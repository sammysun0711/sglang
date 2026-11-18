"""Startup warmup for DFlash's HIP decode-allocation bookkeeping."""

import logging

import torch

from sglang.srt.mem_cache.triton_ops.common import get_last_loc_triton_safe
from sglang.srt.speculative.triton_ops.cache_locs import assign_req_to_token_pool_func

logger = logging.getLogger(__name__)


def warmup_dflash_decode_bookkeeping(
    req_to_token: torch.Tensor, max_batch_size: int, page_size: int
) -> None:
    """Warm int32-length DFlash variants without changing the request table.

    Allocation runs outside the captured model forward. In particular, a rolling
    batch of one request can miss both the explicit ``bs_upper=1`` assignment
    variant and Triton's implicit scalar ``num_tokens=1`` specialization.

    Powers of two cover assignment's batch buckets and the last-location
    kernel's scalar classes (one, ordinary, and divisible by16). Include the
    actual maximum to warm its final, possibly partial bucket. Use the real
    table: dtype, row stride, alignment and Triton's pointer-range metadata are
    part of specialization; a small dummy table can produce different code.

    All prefix/start/end lengths are zero. The safe lookup masks every table
    read, and assignment executes zero write iterations. No KV slots, request
    ownership, model data or RNG state are changed.
    """
    if max_batch_size < 1:
        return

    device = req_to_token.device
    req_indices = torch.zeros(max_batch_size, dtype=torch.int64, device=device)
    starts = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
    ends = torch.zeros_like(starts)
    cache_locs = torch.empty(max_batch_size, dtype=torch.int64, device=device)

    batch_sizes = []
    batch_size = 1
    while batch_size < max_batch_size:
        batch_sizes.append(batch_size)
        batch_size *= 2
    batch_sizes.append(max_batch_size)

    for batch_size in batch_sizes:
        indices = req_indices[:batch_size]
        prefix = starts[:batch_size]
        if page_size > 1:
            get_last_loc_triton_safe(req_to_token, indices, prefix)
        assign_req_to_token_pool_func(
            indices,
            req_to_token,
            prefix,
            ends[:batch_size],
            cache_locs,
            batch_size,
        )

    torch.get_device_module(device).synchronize()
    logger.info(
        "Warmed HIP DFlash KV bookkeeping: batches=%s, row_stride=%d, "
        "table_dtype=%s, prefix_dtype=int32, paged=%s",
        batch_sizes,
        req_to_token.stride(0),
        req_to_token.dtype,
        page_size > 1,
    )
