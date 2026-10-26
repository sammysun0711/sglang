"""Build DFlash's compact request table directly from the target request table."""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_dflash_req_to_token_kernel(
    prefix_lens,
    req_indices,
    target_table,
    draft_table,
    verify_locs,
    compact_lens,
    prefix_stride,
    req_stride,
    target_row_stride,
    draft_row_stride,
    verify_row_stride,
    compact_stride,
    WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
    VERIFY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(req_indices + row * req_stride).to(tl.int64)
    prefix = tl.load(prefix_lens + row * prefix_stride).to(tl.int64)
    visible_start = tl.maximum(prefix - WINDOW, 0)
    start = visible_start // PAGE * PAGE
    length = prefix - start
    cols = tl.arange(0, BLOCK)

    # Input and output are distinct tables. Each request's prefix and verify
    # slots are written once, with the same ordering as gather + two assigns.
    old = tl.load(
        target_table + req * target_row_stride + start + cols,
        mask=cols < length,
        other=0,
    )
    verify_col = cols.to(tl.int64) - length
    new = tl.load(
        verify_locs + row * verify_row_stride + verify_col,
        mask=(verify_col >= 0) & (verify_col < VERIFY),
        other=0,
    )
    slots = tl.where(cols < length, old, new)
    tl.store(
        draft_table + req * draft_row_stride + cols, slots, mask=cols < length + VERIFY
    )
    tl.store(compact_lens + row * compact_stride, length)


def compact_dflash_req_to_token(
    prefix_lens: torch.Tensor,
    req_indices: torch.Tensor,
    target_table: torch.Tensor,
    draft_table: torch.Tensor,
    verify_locs: torch.Tensor,
    compact_lens: torch.Tensor,
    *,
    window_size: int,
    page_size: int,
) -> None:
    """Replace the compact suffix gather and both request-table assignments.

    Request indices are unique and valid in both tables; prefix lengths refer
    to committed target slots. These are the same invariants required by the
    existing DFlash gather/assign path. No device scalar is read on the host.
    Storage after each request's compact prefix plus verify block is untouched.
    """
    if window_size <= 0 or page_size <= 0:
        raise ValueError("DFlash compaction requires positive window and page sizes")
    batch = prefix_lens.numel()
    if (
        prefix_lens.ndim != 1
        or req_indices.shape != (batch,)
        or compact_lens.shape != (batch,)
    ):
        raise ValueError("DFlash compact lengths and request indices must be vectors")
    if (
        verify_locs.ndim != 2
        or verify_locs.shape[0] != batch
        or verify_locs.stride(1) != 1
    ):
        raise ValueError(
            "verify locations must be [batch, block] with contiguous columns"
        )
    if (
        target_table.ndim != 2
        or draft_table.ndim != 2
        or target_table.stride(1) != 1
        or draft_table.stride(1) != 1
    ):
        raise ValueError("request tables must have contiguous columns")
    if (
        target_table.untyped_storage().data_ptr()
        == draft_table.untyped_storage().data_ptr()
    ):
        raise ValueError("compact draft and target request tables must not alias")
    tensors = (
        prefix_lens,
        req_indices,
        target_table,
        draft_table,
        verify_locs,
        compact_lens,
    )
    if any(
        t.device != prefix_lens.device or t.dtype not in (torch.int32, torch.int64)
        for t in tensors
    ):
        raise ValueError("DFlash compaction requires int32/int64 tensors on one device")
    if not prefix_lens.is_cuda:
        raise ValueError("DFlash compaction requires a CUDA/HIP device")
    verify = verify_locs.shape[1]
    capacity = window_size + page_size - 1 + verify
    if verify <= 0 or draft_table.shape[1] < capacity:
        raise ValueError(
            "draft request table is too small for the aligned window and verify block"
        )
    if batch == 0:
        return
    _compact_dflash_req_to_token_kernel[(batch,)](
        prefix_lens,
        req_indices,
        target_table,
        draft_table,
        verify_locs,
        compact_lens,
        prefix_lens.stride(0),
        req_indices.stride(0),
        target_table.stride(0),
        draft_table.stride(0),
        verify_locs.stride(0),
        compact_lens.stride(0),
        WINDOW=int(window_size),
        PAGE=int(page_size),
        VERIFY=verify,
        BLOCK=triton.next_power_of_2(capacity),
        num_warps=4,
    )
