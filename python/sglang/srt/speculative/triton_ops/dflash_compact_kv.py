"""Build DFlash's compact request table directly from the target request table."""

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dflash_draft_ring import dflash_draft_ring_size


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


@triton.jit
def _ring_dflash_req_to_token_kernel(
    prefix_lens,
    req_indices,
    draft_table,
    block_locs,
    compact_lens,
    prefix_stride,
    req_stride,
    draft_row_stride,
    block_row_stride,
    compact_stride,
    WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
    VERIFY: tl.constexpr,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
    VERIFY_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(req_indices + row * req_stride).to(tl.int64)
    prefix = tl.load(prefix_lens + row * prefix_stride).to(tl.int64)
    visible_start = tl.maximum(prefix - WINDOW, 0)
    start = visible_start // PAGE * PAGE
    length = prefix - start
    base = req * RING

    # Position p of request r lives at r * RING + p % RING. RING is a whole
    # number of pages and start is page aligned, so pages stay contiguous.
    cols = tl.arange(0, BLOCK).to(tl.int64)
    slots = base + (start + cols) % RING
    tl.store(
        draft_table + req * draft_row_stride + cols, slots, mask=cols < length + VERIFY
    )
    tl.store(compact_lens + row * compact_stride, length)

    verify_cols = tl.arange(0, VERIFY_BLOCK).to(tl.int64)
    tl.store(
        block_locs + row * block_row_stride + verify_cols,
        base + (prefix + verify_cols) % RING,
        mask=verify_cols < VERIFY,
    )


def ring_dflash_req_to_token(
    prefix_lens: torch.Tensor,
    req_indices: torch.Tensor,
    draft_table: torch.Tensor,
    block_locs: torch.Tensor,
    compact_lens: torch.Tensor,
    *,
    window_size: int,
    page_size: int,
    ring_size: int,
) -> None:
    """Fill the draft request table and block slots for a per-request KV ring.

    Unlike ``compact_dflash_req_to_token`` no target slot is read: the draft KV
    pool is addressed by request row and position only. ``block_locs`` receives
    the slots of the block positions ``prefix .. prefix + block - 1``; the same
    slots are appended to the draft table after the visible prefix.
    """
    batch = prefix_lens.numel()
    if (
        prefix_lens.ndim != 1
        or req_indices.shape != (batch,)
        or compact_lens.shape != (batch,)
    ):
        raise ValueError("DFlash ring lengths and request indices must be vectors")
    if (
        block_locs.ndim != 2
        or block_locs.shape[0] != batch
        or block_locs.stride(1) != 1
    ):
        raise ValueError("block slots must be [batch, block] with contiguous columns")
    if draft_table.ndim != 2 or draft_table.stride(1) != 1:
        raise ValueError("draft request table must have contiguous columns")
    tensors = (prefix_lens, req_indices, draft_table, block_locs, compact_lens)
    if any(
        t.device != prefix_lens.device or t.dtype not in (torch.int32, torch.int64)
        for t in tensors
    ):
        raise ValueError("DFlash ring requires int32/int64 tensors on one device")
    if not prefix_lens.is_cuda:
        raise ValueError("DFlash ring requires a CUDA/HIP device")
    verify = block_locs.shape[1]
    if verify <= 0:
        raise ValueError("DFlash ring requires a non-empty block")
    if ring_size != dflash_draft_ring_size(window_size, page_size, verify):
        raise ValueError("DFlash ring size does not match window, page and block")
    capacity = window_size + page_size - 1 + verify
    if draft_table.shape[1] < capacity:
        raise ValueError("draft request table is too small for the ring window")
    if batch == 0:
        return
    _ring_dflash_req_to_token_kernel[(batch,)](
        prefix_lens,
        req_indices,
        draft_table,
        block_locs,
        compact_lens,
        prefix_lens.stride(0),
        req_indices.stride(0),
        draft_table.stride(0),
        block_locs.stride(0),
        compact_lens.stride(0),
        WINDOW=int(window_size),
        PAGE=int(page_size),
        VERIFY=verify,
        RING=int(ring_size),
        BLOCK=triton.next_power_of_2(capacity),
        VERIFY_BLOCK=triton.next_power_of_2(verify),
        num_warps=4,
    )


def warmup_ring_dflash_req_to_token(
    device, *, window_size: int, page_size: int, block_size: int, ring_size: int
) -> None:
    """Compile the ring kernel at startup instead of on the first decode step.

    Runs it on scratch tensors for the length and request-index dtypes a batch
    can carry; nothing of the draft pool is touched.
    """
    table = torch.zeros((2, ring_size), dtype=torch.int32, device=device)
    block = torch.zeros((1, block_size), dtype=torch.int64, device=device)
    lengths = torch.zeros(1, dtype=torch.int32, device=device)
    for prefix_dtype in (torch.int32, torch.int64):
        for request_dtype in (torch.int32, torch.int64):
            ring_dflash_req_to_token(
                torch.ones(1, dtype=prefix_dtype, device=device),
                torch.zeros(1, dtype=request_dtype, device=device),
                table,
                block,
                lengths,
                window_size=window_size,
                page_size=page_size,
                ring_size=ring_size,
            )
