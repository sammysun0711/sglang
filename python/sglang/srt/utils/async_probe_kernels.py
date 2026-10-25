"""Fused asynchronous integer bounds checks; assertions stay enabled in the kernel."""

import triton
import triton.language as tl


@triton.jit
def _check_index_bounds_kernel(
    indices,
    size,
    low,
    high,
    LOW_MESSAGE: tl.constexpr,
    HIGH_MESSAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < size
    values = tl.load(indices + offsets, mask=active, other=0)
    tl.device_assert((~active) | (values >= low), LOW_MESSAGE)
    tl.device_assert((~active) | (values < high), HIGH_MESSAGE)


def check_index_bounds(indices, low: int, high: int, msg: str) -> None:
    """Check contiguous integer indices with one kernel, including on graph replay."""
    _check_index_bounds_kernel[(triton.cdiv(indices.numel(), 256),)](
        indices,
        indices.numel(),
        low,
        high,
        LOW_MESSAGE=f"index < {low} (negative / unmasked sentinel?): {msg}",
        HIGH_MESSAGE=f"index >= {high} (out of range): {msg}",
        BLOCK=256,
        num_warps=4,
        # tl.device_assert is otherwise compiled out. This is a correctness
        # requirement, independent of the caller's TRITON_DEBUG environment.
        debug=True,
    )
