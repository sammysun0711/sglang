"""Per-request ring for the windowed DFlash draft KV, and the PD transfer of the draft window.

A windowed draft only reads the page-aligned window before a request's committed
prefix, and the draft KV of a position is materialized from the target hidden
state of that position alone. A request therefore needs draft KV for its window
plus the block being drafted, not for every target token.

Ring storage (SGLANG_ENABLE_DFLASH_DRAFT_KV_RING; decode and non-disaggregated
servers). The draft pool holds one ring per row of the target request table:

    slot(row, pos) = row * ring + pos % ring

``ring`` is a whole number of pages, so ring pages line up with sequence pages.
Row 0 of the request table is its padding row (ReqToTokenPool and
DecodeReqToTokenPool never hand it out), so row 0's ring takes the writes that no
window reads: padded graph rows, and the head of a prompt longer than the ring.

Window transfer (SGLANG_ENABLE_DFLASH_DRAFT_WINDOW_TRANSFER; PD). Prefill sends
only the pages of the draft window, as the DRAFT_WINDOW state component, instead
of draft KV for every prompt page. Each server lists the pages of the same window
positions in ascending order, in its own layout: target slots where the draft KV
shares them (prefill, or a decode server without a ring), ring pages where it
keeps a ring. The two lists pair up page by page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

# Attribute of the draft KV pool that carries its DFlashDraftWindow.
DRAFT_WINDOW_ATTR = "dflash_draft_window"
# Transfer backends that send the DRAFT_WINDOW state component ("fake" sends nothing).
DRAFT_WINDOW_TRANSFER_BACKENDS = ("mooncake", "fake")

_CHECKSUM_MODULUS = (1 << 31) - 1
_CHECKSUM_COLUMN_PERIOD = 8191


def dflash_draft_kv_ring_problems(
    server_args: ServerArgs, *, windowed: bool, fused_compact_kv: bool
) -> List[str]:
    """What a server lacks to keep its draft KV in a per-request ring; empty if nothing."""
    if server_args.disaggregation_mode == "prefill":
        return [
            "a decode or non-disaggregated server (a prefill server keeps its draft "
            "KV at target slots: set SGLANG_ENABLE_DFLASH_DRAFT_WINDOW_TRANSFER=1 "
            "there to pair it with a decode server that keeps the ring)"
        ]
    problems = []
    if not windowed:
        problems.append("a draft window (--speculative-draft-window-size)")
    if not fused_compact_kv:
        problems.append("SGLANG_DFLASH_FUSED_COMPACT_KV=1 on a CUDA/HIP device")
    if server_args.max_running_requests is None:
        # The ring is sized per request row before the target pool is sized.
        problems.append("an explicit --max-running-requests")
    if server_args.enable_hierarchical_cache:
        problems.append("no hierarchical cache (host slots mirror target slots)")
    # A ring row belongs to one request: a prefix reused from the radix cache
    # would arrive without the draft KV of its window.
    if not server_args.disable_radix_cache and not (
        server_args.disaggregation_mode == "decode"
        and not server_args.disaggregation_decode_enable_radix_cache
    ):
        problems.append("no prefix reuse (--disable-radix-cache)")
    return problems


def dflash_draft_window_transfer_problems(
    server_args: ServerArgs, *, windowed: bool
) -> List[str]:
    """What a PD server lacks to transfer only the draft window; empty if nothing."""
    problems = []
    if not windowed:
        problems.append("a draft window (--speculative-draft-window-size)")
    backend = server_args.disaggregation_transfer_backend
    if backend not in DRAFT_WINDOW_TRANSFER_BACKENDS:
        problems.append(
            "the mooncake transfer backend (the DRAFT_WINDOW state is not "
            f"implemented for {backend!r})"
        )
    return problems


def dflash_draft_ring_size(window_size: int, page_size: int, block_size: int) -> int:
    """Tokens one request needs in a windowed draft KV ring.

    The draft reads its page-aligned window plus the block being drafted, so at
    most ``window + page - 1 + block`` consecutive positions are live at once.
    Rounding up to whole pages keeps ring pages aligned with sequence pages.
    """
    if window_size <= 0 or page_size <= 0 or block_size <= 0:
        raise ValueError("DFlash draft ring requires positive window, page and block")
    live = window_size + page_size - 1 + block_size
    return (live + page_size - 1) // page_size * page_size


def dflash_draft_window_start(seq_len: int, window_size: int, page_size: int) -> int:
    """First position of the page-aligned draft window of a ``seq_len`` prefix."""
    return max(0, seq_len - window_size) // page_size * page_size


def dflash_draft_ring_extend_slots(
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    positions: torch.Tensor,
    ring_size: int,
) -> torch.Tensor:
    """Ring slots for the prompt tokens of an extend batch, in token order.

    A chunk longer than the ring would write a slot more than once; only its
    last ``ring_size`` positions can still be read, so earlier ones are sent
    to the ring of the padding request row (row 0), which no request reads.
    """
    extend_lens = extend_lens.to(torch.int64)
    # output_size: without it repeat_interleave reads the sum back to the host.
    num_tokens = int(positions.numel())
    rows = torch.repeat_interleave(
        req_pool_indices.to(torch.int64), extend_lens, output_size=num_tokens
    )
    seq_ends = torch.repeat_interleave(
        prefix_lens.to(torch.int64) + extend_lens, extend_lens, output_size=num_tokens
    )
    positions = positions.to(torch.int64)
    rows = torch.where(positions >= seq_ends - ring_size, rows, torch.zeros_like(rows))
    return rows * ring_size + positions % ring_size


def dflash_draft_ring_rows(server_args: ServerArgs, max_num_reqs: int) -> int:
    """Upper bound on request-table rows of one DP worker, padding row included.

    A windowed draft KV ring holds one ring per request-table row, so its size
    must be reserved before the target pool is sized. Mirrors the request
    table built by the model runner.
    """
    rows = max_num_reqs + 1
    if server_args.disaggregation_mode == "decode":
        pre_alloc_size = envs.SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS.get()
        rows += max_num_reqs * 2 if max_num_reqs <= 32 else pre_alloc_size
    return rows


@dataclass(frozen=True)
class DFlashDraftWindow:
    """How a server stores the draft window that PD transfers on its own.

    ``ring_size`` is None where the draft KV shares the target slots.
    """

    window_size: int
    ring_size: Optional[int] = None

    def page_indices(
        self,
        req_to_token: torch.Tensor,
        req_pool_idx: int,
        seq_len: int,
        page_size: int,
    ) -> np.ndarray:
        """Draft pool pages holding the window of a ``seq_len`` prompt, by position.

        Prefill and decode both call this with their own request row; equal
        ``seq_len``, window and page size give lists of equal length whose i-th
        entries hold the same positions.
        """
        start = dflash_draft_window_start(seq_len, self.window_size, page_size)
        if self.ring_size is None:
            slots = req_to_token[req_pool_idx, start:seq_len].cpu().numpy()
            return (slots[::page_size] // page_size).astype(np.int32)
        if self.ring_size % page_size != 0:
            raise ValueError("DFlash draft ring must be a whole number of pages")
        positions = np.arange(start, seq_len, page_size, dtype=np.int64)
        slots = int(req_pool_idx) * self.ring_size + positions % self.ring_size
        return (slots // page_size).astype(np.int32)


def get_dflash_draft_window(draft_token_to_kv_pool) -> Optional[DFlashDraftWindow]:
    """The window layout the DFlash worker attached to its draft pool, if any."""
    return getattr(draft_token_to_kv_pool, DRAFT_WINDOW_ATTR, None)


def dflash_draft_window_checksum(draft_token_to_kv_pool, page_indices) -> int:
    """Position-sensitive checksum of the listed draft KV pages (debug only).

    Prefill and decode list the pages of one window in the same order, so equal
    values on both sides mean every window page arrived at the place decode reads.
    """
    pool = draft_token_to_kv_pool
    pages = torch.as_tensor(np.asarray(page_indices), dtype=torch.int64)
    total = 0
    if pages.numel() == 0:
        return total
    buffers = [
        get(layer)
        for get in (pool._get_key_buffer, pool._get_value_buffer)
        for layer in range(pool.start_layer, pool.start_layer + pool.layer_num)
    ]
    page_weights = torch.arange(1, pages.numel() + 1, dtype=torch.int64)
    for index, buffer in enumerate(buffers):
        page_elements = pool._disagg_page_nbytes(buffer) // buffer.element_size()
        rows = buffer.reshape(-1, page_elements).index_select(
            0, pages.to(buffer.device)
        )
        values = rows.contiguous().view(torch.int16).to(torch.int64)
        column_weights = (
            torch.arange(values.shape[1], dtype=torch.int64, device=values.device)
            % _CHECKSUM_COLUMN_PERIOD
            + 1
        )
        per_page = (values * column_weights).sum(dim=1) % _CHECKSUM_MODULUS
        layer_sum = int((per_page.cpu() * page_weights).sum() % _CHECKSUM_MODULUS)
        total = (total * 31 + layer_sum + index + 1) % _CHECKSUM_MODULUS
    return total
