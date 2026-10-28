"""Page-64 working banks for the MiMo DFlash draft, independent of target IDs."""

import weakref

import torch
import triton
import triton.language as tl
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.speculative.dflash_compact_layout import CompactDFlashLayout


class FixedDFlashAllocator(BaseTokenToKVPoolAllocator):
    """Expose the draft pool without allowing global allocation/free operations."""

    def __init__(self, pool):
        super().__init__(
            pool.size, pool.page_size, pool.dtype, pool.device, pool, False
        )
        self.clear()

    def clear(self):
        self.free_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self.release_pages = torch.empty_like(self.free_pages)

    def alloc(self, *args, **kwargs):
        raise RuntimeError(
            "compact draft banks are addressed by request owner, not allocated globally"
        )

    alloc_extend = alloc
    alloc_decode = alloc

    def free(self, *args, **kwargs):
        raise RuntimeError(
            "target KV indices must never be freed through the compact draft allocator"
        )


@triton.jit
def _copy_banks_5d(
    k_ptrs,
    v_ptrs,
    src_bases,
    dst_bases,
    lengths,
    PAGE: tl.constexpr,
    DIM: tl.constexpr,
    ZERO: tl.constexpr,
    ROWS: tl.constexpr = 16,
):
    request = tl.program_id(1)
    layer = tl.program_id(2)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    dims = tl.arange(0, DIM)
    valid = rows[:, None] < tl.load(lengths + request)
    dst = tl.load(dst_bases + request) + rows
    k = tl.load(k_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    v = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    # One TP-local KV head; K [page,1,D/8,P,8], V [page,1,P/8,D,8].
    dk = (
        (dst[:, None] // PAGE * (DIM // 8) + dims[None, :] // 8) * PAGE
        + dst[:, None] % PAGE
    ) * 8 + dims[None, :] % 8
    dv = (
        (dst[:, None] // PAGE * (PAGE // 8) + dst[:, None] % PAGE // 8) * DIM
        + dims[None, :]
    ) * 8 + dst[:, None] % 8
    if ZERO:
        kval = tl.full((ROWS, DIM), 0, tl.bfloat16)
        vval = tl.full((ROWS, DIM), 0, tl.bfloat16)
    else:
        src = tl.load(src_bases + request) + rows
        sk = (
            (src[:, None] // PAGE * (DIM // 8) + dims[None, :] // 8) * PAGE
            + src[:, None] % PAGE
        ) * 8 + dims[None, :] % 8
        sv = (
            (src[:, None] // PAGE * (PAGE // 8) + src[:, None] % PAGE // 8) * DIM
            + dims[None, :]
        ) * 8 + src[:, None] % 8
        kval = tl.load(k + sk, valid, other=0)
        vval = tl.load(v + sv, valid, other=0)
    tl.store(k + dk, kval, valid)
    tl.store(v + dv, vval, valid)


@triton.jit
def _write_request_table(
    table, owners, bases, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    req = tl.program_id(0)
    owner = tl.load(owners + req)
    col = tl.arange(0, BLOCK)
    tl.store(table + owner * WIDTH + col, tl.load(bases + req) + col, col < WIDTH)


class CompactDFlashCache:
    """Owns fixed physical banks; request identities guard slot reuse.

    Operations run outside the model's captured graph on its forward stream.
    A new owner waits for the last use of its banks; old requests are held only
    through weak references. Cancellation does not need to free GPU storage:
    the fixed bank pair is safely reinitialized when the request slot is reused.
    """

    def __init__(self, layout: CompactDFlashLayout, pool, request_table: torch.Tensor):
        if (
            pool.kv_cache_layout != "vectorized_5d"
            or pool.dtype != torch.bfloat16
            or pool.page_size != 64
            or pool.head_num != 1
            or pool.head_dim != 128
            or pool.v_head_dim != 128
            or pool.size != layout.physical_tokens
        ):
            raise ValueError(
                "compact DFlash requires its own BF16 H1/D128/V128 page-64 pool"
            )
        if request_table.shape != (layout.request_rows, layout.bank_rows):
            raise ValueError("compact draft request table disagrees with layout")
        self.layout = layout
        self.pool = pool
        self.table = request_table
        self.device = request_table.device
        self.banks = torch.zeros(
            layout.request_rows, dtype=torch.int64, device=self.device
        )
        self.starts = torch.zeros_like(self.banks)
        self.ends = torch.zeros_like(self.banks)
        self.k_ptrs = pool.k_data_ptrs
        self.v_ptrs = pool.v_data_ptrs
        self.block_offsets = torch.arange(
            layout.block_size, dtype=torch.int64, device=self.device
        )
        self._requests = [None] * layout.request_rows
        self._events = [None] * layout.request_rows
        self.generations = [0] * layout.request_rows
        self._retractions = [None] * layout.request_rows

    def _owners(self, reqs, *, allow_new: bool):
        owners = []
        fresh = []
        for req in reqs:
            owner = req.req_pool_idx
            if owner is None or not 1 <= owner <= self.layout.owner_count:
                raise ValueError(f"invalid compact draft request index {owner}")
            owners.append(owner)
            previous = self._requests[owner]
            retractions = getattr(req, "retraction_count", 0)
            if (
                previous is None
                or previous() is not req
                or self._retractions[owner] != retractions
            ):
                if not allow_new:
                    raise RuntimeError("draft decode has no committed owner state")
                event = self._events[owner]
                if event is not None:
                    torch.cuda.current_stream(self.device).wait_event(event)
                self._requests[owner] = weakref.ref(req)
                self._retractions[owner] = retractions
                self.generations[owner] += 1
                fresh.append(owner)
        if len(set(owners)) != len(owners):
            raise ValueError("one forward cannot contain a request slot twice")
        if fresh:
            self.banks[fresh] = 0
            self.starts[fresh] = 0
            self.ends[fresh] = 0
        return torch.tensor(owners, dtype=torch.int64, device=self.device), fresh

    def _copy(self, src, dst, lengths, *, zero=False):
        if not lengths.numel():
            return
        _copy_banks_5d[
            (
                triton.cdiv(self.layout.bank_rows, 16),
                lengths.numel(),
                self.pool.layer_num,
            )
        ](
            self.k_ptrs,
            self.v_ptrs,
            src,
            dst,
            lengths,
            PAGE=self.layout.page_size,
            DIM=128,
            ZERO=zero,
        )

    def _start(self, lengths):
        return (
            torch.div(
                torch.clamp(lengths - self.layout.window_size, min=0),
                self.layout.page_size,
                rounding_mode="floor",
            )
            * self.layout.page_size
        )

    def _base(self, owners, banks):
        return (2 * owners + banks) * self.layout.bank_rows

    def _check_lengths(self, owners, prefix_lens):
        torch._assert_async(
            (self.ends[owners] == prefix_lens).all(),
            "compact draft committed length mismatch",
        )

    def prepare_decode(self, reqs, prefix_lens):
        owners, _ = self._owners(reqs, allow_new=False)
        prefix_lens = prefix_lens.to(torch.int64)
        self._check_lengths(owners, prefix_lens)
        old_bank = self.banks[owners]
        work_bank = 1 - old_bank
        starts = self._start(prefix_lens)
        lengths = prefix_lens - starts
        old_starts = self.starts[owners]
        torch._assert_async(
            (starts >= old_starts).all(),
            "compact draft suffix precedes retained history",
        )
        src = self._base(owners, old_bank) + starts - old_starts
        dst = self._base(owners, work_bank)
        self._copy(src, dst, lengths)
        _write_request_table[(len(reqs),)](
            self.table,
            owners,
            dst,
            WIDTH=self.layout.bank_rows,
            BLOCK=triton.next_power_of_2(self.layout.bank_rows),
        )
        self.banks[owners] = work_bank
        self.starts[owners] = starts
        locs = dst[:, None] + lengths[:, None] + self.block_offsets[None, :]
        return owners, lengths.to(torch.int32), locs

    def prepare_prefill(self, reqs, prefix_lens, extend_lens):
        owners, fresh = self._owners(reqs, allow_new=True)
        for req, prefix in zip(reqs, prefix_lens):
            if req.req_pool_idx in fresh and prefix != 0:
                raise RuntimeError(
                    "compact draft cannot restore an uncached real prefix"
                )
        prefixes = torch.tensor(prefix_lens, dtype=torch.int64, device=self.device)
        ends = prefixes + torch.tensor(
            extend_lens, dtype=torch.int64, device=self.device
        )
        self._check_lengths(owners, prefixes)
        starts = self._start(ends)
        old_bank = self.banks[owners]
        work_bank = 1 - old_bank
        dst = self._base(owners, work_bank)
        retained = torch.clamp(prefixes - starts, min=0)
        old_starts = self.starts[owners]
        torch._assert_async(
            ((retained == 0) | (starts >= old_starts)).all(),
            "compact draft prefill history mismatch",
        )
        self._copy(self._base(owners, old_bank) + starts - old_starts, dst, retained)

        hidden_indices = []
        write_locs = []
        offset = 0
        for i, (prefix, count) in enumerate(zip(prefix_lens, extend_lens)):
            end = prefix + count
            start = self.layout.visible_start(end)
            first = max(prefix, start)
            hidden_indices.append(
                torch.arange(
                    offset + first - prefix,
                    offset + count,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
            write_locs.append(
                dst[i]
                + torch.arange(
                    first - start, end - start, dtype=torch.int64, device=self.device
                )
            )
            offset += count
        self.banks[owners] = work_bank
        self.starts[owners] = starts
        return owners, ends, torch.cat(hidden_indices), torch.cat(write_locs)

    def finish(self, reqs, owners, ends):
        self.ends[owners] = ends.to(torch.int64)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        for req in reqs:
            self._events[req.req_pool_idx] = event

    def clear(self):
        stream = torch.cuda.current_stream(self.device)
        for event in {e for e in self._events if e is not None}:
            stream.wait_event(event)
        self.banks.zero_()
        self.starts.zero_()
        self.ends.zero_()
        self._requests = [None] * self.layout.request_rows
        self._retractions = [None] * self.layout.request_rows
        self._events = [None] * self.layout.request_rows

    def describe(self):
        return {
            "owner_count": self.layout.owner_count,
            "bank_rows": self.layout.bank_rows,
            "banks_per_owner": 2,
            "draft_tokens": self.layout.physical_tokens,
            "request_table_shape": list(self.table.shape),
            "qkv_projection": "torch_bf16_linear",
            "communication": "rccl",
            "kv_bytes": sum(
                t.numel() * t.element_size()
                for t in self.pool.k_buffer + self.pool.v_buffer
            ),
            "persistent_metadata_bytes": sum(
                t.numel() * t.element_size()
                for t in (
                    self.table,
                    self.banks,
                    self.starts,
                    self.ends,
                    self.k_ptrs,
                    self.v_ptrs,
                    self.pool.data_ptrs,
                    self.pool.data_strides,
                    self.block_offsets,
                )
            ),
        }
