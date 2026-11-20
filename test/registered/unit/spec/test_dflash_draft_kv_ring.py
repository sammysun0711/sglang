"""Per-request DFlash draft KV ring and the PD transfer of the draft window.

Covers the ring layout (slots, prefill writes, the request-table rewrite), the
page lists prefill and decode exchange for the draft window, the bootstrap
check that both servers transfer it, and the switch validation.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import PrefillServerInfo
from sglang.srt.disaggregation.utils import check_pd_dflash_draft_window
from sglang.srt.speculative.dflash_draft_ring import (
    DRAFT_WINDOW_ATTR,
    DFlashDraftWindow,
    dflash_draft_kv_ring_problems,
    dflash_draft_ring_extend_slots,
    dflash_draft_ring_rows,
    dflash_draft_ring_size,
    dflash_draft_window_checksum,
    dflash_draft_window_start,
    dflash_draft_window_transfer_problems,
    get_dflash_draft_window,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WINDOW, PAGE, BLOCK = 1024, 64, 8
RING = dflash_draft_ring_size(WINDOW, PAGE, BLOCK)


def ring_table(prefix, row, window=WINDOW, page=PAGE, block=BLOCK, ring=RING):
    """Reference of the ring kernel: (draft table row, block slots, compact length)."""
    start = max(prefix - window, 0) // page * page
    length = prefix - start
    table = row * ring + (start + np.arange(length + block)) % ring
    return table, row * ring + (prefix + np.arange(block)) % ring, length


@pytest.mark.parametrize(
    "window,page,block,expected",
    [(1024, 64, 8, 1152), (128, 64, 8, 256), (16, 1, 8, 24), (1024, 64, 16, 1152)],
)
def test_ring_holds_window_alignment_slack_and_block(window, page, block, expected):
    ring = dflash_draft_ring_size(window, page, block)
    assert ring == expected
    assert ring % page == 0 and ring >= window + page - 1 + block


def test_ring_size_rejects_empty_geometry():
    with pytest.raises(ValueError):
        dflash_draft_ring_size(0, 64, 8)


def test_ring_rows_follow_the_request_table():
    plain = SimpleNamespace(disaggregation_mode="null")
    decode = SimpleNamespace(disaggregation_mode="decode")
    assert dflash_draft_ring_rows(plain, 257) == 258
    # Small decode pools pre-allocate twice their size, like the request table.
    assert dflash_draft_ring_rows(decode, 16) == 16 + 32 + 1


def test_extend_slots_write_each_live_slot_once():
    # Two requests in one extend batch: a short chunk and one longer than the ring.
    rows = torch.tensor([3, 5])
    prefix = torch.tensor([10, 4096])
    extend = torch.tensor([7, 3 * RING + 5])
    positions = torch.cat(
        [torch.arange(10, 17), torch.arange(4096, 4096 + 3 * RING + 5)]
    )
    slots = dflash_draft_ring_extend_slots(rows, prefix, extend, positions, RING)
    assert slots[:7].tolist() == [3 * RING + p for p in range(10, 17)]
    long_slots, long_positions = slots[7:], positions[7:]
    kept = long_slots >= RING
    # Only the last RING positions of the long chunk stay, each on its own slot.
    assert int(kept.sum()) == RING
    assert torch.all(long_positions[kept] >= 4096 + 3 * RING + 5 - RING)
    assert long_slots[kept].unique().numel() == RING
    assert torch.all(long_slots[kept] == 5 * RING + long_positions[kept] % RING)
    # Everything else lands in the padding row's ring.
    assert torch.all(long_slots[~kept] < RING)


@pytest.mark.parametrize(
    "prompt_len", [1, 63, 64, 1023, 1024, 1087, 1088, 1151, 1152, 1153, 2500, 40001]
)
@pytest.mark.parametrize("chunk", [1, 64, 1152, 1153, 16384])
def test_window_reads_back_its_own_positions(prompt_len, chunk):
    """Chunked prefill followed by decode steps never leaves the window stale."""
    if chunk < 64 and prompt_len > 3000:
        pytest.skip("one-token chunks are only exercised on short prompts")
    row, other = 2, 1
    owner = np.full(4 * RING, -1, dtype=np.int64)
    owner[row * RING : (row + 1) * RING] = -7  # left over from an earlier request
    owner[other * RING : (other + 1) * RING] = 10**9  # a neighbour that must survive
    for start in range(0, prompt_len, chunk):
        end = min(prompt_len, start + chunk)
        positions = torch.arange(start, end)
        slots = dflash_draft_ring_extend_slots(
            torch.tensor([row]),
            torch.tensor([start]),
            torch.tensor([end - start]),
            positions,
            RING,
        ).numpy()
        owner[slots] = np.where(slots >= RING, positions.numpy(), -9)
    prefix = prompt_len
    rng = np.random.default_rng(prompt_len * 31 + chunk)
    for _ in range(400):
        table, block_slots, length = ring_table(prefix, row)
        # Pages of the draft table are whole, aligned ring pages.
        assert np.all(table[::PAGE] % PAGE == 0)
        assert np.all(owner[table[:length]] == prefix - length + np.arange(length))
        # The block being drafted never overwrites a slot the window still reads.
        assert not np.intersect1d(block_slots, table[:length]).size
        assert np.array_equal(block_slots, table[length:])
        owner[block_slots] = -1
        commit = int(rng.choice([1, 3, 4, 8]))
        owner[block_slots[:commit]] = prefix + np.arange(commit)
        prefix += commit
    assert np.all(owner[other * RING : (other + 1) * RING] == 10**9)


def test_request_tables_reserve_row_zero_for_padding():
    """Row 0's ring is the sink of the draft ring: no request may own row 0."""
    try:
        from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    except (ImportError, RuntimeError) as error:
        # aiter probes the GPU at import on ROCm hosts
        pytest.skip(f"memory pools cannot be imported here: {error}")
    plain = ReqToTokenPool(
        size=4, max_context_len=8, device="cpu", enable_memory_saver=False
    )
    decode = DecodeReqToTokenPool(
        size=4,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=2,
    )
    for pool, rows in ((plain, 5), (decode, 7)):
        assert pool.req_to_token.shape[0] == rows
        assert sorted(pool.free_slots) == list(range(1, rows))


def test_ring_decode_holds_transfer_metadata_until_flushed():
    """A reused ring row must not receive its window before the old request's last write."""
    try:
        from sglang.srt.disaggregation.decode import DecodePreallocQueue
    except (ImportError, RuntimeError) as error:
        # aiter probes the GPU at import on ROCm hosts
        pytest.skip(f"the decode queue cannot be imported here: {error}")
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue._deferred_metadata = []
    for pool, expected in (
        (None, False),
        (SimpleNamespace(), False),
        (SimpleNamespace(**{DRAFT_WINDOW_ATTR: DFlashDraftWindow(WINDOW)}), False),
        (SimpleNamespace(**{DRAFT_WINDOW_ATTR: DFlashDraftWindow(WINDOW, RING)}), True),
    ):
        queue.draft_token_to_kv_pool = pool
        assert queue._defers_metadata() is expected
    sent = []
    live = SimpleNamespace(conclude_state=None)
    aborted = SimpleNamespace(conclude_state=KVPoll.Failed)
    queue._deferred_metadata = [
        (live, lambda: sent.append("a")),
        (aborted, lambda: sent.append("aborted")),
        (live, lambda: sent.append("b")),
    ]
    queue.flush_deferred_metadata()
    # In order, once, and never for a request aborted since pre-allocation.
    assert sent == ["a", "b"] and queue._deferred_metadata == []
    queue.flush_deferred_metadata()
    assert sent == ["a", "b"]


def scrambled_slot_table(seq_len, row, rows=4, seed=0):
    """Request table whose row holds page-aligned target slots in a scrambled page order."""
    num_pages = max(1, (seq_len + PAGE - 1) // PAGE)
    rng = np.random.default_rng(seq_len * 7 + seed)
    page_ids = rng.permutation(4 * num_pages)[:num_pages] + 1
    slots = (page_ids[:, None] * PAGE + np.arange(PAGE)[None, :]).reshape(-1)
    table = torch.zeros((rows, num_pages * PAGE), dtype=torch.int32)
    table[row] = torch.from_numpy(slots.astype(np.int32))
    return table, page_ids


SEQ_LENS = [1, 5, 63, 64, 65, 1000, 1023, 1024, 1025, 1087, 1088, 1089, 1151, 1152]
SEQ_LENS += [1153, 2303, 2304, 5000, 123457]


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_prefill_and_decode_list_the_same_window_pages(seq_len):
    """Prefill lists target-slot pages, a ring decode lists ring pages, position by position."""
    table, page_ids = scrambled_slot_table(seq_len, row=3)
    prefill_pages = DFlashDraftWindow(WINDOW).page_indices(table, 3, seq_len, PAGE)
    decode_pages = DFlashDraftWindow(WINDOW, RING).page_indices(None, 7, seq_len, PAGE)

    window_start = dflash_draft_window_start(seq_len, WINDOW, PAGE)
    assert window_start == max(0, seq_len - WINDOW) // PAGE * PAGE
    first_positions = np.arange(window_start, seq_len, PAGE)
    assert len(prefill_pages) == len(decode_pages) == len(first_positions)
    assert prefill_pages.dtype == decode_pages.dtype == np.int32
    assert np.array_equal(prefill_pages, page_ids[first_positions // PAGE])
    assert np.array_equal(decode_pages, (7 * RING + first_positions % RING) // PAGE)
    # A window never needs more pages than the ring has, so no page is listed twice.
    assert len(set(decode_pages.tolist())) == len(decode_pages) <= RING // PAGE - 1
    # The ring pages are exactly where the decode-side kernel reads the window.
    ring_row, _, length = ring_table(seq_len, 7)
    assert np.array_equal(decode_pages, ring_row[:length:PAGE] // PAGE)


@pytest.mark.parametrize("decode_has_ring", [True, False])
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_transferred_window_is_what_the_decode_draft_reads(seq_len, decode_has_ring):
    """Copy the listed pages from a prefill pool to a decode pool and read the window.

    Tokens are tagged with their position. After the page-by-page copy, the
    decode draft must find position p at the slot its request table names for p,
    for every p of the window, whatever the pools held before.
    """
    prefill_row, decode_row = 3, 6
    table, _ = scrambled_slot_table(seq_len, prefill_row)
    prefill_pool = np.full(int(table.max()) + PAGE, -1, dtype=np.int64)
    prefill_pool[table[prefill_row, :seq_len].numpy()] = np.arange(seq_len)
    src_pages = DFlashDraftWindow(WINDOW).page_indices(
        table, prefill_row, seq_len, PAGE
    )

    if decode_has_ring:
        decode_table = None
        decode_pool = np.full(8 * RING, -5, dtype=np.int64)  # stale content everywhere
        window = DFlashDraftWindow(WINDOW, RING)
    else:
        decode_table, _ = scrambled_slot_table(seq_len, decode_row, rows=8, seed=1)
        decode_pool = np.full(int(decode_table.max()) + PAGE, -5, dtype=np.int64)
        window = DFlashDraftWindow(WINDOW)
    dst_pages = window.page_indices(decode_table, decode_row, seq_len, PAGE)
    assert len(src_pages) == len(dst_pages)
    untouched = decode_pool.copy()
    for src, dst in zip(src_pages.tolist(), dst_pages.tolist()):
        decode_pool[dst * PAGE : (dst + 1) * PAGE] = prefill_pool[
            src * PAGE : (src + 1) * PAGE
        ]

    start = dflash_draft_window_start(seq_len, WINDOW, PAGE)
    positions = np.arange(start, seq_len)
    if decode_has_ring:
        read_slots, _, length = ring_table(seq_len, decode_row)
        assert length == seq_len - start
        read_slots = read_slots[:length]
    else:
        read_slots = decode_table[decode_row, start:seq_len].numpy()
    assert np.array_equal(decode_pool[read_slots], positions)
    # Only the listed pages were written.
    written = np.zeros(decode_pool.size, dtype=bool)
    for dst in dst_pages.tolist():
        written[dst * PAGE : (dst + 1) * PAGE] = True
    assert np.array_equal(decode_pool[~written], untouched[~written])
    if decode_has_ring:
        owned = np.zeros(decode_pool.size, dtype=bool)
        owned[decode_row * RING : (decode_row + 1) * RING] = True
        assert not np.any(written & ~owned)


def test_window_layout_is_read_from_the_draft_pool():
    assert get_dflash_draft_window(None) is None
    assert get_dflash_draft_window(SimpleNamespace()) is None
    pool = SimpleNamespace()
    setattr(pool, DRAFT_WINDOW_ATTR, DFlashDraftWindow(WINDOW, RING))
    assert get_dflash_draft_window(pool) == DFlashDraftWindow(WINDOW, RING)
    with pytest.raises(ValueError):
        DFlashDraftWindow(WINDOW, RING + 1).page_indices(None, 1, 100, PAGE)


def test_bootstrap_rejects_a_draft_window_mismatch():
    check_pd_dflash_draft_window(None, None)
    check_pd_dflash_draft_window(WINDOW, WINDOW)
    # None is a server that does not transfer the window, or an older prefill build.
    with pytest.raises(RuntimeError, match="only the decode server"):
        check_pd_dflash_draft_window(None, WINDOW)
    with pytest.raises(RuntimeError, match="only the prefill server"):
        check_pd_dflash_draft_window(WINDOW, None)
    with pytest.raises(RuntimeError, match="speculative-draft-window-size"):
        check_pd_dflash_draft_window(WINDOW, 2 * WINDOW)


def test_prefill_info_carries_the_draft_window_size():
    base = dict(
        attn_tp_size=4,
        attn_cp_size=1,
        dp_size=2,
        pp_size=1,
        page_size=64,
        kv_cache_dtype="bf16",
        follow_bootstrap_room=True,
    )
    # The bootstrap reply leaves the key out when prefill does not transfer the
    # window, so a reply without it must still parse.
    assert PrefillServerInfo(**base).dflash_draft_window_size is None
    info = PrefillServerInfo(**base, dflash_draft_window_size=WINDOW)
    assert info.dflash_draft_window_size == WINDOW


def ring_args(**overrides):
    args = dict(
        disaggregation_mode="decode",
        max_running_requests=64,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        disaggregation_decode_enable_radix_cache=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def ring_problems(args, windowed=True, fused_compact_kv=True):
    return dflash_draft_kv_ring_problems(
        args, windowed=windowed, fused_compact_kv=fused_compact_kv
    )


def test_ring_switch_is_rejected_on_a_prefill_server():
    problems = ring_problems(ring_args(disaggregation_mode="prefill"))
    assert len(problems) == 1 and "DRAFT_WINDOW_TRANSFER" in problems[0]


def test_ring_requirements():
    assert ring_problems(ring_args()) == []
    assert (
        ring_problems(ring_args(disaggregation_mode="null", disable_radix_cache=True))
        == []
    )
    cases = [
        (dict(windowed=False), {}, "draft window"),
        (dict(fused_compact_kv=False), {}, "FUSED_COMPACT_KV"),
        ({}, dict(max_running_requests=None), "max-running-requests"),
        ({}, dict(enable_hierarchical_cache=True), "hierarchical"),
        ({}, dict(disaggregation_mode="null"), "prefix reuse"),
        ({}, dict(disaggregation_decode_enable_radix_cache=True), "prefix reuse"),
    ]
    for flags, overrides, needle in cases:
        problems = ring_problems(ring_args(**overrides), **flags)
        assert len(problems) == 1 and needle in problems[0], (flags, overrides)


@pytest.mark.parametrize("backend", ["mooncake", "fake"])
def test_window_transfer_accepts_validated_backends(backend):
    args = SimpleNamespace(disaggregation_transfer_backend=backend)
    assert dflash_draft_window_transfer_problems(args, windowed=True) == []


@pytest.mark.parametrize(
    "windowed,backend,needle",
    [
        (True, "nixl", "mooncake"),
        (True, "mori", "mooncake"),
        (False, "mooncake", "draft window"),
    ],
)
def test_window_transfer_rejects_unsupported_setups(windowed, backend, needle):
    args = SimpleNamespace(disaggregation_transfer_backend=backend)
    problems = dflash_draft_window_transfer_problems(args, windowed=windowed)
    assert len(problems) == 1 and needle in problems[0]


class FakeDraftPool:
    """Two layers of K and V pages, enough for the checksum's buffer access."""

    start_layer, layer_num = 0, 2

    def __init__(self, num_pages, key_width=6, value_width=4):
        self.keys = [
            torch.zeros((num_pages * PAGE, key_width), dtype=torch.bfloat16)
            for _ in range(self.layer_num)
        ]
        self.values = [
            torch.zeros((num_pages * PAGE, value_width), dtype=torch.bfloat16)
            for _ in range(self.layer_num)
        ]

    def _get_key_buffer(self, layer):
        return self.keys[layer]

    def _get_value_buffer(self, layer):
        return self.values[layer]

    def _disagg_page_nbytes(self, buffer):
        return int(buffer[0].nbytes * PAGE)

    def fill_pages(self, pages, content):
        for buffers, key in ((self.keys, "k"), (self.values, "v")):
            for layer, buffer in enumerate(buffers):
                for page, data in zip(pages, content[key][layer]):
                    buffer[page * PAGE : (page + 1) * PAGE] = data


def test_checksum_follows_window_order_and_content():
    generator = torch.Generator().manual_seed(7)
    num = 5
    content = {
        key: [
            torch.randn((num, PAGE, width), generator=generator).to(torch.bfloat16)
            for _ in range(FakeDraftPool.layer_num)
        ]
        for key, width in (("k", 6), ("v", 4))
    }
    prefill, decode = FakeDraftPool(40), FakeDraftPool(40)
    src_pages, dst_pages = [31, 2, 17, 8, 23], [36, 37, 38, 21, 22]
    prefill.fill_pages(src_pages, content)
    decode.fill_pages(dst_pages, content)
    reference = dflash_draft_window_checksum(prefill, src_pages)
    assert reference == dflash_draft_window_checksum(decode, np.array(dst_pages))
    # Two pages in the wrong order, one wrong element, or a missing page all show.
    swapped = [dst_pages[1], dst_pages[0]] + dst_pages[2:]
    assert dflash_draft_window_checksum(decode, swapped) != reference
    assert dflash_draft_window_checksum(decode, dst_pages[:-1]) != reference
    decode.values[1][dst_pages[3] * PAGE + 5, 2] += 1
    assert dflash_draft_window_checksum(decode, dst_pages) != reference
    assert dflash_draft_window_checksum(decode, []) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_ring_kernel_matches_reference(index_dtype):
    from sglang.srt.speculative.triton_ops.dflash_compact_kv import (
        ring_dflash_req_to_token,
    )

    lens = [0, 1, 63, 64, 65, 1023, 1024, 1025, 1087, 1088, 1151, 1152, 1153, 1048000]
    rows = len(lens) + 5
    prefix = torch.tensor(lens, device="cuda", dtype=index_dtype)
    requests = torch.randperm(rows, device="cuda", dtype=index_dtype)[: len(lens)]
    draft = torch.full((rows, RING + 9), -317, device="cuda", dtype=torch.int32)
    block = torch.full((len(lens), BLOCK), -5, device="cuda", dtype=torch.int64)
    lengths = torch.full((len(lens),), -1, device="cuda", dtype=torch.int32)
    ring_dflash_req_to_token(
        prefix,
        requests,
        draft[:, :RING],
        block,
        lengths,
        window_size=WINDOW,
        page_size=PAGE,
        ring_size=RING,
    )
    expected = torch.full_like(draft, -317)
    for i, (n, row) in enumerate(zip(lens, requests.tolist())):
        table, block_slots, length = ring_table(n, row)
        expected[row, : len(table)] = torch.from_numpy(table.astype(np.int32))
        assert lengths[i].item() == length
        assert block[i].tolist() == block_slots.tolist()
    torch.testing.assert_close(draft, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_ring_kernel_graph_replay_follows_new_lengths():
    from sglang.srt.speculative.triton_ops.dflash_compact_kv import (
        ring_dflash_req_to_token,
    )

    prefix = torch.tensor([1, 1024, 1089, 4095], device="cuda", dtype=torch.int64)
    requests = torch.tensor([6, 2, 4, 1], device="cuda", dtype=torch.int32)
    draft = torch.full((8, RING), -23, device="cuda", dtype=torch.int32)
    block = torch.empty((4, BLOCK), device="cuda", dtype=torch.int64)
    lengths = torch.empty(4, device="cuda", dtype=torch.int32)

    def run():
        ring_dflash_req_to_token(
            prefix,
            requests,
            draft,
            block,
            lengths,
            window_size=WINDOW,
            page_size=PAGE,
            ring_size=RING,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for next_prefix in [[63, 65, 1023, 1088], [4096, 4097, 8191, 1152]]:
        prefix.copy_(torch.tensor(next_prefix, device="cuda"))
        requests.copy_(requests.roll(1))
        graph.replay()
        for i, (n, row) in enumerate(zip(next_prefix, requests.tolist())):
            table, block_slots, length = ring_table(n, row)
            assert draft[row, : len(table)].tolist() == table.tolist()
            assert block[i].tolist() == block_slots.tolist()
            assert lengths[i].item() == length


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_ring_kernel_warmup_runs_on_scratch_tensors():
    from sglang.srt.speculative.triton_ops.dflash_compact_kv import (
        warmup_ring_dflash_req_to_token,
    )

    warmup_ring_dflash_req_to_token(
        "cuda", window_size=WINDOW, page_size=PAGE, block_size=BLOCK, ring_size=RING
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
