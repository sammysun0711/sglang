"""Exact request-table equivalence for fused DFlash window compaction."""

import pytest
import torch

from sglang.srt.speculative.triton_ops.dflash_compact_kv import (
    compact_dflash_req_to_token,
)


def reference(prefix, requests, target, draft, verify, window, page):
    expected = draft.clone()
    lengths = []
    for row, (n, req) in enumerate(zip(prefix.tolist(), requests.tolist())):
        start = max(0, n - window) // page * page
        length = n - start
        expected[req, :length] = target[req, start:n]
        expected[req, length : length + verify.shape[1]] = verify[row]
        lengths.append(length)
    return expected, torch.tensor(lengths, dtype=torch.int32, device=prefix.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("window,page", [(16, 1), (128, 64), (1024, 64)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_compact_matches_gather_and_assign(window, page, index_dtype, strided):
    torch.manual_seed(719)
    lens = [0, 1, page - 1, page, page + 1, window - 1, window, window + 1, 4095]
    batch, rows, verify_size = len(lens), len(lens) + 7, 8
    step = 2 if strided else 1
    prefix = torch.empty(batch * step, device="cuda", dtype=index_dtype)[::step]
    prefix.copy_(torch.tensor(lens, device="cuda", dtype=index_dtype))
    requests = torch.randperm(rows, device="cuda", dtype=index_dtype)[:batch]
    target_storage = torch.randint(
        1, 2**28, (rows, 4160 + 32), device="cuda", dtype=index_dtype
    )
    target = target_storage[:, :4160]
    width = window + page - 1 + verify_size + 11
    draft_storage = torch.full(
        (rows, width + 17), -317, device="cuda", dtype=index_dtype
    )
    draft = draft_storage[:, :width]
    verify = torch.randint(
        1, 2**28, (batch, verify_size + 3), device="cuda", dtype=torch.int64
    )[:, :verify_size]
    lengths = torch.full((batch * step,), -1, device="cuda", dtype=torch.int32)[::step]
    before = target.clone()
    expected, expected_lengths = reference(
        prefix, requests, target, draft, verify, window, page
    )
    compact_dflash_req_to_token(
        prefix,
        requests,
        target,
        draft,
        verify,
        lengths,
        window_size=window,
        page_size=page,
    )
    torch.testing.assert_close(draft, expected, rtol=0, atol=0)
    torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)
    torch.testing.assert_close(target, before, rtol=0, atol=0)
    assert torch.all(draft_storage[:, width:] == -317)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_compact_graph_replay_changes_lengths_and_request_order():
    torch.manual_seed(920)
    prefix = torch.tensor([1, 1024, 1089, 4095], device="cuda", dtype=torch.int64)
    requests = torch.tensor([6, 2, 4, 0], device="cuda", dtype=torch.int32)
    target = torch.randint(0, 2**28, (8, 8192), device="cuda", dtype=torch.int32)
    draft = torch.full((8, 1152), -23, device="cuda", dtype=torch.int32)
    verify = torch.randint(0, 2**28, (4, 8), device="cuda", dtype=torch.int64)
    lengths = torch.empty(4, device="cuda", dtype=torch.int32)

    def run():
        compact_dflash_req_to_token(
            prefix,
            requests,
            target,
            draft,
            verify,
            lengths,
            window_size=1024,
            page_size=64,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for next_prefix in [[63, 65, 1023, 1088], [4096, 4097, 8191, 0]]:
        prefix.copy_(torch.tensor(next_prefix, device="cuda"))
        requests.copy_(requests.roll(1))
        verify.add_(31)
        draft.fill_(-23)
        expected, expected_lengths = reference(
            prefix, requests, target, draft, verify, 1024, 64
        )
        graph.replay()
        torch.testing.assert_close(draft, expected, rtol=0, atol=0)
        torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_compact_uses_64bit_request_table_offsets():
    # A short view of the final row has a >2 GiB byte offset from the base.
    target = torch.empty((513, 1048576), device="cuda", dtype=torch.int32)
    prefix = torch.tensor([1048575], device="cuda", dtype=torch.int64)
    requests = torch.tensor([512], device="cuda", dtype=torch.int64)
    start = (1048575 - 1024) // 64 * 64
    target[512, start:1048575] = torch.arange(1048575 - start, device="cuda")
    draft = torch.full((513, 1152), -9, device="cuda", dtype=torch.int32)
    verify = torch.arange(8, device="cuda", dtype=torch.int64).view(1, 8) + 98765
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    compact_dflash_req_to_token(
        prefix, requests, target, draft, verify, lengths, window_size=1024, page_size=64
    )
    length = 1048575 - start
    assert lengths.item() == length
    torch.testing.assert_close(
        draft[512, :length],
        torch.arange(length, device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        draft[512, length : length + 8], verify[0].int(), rtol=0, atol=0
    )
    assert torch.all(draft[:512] == -9)


def test_compact_rejects_aliasing_tables():
    table = torch.zeros((1, 128), dtype=torch.int32)
    with pytest.raises(ValueError, match="must not alias"):
        compact_dflash_req_to_token(
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            table,
            table,
            torch.zeros((1, 8), dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
            window_size=16,
            page_size=1,
        )
