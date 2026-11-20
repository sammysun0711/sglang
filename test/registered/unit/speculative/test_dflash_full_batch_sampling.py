"""CPU checks for full-batch draft LM-head sampling with TP4 candidate exchange."""

import math
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative import dflash_worker_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")

# SGLANG_DFLASH_GREEDY_CHUNK_SIZE of the MiMo EP8 decode launcher: one GEMM for the
# whole candidate batch (7 rows per request, up to 585 requests per DP replica).
FULL_BATCH_CHUNK_SIZE = 4096


def worker_buffers():
    worker = SimpleNamespace()
    for name in ("local", "gather", "index"):
        setattr(worker, f"_draft_greedy_{name}_cap", 0)
    for name in (
        "local_max",
        "local_arg",
        "gathered_max",
        "gathered_ids",
        "best_rank",
        "rank_index",
        "selected_ids",
    ):
        setattr(worker, f"_draft_greedy_{name}_buf", None)
    return worker


class CandidateGroup:
    world_size = 4

    def __init__(self, rank, values, ids):
        self.rank = rank
        self.values, self.ids = values, ids
        self.calls = 0
        self.offset = 0

    def all_gather_into_tensor(self, output, local):
        # Only one maximum or token ID per row may cross TP, never vocab logits.
        assert local.ndim == 1
        rows = local.numel()
        payload = self.values if self.calls % 2 == 0 else self.ids
        current = payload[:, self.offset : self.offset + rows]
        torch.testing.assert_close(local, current[self.rank], rtol=0, atol=0)
        output.copy_(current.reshape(-1))
        if self.calls % 2:
            self.offset += rows
        self.calls += 1


@pytest.mark.parametrize("rows", [0, 1, 257, 1680, 2135, FULL_BATCH_CHUNK_SIZE])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_batch_matches_full_vocab_argmax_with_one_gemm(monkeypatch, rows, dtype):
    check_sampling(monkeypatch, rows, dtype, chunk_size=FULL_BATCH_CHUNK_SIZE)


def test_chunked_fallback_matches_full_batch(monkeypatch):
    check_sampling(monkeypatch, 1680, torch.float32, chunk_size=256)


def check_sampling(monkeypatch, rows, dtype, chunk_size):
    generator = torch.Generator().manual_seed(12345)
    local_vocab, hidden = 16, 8
    states = torch.randint(-2, 3, (rows, hidden), generator=generator).to(dtype)
    weight = torch.randint(-2, 3, (4 * local_vocab, hidden), generator=generator).to(
        dtype
    )
    if rows:
        states[0].zero_()  # A cross-shard tie must select the first global token.
    full_logits = states @ weight.T
    expected = full_logits.argmax(-1)
    values, ids = [], []
    for rank in range(4):
        value, token_id = full_logits[
            :, rank * local_vocab : (rank + 1) * local_vocab
        ].max(-1)
        values.append(value)
        ids.append(token_id + rank * local_vocab)
    values, ids = torch.stack(values), torch.stack(ids)
    original_matmul = torch.matmul
    calls = []

    def count_matmul(lhs, rhs):
        calls.append(tuple(lhs.shape))
        return original_matmul(lhs, rhs)

    def wrong_group():
        raise AssertionError("Draft vocabulary sampling must use attention TP4")

    monkeypatch.setattr(torch, "matmul", count_matmul)
    monkeypatch.setattr(dflash_worker_v2, "get_tp_group", wrong_group)
    expected_gemms = math.ceil(rows / chunk_size)
    for rank in range(4):
        group = CandidateGroup(rank, values, ids)
        monkeypatch.setattr(dflash_worker_v2, "get_attention_tp_group", lambda: group)
        head = SimpleNamespace(
            use_attn_tp_group=True,
            weight=weight[rank * local_vocab : (rank + 1) * local_vocab],
            shard_indices=SimpleNamespace(
                num_org_elements=local_vocab,
                num_org_elements_padded=local_vocab,
                num_added_elements=0,
                org_vocab_start_index=rank * local_vocab,
                added_vocab_start_index=4 * local_vocab,
            ),
        )
        calls.clear()
        actual = (
            dflash_worker_v2.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
                worker_buffers(),
                hidden_states=states,
                lm_head=head,
                chunk_size=chunk_size,
            )
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(calls) == expected_gemms
        assert group.calls == 2 * expected_gemms
        assert group.offset == rows
        if rows and chunk_size >= rows:
            assert calls == [(rows, hidden)]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
