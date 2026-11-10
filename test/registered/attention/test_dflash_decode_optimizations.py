"""Numerical and replay checks for the DPA decode verification helpers."""
import pytest
import torch
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")
pytestmark = pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm")


@pytest.mark.parametrize("reference_rows,owned_rows", [(240, 60), (480, 120)])
def test_request_owner_scan_matches_full_batch(reference_rows, owned_rows):
    from sglang.srt.speculative.request_owner_scan import cumsum_like_full_batch
    torch.manual_seed(17)
    full = torch.rand(reference_rows, 152576, device="cuda", dtype=torch.float32)
    owned = full[:owned_rows]
    actual = cumsum_like_full_batch(owned, reference_rows)
    torch.testing.assert_close(actual, full.cumsum(-1)[:owned_rows], atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = cumsum_like_full_batch(owned, reference_rows)
    full.mul_(.75)
    graph.replay()
    torch.testing.assert_close(actual, full.cumsum(-1)[:owned_rows], atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 4])
def test_probability_sort_keeps_values_and_token_ids(rows):
    from sglang.srt.speculative.probability_sort32 import sort_probabilities
    torch.manual_seed(19)
    probabilities = torch.randint(0, 16, (rows, 152576), device="cuda").float() / 16
    actual, ids = sort_probabilities(probabilities)
    expected, expected_ids = probabilities.sort(dim=-1, descending=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    assert ids.dtype == torch.int64
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, ids = sort_probabilities(probabilities)
    probabilities.copy_(torch.rand_like(probabilities))
    graph.replay()
    expected, expected_ids = probabilities.sort(dim=-1, descending=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)


@pytest.mark.parametrize("threshold_single,threshold_acc", [(1., 1.), (1., .7), (.5, 1.3)])
def test_fused_chain_matches_prefix_and_rejection_weights(threshold_single, threshold_acc):
    from sglang.srt.speculative.dflash_chain_fused import chain_weights
    torch.manual_seed(23)
    batch, tokens, vocab = 8, 8, 33
    probs = torch.softmax(torch.randn(batch, tokens, vocab, device="cuda"), dim=-1)
    candidates = torch.randint(vocab, (batch, tokens), device="cuda", dtype=torch.int64)
    coins = torch.ones(batch, tokens, device="cuda")
    for row in range(batch):
        coins[row, :row] = 0

    def reference():
        probability = probs[:, :-1].gather(-1, candidates[:, 1:, None]).squeeze(-1)
        accepted = (coins[:, :-1] <= probability / max(threshold_acc, 1e-9)) | (probability >= threshold_single)
        count = accepted.int().cumprod(-1).sum(-1)
        weights = probs[torch.arange(batch, device="cuda"), count].clone()
        rejected = count < tokens - 1
        rows = torch.arange(batch, device="cuda")[rejected]
        weights[rows, candidates[rows, count[rejected] + 1]] = 0
        return count.int(), weights

    def run():
        return chain_weights(probs, candidates, coins, threshold_single=threshold_single, threshold_acc=threshold_acc)

    result = run()
    for actual, expected in zip(result, reference()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    coins.copy_(torch.rand_like(coins))
    graph.replay()
    for actual, expected in zip(result, reference()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
