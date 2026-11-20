"""Numerical and graph regressions for the opt-in MiMo stable-weight fusion."""

import sys

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="MiMo stable-router weight fusion is enabled on ROCm",
)

if torch.cuda.is_available() and torch.version.hip:
    from sglang.srt.layers.moe.stable_router_weights import stable_router_weights


def reference(logits, ids, renormalize=True, scale=1.0):
    selected = logits.float().gather(1, ids.long())
    logp = F.logsigmoid(selected)
    return F.softmax(logp, dim=-1) * scale if renormalize else logp.exp()


@pytest.mark.parametrize("rows", [1, 2, 32, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("renormalize,scale", [(True, 1.0), (True, 2.5), (False, 1.0)])
def test_selected_weights(rows, dtype, renormalize, scale):
    torch.manual_seed(12345)
    logits = (torch.randn(rows, 512, device="cuda") * 35).to(dtype)[:, ::2]
    # Test both tensor strides, not just contiguous rows.
    ids = torch.randint(0, 256, (rows, 16), device="cuda", dtype=torch.int32)[:, ::2]
    actual = stable_router_weights(
        logits,
        ids,
        renormalize=renormalize,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=True,
    )
    expected = reference(logits, ids, renormalize, scale)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=3e-6)
    assert actual.dtype == torch.float32


def test_extreme_logits_and_nan_propagation():
    logits = torch.tensor(
        [
            [
                -10000.0,
                -10001.0,
                -10002.0,
                -10003.0,
                -10004.0,
                -10005.0,
                -10006.0,
                -10007.0,
            ],
            [float("inf"), -float("inf"), 0.0, 50.0, -50.0, 1000.0, -1000.0, -1.0],
            [-float("inf")] * 8,
            [float("nan"), 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
        device="cuda",
    )
    ids = torch.arange(8, dtype=torch.int64, device="cuda").expand(4, -1)
    for normalized in (True, False):
        got = stable_router_weights(logits, ids, renormalize=normalized)
        torch.testing.assert_close(
            got,
            reference(logits, ids, normalized),
            atol=2e-7,
            rtol=3e-6,
            equal_nan=True,
        )


def test_graph_reads_updated_inputs():
    logits = torch.randn(8, 256, device="cuda")
    ids = torch.arange(8, device="cuda", dtype=torch.int32).expand(8, -1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stable_router_weights(logits, ids, renormalize=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = stable_router_weights(logits, ids, renormalize=True)
    for shift in (0.0, -1000.0, 1000.0):
        logits.copy_(torch.randn_like(logits) + shift)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference(logits, ids), atol=2e-7, rtol=3e-6)


def padded_reference(logits, ids, num_valid, renormalize=True):
    """The separate HIP padding fills: expert id 0 and zero weight beyond num_valid."""
    weights, expected_ids = reference(logits, ids, renormalize), ids.clone()
    weights[num_valid:] = 0
    expected_ids[num_valid:] = 0
    return weights, expected_ids


@pytest.mark.parametrize(
    "rows,num_valid", [(1, 0), (1, 1), (64, 37), (256, 256), (257, 0), (1024, 1023)]
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
def test_fused_padding_matches_separate_fills(
    rows, num_valid, renormalize, count_dtype
):
    torch.manual_seed(2718)
    logits = (torch.randn(rows, 256, device="cuda") * 20).to(torch.bfloat16)
    # NaN in a padded row must not leak into its (zero) weights.
    logits[rows - 1, :] = float("nan")
    ids = torch.randint(0, 256, (rows, 8), device="cuda", dtype=torch.int32)
    expected, expected_ids = padded_reference(logits, ids, num_valid, renormalize)
    actual = stable_router_weights(
        logits,
        ids,
        renormalize=renormalize,
        num_token_non_padded=torch.tensor(
            [num_valid], device="cuda", dtype=count_dtype
        ),
    )
    torch.testing.assert_close(
        actual, expected, atol=2e-7, rtol=3e-6, equal_nan=num_valid == rows
    )
    assert torch.equal(ids, expected_ids)


def test_fused_padding_graph_follows_the_row_count():
    torch.manual_seed(31)
    rows = 96
    logits = torch.randn(rows, 256, device="cuda")
    source_ids = torch.randint(0, 256, (rows, 8), device="cuda", dtype=torch.int32)
    ids = source_ids.clone()
    num_valid = torch.tensor([rows], device="cuda", dtype=torch.int32)

    def run():
        # As in the router: the grouped top-k rewrites every id before the weights.
        ids.copy_(source_ids)
        return stable_router_weights(
            logits, ids, renormalize=True, num_token_non_padded=num_valid
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for count in (rows, 0, 41, 95, 1):
        num_valid.fill_(count)
        logits.copy_(torch.randn_like(logits) * 30)
        source_ids.copy_(torch.randint_like(source_ids, 0, 256))
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_ids = padded_reference(logits, source_ids, count)
        torch.testing.assert_close(output, expected, atol=2e-7, rtol=3e-6)
        assert torch.equal(ids, expected_ids)


def test_fused_padding_rejects_ids_it_cannot_rewrite():
    logits = torch.randn(4, 256, device="cuda")
    count = torch.tensor([2], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError):
        stable_router_weights(
            logits,
            torch.zeros(4, 8, dtype=torch.int64, device="cuda"),
            renormalize=True,
            num_token_non_padded=count,
        )
    with pytest.raises(ValueError):
        stable_router_weights(
            logits,
            torch.zeros(4, 8, dtype=torch.int32, device="cuda"),
            renormalize=True,
            num_token_non_padded=torch.tensor([2, 2], device="cuda", dtype=torch.int32),
        )


def test_empty_batch():
    actual = stable_router_weights(
        torch.empty(0, 256, device="cuda"),
        torch.empty(0, 8, dtype=torch.int32, device="cuda"),
        renormalize=True,
    )
    assert actual.shape == (0, 8) and actual.dtype == torch.float32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
