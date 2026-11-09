"""Numerical regression tests for the gfx942 PyHIP blockscale linear path."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


@pytest.fixture
def gfx942(monkeypatch):
    # Keep hardware queries and optional GPU imports out of test collection.
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("Requires ROCm gfx942")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if "gfx942" not in properties.gcnArchName:
        pytest.skip("Requires gfx942")

    monkeypatch.setenv("SGLANG_USE_AITER", "1")
    monkeypatch.setenv("SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE", "1")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization import pyhip_fp8_blockscale_gemm as kernel

    # These globals may already be initialized by another collected test.
    monkeypatch.setattr(fp8_utils, "_use_aiter_bpreshuffle_gfx95", False)
    monkeypatch.setattr(fp8_utils, "_use_aiter_gfx95", False)
    monkeypatch.setattr(fp8_utils, "_use_aiter_ck_blockscale_bpreshuffle_gfx942", True)
    monkeypatch.setattr(
        fp8_utils,
        "gemm_a8w8_blockscale_bpreshuffle",
        gemm_a8w8_blockscale_bpreshuffle,
        raising=False,
    )
    calls = []

    def tracked_gemm(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return kernel.gemm_a8w8_blockscale_942(*args, **kwargs)

    monkeypatch.setattr(
        fp8_utils, "_load_pyhip_fp8_blockscale_gemm", lambda: tracked_gemm
    )
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield SimpleNamespace(
            fp8=fp8_utils,
            kernel=kernel,
            shuffle=shuffle_weight,
            device=device,
            cus=properties.multi_processor_count,
            generator=torch.Generator(device=device).manual_seed(12345),
            calls=calls,
        )
    finally:
        torch.set_float32_matmul_precision(precision)


def _inputs_and_reference(state, m, n, k, packed):
    kwargs = dict(device=state.device, dtype=torch.float32, generator=state.generator)
    stride = k + (k // 128 * 4 if packed else 0)
    storage = torch.randn(m, stride, **kwargs).to(torch.float8_e4m3fnuz)
    x = storage[:, :k]
    weight = torch.randn(n, k, **kwargs).to(torch.float8_e4m3fnuz)
    x_scale = torch.rand(k // 128, m, **kwargs) * 0.02 + 0.01
    w_scale = torch.rand((n + 127) // 128, k // 128, **kwargs) * 0.02 + 0.01
    # Independent reference: dequantize each block, then multiply in FP32.
    x_float = x.float() * x_scale.t().repeat_interleave(128, dim=1)
    w_float = weight.float() * w_scale.repeat_interleave(128, dim=0)[
        :n
    ].repeat_interleave(128, dim=1)
    expected = x_float @ w_float.t()
    return x, state.shuffle(weight, (16, 16)), x_scale, w_scale, expected


def _assert_numerics(actual, expected):
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    # Inspect boundary regions separately so partial-tile errors are not diluted
    # by the much larger interior. Match the existing manual check's RMS bound.
    regions = [(actual, expected), (actual[-1:], expected[-1:])]
    if expected.shape[1] % 256:
        tail = expected.shape[1] % 256
        regions.append((actual[:, -tail:], expected[:, -tail:]))
    for output, reference in regions:
        relative_rms = torch.linalg.vector_norm(output.float() - reference) / (
            torch.linalg.vector_norm(reference).clamp_min(1e-12)
        )
        assert relative_rms.item() < 0.003, relative_rms.item()


@pytest.mark.parametrize(
    "m,n,k,packed",
    [
        (16384, 3392, 4096, True),  # Full-attention TP QKV, partial N tile.
        (16384, 3712, 4096, True),  # SWA TP QKV, partial N tile.
        (4096, 14848, 4096, False),  # SWA SP full QKV.
        (4096, 32768, 4096, False),  # Dense gate/up projection.
        (4096, 4096, 16384, False),  # Dense down projection, larger K.
        (2049, 3712, 4096, True),  # One valid row in the final M tile.
        (4063, 14848, 4096, False),  # Partial M tile with full-head QKV.
    ],
)
@torch.inference_mode()
def test_pyhip_linear_matches_dequantized_reference(
    gfx942, monkeypatch, m, n, k, packed
):
    x, weight, xs, ws, expected = _inputs_and_reference(gfx942, m, n, k, packed)
    assert x.stride(0) == k + (k // 128 * 4 if packed else 0)
    # The linear wrapper also accepts this logical [M,K/128] presentation of
    # the column-major activation scale storage.
    scale_arg = xs.view(m, k // 128)
    for enabled in (False, True):
        monkeypatch.setattr(gfx942.fp8, "_use_pyhip_fp8_blockscale_gemm", enabled)
        before = len(gfx942.calls)
        actual = gfx942.fp8.aiter_w8a8_block_fp8_linear(
            x, weight, [128, 128], ws, input_scale=scale_arg
        )
        assert len(gfx942.calls) == before + int(enabled)
        _assert_numerics(actual, expected)


def _rows_for_plan(kernel, n, k, cus, split):
    # Select inputs using the real CU count, so both execution paths are covered
    # on gfx942 configurations with different numbers of available CUs.
    for m in range(2048, 32769, 256):
        tiles = ((m + 255) // 256) * ((n + 255) // 256)
        full, tail, _ = kernel._plan(tiles, k // 128, cus)
        if bool(tail) == split and full % 8 == 0:
            return m
    pytest.fail(f"No {'split' if split else 'unsplit'} plan found for {cus} CUs")


@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_split_and_unsplit_reuse_executable_with_different_m(
    gfx942, monkeypatch, packed
):
    kernel = gfx942.kernel
    n, k = 2080, 4096  # Also exercise the final partial N tile and scale block.
    split_m = _rows_for_plan(kernel, n, k, gfx942.cus, split=True)
    unsplit_m = _rows_for_plan(kernel, n, k, gfx942.cus, split=False)
    monkeypatch.setattr(kernel, "_compiled", {})
    executable = None
    for m, expect_split in (
        (split_m, True),
        (split_m - 1, True),
        (unsplit_m - 1, False),
    ):
        x, weight, xs, ws, expected = _inputs_and_reference(gfx942, m, n, k, packed)
        tiles = ((m + 255) // 256) * ((n + 255) // 256)
        full, tail, splits = kernel._plan(tiles, k // 128, gfx942.cus)
        assert bool(tail) == expect_split and full % 8 == 0
        kernel._partial_buffer(gfx942.device, gfx942.cus).fill_(float("nan"))
        guarded = torch.full(
            (m + 2, n), 42.0, dtype=torch.bfloat16, device=gfx942.device
        )
        out = guarded[1:-1]
        actual = kernel.gemm_a8w8_blockscale_942(
            x, weight, xs, ws, out=out, w_preshuffled=True
        )
        assert actual.data_ptr() == out.data_ptr()
        _assert_numerics(actual, expected)
        assert (guarded[0] == 42).all() and (guarded[-1] == 42).all()
        key = (gfx942.device, n, k, x.stride(0), True, True)
        if executable is None:
            executable = kernel._compiled[key]
        assert kernel._compiled[key] is executable
        if expect_split:
            assert splits >= 4
            assert ("reduce", gfx942.device, n, splits) in kernel._compiled


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
