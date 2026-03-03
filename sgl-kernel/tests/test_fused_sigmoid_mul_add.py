# Unit test for fused sigmoid-mul-add kernel (PR #19631).
# Dev tree: /root/workspace/qwen3.5/fused_sigmoid_mul_add/sglang
# Uses a local copy of the Gluon kernel (fused_sigmoid_mul_add_gluon.py) so tests
# run without registering torch.ops.sgl_kernel.
#
# Kernel API: fused_sigmoid_mul_add_gluon(gate, shared_output, final_hidden_states)
#   In-place: final_hidden_states += sigmoid(gate) * shared_output
#   gate: [num_tokens, 1], shared_output / final_hidden_states: [num_tokens, hidden_size]
#
# Production shapes from _forward_shared_experts (qwen3.5_moe_shape_disable_cudagraph.log):
#   hidden_states / shared_expert_out: [80, 4096] torch.bfloat16 (Qwen3.5-397B-A17B prefill)

import os
import sys

import pytest
import torch

# Ensure tests dir is on path so we can import the local gluon kernel copy.
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Prefer local Gluon kernel copy so we don't need torch.ops registration.
try:
    from fused_sigmoid_mul_add_gluon import fused_sigmoid_mul_add_gluon as _gluon_kernel
    _GLUON_AVAILABLE = True
except Exception:
    _GLUON_AVAILABLE = False


def _ref_fused_sigmoid_mul_add(gate: torch.Tensor, shared: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Reference: out + sigmoid(gate) * shared (gate broadcast per row)."""
    # gate [N, 1], shared [N, H], out [N, H]
    return out + torch.sigmoid(gate) * shared


def _get_kernel():
    """Return the kernel to use: local Gluon copy if available, else torch.ops."""
    if _GLUON_AVAILABLE:
        return _gluon_kernel
    try:
        op = getattr(torch.ops.sgl_kernel, "fused_sigmoid_mul_add", None)
        if op is not None:
            return getattr(op, "default", op)
        op = getattr(torch.ops.sgl_kernel, "fused_sigmoid_mul_add_gluon", None)
        if op is not None:
            return getattr(op, "default", op)
    except Exception:
        pass
    return None


_KERNEL = _get_kernel()
_KERNEL_MISSING = _KERNEL is None


def _run_gluon_kernel(gate, shared, out, device, dtype):
    """Run Gluon kernel (in-place): out += sigmoid(gate) * shared."""
    out_copy = out.clone()
    _gluon_kernel(gate, shared, out_copy)
    return out_copy


# All (num_tokens, hidden_size, dtype) cases: grid (1,8,128)x(128,4096)x(fp16,bf16) + MoE production + edge.
# MoE/edge from qwen3.5_moe_shape_disable_cudagraph.log and small/large hidden_size.
FUSED_SIGMOID_MUL_ADD_CASES = [
    # Grid: num_tokens x hidden_size x dtype
    pytest.param(1, 128, torch.float16, id="1x128_fp16"),
    pytest.param(1, 128, torch.bfloat16, id="1x128_bf16"),
    pytest.param(1, 4096, torch.float16, id="1x4096_fp16"),
    pytest.param(1, 4096, torch.bfloat16, id="1x4096_bf16"), # decode
    pytest.param(8, 128, torch.float16, id="8x128_fp16"),
    pytest.param(8, 128, torch.bfloat16, id="8x128_bf16"),
    pytest.param(8, 4096, torch.float16, id="8x4096_fp16"),
    pytest.param(8, 4096, torch.bfloat16, id="8x4096_bf16"),
    pytest.param(128, 128, torch.float16, id="128x128_fp16"),
    pytest.param(128, 128, torch.bfloat16, id="128x128_bf16"),
    pytest.param(128, 4096, torch.float16, id="128x4096_fp16"),
    pytest.param(128, 4096, torch.bfloat16, id="128x4096_bf16"),
    # MoE production + edge
    pytest.param(80, 4096, torch.bfloat16, id="80x4096_bf16"), # prefill
    pytest.param(1, 16, torch.float16, id="1x16_fp16"),
    pytest.param(1, 16, torch.bfloat16, id="1x16_bf16"),
    pytest.param(2, 256, torch.float16, id="2x256_fp16"),
    pytest.param(2, 256, torch.bfloat16, id="2x256_bf16"),
    pytest.param(1024, 5120, torch.float16, id="1024x5120_fp16"),
    pytest.param(1024, 5120, torch.bfloat16, id="1024x5120_bf16"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(_KERNEL_MISSING, reason="fused_sigmoid_mul_add not available (need gluon or PR #19631)")
@pytest.mark.parametrize("num_tokens, hidden_size, dtype", FUSED_SIGMOID_MUL_ADD_CASES)
def test_fused_sigmoid_mul_add_kernel(num_tokens, hidden_size, dtype):
    """Test fused sigmoid-mul-add: out += sigmoid(gate) * shared vs reference.

    Covers grid (1,8,128)x(128,4096)x(fp16,bf16), MoE production shapes from
    qwen3.5_moe_shape_disable_cudagraph.log, and edge cases (small/large hidden_size).
    """
    device = torch.device("cuda", 0)
    gate = torch.randn(num_tokens, 1, device=device, dtype=dtype)
    shared = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    z = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    ref = _ref_fused_sigmoid_mul_add(gate, shared, z)

    if _GLUON_AVAILABLE:
        out = _run_gluon_kernel(gate, shared, z, device, dtype)
    else:
        out = torch.empty_like(z)
        try:
            _KERNEL(out, gate, shared, z)
        except TypeError:
            out = _KERNEL(gate, shared, z)

    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def _benchmark_impl(run_fn, num_tokens: int, hidden_size: int, dtype: torch.dtype, warmup: int = 10, repeat: int = 100):
    """Time run_fn() with torch.cuda.Event; returns mean ms and GB/s (same bytes formula as kernel)."""
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(repeat):
        run_fn()
    end_ev.record()
    torch.cuda.synchronize()
    elapsed_ms = start_ev.elapsed_time(end_ev)
    mean_ms = elapsed_ms / repeat
    # Bytes read (gate, shared, out) + written (out)
    es = torch.empty(0, dtype=dtype).element_size()
    bytes_per_run = (num_tokens + num_tokens * hidden_size + 2 * num_tokens * hidden_size) * es
    gbps = (bytes_per_run / 1e9) / (mean_ms / 1000) if mean_ms > 0 else 0
    return mean_ms, gbps


def _benchmark_kernel(num_tokens: int, hidden_size: int, dtype: torch.dtype, warmup: int = 10, repeat: int = 100):
    """Benchmark fused kernel; returns (mean_ms, gbps)."""
    device = torch.device("cuda", 0)
    gate = torch.randn(num_tokens, 1, device=device, dtype=dtype)
    shared = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    z = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    if _GLUON_AVAILABLE:
        def run():
            out = z.clone()
            _gluon_kernel(gate, shared, out)
    else:
        def run():
            out = z.clone()
            try:
                _KERNEL(out, gate, shared, z)
            except TypeError:
                _KERNEL(gate, shared, out)

    return _benchmark_impl(run, num_tokens, hidden_size, dtype, warmup, repeat)


def _benchmark_reference(num_tokens: int, hidden_size: int, dtype: torch.dtype, warmup: int = 10, repeat: int = 100):
    """Benchmark PyTorch one-liner (may be fused by PyTorch); returns (mean_ms, gbps)."""
    device = torch.device("cuda", 0)
    gate = torch.randn(num_tokens, 1, device=device, dtype=dtype)
    shared = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    z = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    def run():
        _ref_fused_sigmoid_mul_add(gate, shared, z)

    return _benchmark_impl(run, num_tokens, hidden_size, dtype, warmup, repeat)


# Benchmark shapes: MoE production + a couple of sizes.
BENCHMARK_CASES = [
    pytest.param(1, 4096, torch.bfloat16, id="1x4096_bf16"),
    pytest.param(80, 4096, torch.bfloat16, id="80x4096_bf16_moe"),
    pytest.param(128, 4096, torch.bfloat16, id="128x4096_bf16"),
    pytest.param(1024, 5120, torch.bfloat16, id="1024x5120_bf16"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(_KERNEL_MISSING, reason="fused_sigmoid_mul_add not available (need gluon or PR #19631)")
@pytest.mark.parametrize("num_tokens, hidden_size, dtype", BENCHMARK_CASES)
def test_benchmark_fused_sigmoid_mul_add_kernel(num_tokens, hidden_size, dtype):
    """Benchmark fused vs reference using torch.cuda.Event; report mean ms, GB/s, and speedup.

    Ref can be faster: PyTorch fuses the one-liner into an optimized kernel; our kernel uses
    FP32 for sigmoid (gl.exp requirement) which increases bandwidth. See fused_sigmoid_mul_add_gluon.py
    PERFORMANCE note for details.
    """
    fused_ms, fused_gbps = _benchmark_kernel(num_tokens, hidden_size, dtype)
    ref_ms, ref_gbps = _benchmark_reference(num_tokens, hidden_size, dtype)
    fused_us = fused_ms * 1000
    ref_us = ref_ms * 1000
    speedup = ref_ms / fused_ms if fused_ms > 0 else 0.0
    assert fused_us >= 0 and fused_us < 1e9, f"unexpected fused mean time: {fused_us} us"
    assert ref_us >= 0 and ref_us < 1e9, f"unexpected ref mean time: {ref_us} us"
    print(
        f"\n[benchmark] {num_tokens}x{hidden_size} {dtype}:\n"
        f"  fused: {fused_us:.2f} us/iter, {fused_gbps:.2f} GB/s\n"
        f"  ref:   {ref_us:.2f} us/iter, {ref_gbps:.2f} GB/s\n"
        f"  speedup: {speedup:.2f}x"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
