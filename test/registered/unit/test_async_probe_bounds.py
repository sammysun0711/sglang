"""The fused probe must still reject invalid values, including after graph capture."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.utils.async_probe import maybe_detect_oob


@pytest.fixture(autouse=True)
def enable_probes(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "1")
    monkeypatch.setenv("SGLANG_FUSED_ASYNC_OOB_CHECKS", "1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("size", [1, 8, 257, 16384])
def test_valid_bounds_and_partial_blocks(dtype, size):
    indices = torch.arange(size, device="cuda", dtype=dtype) % 129 - 4
    before = indices.clone()
    maybe_detect_oob(indices, -4, 125, "valid indices")
    torch.cuda.synchronize()
    torch.testing.assert_close(indices, before, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_valid_bounds_graph_replay_and_wide_int64_values():
    low = 2**33
    indices = torch.arange(257, device="cuda", dtype=torch.int64) + low
    maybe_detect_oob(indices, low, low + 1024, "wide indices")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        maybe_detect_oob(indices, low, low + 1024, "wide indices")
    for delta in [1, 7, 31]:
        indices.add_(delta)
        graph.replay()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_noncontiguous_indices_keep_the_existing_probe():
    storage = torch.full((16,), 999, device="cuda", dtype=torch.int64)
    indices = storage[::2]
    indices.copy_(torch.arange(8, device="cuda"))
    maybe_detect_oob(indices, 0, 8, "strided indices")
    torch.cuda.synchronize()


def test_cpu_fallback_still_checks_bounds():
    with pytest.raises(RuntimeError, match="index < 0"):
        maybe_detect_oob(torch.tensor([-1, 3]), 0, 4, "CPU fallback")


def test_disabled_and_empty_probes(monkeypatch):
    maybe_detect_oob(None, 0, 4, "none")
    maybe_detect_oob(torch.empty(0, dtype=torch.int64), 0, 4, "empty")
    monkeypatch.setenv("SGLANG_ENABLE_ASYNC_ASSERT", "0")
    maybe_detect_oob(torch.tensor([-99, 999]), 0, 4, "disabled")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
@pytest.mark.parametrize(
    "value,message", [(-1, "negative / unmasked sentinel"), (17, "out of range")]
)
@pytest.mark.parametrize("graph", [False, True])
def test_invalid_gpu_values_raise_even_with_triton_debug_disabled(
    value, message, graph
):
    # Device assertions poison the CUDA/HIP context, so isolate each negative case.
    source = f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import torch
from sglang.srt.utils.async_probe import maybe_detect_oob
x = torch.arange(8, device='cuda', dtype=torch.int64)
maybe_detect_oob(x, 0, 17, 'negative test')
torch.cuda.synchronize()
if {graph!r}:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        maybe_detect_oob(x, 0, 17, 'negative test')
    x[7] = {value}
    g.replay()
else:
    x[7] = {value}
    maybe_detect_oob(x, 0, 17, 'negative test')
torch.cuda.synchronize()
raise SystemExit('ERROR: invalid index was accepted')
"""
    env = os.environ.copy()
    env["TRITON_DEBUG"] = "0"
    python_root = str(Path(__file__).resolve().parents[3] / "python")
    env["PYTHONPATH"] = python_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "ERROR: invalid index was accepted" not in result.stdout + result.stderr
    assert message in result.stdout + result.stderr
    diagnostic = (result.stdout + result.stderr).lower()
    assert "device assertion failed" in diagnostic or "device-side assert" in diagnostic
