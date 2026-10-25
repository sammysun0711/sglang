"""Validate reusable KV gather storage for growing MiMo contexts."""

import pytest
import torch

from sglang.srt.layers.attention.utils import (
    launch_gather_shuffle_5d_to_linear,
    launch_reshape_and_cache_shuffle_5d,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype,vector_width", [(torch.bfloat16, 8), (torch.uint8, 16)])
@pytest.mark.parametrize("value_dim", [128, 192])
def test_growing_and_reused_workspace_preserves_paged_kv(
    dtype, vector_width, value_dim
):
    workspace = {}
    # Full and SWA attention have different local head counts. They share the
    # backend but must retain independent scratch storage.
    for heads in (1, 2):
        tokens = 257
        slots = torch.randperm(320, device="cuda")[:tokens].to(torch.int64)
        key = (
            torch.arange(tokens * heads * 192, device="cuda")
            .remainder(251)
            .to(dtype)
            .view(tokens, heads, 192)
        )
        value = (
            torch.arange(tokens * heads * value_dim, device="cuda")
            .add(17)
            .remainder(251)
            .to(dtype)
            .view(tokens, heads, value_dim)
        )
        key_cache = torch.zeros(
            (5, heads, 192 // vector_width, 64, vector_width),
            dtype=dtype,
            device="cuda",
        )
        value_cache = torch.zeros(
            (5, heads, 64 // vector_width, value_dim, vector_width),
            dtype=dtype,
            device="cuda",
        )
        launch_reshape_and_cache_shuffle_5d(key, value, key_cache, value_cache, slots)

        for count in (33, 65, 129, 257):
            gathered = launch_gather_shuffle_5d_to_linear(
                key_cache, value_cache, slots[:count], workspace=workspace
            )
            torch.testing.assert_close(gathered[0], key[:count], rtol=0, atol=0)
            torch.testing.assert_close(gathered[1], value[:count], rtol=0, atol=0)

        pointers = tuple(buffer.data_ptr() for buffer in gathered)
        # Changing slot order must refresh contents without allocating another
        # pair of buffers for a smaller context.
        reordered = launch_gather_shuffle_5d_to_linear(
            key_cache, value_cache, slots[:65].flip(0), workspace=workspace
        )
        assert tuple(buffer.data_ptr() for buffer in reordered) == pointers
        torch.testing.assert_close(reordered[0], key[:65].flip(0), rtol=0, atol=0)
        torch.testing.assert_close(reordered[1], value[:65].flip(0), rtol=0, atol=0)
    assert len(workspace) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_graph_capture_does_not_alias_resizable_workspace():
    key_cache = torch.ones((2, 1, 24, 64, 8), device="cuda", dtype=torch.bfloat16)
    value_cache = torch.full((2, 1, 8, 128, 8), 2, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(16, device="cuda", dtype=torch.int64)
    workspace = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup = launch_gather_shuffle_5d_to_linear(
            key_cache, value_cache, slots, workspace=workspace
        )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch_gather_shuffle_5d_to_linear(
            key_cache, value_cache, slots, workspace=workspace
        )
    assert captured[0].data_ptr() != warmup[0].data_ptr()
    launch_gather_shuffle_5d_to_linear(
        key_cache,
        value_cache,
        torch.arange(128, device="cuda", dtype=torch.int64),
        workspace=workspace,
    )
    key_cache.fill_(3)
    value_cache.fill_(4)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured[0], torch.full_like(captured[0], 3))
    torch.testing.assert_close(captured[1], torch.full_like(captured[1], 4))
