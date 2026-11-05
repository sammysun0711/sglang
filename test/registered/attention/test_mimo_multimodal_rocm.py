"""Numerical checks for the real MiMo audio and vision paths on ROCm."""

import math

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import VisionAiterAttention
from sglang.srt.models.mimo_audio import AudioEncoderAttention, AudioRotaryEmbedding
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="Requires ROCm"
)


def packed_reference(q, k, v, lengths, *, causal=False, window=(-1, -1), sinks=None):
    """FP32 dense reference, independently masked per packed sequence."""
    results = []
    offset = 0
    for length in lengths:
        qs, ks, vs = [
            x[offset : offset + length].float().transpose(0, 1) for x in (q, k, v)
        ]
        ks = ks.repeat_interleave(qs.shape[0] // ks.shape[0], dim=0)
        vs = vs.repeat_interleave(qs.shape[0] // vs.shape[0], dim=0)
        row = torch.arange(length, device=q.device)[:, None]
        col = torch.arange(length, device=q.device)[None, :]
        mask = torch.ones((length, length), dtype=torch.bool, device=q.device)
        if causal:
            mask &= col <= row
        if window[0] >= 0:
            mask &= col >= row - window[0]
        if window[1] >= 0:
            mask &= col <= row + window[1]
        logits = (qs @ ks.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        logits.masked_fill_(~mask, float("-inf"))
        if sinks is not None:
            logits = torch.cat(
                (logits, sinks.float()[:, None, None].expand(-1, length, 1)), dim=-1
            )
        weights = torch.softmax(logits, dim=-1)[..., :length]
        results.append((weights @ vs).transpose(0, 1))
        offset += length
    return torch.cat(results)


@pytest.mark.parametrize(
    "window,causal", [((-1, -1), True), ((128, 0), True), ((-1, -1), False)]
)
@pytest.mark.parametrize("lengths", [(17,), (17, 257)])
@torch.inference_mode()
def test_audio_projection_rope_and_packed_attention(window, causal, lengths):
    torch.manual_seed(641)
    layer = (
        AudioEncoderAttention(1024, 16, window_size=window, causal=causal)
        .cuda()
        .bfloat16()
    )
    hidden = torch.randn(sum(lengths), 1024, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    rope = AudioRotaryEmbedding(10000, 64, max(lengths), device="cuda")
    cos, sin = rope(hidden, positions)
    actual = layer(hidden, cu, max(lengths), (cos, sin))
    q, k, v = [
        projection(hidden).reshape(-1, 16, 64)
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj)
    ]
    # Apply the checkpoint's split-half RoPE independently of the fused backend.
    q = q * cos[:, None] + torch.cat((-q[..., 32:], q[..., :32]), dim=-1) * sin[:, None]
    k = k * cos[:, None] + torch.cat((-k[..., 32:], k[..., :32]), dim=-1) * sin[:, None]
    expected = packed_reference(q, k, v, lengths, causal=causal, window=window)
    expected = F.linear(
        expected.reshape(-1, 1024).bfloat16(),
        layer.out_proj.weight,
        layer.out_proj.bias,
    )
    torch.testing.assert_close(actual, expected, atol=1.5e-2, rtol=1.5e-2)
    if len(lengths) > 1:
        changed = hidden.clone()
        changed[lengths[0] :] += 5
        second = layer(changed, cu, max(lengths), (cos, sin))
        torch.testing.assert_close(
            actual[: lengths[0]], second[: lengths[0]], atol=0, rtol=0
        )


@pytest.mark.parametrize("window", [(-1, -1), (128, 0)])
@torch.inference_mode()
def test_audio_prefix_cannot_read_future_frames(window):
    torch.manual_seed(175)
    layer = (
        AudioEncoderAttention(1024, 16, window_size=window, causal=True)
        .cuda()
        .bfloat16()
    )
    hidden = torch.randn(257, 1024, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 257], device="cuda", dtype=torch.int32)
    full = layer(hidden, cu, 257)
    changed = hidden.clone()
    changed[17:] += 5
    perturbed = layer(changed, cu, 257)
    torch.testing.assert_close(full[:17], perturbed[:17], atol=0, rtol=0)
    short = layer(
        hidden[:17], torch.tensor([0, 17], device="cuda", dtype=torch.int32), 17
    )
    torch.testing.assert_close(full[:17], short, atol=1.5e-2, rtol=1.5e-2)


@pytest.mark.parametrize("window,use_sinks", [((-1, -1), False), ((64, 64), True)])
@torch.inference_mode()
def test_vision_window_sinks_and_gqa(window, use_sinks):
    torch.manual_seed(99)
    lengths = (17, 257)
    q = torch.randn(sum(lengths), 8, 64, device="cuda", dtype=torch.bfloat16) * 0.3
    k = torch.randn(sum(lengths), 2, 64, device="cuda", dtype=torch.bfloat16) * 0.3
    v = torch.randn_like(k) + 0.5
    cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    sinks = torch.full((8,), math.log(257) + 2, device="cuda") if use_sinks else None
    actual = VisionAiterAttention()(
        q, k, v, cu, 1, sum(lengths), window_size=window, s_aux=sinks
    )
    expected = packed_reference(q, k, v, lengths, window=window, sinks=sinks)
    torch.testing.assert_close(actual.float(), expected, atol=1.5e-2, rtol=1.5e-2)
