"""Eight-rank PyHIP TP MoE versus AITER experts and RCCL reduce-scatter."""

import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=240, suite="stage-c-test-large-8-gpu-amd")

RANKS, HIDDEN, INTER, EXPERTS, TOPK = 8, 4096, 256, 256, 8
SIZES = (
    (64,) * RANKS,
    (0, 1, 17, 64, 0, 16, 31, 3),
    (3, 31, 16, 0, 64, 17, 1, 0),
    (96, 0, 5, 33, 2, 64, 16, 1),
)


class _Group:
    world_size = RANKS

    def __init__(self, rank):
        self.rank_in_group = rank

    def all_gather_into_tensor(self, output, value):
        dist.all_gather_into_tensor(output, value)


def _weights(rank, device):
    from aiter.ops.shuffle import shuffle_weight

    generator = torch.Generator(device=device).manual_seed(4100 + rank)
    kwargs = dict(device=device, generator=generator)
    w13 = shuffle_weight(
        torch.randn(EXPERTS, 2 * INTER, HIDDEN, **kwargs).to(torch.float8_e4m3fnuz),
        (16, 16),
    )
    w2 = shuffle_weight(
        torch.randn(EXPERTS, HIDDEN, INTER, **kwargs).to(torch.float8_e4m3fnuz),
        (16, 16),
    )
    w13.is_shuffled = w2.is_shuffled = True
    return SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale_inv=torch.rand(EXPERTS, 4, 32, **kwargs) * 0.02 + 0.01,
        w2_weight_scale_inv=torch.rand(EXPERTS, 32, 2, **kwargs) * 0.02 + 0.01,
    )


def _inputs(index, sizes, device):
    # Every TP rank sees the same gathered activations/routes; expert weights
    # differ by rank. Distinct block scales catch transposed-scale mistakes.
    generator = torch.Generator(device=device).manual_seed(6100 + index)
    kwargs = dict(device=device, generator=generator)
    tokens = sum(sizes)
    x = torch.randn(tokens, HIDDEN, **kwargs).to(torch.float8_e4m3fnuz)
    scale = torch.rand(HIDDEN // 128, tokens, **kwargs) * 0.05 + 0.025
    ids = torch.rand(tokens, EXPERTS, **kwargs).topk(TOPK, dim=-1).indices.int()
    if index % 2:
        # Many empty experts and long expert runs in addition to partial tiles.
        ids[: tokens // 2] = torch.arange(TOPK, dtype=torch.int32, device=device)
    weights = torch.rand(tokens, TOPK, **kwargs)
    weights /= weights.sum(-1, keepdim=True)
    return x, scale, weights, ids


def _reference(layer, inputs, sizes, rank):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    x, scale, weights, ids = inputs
    partial = fused_moe(
        x,
        layer.w13_weight,
        layer.w2_weight,
        weights,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        doweight_stage1=False,
        w1_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
        a1_scale=scale.t().contiguous(),
        dtype=torch.bfloat16,
    )
    # Pad unequal shards for RCCL's fixed-count API. FP32 rank reduction with
    # one final BF16 rounding matches the fused IPC kernel's arithmetic class.
    maximum = max(sizes)
    packed = torch.zeros(RANKS, maximum, HIDDEN, device=x.device, dtype=torch.float32)
    offset = 0
    for owner, size in enumerate(sizes):
        packed[owner, :size] = partial[offset : offset + size]
        offset += size
    result = torch.empty(maximum, HIDDEN, device=x.device, dtype=torch.float32)
    dist.reduce_scatter_tensor(result, packed.view(-1, HIDDEN))
    return result[: sizes[rank]].to(torch.bfloat16)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=RANKS,
        timeout=timedelta(seconds=900),
    )
    try:
        from sglang.srt.layers.moe import pyhip_fp8_blockscale_moe as kernel

        device = torch.device("cuda", rank)
        layer = _weights(rank, device)
        group = _Group(rank)
        fixtures = [_inputs(i, sizes, device) for i, sizes in enumerate(SIZES)]
        # Queue changing shard sizes without host synchronization or barriers
        # between calls. This reuses the same route workspaces, inbox and epochs.
        outputs = [
            kernel.moe_reduce_scatter(layer, *inputs, list(sizes), group, max_rows=96)
            for inputs, sizes in zip(fixtures, SIZES)
        ]
        torch.cuda.synchronize()
        for index, (actual, inputs, sizes) in enumerate(zip(outputs, fixtures, SIZES)):
            expected = _reference(layer, inputs, sizes, rank)
            assert actual.shape == expected.shape == (sizes[rank], HIDDEN)
            assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
            if actual.numel():
                error = actual.float() - expected.float()
                rms = expected.float().square().mean().sqrt().clamp_min(1e-12)
                relative_rms = error.square().mean().sqrt() / rms
                normalized_max = error.abs().max() / rms
                print(
                    f"rank={rank} case={index} rows={sizes[rank]} "
                    f"relative_rms={relative_rms.item():.6f} "
                    f"normalized_max={normalized_max.item():.6f}",
                    flush=True,
                )
                assert relative_rms < 0.01
                assert normalized_max < 0.1
        assert kernel._reduce_scatter.epoch == len(SIZES)
    finally:
        dist.destroy_process_group()


def test_tp_moe_matches_aiter_and_rccl(monkeypatch):
    if not torch.version.hip or torch.cuda.device_count() < RANKS:
        pytest.skip("Requires eight gfx942 GPUs on one node")
    if any(
        "gfx942" not in torch.cuda.get_device_properties(i).gcnArchName
        for i in range(RANKS)
    ):
        pytest.skip("Requires gfx942")
    monkeypatch.setenv("AITER_FMOE_FP8_BLOCKSCALE_FLYDSL", "0")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _worker,
            args=(str(Path(directory) / "rendezvous"),),
            nprocs=RANKS,
            join=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
