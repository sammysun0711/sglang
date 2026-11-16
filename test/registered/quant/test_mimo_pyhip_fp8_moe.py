"""Eight-rank PyHIP TP MoE versus AITER experts and RCCL reduce-scatter."""

import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=240,
    suite="stage-c-test-large-8-gpu-amd",
    disabled="Manual qualification in this fork",
)

RANKS, HIDDEN, INTER, EXPERTS, TOPK = 8, 4096, 256, 256, 8
SIZES = (
    (64,) * RANKS,
    (0, 1, 17, 64, 0, 16, 31, 3),
    (3, 31, 16, 0, 64, 17, 1, 0),
    (96, 0, 5, 33, 2, 64, 16, 1),
    (0,) * RANKS,
    (16,) * RANKS,
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
    if not sum(sizes):
        return torch.empty(0, HIDDEN, device=x.device, dtype=torch.bfloat16)
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


def _check_reduce_scatter_epochs(rank, group, kernel, device, diagnostics):
    reducer = kernel._SumReduceScatter(group, 96, device)
    outputs = []
    for index, sizes in enumerate(SIZES):
        if rank == 6 and index == 1:
            # Delay this producer while empty owners can otherwise advance.
            torch.cuda.synchronize()
            time.sleep(0.05)
        tokens = sum(sizes)
        route = torch.arange(tokens * TOPK, device=device)
        values = (route // TOPK % 4).float() / 16 + rank + 1 + index * 0.5
        y = values[:, None].expand(-1, HIDDEN).contiguous().to(torch.bfloat16)
        inv = route.reshape(tokens, TOPK).int()
        outputs.append(reducer(y, inv, list(sizes)))
    torch.cuda.synchronize()
    failures = []
    for index, (actual, sizes) in enumerate(zip(outputs, SIZES)):
        begin = sum(sizes[:rank])
        token = torch.arange(begin, begin + sizes[rank], device=device)
        # All additions are exactly representable in BF16 for this fixture.
        expected = (288 + 32 * index + 4 * (token % 4)).to(torch.bfloat16)
        expected = expected[:, None].expand(-1, HIDDEN)
        if not torch.equal(actual, expected):
            torch.save(
                dict(actual=actual.cpu(), expected=expected.cpu(), sizes=sizes),
                Path(diagnostics) / f"ipc-rank{rank}-case{index}.pt",
            )
            failures.append(index)
    assert not failures, f"IPC epoch mismatch on rank {rank}: cases {failures}"
    assert reducer.epoch == sum(bool(sum(s)) for s in SIZES)


def _worker(rank, rendezvous, diagnostics):
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
        group = _Group(rank)
        _check_reduce_scatter_epochs(rank, group, kernel, device, diagnostics)
        layer = _weights(rank, device)
        fixtures = [_inputs(i, sizes, device) for i, sizes in enumerate(SIZES)]
        # Queue changing shard sizes without host synchronization or barriers
        # between calls. This reuses the same route workspaces, inbox and epochs.
        outputs = [
            kernel.moe_reduce_scatter(layer, *inputs, list(sizes), group, max_rows=96)
            for inputs, sizes in zip(fixtures, SIZES)
        ]
        torch.cuda.synchronize()
        failures = []
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
                if not (relative_rms < 0.01 and normalized_max < 0.1):
                    torch.save(
                        dict(actual=actual.cpu(), expected=expected.cpu(), sizes=sizes),
                        Path(diagnostics) / f"moe-rank{rank}-case{index}.pt",
                    )
                    failures.append((index, relative_rms.item(), normalized_max.item()))
        assert not failures, f"MoE mismatch on rank {rank}: {failures}"
        assert kernel._reduce_scatter.epoch == sum(bool(sum(s)) for s in SIZES)
    finally:
        dist.destroy_process_group()


def test_tp_moe_matches_aiter_and_rccl(monkeypatch, tmp_path):
    if not torch.version.hip or torch.cuda.device_count() < RANKS:
        pytest.skip("Requires eight gfx942 GPUs on one node")
    if any(
        "gfx942" not in torch.cuda.get_device_properties(i).gcnArchName
        for i in range(RANKS)
    ):
        pytest.skip("Requires gfx942")
    monkeypatch.setenv("AITER_FMOE_FP8_BLOCKSCALE_FLYDSL", "0")
    mp.spawn(
        _worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=RANKS,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
