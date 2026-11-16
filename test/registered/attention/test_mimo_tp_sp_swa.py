"""Numerical coverage of TP4 SWA packing, cache writes and attention on gfx942."""

import math
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-c-test-4-gpu-amd")


class _AttentionGroup:
    def all_to_all_single(self, output, value):
        dist.all_to_all_single(output, value)

    def all_gather_into_tensor(self, output, value):
        dist.all_gather_into_tensor(output, value)


def _case(rank, lengths, prefixes, v_scale, with_sink, pretranslated):
    from sglang.srt.layers.attention import mimo_sp
    from sglang.srt.model_executor import forward_context

    device = torch.device("cuda", rank)
    heads, kv_heads, dq, dv, page = 64, 8, 192, 128, 64
    local_heads = kv_heads // 4
    live = sum(lengths)
    rows = (live + 3) // 4
    padded = rows * 4
    generator = torch.Generator().manual_seed(12345)

    def random(shape):
        return (torch.randn(shape, generator=generator) * 0.3).to(
            device=device, dtype=torch.bfloat16
        )

    q = random((padded, heads, dq))
    keys = [random((p + n, kv_heads, dq)) for n, p in zip(lengths, prefixes)]
    values = [random((p + n, kv_heads, dv)) for n, p in zip(lengths, prefixes)]
    scaled_values = [
        v if v_scale is None else (v.float() * v_scale).to(v.dtype) for v in values
    ]
    k = torch.cat([x[p:] for x, p in zip(keys, prefixes)])
    v = torch.cat([x[p:] for x, p in zip(values, prefixes)])
    k = torch.cat([k, k.new_full((padded - live, kv_heads, dq), float("nan"))])
    v = torch.cat([v, v.new_full((padded - live, kv_heads, dv), float("nan"))])
    q[live:] = float("nan")

    width = max(n + p for n, p in zip(lengths, prefixes))
    pages_per_request = (width + page - 1) // page
    count = len(lengths) * pages_per_request + 1  # spare page for padded rows
    full_ids = torch.randperm(count, generator=generator).to(device) + 1
    swa_ids = full_ids.roll(1)
    full_to_swa = torch.zeros((count + 1) * page, dtype=torch.long, device=device)
    offsets = torch.arange(page, device=device)
    full_slots = full_ids[:, None] * page + offsets
    swa_slots = swa_ids[:, None] * page + offsets
    full_to_swa[full_slots] = swa_slots
    table = full_slots[:-1].reshape(len(lengths), pages_per_request * page)
    current_loc = torch.cat(
        [table[i, p : p + n] for i, (n, p) in enumerate(zip(lengths, prefixes))]
    )
    current_loc = torch.cat([current_loc, full_slots[-1, : padded - live]])
    current_swa = full_to_swa[current_loc]
    kc = k.new_full((count + 1, local_heads, dq // 8, page, 8), float("nan"))
    vc = v.new_full((count + 1, local_heads, page // 8, dv, 8), float("nan"))

    def store(locations, key, value):
        pg, offset = locations // page, locations % page
        kc[pg, :, :, offset, :] = key.reshape(-1, local_heads, dq // 8, 8)
        vc[pg, :, offset // 8, :, offset % 8] = value

    hs = slice(rank * local_heads, (rank + 1) * local_heads)
    for i, prefix in enumerate(prefixes):
        store(
            full_to_swa[table[i, :prefix]],
            keys[i][:prefix, hs],
            scaled_values[i][:prefix, hs],
        )

    def write_cache(layer, loc, new_k, new_v, *_):
        torch.testing.assert_close(loc.loc, current_loc)
        locations = full_to_swa[loc.loc] if loc.swa_loc is None else loc.swa_loc
        torch.testing.assert_close(locations, current_swa)
        store(locations, new_k, new_v)

    pool = SimpleNamespace(
        get_key_buffer=lambda _: kc,
        get_value_buffer=lambda _: vc,
        set_kv_buffer=write_cache,
        translate_loc_from_full_to_swa=lambda loc: full_to_swa[loc],
    )
    backend = SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(
            swa_out_cache_loc=current_swa if pretranslated else None
        ),
    )
    begin, end = rank * rows, (rank + 1) * rows
    qkv = torch.cat(
        [q[begin:end].flatten(1), k[begin:end].flatten(1), v[begin:end].flatten(1)],
        dim=1,
    )
    sinks = torch.linspace(-1, 1, heads, device=device) if with_sink else None
    attention = SimpleNamespace(
        _ep_sp_swa=False,
        total_num_heads=heads,
        total_num_kv_heads=kv_heads,
        head_dim=dq,
        original_v_head_dim=dv,
        qkv_proj_full=lambda _: (qkv, None),
        rotary_emb=lambda positions, q, k: (q, k),
        v_scale=v_scale,
        attn=SimpleNamespace(
            sliding_window_size=128, tp_k_head_num=local_heads, layer_id=0
        ),
        attention_sink_full=sinks,
        scaling=1 / math.sqrt(dq),
        # Expose the attention output directly; linear projections are separate.
        o_proj=SimpleNamespace(quant_method=SimpleNamespace(apply=lambda layer, x: x)),
        o_proj_full=None,
    )
    batch = SimpleNamespace(
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens_cpu=[0] * len(lengths),
        seq_lens_cpu=[n + p for n, p in zip(lengths, prefixes)],
        req_pool_indices=torch.arange(len(lengths), device=device),
        out_cache_loc=current_loc,
        positions=torch.arange(padded, device=device),
    )
    with (
        patch.object(mimo_sp, "get_attention_tp_size", return_value=4),
        patch.object(mimo_sp, "get_attention_tp_rank", return_value=rank),
        patch.object(mimo_sp, "get_attention_tp_group", return_value=_AttentionGroup()),
        patch.object(forward_context, "get_attn_backend", return_value=backend),
        patch.object(
            forward_context,
            "get_req_to_token_pool",
            return_value=SimpleNamespace(req_to_token=table),
        ),
        patch.object(forward_context, "get_token_to_kv_pool", return_value=pool),
    ):
        actual = mimo_sp.run_mimo_sp_swa(
            attention, batch.positions, q.new_zeros((rows, 1)), batch
        )

    expected = q.new_zeros((rows, heads, dv))
    offset = 0
    for n, prefix, key, value in zip(lengths, prefixes, keys, scaled_values):
        qa, qb = max(offset, begin), min(offset + n, end)
        if qa < qb:
            query = q[qa:qb].float().transpose(0, 1)
            key = (
                key.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
            )
            value = (
                value.float()
                .repeat_interleave(heads // kv_heads, dim=1)
                .transpose(0, 1)
            )
            positions = prefix + torch.arange(qa - offset, qb - offset, device=device)
            key_pos = torch.arange(prefix + n, device=device)
            valid = (key_pos[None, :] <= positions[:, None]) & (
                key_pos[None, :] >= positions[:, None] - 127
            )
            scores = (query @ key.transpose(1, 2)) * attention.scaling
            scores.masked_fill_(~valid[None], -torch.inf)
            if sinks is not None:
                scores = torch.cat(
                    [scores, sinks[:, None, None].expand(-1, qb - qa, 1)], dim=-1
                )
            probabilities = scores.softmax(-1)[..., : prefix + n]
            expected[qa - begin : qb - begin] = (
                (probabilities @ value).transpose(0, 1).to(expected.dtype)
            )
        offset += n
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.view_as(expected), expected, atol=2e-3, rtol=2e-2)
    pg, pos = current_swa // page, current_swa % page
    torch.testing.assert_close(
        kc[pg, :, :, pos, :].reshape(padded, local_heads, dq),
        k[:, hs],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    scaled = v if v_scale is None else (v.float() * v_scale).to(v.dtype)
    torch.testing.assert_close(
        vc[pg, :, pos // 8, :, pos % 8], scaled[:, hs], rtol=0, atol=0, equal_nan=True
    )


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    try:
        for args in (
            ([2048], [0], None, False, False),
            ([17, 2050, 6], [129, 257, 65], None, True, True),
            ([0, 1027, 1046], [64, 513, 31], 0.5, True, False),
        ):
            _case(rank, *args)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_tp_swa_matches_unsharded_reference():
    if torch.version.hip is None or torch.cuda.device_count() < 4:
        pytest.skip("requires four MI300X GPUs")
    if any(
        "gfx942" not in torch.cuda.get_device_properties(i).gcnArchName
        for i in range(4)
    ):
        pytest.skip("gfx942 TP SWA path")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _worker, args=(str(Path(directory) / "rendezvous"),), nprocs=4, join=True
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
