"""Exercise compact banks through the actual HIP writers and FlyDSL attention."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.utils import launch_gather_shuffle_5d_to_linear
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.speculative.dflash_compact_cache import CompactDFlashCache
from sglang.srt.speculative.dflash_compact_layout import CompactDFlashLayout
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


class Request:
    def __init__(self, owner):
        self.req_pool_idx = owner


@pytest.fixture(scope="module")
def cache():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    layout = CompactDFlashLayout(54)
    pool = MHATokenToKVPool(
        size=layout.physical_tokens,
        page_size=64,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=128,
        v_head_dim=128,
        layer_num=5,
        device="cuda",
        enable_memory_saver=False,
        kv_cache_layout="vectorized_5d",
    )
    table = torch.zeros(
        (layout.request_rows, layout.bank_rows), dtype=torch.int32, device="cuda"
    )
    return CompactDFlashCache(layout, pool, table)


def kv_values(positions, owner, layer, *, noise=False):
    dims = torch.arange(128, device="cuda", dtype=torch.float32)
    phase = positions.float()[:, None] % 8192 * 0.03 + dims[None, :] * 0.013
    phase = phase + owner * 0.19 + layer * 0.31 + (1.7 if noise else 0)
    return (phase.sin() * 0.2).to(torch.bfloat16)[:, None, :], (phase.cos() * 0.2).to(
        torch.bfloat16
    )[:, None, :]


def materialize_prefill(cache, reqs, prefixes, counts):
    owners, ends, hidden_indices, locs = cache.prepare_prefill(reqs, prefixes, counts)
    positions = [
        torch.arange(
            max(prefix, cache.layout.visible_start(prefix + count)),
            prefix + count,
            device="cuda",
        )
        for prefix, count in zip(prefixes, counts)
    ]
    packed_offset = 0
    expected_indices = []
    for prefix, count, pos in zip(prefixes, counts, positions):
        expected_indices.append(packed_offset + pos - prefix)
        packed_offset += count
    torch.testing.assert_close(
        hidden_indices, torch.cat(expected_indices), rtol=0, atol=0
    )
    for layer in range(5):
        kv = [
            kv_values(pos, req.req_pool_idx, layer) for pos, req in zip(positions, reqs)
        ]
        cache.pool.set_kv_buffer(
            SimpleNamespace(layer_id=layer),
            locs,
            torch.cat([x[0] for x in kv]),
            torch.cat([x[1] for x in kv]),
        )
    cache.finish(reqs, owners, ends)


def assert_prefix(cache, reqs, prefixes, lengths):
    expected_kv = []
    for layer in range(5):
        locs = []
        expected_k = []
        expected_v = []
        for req, prefix, length in zip(reqs, prefixes, lengths.tolist()):
            locs.append(cache.table[req.req_pool_idx, :length].to(torch.int64))
            pos = torch.arange(prefix - length, prefix, device="cuda")
            k, v = kv_values(pos, req.req_pool_idx, layer)
            expected_k.append(k)
            expected_v.append(v)
        k, v = launch_gather_shuffle_5d_to_linear(
            cache.pool.k_buffer[layer], cache.pool.v_buffer[layer], torch.cat(locs)
        )
        torch.testing.assert_close(k, torch.cat(expected_k), rtol=0, atol=0)
        torch.testing.assert_close(v, torch.cat(expected_v), rtol=0, atol=0)
        expected_kv.append((expected_k, expected_v))
    return expected_kv


@pytest.mark.parametrize("residue", range(64))
@pytest.mark.parametrize("prefix_base", [0, 1024])
def test_partial_pages_attention_and_prefix_commit(cache, residue, prefix_base):
    from aiter.ops.flydsl import flydsl_paged_attention_swa_bf16

    reqs = [Request(1), Request(54)]
    prefixes = [prefix_base + residue, 1048576 - 64 + residue]
    for buffer in cache.pool.k_buffer + cache.pool.v_buffer:
        buffer.fill_(float("nan"))
    materialize_prefill(cache, reqs, [0, 0], prefixes)
    owners, lengths, locs = cache.prepare_decode(
        reqs, torch.tensor(prefixes, device="cuda")
    )
    expected = assert_prefix(cache, reqs, prefixes, lengths)
    for layer in range(5):
        kv = [
            kv_values(
                torch.arange(p, p + 8, device="cuda"), r.req_pool_idx, layer, noise=True
            )
            for r, p in zip(reqs, prefixes)
        ]
        cache.pool.set_kv_buffer(
            SimpleNamespace(layer_id=layer),
            locs.reshape(-1),
            torch.cat([x[0] for x in kv]),
            torch.cat([x[1] for x in kv]),
        )

    # Compare attention to an independent dense causal + sink + SWA reference.
    torch.manual_seed(residue)
    q = (torch.randn(16, 16, 128, device="cuda") * 0.2).to(torch.bfloat16)
    sinks = torch.linspace(-0.4, 0.3, 16, device="cuda")
    lens = [int(x) + 8 for x in lengths.tolist()]
    pages = [(x + 63) // 64 for x in lens]
    page_ids = torch.cat(
        [cache.table[r.req_pool_idx, : n * 64 : 64] // 64 for r, n in zip(reqs, pages)]
    ).int()
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device="cuda")
    one = torch.ones(1, device="cuda")
    actual = flydsl_paged_attention_swa_bf16(
        q,
        cache.pool.k_buffer[0],
        cache.pool.v_buffer[0],
        i32([0, 8, 16]),
        i32([0, pages[0], sum(pages)]),
        page_ids,
        8,
        max(lens),
        window_left=1023,
        kv_last_page_lens=i32([(n - 1) % 64 + 1 for n in lens]),
        q_descale=one,
        k_descale=one,
        v_descale=one,
        sink_ptr=sinks,
    )
    references = []
    for i, (req, prefix, length) in enumerate(zip(reqs, prefixes, lens)):
        nk, nv = kv_values(
            torch.arange(prefix, prefix + 8, device="cuda"),
            req.req_pool_idx,
            0,
            noise=True,
        )
        k = torch.cat((expected[0][0][i], nk))[:, 0].float()
        v = torch.cat((expected[0][1][i], nv))[:, 0].float()
        scores = (
            torch.einsum("qhd,kd->hqk", q[i * 8 : (i + 1) * 8].float(), k) * 128**-0.5
        )
        qp = torch.arange(length - 8, length, device="cuda")
        kp = torch.arange(length, device="cuda")
        visible = (kp[None, :] <= qp[:, None]) & (kp[None, :] >= qp[:, None] - 1023)
        scores.masked_fill_(~visible[None], -float("inf"))
        prob = torch.softmax(
            torch.cat((scores, sinks[:, None, None].expand(16, 8, 1)), dim=-1), dim=-1
        )[..., :-1]
        references.append(torch.einsum("hqk,kd->qhd", prob, v))
    torch.testing.assert_close(
        actual.float(), torch.cat(references), atol=0.003, rtol=0.02
    )
    repeated = flydsl_paged_attention_swa_bf16(
        q,
        cache.pool.k_buffer[0],
        cache.pool.v_buffer[0],
        i32([0, 8, 16]),
        i32([0, pages[0], sum(pages)]),
        page_ids,
        8,
        max(lens),
        window_left=1023,
        kv_last_page_lens=i32([(n - 1) % 64 + 1 for n in lens]),
        q_descale=one,
        k_descale=one,
        v_descale=one,
        sink_ptr=sinks,
    )
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)

    commits = [residue % 8 + 1, 8 - residue % 8]
    for layer in range(5):
        kv = [
            kv_values(torch.arange(p, p + 8, device="cuda"), r.req_pool_idx, layer)
            for r, p in zip(reqs, prefixes)
        ]
        cache.pool.set_kv_buffer_prefix_valid(
            SimpleNamespace(layer_id=layer),
            locs,
            i32(commits),
            torch.cat([x[0] for x in kv]),
            torch.cat([x[1] for x in kv]),
        )
    ends = [p + n for p, n in zip(prefixes, commits)]
    cache.finish(reqs, owners, torch.tensor(ends, device="cuda"))
    _, next_lengths, _ = cache.prepare_decode(reqs, torch.tensor(ends, device="cuda"))
    assert_prefix(cache, reqs, ends, next_lengths)


def test_chunked_prefill_reuse_and_flush(cache):
    req = Request(54)
    materialize_prefill(cache, [req], [0], [4097])
    materialize_prefill(cache, [req], [4097], [19])
    materialize_prefill(cache, [req], [4116], [3000])
    _, lengths, _ = cache.prepare_decode([req], torch.tensor([7116], device="cuda"))
    assert_prefix(cache, [req], [7116], lengths)
    generation = cache.generations[54]
    replacement = Request(54)
    materialize_prefill(cache, [replacement], [0], [67])
    assert cache.generations[54] == generation + 1
    _, lengths, _ = cache.prepare_decode(
        [replacement], torch.tensor([67], device="cuda")
    )
    assert_prefix(cache, [replacement], [67], lengths)
    # A retraction can re-prefill the same Req object in the same request slot.
    replacement.retraction_count = 1
    materialize_prefill(cache, [replacement], [0], [81])
    _, lengths, _ = cache.prepare_decode(
        [replacement], torch.tensor([81], device="cuda")
    )
    assert_prefix(cache, [replacement], [81], lengths)
    cache.clear()
    with pytest.raises(RuntimeError, match="no committed owner"):
        cache.prepare_decode([replacement], torch.tensor([67], device="cuda"))


@pytest.mark.parametrize("residue", range(64))
def test_fake_zero_suffix_preallocated_owners_and_reuse(cache, residue):
    owners = [1, 18, 19, 36, 37, 54]
    reqs = [Request(owner) for owner in owners]
    prefixes = (
        torch.tensor([0, 63, 1024, 65536, 262144, 1048512], device="cuda") + residue
    )
    # Dirty every owner bank, including speculative/rejected rows left by a
    # previous lifetime. Fake admission must explicitly initialize its suffix.
    for buffer in cache.pool.k_buffer + cache.pool.v_buffer:
        buffer.fill_(0.734375)
    for wave in range(4):
        reqs = [Request(owner) for owner in owners]
        cache.initialize_fake_prefix(reqs, prefixes)
        actual_owners, lengths, locs = cache.prepare_decode(reqs, prefixes)
        assert actual_owners.tolist() == owners
        for layer in range(5):
            visible = torch.cat(
                [
                    cache.table[owner, : int(length)]
                    for owner, length in zip(owners, lengths)
                ]
            ).long()
            k, v = launch_gather_shuffle_5d_to_linear(
                cache.pool.k_buffer[layer], cache.pool.v_buffer[layer], visible
            )
            assert torch.all(k == 0) and torch.all(v == 0)
            assert torch.all(locs >= 0) and torch.all(
                locs < cache.layout.physical_tokens
            )
            # Simulate valid commits to dirty the next lifetime's history.
            cache.pool.set_kv_buffer(
                SimpleNamespace(layer_id=layer),
                locs.reshape(-1),
                torch.ones(
                    len(owners) * 8, 1, 128, device="cuda", dtype=torch.bfloat16
                ),
                torch.ones(
                    len(owners) * 8, 1, 128, device="cuda", dtype=torch.bfloat16
                ),
            )
        cache.finish(reqs, actual_owners, prefixes + 8)
    # Row zero and an unrelated owner never become another owner's cache.
    guards = torch.tensor([0, cache.layout.bank_base(2, 0)], device="cuda")
    for kbuf, vbuf in zip(cache.pool.k_buffer, cache.pool.v_buffer):
        k, v = launch_gather_shuffle_5d_to_linear(kbuf, vbuf, guards)
        assert torch.all(k == 0.734375) and torch.all(v == 0.734375)


@pytest.fixture(scope="module")
def checkpoint_materializer(cache):
    """Real TP-rank-0 projection weights, K norm and MiMo partial RoPE."""
    from safetensors import safe_open
    from sglang.srt import server_args
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.models.dflash import DFlashAttention
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
        FusedKVMaterializeHelper,
    )

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        server_args,
        "_global_server_args",
        SimpleNamespace(
            rl_on_policy_target=None,
            enable_aiter_allreduce_fusion=False,
            speculative_dflash_compact_cache=True,
        ),
    )

    root = Path(
        os.environ.get(
            "DFLASH_TEST_CHECKPOINT", "/models/MiMo-V2.5-Pro-FP4-DFlash/dflash"
        )
    )
    if not (root / "model.safetensors.index.json").is_file():
        pytest.skip("requires the MiMo DFlash checkpoint")
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def weight(name, rows=None):
        with safe_open(root / index[name], framework="pt") as shard:
            tensor = shard.get_slice(name)[:128] if rows else shard.get_tensor(name)
            return tensor.to(device="cuda", dtype=torch.bfloat16)

    fc = torch.nn.Linear(30720, 6144, bias=False, device="cuda", dtype=torch.bfloat16)
    fc.weight.data.copy_(weight("fc.weight"))
    hidden_norm = RMSNorm(6144, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
    hidden_norm.weight.data.copy_(weight("hidden_norm.weight"))
    rotary = get_rope(
        128,
        rotary_dim=128,
        max_position=1048576,
        base=5000000,
        partial_rotary_factor=0.5,
        is_neox_style=True,
    )
    layers = []
    for i in range(5):
        attn = object.__new__(DFlashAttention)
        torch.nn.Module.__init__(attn)
        attn.num_kv_heads = 1
        attn.head_dim = attn.kv_size = 128
        attn.q_size = 0
        attn.v_scale = 0.612
        attn.qkv_proj = SimpleNamespace(
            weight=torch.cat(
                [
                    weight(f"layers.{i}.self_attn.{kind}_proj.weight", rows=True)
                    for kind in ("k", "v")
                ]
            ),
            bias=None,
            quant_method=UnquantizedLinearMethod(),
        )
        attn.k_norm = RMSNorm(128, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
        attn.k_norm.weight.data.copy_(weight(f"layers.{i}.self_attn.k_norm.weight"))
        attn.rotary_emb = rotary
        attn.attn = SimpleNamespace(layer_id=i, k_scale=None, v_scale=None)
        layers.append(SimpleNamespace(self_attn=attn))
    worker = object.__new__(DFlashWorkerV2)
    worker.model_runner = SimpleNamespace(device=torch.device("cuda"))
    worker.draft_model_runner = SimpleNamespace(token_to_kv_pool=cache.pool)
    worker.draft_model = SimpleNamespace(
        layers=layers,
        project_target_hidden=lambda x: hidden_norm(fc(x)),
    )
    worker._fused_kv_helper = FusedKVMaterializeHelper(
        layers,
        rotary,
        1,
        128,
        torch.device("cuda"),
        max_position_hint=1048576,
    )
    yield worker
    patcher.undo()


@pytest.mark.parametrize("prefix_valid,commit", [(False, None), (True, 1), (True, 8)])
def test_checkpoint_sequential_and_fused_materialization(
    cache, checkpoint_materializer, prefix_valid, commit
):
    worker = checkpoint_materializer
    count = 8 if prefix_valid else 1087
    torch.manual_seed(17)
    hidden = (torch.randn(count, 30720, device="cuda") * 0.2).to(torch.bfloat16)
    positions = torch.arange(1048576 - count, 1048576, device="cuda")
    locations = [
        torch.arange(
            cache.layout.bank_base(2, b),
            cache.layout.bank_base(2, b) + count,
            device="cuda",
            dtype=torch.int64,
        )
        for b in (0, 1)
    ]
    sentinel = 0.734375
    for buffer in cache.pool.k_buffer + cache.pool.v_buffer:
        buffer.fill_(sentinel)
    observations = []
    valid = commit if prefix_valid else count
    for fused, loc in zip((False, True), locations):
        worker._use_fused_kv_materialize = fused
        kwargs = {}
        if prefix_valid:
            kwargs = {
                "cache_loc_2d": loc.reshape(1, 8),
                "commit_lens": torch.tensor([commit], dtype=torch.int32, device="cuda"),
            }
        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden,
            cache_loc=loc,
            positions=positions,
            **kwargs,
        )
        assert worker._use_fused_kv_materialize == fused
        # Legacy global pages may begin at any within-page offset. Use the
        # same writer/projection shape there to isolate physical-address parity
        # from the existing sequential/fused arithmetic rounding differences.
        reference_locs = loc - loc[0] + cache.layout.bank_base(3, 0) + 1
        reference_kwargs = dict(kwargs)
        if prefix_valid:
            reference_kwargs["cache_loc_2d"] = reference_locs.reshape(1, 8)
        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden,
            cache_loc=reference_locs,
            positions=positions,
            **reference_kwargs,
        )
        for k, v in zip(cache.pool.k_buffer, cache.pool.v_buffer):
            bank_kv = launch_gather_shuffle_5d_to_linear(k, v, loc)
            reference_kv = launch_gather_shuffle_5d_to_linear(k, v, reference_locs)
            for actual, expected in zip(bank_kv, reference_kv):
                torch.testing.assert_close(
                    actual[:valid], expected[:valid], rtol=0, atol=0
                )
        observations.append(
            [
                launch_gather_shuffle_5d_to_linear(k, v, loc)
                for k, v in zip(cache.pool.k_buffer, cache.pool.v_buffer)
            ]
        )
    differences = []
    for layer, (sequential, fused) in enumerate(zip(*observations)):
        for kind, (a, b) in zip(("k", "v"), zip(sequential, fused)):
            differences.append(
                {
                    "layer": layer,
                    "kind": kind,
                    "max_abs": (a[:valid].float() - b[:valid].float())
                    .abs()
                    .max()
                    .item(),
                    "bitwise_equal": torch.equal(a[:valid], b[:valid]),
                }
            )
            if prefix_valid:
                assert torch.all(a[valid:] == sentinel)
                assert torch.all(b[valid:] == sentinel)
    print("Existing sequential/fused arithmetic differences:", differences)
    # Unrelated banks and row zero (graph dummy storage) remain untouched.
    guards = torch.tensor(
        [0, cache.layout.bank_base(4, 0), cache.layout.physical_tokens - 1],
        device="cuda",
        dtype=torch.int64,
    )
    for k, v in zip(cache.pool.k_buffer, cache.pool.v_buffer):
        a, b = launch_gather_shuffle_5d_to_linear(k, v, guards)
        assert torch.all(a == sentinel) and torch.all(b == sentinel)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
