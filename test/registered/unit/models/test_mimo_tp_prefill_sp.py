"""CPU regressions for TP SP decisions and the layouts consumed by its GPU kernels.

Load the production functions without importing the ROCm model stack: that stack
queries GPU properties even when all accelerators are hidden. GPU numerical and
collective qualification is separate from these host-side regressions.
"""

import ast
import dataclasses
import enum
import itertools
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")

SRC = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load(relative, names, **bindings):
    tree = ast.parse((SRC / relative).read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    module = types.ModuleType(f"mimo_tp_sp_test_{len(sys.modules)}")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        torch=torch,
        dataclass=dataclasses.dataclass,
        itertools=itertools,
        IntEnum=enum.IntEnum,
        auto=enum.auto,
    )
    module.__dict__.update(bindings)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(SRC / relative), "exec"), module.__dict__)
    return module


Mode = load("model_executor/forward_batch_info.py", ["ForwardMode"]).ForwardMode
global_prefill = load(
    "managers/scheduler_components/dp_attn.py",
    ["_all_active_dp_ranks_prefill"],
    ForwardMode=Mode,
)._all_active_dp_ranks_prefill


@pytest.mark.parametrize("left,right", itertools.product(list(Mode), repeat=2))
def test_global_plain_prefill_excludes_verify_and_mixed(left, right):
    active = set((left, right)) - {Mode.IDLE}
    assert global_prefill([left, right]) == (active == {Mode.EXTEND})


@pytest.mark.parametrize(
    "modes,sizes,expected",
    [
        ([Mode.EXTEND, Mode.IDLE], [32768, 0], [True, True]),
        ([Mode.IDLE, Mode.EXTEND], [0, 16384], [True, True]),
        ([Mode.EXTEND, Mode.EXTEND], [1024, 1024], [True, True]),
        ([Mode.EXTEND, Mode.EXTEND], [512, 512], [False, False]),
        ([Mode.EXTEND, Mode.TARGET_VERIFY], [16384, 1920], [False, False]),
        ([Mode.TARGET_VERIFY, Mode.TARGET_VERIFY], [1920, 1920], [False, False]),
        ([Mode.EXTEND, Mode.EXTEND], [1025, 1024], [False, False]),
    ],
)
def test_moe_collective_decision_is_identical_on_idle_and_active_replicas(
    modes, sizes, expected
):
    fn = load(
        "models/mimo_v2.py",
        ["_mimo_sp_moe_active"],
        ForwardMode=Mode,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)
        ),
        get_dp_global_num_tokens=lambda: sizes,
        _MIMO_SP_MIN_TOKENS=2048,
        _log_mimo_tp_sp=lambda: None,
    )._mimo_sp_moe_active
    decisions = [
        fn(
            SimpleNamespace(
                forward_mode=mode,
                all_extend_in_batch=global_prefill(modes),
                can_run_tbo=False,
            )
        )
        for mode in modes
    ]
    assert decisions == expected


@pytest.mark.parametrize(
    "mode,capturing,tbo",
    [
        (Mode.TARGET_VERIFY, False, False),
        (Mode.DECODE, False, False),
        (Mode.DRAFT_EXTEND_V2, False, False),
        (Mode.MIXED, False, False),
        (Mode.EXTEND, True, False),
        (Mode.EXTEND, False, True),
    ],
)
def test_stale_prefill_flag_cannot_enable_decode_capture_or_tbo(mode, capturing, tbo):
    fn = load(
        "models/mimo_v2.py",
        ["_mimo_sp_moe_active"],
        ForwardMode=Mode,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing)
        ),
        get_dp_global_num_tokens=lambda: [16384, 16384],
        _MIMO_SP_MIN_TOKENS=2048,
        _log_mimo_tp_sp=lambda: None,
    )._mimo_sp_moe_active
    assert not fn(
        SimpleNamespace(forward_mode=mode, all_extend_in_batch=True, can_run_tbo=tbo)
    )


@pytest.mark.parametrize(
    "tp,dp,attn_tp,ep,a2a,enabled,expected",
    [
        (8, 2, 4, 1, "none", True, True),
        (8, 2, 4, 1, "none", False, False),
        (8, 2, 4, 2, "none", True, False),
        (8, 2, 4, 8, "none", True, False),
        (8, 2, 4, 8, "mori", True, False),
        (8, 2, 4, 1, "deepep", True, False),
        (4, 2, 2, 1, "none", True, False),
        (8, 1, 8, 1, "none", True, False),
    ],
)
def test_topology_contract(tp, dp, attn_tp, ep, a2a, enabled, expected):
    fn = load(
        "layers/moe/utils.py",
        ["should_use_dpa_sp_moe"],
        envs=SimpleNamespace(SGLANG_ENABLE_DPA_SP=SimpleNamespace(get=lambda: enabled)),
        is_dp_attention_enabled=lambda: True,
        get_tensor_model_parallel_world_size=lambda: tp,
        get_attention_dp_size=lambda: dp,
        get_attention_tp_size=lambda: attn_tp,
        get_moe_expert_parallel_world_size=lambda: ep,
        get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: a2a == "none"),
        should_use_flashinfer_cutlass_moe_fp4_allgather=lambda: False,
    ).should_use_dpa_sp_moe
    assert fn() == expected


@pytest.mark.parametrize(
    "changed,value",
    [
        (None, None),
        ("native_mxfp4", True),
        ("gfx942", False),
        ("flypa", False),
        ("native_v_cache", False),
        ("kv_cache_dtype", "fp8"),
        ("page_size", 16),
        ("attention_backend", "triton"),
        ("prefill_attention_backend", "triton"),
        ("attn_cp_size", 2),
        ("pp_size", 2),
        ("moe_dense_tp_size", None),
        ("enable_two_batch_overlap", True),
        ("layout", "nhd"),
        ("aiter_moe", False),
    ],
)
def test_unsupported_runtime_keeps_original_layout(changed, value):
    settings = dict(
        native_mxfp4=False,
        gfx942=True,
        flypa=True,
        native_v_cache=True,
        attention_backend="aiter",
        prefill_attention_backend=None,
        kv_cache_dtype="bf16",
        page_size=64,
        attn_cp_size=1,
        pp_size=1,
        moe_dense_tp_size=1,
        enable_two_batch_overlap=False,
        layout="vectorized_5d",
        aiter_moe=True,
    )
    if changed:
        settings[changed] = value
    fn = load(
        "models/mimo_v2.py",
        ["_mimo_sp_attn_enabled"],
        should_use_dpa_sp_moe=lambda: True,
        get_global_server_args=lambda: SimpleNamespace(**settings),
        is_gfx942_supported=lambda: settings["gfx942"],
        get_bool_env_var=lambda _: settings["flypa"],
        mimo_flydsl_target_verify_enabled=lambda: settings["native_v_cache"],
        get_moe_runner_backend=lambda: SimpleNamespace(
            is_aiter=lambda: settings["aiter_moe"], is_auto=lambda: False
        ),
        envs=SimpleNamespace(
            SGLANG_MXFP4_TO_FP8_128BLOCK=SimpleNamespace(
                get=lambda: not settings["native_mxfp4"]
            ),
            SGLANG_AITER_KV_CACHE_LAYOUT=SimpleNamespace(
                get=lambda: settings["layout"]
            ),
        ),
    )._mimo_sp_attn_enabled
    assert fn() == (changed is None)


@pytest.mark.parametrize(
    "sizes", [[3] * 8, [3] * 4 + [5] * 4, [0] * 4 + [7] * 4, [7] * 4 + [0] * 4, [0] * 8]
)
@pytest.mark.parametrize("groups", [1, 32])
def test_fp8_column_major_scales_preserve_rank_and_token_order(sizes, groups):
    fn = load(
        "models/mimo_v2.py", ["_mimo_sp_moe_gather_scales"]
    )._mimo_sp_moe_gather_scales
    logical = [
        torch.arange(n * groups, dtype=torch.float32).view(n, groups) + r * 10000
        for r, n in enumerate(sizes)
    ]
    packed = torch.cat([x.T.contiguous().reshape(x.shape) for x in logical])
    result = fn(packed, sizes)
    torch.testing.assert_close(
        result.view(groups, sum(sizes)).T, torch.cat(logical), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "lens,prefixes",
    [
        ([16384], [0]),
        ([257], [513]),
        ([31, 260, 13], [0, 155, 64]),
        ([37, 14], [300, 2]),
        ([], []),
        ([0, 0], [63, 100]),
    ],
)
@pytest.mark.parametrize("rank", range(4))
def test_swa_halos_recover_zero_reported_chunk_prefix(
    lens, prefixes, rank, monkeypatch
):
    helper = load("layers/attention/aiter_utils.py", ["_mimo_effective_prefix_lengths"])
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.attention.aiter_utils", helper)
    window = 127
    rows = max(1, (sum(lens) + 3) // 4)
    indices = torch.arange(len(lens)) + 3
    table = torch.arange(2000)[None, :] + torch.arange(len(lens) + 5)[:, None] * 1000000
    context = types.ModuleType("sglang.srt.model_executor.forward_context")
    context.get_req_to_token_pool = lambda: SimpleNamespace(req_to_token=table)
    context.get_token_to_kv_pool = lambda: SimpleNamespace(
        translate_loc_from_full_to_swa=lambda x: x + 10000000
    )
    monkeypatch.setitem(sys.modules, context.__name__, context)
    module = load(
        "models/mimo_v2.py",
        ["_MimoSpSwaMetadata", "_mimo_sp_swa_metadata"],
        _SP_SWA_PAGE=64,
        get_attention_tp_size=lambda: 4,
        get_attention_tp_rank=lambda: rank,
    )
    fb = SimpleNamespace(
        extend_seq_lens_cpu=lens,
        extend_prefix_lens_cpu=[0] * len(lens),
        seq_lens_cpu=[p + n for p, n in zip(prefixes, lens)],
        positions=torch.arange(rows * 4),
        req_pool_indices=indices,
    )
    meta = module._mimo_sp_swa_metadata(fb, window, rows)
    queries = []
    cached = []
    for i, (n, prefix) in enumerate(zip(lens, prefixes)):
        pool = int(indices[i])
        queries += [pool * 1000000 + prefix + j for j in range(n)]
        if n:
            cached += [
                pool * 1000000 + j for j in range(max(0, prefix - window), prefix)
            ]
    queries += [-999] * (rows * 4 - len(queries))
    if cached:
        torch.testing.assert_close(
            meta.cache_loc, torch.tensor(cached) + 10000000, rtol=0, atol=0
        )
    else:
        assert meta.cache_loc is None
    tail = min(window, rows)
    tails = [
        v for r in range(4) for v in queries[r * rows + rows - tail : (r + 1) * rows]
    ]
    packed = torch.tensor(
        cached + tails + queries[rank * rows : (rank + 1) * rows] + [-1],
        dtype=torch.int64,
    )
    expected = []
    start = 0
    for i, (n, prefix) in enumerate(zip(lens, prefixes)):
        qa, qb = max(start, rank * rows), min(start + n, (rank + 1) * rows)
        if qa < qb:
            pool = int(indices[i])
            keys = [
                pool * 1000000 + j
                for j in range(
                    max(0, prefix + qa - start - window), prefix + qb - start
                )
            ]
            expected.append((qb - qa, keys))
        start += n
    assert meta.q_rows == sum(q for q, _ in expected)
    for i, (q, keys) in enumerate(expected):
        begin, end = [int(x) * 64 for x in meta.kv_indptr[i : i + 2]]
        actual = packed[meta.kv_src[begin:end]].tolist()
        assert actual[: len(keys)] == keys
        assert actual[len(keys) :] == [-1] * (len(actual) - len(keys))
        assert int(meta.last_page_lens[i]) == (len(keys) - 1) % 64 + 1
        assert int(meta.cu_seqlens_q[i + 1] - meta.cu_seqlens_q[i]) == q


def test_skip_sync_clears_old_prefill_eligibility():
    update = load(
        "managers/scheduler_components/dp_attn.py", ["_update_gather_batch"]
    )._update_gather_batch
    info = SimpleNamespace(
        num_tokens=1,
        num_tokens_for_logprob=1,
        global_num_tokens=[1, 1],
        global_num_tokens_for_logprob=[1, 1],
        all_extend_in_batch=True,
        is_extend_in_batch=True,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        can_cuda_graph=False,
        can_run_breakable_cuda_graph=False,
    )
    batch = SimpleNamespace(all_extend_in_batch=True)
    update(batch, info, require_mlp_tp_gather=True, skip_all_gather=True)
    assert not batch.all_extend_in_batch


@pytest.mark.parametrize(
    "modes,expected",
    [
        ([Mode.EXTEND, Mode.IDLE], True),
        ([Mode.TARGET_VERIFY, Mode.EXTEND], False),
        ([Mode.EXTEND, Mode.EXTEND], True),
        ([Mode.MIXED, Mode.EXTEND], False),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_scheduler_uses_existing_all_gather_for_uniform_prefill_flag(
    monkeypatch, modes, expected, enabled
):
    gathered = torch.zeros(2, 4, 7, dtype=torch.int64)
    for rank, mode in enumerate(modes):
        gathered[rank, :, 0] = 0 if mode == Mode.IDLE else 16384
        gathered[rank, :, 1] = gathered[rank, :, 0]
        gathered[rank, :, 3] = int(mode.is_extend())
        gathered[rank, :, 5] = mode.value
    calls = []

    def all_gather(output, local, group):
        calls.append(1)
        output.copy_(gathered.flatten())

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather)
    module = load(
        "managers/scheduler_components/dp_attn.py",
        ["MLPSyncBatchInfo", "_update_gather_batch", "_all_active_dp_ranks_prefill"],
        ForwardMode=Mode,
        _ENABLE_METRICS_DP_ATTENTION=False,
        get_tp_group=lambda: SimpleNamespace(active_ranks_cpu=torch.ones(8)),
        envs=SimpleNamespace(SGLANG_ENABLE_DPA_SP=SimpleNamespace(get=lambda: enabled)),
    )
    info = module.MLPSyncBatchInfo(
        dp_size=2,
        tp_size=4,
        cp_size=1,
        num_tokens=16384,
        num_tokens_for_logprob=16384,
        can_cuda_graph=False,
        is_extend_in_batch=True,
        local_can_run_tbo=False,
        local_forward_mode=Mode.EXTEND.value,
        can_run_breakable_cuda_graph=False,
    )
    info.all_gather(device="cpu", group=None)
    assert calls == [1]
    assert info.all_extend_in_batch == (enabled and expected)
    for _ in modes:
        batch = SimpleNamespace(all_extend_in_batch=True)
        module._update_gather_batch(batch, info, require_mlp_tp_gather=True)
        assert batch.all_extend_in_batch == (enabled and expected)
