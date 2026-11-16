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
from functools import cached_property
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
        cached_property=cached_property,
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
        "layers/moe/mimo_sp.py",
        ["mimo_tp_sp_active"],
        ForwardMode=Mode,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)
        ),
        get_dp_global_num_tokens=lambda: sizes,
        _MIMO_SP_MIN_TOKENS=2048,
        _log_mimo_tp_sp=lambda: None,
    ).mimo_tp_sp_active
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
        "layers/moe/mimo_sp.py",
        ["mimo_tp_sp_active"],
        ForwardMode=Mode,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing)
        ),
        get_dp_global_num_tokens=lambda: [16384, 16384],
        _MIMO_SP_MIN_TOKENS=2048,
        _log_mimo_tp_sp=lambda: None,
    ).mimo_tp_sp_active
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
        envs=SimpleNamespace(
            SGLANG_ENABLE_MIMO_SWA_SP=SimpleNamespace(get=lambda: enabled)
        ),
        is_dp_attention_enabled=lambda: True,
        get_tensor_model_parallel_world_size=lambda: tp,
        get_attention_dp_size=lambda: dp,
        get_attention_tp_size=lambda: attn_tp,
        get_moe_expert_parallel_world_size=lambda: ep,
        get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: a2a == "none"),
        should_use_flashinfer_cutlass_moe_fp4_allgather=lambda: False,
    ).should_use_dpa_sp_moe
    assert fn() == expected


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ep,a2a", [(1, "none"), (8, "mori")])
def test_shared_sp_flag_selects_only_the_supported_backend(enabled, ep, a2a):
    flags = SimpleNamespace(
        SGLANG_ENABLE_MIMO_SWA_SP=SimpleNamespace(get=lambda: enabled),
        SGLANG_AITER_KV_CACHE_LAYOUT=SimpleNamespace(get=lambda: "vectorized_5d"),
    )
    backend = SimpleNamespace(
        is_none=lambda: a2a == "none", is_mori=lambda: a2a == "mori"
    )
    tp = load(
        "layers/moe/utils.py",
        ["should_use_dpa_sp_moe"],
        envs=flags,
        is_dp_attention_enabled=lambda: True,
        get_tensor_model_parallel_world_size=lambda: 8,
        get_attention_dp_size=lambda: 2,
        get_attention_tp_size=lambda: 4,
        get_moe_expert_parallel_world_size=lambda: ep,
        get_moe_a2a_backend=lambda: backend,
        should_use_flashinfer_cutlass_moe_fp4_allgather=lambda: False,
    ).should_use_dpa_sp_moe
    ep_supported = load(
        "models/mimo_v2.py",
        ["_mimo_ep_sp_supported"],
        get_global_server_args=lambda: SimpleNamespace(
            enable_two_batch_overlap=False,
            attention_backend="aiter",
            page_size=64,
            kv_cache_dtype="bf16",
            attn_cp_size=1,
        ),
        is_gfx942_supported=lambda: True,
        is_dp_attention_enabled=lambda: True,
        get_attention_tp_size=lambda: 4,
        enable_moe_dense_fully_dp=lambda: True,
        get_moe_a2a_backend=lambda: backend,
        envs=flags,
    )._mimo_ep_sp_supported
    tree = ast.parse((SRC / "models/mimo_v2.py").read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_ep_sp_swa"
            for t in node.targets
        )
    )
    attention = SimpleNamespace()
    exec(
        compile(ast.Module(body=[assignment], type_ignores=[]), "ep-selection", "exec"),
        dict(self=attention, envs=flags, _mimo_ep_sp_supported=ep_supported),
    )
    assert tp() == (enabled and ep == 1)
    assert attention._ep_sp_swa == (enabled and ep == 8)


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
        "layers/moe/mimo_sp.py",
        ["mimo_tp_sp_enabled"],
        should_use_dpa_sp_moe=lambda: True,
        get_global_server_args=lambda: SimpleNamespace(**settings),
        is_gfx942_supported=lambda: settings["gfx942"],
        mimo_flydsl_target_verify_enabled=lambda: settings["native_v_cache"],
        get_moe_runner_backend=lambda: SimpleNamespace(
            is_aiter=lambda: settings["aiter_moe"], is_auto=lambda: False
        ),
        envs=SimpleNamespace(
            SGLANG_FLYPA_MIMO_PREFILL=SimpleNamespace(get=lambda: settings["flypa"]),
            SGLANG_MXFP4_TO_FP8_128BLOCK=SimpleNamespace(
                get=lambda: not settings["native_mxfp4"]
            ),
            SGLANG_AITER_KV_CACHE_LAYOUT=SimpleNamespace(
                get=lambda: settings["layout"]
            ),
        ),
    ).mimo_tp_sp_enabled
    assert fn() == (changed is None)


@pytest.mark.parametrize(
    "sizes", [[3] * 8, [3] * 4 + [5] * 4, [0] * 4 + [7] * 4, [7] * 4 + [0] * 4, [0] * 8]
)
@pytest.mark.parametrize("groups", [1, 32])
def test_fp8_column_major_scales_preserve_rank_and_token_order(sizes, groups):
    fn = load(
        "layers/moe/mimo_sp.py", ["_mimo_sp_moe_gather_scales"]
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
    helper = load("models/mimo_v2_utils.py", ["mimo_effective_prefix_lengths"])
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
        "layers/attention/mimo_sp.py",
        ["_MimoSpSwaMetadata", "_mimo_sp_swa_metadata"],
        mimo_effective_prefix_lengths=helper.mimo_effective_prefix_lengths,
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
        torch.testing.assert_close(
            meta.cache_page * 64 + meta.cache_offset, meta.cache_loc, rtol=0, atol=0
        )
        assert meta.cache_page is meta.cache_page
        assert meta.cache_offset is meta.cache_offset
    else:
        assert meta.cache_loc is None
        assert meta.cache_page is None and meta.cache_offset is None
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


@pytest.mark.parametrize("ep", [False, True])
def test_shared_swa_entry_point_keeps_tp_and_ep_execution_distinct(ep):
    calls = []

    def record(path):
        def run(*args):
            calls.append((path, args))
            return path

        return run

    run = load(
        "layers/attention/mimo_sp.py",
        ["run_mimo_sp_swa"],
        _run_mimo_tp_sp_swa=record("tp"),
        _run_mimo_ep_sp_swa=record("ep"),
    ).run_mimo_sp_swa
    attention = SimpleNamespace(_ep_sp_swa=ep)
    args = (attention, object(), object(), object())
    expected = "ep" if ep else "tp"
    assert run(*args) == expected
    assert calls == [(expected, args)]


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("nnodes", [1, 2])
def test_tp_moe_selection_preserves_layer_and_workspace_fallbacks(
    monkeypatch, supported, nnodes
):
    checks = []
    package = types.ModuleType("sglang.srt.layers.moe")
    package.pyhip_fp8_blockscale_moe = SimpleNamespace(
        supports_layer=lambda *args: checks.append("layer") or supported,
        fits=lambda sizes, experts, cap: checks.append("workspace")
        or max(sizes) <= cap,
    )
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = load(
        "layers/moe/mimo_sp.py",
        ["_use_pyhip_moe"],
        get_tp_group=lambda: SimpleNamespace(world_size=8),
        get_global_server_args=lambda: SimpleNamespace(nnodes=nnodes),
        _mimo_sp_moe_max_rows=lambda: 4096,
    )
    layer = SimpleNamespace(
        _pyhip_moe=None,
        experts=SimpleNamespace(w13_weight=SimpleNamespace(shape=(256,))),
    )
    assert not module._use_pyhip_moe(layer, [8192] * 8)
    assert module._use_pyhip_moe(layer, [4096] * 8) == (supported and nnodes == 1)
    if nnodes != 1:
        assert not checks
        return
    assert checks.count("layer") == 1
    assert checks.count("workspace") == (2 if supported else 0)


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
        envs=SimpleNamespace(
            SGLANG_ENABLE_MIMO_SWA_SP=SimpleNamespace(get=lambda: enabled)
        ),
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


def test_empty_tp_moe_does_not_allocate_or_advance_ipc_epoch():
    def forbidden(*args, **kwargs):
        raise AssertionError("An empty call must not route or allocate IPC buffers")

    module = load(
        "layers/moe/pyhip_fp8_blockscale_moe.py",
        ["moe_reduce_scatter", "_SumReduceScatter"],
        HIDDEN=4096,
        _reduce_scatter=None,
        _HipIpcBuffer=forbidden,
        moe_route_rows=forbidden,
    )
    x = torch.empty((0, 4096), dtype=torch.bfloat16)
    output = module.moe_reduce_scatter(None, x, None, None, None, [0] * 8, None, 96)
    assert output.shape == x.shape and output.dtype == torch.bfloat16
    assert module._reduce_scatter is None
    reducer = module._SumReduceScatter.__new__(module._SumReduceScatter)
    reducer.epoch = 7
    output = reducer(x, None, [0] * 8)
    assert output.shape == x.shape and reducer.epoch == 7


@pytest.mark.parametrize("ep", [False, True])
def test_full_attention_and_ep_skip_tp_weight_reassembly(monkeypatch, ep):
    def forbidden(*args):
        raise AssertionError("Full attention and EP must not gather TP SWA weights")

    package = types.ModuleType("sglang.srt.layers.attention.mimo_sp")
    package._mimo_attn_tp_gather = forbidden
    monkeypatch.setitem(sys.modules, package.__name__, package)
    cls = load("models/mimo_v2.py", ["MiMoV2Attention"], nn=torch.nn).MiMoV2Attention
    existing = object() if ep else None
    layer = SimpleNamespace(
        _ep_sp_swa=ep,
        qkv_proj_full=object() if ep else None,
        o_proj_full=existing,
    )
    cls.build_sp_weights(layer)
    assert layer.o_proj_full is existing


@pytest.mark.parametrize("rows,columns", [(3712, 2), (29, 32)])
def test_tp_swa_weight_assembly_preserves_projection_and_qkv_order(
    monkeypatch, rows, columns
):
    class Unquantized:
        pass

    tp = 4

    def gather(value):
        return torch.cat([value + rank * 100000 for rank in range(tp)])

    package = types.ModuleType("sglang.srt.layers.attention.mimo_sp")
    package._mimo_attn_tp_gather = gather
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = load(
        "models/mimo_v2.py",
        ["MiMoV2Attention", "_mimo_reorder_fused_qkv"],
        nn=torch.nn,
        UnquantizedLinearMethod=Unquantized,
        get_attention_tp_size=lambda: tp,
    )
    shard = torch.arange(rows * columns, dtype=torch.float32).view(rows, columns)
    full = torch.empty(tp * rows, columns)
    projection = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer = SimpleNamespace(
        _ep_sp_swa=False,
        qkv_proj=SimpleNamespace(named_parameters=lambda: [("weight", shard)]),
        qkv_proj_full=SimpleNamespace(named_parameters=lambda: [("weight", full)]),
        o_proj=SimpleNamespace(weight=projection, quant_method=Unquantized()),
        attention_sink_bias=None,
        q_size=3072,
        k_size=384,
        v_size=256,
    )
    module.MiMoV2Attention.build_sp_weights(layer)
    parts = [rows * size // 3712 for size in (3072, 384, 256)]
    rank_parts = [(shard + rank * 100000).split(parts) for rank in range(tp)]
    expected = torch.cat([torch.cat(group) for group in zip(*rank_parts)])
    torch.testing.assert_close(full, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        layer.o_proj_full,
        torch.cat([projection + rank * 100000 for rank in range(tp)], dim=1),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("mode", ["tp_full", "tp_swa", "ep_swa"])
def test_decoder_keeps_normal_mlp_reduction_and_swa_dispatch(mode):
    calls = []
    swa = mode != "tp_full"
    local_rows = 4 if swa else 2

    def reduce_scatter(output, value):
        calls.append(("reduce_scatter", "attn_tp"))
        output.copy_((4 * value)[:local_rows])

    communication = load(
        "layers/communicator.py",
        ["ScatterMode", "CommunicateWithAllReduceAndLayerNormFn"],
        Enum=enum.Enum,
        attn_tp_reduce_scatter_tensor=reduce_scatter,
    )

    def communicator(name):
        def prepare_attn(x, residual, batch, **kwargs):
            calls.append((name, "attn"))
            if name == "swa":
                return x[:local_rows], residual[:local_rows]
            return x, residual

        def prepare_mlp(x, residual, batch, **kwargs):
            calls.append((name, "mlp"))
            if name == "swa":
                return x, residual
            return communication.CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual(
                x,
                residual,
                batch,
                lambda value, skip: (value + skip, skip),
                SimpleNamespace(attn_tp_size=4, attn_tp_rank=0),
                residual_input_mode=communication.ScatterMode.TP_ATTN_FULL,
            )

        return SimpleNamespace(
            prepare_attn_and_capture_last_layer_outputs=prepare_attn,
            prepare_mlp=prepare_mlp,
            should_fuse_mlp_allreduce_with_next_layer=lambda batch: False,
            should_use_reduce_scatter=lambda batch: False,
            postprocess_layer=lambda x, residual, batch: (x, residual),
        )

    class Attention:
        _ep_sp_swa = mode == "ep_swa"
        o_proj_full = object() if swa else None

        def __call__(self, positions, hidden_states, forward_batch, sp_swa):
            assert sp_swa == swa
            return hidden_states + 1

    def moe(x, batch, fused, reduce_scatter):
        assert x.shape[0] == local_rows
        return x + 2

    cls = load(
        "models/mimo_v2.py",
        ["MiMoV2DecoderLayer"],
        nn=torch.nn,
        ForwardMode=Mode,
        mimo_tp_sp_active=lambda batch: True,
        _mimo_ep_sp_active=lambda batch: True,
        _mimo_hidden_num_tokens=lambda x: x.shape[0],
    ).MiMoV2DecoderLayer
    layer = SimpleNamespace(
        layer_communicator=communicator("normal"),
        layer_communicator_sp=None if mode == "ep_swa" else communicator("tp"),
        layer_communicator_sp_attn=communicator("swa") if swa else None,
        self_attn=Attention(),
        _fused_rms_qkv_quant_format="",
        _prefill_moe_quant_format=lambda batch: "",
        mlp=moe,
    )
    x = torch.arange(32, dtype=torch.float32).view(8, 4)
    residual = torch.zeros_like(x)
    batch = SimpleNamespace(forward_mode=Mode.EXTEND, extend_seq_lens_cpu=[8])
    output, _ = cls.forward(layer, torch.arange(8), x, batch, residual)
    selected = "swa" if swa else "tp"
    expected_calls = [(selected, "attn"), (selected, "mlp")]
    if not swa:
        expected_calls.append(("reduce_scatter", "attn_tp"))
    assert calls == expected_calls
    expected = x[:local_rows] + 3 if swa else 4 * (x[:local_rows] + 1) + 2
    torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
