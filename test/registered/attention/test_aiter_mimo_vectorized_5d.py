from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import aiter_backend, aiter_utils
from sglang.srt.layers.attention.utils import (
    launch_gather_shuffle_5d_to_linear,
    launch_reshape_and_cache_shuffle_5d,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.model_executor import model_runner_kv_cache_mixin
from sglang.srt.models import mimo_v2


def test_mimo_model_uses_native_v128_only_for_target_vectorized_5d(monkeypatch):
    server_args = SimpleNamespace(attention_backend="aiter")
    layout = SimpleNamespace(get=lambda: "vectorized_5d")
    monkeypatch.setattr(mimo_v2.envs, "SGLANG_AITER_KV_CACHE_LAYOUT", layout)

    kwargs = dict(
        head_dim=192,
        v_head_dim=128,
        num_kv_heads=1,
        server_args=server_args,
    )
    assert not mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)
    assert mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=True)

    kwargs["num_kv_heads"] = 2
    assert mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)
    kwargs["num_kv_heads"] = 1

    layout.get = lambda: "nhd"
    assert mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)

    server_args.attention_backend = "triton"
    assert not mimo_v2._mimo_needs_v_padding(**kwargs, force_v_pad=False)


@pytest.mark.parametrize(
    "full_kv_heads,swa_kv_heads,expected",
    [(1, 1, True), (2, 1, False), (1, 2, False)],
)
def test_mimo_pool_native_v128_requires_one_tp_local_kv_head(
    full_kv_heads, swa_kv_heads, expected
):
    text_config = SimpleNamespace(
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
    )
    model_config = SimpleNamespace(
        head_dim=192,
        get_num_kv_heads=lambda tp_size: full_kv_heads,
        get_swa_num_kv_heads=lambda tp_size: swa_kv_heads,
    )

    assert (
        model_runner_kv_cache_mixin._use_native_mimo_vectorized_v_cache(
            model_config=model_config,
            text_config=text_config,
            attention_backend="aiter",
            kv_cache_layout="vectorized_5d",
            tensor_parallel_size=8,
        )
        is expected
    )


def _make_fresh_asm_case(sequence_length=256, batch_size=2):
    total_tokens = sequence_length * batch_size
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        logits_soft_cap=0.0,
        forward_metadata=SimpleNamespace(max_q_len=sequence_length),
        qo_indptr=torch.arange(
            0,
            total_tokens + 1,
            sequence_length,
            dtype=torch.int32,
        ),
    )
    layer = SimpleNamespace(
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens_cpu=[sequence_length] * batch_size,
    )
    q = torch.zeros((total_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((total_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((total_tokens, 1, 128), dtype=torch.bfloat16)
    return backend, layer, forward_batch, q, k, v


def _make_fresh_swa_case(lengths=(255, 257), sink_dtype=torch.bfloat16):
    total_tokens = sum(lengths)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        logits_soft_cap=0.0,
        forward_metadata=SimpleNamespace(max_q_len=max(lengths)),
        qo_indptr=torch.tensor(cumulative, dtype=torch.int32),
    )
    layer = SimpleNamespace(
        sliding_window_size=128,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
        mimo_original_v_head_dim=128,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0] * len(lengths),
        extend_seq_lens_cpu=list(lengths),
    )
    q = torch.zeros((total_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((total_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((total_tokens, 1, 128), dtype=torch.bfloat16)
    sinks = torch.linspace(-1.0, 1.0, 16, dtype=sink_dtype)
    return backend, layer, forward_batch, q, k, v, sinks


def _make_cached_bf16_chunk_case(prefix_len=256, extend_len=256):
    seq_len = prefix_len + extend_len
    num_blocks = (seq_len + 63) // 64
    k_buf = torch.zeros((num_blocks, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_buf = torch.zeros((num_blocks, 1, 8, 128, 8), dtype=torch.bfloat16)
    pool = SimpleNamespace(
        dtype=torch.bfloat16,
        store_dtype=torch.bfloat16,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    metadata = SimpleNamespace(
        swa_page_table=None,
        kv_indices=torch.arange(seq_len, dtype=torch.int32),
        kv_indptr=torch.tensor([0, seq_len], dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        max_q_len=extend_len,
        max_kv_len=seq_len,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor([0, extend_len], dtype=torch.int32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        scaling=192**-0.5,
        mimo_original_v_head_dim=128,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[prefix_len],
        extend_seq_lens_cpu=[extend_len],
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_sum=seq_len,
    )
    q = torch.zeros((extend_len, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((extend_len, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((extend_len, 1, 128), dtype=torch.bfloat16)
    return backend, layer, forward_batch, q, k, v


@pytest.mark.parametrize("lengths", [(256, 256), (255, 257)])
def test_fresh_mimo_swa_uses_native_v128_ck_varlen(monkeypatch, lengths):
    backend, layer, forward_batch, q, k, v, sinks = _make_fresh_swa_case(
        lengths
    )
    captured = {}

    def fake_varlen(q_varlen, k_varlen, v_varlen, *args, **kwargs):
        captured.update(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_q=args[0],
            cu_k=args[1],
            max_q=args[2],
            max_k=args[3],
            kwargs=kwargs,
        )
        kwargs["out"].fill_(7.0)
        return kwargs["out"]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified fresh SWA input must not use CK prefill")

    monkeypatch.setattr(
        aiter_utils, "MIMO_FRESH_BF16_SWA_VARLEN_ENABLED", True
    )
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "flash_attn_varlen_func", fake_varlen)
    monkeypatch.setattr(
        aiter_utils, "mha_batch_prefill_func", reject_batch_prefill
    )

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=len(lengths) + 1,
        window_size=(128, -1),
        sinks=sinks,
    ).view(sum(lengths), 16, 128)

    assert captured["q"].shape == (sum(lengths), 16, 192)
    assert captured["k"].shape == (sum(lengths), 1, 192)
    assert captured["v"].shape == (sum(lengths), 1, 128)
    assert captured["v"].stride(-2) == 128
    assert captured["cu_q"].tolist() == [0, lengths[0], sum(lengths)]
    assert captured["cu_k"].tolist() == [0, lengths[0], sum(lengths)]
    assert captured["max_q"] == max(lengths)
    assert captured["max_k"] == max(lengths)
    assert captured["kwargs"]["min_seqlen_q"] == 0
    assert captured["kwargs"]["window_size"] == (128, 0, 0)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["sink_ptr"].dtype == torch.float32
    assert captured["kwargs"]["out"].shape == (sum(lengths), 16, 128)
    assert captured["kwargs"]["out"].stride(-2) == 128
    assert torch.all(output == 7.0)


def test_fresh_mimo_swa_gfx942_fallback_uses_asymmetric_batch_prefill(
    monkeypatch,
):
    backend, layer, forward_batch, q, k, v, sinks = _make_fresh_swa_case()
    captured = {}

    def fake_batch_prefill(q_in, k_in, v_in, *args, **kwargs):
        captured.update(q=q_in, k=k_in, v=v_in, args=args, kwargs=kwargs)
        return torch.full(
            (q_in.shape[0], q_in.shape[1], 128),
            13.0,
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(
        aiter_utils, "MIMO_FRESH_BF16_SWA_VARLEN_ENABLED", True
    )
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: False)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", fake_batch_prefill)

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(128, -1),
        sinks=sinks,
    ).view(512, 16, 128)

    assert captured["q"].shape == (512, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["args"][0].tolist() == [0, 255, 512]
    assert captured["args"][1].tolist() == [0, 255, 512]
    assert captured["args"][2].tolist() == list(range(512))
    assert captured["args"][3:5] == (257, 257)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["window_size"] == (128, -1)
    assert captured["kwargs"]["sink_ptr"] is sinks
    assert output.shape == (512, 16, 128)
    assert torch.all(output == 13.0)


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "wrong_arch",
        "short",
        "wrong_window",
        "no_sink",
        "wrong_sink_shape",
        "non_mimo",
        "wrong_v_shape",
        "logit_cap",
    ],
)
def test_fresh_mimo_swa_varlen_contract_guards_fall_back(monkeypatch, guard):
    backend, layer, forward_batch, q, k, v, sinks = _make_fresh_swa_case()
    enabled = guard != "disabled"
    is_gfx950 = guard != "wrong_arch"
    window_size = (128, -1)

    if guard == "short":
        forward_batch.extend_seq_lens_cpu = [128, 384]
    elif guard == "wrong_window":
        window_size = (127, -1)
    elif guard == "no_sink":
        sinks = None
    elif guard == "wrong_sink_shape":
        sinks = torch.zeros(8, dtype=torch.float32)
    elif guard == "non_mimo":
        layer.mimo_original_v_head_dim = None
    elif guard == "wrong_v_shape":
        v = torch.zeros((512, 1, 192), dtype=torch.bfloat16)
    elif guard == "logit_cap":
        backend.logits_soft_cap = 50.0

    monkeypatch.setattr(
        aiter_utils, "MIMO_FRESH_BF16_SWA_VARLEN_ENABLED", enabled
    )
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: is_gfx950)
    monkeypatch.setattr(aiter_utils, "flash_attn_varlen_func", lambda: None)

    assert not aiter_utils.can_use_mimo_fresh_bf16_swa_varlen(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        window_size,
        sinks,
    )


def test_fresh_uniform_mimo_extend_uses_gfx950_bf16_asm(monkeypatch):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    captured = {}

    def fake_asm(q_4d, k_4d, v_4d, *args):
        out = args[8]
        captured.update(q=q_4d, k=k_4d, v=v_4d, out=out)
        out.fill_(3.0)
        return [out]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified fresh uniform input must not use CK prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_fwd", fake_asm)
    monkeypatch.setattr(
        aiter_utils, "mha_batch_prefill_func", reject_batch_prefill
    )

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(-1, -1),
        sinks=None,
    ).view(2, 256, 16, 128)

    assert captured["q"].shape == (2, 256, 16, 192)
    assert captured["k"].shape == (2, 256, 1, 192)
    assert captured["v"].shape == (2, 256, 1, 128)
    assert captured["out"].shape == (2, 256, 16, 128)
    assert captured["out"].stride(-2) == 128
    assert torch.all(output == 3.0)
    assert torch.isfinite(output).all()


def test_fresh_ragged_mimo_extend_uses_gfx950_bf16_varlen_asm(monkeypatch):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    forward_batch.extend_seq_lens_cpu = [255, 257]
    backend.qo_indptr = torch.tensor([0, 255, 512], dtype=torch.int32)
    captured = {}

    def fake_varlen_asm(q_varlen, k_varlen, v_varlen, *args):
        out = args[15]
        captured.update(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_q=args[0],
            cu_k=args[1],
            max_q=args[2],
            min_q=args[4],
            out=out,
        )
        out.fill_(5.0)
        return [out]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified fresh ragged input must not use CK prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(
        aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True
    )
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", fake_varlen_asm)
    monkeypatch.setattr(
        aiter_utils, "mha_batch_prefill_func", reject_batch_prefill
    )

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=3,
        window_size=(-1, -1),
        sinks=None,
    ).view(512, 16, 128)

    assert captured["q"].shape == (512, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["cu_q"].tolist() == [0, 255, 512]
    assert captured["cu_k"].tolist() == [0, 255, 512]
    assert captured["max_q"] == 257
    assert captured["min_q"] == 255
    assert captured["out"].shape == (512, 16, 128)
    assert captured["out"].stride(-2) == 128
    assert torch.all(output == 5.0)
    assert torch.isfinite(output).all()


def test_cached_bf16_chunk_prefill_uses_gfx950_varlen_asm(monkeypatch):
    backend, layer, forward_batch, q, k, v = _make_cached_bf16_chunk_case()
    metadata = backend.forward_metadata
    captured = {}

    def fake_gather(k_buf, v_buf, slot_ids):
        captured["gather_slot_ids"] = slot_ids
        seq_len = slot_ids.numel()
        return (
            torch.zeros((seq_len, 1, 192), dtype=torch.bfloat16),
            torch.zeros((seq_len, 1, 128), dtype=torch.bfloat16),
        )

    def fake_varlen_asm(q_varlen, k_varlen, v_varlen, *args):
        out = args[15]
        captured.update(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            cu_q=args[0],
            cu_k=args[1],
            max_q=args[2],
            max_k=args[3],
            min_q=args[4],
            causal=args[9],
            window_left=args[10],
            window_right=args[11],
            out=out,
        )
        out.fill_(11.0)
        return [out]

    def reject_batch_prefill(*args, **kwargs):
        raise AssertionError("qualified BF16 chunk must not use CK prefill")

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(
        aiter_utils, "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED", True
    )
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", fake_varlen_asm)
    monkeypatch.setattr(
        aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather
    )
    monkeypatch.setattr(
        aiter_utils, "mha_batch_prefill_func", reject_batch_prefill
    )

    output = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=None,
    ).view(q.shape[0], 16, 128)

    assert torch.equal(captured["gather_slot_ids"], metadata.kv_indices)
    assert captured["q"].shape == (256, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["v"].stride(-2) == 128
    assert captured["cu_q"].tolist() == [0, 256]
    assert captured["cu_k"].tolist() == [0, 512]
    assert captured["max_q"] == 256
    assert captured["max_k"] == 512
    assert captured["min_q"] == 256
    assert captured["causal"] is True
    assert captured["window_left"] == -1
    assert captured["window_right"] == -1
    assert captured["out"].shape == (256, 16, 128)
    assert captured["out"].stride(-2) == 128
    assert torch.all(output == 11.0)


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "short",
        "swa",
        "sink",
        "window",
        "logit_cap",
        "wrong_shape",
        "wrong_total",
    ],
)
def test_fresh_bf16_varlen_asm_contract_guards_fall_back(monkeypatch, guard):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    forward_batch.extend_seq_lens_cpu = [255, 257]
    backend.qo_indptr = torch.tensor([0, 255, 512], dtype=torch.int32)
    window_size = (-1, -1)
    sinks = None
    if guard == "short":
        forward_batch.extend_seq_lens_cpu = [128, 384]
    elif guard == "swa":
        layer.sliding_window_size = 4096
    elif guard == "sink":
        sinks = torch.zeros(16, dtype=torch.float32)
    elif guard == "window":
        window_size = (4096, -1)
    elif guard == "logit_cap":
        backend.logits_soft_cap = 50.0
    elif guard == "wrong_shape":
        layer.tp_q_head_num = 8
    elif guard == "wrong_total":
        forward_batch.extend_seq_lens_cpu = [255, 256]

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", True)
    monkeypatch.setattr(
        aiter_utils,
        "MIMO_FRESH_BF16_ASM_VARLEN_ENABLED",
        guard != "disabled",
    )
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_varlen_fwd", lambda *args: None)
    assert not aiter_utils.can_use_mimo_fresh_bf16_varlen_asm(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        window_size,
        sinks,
    )


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "nonuniform",
        "short",
        "swa",
        "sink",
        "window",
        "logit_cap",
        "wrong_shape",
    ],
)
def test_fresh_bf16_asm_contract_guards_fall_back(monkeypatch, guard):
    backend, layer, forward_batch, q, k, v = _make_fresh_asm_case()
    enabled = guard != "disabled"
    window_size = (-1, -1)
    sinks = None
    if guard == "nonuniform":
        forward_batch.extend_seq_lens_cpu = [255, 257]
    elif guard == "short":
        backend, layer, forward_batch, q, k, v = _make_fresh_asm_case(
            sequence_length=128
        )
    elif guard == "swa":
        layer.sliding_window_size = 4096
    elif guard == "sink":
        sinks = torch.zeros(16, dtype=torch.float32)
    elif guard == "window":
        window_size = (4096, -1)
    elif guard == "logit_cap":
        backend.logits_soft_cap = 50.0
    elif guard == "wrong_shape":
        layer.tp_q_head_num = 8

    monkeypatch.setattr(aiter_utils, "MIMO_FRESH_BF16_ASM_ENABLED", enabled)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "fmha_v3_fwd", lambda *args: [args[11]])
    assert not aiter_utils.can_use_mimo_fresh_bf16_asm(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        window_size,
        sinks,
    )


def _run_vectorized_prefill_scale_case(
    monkeypatch,
    *,
    direct_paged: bool,
    flydsl_prefill: bool = False,
    flydsl_model_marker: bool = True,
    flypa_env: bool = False,
    swa: bool = False,
):
    captured = {}
    q_descale = torch.tensor([0.125], dtype=torch.float32)
    k_descale = torch.tensor([0.25], dtype=torch.float32)
    v_descale = torch.tensor([0.5], dtype=torch.float32)

    def fake_quantize(q):
        return q.to(fp8_dtype), q_descale

    def fake_batch_prefill(q, k, v, *args, **kwargs):
        captured.update(q=q, k=k, v=v, kwargs=kwargs, selected="ck")
        return torch.zeros(
            (q.shape[0], q.shape[1], 128), dtype=torch.bfloat16, device=q.device
        )

    def fake_flydsl_prefill(q, k, v, *args, **kwargs):
        captured.update(
            q=q, k=k, v=v, args=args, kwargs=kwargs, selected="flydsl"
        )
        return torch.zeros(
            (q.shape[0], q.shape[1], 128), dtype=torch.bfloat16, device=q.device
        )

    def fake_flypa(**compile_kwargs):
        def run(q, k, v, *args):
            captured.update(
                q=q,
                k=k,
                v=v,
                args=args,
                compile=compile_kwargs,
                selected="flypa",
            )
            return torch.zeros(
                (
                    q.shape[0],
                    compile_kwargs["num_qo_heads"],
                    compile_kwargs["head_dim_v"],
                ),
                dtype=torch.bfloat16,
            )

        return run

    def fake_gather(k_buf, v_buf, slot_ids):
        captured["gather_slot_ids"] = slot_ids
        return (
            torch.zeros((1, 1, 192), dtype=torch.uint8),
            torch.zeros((1, 1, 128), dtype=torch.uint8),
        )

    monkeypatch.setenv(
        aiter_utils.FLYPA_MIMO_PREFILL_ENV, "1" if flypa_env else "0"
    )
    monkeypatch.setenv(
        aiter_utils.FLYDSL_MIMO_PREFILL_ENV, "1" if flydsl_prefill else "0"
    )
    monkeypatch.setattr(
        aiter_utils, "quantize_query_per_tensor_fp8", fake_quantize
    )
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", fake_batch_prefill)
    monkeypatch.setattr(
        aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather
    )
    monkeypatch.setattr(aiter_utils, "flypa", fake_flypa)
    if flydsl_prefill or flypa_env:
        monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
        monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: False)
    if flydsl_prefill:
        monkeypatch.setattr(
            aiter_utils,
            "load_flydsl_mimo_prefill_kernel",
            lambda: SimpleNamespace(run=fake_flydsl_prefill),
        )

    k_buf = torch.zeros((2, 1, 12, 64, 16), dtype=torch.uint8)
    v_buf = torch.zeros((2, 1, 4, 128, 16), dtype=torch.uint8)
    pool = SimpleNamespace(
        dtype=fp8_dtype,
        store_dtype=torch.uint8,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    metadata = SimpleNamespace(
        swa_page_table=(
            torch.tensor([1], dtype=torch.int32) if swa else None
        ),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        max_q_len=4096 if flydsl_prefill else 1,
        max_kv_len=8192 if flydsl_prefill else 1,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=fp8_dtype,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        k_scale=k_descale,
        v_scale=v_descale,
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=128 if swa else -1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        k_scale=None,
        v_scale=None,
        mimo_original_v_head_dim=(
            128 if flydsl_prefill and flydsl_model_marker else None
        ),
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[1],
        seq_lens_sum=1,
    )
    q = torch.zeros((1, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((1, 192), dtype=torch.bfloat16)
    v = torch.zeros((1, 128), dtype=torch.bfloat16)
    sinks = None if direct_paged else torch.zeros(16, dtype=torch.float32)
    window_size = (128, -1) if swa else (-1, -1)

    out = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=window_size,
        sinks=sinks,
    )

    assert out.shape == (1, 16 * 128)
    assert captured["q"].dtype == fp8_dtype
    assert captured["kwargs"]["q_descale"] is q_descale
    assert captured["kwargs"]["k_descale"] is k_descale
    assert captured["kwargs"]["v_descale"] is v_descale
    if direct_paged:
        assert "gather_slot_ids" not in captured
        assert captured["k"].data_ptr() == k_buf.data_ptr()
        assert captured["v"].data_ptr() == v_buf.data_ptr()
    else:
        expected_slot_ids = (
            metadata.swa_page_table if swa else metadata.kv_indices
        )
        assert torch.equal(captured["gather_slot_ids"], expected_slot_ids)
    return captured


@pytest.mark.parametrize("direct_paged", [True, False])
def test_vectorized_prefill_forwards_independent_qkv_descales(
    monkeypatch, direct_paged
):
    _run_vectorized_prefill_scale_case(monkeypatch, direct_paged=direct_paged)


def test_cached_fp8_swa_gathers_v128_and_forwards_window_sink(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch,
        direct_paged=False,
        swa=True,
    )
    assert captured["gather_slot_ids"].tolist() == [1]
    assert captured["k"].shape == (1, 1, 192)
    assert captured["v"].shape == (1, 1, 128)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["window_size"] == (128, -1)
    assert captured["kwargs"]["sink_ptr"].shape == (16,)


def test_qualified_long_mimo_prefill_selects_flydsl(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch, direct_paged=True, flydsl_prefill=True
    )
    assert captured["selected"] == "flydsl"
    assert captured["kwargs"]["max_seqlen_q"] == 4096
    assert captured["kwargs"]["max_seqlen_kv"] == 8192


def test_gfx950_fp8_flydsl_is_preferred_over_flypa(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch,
        direct_paged=True,
        flydsl_prefill=True,
        flypa_env=True,
    )
    assert captured["selected"] == "flydsl"
    assert captured["kwargs"]["max_seqlen_q"] == 4096
    assert captured["kwargs"]["max_seqlen_kv"] == 8192


def test_flydsl_env_falls_back_to_ck_without_v128_mimo_marker(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch,
        direct_paged=True,
        flydsl_prefill=True,
        flydsl_model_marker=False,
    )
    assert captured["selected"] == "ck"


def _run_flypa_prefill_case(
    monkeypatch,
    *,
    gfx942: bool,
    gfx950: bool,
    flypa_env: bool,
    kv_dtype,
    with_paged_metadata: bool = True,
    prefix_lens=None,
    flydsl_env: bool = False,
):
    captured = {}

    def fake_flypa(**compile_kwargs):
        def run(q, k, v, *args):
            captured.update(
                q=q,
                k=k,
                v=v,
                args=args,
                compile=compile_kwargs,
                selected="flypa",
            )
            return torch.zeros(
                (q.shape[0], compile_kwargs["num_qo_heads"], compile_kwargs["head_dim_v"]),
                dtype=torch.bfloat16,
            )

        return run

    def fake_batch_prefill(q, k, v, *args, **kwargs):
        captured.update(q=q, k=k, v=v, kwargs=kwargs, selected="ck")
        return torch.zeros(
            (q.shape[0], q.shape[1], 128), dtype=torch.bfloat16, device=q.device
        )

    def fake_gather(k_buf, v_buf, slot_ids):
        captured["gather_slot_ids"] = slot_ids
        return (
            torch.zeros((slot_ids.numel(), 1, 192), dtype=kv_dtype),
            torch.zeros((slot_ids.numel(), 1, 128), dtype=kv_dtype),
        )

    monkeypatch.setenv(aiter_utils.FLYPA_MIMO_PREFILL_ENV, "1" if flypa_env else "0")
    monkeypatch.setenv(
        aiter_utils.FLYDSL_MIMO_PREFILL_ENV, "1" if flydsl_env else "0"
    )
    monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: gfx942)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: gfx950)
    monkeypatch.setattr(aiter_utils, "flypa", fake_flypa)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", fake_batch_prefill)
    monkeypatch.setattr(
        aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather
    )
    if flydsl_env:

        def fake_flydsl_prefill(q, k, v, *args, **kwargs):
            captured.update(
                q=q, k=k, v=v, args=args, kwargs=kwargs, selected="flydsl"
            )
            return torch.zeros(
                (q.shape[0], q.shape[1], 128),
                dtype=torch.bfloat16,
                device=q.device,
            )

        monkeypatch.setattr(
            aiter_utils,
            "load_flydsl_mimo_prefill_kernel",
            lambda: SimpleNamespace(run=fake_flydsl_prefill),
        )
    if kv_dtype == fp8_dtype:
        q_descale = torch.tensor([0.125], dtype=torch.float32)
        monkeypatch.setattr(
            aiter_utils,
            "quantize_query_per_tensor_fp8",
            lambda q: (q.to(fp8_dtype), q_descale),
        )
        pack = 16
        store_dtype = torch.uint8
        k_buf = torch.zeros((2, 1, 12, 64, 16), dtype=torch.uint8)
        v_buf = torch.zeros((2, 1, 4, 128, 16), dtype=torch.uint8)
    else:
        pack = 8
        store_dtype = torch.bfloat16
        k_buf = torch.zeros((2, 1, 24, 64, 8), dtype=torch.bfloat16)
        v_buf = torch.zeros((2, 1, 8, 128, 8), dtype=torch.bfloat16)

    pool = SimpleNamespace(
        dtype=kv_dtype,
        store_dtype=store_dtype,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    metadata = SimpleNamespace(
        swa_page_table=None,
        kv_indices=torch.arange(64, dtype=torch.int32),
        kv_indptr=torch.tensor([0, 64], dtype=torch.int32),
        paged_kv_indptr=(
            torch.tensor([0, 1], dtype=torch.int32) if with_paged_metadata else None
        ),
        paged_kv_indices=(
            torch.tensor([0], dtype=torch.int32) if with_paged_metadata else None
        ),
        paged_kv_last_page_len=(
            torch.tensor([64], dtype=torch.int32) if with_paged_metadata else None
        ),
        max_q_len=64,
        max_kv_len=64,
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        page_size=64,
        logits_soft_cap=0.0,
        token_to_kv_pool=pool,
        forward_metadata=metadata,
        qo_indptr=torch.tensor([0, 64], dtype=torch.int32),
        k_scale=torch.tensor([0.25], dtype=torch.float32),
        v_scale=torch.tensor([0.5], dtype=torch.float32),
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        head_dim=192,
        k_scale=None,
        v_scale=None,
        mimo_original_v_head_dim=128,
    )
    prefix_lens = [64] if prefix_lens is None else list(prefix_lens)
    extend_len = 64
    seq_len = prefix_lens[0] + extend_len
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens_cpu=[extend_len],
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_sum=seq_len,
    )
    q = torch.zeros((64, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((64, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((64, 1, 128), dtype=torch.bfloat16)
    out = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=None,
    )
    assert out.shape == (64, 16 * 128)
    return captured, k_buf, v_buf, pack


def test_gfx942_bf16_flypa_prefill_skips_gather(monkeypatch):
    captured, k_buf, v_buf, pack = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "flypa"
    assert "gather_slot_ids" not in captured
    assert captured["k"].data_ptr() == k_buf.data_ptr()
    assert captured["v"].data_ptr() == v_buf.data_ptr()
    assert captured["q"].shape == (64, 16, 192)
    assert captured["compile"]["head_dim_v"] == 128
    assert pack == 8


def test_gfx942_fp8_flypa_prefill_skips_ck(monkeypatch):
    captured, k_buf, v_buf, pack = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=fp8_dtype,
    )
    assert captured["selected"] == "flypa"
    assert captured["k"].data_ptr() == k_buf.data_ptr()
    assert pack == 16
    assert captured["q"].dtype == fp8_dtype


def test_gfx950_fp8_below_flydsl_size_falls_back_to_flypa(monkeypatch):
    captured, k_buf, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=fp8_dtype,
        flydsl_env=True,
    )
    assert captured["selected"] == "flypa"
    assert captured["k"].data_ptr() == k_buf.data_ptr()


def test_gfx950_bf16_flypa_prefill_is_enabled(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "flypa"


def test_gfx950_flypa_env_keeps_fresh_asm_shortcut(monkeypatch):
    selected = {}

    def fake_asm(q, k, v, layer, forward_batch):
        selected["path"] = "asm"
        return torch.zeros((q.shape[0], 16 * 128), dtype=torch.bfloat16)

    monkeypatch.setattr(
        aiter_utils, "can_use_mimo_fresh_bf16_asm", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(aiter_utils, "mimo_fresh_bf16_asm", fake_asm)
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        prefix_lens=[0],
    )
    assert selected["path"] == "asm"
    assert captured.get("selected") != "flypa"


def test_gfx942_flypa_env_uses_paged_path_on_fresh_chunk(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        prefix_lens=[0],
    )
    assert captured["selected"] == "flypa"


def test_gfx950_cached_asm_is_preferred_over_flypa(monkeypatch):
    selected = {}

    def fake_chunk_asm(*args, **kwargs):
        selected["path"] = "chunk_asm"
        q = args[1]
        return torch.zeros((q.shape[0], 16 * 128), dtype=torch.bfloat16)

    monkeypatch.setattr(
        aiter_utils, "can_use_mimo_chunk_bf16_varlen_asm", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(aiter_utils, "mimo_chunk_bf16_varlen_asm", fake_chunk_asm)
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=True,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        prefix_lens=[256],
    )
    assert selected["path"] == "chunk_asm"
    assert captured.get("selected") != "flypa"


def test_flypa_env_off_gfx942_bf16_gathers(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=False,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "ck"
    assert "gather_slot_ids" in captured


def test_flypa_ignored_without_gfx942_or_gfx950(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=False,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
    )
    assert captured["selected"] == "ck"
    assert "gather_slot_ids" in captured


def test_gfx942_bf16_flypa_requires_paged_kv_metadata(monkeypatch):
    captured, _, _, _ = _run_flypa_prefill_case(
        monkeypatch,
        gfx942=True,
        gfx950=False,
        flypa_env=True,
        kv_dtype=torch.bfloat16,
        with_paged_metadata=False,
    )
    assert captured["selected"] == "ck"
    assert "gather_slot_ids" in captured


def _mimo_paged_metadata_kwargs(**overrides):
    kwargs = dict(
        kv_cache_is_vectorized_5d=True,
        page_size=64,
        kv_cache_dtype=torch.bfloat16,
        q_dtype=torch.bfloat16,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=192,
    )
    kwargs.update(overrides)
    return kwargs


def test_paged_kv_metadata_enabled_for_bf16_and_fp8_shuffle_5d():
    assert aiter_utils.can_build_mimo_paged_kv_metadata(**_mimo_paged_metadata_kwargs())
    assert aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(kv_cache_dtype=fp8_dtype)
    )


def test_paged_kv_metadata_rejected_outside_mimo_contract():
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(kv_cache_is_vectorized_5d=False)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(page_size=16)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(kv_cache_dtype=torch.float16)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(num_kv_heads=2)
    )
    assert not aiter_utils.can_build_mimo_paged_kv_metadata(
        **_mimo_paged_metadata_kwargs(num_qo_heads=8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_query_fp8_quantization_returns_its_own_non_unit_descale():
    q = torch.linspace(
        -96.0,
        96.0,
        2 * 16 * 192,
        dtype=torch.bfloat16,
        device="cuda",
    ).view(2, 16 * 192)

    q_fp8, scale = aiter_utils.quantize_query_per_tensor_fp8(q)
    q_dequant = q_fp8.float() * scale

    assert q_fp8.dtype == fp8_dtype
    assert q_fp8.shape == q.shape
    assert scale.shape == (1,)
    assert scale.dtype == torch.float32
    assert scale.item() != pytest.approx(1.0)
    expected_scale = q.abs().max().float() / torch.finfo(fp8_dtype).max
    torch.testing.assert_close(
        scale, expected_scale.view(1), rtol=1e-3, atol=1e-6
    )
    torch.testing.assert_close(q_dequant, q.float(), rtol=0.125, atol=0.5)


def test_query_fp8_quantization_rejects_prequantized_input():
    q = torch.zeros((1, 16 * 192), dtype=fp8_dtype)
    with pytest.raises(ValueError, match="non-FP8"):
        aiter_utils.quantize_query_per_tensor_fp8(q)


def test_flydsl_target_verify_accepts_bf16_vectorized_5d():
    batch_size = 2
    query_length = 4
    partitions = 8
    equivalent_group = query_length * 16
    scalar_numel = batch_size * partitions * equivalent_group
    captured = {}

    def fake_pa_decode_tile(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(3.0)

    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    backend = SimpleNamespace(
        _flydsl_pa_decode_tile=fake_pa_decode_tile,
        _flydsl_pa_decode_num_partitions=partitions,
        _flydsl_pa_decode_pmax=torch.empty(scalar_numel, dtype=torch.float32),
        _flydsl_pa_decode_psum=torch.empty(scalar_numel, dtype=torch.float32),
        _flydsl_pa_decode_pout=torch.empty(
            scalar_numel * 128, dtype=torch.bfloat16
        ),
        _flydsl_pa_decode_context_lengths=torch.empty(
            batch_size, dtype=torch.int32
        ),
        _flydsl_pa_decode_workspace_max_bs=batch_size,
        forward_metadata=SimpleNamespace(
            max_q_len=query_length,
            kv_indices=block_tables,
        ),
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
    )
    layer = SimpleNamespace(
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        scaling=192**-0.5,
        logit_cap=0.0,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([100, 120], dtype=torch.int64),
    )
    q = torch.zeros(
        (batch_size * query_length, 16 * 192), dtype=torch.bfloat16
    )
    output = torch.empty(
        (batch_size * query_length, 16 * 128), dtype=torch.bfloat16
    )
    k_cache = torch.zeros((4, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.zeros((4, 1, 8, 128, 8), dtype=torch.bfloat16)

    aiter_utils.forward_target_verify_flydsl_5d(
        backend,
        q,
        layer,
        forward_batch,
        k_cache,
        v_cache,
        output,
        sinks=None,
    )

    assert captured["key_cache"] is k_cache
    assert captured["value_cache"] is v_cache
    assert captured["key_scale"] is None
    assert captured["value_scale"] is None
    assert captured["context_lengths"].tolist() == [104, 124]
    assert captured["query"].shape == (batch_size * query_length, 16, 192)
    assert captured["output"].shape == (batch_size * query_length, 16, 128)
    assert captured["pout"].shape == (batch_size, 1, partitions, 64, 128)
    assert torch.all(output == 3.0)


@pytest.mark.parametrize("sliding_window", [-1, 128], ids=["full", "swa"])
def test_gluon_decode_uses_native_v128_workspace(monkeypatch, sliding_window):
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(2.0)

    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", fake_decode)
    monkeypatch.setattr(aiter_utils, "get_recommended_splits", lambda *_: 2)

    batch_size = 2
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            kv_indices=torch.tensor([[0], [1]], dtype=torch.int32),
            swa_page_table=(
                torch.tensor([[1], [0]], dtype=torch.int32)
                if sliding_window > 0
                else None
            ),
        ),
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
    )
    layer = SimpleNamespace(
        sliding_window_size=sliding_window,
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        scaling=192**-0.5,
        k_scale=None,
        v_scale=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([64, 96], dtype=torch.int32),
    )
    q = torch.zeros((batch_size, 16 * 192), dtype=torch.bfloat16)
    output = torch.empty((batch_size, 16 * 128), dtype=torch.bfloat16)
    k_cache = torch.zeros((2, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.zeros((2, 1, 8, 128, 8), dtype=torch.bfloat16)

    aiter_utils.forward_decode_vectorized_5d(
        backend,
        q,
        layer,
        forward_batch,
        k_cache,
        v_cache,
        output,
        sinks=None,
    )

    assert captured["output"].shape == (batch_size, 16, 128)
    assert captured["query"].shape == (batch_size, 16, 192)
    assert captured["temporary_output"].shape[-1] == 128
    assert captured["sliding_window"] == max(sliding_window, 0)
    assert torch.all(output == 2.0)


@pytest.mark.parametrize("sliding_window", [-1, 128], ids=["full", "swa"])
def test_gluon_target_verify_accepts_native_v128(monkeypatch, sliding_window):
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        kwargs["output"].fill_(4.0)

    monkeypatch.setattr(aiter_utils, "pa_decode_gluon", fake_decode)
    monkeypatch.setattr(aiter_utils, "get_recommended_splits", lambda *_: 2)

    batch_size = 2
    query_length = 4
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            max_q_len=query_length,
            kv_indices=torch.tensor([[0], [1]], dtype=torch.int32),
            swa_page_table=(
                torch.tensor([[1], [0]], dtype=torch.int32)
                if sliding_window > 0
                else None
            ),
        ),
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
    )
    layer = SimpleNamespace(
        sliding_window_size=sliding_window,
        tp_q_head_num=16,
        tp_k_head_num=1,
        qk_head_dim=192,
        v_head_dim=128,
        scaling=192**-0.5,
        logit_cap=0.0,
        k_scale=None,
        v_scale=None,
    )
    forward_batch = SimpleNamespace(
        batch_size=batch_size,
        seq_lens=torch.tensor([64, 96], dtype=torch.int32),
    )
    q = torch.zeros(
        (batch_size * query_length, 16 * 192), dtype=torch.bfloat16
    )
    output = torch.empty(
        (batch_size * query_length, 16 * 128), dtype=torch.bfloat16
    )
    k_cache = torch.zeros((2, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_cache = torch.zeros((2, 1, 8, 128, 8), dtype=torch.bfloat16)

    aiter_utils.forward_target_verify_vectorized_5d(
        backend,
        q,
        layer,
        forward_batch,
        k_cache,
        v_cache,
        output,
        sinks=None,
    )

    assert captured["output"].shape == (
        batch_size * query_length,
        16,
        128,
    )
    assert captured["query"].shape == (
        batch_size * query_length,
        16,
        192,
    )
    assert captured["temporary_output"].shape[-1] == 128
    assert captured["context_lengths"].tolist() == [68, 100]
    assert captured["sliding_window"] == max(sliding_window, 0)
    assert torch.all(output == 4.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "dtype,vector_width",
    [(torch.bfloat16, 8), (torch.uint8, 16)],
    ids=["bf16", "fp8-storage"],
)
def test_shuffle_5d_asymmetric_writer_gather_round_trip(dtype, vector_width):
    device = "cuda"
    num_tokens = 5
    num_heads = 1
    page_size = 64
    slots = torch.tensor([0, 65, 3, 127, 64], dtype=torch.int64, device=device)
    key = torch.arange(
        num_tokens * num_heads * 192,
        dtype=torch.int32,
        device=device,
    ).remainder(251).to(dtype).view(num_tokens, num_heads, 192)
    value = torch.arange(
        num_tokens * num_heads * 128,
        dtype=torch.int32,
        device=device,
    ).add(17).remainder(251).to(dtype).view(num_tokens, num_heads, 128)
    key_cache = torch.zeros(
        (2, num_heads, 192 // vector_width, page_size, vector_width),
        dtype=dtype,
        device=device,
    )
    value_cache = torch.zeros(
        (2, num_heads, page_size // vector_width, 128, vector_width),
        dtype=dtype,
        device=device,
    )

    launch_reshape_and_cache_shuffle_5d(
        key,
        value,
        key_cache,
        value_cache,
        slots,
    )
    gathered_key, gathered_value = launch_gather_shuffle_5d_to_linear(
        key_cache,
        value_cache,
        slots,
    )

    assert gathered_key.shape == key.shape
    assert gathered_value.shape == value.shape
    torch.testing.assert_close(gathered_key, key, rtol=0, atol=0)
    torch.testing.assert_close(gathered_value, value, rtol=0, atol=0)


def _make_flydsl_bf16_backend():
    backend = object.__new__(aiter_backend.AiterAttnBackend)
    backend.kv_cache_is_vectorized_5d = True
    backend.use_mla = False
    backend.topk = 1
    backend.page_size = 64
    backend.max_context_len = 1_048_576
    backend.num_head = 16
    backend.num_kv_head = 1
    backend.head_dim = 192
    backend.v_head_dim = 128
    backend.input_dtype = torch.bfloat16
    backend.kv_cache_dtype = torch.bfloat16
    backend._flydsl_pa_decode_workspace_max_bs = 0
    backend._flydsl_pa_decode_compiled = False
    return backend


def _patch_flydsl_backend_dependencies(monkeypatch, *, gfx942, gfx950):
    captured = {}

    def fake_compile_pa_decode_tile(**kwargs):
        captured["tile"] = kwargs

    def fake_compile_pa_decode_reduce(**kwargs):
        captured["reduce"] = kwargs

    kernels = SimpleNamespace(
        pa_decode_tile=lambda **kwargs: None,
        compile_pa_decode_tile=fake_compile_pa_decode_tile,
        compile_pa_decode_reduce=fake_compile_pa_decode_reduce,
        version="test",
        runtime_path="test-runtime",
        kernel_path="test-kernel",
    )
    monkeypatch.setattr(
        aiter_backend.envs,
        "SGLANG_AITER_PA_DECODE_IMPL",
        SimpleNamespace(get=lambda: "flydsl"),
    )
    monkeypatch.setattr(aiter_backend, "is_gfx942_supported", lambda: gfx942)
    monkeypatch.setattr(aiter_backend, "is_gfx95_supported", lambda: gfx950)
    monkeypatch.setattr(aiter_backend, "get_flydsl_mimo_num_partitions", lambda: 8)
    monkeypatch.setattr(
        aiter_backend, "load_flydsl_pa_decode_kernels", lambda: kernels
    )
    monkeypatch.setattr(
        aiter_backend.AiterAttnBackend,
        "_ensure_flydsl_pa_decode_workspace",
        lambda self, max_bs: None,
    )
    return captured


@pytest.mark.parametrize(
    "gfx942,gfx950",
    [(True, False), (False, True)],
    ids=["gfx942", "gfx950"],
)
def test_flydsl_bf16_decode_configuration_accepts_supported_arches(
    monkeypatch, gfx942, gfx950
):
    captured = _patch_flydsl_backend_dependencies(
        monkeypatch, gfx942=gfx942, gfx950=gfx950
    )
    backend = _make_flydsl_bf16_backend()

    backend._configure_flydsl_pa_decode(max_bs=1)
    backend._compile_flydsl_pa_decode()

    assert backend._use_flydsl_pa_decode
    assert backend._flydsl_pa_decode_num_partitions == 8
    assert backend._flydsl_pa_decode_tile is not None
    assert captured["tile"]["bf16_kv"] is True
    assert captured["tile"]["head_dim"] == 192
    assert captured["tile"]["v_head_dim"] == 128
    assert captured["reduce"]["head_size"] == 128


def test_flydsl_fp8_decode_compile_disables_bf16_kv(monkeypatch):
    captured = _patch_flydsl_backend_dependencies(
        monkeypatch, gfx942=False, gfx950=True
    )
    backend = _make_flydsl_bf16_backend()
    backend.kv_cache_dtype = fp8_dtype

    backend._configure_flydsl_pa_decode(max_bs=1)
    backend._compile_flydsl_pa_decode()

    assert captured["tile"]["bf16_kv"] is False


def test_flydsl_pa_decode_loader_rejects_legacy_compile_api(monkeypatch):
    def legacy_compile_pa_decode_tile(*, head_dim):
        pass

    modules = {
        "flydsl": SimpleNamespace(__version__="0.2.4", __file__="test-runtime"),
        "kernels.attention.pa_decode_tile": SimpleNamespace(
            __file__="legacy-pa-decode-tile",
            pa_decode_tile=lambda **kwargs: None,
            compile_pa_decode_tile=legacy_compile_pa_decode_tile,
        ),
        "kernels.attention.pa_decode_swa": SimpleNamespace(
            compile_pa_decode_sw_reduce=lambda **kwargs: None
        ),
        "kernels.attention.pa_decode_fp8": SimpleNamespace(),
    }
    monkeypatch.setattr(
        aiter_utils.importlib, "import_module", lambda name: modules[name]
    )
    aiter_utils.load_flydsl_pa_decode_kernels.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="v_head_dim"):
            aiter_utils.load_flydsl_pa_decode_kernels()
    finally:
        aiter_utils.load_flydsl_pa_decode_kernels.cache_clear()


def test_flydsl_bf16_decode_configuration_rejects_unsupported_arch(monkeypatch):
    _patch_flydsl_backend_dependencies(
        monkeypatch, gfx942=False, gfx950=False
    )
    backend = _make_flydsl_bf16_backend()

    with pytest.raises(RuntimeError, match="validated only on gfx942 or gfx950"):
        backend._configure_flydsl_pa_decode(max_bs=1)
