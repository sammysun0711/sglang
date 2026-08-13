from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import aiter_utils
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype


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
        v_head_dim=192,
        head_dim=192,
        scaling=192**-0.5,
    )
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens_cpu=[sequence_length] * batch_size,
    )
    q = torch.zeros((total_tokens, 16 * 192), dtype=torch.bfloat16)
    k = torch.zeros((total_tokens, 1, 192), dtype=torch.bfloat16)
    v = torch.zeros((total_tokens, 1, 192), dtype=torch.bfloat16)
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
        v_head_dim=192,
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
    v = torch.zeros((total_tokens, 1, 192), dtype=torch.bfloat16)
    sinks = torch.linspace(-1.0, 1.0, 16, dtype=sink_dtype)
    return backend, layer, forward_batch, q, k, v, sinks


def _make_cached_bf16_chunk_case(prefix_len=256, extend_len=256):
    seq_len = prefix_len + extend_len
    num_blocks = (seq_len + 63) // 64
    k_buf = torch.zeros((num_blocks, 1, 24, 64, 8), dtype=torch.bfloat16)
    v_buf = torch.zeros((num_blocks, 1, 8, 192, 8), dtype=torch.bfloat16)
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
        v_head_dim=192,
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
    v = torch.zeros((extend_len, 1, 192), dtype=torch.bfloat16)
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
    ).view(sum(lengths), 16, 192)

    assert captured["q"].shape == (sum(lengths), 16, 192)
    assert captured["k"].shape == (sum(lengths), 1, 192)
    assert captured["v"].shape == (sum(lengths), 1, 128)
    assert captured["v"].stride(-2) == 192
    assert captured["cu_q"].tolist() == [0, lengths[0], sum(lengths)]
    assert captured["cu_k"].tolist() == [0, lengths[0], sum(lengths)]
    assert captured["max_q"] == max(lengths)
    assert captured["max_k"] == max(lengths)
    assert captured["kwargs"]["min_seqlen_q"] == 0
    assert captured["kwargs"]["window_size"] == (128, 0, 0)
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["sink_ptr"].dtype == torch.float32
    assert captured["kwargs"]["out"].shape == (sum(lengths), 16, 128)
    assert captured["kwargs"]["out"].stride(-2) == 192
    assert torch.all(output[..., :128] == 7.0)


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
        v = torch.zeros((512, 1, 128), dtype=torch.bfloat16)
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
    ).view(2, 256, 16, 192)

    assert captured["q"].shape == (2, 256, 16, 192)
    assert captured["k"].shape == (2, 256, 1, 192)
    assert captured["v"].shape == (2, 256, 1, 128)
    assert captured["out"].shape == (2, 256, 16, 128)
    assert captured["out"].stride(-2) == 192
    assert torch.all(output[..., :128] == 3.0)
    assert torch.all(output[..., 128:] == 0.0)
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
    ).view(512, 16, 192)

    assert captured["q"].shape == (512, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["cu_q"].tolist() == [0, 255, 512]
    assert captured["cu_k"].tolist() == [0, 255, 512]
    assert captured["max_q"] == 257
    assert captured["min_q"] == 255
    assert captured["out"].shape == (512, 16, 128)
    assert captured["out"].stride(-2) == 192
    assert torch.all(output[..., :128] == 5.0)
    assert torch.all(output[..., 128:] == 0.0)
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
            torch.zeros((seq_len, 1, 192), dtype=torch.bfloat16),
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
    ).view(q.shape[0], 16, 192)

    assert torch.equal(captured["gather_slot_ids"], metadata.kv_indices)
    assert captured["q"].shape == (256, 16, 192)
    assert captured["k"].shape == (512, 1, 192)
    assert captured["v"].shape == (512, 1, 128)
    assert captured["v"].stride(-2) == 192
    assert captured["cu_q"].tolist() == [0, 256]
    assert captured["cu_k"].tolist() == [0, 512]
    assert captured["max_q"] == 256
    assert captured["max_k"] == 512
    assert captured["min_q"] == 256
    assert captured["causal"] is True
    assert captured["window_left"] == -1
    assert captured["window_right"] == -1
    assert captured["out"].shape == (256, 16, 128)
    assert captured["out"].stride(-2) == 192
    assert torch.all(output[..., :128] == 11.0)
    assert torch.all(output[..., 128:] == 0.0)


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
            (q.shape[0], q.shape[1], 192), dtype=torch.bfloat16, device=q.device
        )

    def fake_flydsl_prefill(q, k, v, *args, **kwargs):
        captured.update(
            q=q, k=k, v=v, args=args, kwargs=kwargs, selected="flydsl"
        )
        return torch.zeros(
            (q.shape[0], q.shape[1], 192), dtype=torch.bfloat16, device=q.device
        )

    def fake_gather(k_buf, v_buf, slot_ids):
        captured["gather_slot_ids"] = slot_ids
        return (
            torch.zeros((1, 1, 192), dtype=torch.uint8),
            torch.zeros((1, 1, 192), dtype=torch.uint8),
        )

    monkeypatch.setattr(
        aiter_utils, "quantize_query_per_tensor_fp8", fake_quantize
    )
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", fake_batch_prefill)
    monkeypatch.setattr(
        aiter_utils, "launch_gather_shuffle_5d_to_linear", fake_gather
    )
    if flydsl_prefill:
        monkeypatch.setenv(aiter_utils.FLYDSL_MIMO_PREFILL_ENV, "1")
        monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
        monkeypatch.setattr(
            aiter_utils,
            "load_flydsl_mimo_prefill_kernel",
            lambda: SimpleNamespace(run=fake_flydsl_prefill),
        )

    k_buf = torch.zeros((2, 1, 12, 64, 16), dtype=torch.uint8)
    v_buf = torch.zeros((2, 1, 4, 192, 16), dtype=torch.uint8)
    pool = SimpleNamespace(
        dtype=fp8_dtype,
        store_dtype=torch.uint8,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    metadata = SimpleNamespace(
        swa_page_table=None,
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
        sliding_window_size=-1,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        v_head_dim=192,
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
    v = torch.zeros((1, 192), dtype=torch.bfloat16)
    sinks = None if direct_paged else torch.zeros(16, dtype=torch.float32)

    out = aiter_utils.forward_extend_vectorized_5d(
        backend,
        q,
        k,
        v,
        layer,
        forward_batch,
        bs0=2,
        window_size=(-1, -1),
        sinks=sinks,
    )

    assert out.shape == (1, 16 * 192)
    assert captured["q"].dtype == fp8_dtype
    assert captured["kwargs"]["q_descale"] is q_descale
    assert captured["kwargs"]["k_descale"] is k_descale
    assert captured["kwargs"]["v_descale"] is v_descale
    if direct_paged:
        assert "gather_slot_ids" not in captured
        assert captured["k"].data_ptr() == k_buf.data_ptr()
        assert captured["v"].data_ptr() == v_buf.data_ptr()
    else:
        assert torch.equal(captured["gather_slot_ids"], metadata.kv_indices)
    return captured


@pytest.mark.parametrize("direct_paged", [True, False])
def test_vectorized_prefill_forwards_independent_qkv_descales(
    monkeypatch, direct_paged
):
    _run_vectorized_prefill_scale_case(monkeypatch, direct_paged=direct_paged)


def test_qualified_long_mimo_prefill_selects_flydsl(monkeypatch):
    captured = _run_vectorized_prefill_scale_case(
        monkeypatch, direct_paged=True, flydsl_prefill=True
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
