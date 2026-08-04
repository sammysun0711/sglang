from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import aiter_utils
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype


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
