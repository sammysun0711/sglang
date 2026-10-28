from types import SimpleNamespace

import pytest
from sglang.srt.speculative.dflash_compact_config import (
    _validate_compact_target_profile,
    validate_compact_dflash_args,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def valid_args():
    return SimpleNamespace(
        speculative_dflash_compact_cache=True,
        speculative_algorithm="DFLASH",
        tp_size=8,
        pp_size=1,
        dp_size=1,
        attn_cp_size=1,
        page_size=64,
        attention_backend="aiter",
        speculative_draft_attention_backend="aiter",
        speculative_num_draft_tokens=8,
        speculative_draft_window_size=1024,
        kv_cache_dtype="bf16",
        speculative_draft_kv_cache_dtype="bf16",
        disable_radix_cache=True,
        disable_custom_all_reduce=True,
        max_running_requests=18,
        disaggregation_mode="null",
    )


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("SGLANG_AITER_KV_CACHE_LAYOUT", "vectorized_5d")
    monkeypatch.setenv("SGLANG_AITER_DFLASH_SWA_IMPL", "flydsl")
    monkeypatch.setenv("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")


def test_supported_scope_and_off_switch():
    validate_compact_dflash_args(valid_args())
    validate_compact_dflash_args(
        SimpleNamespace(speculative_dflash_compact_cache=False)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("speculative_algorithm", "EAGLE"),
        ("page_size", 1),
        ("tp_size", 4),
        ("pp_size", 2),
        ("dp_size", 2),
        ("attn_cp_size", 2),
        ("disable_radix_cache", False),
        ("disable_custom_all_reduce", False),
        ("enable_hierarchical_cache", True),
        ("enable_hisparse", True),
        ("enable_two_batch_overlap", True),
        ("disable_hybrid_swa_memory", True),
        ("enable_mixed_chunk", True),
        ("enable_prefill_cp", True),
        ("enable_prefill_context_parallel", True),
        ("enable_dsa_prefill_context_parallel", True),
        ("speculative_draft_window_size", 128),
        ("speculative_num_draft_tokens", 4),
        ("speculative_draft_kv_cache_dtype", "fp8_e4m3"),
        ("kv_cache_dtype", "fp8_e4m3"),
        ("speculative_draft_attention_backend", "triton"),
        ("max_running_requests", 0),
        ("disaggregation_mode", "decode"),
        ("disaggregation_mode", "prefill"),
    ],
)
def test_unqualified_configurations_fail(field, value):
    args = valid_args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match="Unsupported compact DFlash"):
        validate_compact_dflash_args(args)


@pytest.mark.parametrize(
    "name,value",
    [
        ("SGLANG_AITER_KV_CACHE_LAYOUT", "nhd"),
        ("SGLANG_AITER_DFLASH_SWA_IMPL", "gluon"),
        ("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "1"),
    ],
)
def test_unqualified_backends_fail(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="Unsupported compact DFlash"):
        validate_compact_dflash_args(valid_args())


@pytest.mark.parametrize("fp32_head", [False, True])
@pytest.mark.parametrize(
    "quantization,quant_config,is_fp8",
    [
        (None, {"quant_method": "fp8"}, True),
        ("fp8", {}, True),
        (None, {"quant_method": "fp8", "store_dtype": "mxfp4"}, False),
        ("fp8", {"quant_method": "fp8", "store_dtype": "mxfp4"}, False),
    ],
)
def test_target_profile_requires_fp32_head_only_for_fp8_weights(
    fp32_head, quantization, quant_config, is_fp8
):
    args = SimpleNamespace(enable_fp32_lm_head=fp32_head)
    target = SimpleNamespace(
        quantization=quantization,
        hf_config=SimpleNamespace(quantization_config=quant_config),
    )
    if is_fp8 and not fp32_head:
        with pytest.raises(ValueError, match="--enable-fp32-lm-head"):
            _validate_compact_target_profile(args, target)
    else:
        _validate_compact_target_profile(args, target)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
