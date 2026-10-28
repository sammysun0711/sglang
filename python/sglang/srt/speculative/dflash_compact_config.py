"""Fail-closed capability checks for the initial MiMo compact-cache path."""

import os

from sglang.srt.speculative.dflash_compact_layout import (
    CompactDFlashLayout,
    draft_owner_count,
)


def compact_dflash_transfer_pool(args, draft_pool):
    """Prevent shared target transfer indices from reaching fixed draft banks.

    Fake decode has no draft payload: admission initializes its visible zero
    suffix locally. Real transfer needs a distinct draft protocol and remains
    rejected even if a caller bypasses startup capability checks.
    """
    if not getattr(args, "speculative_dflash_compact_cache", False):
        return draft_pool
    if (
        args.disaggregation_mode != "decode"
        or args.disaggregation_transfer_backend != "fake"
    ):
        raise ValueError("compact DFlash does not support real draft KV transfer")
    return None


def validate_compact_dflash_args(args):
    if not args.speculative_dflash_compact_cache:
        return
    required = {
        "speculative_algorithm": "DFLASH",
        "tp_size": 8,
        "pp_size": 1,
        "dp_size": 1,
        "attn_cp_size": 1,
        "page_size": 64,
        "attention_backend": "aiter",
        "speculative_draft_attention_backend": "aiter",
        "speculative_num_draft_tokens": 8,
        "speculative_draft_window_size": 1024,
    }
    errors = [
        f"{key}={getattr(args, key, None)!r} (requires {value!r})"
        for key, value in required.items()
        if getattr(args, key, None) != value
    ]
    if args.kv_cache_dtype not in ("bf16", "bfloat16"):
        errors.append("target KV must be BF16")
    if args.speculative_draft_kv_cache_dtype not in (None, "bf16", "bfloat16"):
        errors.append("draft KV must be BF16")
    if not args.disable_radix_cache:
        errors.append("radix cache must be disabled")
    if not getattr(args, "disable_custom_all_reduce", False):
        errors.append(
            "custom all-reduce must be disabled until compact graph communication is qualified"
        )
    if args.max_running_requests is None or args.max_running_requests <= 0:
        errors.append("max-running-requests must be positive")
    for key in (
        "enable_hierarchical_cache",
        "enable_hisparse",
        "enable_dp_attention",
        "enable_two_batch_overlap",
        "enable_lora",
        "disable_hybrid_swa_memory",
        "enable_mixed_chunk",
        "enable_prefill_cp",
        "enable_prefill_context_parallel",
        "enable_dsa_prefill_context_parallel",
    ):
        if getattr(args, key, False):
            errors.append(f"{key} is not supported")
    disaggregation_mode = getattr(args, "disaggregation_mode", "null")
    if disaggregation_mode not in (None, "null") and not (
        disaggregation_mode == "decode"
        and getattr(args, "disaggregation_transfer_backend", None) == "fake"
    ):
        errors.append("compact DFlash only supports fake decode disaggregation")
    if os.environ.get("SGLANG_AITER_KV_CACHE_LAYOUT", "") != "vectorized_5d":
        errors.append("SGLANG_AITER_KV_CACHE_LAYOUT must be vectorized_5d")
    if os.environ.get("SGLANG_AITER_DFLASH_SWA_IMPL", "") != "flydsl":
        errors.append("SGLANG_AITER_DFLASH_SWA_IMPL must be flydsl")
    if os.environ.get("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0").lower() not in (
        "0",
        "false",
    ):
        errors.append("overlap plan stream is not supported")
    if errors:
        raise ValueError(
            "Unsupported compact DFlash configuration: " + "; ".join(errors)
        )


def _validate_compact_target_profile(args, target):
    quant_config = getattr(target.hf_config, "quantization_config", None) or {}
    quant_method = getattr(target, "quantization", None) or quant_config.get(
        "quant_method"
    )
    # MiMo's MXFP4 checkpoint also declares quant_method="fp8". Its storage
    # format distinguishes it from the FP8 target that needs the FP32 LM head.
    is_fp8_target = quant_method == "fp8" and quant_config.get("store_dtype") != "mxfp4"
    if is_fp8_target and not getattr(args, "enable_fp32_lm_head", False):
        raise ValueError(
            "compact DFlash with FP8 target weights requires --enable-fp32-lm-head; "
            "the BF16 LM-head profile is not qualified"
        )


def resolve_compact_dflash_layout(args, target, draft, *, decode_preallocated=0):
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
        get_dflash_layer_types,
    )

    if target.hf_config.architectures != ["MiMoV2ForCausalLM"]:
        raise ValueError("compact DFlash currently supports the MiMoV2 target only")
    _validate_compact_target_profile(args, target)
    if draft.hf_config.architectures != ["DFlashDraftModel"]:
        raise ValueError("compact DFlash requires DFlashDraftModel")
    if get_dflash_layer_types(draft.hf_config) != ["sliding_attention"] * 5:
        raise ValueError("compact DFlash requires five uniformly sliding draft layers")
    if get_dflash_attention_sliding_window_size(draft.hf_config) != 1023:
        raise ValueError(
            "compact DFlash requires checkpoint window 1024 (window_left=1023)"
        )
    geometry = (
        draft.get_num_kv_heads(8),
        draft.head_dim,
        draft.v_head_dim,
        draft.hf_config.num_hidden_layers,
    )
    if geometry != (1, 128, 128, 5):
        raise ValueError(f"unsupported compact draft geometry: {geometry}")
    owners = draft_owner_count(
        args.max_running_requests,
        decode=args.disaggregation_mode == "decode",
        preallocated_requests=decode_preallocated,
    )
    return (
        CompactDFlashLayout(owners),
        geometry[0] * (geometry[1] + geometry[2]) * geometry[3] * 2,
    )
