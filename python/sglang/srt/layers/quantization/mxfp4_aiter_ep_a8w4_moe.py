"""MXFP4 expert backend for expert parallelism running the gfx942 EP8 A8W4 MoE.

``Fp8Config`` selects this for FP4-expert checkpoints under
``--moe-runner-backend aiter`` with ``--ep-size > 1``. Two token paths:

  * ``--moe-a2a-backend none``: every rank sees all tokens with *global* top-k
    ids and computes the contribution of its local experts
    ``[ep_rank * E, ep_rank * E + E)``; the MoE all-reduce sums the partials.
  * ``--moe-a2a-backend mori``: MORI dispatch delivers a capacity-sized receive
    buffer (only the first ``total_recv`` rows valid, a device count) with the
    tokens' global top-k ids. The activations arrive fp8 with one f32 scale per
    (token, 128) block (``SGLANG_MORI_DISPATCH_DTYPE=fp8``, the default this
    method requests) or bf16; the kernels read the valid row count on device,
    gemm1 consumes the blockwise scales directly, and the output (one row per
    received token, already weighted by its local routes) goes to MORI combine.

Kernels (``flydsl_moe_ep_w4a8_gfx942``, from tingqli/aiter ``tq/mimo-moe``
``doc-mimo`` section 8), 3 launches per call:

  * ``ep_prep``: local-expert routing sort + fp8 row quant of x + GEMM tile
    schedule (replaces ``moe_sorting`` + ``quantize_rows``);
  * ``ep_gemm1``: gate|up GEMM + SiLU, fp8 output with one scale per
    (row, 128 cols); also zeroes the output rows that gemm2 accumulates into;
  * ``ep_gemm2``: down GEMM, routed-weight bf16x2 atomic scatter (tokens with a
    single local route are stored directly).

The checkpoint's separated ``[gate; up]`` w13 rows are standard-preshuffled
(``shuffle_*_a16w4``) and then rewritten in place into the ``fast_dec=2`` form
consumed by both GEMMs (see ``ep8_moe.prepare_weights``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from torch.nn import Module

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
from sglang.srt.utils import get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import RoutingMethodType

# ``ep_prep`` encodes the token index in 24 bits and needs M < 65536; it also
# keeps all sort WGs resident (M * topk / 2048 <= CU count). Larger batches are
# run in chunks of this many tokens.
EP8_MAX_TOKENS = 32768


def _mori_token_bounds(capacity: int) -> Tuple[int, int]:
    """(rows to process, estimated valid rows) of a MORI receive buffer.

    Every rank dispatches its share of the tokens once per destination rank, so a
    rank receives at most the global token count of the forward pass: the DP
    gather buffer length under DP attention (exact incl. padding, also when
    captured in a CUDA graph). Without DP attention fall back to the capacity."""
    from sglang.srt.layers.dp_attention import (
        get_dp_global_num_tokens,
        get_global_dp_buffer_len,
        is_dp_attention_enabled,
    )

    if not is_dp_attention_enabled():
        return capacity, capacity
    try:
        bound = int(get_global_dp_buffer_len())
        est = int(sum(get_dp_global_num_tokens()))
    except (AttributeError, TypeError):
        return capacity, capacity
    bound = min(capacity, bound)
    return bound, max(1, min(bound, est))


class Mxfp4AiterEpA8W4MoEMethod:
    """MXFP4 (E8M0 scales) EP MoE: fp8 activations x MXFP4 weights (gfx942)."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        if with_bias:
            raise NotImplementedError(
                "Mxfp4AiterEpA8W4MoEMethod does not support MoE GEMM biases."
            )
        # The streamed-W K loop (prefetch 2, unrolled 2x) needs >= 8 K tiles of
        # 256 (or 128) fp8 bytes and a multiple of 4 tiles for both GEMMs.
        for name, k in (
            ("hidden_size", hidden_size),
            ("intermediate_size_per_partition", intermediate_size_per_partition),
        ):
            if k % 1024 or k < 2048:
                raise ValueError(
                    f"EP A8W4 MoE requires {name} to be a multiple of 1024 and "
                    f">= 2048, got {k}"
                )
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        if any(
            limit is not None and limit != 0
            for limit in (
                moe_runner_config.swiglu_limit,
                moe_runner_config.gemm1_clamp_limit,
            )
        ):
            raise ValueError(
                "EP A8W4 MoE does not support activation clamps; "
                "swiglu_limit and gemm1_clamp_limit must be unset or 0."
            )
        if moe_runner_config.activation != "silu" or not moe_runner_config.is_gated:
            raise ValueError("EP A8W4 MoE only supports gated SiLU experts.")
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError("EP A8W4 MoE does not support apply_router_weight_on_input.")
        a2a = get_moe_a2a_backend()
        if not (a2a.is_none() or a2a.is_mori()):
            raise ValueError(
                "EP A8W4 MoE supports --moe-a2a-backend none (global top-k ids on "
                f"every rank) or mori, got {a2a.value}."
            )
        if layer.num_fused_shared_experts:
            raise ValueError("EP A8W4 MoE does not support fused shared experts.")
        if layer.num_local_experts > 32:
            raise ValueError(
                "EP A8W4 MoE supports at most 32 local experts per rank, got "
                f"{layer.num_local_experts}."
            )
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.runner.runner_backend

    def get_moe_routing_method_type(
        self, layer: Module, requested_method: RoutingMethodType | None
    ) -> RoutingMethodType | None:
        return requested_method

    def process_weights_after_loading(self, layer: Module) -> None:
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        from sglang.srt.layers.quantization.flydsl_moe_ep_w4a8_gfx942.ep8_moe import (
            prepare_weights,
        )

        num_experts = layer.w13_weight.shape[0]
        fp4_dtype = torch.float4_e2m1fn_x2

        # Standard preshuffle; gate_up=False keeps the [gate; up] row order.
        w1 = shuffle_weight_a16w4(layer.w13_weight.data.view(fp4_dtype), 16, False)
        w2 = shuffle_weight_a16w4(layer.w2_weight.data.view(fp4_dtype), 16, False)
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data
        s1 = shuffle_scale_a16w4(
            w13_scale.view(-1, w13_scale.shape[-1]), num_experts, False
        )
        s2 = shuffle_scale_a16w4(
            w2_scale.view(-1, w2_scale.shape[-1]), num_experts, False
        )
        weights = prepare_weights(w1, w2, s1, s2)
        del w1, w2, s1, s2

        # Swap the layer storage to the rewritten bytes so the originals are freed.
        layer.w13_weight.data = weights["w1"].view(fp4_dtype)
        layer.w2_weight.data = weights["w2"].view(fp4_dtype)
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            weights["w1_scale"], requires_grad=False
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            weights["w2_scale"], requires_grad=False
        )
        for weight in (layer.w13_weight, layer.w2_weight):
            weight.is_shuffled = True
            weight.gate_up_interleave = False
        layer.ep8_weights = weights

        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        # MORI picks its dispatch dtype from the reported weight dtype: request
        # blockwise fp8 (consumed directly by gemm1) instead of the MXFP4 default
        # (SGLANG_MORI_DISPATCH_DTYPE still overrides; bf16 is also supported).
        dispatch_weight_dtype = (
            torch.float8_e4m3fnuz
            if get_moe_a2a_backend().is_mori()
            else layer.w13_weight.dtype
        )
        layer.dispatcher.set_quant_config(
            {"weight_dtype": dispatch_weight_dtype, "use_mxfp8": False}
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )
        from sglang.srt.layers.quantization.flydsl_moe_ep_w4a8_gfx942.ep8_moe import (
            ep8_moe,
        )

        if hasattr(dispatch_output, "origin_topk_ids"):  # MORI dispatch output
            return self._apply_mori(layer, dispatch_output)
        if getattr(dispatch_output, "hidden_states_scale", None) is not None:
            raise NotImplementedError(
                "EP A8W4 MoE quantizes bf16 activations itself; prequantized "
                "MoE input is not supported."
            )
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return StandardCombineInput(hidden_states=x)
        x = x.contiguous()
        rank = layer.moe_ep_rank
        W = layer.ep8_weights

        if num_tokens <= EP8_MAX_TOKENS:
            out = ep8_moe(x, W, topk_weights, topk_ids, rank=rank, adt="fp8")
        else:
            out = torch.empty_like(x)
            for s in range(0, num_tokens, EP8_MAX_TOKENS):
                e = min(s + EP8_MAX_TOKENS, num_tokens)
                out[s:e] = ep8_moe(
                    x[s:e], W, topk_weights[s:e], topk_ids[s:e], rank=rank, adt="fp8"
                )
        return StandardCombineInput(hidden_states=out)

    def _apply_mori(self, layer: Module, dispatch_output) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPNormalCombineInput,
        )
        from sglang.srt.layers.quantization.flydsl_moe_ep_w4a8_gfx942.ep8_moe import (
            ep8_moe_dyn,
        )

        x = dispatch_output.hidden_states
        x_scale = dispatch_output.hidden_states_scale
        topk_ids = dispatch_output.topk_ids
        topk_weights = dispatch_output.topk_weights
        # mori: [total_recv] int32 on device (no host sync)
        total_recv = dispatch_output.num_recv_tokens_per_expert
        if x.dtype == torch.float8_e4m3fnuz:
            if x_scale is None or x_scale.dtype != torch.float32:
                raise ValueError("EP A8W4 MoE: fp8 dispatch needs f32 scales.")
        elif x.dtype == torch.bfloat16:
            x_scale = None
        else:
            raise NotImplementedError(
                f"EP A8W4 MoE: unsupported MORI dispatch dtype {x.dtype}; use "
                "SGLANG_MORI_DISPATCH_DTYPE=fp8 (default) or bf16."
            )

        capacity = x.shape[0]
        mori_max = get_int_env_var("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0)
        if mori_max > 0:
            capacity = min(capacity, mori_max)
        rows, est = _mori_token_bounds(capacity)

        out = torch.empty((rows, x.shape[1]), dtype=torch.bfloat16, device=x.device)
        if rows > 0:
            topk_ids = topk_ids[:rows].to(torch.int32)
            topk_weights = topk_weights[:rows].to(torch.float32)
            for s in range(0, rows, EP8_MAX_TOKENS):
                e = min(s + EP8_MAX_TOKENS, rows)
                m_valid = (
                    total_recv
                    if s == 0
                    else (total_recv - s).clamp(min=0, max=e - s).to(torch.int32)
                )
                ep8_moe_dyn(
                    x[s:e],
                    x_scale[s:e] if x_scale is not None else None,
                    layer.ep8_weights,
                    topk_weights[s:e].contiguous(),
                    topk_ids[s:e].contiguous(),
                    m_valid,
                    rank=layer.moe_ep_rank,
                    est_tokens=max(1, min(e - s, est - s)),
                    out=out[s:e],
                )

        combine_cls = (
            MoriEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else MoriEPLLCombineInput
        )
        return combine_cls(
            hidden_states=out,
            topk_ids=dispatch_output.origin_topk_ids,
            topk_weights=dispatch_output.origin_topk_weights,
        )
