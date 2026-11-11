"""MXFP4 expert backend running fp8-activation x MXFP4-weight (A8W4) MoE on gfx942.

``Fp8Config`` selects this for FP4-expert checkpoints under
``--moe-runner-backend aiter``. The kernels are the FlyDSL A8W4 MoE of
``pyhip.contrib.flydsl.a8w4`` (one package for TP and EP; per batch size it runs the
fastest kernel of its tuned table). One method serves every layout:

  * TP (``--ep-size 1``): each rank holds all experts and a slice of inter_dim;
  * EP, ``--moe-a2a-backend none``: every rank sees all tokens with *global* top-k
    ids and computes its local experts ``[ep_rank * E, ep_rank * E + E)``;
  * EP, ``--moe-a2a-backend mori``: the capacity-sized MORI receive buffer (only the
    first ``total_recv`` rows valid, a device count; fp8 with one f32 scale per
    (token, 128) block, or bf16) goes to ``a8w4_moe_dyn`` (no host sync, CUDA-graph
    safe) and its output to MORI combine.

The output is this rank's partial sum; the MoE all-reduce / MORI combine reduces
it. The checkpoint's separated ``[gate; up]`` w13 rows are preshuffled
(``shuffle_mxfp4``) and rewritten by ``prepare_weights`` (fast fp4 decode form
shared by all kernels), replacing the layer's weight storage.

Activations (every kernel family): ``silu``, or ``situv2``
(``beta * tanh(g / beta) * sigmoid(g) * lb * tanh(u / lb)`` with ``situ_beta`` /
``situ_linear_beta`` from the runner config and an optional clamp ``g <= L``,
``|u| <= L`` from ``swiglu_limit`` / ``gemm1_clamp_limit``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from torch.nn import Module

from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
from sglang.srt.layers.moe.utils import get_moe_a2a_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import RoutingMethodType

# The EP prep kernel of ``a8w4_moe_dyn`` takes < 65536 rows per call (24-bit token
# ids, all sort workgroups resident); larger MORI receive buffers run in chunks.
A8W4_DYN_MAX_TOKENS = 32768


def _expert_layout(layer: Module) -> Tuple[int, int]:
    """(first global expert id of this rank, global expert count). TP: (0, E)."""
    return layer.moe_ep_rank * layer.num_local_experts, layer.num_experts


def _check_kernel_shape(shape: Tuple[int, int, int, int, int]) -> None:
    """Fail at load time on (hidden, intermediate per partition, local experts,
    experts, top-k) the A8W4 kernels cannot run."""
    from pyhip.contrib.flydsl.a8w4 import dyn_config, supported

    if get_moe_a2a_backend().is_mori():
        dyn_config(shape, 1)  # raises: the MORI path needs the EP kernels
    elif not supported(dict(kind="tp2k", block_m=64, g1_bcm=1, g2="base"), shape):
        raise ValueError(
            "A8W4 MoE needs hidden_size % 512 == 0 and intermediate size per "
            f"partition % 256 == 0, got (hidden, inter, E_local, E, topk) = {shape}."
        )


def _mori_token_bounds(capacity: int, total_recv: torch.Tensor) -> Tuple[int, int]:
    """Return (rows to process, estimated valid rows) for a MORI receive buffer.

    Eager mode reads the device count (one device-to-host synchronization per MoE
    layer). Capture processes the full capacity because the device count cannot be
    read on the host; the kernels read it and ignore the invalid suffix.

    During capture, DP token metadata only supplies a performance estimate for
    config / grid selection, not a received-row upper bound. When only local DP
    metadata is available, extrapolate assuming equal token counts across DPs.
    Other DPs may replay different batch sizes, so this estimate can be low; the
    GEMM workgroups stride over the actual tiles to preserve correctness.
    """
    if total_recv.numel() == 0:
        return 0, 1
    if not torch.cuda.is_current_stream_capturing():
        n = min(capacity, int(total_recv.reshape(-1)[0].item()))
        return n, max(1, n)

    from sglang.srt.layers.dp_attention import (
        get_attention_dp_size,
        get_dp_global_num_tokens,
        is_dp_attention_enabled,
    )

    if not is_dp_attention_enabled():
        return capacity, capacity
    try:
        gnt = [int(n) for n in get_dp_global_num_tokens()]
    except (AttributeError, TypeError):
        return capacity, capacity
    if not gnt:
        return capacity, capacity
    dp_size = get_attention_dp_size()
    est = sum(gnt) if len(gnt) == dp_size else dp_size * max(gnt)
    return capacity, max(1, min(capacity, est))


class Mxfp4AiterA16W4MoEMethod:
    """MXFP4 (E8M0 scales) MoE on gfx942: fp8 activations x MXFP4 weights."""

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
                "Mxfp4AiterA16W4MoEMethod does not support MoE GEMM biases."
            )
        _check_kernel_shape(
            (
                hidden_size,
                intermediate_size_per_partition,
                num_experts,
                layer.num_experts,
                layer.top_k,
            )
        )
        # shuffle_scale_a16w4 passes fp32 scales through untouched, so keep E8M0.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            fp4_scale_dtype=torch.float8_e8m0fnu,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        cfg = moe_runner_config
        # Either spelling of the gate/up clamp limit (gpt-oss: gemm1_clamp_limit).
        clamp = cfg.gemm1_clamp_limit or cfg.swiglu_limit
        if cfg.activation not in ("silu", "situv2") or not cfg.is_gated:
            raise ValueError(
                "A8W4 MoE only supports gated silu or situv2 experts, got "
                f"activation={cfg.activation!r}, is_gated={cfg.is_gated}."
            )
        if cfg.activation == "silu":
            # The A8W4 SiLU epilogue does not apply an activation clamp. Reject
            # either config spelling before runner setup rather than silently
            # changing the model.
            if clamp:
                raise ValueError(
                    "Native MXFP4 A16W4/A8W4 does not support activation clamps; "
                    "swiglu_limit and gemm1_clamp_limit must be unset or 0."
                )
            act_kwargs = dict(act="silu")
        else:
            beta, linear_beta = cfg.situ_beta, cfg.situ_linear_beta
            limit = float(clamp) if clamp else float("inf")  # unset / 0: no clamp
            if not (
                beta is not None
                and beta > 0
                and linear_beta is not None
                and linear_beta > 0
                and limit > 0
            ):
                raise ValueError(
                    "A8W4 situv2 needs situ_beta > 0, situ_linear_beta > 0 and a "
                    f"clamp limit that is unset or > 0, got {beta}, {linear_beta}, "
                    f"{clamp}."
                )
            situ = dict(
                situ_beta=float(beta),
                situ_linear_beta=float(linear_beta),
                swiglu_limit=limit,
            )
            act_kwargs = dict(act="situv2", situ=situ)
        if cfg.apply_router_weight_on_input:
            raise ValueError("A8W4 MoE does not support apply_router_weight_on_input.")
        if cfg.num_fused_shared_experts:
            raise ValueError("A8W4 MoE does not support fused shared experts.")
        a2a = get_moe_a2a_backend()
        if not (a2a.is_none() or a2a.is_mori()):
            raise ValueError(
                "A8W4 MoE supports --moe-a2a-backend none (global top-k ids on every "
                f"rank) or mori, got {a2a.value}."
            )
        self._act_kwargs = act_kwargs
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.runner.runner_backend

    def get_moe_routing_method_type(
        self, layer: Module, requested_method: RoutingMethodType | None
    ) -> RoutingMethodType | None:
        return requested_method

    def process_weights_after_loading(self, layer: Module) -> None:
        from pyhip.contrib.flydsl.a8w4 import prepare_weights, shuffle_mxfp4

        fp4_dtype = torch.float4_e2m1fn_x2
        # aiter's a16w4 preshuffle (keeps the [gate; up] row order), then the fast
        # fp4 decode form used by every A8W4 kernel (bit-exact): nibbles reordered,
        # per-group scale deltas vs the channel max, W2's channel max codes
        # appended to its scales (``w2_exp`` is a view of them).
        weights = prepare_weights(
            *shuffle_mxfp4(
                layer.w13_weight.data.view(fp4_dtype),
                layer.w13_weight_scale_inv.data,
                layer.w2_weight.data.view(fp4_dtype),
                layer.w2_weight_scale_inv.data,
            )
        )

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
        layer.a8w4_weights = weights

        # MORI picks its dispatch dtype from the reported weight dtype: fp8 makes it
        # dispatch blockwise fp8, consumed directly by the kernels, instead of the
        # MXFP4 default (SGLANG_MORI_DISPATCH_DTYPE still overrides; bf16 also works).
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
        if hasattr(dispatch_output, "origin_topk_ids"):  # MORI dispatch output
            return self._apply_mori(layer, dispatch_output)

        from pyhip.contrib.flydsl.a8w4 import a8w4_moe

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        if getattr(dispatch_output, "hidden_states_scale", None) is not None:
            raise NotImplementedError(
                "A8W4 MoE quantizes bf16 activations itself; prequantized MoE "
                "input is not supported."
            )
        x = dispatch_output.hidden_states
        if x.shape[0] == 0:
            return StandardCombineInput(hidden_states=x)
        topk_output = dispatch_output.topk_output
        # TP: all experts local (offset 0); EP: global ids, this rank's experts.
        expert_offset, num_experts = _expert_layout(layer)
        out = a8w4_moe(
            x.contiguous(),
            layer.a8w4_weights,
            topk_output.topk_weights.to(torch.float32).contiguous(),
            topk_output.topk_ids.to(torch.int32).contiguous(),
            expert_offset=expert_offset,
            num_experts=num_experts,
            **self._act_kwargs,
        )
        return StandardCombineInput(hidden_states=out)

    def _apply_mori(self, layer: Module, dispatch_output) -> CombineInput:
        from pyhip.contrib.flydsl.a8w4 import a8w4_moe_dyn

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPNormalCombineInput,
        )

        x = dispatch_output.hidden_states
        x_scale = dispatch_output.hidden_states_scale
        # [1] int32 on device: the valid rows of the receive buffer
        total_recv = dispatch_output.num_recv_tokens_per_expert
        if x.dtype == torch.float8_e4m3fnuz:
            if x_scale is None or x_scale.dtype != torch.float32:
                raise ValueError("A8W4 MoE: fp8 MORI dispatch needs f32 scales.")
        elif x.dtype == torch.bfloat16:
            x_scale = None
        else:
            raise NotImplementedError(
                f"A8W4 MoE: unsupported MORI dispatch dtype {x.dtype}; use "
                "SGLANG_MORI_DISPATCH_DTYPE=fp8 (this method's default) or bf16."
            )

        capacity = x.shape[0]
        mori_max = envs.SGLANG_MORI_MOE_MAX_INPUT_TOKENS.get()
        rows, est = _mori_token_bounds(
            min(capacity, mori_max) if mori_max > 0 else capacity, total_recv
        )
        # The output keeps the receive-buffer capacity: MORI combine reads rows
        # [0, total_recv). Only the first ``rows`` are computed.
        out = torch.empty((capacity, x.shape[1]), dtype=torch.bfloat16, device=x.device)
        expert_offset, num_experts = _expert_layout(layer)
        topk_ids = dispatch_output.topk_ids[:rows].to(torch.int32)
        topk_weights = dispatch_output.topk_weights[:rows].to(torch.float32)
        for s in range(0, rows, A8W4_DYN_MAX_TOKENS):
            e = min(s + A8W4_DYN_MAX_TOKENS, rows)
            m_valid = (
                total_recv
                if s == 0
                else (total_recv - s).clamp(min=0, max=e - s).to(torch.int32)
            )
            a8w4_moe_dyn(
                x[s:e],
                layer.a8w4_weights,
                topk_weights[s:e].contiguous(),
                topk_ids[s:e].contiguous(),
                m_valid,
                x_scale=x_scale[s:e] if x_scale is not None else None,
                expert_offset=expert_offset,
                num_experts=num_experts,
                est_tokens=max(1, min(e - s, est - s)),
                out=out[s:e],
                **self._act_kwargs,
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
