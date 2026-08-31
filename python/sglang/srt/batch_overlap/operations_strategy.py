from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.batch_overlap import operations
from sglang.srt.batch_overlap.operations import Operation
from sglang.srt.environ import envs
from sglang.srt.layers.moe import get_deepep_mode, get_moe_a2a_backend
from sglang.srt.layers.moe.token_dispatcher import DeepEPConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@dataclass
class OperationsStrategy:
    operations: List[Operation]
    deep_gemm_num_sms: Optional[int] = None
    tbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            deep_gemm_num_sms=_assert_all_same(
                [item.deep_gemm_num_sms for item in items]
            ),
            tbo_delta_stages=_assert_all_same(
                [item.tbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: ForwardMode,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        if layer_name == "DeepseekV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "Qwen3MoeDecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_qwen3_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "MiMoV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_mimov2_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# -------------------------------- Strategy for DeepSeek ---------------------------------------


# TODO can refactor to make it more fancy if we have more complex strategies
def _compute_moe_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "dense layer TBO not yet implemented"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_deepseek_blog_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_deepseek_blog_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_deepseek_blog_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = None
    if not _is_hip:
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_deepseek_blog_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            layer.mlp.op_shared_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            operations.YieldOperation(),
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


# -------------------------------- Strategy for Qwen3 ---------------------------------------


# TODO: unstable, current strategy is almost the same as DeepSeek, keep redundant code here for
# convenience to adjust strategy
def _compute_moe_qwen3_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "qwen3 moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_qwen3_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_qwen3_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_qwen3_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = None
    if not _is_hip:
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_qwen3_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )


# -------------------------------- Strategy for MiMoV2DecoderLayer ---------------------------------------


# TODO: unstable; current strategy matches DeepSeek for the common operations (MiMoV2 has no op_shared_experts),
# so we keep this redundant code here for convenience when adjusting the strategy
def _compute_moe_mimov2_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "MiMoV2DecoderLayer moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        if get_moe_a2a_backend().is_none():
            return _compute_moe_mimov2_no_ep_prefill(layer)
        return _compute_moe_mimov2_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_mimov2_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_mimov2_no_ep_prefill(layer):
    """Overlap TP all-reduces with the other child's MHA/MoE computation."""

    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_tbo_no_ep_attn_all_reduce_launch,
            operations.YieldOperation(),
            layer.op_tbo_no_ep_attn_all_reduce_wait_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_tbo_no_ep_experts,
            layer.mlp.op_tbo_no_ep_all_reduce_launch,
            operations.YieldOperation(),
            layer.mlp.op_tbo_no_ep_all_reduce_wait,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_mimov2_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = None
    if not _is_hip:
        # DeepEPConfig reads CUDA-only state from the deep_ep package. ROCm
        # dispatcher backends (including MORI) do not reserve DeepGEMM SMs.
        deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms
    enable_attn_comm_tbo = envs.SGLANG_MIMO_TBO_ATTN_COMM.get()

    if enable_attn_comm_tbo:
        if not _is_hip:
            raise RuntimeError("MiMo attention-communication TBO requires ROCm")
        if not get_moe_a2a_backend().is_mori():
            raise RuntimeError(
                "MiMo attention-communication TBO requires the MORI A2A backend"
            )
        if not get_deepep_mode().resolve(is_extend_in_batch=True).is_normal():
            raise RuntimeError(
                "MiMo attention-communication TBO currently supports MORI normal mode only"
            )
        if not layer.layer_communicator.supports_tbo_attn_communication():
            raise RuntimeError(
                "MiMo attention-communication TBO requires either a no-op or TP "
                "all-gather before attention, plus TP reduce-scatter before MoE "
                "on the same group"
            )
        prepare_attn_operations = (
            [
                layer.op_comm_prepare_attn_a,
                operations.YieldOperation(),
                layer.op_comm_prepare_attn_b,
            ]
            if layer.layer_communicator.has_tbo_attn_all_gather()
            else [layer.op_comm_prepare_attn]
        )
        return OperationsStrategy(
            deep_gemm_num_sms=deep_gemm_num_sms,
            tbo_delta_stages=0,
            operations=[
                *prepare_attn_operations,
                layer.self_attn.op_prepare,
                layer.self_attn.op_core,
                layer.op_comm_prepare_mlp_a,
                operations.YieldOperation(),
                layer.op_comm_prepare_mlp_b,
                layer.mlp.op_gate,
                layer.mlp.op_select_experts,
                layer.mlp.op_dispatch_a,
                operations.YieldOperation(),
                layer.mlp.op_dispatch_b,
                layer.mlp.op_experts,
                layer.mlp.op_combine_a,
                operations.YieldOperation(),
                layer.mlp.op_combine_b,
                layer.mlp.op_output,
                layer.op_comm_postprocess_layer,
            ],
        )

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_mimov2_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )
