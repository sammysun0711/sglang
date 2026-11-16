# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MiMo TP prefill SP eligibility, routing and expert execution.

Model layers retain ownership of their gate, experts and checkpoint parameters.
The helper runs the local token shard through the existing TP collective paths.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    get_dp_global_num_tokens,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import get_moe_runner_backend, should_use_dpa_sp_moe
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.mimo_v2_utils import mimo_flydsl_target_verify_enabled
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_gfx942_supported

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)
_MIMO_SP_MIN_TOKENS = 2048


def split_mimo_moe_inputs(hidden_states):
    """Split fused MiMo MLP input into router BF16 and FMoE activation."""
    if isinstance(hidden_states, tuple) and len(hidden_states) == 3:
        normalized, quantized, scale = hidden_states
        return normalized, (quantized, scale)
    return hidden_states, hidden_states


def _mimo_sp_moe_sizes() -> List[int]:
    """Token rows per TP rank under SGLANG_ENABLE_MIMO_SWA_SP: TP rank d * attn_tp + r holds the
    r-th of attn_tp equal slices of DP group d's (attn_tp-aligned) tokens."""
    attn_tp = get_attention_tp_size()
    return [n // attn_tp for n in get_dp_global_num_tokens() for _ in range(attn_tp)]


def _mimo_sp_moe_max_rows() -> int:
    """Rows per rank of the PyHIP fused reduce-scatter buffer: one prefill chunk's shard
    (chunked_prefill_size is per DP group), at most 8192 (1 GiB). Larger shards fall back.
    """
    chunk = get_global_server_args().chunked_prefill_size
    rows = -(-chunk // get_attention_tp_size()) if chunk and chunk > 0 else 8192
    return min(8192, -(-rows // 1024) * 1024)


@functools.lru_cache(maxsize=1)
def _log_mimo_tp_sp():
    logger.info("Using MiMo TP prefill sequence parallel (TP8/DP2/EP1, static FP8)")


@functools.lru_cache(maxsize=1)
def _log_mimo_prefill_pyhip_moe():
    logger.info("Using PR40 PyHIP FP8 prefill MoE with fused reduce-scatter")


def mimo_tp_sp_active(forward_batch: Optional[ForwardBatch]) -> bool:
    """Use the SGLANG_ENABLE_MIMO_SWA_SP layout for this forward: the same decision on every rank
    and layer (global token count); small passes and graph capture keep the all-reduce path.
    """
    if (
        forward_batch is None
        or not forward_batch.all_extend_in_batch
        or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.IDLE)
        or forward_batch.can_run_tbo
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    sizes = get_dp_global_num_tokens()
    active = (
        sizes is not None
        and len(sizes) == 2
        and all(n >= 0 and n % 4 == 0 for n in sizes)
        and sum(sizes) >= _MIMO_SP_MIN_TOKENS
    )
    if active:
        _log_mimo_tp_sp()
    return active


def _mimo_sp_moe_quant(hidden_states: torch.Tensor):
    """FP8 per-1x128 quantization with AITER's column-major scale storage: the same
    kernel aiter.fused_moe applies to a BF16 input, so gathering it is lossless."""
    from aiter.ops.quant import per_group_quant_hip

    from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

    return per_group_quant_hip(
        hidden_states, quant_dtype=fp8_dtype, group_size=128, transpose_scale=True
    )


def _mimo_sp_moe_gather_scales(scale: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    """Per-rank column-major [rows, groups] scale blocks -> one column-major [N, groups]."""
    groups = scale.shape[1]
    total = sum(sizes)
    if total == 0:
        return scale
    if all(size == sizes[0] for size in sizes):
        merged = (
            scale.view(len(sizes), groups, sizes[0])
            .transpose(0, 1)
            .reshape(groups, total)
        )
    else:
        blocks, offset = [], 0
        for size in sizes:
            blocks.append(scale[offset : offset + size].reshape(groups, size))
            offset += size
        merged = torch.cat(blocks, dim=1)
    merged = merged.view(total, groups)
    merged._aiter_moe_scale_is_transposed = True
    return merged


def mimo_tp_sp_enabled() -> bool:
    """Shared, construction-time eligibility for the TP MoE and attention layouts."""
    if not should_use_dpa_sp_moe():
        return False
    args = get_global_server_args()
    return (
        is_gfx942_supported()
        and envs.SGLANG_MXFP4_TO_FP8_128BLOCK.get()
        and envs.SGLANG_FLYPA_MIMO_PREFILL.get()
        and mimo_flydsl_target_verify_enabled()
        and (get_moe_runner_backend().is_aiter() or get_moe_runner_backend().is_auto())
        and args.attention_backend == "aiter"
        and args.prefill_attention_backend in (None, "aiter")
        and envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower() == "vectorized_5d"
        and args.page_size == 64
        and args.kv_cache_dtype == "bf16"
        and args.attn_cp_size == 1
        and args.pp_size == 1
        and args.moe_dense_tp_size == 1
        and not args.enable_two_batch_overlap
    )


def run_mimo_tp_sp_moe(layer, hidden_states) -> torch.Tensor:
    """TP MoE on attention-TP token shards (SGLANG_ENABLE_MIMO_SWA_SP).

    Route the local shard, all-gather its FP8 activations, 1x128 scales and top-k over
    the TP group, run the TP-sharded experts on all tokens, then reduce-scatter the
    partial sums back to the shard. Same quantized expert input as the all-reduce path.
    """
    router_hidden_states, expert_hidden_states = split_mimo_moe_inputs(hidden_states)
    if router_hidden_states.shape[0] > 0:
        router_logits = layer.gate(router_hidden_states)
        topk_output = layer.topk(router_hidden_states, router_logits)
    else:
        topk_output = layer.topk.empty_topk_output(router_hidden_states.device)
    if isinstance(expert_hidden_states, tuple):
        quantized, scale = expert_hidden_states
        assert getattr(scale, "_aiter_moe_scale_is_transposed", False)
    elif expert_hidden_states.shape[0] == 0:
        # Idle shard: no quant launch (an empty grid is an invalid configuration).
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        rows, hidden = expert_hidden_states.shape
        quantized = expert_hidden_states.new_empty((rows, hidden), dtype=fp8_dtype)
        scale = expert_hidden_states.new_empty(
            (rows, hidden // 128), dtype=torch.float32
        )
    else:
        quantized, scale = _mimo_sp_moe_quant(expert_hidden_states)
    sizes = _mimo_sp_moe_sizes()
    if sum(sizes) == 0:
        return router_hidden_states
    tp_group = get_tp_group()
    topk_weights, topk_ids, quantized_bytes, scale = tp_group.all_gatherv(
        [
            topk_output.topk_weights.to(torch.float32),
            topk_output.topk_ids.to(torch.int32),
            quantized.view(torch.uint8),
            scale,
        ],
        sizes=sizes,
    )
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=topk_weights.new_empty((0,)),
    )
    expert_input = (
        quantized_bytes.view(quantized.dtype),
        _mimo_sp_moe_gather_scales(scale, sizes),
    )
    if _use_pyhip_moe(layer, sizes):
        _log_mimo_prefill_pyhip_moe()
        from sglang.srt.layers.moe import pyhip_fp8_blockscale_moe

        x8, x_scale = expert_input
        tokens, hidden = x8.shape
        return pyhip_fp8_blockscale_moe.moe_reduce_scatter(
            layer.experts,
            x8,
            torch.as_strided(x_scale, (hidden // 128, tokens), (tokens, 1)),
            topk_weights,
            topk_ids,
            sizes,
            tp_group,
            _mimo_sp_moe_max_rows(),
        )
    final_hidden_states = layer.experts(expert_input, topk_output)
    if all(size == sizes[0] for size in sizes):
        output = final_hidden_states.new_empty((sizes[0], final_hidden_states.shape[1]))
        tp_group.reduce_scatter_tensor(output, final_hidden_states)
        return output
    return tp_group.reduce_scatterv(final_hidden_states, sizes=sizes)


def _use_pyhip_moe(layer, sizes: List[int]) -> bool:
    """Select the supported PyHIP TP MoE path after weight post-processing."""
    # The fused reduction maps every TP rank's buffer through HIP IPC.
    # A TP group spanning nodes must retain the AITER/RCCL fallback.
    if get_global_server_args().nnodes != 1:
        return False
    from sglang.srt.layers.moe import pyhip_fp8_blockscale_moe

    if layer._pyhip_moe is None:
        layer._pyhip_moe = pyhip_fp8_blockscale_moe.supports_layer(
            layer.experts, get_tp_group().world_size
        )
    if not layer._pyhip_moe:
        return False

    return pyhip_fp8_blockscale_moe.fits(
        sizes, layer.experts.w13_weight.shape[0], _mimo_sp_moe_max_rows()
    )
