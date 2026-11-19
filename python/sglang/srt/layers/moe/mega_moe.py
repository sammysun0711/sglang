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
"""Mega-MoE forward path and expert-weight prep shared by Deepseek V2/V4 (DeepGEMM)
and MiMo-V2 (pyhip's MOE_EP on ROCm)."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.dsv4 import mega_moe_pre_dispatch
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    get_dp_global_num_tokens,
)
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from deep_gemm import SymmBuffer

    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    from sglang.srt.models.mimo_v2 import MiMoV2MoE


_MEGA_MOE_SYMM_BUFFER: dict = {}
_MEGA_MOE_DG_ENV_APPLIED = False


def _apply_mega_moe_dg_env() -> None:
    """Forward sglang's FP4/MXF4 opt-in flags to DeepGEMM via env vars.

    DeepGEMM reads `DG_USE_FP4_ACTS` (and `DG_USE_MXF4_KIND`) at host-function
    call time — both `get_symm_buffer_for_mega_moe` and `fp8_fp4_mega_moe`.
    Forwarding once at first use is sufficient (these are static config
    flags, not per-request state) and matches the `setdefault` pattern so
    explicit `DG_USE_*` overrides from outside still win.
    """
    global _MEGA_MOE_DG_ENV_APPLIED
    if _MEGA_MOE_DG_ENV_APPLIED:
        return
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get():
        os.environ.setdefault("DG_USE_FP4_ACTS", "1")
    if envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND.get():
        os.environ.setdefault("DG_USE_MXF4_KIND", "1")
    _MEGA_MOE_DG_ENV_APPLIED = True


def _get_mega_moe_symm_buffer(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> SymmBuffer:
    import deep_gemm

    _apply_mega_moe_dg_env()

    key = (
        id(group),
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    buf = _MEGA_MOE_SYMM_BUFFER.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        _MEGA_MOE_SYMM_BUFFER[key] = buf
    return buf


def should_use_mega_moe(moe: DeepseekV2MoE, hidden_states: torch.Tensor) -> bool:
    if not get_moe_a2a_backend().is_megamoe():
        return False
    if not getattr(moe.experts, "_mega_moe_weights_built", False):
        return False
    if getattr(moe.experts, "mega_moe_comm", None) is not None:
        # pyhip MOE_EP: a collective with no other EP path; the tokens a rank
        # brings are bounded at load time (check_pyhip_mega_moe_config).
        return True
    if get_is_capture_mode():
        return True

    global_num_tokens = get_dp_global_num_tokens()
    if global_num_tokens:
        max_tokens_per_rank = max(global_num_tokens)
    else:
        max_tokens_per_rank = hidden_states.shape[0]
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap


def forward_mega_moe(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if getattr(moe.experts, "mega_moe_comm", None) is not None:
        return _forward_pyhip_mega_moe(moe, hidden_states, forward_batch)

    num_tokens = hidden_states.shape[0]

    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        current_stream = torch.cuda.current_stream()
        moe.alt_stream.wait_stream(current_stream)
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = torch.cuda.stream(moe.alt_stream)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)
        mega_stream_ctx = nullcontext()

    with mega_stream_ctx:
        y = _run_mega_routed(
            moe, hidden_states, forward_batch, input_ids_global, num_tokens
        )

    if sbo_overlap_flag:
        current_stream.wait_stream(moe.alt_stream)

    if shared_output is not None:
        y.add_(shared_output)
    return y


def _run_mega_routed(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    input_ids_global: Optional[torch.Tensor],
    num_tokens: int,
) -> torch.Tensor:
    import deep_gemm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    hidden_size = moe.config.hidden_size

    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = {"input_ids": input_ids_global} if moe.is_hash else {}
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=(
                forward_batch.num_token_non_padded
                if forward_batch is not None
                else None
            ),
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
            **topk_kwargs,
        )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
    else:
        topk_ids = None
        topk_weights = None

    ep_group = get_moe_ep_group().device_group
    num_experts = moe.experts.num_experts
    top_k = moe.config.num_experts_per_tok + moe.num_fused_shared_experts
    intermediate_size = moe.config.moe_intermediate_size
    num_max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    assert num_tokens <= num_max_tokens_per_rank, (
        f"mega MoE: num_tokens={num_tokens} exceeds cap "
        f"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
        f"{num_max_tokens_per_rank}; raise the env var or shrink "
        f"cuda_graph_max_bs / chunked_prefill_size accordingly"
    )

    buf = _get_mega_moe_symm_buffer(
        ep_group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=top_k,
        hidden=hidden_size,
        intermediate_hidden=intermediate_size,
    )

    if num_tokens > 0:
        topk_ids_in = topk_ids.to(torch.int32)
        topk_weights_in = topk_weights.to(torch.float32)
    else:
        topk_ids_in = hidden_states.new_empty((0, top_k), dtype=torch.int32)
        topk_weights_in = hidden_states.new_empty((0, top_k), dtype=torch.float32)

    use_fp4_acts = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS.get()
    if use_fp4_acts:
        # FP4 path goes through DeepGEMM's mega_moe_pre_dispatch which
        # handles the E2M1 packing variant. The jit implementation
        # only emits FP8.
        deep_gemm.mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            num_tokens=num_tokens,
            group_size=32,
            use_fp4_acts=True,
        )
    else:
        mega_moe_pre_dispatch(
            hidden_states,
            topk_ids_in,
            topk_weights_in,
            buf.x,
            buf.x_sf,
            buf.topk_idx,
            buf.topk_weights,
            quant_group_size=32,
        )

    # Allocate at least one row so y has a non-null CUDA data_ptr;
    # the DeepGEMM tvm-ffi binding rejects nullptr in convert_to_torch_tensor().
    y = torch.empty(
        (max(num_tokens, 1), hidden_size),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    swiglu_limit = getattr(moe.config, "swiglu_limit", None)
    deep_gemm.fp8_fp4_mega_moe(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=swiglu_limit,
        fast_math=True,
    )
    y = y[:num_tokens]

    if not moe.experts.should_fuse_routed_scaling_factor_in_topk:
        y.mul_(moe.routed_scaling_factor)
    return y


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # Build the interleaved L1 weight + scale once; share the weight buffer
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
        # the deep-ep path consumes the non-transposed interleaved scale and a
        # swizzle-aware activation kernel. L2 weight is untouched by the mega
        # transform, so the existing `w2_weight.data` is shared directly.
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)

        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
    else:
        l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))

        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_weights_built = True


# ---- pyhip MOE_EP (ROCm) ------------------------------------------------------------
# pyhip.contrib.flydsl.a8w4.moe_ep: the EP layer of MiMo-V2.6-Flash (EP8, 32 MXFP4
# experts per rank) as fused kernels -- dispatch, A8W4 experts, combine -- that talk
# through MORI shmem. Every rank calls MOE_EP once per layer, idle ranks with 0 rows,
# with the same ``tokens`` (the batch over all ranks), which picks the kernel family
# from pyhip's tuned table: ranks that pick differently would wait on each other.

# Tokens per rank and call the buffers hold: pyhip's tuned size, raised to a prefill
# chunk's share of one attention-TP rank, up to what the kernels can index.
PYHIP_MEGA_MOE_MIN_TOKENS_PER_RANK = 4096
PYHIP_MEGA_MOE_MAX_TOKENS_PER_RANK = 8192

# MOE_EP solver -> the EpComm method that builds its buffers.
_PYHIP_MEGA_MOE_FAMILIES = {
    "mega": "mega",
    "mega_small": "small",
    "mega_large": "large",
    "mori": "mori",
    "mori_eager": "mori",
}

# One EpComm per (EP group, shape), shared by every layer of that shape.
_PYHIP_MEGA_MOE_COMM: dict = {}


def _pyhip_mega_moe_problem(config: MoeRunnerConfig) -> dict:
    """MOE_EP's problem for a layer's shape (tokens, act, routing and x quant left to
    the defaults)."""
    from pyhip.contrib.flydsl.a8w4.moe_ep import MOE_EP

    from sglang.srt.distributed import get_moe_expert_parallel_world_size

    return dict(
        dict.fromkeys(MOE_EP.dim_names),
        model_dim=config.hidden_size,
        inter_dim=config.intermediate_size_per_partition,
        experts_global=config.num_experts,
        world=get_moe_expert_parallel_world_size(),
        topk=config.top_k,
    )


def _pyhip_mega_moe_max_tokens() -> int:
    """Tokens per rank and call: a prefill chunk is scattered over the attention-TP
    ranks of its DP group."""
    server_args = get_global_server_args()
    per_rank = 0
    if server_args.disaggregation_mode != "decode":
        chunk = server_args.chunked_prefill_size
        if chunk is None or chunk <= 0:
            raise ValueError(
                "--moe-a2a-backend megamoe (pyhip MOE_EP) needs chunked prefill: "
                "--chunked-prefill-size bounds the tokens a rank brings to a call."
            )
        per_rank = -(-chunk // get_attention_tp_size())
    if per_rank > PYHIP_MEGA_MOE_MAX_TOKENS_PER_RANK:
        raise ValueError(
            "--moe-a2a-backend megamoe (pyhip MOE_EP) takes at most "
            f"{PYHIP_MEGA_MOE_MAX_TOKENS_PER_RANK} tokens per rank and call, but a "
            f"prefill chunk brings {per_rank}; lower --chunked-prefill-size."
        )
    return max(PYHIP_MEGA_MOE_MIN_TOKENS_PER_RANK, per_rank)


def check_pyhip_mega_moe_config(config: MoeRunnerConfig) -> None:
    """Fail at load time on what MOE_EP cannot run."""
    from pyhip.contrib.flydsl.a8w4.moe_ep import MOE_EP

    server_args = get_global_server_args()
    name = "--moe-a2a-backend megamoe (pyhip MOE_EP)"
    if server_args.enable_two_batch_overlap or server_args.enable_single_batch_overlap:
        raise ValueError(
            f"{name} does not support two / single batch overlap: its output is a "
            "buffer that the next MoE call overwrites."
        )
    if envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get():
        raise ValueError(
            f"{name} needs the token counts the scheduler all-gathers; unset "
            "SGLANG_SCHEDULER_SKIP_ALL_GATHER."
        )
    if config.num_fused_shared_experts or config.apply_router_weight_on_input:
        raise ValueError(
            f"{name} does not support fused shared experts or "
            "apply_router_weight_on_input."
        )
    problem = _pyhip_mega_moe_problem(config)
    if not MOE_EP.points(problem):
        raise ValueError(
            f"{name} has no tuned config for {problem}; its kernels run EP8 with 32 "
            "experts per rank (MiMo-V2.6-Flash)."
        )
    _pyhip_mega_moe_max_tokens()


def _init_mori_shmem(group) -> None:
    """MORI shmem over ``group``, registered as sglang's MORI dispatcher does
    (``init_mori_op``): a group already registered keeps its shmem."""
    import mori

    try:
        torch._C._distributed_c10d._register_process_group("mori", group.cpu_group)
    except Exception as e:
        if "already registered" not in str(e):
            raise
    else:
        mori.shmem.shmem_torch_process_group_init("mori")


def build_pyhip_mega_moe(experts, weights: dict) -> None:
    """Run ``experts`` (``weights``: pyhip ``prepare_weights`` of its local experts)
    with MOE_EP. The first layer builds the shared EpComm, every rank in the same
    order: MORI shmem and the buffers of every solver family the tuned table can
    pick. Their constructors synchronize all ranks, which CUDA-graph capture does not
    allow, and built at load time their memory counts in sglang's profiling."""
    from pyhip.contrib.flydsl.a8w4.moe_ep import MOE_EP, EpComm

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    config = experts.moe_runner_config
    group = get_moe_ep_group()
    max_tokens = _pyhip_mega_moe_max_tokens()
    key = (
        id(group),
        max_tokens,
        config.num_experts,
        config.top_k,
        config.hidden_size,
        config.intermediate_size_per_partition,
    )
    comm = _PYHIP_MEGA_MOE_COMM.get(key)
    if comm is None:
        _init_mori_shmem(group)
        comm = EpComm(
            rank=group.rank_in_group, world=group.world_size, max_tokens=max_tokens
        )
        problem = _pyhip_mega_moe_problem(config)
        solvers = {
            MOE_EP.pick({**problem, "tokens": tokens}, capturing=capturing)["solver"]
            for tokens, _ in MOE_EP.points(problem)
            for capturing in (False, True)
        }
        # Sorted: symmetric buffers must be allocated in the same order on all ranks.
        for solver in sorted(solvers):
            getattr(comm, _PYHIP_MEGA_MOE_FAMILIES[solver])(weights, config.top_k)
        _PYHIP_MEGA_MOE_COMM[key] = comm

    experts.mega_moe_weights = weights
    experts.mega_moe_comm = comm
    experts._mega_moe_weights_built = True


def _forward_pyhip_mega_moe(
    moe: MiMoV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    from pyhip.contrib.flydsl.a8w4.moe_ep import MOE_EP

    if hidden_states.shape[0] > 0:
        router_logits = moe.gate(hidden_states)
        topk_output = moe.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=forward_batch.num_token_non_padded,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=moe.layer_id,
            ),
        )
    else:
        topk_output = moe.topk.empty_topk_output(hidden_states.device)

    # tokens: the DP ranks' token counts the scheduler all-gathers (a CUDA graph's
    # batch while it is captured), the same on every rank. The output is this rank's
    # tokens summed over all experts: a view of a buffer the next call overwrites.
    return MOE_EP(
        hidden_states,
        moe.experts.mega_moe_weights,
        topk_output.topk_weights,
        topk_output.topk_ids,
        comm=moe.experts.mega_moe_comm,
        tokens=sum(get_dp_global_num_tokens()),
    )
