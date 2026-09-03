"""SHUFFLE 5D KV pool helpers for the AITER attention backend.

This module hosts the attention pathways that are specific to the
``SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d`` (SHUFFLE 5D) physical layout.
They live here rather than inline in
:mod:`sglang.srt.layers.attention.aiter_backend` so the main backend
file keeps focused on the legacy NHD path and on dispatch wiring.

Each entry point takes the :class:`AiterAttnBackend` instance as its
first argument so it can reach the shared per-step metadata
(``forward_metadata``, ``qo_indptr``, ``input_dtype``, …) without
needing to be a method on the class.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import torch

from .flydsl_pa import PagedAttention as flypa

try:
    # `mha_batch_prefill_func` is re-exported at the aiter top level via
    # `aiter/__init__.py` (`from .ops.mha import *`). Note: a bare
    # `from aiter.mha import ...` does NOT work — that module path only
    # exists as `aiter.ops.mha`.
    from aiter import (
        flash_attn_varlen_func,
        fmha_v3_fwd,
        fmha_v3_varlen_fwd,
        mha_batch_prefill_func,
    )
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )
except ImportError:  # pragma: no cover - import-time guard mirrors aiter_backend
    flash_attn_varlen_func = None
    fmha_v3_fwd = None
    fmha_v3_varlen_fwd = None
    mha_batch_prefill_func = None
    pa_decode_gluon = None
    get_recommended_splits = None

from sglang.srt.layers.attention.utils import launch_gather_shuffle_5d_to_linear
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, scaled_fp8_quant
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


FLYDSL_MIMO_QUERY_LENGTH = 4
FLYDSL_MIMO_QUERY_HEADS = 16
FLYDSL_MIMO_KV_HEADS = 1
FLYDSL_MIMO_HEAD_DIM = 192
FLYDSL_MIMO_VALUE_HEAD_DIM = 128
FLYDSL_MIMO_PAGE_SIZE = 64
FLYDSL_MIMO_DEFAULT_NUM_PARTITIONS = 8
FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS = (8, 16, 24, 32)
FLYDSL_MIMO_NUM_PARTITIONS_ENV = "SGLANG_FLYDSL_PA_NUM_PARTITIONS"
FLYDSL_MIMO_EQUIVALENT_GROUP_SIZE = 64
FLYDSL_MIMO_PREFILL_ENV = "SGLANG_FLYDSL_MIMO_PREFILL"
FLYDSL_MIMO_PREFILL_MIN_Q = 4096
FLYDSL_MIMO_PREFILL_MIN_KV = 8192
FLYPA_MIMO_PREFILL_ENV = "SGLANG_FLYPA_MIMO_PREFILL"

CK_MIMO_PREFILL_QUERY_HEADS = 16
CK_MIMO_PREFILL_KV_HEADS = 1
CK_MIMO_PREFILL_HEAD_DIM = 192
CK_MIMO_PREFILL_VALUE_HEAD_DIM = 128
CK_MIMO_PREFILL_PAGE_SIZE = 64
MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS = 8

MIMO_FRESH_BF16_ASM_ENV = "SGLANG_AITER_MIMO_FRESH_BF16_ASM"
MIMO_FRESH_BF16_ASM_ENABLED = get_bool_env_var(MIMO_FRESH_BF16_ASM_ENV, "false")
MIMO_FRESH_BF16_ASM_VARLEN_ENV = "SGLANG_AITER_MIMO_FRESH_BF16_ASM_VARLEN"
MIMO_FRESH_BF16_ASM_VARLEN_ENABLED = get_bool_env_var(
    MIMO_FRESH_BF16_ASM_VARLEN_ENV, "false"
)
MIMO_FRESH_BF16_ASM_V_HEAD_DIM = 128
MIMO_FRESH_BF16_SWA_VARLEN_ENV = "SGLANG_AITER_MIMO_FRESH_BF16_SWA_VARLEN"
MIMO_FRESH_BF16_SWA_VARLEN_ENABLED = get_bool_env_var(
    MIMO_FRESH_BF16_SWA_VARLEN_ENV, "false"
)
MIMO_FRESH_BF16_SWA_WINDOW_SIZE = 128


@lru_cache(maxsize=1)
def is_gfx950() -> bool:
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    return arch == "gfx950"


@lru_cache(maxsize=1)
def is_gfx942() -> bool:
    if not torch.version.hip or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    return arch == "gfx942"


def is_mimo_flypa_arch() -> bool:
    """FlyPA prefill is the gfx942 path; gfx950 is also accepted."""
    return is_gfx942() or is_gfx950()


def _is_scalar_f32_scale(scale) -> bool:
    return (
        isinstance(scale, torch.Tensor)
        and scale.dtype == torch.float32
        and scale.numel() == 1
    )


def _mimo_logical_qkv_views(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: ForwardBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return live-token Q/K/V views and the physical query row count.

    TBO pads token-major tensors to the attention-TP width while keeping
    ``extend_seq_lens_cpu`` logical. MiMo's ASM kernels consume only the live
    prefix; downstream TBO stages still require an output with the original
    physical row count.
    """
    lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    physical_tokens = q.shape[0]
    if lengths is None:
        return q, k, v, physical_tokens

    logical_tokens = sum(int(length) for length in lengths)
    if logical_tokens < 0:
        raise ValueError(f"MiMo extend token count must be non-negative: {lengths}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.shape[0] < logical_tokens:
            raise ValueError(
                f"MiMo {name} has {tensor.shape[0]} physical rows but metadata "
                f"requires {logical_tokens} live rows"
            )

    return (
        q[:logical_tokens],
        k[:logical_tokens],
        v[:logical_tokens],
        physical_tokens,
    )


def _allocate_mimo_asm_output(
    q: torch.Tensor,
    logical_tokens: int,
    output_tokens: int,
    num_heads: int,
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a physical TBO output and return its live-token prefix."""
    if output_tokens < logical_tokens:
        raise ValueError(
            f"MiMo ASM output has {output_tokens} rows for {logical_tokens} live tokens"
        )
    output = q.new_empty((output_tokens, num_heads, value_head_dim))
    if output_tokens > logical_tokens:
        output[logical_tokens:].zero_()
    return output, output[:logical_tokens]


def can_use_mimo_fresh_bf16_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
) -> bool:
    """Return whether this extend is the qualified gfx950 MiMo ASM contract."""
    lengths = forward_batch.extend_seq_lens_cpu
    uniform_length = (
        lengths is not None
        and len(lengths) > 0
        and lengths[0] > 128
        and all(length == lengths[0] for length in lengths)
    )
    total_tokens = 0 if not uniform_length else len(lengths) * lengths[0]
    return (
        MIMO_FRESH_BF16_ASM_ENABLED
        and fmha_v3_fwd is not None
        and is_gfx950()
        and uniform_length
        and q.shape[0] == total_tokens
        and k.shape[0] == total_tokens
        and v.shape[0] == total_tokens
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
        and tuple(window_size) == (-1, -1)
        and sinks is None
        and float(backend.logits_soft_cap) == 0.0
    )


def mimo_fresh_bf16_asm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    output_token_count: int | None = None,
) -> torch.Tensor:
    """Run gfx950 D192/V128 ASM with a native V128 model/cache ABI."""
    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = len(lengths)
    sequence_length = lengths[0]
    q_4d = q.view(
        batch_size,
        sequence_length,
        layer.tp_q_head_num,
        layer.qk_head_dim,
    )
    k_4d = k.view(
        batch_size,
        sequence_length,
        layer.tp_k_head_num,
        layer.qk_head_dim,
    )
    v_4d = v.view(
        batch_size,
        sequence_length,
        layer.tp_v_head_num,
        layer.v_head_dim,
    )

    logical_tokens = batch_size * sequence_length
    output_storage, output_live = _allocate_mimo_asm_output(
        q,
        logical_tokens,
        logical_tokens if output_token_count is None else output_token_count,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    output = output_live.view(
        batch_size,
        sequence_length,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    result = fmha_v3_fwd(
        q_4d,
        k_4d,
        v_4d,
        0.0,
        layer.scaling,
        True,
        -1,
        -1,
        False,
        False,
        0,
        output,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if result[0].data_ptr() != output.data_ptr():
        raise RuntimeError("gfx950 MiMo fresh BF16 ASM ignored its output buffer")
    return output_storage.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_fresh_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
) -> bool:
    """Return whether this fresh ragged extend matches MiMo varlen ASM."""
    lengths = forward_batch.extend_seq_lens_cpu
    valid_lengths = (
        lengths is not None
        and len(lengths) > 0
        and min(lengths) > 128
    )
    total_tokens = 0 if not valid_lengths else sum(lengths)
    return (
        MIMO_FRESH_BF16_ASM_ENABLED
        and MIMO_FRESH_BF16_ASM_VARLEN_ENABLED
        and fmha_v3_varlen_fwd is not None
        and is_gfx950()
        and valid_lengths
        and q.shape[0] == total_tokens
        and k.shape[0] == total_tokens
        and v.shape[0] == total_tokens
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and (layer.sliding_window_size is None or layer.sliding_window_size < 0)
        and tuple(window_size) == (-1, -1)
        and sinks is None
        and float(backend.logits_soft_cap) == 0.0
    )


def mimo_fresh_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    output_token_count: int | None = None,
) -> torch.Tensor:
    """Run gfx950's grouped D192/V128 ASM on a fresh ragged MiMo batch."""
    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    q_varlen = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k_varlen = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v_varlen = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    logical_tokens = q.shape[0]
    output_storage, output = _allocate_mimo_asm_output(
        q,
        logical_tokens,
        logical_tokens if output_token_count is None else output_token_count,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    cu_seqlens = backend.qo_indptr[: batch_size + 1]
    result = fmha_v3_varlen_fwd(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens,
        cu_seqlens,
        max_length,
        max_length,
        min_length,
        0.0,
        layer.scaling,
        0.0,
        False,
        True,
        -1,
        -1,
        False,
        False,
        0,
        output,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if result[0].data_ptr() != output.data_ptr():
        raise RuntimeError(
            "gfx950 MiMo fresh BF16 varlen ASM ignored its output buffer"
        )
    return output_storage.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_chunk_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
    is_swa_layer: bool,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> bool:
    """Return whether cached ragged prefill can reuse MiMo BF16 varlen ASM."""
    extend_lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefix_lengths = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    valid_ragged_batch = (
        extend_lengths is not None
        and prefix_lengths is not None
        and seq_lens_cpu is not None
        and len(extend_lengths) > 0
        and len(extend_lengths) == len(prefix_lengths) == len(seq_lens_cpu)
    )
    if not valid_ragged_batch:
        return False

    extend_lengths = [int(length) for length in extend_lengths]
    prefix_lengths = [int(length) for length in prefix_lengths]
    seq_lengths = [int(length) for length in seq_lens_cpu]
    total_q = sum(extend_lengths)
    total_kv = int(forward_batch.seq_lens_sum)
    batch_size = len(extend_lengths)
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
        CK_MIMO_PREFILL_PAGE_SIZE,
        MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        MIMO_BF16_KV_CACHE_INNER_PACK_ELEMS,
    )
    return (
        MIMO_FRESH_BF16_ASM_ENABLED
        and MIMO_FRESH_BF16_ASM_VARLEN_ENABLED
        and fmha_v3_varlen_fwd is not None
        and is_gfx950()
        and not is_swa_layer
        and sinks is None
        and tuple(window_size) == (-1, -1)
        and any(prefix_len > 0 for prefix_len in prefix_lengths)
        and all(prefix_len >= 0 for prefix_len in prefix_lengths)
        and all(extend_len > 0 for extend_len in extend_lengths)
        and max(extend_lengths) > 128
        and all(
            seq_len == prefix_len + extend_len
            for seq_len, prefix_len, extend_len in zip(
                seq_lengths, prefix_lengths, extend_lengths
            )
        )
        and total_kv == sum(seq_lengths)
        and q.shape[0] == total_q
        and k.shape[0] == total_q
        and v.shape[0] == total_q
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and backend.kv_cache_dtype != fp8_dtype
        and sub_pool.dtype == torch.bfloat16
        and sub_pool.store_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and getattr(layer, "mimo_original_v_head_dim", None)
        == MIMO_FRESH_BF16_ASM_V_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and float(backend.logits_soft_cap) == 0.0
        and metadata.kv_indices is not None
        and metadata.kv_indptr is not None
        and metadata.max_kv_len is not None
        and metadata.kv_indices.dtype == torch.int32
        and metadata.kv_indptr.dtype == torch.int32
        and backend.qo_indptr.dtype == torch.int32
        and metadata.kv_indices.device == q.device
        and metadata.kv_indptr.device == q.device
        and backend.qo_indptr.device == q.device
        and metadata.kv_indices.numel() >= total_kv
        and metadata.kv_indptr.numel() >= batch_size + 1
        and backend.qo_indptr.numel() >= batch_size + 1
        and int(metadata.max_q_len) == max(extend_lengths)
        and int(metadata.max_kv_len) == max(seq_lengths)
        and k_buf.ndim == 5
        and v_buf.ndim == 5
        and tuple(k_buf.shape[1:]) == expected_k_tail
        and tuple(v_buf.shape[1:]) == expected_v_tail
        and k_buf.shape[0] == v_buf.shape[0]
        and k_buf.element_size() == 2
        and v_buf.element_size() == 2
    )


def mimo_chunk_bf16_varlen_asm(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
    output_token_count: int | None = None,
) -> torch.Tensor:
    """Run gfx950 D192/V128 ASM for a cached ragged MiMo prefill batch."""
    extend_lengths = [int(length) for length in forward_batch.extend_seq_lens_cpu]
    seq_lengths = [int(length) for length in forward_batch.seq_lens_cpu]
    batch_size = len(extend_lengths)
    total_q = sum(extend_lengths)
    total_kv = sum(seq_lengths)
    slot_ids = metadata.kv_indices[:total_kv]
    k_full, v_full = launch_gather_shuffle_5d_to_linear(k_buf, v_buf, slot_ids)
    q_varlen = q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k_varlen = k_full.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v_varlen = v_full.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    output_storage, output = _allocate_mimo_asm_output(
        q,
        total_q,
        total_q if output_token_count is None else output_token_count,
        layer.tp_q_head_num,
        layer.v_head_dim,
    )
    result = fmha_v3_varlen_fwd(
        q_varlen,
        k_varlen,
        v_varlen,
        backend.qo_indptr[: batch_size + 1],
        metadata.kv_indptr[: batch_size + 1],
        max(extend_lengths),
        max(seq_lengths),
        min(extend_lengths),
        0.0,
        layer.scaling,
        0.0,
        False,
        True,
        -1,
        -1,
        False,
        False,
        0,
        output,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if result[0].data_ptr() != output.data_ptr():
        raise RuntimeError(
            "gfx950 MiMo chunk BF16 varlen ASM ignored its output buffer"
        )
    return output_storage.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_fresh_bf16_swa_varlen(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks,
) -> bool:
    """Return whether this fresh extend matches MiMo's gfx950 SWA contract."""
    lengths = forward_batch.extend_seq_lens_cpu
    valid_lengths = (
        lengths is not None
        and len(lengths) > 0
        and min(lengths) > MIMO_FRESH_BF16_SWA_WINDOW_SIZE
    )
    total_tokens = 0 if not valid_lengths else sum(lengths)
    valid_sinks = (
        isinstance(sinks, torch.Tensor)
        and sinks.device == q.device
        and sinks.shape == (CK_MIMO_PREFILL_QUERY_HEADS,)
        and sinks.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and sinks.stride(-1) == 1
    )
    return (
        MIMO_FRESH_BF16_SWA_VARLEN_ENABLED
        and flash_attn_varlen_func is not None
        and is_gfx950()
        and valid_lengths
        and q.shape[0] == total_tokens
        and k.shape[0] == total_tokens
        and v.shape[0] == total_tokens
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and backend.input_dtype == torch.bfloat16
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and getattr(layer, "mimo_original_v_head_dim", None)
        == MIMO_FRESH_BF16_ASM_V_HEAD_DIM
        and q.shape[-1] == layer.tp_q_head_num * layer.head_dim
        and k.shape[-2:] == (layer.tp_k_head_num, layer.qk_head_dim)
        and v.shape[-2:] == (layer.tp_v_head_num, layer.v_head_dim)
        and q.stride(-1) == 1
        and k.stride(-1) == 1
        and v.stride(-1) == 1
        and layer.sliding_window_size == MIMO_FRESH_BF16_SWA_WINDOW_SIZE
        and tuple(window_size)
        == (MIMO_FRESH_BF16_SWA_WINDOW_SIZE, -1)
        and valid_sinks
        and float(backend.logits_soft_cap) == 0.0
    )


def mimo_fresh_bf16_swa_varlen(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    window_size,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Run native-D128 CK varlen SWA with a native V128 ABI."""
    lengths = forward_batch.extend_seq_lens_cpu
    batch_size = len(lengths)
    q_varlen = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k_varlen = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    v_varlen = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

    output = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
    cu_seqlens = backend.qo_indptr[: batch_size + 1]
    sink_ptr = sinks if sinks.dtype == torch.float32 else sinks.float()
    result = flash_attn_varlen_func(
        q_varlen,
        k_varlen,
        v_varlen,
        cu_seqlens,
        cu_seqlens,
        max(lengths),
        max(lengths),
        # This is a cascade-attention exclusion threshold, not the shortest
        # sequence length.  Zero is required to execute every request.
        min_seqlen_q=0,
        softmax_scale=layer.scaling,
        causal=True,
        window_size=(int(window_size[0]), 0, 0),
        sink_ptr=sink_ptr,
        out=output,
    )
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("gfx950 MiMo fresh SWA varlen ignored its output buffer")
    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def quantize_query_per_tensor_fp8(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 query and return its independent FP8 descale.

    A raw ``q.to(fp8_dtype)`` has an implicit descale of one.  Reusing the KV
    cache's K descale for that tensor changes Q by an unrelated factor and is
    only hidden when every cache scale happens to be 1.0.  The batch-prefill
    FP8 ABI expects ``dequantized_q = q_fp8 * q_descale``, so use SGLang's
    dynamic per-tensor quantizer and forward the scale it actually produced.

    Prefill queries arrive as a two-dimensional token-major view whose outer
    stride may still reference the fused QKV allocation.  The quantization
    kernels consume a dense matrix, and the old dtype conversion also
    materialized one, so make that requirement explicit here.
    """
    if q.ndim != 2:
        raise ValueError(
            "AITER FP8 batch-prefill query quantization requires a 2D "
            f"[tokens, heads * head_dim] tensor; got shape {tuple(q.shape)}"
        )
    if not q.is_floating_point() or q.dtype == fp8_dtype:
        raise ValueError(
            "AITER FP8 batch-prefill query quantization requires a non-FP8 "
            f"floating-point query; got {q.dtype}"
        )

    q_fp8, q_descale = scaled_fp8_quant(
        q.contiguous(),
        scale=None,
        use_per_token_if_dynamic=False,
    )
    if q_descale.numel() != 1 or q_descale.dtype != torch.float32:
        raise RuntimeError(
            "AITER FP8 batch-prefill requires one FP32 Q descale; got "
            f"shape={tuple(q_descale.shape)}, dtype={q_descale.dtype}"
        )
    return q_fp8, q_descale


@dataclass(frozen=True)
class FlyDSLMiMoPrefillKernel:
    run: Callable
    version: str
    runtime_path: str
    kernel_path: str


@lru_cache(maxsize=1)
def load_flydsl_mimo_prefill_kernel() -> FlyDSLMiMoPrefillKernel:
    """Lazily load the optional gfx950 MiMo D192/V128 paged kernel."""

    try:
        flydsl = importlib.import_module("flydsl")
        module = importlib.import_module(
            "kernels.attention.flash_attn_fp8_mimo_paged_gfx950"
        )
    except Exception as exc:
        raise RuntimeError(
            f"{FLYDSL_MIMO_PREFILL_ENV}=1 requires the compatible FlyDSL "
            "runtime and the local FlyDSL repository-root `kernels` package "
            "on PYTHONPATH"
        ) from exc

    kernel = FlyDSLMiMoPrefillKernel(
        run=module.mimo_paged_flash_attn_fp8,
        version=str(getattr(flydsl, "__version__", "unknown")),
        runtime_path=str(getattr(flydsl, "__file__", "unknown")),
        kernel_path=str(getattr(module, "__file__", "unknown")),
    )
    logger.info(
        "Loaded FlyDSL MiMo cached-prefill kernel: runtime=%s version=%s kernel=%s",
        kernel.runtime_path,
        kernel.version,
        kernel.kernel_path,
    )
    return kernel


def get_flydsl_mimo_num_partitions() -> int:
    raw_value = os.getenv(
        FLYDSL_MIMO_NUM_PARTITIONS_ENV,
        str(FLYDSL_MIMO_DEFAULT_NUM_PARTITIONS),
    )
    try:
        num_partitions = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{FLYDSL_MIMO_NUM_PARTITIONS_ENV} must be one of "
            f"{FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS}; got {raw_value!r}"
        ) from exc
    if num_partitions not in FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS:
        raise ValueError(
            f"{FLYDSL_MIMO_NUM_PARTITIONS_ENV} must be one of "
            f"{FLYDSL_MIMO_SUPPORTED_NUM_PARTITIONS}; got {num_partitions}"
        )
    return num_partitions


@dataclass(frozen=True)
class FlyDSLPADecodeKernels:
    pa_decode_tile: Callable
    compile_pa_decode_tile: Callable
    compile_pa_decode_reduce: Callable
    version: str
    runtime_path: str
    kernel_path: str


@lru_cache(maxsize=1)
def load_flydsl_pa_decode_kernels() -> FlyDSLPADecodeKernels:
    """Load the optional local FlyDSL page-64 tile implementation.

    FlyDSL is intentionally not imported at module scope: the normal AITER
    Gluon configuration must continue to work without FlyDSL installed. The
    BF16 integration requires the 0.2.4 native runtime and the
    ``mimo_flydsl_kernels`` 0.1.2 compile API.
    """

    try:
        flydsl = importlib.import_module("flydsl")
        tile_module = importlib.import_module("kernels.attention.pa_decode_tile")
        reduce_module = importlib.import_module("kernels.attention.pa_decode_swa")
        # pa_decode_tile imports graph-capture and dtype helpers from this
        # module inside its host wrapper. Load it now so capture never performs
        # the first import.
        importlib.import_module("kernels.attention.pa_decode_fp8")
    except Exception as exc:
        raise RuntimeError(
            "SGLANG_AITER_PA_DECODE_IMPL=flydsl requires a compatible FlyDSL "
            "native runtime and the local FlyDSL repository-root `kernels` "
            "package on PYTHONPATH. The MiMo setup expects the FlyDSL 0.2.4 "
            "runtime plus mimo_flydsl_kernels 0.1.2 or newer."
        ) from exc

    version = str(getattr(flydsl, "__version__", "unknown"))
    if version != "0.2.4":
        raise RuntimeError(
            "MiMo phase-1 FlyDSL integration requires the validated 0.2.4 "
            f"native runtime; imported version {version!r} from "
            f"{getattr(flydsl, '__file__', 'unknown')}"
        )

    compile_tile = getattr(tile_module, "compile_pa_decode_tile", None)
    try:
        compile_tile_parameters = (
            inspect.signature(compile_tile).parameters
            if compile_tile is not None
            else {}
        )
        supports_native_v = {
            "bf16_kv",
            "v_head_dim",
        }.issubset(compile_tile_parameters)
    except (TypeError, ValueError):
        supports_native_v = False
    if not supports_native_v:
        raise RuntimeError(
            "SGLANG_AITER_PA_DECODE_IMPL=flydsl requires "
            "a compatible FlyDSL source whose compile_pa_decode_tile accepts "
            "both `bf16_kv` and `v_head_dim`; imported "
            f"incompatible kernel from {getattr(tile_module, '__file__', 'unknown')}"
        )

    return FlyDSLPADecodeKernels(
        pa_decode_tile=tile_module.pa_decode_tile,
        compile_pa_decode_tile=compile_tile,
        compile_pa_decode_reduce=reduce_module.compile_pa_decode_sw_reduce,
        version=version,
        runtime_path=str(getattr(flydsl, "__file__", "unknown")),
        kernel_path=str(getattr(tile_module, "__file__", "unknown")),
    )


def can_build_mimo_paged_kv_metadata(
    kv_cache_is_vectorized_5d: bool,
    page_size: int,
    kv_cache_dtype,
    q_dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> bool:
    """Whether to scatter ragged slot ids into page-64 tables.

    FlyPA (gfx942/gfx950, BF16 or FP8 SHUFFLE-5D) and CK FP8 paged prefill
    both consume ``paged_kv_indptr`` / ``paged_kv_indices`` /
    ``paged_kv_last_page_len``. Without these tables ``can_use_mimo_flypa_prefill``
    always returns False.
    """
    return (
        kv_cache_is_vectorized_5d
        and page_size == CK_MIMO_PREFILL_PAGE_SIZE
        and kv_cache_dtype in (fp8_dtype, torch.bfloat16)
        and q_dtype == torch.bfloat16
        and num_qo_heads == CK_MIMO_PREFILL_QUERY_HEADS
        and num_kv_heads == CK_MIMO_PREFILL_KV_HEADS
        and head_dim == CK_MIMO_PREFILL_HEAD_DIM
    )


def can_use_mimo_flypa_prefill(
    backend: AiterAttnBackend,
    layer: RadixAttention,
    window_size,
    sinks,
    is_swa_layer: bool,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> bool:
    """Return whether FlyPA can consume this cached MiMo 5D extend.

    Unlike CK paged prefill, FlyPA is not FP8-only. gfx950 FP8 prefers the
    FlyDSL paged kernel when that path qualifies; otherwise FlyPA covers
    gfx942/gfx950 BF16 or FP8 SHUFFLE-5D full-attention extends with
    page-64 metadata.
    """
    if not get_bool_env_var(FLYPA_MIMO_PREFILL_ENV, "false"):
        return False
    if not is_mimo_flypa_arch():
        return False
    if (
        is_swa_layer
        or sinks is not None
        or tuple(window_size) != (-1, -1)
        or backend.input_dtype != torch.bfloat16
        or backend.page_size != CK_MIMO_PREFILL_PAGE_SIZE
        or float(backend.logits_soft_cap) != 0.0
    ):
        return False
    if sub_pool.dtype not in (fp8_dtype, torch.bfloat16):
        return False
    if (
        layer.tp_q_head_num != CK_MIMO_PREFILL_QUERY_HEADS
        or layer.tp_k_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.tp_v_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.qk_head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.v_head_dim != CK_MIMO_PREFILL_VALUE_HEAD_DIM
    ):
        return False
    if k_buf.ndim != 5 or v_buf.ndim != 5 or k_buf.shape[0] != v_buf.shape[0]:
        return False
    pack = 16 // k_buf.element_size()
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // pack,
        CK_MIMO_PREFILL_PAGE_SIZE,
        pack,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // pack,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        pack,
    )
    if tuple(k_buf.shape[1:]) != expected_k_tail:
        return False
    if tuple(v_buf.shape[1:]) != expected_v_tail:
        return False
    if any(
        getattr(metadata, name, None) is None
        for name in (
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
        )
    ):
        return False
    if (
        metadata.paged_kv_indptr.dtype != torch.int32
        or metadata.paged_kv_indices.dtype != torch.int32
    ):
        return False
    if sub_pool.dtype == fp8_dtype:
        k_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        v_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
        if not _is_scalar_f32_scale(k_scale) or not _is_scalar_f32_scale(v_scale):
            return False
    return True


def run_mimo_flypa_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    bs0: int,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> torch.Tensor:
    k_paged = (
        k_buf.view(sub_pool.dtype)
        if sub_pool.store_dtype != sub_pool.dtype
        else k_buf
    )
    v_paged = (
        v_buf.view(sub_pool.dtype)
        if sub_pool.store_dtype != sub_pool.dtype
        else v_buf
    )
    if sub_pool.dtype == fp8_dtype:
        q_local, q_descale_local = quantize_query_per_tensor_fp8(q)
        k_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
        v_descale_local = (
            layer.v_scale if layer.v_scale is not None else backend.v_scale
        )
        if not _is_scalar_f32_scale(q_descale_local):
            raise RuntimeError("FlyPA FP8 prefill requires a scalar Q descale")
    else:
        q_local = q
        unit = torch.ones((), dtype=torch.float32, device=q.device)
        q_descale_local = unit
        k_descale_local = unit
        v_descale_local = unit
    q_paged = q_local.contiguous().view(
        -1, layer.tp_q_head_num, layer.qk_head_dim
    )
    o = flypa(
        num_qo_heads=layer.tp_q_head_num,
        num_kv_heads=layer.tp_k_head_num,
        head_dim_qk=layer.qk_head_dim,
        head_dim_v=layer.v_head_dim,
        page_size=backend.page_size,
        is_causal=True,
        quant_query_mode="per-tensor",
    )(
        q_paged,
        k_paged,
        v_paged,
        backend.qo_indptr[:bs0],
        None,
        metadata.paged_kv_indptr[:bs0],
        metadata.paged_kv_indices,
        int(metadata.max_q_len),
        int(metadata.max_kv_len),
        True,
        q_descale_local,
        k_descale_local,
        v_descale_local,
        metadata.paged_kv_last_page_len,
    )
    if o.dtype != backend.input_dtype:
        o = o.to(backend.input_dtype)
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def can_use_mimo_flydsl_fp8_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    window_size,
    sinks,
    is_swa_layer: bool,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> bool:
    """Return whether gfx950 FlyDSL paged FP8 can consume this cached extend.

    Size-gated (``max_q >= 4096``, ``max_kv >= 8192``) and FP8-only. Tried
    before FlyPA so both env flags can stay on without FlyPA stealing the
    gfx950 dual-wave kernel.
    """
    if not get_bool_env_var(FLYDSL_MIMO_PREFILL_ENV, "false"):
        return False
    if not is_gfx950() or is_swa_layer or sinks is not None:
        return False
    if tuple(window_size) != (-1, -1):
        return False
    if (
        backend.input_dtype != torch.bfloat16
        or backend.kv_cache_dtype != fp8_dtype
        or sub_pool.dtype != fp8_dtype
        or backend.page_size != CK_MIMO_PREFILL_PAGE_SIZE
        or float(backend.logits_soft_cap) != 0.0
    ):
        return False
    if (
        layer.tp_q_head_num != CK_MIMO_PREFILL_QUERY_HEADS
        or layer.tp_k_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.tp_v_head_num != CK_MIMO_PREFILL_KV_HEADS
        or layer.qk_head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.head_dim != CK_MIMO_PREFILL_HEAD_DIM
        or layer.v_head_dim != FLYDSL_MIMO_VALUE_HEAD_DIM
        or getattr(layer, "mimo_original_v_head_dim", None) != 128
    ):
        return False
    max_q = getattr(metadata, "max_q_len", None)
    max_kv = getattr(metadata, "max_kv_len", None)
    if (
        max_q is None
        or max_kv is None
        or int(max_q) < FLYDSL_MIMO_PREFILL_MIN_Q
        or int(max_kv) < FLYDSL_MIMO_PREFILL_MIN_KV
    ):
        return False
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // 16,
        CK_MIMO_PREFILL_PAGE_SIZE,
        16,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // 16,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        16,
    )
    if (
        k_buf.ndim != 5
        or v_buf.ndim != 5
        or k_buf.shape[0] != v_buf.shape[0]
        or tuple(k_buf.shape[1:]) != expected_k_tail
        or tuple(v_buf.shape[1:]) != expected_v_tail
        or k_buf.element_size() != 1
        or v_buf.element_size() != 1
    ):
        return False
    if any(
        getattr(metadata, name, None) is None
        for name in (
            "kv_indptr",
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
        )
    ):
        return False
    if (
        metadata.kv_indptr.dtype != torch.int32
        or metadata.paged_kv_indptr.dtype != torch.int32
        or metadata.paged_kv_indices.dtype != torch.int32
    ):
        return False
    k_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
    v_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
    return (
        _is_scalar_f32_scale(k_scale)
        and _is_scalar_f32_scale(v_scale)
        and k_scale.device == q.device
        and v_scale.device == q.device
    )


def run_mimo_flydsl_fp8_prefill(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    bs0: int,
    sub_pool,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    metadata,
) -> torch.Tensor:
    k_paged = (
        k_buf.view(sub_pool.dtype)
        if sub_pool.store_dtype != sub_pool.dtype
        else k_buf
    )
    v_paged = (
        v_buf.view(sub_pool.dtype)
        if sub_pool.store_dtype != sub_pool.dtype
        else v_buf
    )
    q_local, q_descale_local = quantize_query_per_tensor_fp8(q)
    k_descale_local = (
        layer.k_scale if layer.k_scale is not None else backend.k_scale
    )
    v_descale_local = (
        layer.v_scale if layer.v_scale is not None else backend.v_scale
    )
    q_paged = q_local.contiguous().view(
        -1, layer.tp_q_head_num, layer.qk_head_dim
    )
    o = load_flydsl_mimo_prefill_kernel().run(
        q_paged,
        k_paged,
        v_paged,
        backend.qo_indptr[:bs0],
        metadata.kv_indptr[:bs0],
        metadata.paged_kv_indptr[:bs0],
        metadata.paged_kv_indices,
        max_seqlen_q=int(metadata.max_q_len),
        max_seqlen_kv=int(metadata.max_kv_len),
        q_descale=q_descale_local,
        k_descale=k_descale_local,
        v_descale=v_descale_local,
        stream=(
            torch.cuda.current_stream(q.device)
            if q.device.type == "cuda"
            else None
        ),
    )
    if o.dtype != backend.input_dtype:
        o = o.to(backend.input_dtype)
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def forward_extend_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    bs0: int,
    window_size,
    sinks,
) -> torch.Tensor:
    """``forward_extend`` specialization for the SHUFFLE 5D KV pool.

    Four sub-paths handle vectorized-5D extend:

    1. Fresh-prompt shortcut: when every request in the batch has zero
       ``extend_prefix_lens`` (first chunk of a fresh prompt, or any
       path bypassing prefix reuse) the fresh ``(k, v)`` inputs ARE the
       full KV stream — skip pool reads entirely and run on bf16
       ``(k, v)`` directly. No descales needed since no data is read
       from the (possibly fp8) cache.

    2. gfx950 cached BF16 varlen ASM: gather prefix+chunk and reuse the same
       D192/V128 ASM as fresh prefill. This stays ahead of FlyPA on gfx950.

    3. Direct paged FlyDSL / FlyPA / CK: gfx950 FP8 SHUFFLE-5D prefers the
       size-gated FlyDSL paged kernel over FlyPA. FlyPA is the gfx942 (and
       gfx950 fallback) paged kernel for BF16 or FP8. Remaining FP8 pages
       go to CK.

    4. Gather-and-linearize: every unsupported case gathers per-token K/V from the
       SHUFFLE 5D pool via ``launch_gather_shuffle_5d_to_linear``
       (triton inverse of the SHUFFLE writer) into a contiguous
       ``(T, H, D)`` buffer in the cache's ``store_dtype``, then run the
       same LINEAR prefill. fp8-store layers are forwarded to aiter as
       raw fp8 with the per-tensor descales — aiter's LINEAR-mode kernel
       supports fp8 K/V/Q natively, so no host-side dequant is needed.

    The optimized paths are deliberately narrow. SWA/sink, unsupported dtypes,
    other head shapes, and missing metadata retain the established fallback.

    Returns the ``(T, H_q * D_v)`` attention output, ready to be
    returned from ``AiterAttnBackend.forward_extend``.
    """
    asm_q, asm_k, asm_v = q, k, v
    asm_output_kwargs = {}
    if MIMO_FRESH_BF16_ASM_ENABLED and is_gfx950():
        asm_q, asm_k, asm_v, physical_q_tokens = _mimo_logical_qkv_views(
            q, k, v, forward_batch
        )
        if asm_q.shape[0] != physical_q_tokens:
            asm_output_kwargs = {"output_token_count": physical_q_tokens}

    # Path 1: fresh-prompt shortcut.
    extend_no_prefix = forward_batch.extend_prefix_lens_cpu is not None and not any(
        forward_batch.extend_prefix_lens_cpu
    )
    if (
        get_bool_env_var(FLYPA_MIMO_PREFILL_ENV, "false")
        and is_gfx942()
        and not is_gfx950()
        and (layer.sliding_window_size is None or int(layer.sliding_window_size) < 0)
    ):
        # gfx942 has no D192/V128 ASM. Force the paged path so FlyPA covers
        # both the first chunk and later cached chunks. gfx950 keeps the
        # fresh ASM/CK shortcut (and SWA never uses FlyPA).
        extend_no_prefix = False

    if extend_no_prefix:
        if can_use_mimo_fresh_bf16_asm(
            backend,
            asm_q,
            asm_k,
            asm_v,
            layer,
            forward_batch,
            window_size,
            sinks,
        ):
            return mimo_fresh_bf16_asm(
                asm_q,
                asm_k,
                asm_v,
                layer,
                forward_batch,
                **asm_output_kwargs,
            )

        if can_use_mimo_fresh_bf16_varlen_asm(
            backend,
            asm_q,
            asm_k,
            asm_v,
            layer,
            forward_batch,
            window_size,
            sinks,
        ):
            return mimo_fresh_bf16_varlen_asm(
                backend,
                asm_q,
                asm_k,
                asm_v,
                layer,
                forward_batch,
                **asm_output_kwargs,
            )

        if can_use_mimo_fresh_bf16_swa_varlen(
            backend,
            q,
            k,
            v,
            layer,
            forward_batch,
            window_size,
            sinks,
        ):
            return mimo_fresh_bf16_swa_varlen(
                backend,
                q,
                k,
                v,
                layer,
                forward_batch,
                window_size,
                sinks,
            )

        # Q and K are head-aligned views whose last dimension is contiguous.
        # AITER's LINEAR prefill kernel accepts their token stride directly, so
        # avoid materializing full-token copies for the fresh-prompt chunk.
        k_lin = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_lin = v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim)
        total_tokens = k_lin.shape[0]
        kv_indices_lin = torch.arange(
            total_tokens, dtype=torch.int32, device=k_lin.device
        )
        kv_indptr_lin = backend.qo_indptr[:bs0]
        max_q = int(backend.forward_metadata.max_q_len)
        o = mha_batch_prefill_func(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            k_lin,
            v_lin,
            backend.qo_indptr[:bs0],
            kv_indptr_lin,
            kv_indices_lin,
            max_q,
            max_q,
            causal=True,
            logits_soft_cap=backend.logits_soft_cap,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
            window_size=window_size,
            sink_ptr=sinks,
        )
        if o.dtype != backend.input_dtype:
            o = o.to(backend.input_dtype)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # Resolve the raw 5D K/V buffer for this layer (going through the
    # SWA->sub-pool mapping when applicable).
    is_swa_layer = (
        layer.sliding_window_size is not None
        and layer.sliding_window_size > -1
        and backend.forward_metadata.swa_page_table is not None
    )
    pool = backend.token_to_kv_pool
    if hasattr(pool, "layers_mapping"):
        sub_layer_id, sub_is_swa = pool.layers_mapping[layer.layer_id]
        sub_pool = pool.swa_kv_pool if sub_is_swa else pool.full_kv_pool
    else:
        sub_pool = pool
        sub_layer_id = layer.layer_id
    k_buf = sub_pool.k_buffer[sub_layer_id - sub_pool.start_layer]
    v_buf = sub_pool.v_buffer[sub_layer_id - sub_pool.start_layer]

    metadata = backend.forward_metadata
    has_paged_metadata = all(
        getattr(metadata, name, None) is not None
        for name in (
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
        )
    )
    expected_k_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_HEAD_DIM // 16,
        CK_MIMO_PREFILL_PAGE_SIZE,
        16,
    )
    expected_v_tail = (
        CK_MIMO_PREFILL_KV_HEADS,
        CK_MIMO_PREFILL_PAGE_SIZE // 16,
        CK_MIMO_PREFILL_VALUE_HEAD_DIM,
        16,
    )
    # gfx950 cached BF16 full-attn prefers varlen ASM over FlyPA.
    # gfx950 FP8 prefers FlyDSL paged over FlyPA when size gates pass.
    # gfx942 never qualifies for either (is_gfx950() is false).
    if can_use_mimo_chunk_bf16_varlen_asm(
        backend,
        asm_q,
        asm_k,
        asm_v,
        layer,
        forward_batch,
        window_size,
        sinks,
        is_swa_layer,
        sub_pool,
        k_buf,
        v_buf,
        metadata,
    ):
        return mimo_chunk_bf16_varlen_asm(
            backend,
            asm_q,
            layer,
            forward_batch,
            k_buf,
            v_buf,
            metadata,
            **asm_output_kwargs,
        )

    if can_use_mimo_flydsl_fp8_prefill(
        backend,
        q,
        layer,
        window_size,
        sinks,
        is_swa_layer,
        sub_pool,
        k_buf,
        v_buf,
        metadata,
    ):
        return run_mimo_flydsl_fp8_prefill(
            backend,
            q,
            layer,
            bs0,
            sub_pool,
            k_buf,
            v_buf,
            metadata,
        )

    if can_use_mimo_flypa_prefill(
        backend,
        layer,
        window_size,
        sinks,
        is_swa_layer,
        sub_pool,
        k_buf,
        v_buf,
        metadata,
    ):
        return run_mimo_flypa_prefill(
            backend,
            q,
            layer,
            bs0,
            sub_pool,
            k_buf,
            v_buf,
            metadata,
        )

    use_direct_paged = (
        mha_batch_prefill_func is not None
        and not is_swa_layer
        and sinks is None
        and tuple(window_size) == (-1, -1)
        and backend.input_dtype == torch.bfloat16
        and backend.kv_cache_dtype == fp8_dtype
        and sub_pool.dtype == fp8_dtype
        and backend.page_size == CK_MIMO_PREFILL_PAGE_SIZE
        and layer.tp_q_head_num == CK_MIMO_PREFILL_QUERY_HEADS
        and layer.tp_k_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.tp_v_head_num == CK_MIMO_PREFILL_KV_HEADS
        and layer.qk_head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and layer.v_head_dim == CK_MIMO_PREFILL_VALUE_HEAD_DIM
        and layer.head_dim == CK_MIMO_PREFILL_HEAD_DIM
        and float(backend.logits_soft_cap) == 0.0
        and has_paged_metadata
        and k_buf.ndim == 5
        and v_buf.ndim == 5
        and tuple(k_buf.shape[1:]) == expected_k_tail
        and tuple(v_buf.shape[1:]) == expected_v_tail
        and k_buf.shape[0] == v_buf.shape[0]
        and k_buf.element_size() == 1
        and v_buf.element_size() == 1
    )

    if use_direct_paged:
        # FP8 pools may expose uint8 storage because some PyTorch indexing
        # operations do not implement float8. Reinterpret the identical bytes
        # without copying before handing the physical 5D cache to AITER.
        k_paged = (
            k_buf.view(sub_pool.dtype)
            if sub_pool.store_dtype != sub_pool.dtype
            else k_buf
        )
        v_paged = (
            v_buf.view(sub_pool.dtype)
            if sub_pool.store_dtype != sub_pool.dtype
            else v_buf
        )
        q_local, q_descale_local = quantize_query_per_tensor_fp8(q)
        k_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
        v_descale_local = (
            layer.v_scale if layer.v_scale is not None else backend.v_scale
        )
        max_kv = int(metadata.max_kv_len)
        max_q = int(metadata.max_q_len)
        q_paged = q_local.contiguous().view(
            -1, layer.tp_q_head_num, layer.qk_head_dim
        )
        o = mha_batch_prefill_func(
            q_paged,
            k_paged,
            v_paged,
            backend.qo_indptr[:bs0],
            metadata.paged_kv_indptr[:bs0],
            metadata.paged_kv_indices,
            max_q,
            max_kv,
            causal=True,
            logits_soft_cap=0.0,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
            window_size=(-1, -1),
            sink_ptr=None,
            q_descale=q_descale_local,
            k_descale=k_descale_local,
            v_descale=v_descale_local,
            kv_last_page_lens=metadata.paged_kv_last_page_len,
        )
        if o.dtype != backend.input_dtype:
            o = o.to(backend.input_dtype)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # Path 4: gather-and-linearize. SWA layers gather from the SWA sub-pool;
    # full-attention layers gather from the full pool. Both metadata tensors
    # contain per-token absolute slots in request-major order.
    total_kv = int(forward_batch.seq_lens_sum)
    if is_swa_layer:
        slot_ids = metadata.swa_page_table[:total_kv]
    else:
        slot_ids = metadata.kv_indices[:total_kv]

    k_lin, v_lin = launch_gather_shuffle_5d_to_linear(k_buf, v_buf, slot_ids)
    # k_lin / v_lin come out in ``store_dtype`` (uint8 for fp8 pools
    # because ``Tensor.index_put`` isn't implemented for fp8 — see
    # ``MHATokenToKVPool`` ctor). Reinterpret them back to the compute
    # dtype so aiter sees matching q/k/v dtypes. The bytes are
    # identical; this is a zero-copy view.
    if sub_pool.store_dtype != sub_pool.dtype:
        k_lin = k_lin.view(sub_pool.dtype)
        v_lin = v_lin.view(sub_pool.dtype)

    # For fp8 K/V we hand the raw fp8 tensors and the layer's per-tensor
    # descales straight to aiter.
    if sub_pool.dtype == fp8_dtype:
        q_local, q_descale_local = quantize_query_per_tensor_fp8(q)
        k_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
        v_descale_local = (
            layer.v_scale if layer.v_scale is not None else backend.v_scale
        )
    else:
        q_local = q
        q_descale_local = None
        k_descale_local = None
        v_descale_local = None

    kv_indptr_lin = backend.forward_metadata.kv_indptr[:bs0]
    kv_indices_lin = torch.arange(total_kv, dtype=torch.int32, device=k_lin.device)
    max_kv = int(backend.forward_metadata.max_kv_len)
    max_q = int(backend.forward_metadata.max_q_len)

    o = mha_batch_prefill_func(
        q_local.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
        k_lin,
        v_lin,
        backend.qo_indptr[:bs0],
        kv_indptr_lin,
        kv_indices_lin,
        max_q,
        max_kv,
        causal=True,
        logits_soft_cap=backend.logits_soft_cap,
        alibi_slopes=None,
        return_lse=False,
        return_attn_probs=False,
        window_size=window_size,
        sink_ptr=sinks,
        q_descale=q_descale_local,
        k_descale=k_descale_local,
        v_descale=v_descale_local,
    )
    if o.dtype != backend.input_dtype:
        o = o.to(backend.input_dtype)
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


def forward_decode_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o: torch.Tensor,
    sinks,
) -> None:
    """``forward_decode`` specialization for the SHUFFLE 5D KV pool.

    Runs ``pa_decode_gluon`` for both full-attention and sliding-window
    layers — when SHUFFLE 5D is active the SWA sub-pool is also
    allocated 5D (see ``SWAKVPool`` ctor), so we keep one decode kernel
    instead of falling back to ``unified_attention`` for SWA layers.

    The choice between the two layer kinds is purely metadata:

    * Full-attn  → ``kv_indices`` page table + ``sliding_window=0`` +
      ``max_part_num`` recommended by aiter heuristics.
    * SWA layer  → ``swa_page_table`` + ``sliding_window=layer.sliding_window_size``
      + ``max_part_num=1`` (SWA windows are small enough that
      splitting does not help).

    fp8 KV requires per-tensor ``key_scale`` / ``value_scale`` to be
    forwarded; without them the kernel reads the fp8 bytes as fp8
    values without any dequant and produces garbage logits.

    Writes the attention output into ``o`` in place (via a stride-0
    safe ``o.view``).
    """
    bs = forward_batch.batch_size
    num_kv_heads = layer.tp_k_head_num
    num_q_heads = layer.tp_q_head_num
    q_group = num_q_heads // num_kv_heads
    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > -1
    )

    if is_swa_layer:
        block_tables_pa = (
            backend.forward_metadata.swa_page_table
            if backend.forward_metadata.swa_page_table is not None
            else backend.forward_metadata.kv_indices
        )
        ctx_part = 256
        max_part_num = 1
        sliding_window_arg = int(layer.sliding_window_size)
    else:
        block_tables_pa = backend.forward_metadata.kv_indices
        ctx_part = 256
        max_part_num = get_recommended_splits(bs, num_kv_heads)
        sliding_window_arg = 0

    q_in = q.view(-1, num_q_heads, layer.qk_head_dim)
    # Direct view of o as kernel output — saves a per-layer o.copy_ of
    # bs * H_q * D bf16 elementwise.
    o_view = o.view(-1, num_q_heads, layer.v_head_dim)
    exp_sums = torch.empty(
        (bs, num_kv_heads, max_part_num, q_group),
        dtype=torch.float32,
        device=q_in.device,
    )
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        (bs, num_kv_heads, max_part_num, q_group, layer.v_head_dim),
        dtype=q_in.dtype,
        device=q_in.device,
    )

    # For fp8 KV cache the kernel needs per-tensor dequant scales
    # (key_scale / value_scale). Without them the fp8 bytes are
    # interpreted as fp8 values with no dequant.
    key_scale = None
    value_scale = None
    if backend.kv_cache_dtype == fp8_dtype:
        key_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        value_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale

    pa_decode_gluon(
        output=o_view,
        query=q_in,
        key_cache=k_cache,
        value_cache=v_cache,
        context_lengths=forward_batch.seq_lens,
        block_tables=block_tables_pa,
        softmax_scale=layer.scaling,
        query_length=1,
        max_context_partition_num=max_part_num,
        context_partition_size=ctx_part,
        compute_type=backend.input_dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=sinks,
        sliding_window=sliding_window_arg,
        ps=True,
    )


def _get_flydsl_workspace_views(
    backend: AiterAttnBackend,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_partitions = backend._flydsl_pa_decode_num_partitions
    scalar_numel = (
        batch_size
        * FLYDSL_MIMO_KV_HEADS
        * num_partitions
        * FLYDSL_MIMO_EQUIVALENT_GROUP_SIZE
    )
    output_numel = scalar_numel * FLYDSL_MIMO_VALUE_HEAD_DIM
    buffers = (
        ("pmax", backend._flydsl_pa_decode_pmax, scalar_numel),
        ("psum", backend._flydsl_pa_decode_psum, scalar_numel),
        ("pout", backend._flydsl_pa_decode_pout, output_numel),
        ("context lengths", backend._flydsl_pa_decode_context_lengths, batch_size),
    )
    for name, buffer, required_numel in buffers:
        if buffer is None or buffer.numel() < required_numel:
            available = 0 if buffer is None else buffer.numel()
            raise RuntimeError(
                "FlyDSL PA decode workspace is not initialized for this batch: "
                f"{name} requires {required_numel} elements, has {available}"
            )

    scalar_shape = (
        batch_size,
        FLYDSL_MIMO_KV_HEADS,
        num_partitions,
        FLYDSL_MIMO_EQUIVALENT_GROUP_SIZE,
    )
    pmax = backend._flydsl_pa_decode_pmax[:scalar_numel].view(scalar_shape)
    psum = backend._flydsl_pa_decode_psum[:scalar_numel].view(scalar_shape)
    pout = backend._flydsl_pa_decode_pout[:output_numel].view(
        *scalar_shape, FLYDSL_MIMO_VALUE_HEAD_DIM
    )
    context_lengths = backend._flydsl_pa_decode_context_lengths[:batch_size]
    return pmax, psum, pout, context_lengths


def forward_target_verify_flydsl_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    sinks,
) -> None:
    """Run the exact MiMo phase-1 full TARGET_VERIFY shape with FlyDSL."""

    if getattr(backend, "_flydsl_pa_decode_tile", None) is None:
        raise RuntimeError("FlyDSL PA decode was selected but is not initialized")

    query_length = int(backend.forward_metadata.max_q_len)
    batch_size = int(forward_batch.batch_size)
    num_q_heads = int(layer.tp_q_head_num)
    num_kv_heads = int(layer.tp_k_head_num)
    head_dim = int(layer.qk_head_dim)
    v_head_dim = int(layer.v_head_dim)

    if query_length != FLYDSL_MIMO_QUERY_LENGTH:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires query_length=4; " f"got {query_length}"
        )
    if (num_q_heads, num_kv_heads) != (
        FLYDSL_MIMO_QUERY_HEADS,
        FLYDSL_MIMO_KV_HEADS,
    ):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires 16 Q heads and 1 KV head; "
            f"got {num_q_heads}/{num_kv_heads}"
        )
    if (
        head_dim != FLYDSL_MIMO_HEAD_DIM
        or v_head_dim != FLYDSL_MIMO_VALUE_HEAD_DIM
    ):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires QK/V head dimensions "
            f"192/128; got {head_dim}/{v_head_dim}"
        )
    if backend.page_size != FLYDSL_MIMO_PAGE_SIZE:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires page_size=64; "
            f"got {backend.page_size}"
        )
    if layer.sliding_window_size is not None and int(layer.sliding_window_size) > -1:
        raise ValueError("FlyDSL MiMo phase 1 does not support sliding-window layers")
    if sinks is not None:
        raise ValueError("FlyDSL MiMo phase 1 does not support attention sinks")
    if float(layer.logit_cap or 0.0) != 0.0:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY does not support attention logit "
            f"soft-capping; got logit_cap={layer.logit_cap}"
        )
    if backend.kv_cache_dtype not in (fp8_dtype, torch.bfloat16):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires FP8 E4M3 or BF16 KV cache; "
            f"got {backend.kv_cache_dtype}"
        )
    if q.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires BF16 query and output; "
            f"got {q.dtype}/{output.dtype}"
        )
    if q.shape != (batch_size * query_length, num_q_heads * head_dim):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY query shape mismatch: expected "
            f"{(batch_size * query_length, num_q_heads * head_dim)}, got "
            f"{tuple(q.shape)}"
        )
    if output.shape != (batch_size * query_length, num_q_heads * v_head_dim):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY output shape mismatch: expected "
            f"{(batch_size * query_length, num_q_heads * v_head_dim)}, got "
            f"{tuple(output.shape)}"
        )

    kv_vector_width = 8 if backend.kv_cache_dtype == torch.bfloat16 else 16
    expected_k_tail = (
        num_kv_heads,
        head_dim // kv_vector_width,
        FLYDSL_MIMO_PAGE_SIZE,
        kv_vector_width,
    )
    expected_v_tail = (
        num_kv_heads,
        FLYDSL_MIMO_PAGE_SIZE // kv_vector_width,
        v_head_dim,
        kv_vector_width,
    )
    if k_cache.ndim != 5 or tuple(k_cache.shape[1:]) != expected_k_tail:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY K-cache layout mismatch: expected "
            f"[blocks, {', '.join(map(str, expected_k_tail))}], got "
            f"{tuple(k_cache.shape)}"
        )
    if v_cache.ndim != 5 or tuple(v_cache.shape[1:]) != expected_v_tail:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY V-cache layout mismatch: expected "
            f"[blocks, {', '.join(map(str, expected_v_tail))}], got "
            f"{tuple(v_cache.shape)}"
        )
    if k_cache.shape[0] != v_cache.shape[0]:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires K/V caches with the same "
            f"block count; got {k_cache.shape[0]}/{v_cache.shape[0]}"
        )
    expected_element_size = 2 if backend.kv_cache_dtype == torch.bfloat16 else 1
    if (
        k_cache.dtype != backend.kv_cache_dtype
        or v_cache.dtype != backend.kv_cache_dtype
        or k_cache.element_size() != expected_element_size
        or v_cache.element_size() != expected_element_size
    ):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY cache storage does not match the "
            f"configured KV dtype; got {k_cache.dtype}/{v_cache.dtype}"
        )

    block_tables = backend.forward_metadata.kv_indices
    if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires a two-dimensional int32 "
            f"block table; got shape={tuple(block_tables.shape)}, "
            f"dtype={block_tables.dtype}"
        )
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY block-table batch mismatch: "
            f"expected {batch_size}, got {block_tables.shape[0]}"
        )
    if forward_batch.seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires integral sequence lengths; "
            f"got {forward_batch.seq_lens.dtype}"
        )
    if forward_batch.seq_lens.shape != (batch_size,):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY sequence-length shape mismatch: "
            f"expected {(batch_size,)}, got {tuple(forward_batch.seq_lens.shape)}"
        )

    q_view = q.view(-1, num_q_heads, head_dim)
    output_view = output.view(-1, num_q_heads, v_head_dim)
    if q_view.stride(-1) != 1:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires a contiguous query head axis"
        )

    key_scale = None
    value_scale = None
    if backend.kv_cache_dtype == fp8_dtype:
        key_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        value_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
        scales = []
        for name, scale in (("key_scale", key_scale), ("value_scale", value_scale)):
            if not isinstance(scale, torch.Tensor):
                raise ValueError(f"FlyDSL MiMo {name} must be a persistent tensor")
            if scale.dtype != torch.float32 or scale.numel() != 1 or scale.ndim > 1:
                raise ValueError(
                    f"FlyDSL MiMo {name} must be a scalar/length-1 float32 "
                    f"per-tensor scale; got shape={tuple(scale.shape)}, "
                    f"dtype={scale.dtype}"
                )
            # SGLang's KV-cache quant method owns 0-D Parameter scales. FlyDSL's
            # tensor JIT requires at least one stride-1 axis, so expose a zero-copy
            # [1] view while retaining the Parameter's stable device allocation.
            scales.append(scale.view(1) if scale.ndim == 0 else scale)
        key_scale, value_scale = scales

    device = q.device
    tensors = [
        ("output", output),
        ("K cache", k_cache),
        ("V cache", v_cache),
        ("block table", block_tables),
        ("sequence lengths", forward_batch.seq_lens),
    ]
    if key_scale is not None:
        tensors.extend((("key scale", key_scale), ("value scale", value_scale)))
    for name, tensor in tensors:
        if tensor.device != device:
            raise ValueError(
                f"FlyDSL MiMo {name} must be on {device}; got {tensor.device}"
            )

    pmax, psum, pout, context_lengths = _get_flydsl_workspace_views(backend, batch_size)
    for name, workspace, dtype in (
        ("pmax", pmax, torch.float32),
        ("psum", psum, torch.float32),
        ("pout", pout, torch.bfloat16),
        ("context lengths", context_lengths, torch.int32),
    ):
        if workspace.device != device or workspace.dtype != dtype:
            raise ValueError(
                f"FlyDSL MiMo {name} workspace must be {dtype} on {device}; "
                f"got {workspace.dtype} on {workspace.device}"
            )
    # SGLang uses int64 sequence lengths on ROCm.  Write the adjusted lengths
    # directly into the persistent int32 FlyDSL buffer: torch.add's `out=`
    # conversion is capture-safe and avoids a per-layer temporary allocation.
    # The phase-1 context limit (1,048,576) is well within int32 range.
    torch.add(forward_batch.seq_lens, query_length, out=context_lengths)

    backend._flydsl_pa_decode_tile(
        output=output_view,
        query=q_view,
        key_cache=k_cache,
        value_cache=v_cache,
        block_tables=block_tables,
        context_lengths=context_lengths,
        key_scale=key_scale,
        value_scale=value_scale,
        softmax_scale=layer.scaling,
        num_partitions=backend._flydsl_pa_decode_num_partitions,
        pmax=pmax,
        psum=psum,
        pout=pout,
    )


def forward_target_verify_vectorized_5d(
    backend: AiterAttnBackend,
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    sinks,
) -> None:
    """Run top-k-1 target verification directly on a SHUFFLE 5D KV pool.

    ``TARGET_VERIFY`` has a fixed query length per request.  For a linear
    draft chain (top-k 1), its mask is the causal multi-token decode mask that
    ``pa_decode_gluon`` implements for query lengths up to four.  The verify
    page table already contains the freshly written draft-token slots, so the
    kernel receives the physical 5D buffers unchanged and uses
    ``seq_lens + query_length`` as its total KV lengths.

    This helper deliberately validates the complete Gluon contract before
    allocating workspaces.  Falling through to the legacy unified-attention
    path is not safe: reshaping SHUFFLE 5D storage as NHD changes only the view,
    not the physical cache permutation.
    """
    if pa_decode_gluon is None or get_recommended_splits is None:
        raise RuntimeError(
            "AITER pa_decode_gluon is unavailable for vectorized-5D TARGET_VERIFY"
        )

    query_length = int(backend.forward_metadata.max_q_len)
    batch_size = int(forward_batch.batch_size)
    num_q_heads = int(layer.tp_q_head_num)
    num_kv_heads = int(layer.tp_k_head_num)

    if not 1 <= query_length <= 4:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires 1 <= query_length <= 4; "
            f"got {query_length}"
        )
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires Q heads to be divisible by "
            f"KV heads; got {num_q_heads} Q heads and {num_kv_heads} KV heads"
        )

    query_group_size = num_q_heads // num_kv_heads
    equivalent_group_size = query_length * query_group_size
    if equivalent_group_size > 64:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY exceeds pa_decode_gluon's equivalent "
            f"query-group limit: {query_length} * {query_group_size} = "
            f"{equivalent_group_size} > 64"
        )
    asymmetric_mimo = (
        num_kv_heads == FLYDSL_MIMO_KV_HEADS
        and layer.qk_head_dim == FLYDSL_MIMO_HEAD_DIM
        and layer.v_head_dim == FLYDSL_MIMO_VALUE_HEAD_DIM
    )
    if layer.qk_head_dim != layer.v_head_dim and not asymmetric_mimo:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY supports symmetric K/V or the MiMo "
            "one-KV-head QK192/V128 route; got "
            f"{num_kv_heads} KV heads and dimensions "
            f"{layer.qk_head_dim}/{layer.v_head_dim}"
        )
    if backend.page_size not in (16, 64, 1024):
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires page size 16, 64, or 1024; "
            f"got {backend.page_size}"
        )
    if float(layer.logit_cap or 0.0) != 0.0:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY does not support attention logit "
            f"soft-capping; got logit_cap={layer.logit_cap}"
        )
    if k_cache.ndim != 5 or v_cache.ndim != 5:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires 5D K and V caches; got "
            f"K ndim={k_cache.ndim}, V ndim={v_cache.ndim}"
        )
    if k_cache.shape[1] != num_kv_heads or v_cache.shape[1] != num_kv_heads:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY cache/head mismatch: "
            f"layer has {num_kv_heads} KV heads, K/V caches have "
            f"{k_cache.shape[1]}/{v_cache.shape[1]}"
        )
    if k_cache.shape[-2] != backend.page_size:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY cache/page mismatch: "
            f"backend page size is {backend.page_size}, K cache page axis is "
            f"{k_cache.shape[-2]}"
        )
    if q.shape[0] != batch_size * query_length:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires a uniform query length; "
            f"got q.shape[0]={q.shape[0]}, batch_size={batch_size}, and "
            f"query_length={query_length}"
        )

    is_swa_layer = (
        layer.sliding_window_size is not None and layer.sliding_window_size > -1
    )
    if is_swa_layer:
        block_tables = (
            backend.forward_metadata.swa_page_table
            if backend.forward_metadata.swa_page_table is not None
            else backend.forward_metadata.kv_indices
        )
        max_part_num = 1
        sliding_window = int(layer.sliding_window_size)
    else:
        block_tables = backend.forward_metadata.kv_indices
        max_part_num = int(get_recommended_splits(batch_size, num_kv_heads))
        sliding_window = 0

    q_view = q.view(-1, num_q_heads, layer.qk_head_dim)
    output_view = output.view(-1, num_q_heads, layer.v_head_dim)
    workspace_shape = (
        batch_size,
        num_kv_heads,
        max_part_num,
        equivalent_group_size,
    )
    exp_sums = torch.empty(workspace_shape, dtype=torch.float32, device=q_view.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        (*workspace_shape, layer.v_head_dim),
        dtype=q_view.dtype,
        device=q_view.device,
    )

    key_scale = None
    value_scale = None
    if backend.kv_cache_dtype == fp8_dtype:
        key_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
        value_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale

    pa_decode_gluon(
        output=output_view,
        query=q_view,
        key_cache=k_cache,
        value_cache=v_cache,
        context_lengths=forward_batch.seq_lens + query_length,
        block_tables=block_tables,
        softmax_scale=layer.scaling,
        query_length=query_length,
        max_context_partition_num=max_part_num,
        context_partition_size=256,
        compute_type=backend.input_dtype,
        key_scale=key_scale,
        value_scale=value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=sinks,
        sliding_window=sliding_window,
        ps=True,
    )
