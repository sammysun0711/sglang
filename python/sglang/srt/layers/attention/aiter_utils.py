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
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import torch

try:
    # `mha_batch_prefill_func` is re-exported at the aiter top level via
    # `aiter/__init__.py` (`from .ops.mha import *`). Note: a bare
    # `from aiter.mha import ...` does NOT work — that module path only
    # exists as `aiter.ops.mha`.
    from aiter import mha_batch_prefill_func
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )
except ImportError:  # pragma: no cover - import-time guard mirrors aiter_backend
    mha_batch_prefill_func = None
    pa_decode_gluon = None
    get_recommended_splits = None

from sglang.srt.layers.attention.utils import launch_gather_shuffle_5d_to_linear
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

if TYPE_CHECKING:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


FLYDSL_MIMO_QUERY_LENGTH = 4
FLYDSL_MIMO_QUERY_HEADS = 16
FLYDSL_MIMO_KV_HEADS = 1
FLYDSL_MIMO_HEAD_DIM = 192
FLYDSL_MIMO_PAGE_SIZE = 64
FLYDSL_MIMO_NUM_PARTITIONS = 8
FLYDSL_MIMO_EQUIVALENT_GROUP_SIZE = 64


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
    Gluon configuration must continue to work without FlyDSL installed.  The
    phase-1 local integration was validated with the 0.2.4 native runtime and
    the fixed repository-root ``kernels`` source at commit ``c99d5cd``.
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
            "package on PYTHONPATH. The MiMo phase-1 setup expects the FlyDSL "
            "0.2.4 runtime plus source commit c99d5cd."
        ) from exc

    version = str(getattr(flydsl, "__version__", "unknown"))
    if version != "0.2.4":
        raise RuntimeError(
            "MiMo phase-1 FlyDSL integration requires the validated 0.2.4 "
            f"native runtime; imported version {version!r} from "
            f"{getattr(flydsl, '__file__', 'unknown')}"
        )

    return FlyDSLPADecodeKernels(
        pa_decode_tile=tile_module.pa_decode_tile,
        compile_pa_decode_tile=tile_module.compile_pa_decode_tile,
        compile_pa_decode_reduce=reduce_module.compile_pa_decode_sw_reduce,
        version=version,
        runtime_path=str(getattr(flydsl, "__file__", "unknown")),
        kernel_path=str(getattr(tile_module, "__file__", "unknown")),
    )


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

    Two sub-paths, both routing through aiter's 3D LINEAR-mode
    ``mha_batch_prefill_func`` (page_size=1):

    1. Fresh-prompt shortcut: when every request in the batch has zero
       ``extend_prefix_lens`` (first chunk of a fresh prompt, or any
       path bypassing prefix reuse) the fresh ``(k, v)`` inputs ARE the
       full KV stream — skip pool reads entirely and run on bf16
       ``(k, v)`` directly. No descales needed since no data is read
       from the (possibly fp8) cache.

    2. Gather-and-linearize: otherwise gather the per-token K/V from the
       SHUFFLE 5D pool via ``launch_gather_shuffle_5d_to_linear``
       (triton inverse of the SHUFFLE writer) into a contiguous
       ``(T, H, D)`` buffer in the cache's ``store_dtype``, then run the
       same LINEAR prefill. fp8-store layers are forwarded to aiter as
       raw fp8 with the per-tensor descales — aiter's LINEAR-mode kernel
       supports fp8 K/V/Q natively, so no host-side dequant is needed.

    The fallback exists because aiter's paged ``mha_batch_prefill_func``
    lacks a compiled kernel for our
    ``(page_size=64, bf16/fp8, SHUFFLE 5D)`` configuration; calling it
    from the 5D pool aborts with ``"no matching kernel found"``.

    Returns the ``(T, H_q * D_v)`` attention output, ready to be
    returned from ``AiterAttnBackend.forward_extend``.
    """
    # Path 1: fresh-prompt shortcut.
    extend_no_prefix = forward_batch.extend_prefix_lens_cpu is not None and not any(
        forward_batch.extend_prefix_lens_cpu
    )
    if extend_no_prefix:
        k_lin = k.contiguous().view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_lin = v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim)
        total_tokens = k_lin.shape[0]
        kv_indices_lin = torch.arange(
            total_tokens, dtype=torch.int32, device=k_lin.device
        )
        kv_indptr_lin = backend.qo_indptr[:bs0]
        max_q = int(backend.forward_metadata.max_q_len)
        o = mha_batch_prefill_func(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
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

    # Path 2: gather-and-linearize.
    # SWA layers gather from the SWA sub-pool via swa_page_table;
    # full-attn layers gather from the full sub-pool via kv_indices.
    # Both are per-TOKEN slot id lists populated by
    # ``create_flashinfer_kv_indices_triton`` from ``req_to_token`` (one
    # slot id per logical token), so the first ``seq_lens_sum`` entries
    # of either tensor are exactly the per-token absolute pool slot ids
    # in request-major order — no per-token gather metadata to build on
    # host.
    is_swa_layer = (
        layer.sliding_window_size is not None
        and layer.sliding_window_size > -1
        and backend.forward_metadata.swa_page_table is not None
    )
    total_kv = int(forward_batch.seq_lens_sum)
    if is_swa_layer:
        slot_ids = backend.forward_metadata.swa_page_table[:total_kv]
    else:
        slot_ids = backend.forward_metadata.kv_indices[:total_kv]

    # Resolve the raw 5D K/V buffer for this layer (going through the
    # SWA→sub-pool mapping when applicable).
    pool = backend.token_to_kv_pool
    if hasattr(pool, "layers_mapping"):
        sub_layer_id, sub_is_swa = pool.layers_mapping[layer.layer_id]
        sub_pool = pool.swa_kv_pool if sub_is_swa else pool.full_kv_pool
    else:
        sub_pool = pool
        sub_layer_id = layer.layer_id
    k_buf = sub_pool.k_buffer[sub_layer_id - sub_pool.start_layer]
    v_buf = sub_pool.v_buffer[sub_layer_id - sub_pool.start_layer]

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
        q_local = q.to(fp8_dtype)
        q_descale_local = (
            layer.k_scale if layer.k_scale is not None else backend.k_scale
        )
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
        (bs, num_kv_heads, max_part_num, q_group, layer.qk_head_dim),
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
    scalar_numel = (
        batch_size
        * FLYDSL_MIMO_KV_HEADS
        * FLYDSL_MIMO_NUM_PARTITIONS
        * FLYDSL_MIMO_EQUIVALENT_GROUP_SIZE
    )
    output_numel = scalar_numel * FLYDSL_MIMO_HEAD_DIM
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
        FLYDSL_MIMO_NUM_PARTITIONS,
        FLYDSL_MIMO_EQUIVALENT_GROUP_SIZE,
    )
    pmax = backend._flydsl_pa_decode_pmax[:scalar_numel].view(scalar_shape)
    psum = backend._flydsl_pa_decode_psum[:scalar_numel].view(scalar_shape)
    pout = backend._flydsl_pa_decode_pout[:output_numel].view(
        *scalar_shape, FLYDSL_MIMO_HEAD_DIM
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
    if head_dim != FLYDSL_MIMO_HEAD_DIM or v_head_dim != FLYDSL_MIMO_HEAD_DIM:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires equal padded QK/V head "
            f"dimensions of 192; got {head_dim}/{v_head_dim}"
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
    if backend.kv_cache_dtype != fp8_dtype:
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires the configured FP8 E4M3 KV "
            f"cache dtype; got {backend.kv_cache_dtype}"
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

    expected_k_tail = (
        num_kv_heads,
        head_dim // 16,
        FLYDSL_MIMO_PAGE_SIZE,
        16,
    )
    expected_v_tail = (
        num_kv_heads,
        FLYDSL_MIMO_PAGE_SIZE // 16,
        v_head_dim,
        16,
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
    if (
        k_cache.dtype != backend.kv_cache_dtype
        or v_cache.dtype != backend.kv_cache_dtype
        or k_cache.element_size() != 1
        or v_cache.element_size() != 1
    ):
        raise ValueError(
            "FlyDSL MiMo TARGET_VERIFY requires one-byte configured FP8 K/V "
            f"storage; got {k_cache.dtype}/{v_cache.dtype}"
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
    tensors = (
        ("output", output),
        ("K cache", k_cache),
        ("V cache", v_cache),
        ("block table", block_tables),
        ("sequence lengths", forward_batch.seq_lens),
        ("key scale", key_scale),
        ("value scale", value_scale),
    )
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
        num_partitions=FLYDSL_MIMO_NUM_PARTITIONS,
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
    if layer.qk_head_dim != layer.v_head_dim:
        raise ValueError(
            "vectorized-5D TARGET_VERIFY requires equal padded QK and V head "
            f"dimensions; got {layer.qk_head_dim} and {layer.v_head_dim}"
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
        (*workspace_shape, layer.qk_head_dim),
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
