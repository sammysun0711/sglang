# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""MiMo token-sharded prefill attention with the existing head-sharded KV cache.

Attention collectives stay within each DP group's attention TP group. Expert
routing and dispatch/combine are owned by the caller's existing MoE backend.
"""

import itertools
import logging
from dataclasses import dataclass
from functools import cached_property, lru_cache
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.mimo_v2_utils import mimo_effective_prefix_lengths


@lru_cache(maxsize=2)
def _log_path(name):
    logging.getLogger(__name__).info("Using MiMo EP prefill %s", name)


_SP_SWA_PAGE = 64


def _pack_mimo_sp_swa_pages(
    keys: torch.Tensor, kv_heads: int, dq: int, dv: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack live KV rows into bucketed SHUFFLE-5D page buffers."""
    pages = keys.shape[0] // _SP_SWA_PAGE
    # The allocated page count is part of AITER's SWA compilation key. A minimum
    # capacity of 128 pages with power-of-two growth lets request tails reuse
    # compiled kernels. Runtime page IDs and lengths only address the live prefix.
    capacity = max(128, 1 << (pages - 1).bit_length()) if pages else 0
    k_pages = keys.new_empty((capacity, kv_heads, dq // 8, _SP_SWA_PAGE, 8))
    v_pages = keys.new_empty((capacity, kv_heads, _SP_SWA_PAGE // 8, dv, 8))
    # Convert layouts directly into the live prefix so capacity padding does not
    # add another copy through an intermediate contiguous page buffer.
    k_pages[:pages].copy_(
        keys[..., :dq]
        .reshape(pages, _SP_SWA_PAGE, kv_heads, dq // 8, 8)
        .permute(0, 2, 3, 1, 4)
    )
    v_pages[:pages].copy_(
        keys[..., dq:]
        .reshape(pages, _SP_SWA_PAGE // 8, 8, kv_heads, dv)
        .permute(0, 3, 1, 4, 2)
    )
    return k_pages, v_pages


def _mimo_attn_tp_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather along dim 0 over attention TP (1-byte dtypes as bytes: no FP8 collectives)."""
    tensor = tensor.contiguous()
    data = tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
    out = data.new_empty((get_attention_tp_size() * data.shape[0], *data.shape[1:]))
    get_attention_tp_group().all_gather_into_tensor(out, data)
    return out.view(tensor.dtype)


@dataclass
class _MimoSpSwaMetadata:
    """Per-forward layout of one rank's sliding-window attention under SGLANG_ENABLE_MIMO_SWA_SP.

    Keys of every request segment in this rank's token slice are packed into page-64
    SHUFFLE-5D pages: cached keys left of the chunk, keys from earlier slices, then the
    slice's own keys. kv_src indexes rows of [cached, tails, own, zero] (not materialized).
    """

    key: Tuple[int, int]
    q_rows: int
    tail_rows: int
    cache_rows: int
    cache_loc: Optional[torch.Tensor]
    kv_src: torch.Tensor
    cu_seqlens_q: torch.Tensor
    kv_indptr: torch.Tensor
    page_ids: torch.Tensor
    last_page_lens: torch.Tensor
    max_q: int
    max_kv: int

    @cached_property
    def cache_page(self) -> Optional[torch.Tensor]:
        return None if self.cache_loc is None else self.cache_loc // _SP_SWA_PAGE

    @cached_property
    def cache_offset(self) -> Optional[torch.Tensor]:
        return None if self.cache_loc is None else self.cache_loc % _SP_SWA_PAGE


def _mimo_sp_swa_metadata(
    forward_batch: ForwardBatch, window: int, rows: int
) -> _MimoSpSwaMetadata:
    key = (window, rows)
    meta = getattr(forward_batch, "_mimo_sp_swa_metadata", None)
    if meta is not None and meta.key == key:
        return meta
    attn_tp, rank = get_attention_tp_size(), get_attention_tp_rank()
    lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
    prefixes = mimo_effective_prefix_lengths(forward_batch)
    if prefixes is None:
        raise ValueError("MiMo SWA SP needs effective prefix lengths")
    tail = min(window, rows)
    cache_counts = [min(p, window) if n > 0 else 0 for p, n in zip(prefixes, lens)]
    cache_ends = list(itertools.accumulate(cache_counts))
    cache_rows = cache_ends[-1] if cache_ends else 0
    tail_base = cache_rows
    own_base = tail_base + attn_tp * tail
    zero_row = own_base + rows
    s0, s1 = rank * rows, (rank + 1) * rows
    kv_src, cu_q, kv_indptr, last_page = [], [0], [0], []
    max_q = max_kv = 0
    start = 0
    for i, (length, prefix) in enumerate(zip(lens, prefixes)):
        a, b = start, start + length
        start = b
        qa, qb = max(a, s0), min(b, s1)
        if qa >= qb:
            continue
        from_batch = min(qa - a, window)
        from_cache = min(prefix, window - from_batch)
        parts = []
        if from_cache:
            parts.append(torch.arange(cache_ends[i] - from_cache, cache_ends[i]))
        if from_batch:
            t = torch.arange(qa - from_batch, qa)
            sl = t // rows
            parts.append(tail_base + sl * tail + (t - sl * rows) - (rows - tail))
        parts.append(torch.arange(own_base + qa - s0, own_base + qb - s0))
        idx = torch.cat(parts)
        keys = idx.numel()
        pages = -(-keys // _SP_SWA_PAGE)
        pad = pages * _SP_SWA_PAGE - keys
        if pad:
            idx = torch.cat([idx, torch.full((pad,), zero_row, dtype=idx.dtype)])
        kv_src.append(idx)
        cu_q.append(cu_q[-1] + qb - qa)
        kv_indptr.append(kv_indptr[-1] + pages)
        last_page.append(keys - _SP_SWA_PAGE * (pages - 1))
        max_q, max_kv = max(max_q, qb - qa), max(max_kv, keys)

    device = forward_batch.positions.device
    cache_loc = None
    if cache_rows:
        from sglang.srt.model_executor.forward_context import (
            get_req_to_token_pool,
            get_token_to_kv_pool,
        )

        req_rows = torch.cat(
            [torch.full((c,), i) for i, c in enumerate(cache_counts) if c]
        ).to(device, non_blocking=True)
        positions = torch.cat(
            [torch.arange(p - c, p) for p, c in zip(prefixes, cache_counts) if c]
        ).to(device, non_blocking=True)
        full_loc = get_req_to_token_pool().req_to_token[
            forward_batch.req_pool_indices[req_rows].long(), positions
        ]
        cache_loc = (
            get_token_to_kv_pool()
            .translate_loc_from_full_to_swa(full_loc.long())
            .long()
        )

    def to_device(values, dtype):
        return torch.tensor(values, dtype=dtype).to(device, non_blocking=True)

    meta = _MimoSpSwaMetadata(
        key=key,
        q_rows=cu_q[-1],
        tail_rows=tail,
        cache_rows=cache_rows,
        cache_loc=cache_loc,
        kv_src=(torch.cat(kv_src) if kv_src else torch.empty(0, dtype=torch.int64)).to(
            device, non_blocking=True
        ),
        cu_seqlens_q=to_device(cu_q, torch.int32),
        kv_indptr=to_device(kv_indptr, torch.int32),
        page_ids=torch.arange(kv_indptr[-1], dtype=torch.int32, device=device),
        last_page_lens=to_device(last_page, torch.int32),
        max_q=max_q,
        max_kv=max_kv,
    )
    forward_batch._mimo_sp_swa_metadata = meta
    return meta


@triton.jit(do_not_specialize=["rows", "tail_start"])
def _mimo_sp_swa_send_kernel(
    k_ptr,
    v_ptr,
    send_ptr,
    tail_ptr,
    k_stride,
    v_stride,
    v_scale,
    rows,
    tail_start,
    HEADS: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
):
    """send[r, t] = [K | V] of rank r's KV heads for token t (V scaled); tokens from
    tail_start on also go to tail[t - tail_start] = [K | V] of all KV heads."""
    rank = tl.program_id(1)
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    ck = tl.arange(0, BLOCK_K)[None, :]
    cv = tl.arange(0, BLOCK_V)[None, :]
    k_mask = (t < rows) & (ck < LOCAL_HEADS * DQ)
    v_mask = (t < rows) & (cv < LOCAL_HEADS * DV)
    k = tl.load(k_ptr + t * k_stride + rank * LOCAL_HEADS * DQ + ck, mask=k_mask)
    v = tl.load(v_ptr + t * v_stride + rank * LOCAL_HEADS * DV + cv, mask=v_mask)
    if HAS_V_SCALE:
        v = (v.to(tl.float32) * v_scale).to(v.dtype)
    send = send_ptr + (rank * rows + t) * (LOCAL_HEADS * (DQ + DV))
    tl.store(send + ck, k, mask=k_mask)
    tl.store(send + LOCAL_HEADS * DQ + cv, v, mask=v_mask)
    in_tail = t >= tail_start
    tail = tail_ptr + (t - tail_start) * (HEADS * (DQ + DV))
    tl.store(tail + rank * LOCAL_HEADS * DQ + ck, k, mask=k_mask & in_tail)
    tl.store(tail + HEADS * DQ + rank * LOCAL_HEADS * DV + cv, v, mask=v_mask & in_tail)


@triton.jit(do_not_specialize=["cache_rows", "own_base", "own_end"])
def _mimo_sp_swa_pages_kernel(
    kv_src_ptr,
    cached_ptr,
    tails_ptr,
    k_ptr,
    v_ptr,
    k_pages_ptr,
    v_pages_ptr,
    k_stride,
    v_stride,
    v_scale,
    cache_rows,
    own_base,
    own_end,
    HEADS: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    PAGE: tl.constexpr,
    X: tl.constexpr,
    BLOCK_DQ: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
):
    """One SHUFFLE-5D page of one KV head from the kv_src rows of [cached, tails, own,
    zero]: cached is (attn_tp, cache_rows, [K | V] of the rank's heads), tails is
    (rows, [K | V] of all heads), own rows are read from K and V (V scaled here)."""
    page = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.arange(0, PAGE)[:, None]
    dk = tl.arange(0, BLOCK_DQ)[None, :]
    dv = tl.arange(0, DV)[None, :]
    src = tl.load(kv_src_ptr + page * PAGE + i)
    in_cache = src < cache_rows
    in_tail = (src >= cache_rows) & (src < own_base)
    in_own = (src >= own_base) & (src < own_end)
    cached = cached_ptr + ((head // LOCAL_HEADS) * cache_rows + src) * LOCAL_HEADS * (
        DQ + DV
    )
    tails = tails_ptr + (src - cache_rows) * HEADS * (DQ + DV)
    own = src - own_base
    k_mask = dk < DQ
    # Zero for the padding row (zero_row).
    k = tl.load(
        k_ptr + own * k_stride + head * DQ + dk, mask=in_own & k_mask, other=0.0
    )
    v = tl.load(v_ptr + own * v_stride + head * DV + dv, mask=in_own, other=0.0)
    if HAS_V_SCALE:
        v = (v.to(tl.float32) * v_scale).to(v.dtype)
    head_off = head % LOCAL_HEADS
    k = tl.where(
        in_cache, tl.load(cached + head_off * DQ + dk, mask=in_cache & k_mask), k
    )
    v = tl.where(
        in_cache,
        tl.load(cached + LOCAL_HEADS * DQ + head_off * DV + dv, mask=in_cache),
        v,
    )
    k = tl.where(in_tail, tl.load(tails + head * DQ + dk, mask=in_tail & k_mask), k)
    v = tl.where(in_tail, tl.load(tails + HEADS * DQ + head * DV + dv, mask=in_tail), v)
    base = page * HEADS + head
    tl.store(
        k_pages_ptr + base * DQ * PAGE + dk // X * PAGE * X + i * X + dk % X,
        k,
        mask=k_mask,
    )
    tl.store(v_pages_ptr + base * DV * PAGE + i // X * DV * X + dv * X + i % X, v)


@lru_cache(maxsize=None)
def _mimo_unit_scale(device: torch.device) -> torch.Tensor:
    return torch.ones(1, dtype=torch.float32, device=device)


def _run_mimo_ep_sp_swa(
    self,
    positions: torch.Tensor,
    hidden_states,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """SWA prefill on this rank's token slice with all heads."""
    _log_path("SWA sequence parallel (64Q/8KV, head-sharded persistent cache)")
    from aiter.ops.flydsl import flydsl_paged_attention_swa_bf16

    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.model_executor.forward_context import get_attn_backend

    attn_tp, rank = get_attention_tp_size(), get_attention_tp_rank()
    group = get_attention_tp_group()
    rows = (
        hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    ).shape[0]
    heads, kv_heads = self.total_num_heads, self.total_num_kv_heads
    dq, dv = self.head_dim, self.original_v_head_dim
    local_kv_heads = kv_heads // attn_tp
    # Same window as the attention-TP kernel call (see run_mimo_flypa_swa_prefill).
    window = self.attn.sliding_window_size - int(self.attn.tp_k_head_num == 2)
    meta = _mimo_sp_swa_metadata(forward_batch, window, rows)

    qkv, _ = self.qkv_proj_full(hidden_states)
    q, k, v = qkv.split([heads * dq, kv_heads * dq, kv_heads * dv], dim=-1)
    q, k = self.rotary_emb(positions[rank * rows : (rank + 1) * rows], q, k)
    if self.v_scale is not None:
        v = v * self.v_scale
    own = torch.cat(
        [k.reshape(rows, kv_heads, dq), v.reshape(rows, kv_heads, dv)], dim=-1
    )

    # This rank's KV heads for every token of the DP group -> the head-sharded cache.
    send = own.view(rows, attn_tp, local_kv_heads, dq + dv).transpose(0, 1).contiguous()
    received = torch.empty_like(send)
    group.all_to_all_single(received, send)
    received = received.view(attn_tp * rows, local_kv_heads, dq + dv)
    backend = get_attn_backend()
    pool = backend.token_to_kv_pool
    pool.set_kv_buffer(
        self.attn,
        KVWriteLoc(
            forward_batch.out_cache_loc,
            getattr(backend.forward_metadata, "swa_out_cache_loc", None),
        ),
        received[..., :dq].contiguous(),
        received[..., dq:].contiguous(),
        None,
        None,
    )

    # Window halos: the last keys of every slice, and cached keys left of each chunk.
    sources = []
    if meta.cache_rows:
        k_cache = pool.get_key_buffer(self.attn.layer_id)
        v_cache = pool.get_value_buffer(self.attn.layer_id)
        page, offset = meta.cache_page, meta.cache_offset
        cached = torch.cat(
            [
                k_cache[page, :, :, offset, :].reshape(-1, local_kv_heads, dq),
                v_cache[page, :, offset // 8, :, offset % 8],
            ],
            dim=-1,
        )
        cached = _mimo_attn_tp_gather(cached).view(
            attn_tp, meta.cache_rows, local_kv_heads, dq + dv
        )
        sources.append(
            cached.transpose(0, 1).reshape(meta.cache_rows, kv_heads, dq + dv)
        )
    sources.append(_mimo_attn_tp_gather(own[rows - meta.tail_rows :].contiguous()))
    sources.append(own)
    sources.append(own.new_zeros((1, kv_heads, dq + dv)))
    keys = torch.cat(sources)[meta.kv_src]
    k_pages, v_pages = _pack_mimo_sp_swa_pages(keys, kv_heads, dq, dv)

    out = (q.new_zeros if meta.q_rows < rows else q.new_empty)((rows, heads, dv))
    if meta.q_rows:
        unit = torch.ones(1, dtype=torch.float32, device=q.device)
        flydsl_paged_attention_swa_bf16(
            q.view(rows, heads, dq)[: meta.q_rows],
            k_pages,
            v_pages,
            meta.cu_seqlens_q,
            meta.kv_indptr,
            meta.page_ids,
            meta.max_q,
            meta.max_kv,
            window_left=window,
            kv_last_page_lens=meta.last_page_lens,
            q_descale=unit,
            k_descale=unit,
            v_descale=unit,
            out=out[: meta.q_rows],
            sink_ptr=self.attention_sink_full,
            softmax_scale=self.scaling,
            stream=torch.cuda.current_stream(q.device),
        )
    out, _ = self.o_proj_full(out.view(rows, heads * dv))
    return out


def _run_mimo_tp_sp_swa(
    self,
    positions: torch.Tensor,
    hidden_states,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """SWA prefill on this rank's token slice with all heads (SGLANG_ENABLE_MIMO_SWA_SP)."""
    from aiter.ops.flydsl import flydsl_paged_attention_swa_bf16

    from sglang.srt.layers.attention.utils import gather_shuffle_5d_to_linear
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.model_executor.forward_context import get_attn_backend

    attn_tp, rank = get_attention_tp_size(), get_attention_tp_rank()
    group = get_attention_tp_group()
    rows = (
        hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    ).shape[0]
    heads, kv_heads = self.total_num_heads, self.total_num_kv_heads
    dq, dv = self.head_dim, self.original_v_head_dim
    local_kv_heads = kv_heads // attn_tp
    # Same window as the attention-TP kernel call (see run_mimo_flypa_swa_prefill).
    window = self.attn.sliding_window_size - int(self.attn.tp_k_head_num == 2)
    meta = _mimo_sp_swa_metadata(forward_batch, window, rows)

    qkv, _ = self.qkv_proj_full(hidden_states)
    q, k, v = qkv.split([heads * dq, kv_heads * dq, kv_heads * dv], dim=-1)
    q, k = self.rotary_emb(positions[rank * rows : (rank + 1) * rows], q, k)
    k = k.reshape(rows, kv_heads * dq)
    v_scale = 1.0 if self.v_scale is None else self.v_scale
    row = local_kv_heads * (dq + dv)

    # This rank's KV heads for every token of the DP group -> the head-sharded cache;
    # the last keys of every slice are kept for the windows of the next slices.
    send = q.new_empty((attn_tp, rows, row))
    tail = q.new_empty((meta.tail_rows, kv_heads * (dq + dv)))
    _mimo_sp_swa_send_kernel[(triton.cdiv(rows, 16), attn_tp)](
        k,
        v,
        send,
        tail,
        k.stride(0),
        v.stride(0),
        v_scale,
        rows,
        rows - meta.tail_rows,
        HEADS=kv_heads,
        LOCAL_HEADS=local_kv_heads,
        DQ=dq,
        DV=dv,
        BLOCK_T=16,
        BLOCK_K=triton.next_power_of_2(local_kv_heads * dq),
        BLOCK_V=triton.next_power_of_2(local_kv_heads * dv),
        HAS_V_SCALE=self.v_scale is not None,
    )
    received = torch.empty_like(send)
    group.all_to_all_single(received, send)
    received = received.view(attn_tp * rows, row)
    backend = get_attn_backend()
    pool = backend.token_to_kv_pool
    pool.set_kv_buffer(
        self.attn,
        KVWriteLoc(
            forward_batch.out_cache_loc,
            getattr(backend.forward_metadata, "swa_out_cache_loc", None),
        ),
        received[:, : local_kv_heads * dq].view(-1, local_kv_heads, dq),
        received[:, local_kv_heads * dq :].view(-1, local_kv_heads, dv),
        None,
        None,
    )

    # Window halos: cached keys left of each chunk, and the tails of earlier slices.
    cached = tail  # not read without cached rows
    if meta.cache_rows:
        cached = q.new_empty((meta.cache_rows, row))
        k_cache = pool.get_key_buffer(self.attn.layer_id)
        gather_shuffle_5d_to_linear[(meta.cache_rows, 1)](
            k_cache,
            pool.get_value_buffer(self.attn.layer_id),
            cached,
            cached[:, local_kv_heads * dq :],
            meta.cache_loc,
            row,
            row,
            local_kv_heads,
            dq,
            dv,
            _SP_SWA_PAGE,
            X=k_cache.shape[-1],
            HEAD_BLOCK=triton.next_power_of_2(local_kv_heads),
            KEY_BLOCK_D=triton.next_power_of_2(dq),
            VALUE_BLOCK_D=triton.next_power_of_2(dv),
        )
        cached = _mimo_attn_tp_gather(cached)
    tails = _mimo_attn_tp_gather(tail)
    own_base = meta.cache_rows + attn_tp * meta.tail_rows
    pages = meta.kv_src.shape[0] // _SP_SWA_PAGE
    # The FlyDSL SWA kernel compiles once per K.shape[0] (an unused constexpr), about
    # 1.6 s each: power-of-two page buffers keep that to a few variants.
    alloc = max(128, 1 << max(pages - 1, 0).bit_length())
    k_pages = q.new_empty((alloc, kv_heads, dq // 8, _SP_SWA_PAGE, 8))
    v_pages = q.new_empty((alloc, kv_heads, _SP_SWA_PAGE // 8, dv, 8))
    if pages:
        _mimo_sp_swa_pages_kernel[(pages, kv_heads)](
            meta.kv_src,
            cached,
            tails,
            k,
            v,
            k_pages,
            v_pages,
            k.stride(0),
            v.stride(0),
            v_scale,
            meta.cache_rows,
            own_base,
            own_base + rows,
            HEADS=kv_heads,
            LOCAL_HEADS=local_kv_heads,
            DQ=dq,
            DV=dv,
            PAGE=_SP_SWA_PAGE,
            X=8,
            BLOCK_DQ=triton.next_power_of_2(dq),
            HAS_V_SCALE=self.v_scale is not None,
        )

    out = (q.new_zeros if meta.q_rows < rows else q.new_empty)((rows, heads, dv))
    if meta.q_rows:
        unit = _mimo_unit_scale(q.device)
        flydsl_paged_attention_swa_bf16(
            q.view(rows, heads, dq)[: meta.q_rows],
            k_pages,
            v_pages,
            meta.cu_seqlens_q,
            meta.kv_indptr,
            meta.page_ids,
            meta.max_q,
            meta.max_kv,
            window_left=window,
            kv_last_page_lens=meta.last_page_lens,
            q_descale=unit,
            k_descale=unit,
            v_descale=unit,
            out=out[: meta.q_rows],
            sink_ptr=self.attention_sink_full,
            softmax_scale=self.scaling,
            stream=torch.cuda.current_stream(q.device),
        )
    return self.o_proj.quant_method.apply(
        SimpleNamespace(weight=self.o_proj_full), out.view(rows, heads * dv)
    )


def run_mimo_sp_swa(
    attention,
    positions: torch.Tensor,
    hidden_states,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Run the topology-selected SWA path with shared metadata and cache layout."""
    run = _run_mimo_ep_sp_swa if attention._ep_sp_swa else _run_mimo_tp_sp_swa
    return run(attention, positions, hidden_states, forward_batch)
