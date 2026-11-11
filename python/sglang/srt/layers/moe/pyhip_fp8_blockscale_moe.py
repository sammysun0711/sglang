# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2026 <copyright holders>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""gfx942 FP8 TP prefill MoE with fused top-k sum and reduce-scatter.

Ported from luocheng25/pyhip at 7f3706caa7a697e2b1489fc24b43756571075fbf:
- src/pyhip/ops/moe/flydsl/moe_fp8_blockscale_942/{kernels,moe}.py
- benchmarks/moe_fp8_blockscale/probes/sum_rs.hip (adapted fused reduction)
"""

import ctypes
import logging
from typing import List, Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl

from sglang.srt.layers.quantization.pyhip_fp8_blockscale_gemm import (
    _DROP,
    BK,
    BM,
    LDS_BYTES,
    SLOT,
    THREADS,
    _as_f32,
    _buffer,
    _compute,
    _dma,
    _fma,
    _mfma,
    _min,
    _operand,
    _pack_bf16,
    _pin_i32,
    _read_address,
    _read_frag,
    _read_scalar_g2r,
    _scalar_ready,
    _schedule,
    _stage_end,
    _uniform,
    _wait,
    flydsl_disk_cache,
)
from sglang.srt.utils import is_gfx942_supported

logger = logging.getLogger(__name__)

# These two slots were renamed in the current shared GEMM helper module.
P1_SLOT, P2_SLOT = 2 * SLOT, 3 * SLOT


FP8_MAX = 240.0
# Short-K down (K <= 256): 4-wave CTAs, two per CU; W2 streams through a ring of two 64-row LDS chunks.
DOWN_THREADS = 256
DOWN_CHUNK = 64
DOWN_COLS = 1024


def _load_i32(resource, voffset, soffset=0):
    return fx.Int32(
        rocdl.raw_ptr_buffer_load(
            fx.Int32.ir_type,
            resource,
            fx.Int32(voffset).ir_value(),
            fx.Int32(soffset).ir_value(),
            aux=0,
        )
    )


def _load_f32(resource, voffset, soffset=0):
    return fx.Float32(
        rocdl.raw_ptr_buffer_load(
            fx.Float32.ir_type,
            resource,
            fx.Int32(voffset).ir_value(),
            fx.Int32(soffset).ir_value(),
            aux=0,
        )
    )


def _store(value, resource, voffset):
    rocdl.raw_ptr_buffer_store(
        value.ir_value(), resource, fx.Int32(voffset).ir_value(), fx.Int32(0).ir_value()
    )


def _readlane(value, lane):
    return fx.Int32(rocdl.readlane(fx.Int32.ir_type, fx.Int32(value).ir_value(), lane))


def _write_lds_f32(address, value):
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [fx.Int32(address).ir_value(), fx.Float32(value).ir_value()],
        "ds_write_b32 $0, $1",
        "v,v,~{memory}",
        has_side_effects=True,
    )


def _read_lds_f32(address):
    return fx.Float32(
        llvm.inline_asm(
            fx.Float32.ir_type,
            [fx.Int32(address).ir_value()],
            "ds_read_b32 $0, $1",
            "=v,v,~{memory}",
            has_side_effects=True,
        )
    )


def _dma_x_gather(resource, storage, q, table, chunk, h, half, kb):
    """X slot <half>, K-block kb: DMA t covers sorted rows 128 half + 2 q + 8 t + h (table lanes 32 half + 2 t + h)."""
    # Pinned per call so the loop-invariant readlanes stay next to their DMAs instead of 64 SGPRs.
    table = _pin_i32(table)
    for t in range(16):
        lo = _readlane(table, 32 * half + 2 * t)
        hi = _readlane(table, 32 * half + 2 * t + 1)
        _dma(
            resource,
            storage,
            half * SLOT + q * 256 + t * 1024,
            (h == 0).select(lo, hi) + chunk,
            kb * BK,
        )


def _dma_w13(resource, storage, q, lane4, packet, kb, base16, K):
    """Preshuffled W13 row groups base16 .. + 7 (P1 gate / P2 up) of K-block kb: group t / 2 at 2 KiB, 16-byte
    chunk q + 4 (t & 1) at 256 bytes (the dense kernel's preshuffled slot layout)."""
    slot = P1_SLOT if packet == 0 else P2_SLOT
    for t in range(16):
        _dma(
            resource,
            storage,
            slot + q * 256 + t * 1024,
            lane4,
            (base16 + (t >> 1)) * (16 * K) + (kb * 8 + q + 4 * (t & 1)) * 256,
        )


def _load_xs_gather(resource, offsets, soffset):
    return [_load_f32(resource, offsets[mb], soffset) for mb in range(4)]


def _silu_mul(gate, up):
    e = fx.Float32(
        rocdl.exp2(
            fx.Float32.ir_type, (gate * fx.Float32(-1.4426950408889634)).ir_value()
        )
    )
    return (
        gate
        * fx.Float32(rocdl.rcp(fx.Float32.ir_type, (e + fx.Float32(1.0)).ir_value()))
        * up
    )


def _max(a, b):
    return (a > b).select(a, b)


def _abs(a):
    return (a < fx.Float32(0.0)).select(fx.Float32(0.0) - a, a)


def _cvt_fp8x4(values):
    low = rocdl.cvt_pk_fp8_f32(
        fx.Int32.ir_type, values[0], values[1], fx.Int32(0), False
    )
    return fx.Int32(
        rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, values[2], values[3], low, True)
    )


def _xcd_order(block, tiles):
    """Logical tile of block b (< tiles): XCD b % 8 takes a contiguous chunk of the logical order."""
    x8 = block & 7
    return x8 * (tiles >> 3) + _min(x8, tiles & 7) + (block >> 3)


def _retire_last(state):
    acc = [fx.Vector(c) for c in state[:8]]
    # The last two blocks (up N-block 7, M-blocks 2 and 3) are still unretired.
    p2, s2, p3, s3 = (
        fx.Vector(state[8]),
        fx.Float32(state[9]),
        fx.Vector(state[10]),
        fx.Float32(state[11]),
    )
    acc[7] = fx.Vector.from_elements(
        [
            (
                _fma(p2[i - 8], s2, acc[7][i])
                if 8 <= i < 12
                else _fma(p3[i - 12], s3, acc[7][i]) if i >= 12 else acc[7][i]
            )
            for i in range(16)
        ],
        fx.Float32,
    )
    return acc


def _act_epilogue(
    acc, storage, shared, wave, lane, wm, wn, r16, g, SW, A2, A2S, R, row0, rows, jt, I
):
    """SiLU(gate) * up, 1x128 FP8 requant of the tile's first `rows` rows into A2 / A2S (see module docstring)."""
    # act[nbl][4 mb + e]: token 16 mb + r16, column 64 wn + 16 nbl + 4 g + e of this 128-column group.
    act = [
        [_silu_mul(acc[nbl][i], acc[4 + nbl][i]) for i in range(16)] for nbl in range(4)
    ]
    amax = []
    for mb in range(4):
        m = _abs(act[0][4 * mb])
        for i in range(1, 16):
            m = _max(m, _abs(act[i // 4][4 * mb + i % 4]))
        m = _max(m, fx.Float32(m.shuffle_xor(16, 64)))
        amax.append(_max(m, fx.Float32(m.shuffle_xor(32, 64))))
    # The group spans waves wn = 0, 1: exchange row maxima through the wave-private LDS regions (G0 reuses
    # X0 / X1, read for the last time two stages ago; G1 reuses P1 / P2).
    own = (g == 0).select(
        amax[0], (g == 1).select(amax[1], (g == 2).select(amax[2], amax[3]))
    )
    _write_lds_f32(shared + wave * 8192 + lane * 4, own)
    _wait(lgkmcnt=0)
    _stage_end()
    other = shared + (wave ^ 1) * 8192
    partner = [_read_lds_f32(other + (16 * mb + r16) * 4) for mb in range(4)]
    partner_own = _read_lds_f32(other + lane * 4)
    sw_res = _buffer(
        fx.make_view(fx.get_iter(SW) + fx.Int64(row0), fx.make_layout(BM, 1)), BM * 4
    )
    route = _load_f32(sw_res, (wm * 64 + lane) * 4)
    _wait(lgkmcnt=0)

    # The FP8 tile goes through LDS so every store writes whole 128-byte rows: tile row r at 4 KiB of region
    # r >> 5 (clear of the row maxima), 16-byte chunks swizzled by r & 7.
    for mb in range(4):
        top = _max(amax[mb], partner[mb])
        inv = (top > fx.Float32(0.0)).select(fx.Float32(FP8_MAX) / top, fx.Float32(0.0))
        stage = (
            shared
            + (2 * wm + (mb >> 1)) * 8192
            + 4096
            + (16 * (mb & 1) + r16) * 128
            + g * 4
        )
        for nbl in range(4):
            word = _cvt_fp8x4([act[nbl][4 * mb + e] * inv for e in range(4)])
            _write_lds_f32(stage + (((4 * wn + nbl) ^ (r16 & 7)) << 4), _as_f32(word))
    _wait(lgkmcnt=0)
    _stage_end()
    # Wave w stores tile rows 32 w .. + 31; lane l: row 8 i + (l >> 3), chunk l & 7.
    rows8 = [
        _read_address(
            shared
            + wave * 8192
            + 4096
            + (8 * i + (lane >> 3)) * 128
            + (((lane & 7) ^ (lane >> 3)) << 4)
        )
        for i in range(4)
    ]
    a2_res = _buffer(
        fx.make_view(fx.get_iter(A2) + fx.Int64(row0) * I, fx.make_layout(BM * I, 1)),
        rows * I,
    )
    _wait(lgkmcnt=0)
    for i in range(4):
        _store(
            rows8[i],
            a2_res,
            (wave * 32 + 8 * i + (lane >> 3)) * I + jt * 128 + (lane & 7) * 16,
        )
    # Scale (times the route weight) of token `lane` of the wave, written by the wn = 0 wave.
    top = _max(own, partner_own)
    scale = (top > fx.Float32(0.0)).select(
        top * fx.Float32(1.0 / FP8_MAX) * route, fx.Float32(0.0)
    )
    a2s_res = _buffer(A2S, (I // 128) * R * 4)
    _store(
        scale,
        a2s_res,
        ((wn == 0) & (wm * 64 + lane < rows)).select(
            (jt * R + row0 + wm * 64 + lane) * 4, fx.Int32(_DROP)
        ),
    )


@flyc.jit
def _gate_up_body(
    X,
    W,
    XS,
    WS,
    SID,
    SW,
    A2,
    A2S,
    storage,
    T,
    R,
    row0,
    rows,
    jt,
    expert,
    I: fx.Constexpr[int],
    K: fx.Constexpr[int],
    GROUP: fx.Constexpr[int],
):
    """Tile rows row0 .. row0 + rows - 1 (rows = 64, 128, 192 or 256, one expert), intermediate columns
    128 jt .. + 127. Waves whose 64-row band lies past `rows` keep the DMA / barrier schedule but issue no
    MFMAs."""
    # FlyDSL keys its disk cache on helpers named in this function's own code, not in closures.
    _helpers = (
        _dma,
        _dma_x_gather,
        _dma_w13,
        _read_frag,
        _compute,
        _fma,
        _as_f32,
        _load_xs_gather,
        _buffer,
        _min,
        _pin_i32,
        _read_scalar_g2r,
        _scalar_ready,
        _schedule,
        _stage_end,
        _uniform,
        _wait,
        _load_i32,
        _load_f32,
        _store,
        _readlane,
        _write_lds_f32,
        _read_lds_f32,
        _silu_mul,
        _max,
        _abs,
        _cvt_fp8x4,
        _retire_last,
        _act_epilogue,
        _read_address,
    )
    KB = K // BK
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    q = wave & 3
    wm, wn = (q >> 1) + 2 * GROUP, q & 1
    h = lane >> 5
    r16, g = lane & 15, lane >> 4
    active = wm * 64 < rows

    x_res = _buffer(X, T * K)
    w_res = _buffer(
        fx.make_view(
            fx.get_iter(W) + fx.Int64(expert) * (2 * I * K),
            fx.make_layout(2 * I * K, 1),
        ),
        2 * I * K,
    )
    xs_res = _buffer(XS, T * (KB * 4))
    sid_res = _buffer(
        fx.make_view(fx.get_iter(SID) + fx.Int64(row0), fx.make_layout(BM, 1)), BM * 4
    )

    def token_offset(row, unit):
        token = _load_i32(sid_res, row * 4) & 0xFFFFFF
        return ((row < rows) & (token < T)).select(token * unit, fx.Int32(_DROP))

    # Padding rows (token T) and rows past the tile read past the X / scale bounds: zero rows, zero scales.
    table = _pin_i32(
        token_offset(128 * h + 2 * q + 8 * ((lane & 31) >> 1) + (lane & 1), K)
    )
    xs_off = [_pin_i32(token_offset(wm * 64 + 16 * mb + r16, 4)) for mb in range(4)]
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    x_read = [
        _pin_i32(
            shared
            + GROUP * SLOT
            + ((q >> 1) * 64 + r16) * 128
            + (((2 * g + j) ^ (r16 & 7)) << 4)
        )
        for j in range(2)
    ]
    w_read = [
        _pin_i32(shared + wn * 8192 + (2 * g + j) * 256 + r16 * 16) for j in range(2)
    ]
    chunk = _pin_i32(((((lane & 31) >> 2) ^ ((q << 1) | h)) << 4) + ((lane & 3) << 2))
    lane4 = _pin_i32(lane * 4)
    gate16 = jt * 8
    up16 = jt * 8 + I // 16
    wsg_row = (expert * (2 * I // 128) + jt) * KB
    wsu_row = wsg_row + (I // 128) * KB

    @flyc.jit
    def kblock(
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        part0,
        scale0,
        part1,
        scale1,
        xs0,
        xs1,
        xs2,
        xs3,
        wsg_bits,
        wsu_bits,
        kb,
        LAST: fx.Constexpr[bool],
    ):
        acc = [fx.Vector(c) for c in (c0, c1, c2, c3, c4, c5, c6, c7)]
        kb = fx.Int32(kb)
        following = _min(kb + 1, fx.Int32(KB - 1))
        # M1: X and gate rows; G0: DMA up rows (kb), G1: DMA X0(kb+1).
        rocdl.s_setprio(0)
        x = [_read_frag(x_read, mb * 2048) for mb in range(4)]
        w = [_read_frag(w_read, P1_SLOT + nbl * 2048) for nbl in range(4)]
        wsg = _as_f32(wsg_bits)
        scales = tuple(fx.Float32(s) * wsg for s in (xs0, xs1, xs2, xs3))
        if fx.const_expr(GROUP == 0):
            _dma_w13(w_res, storage, q, lane4, 1, kb, up16, K)
        else:
            if fx.const_expr(not LAST):
                _dma_x_gather(x_res, storage, q, table, chunk, h, 0, kb + 1)
        _wait(lgkmcnt=0)
        _stage_end()
        # C1: gate N-blocks 0-3 (idle bands keep their accumulators).
        rocdl.s_setprio(3)
        queue = [
            (fx.Vector(part0), (7, 2), fx.Float32(scale0)),
            (fx.Vector(part1), (7, 3), fx.Float32(scale1)),
        ]
        a0, a1, a2, a3, a4, a5, a6, a7 = acc
        p0, s0, p1, s1 = queue[0][0], queue[0][2], queue[1][0], queue[1][2]
        if active:
            acc_c = [a0, a1, a2, a3, a4, a5, a6, a7]
            queue_c = _compute(w, x, acc_c, queue, scales, 0)
            _schedule(16, 0)
            rocdl.sched_barrier(0)
            a0, a1, a2, a3, a4, a5, a6, a7 = acc_c
            p0, s0, p1, s1 = queue_c[0][0], queue_c[0][2], queue_c[1][0], queue_c[1][2]
        acc = [a0, a1, a2, a3, a4, a5, a6, a7]
        queue = [(p0, (3, 2), s0), (p1, (3, 3), s1)]
        rocdl.sched_barrier(0)
        if fx.const_expr(GROUP == 0):
            # Up rows (kb) for M2 of both groups.
            _wait(vmcnt=0)
        _stage_end()
        # M2: up rows; G0: DMA gate rows (kb+1), G1: DMA X1(kb+1); scales of K-block kb+1.
        rocdl.s_setprio(0)
        w = [_read_frag(w_read, P2_SLOT + nbl * 2048) for nbl in range(4)]
        wsu = _as_f32(wsu_bits)
        scales = tuple(fx.Float32(s) * wsu for s in (xs0, xs1, xs2, xs3))
        wsg_request = _read_scalar_g2r(WS, wsg_row + following)
        wsu_request = _read_scalar_g2r(WS, wsu_row + following)
        if fx.const_expr(not LAST):
            if fx.const_expr(GROUP == 0):
                _dma_w13(w_res, storage, q, lane4, 0, kb + 1, gate16, K)
            else:
                _dma_x_gather(x_res, storage, q, table, chunk, h, 1, kb + 1)
                # X0(kb+1) for G0's next M1; X1(kb+1) may stay in flight.
                _wait(vmcnt=16)
        xs_next = _load_xs_gather(xs_res, xs_off, following * (T * 4))
        _wait(lgkmcnt=0)
        wsg_next = _scalar_ready(wsg_request)
        wsu_next = _scalar_ready(wsu_request)
        _stage_end()
        # C2: up N-blocks 4-7.
        rocdl.s_setprio(3)
        a0, a1, a2, a3, a4, a5, a6, a7 = acc
        p0, s0, p1, s1 = queue[0][0], queue[0][2], queue[1][0], queue[1][2]
        if active:
            acc_c = [a0, a1, a2, a3, a4, a5, a6, a7]
            queue_c = _compute(w, x, acc_c, queue, scales, 4)
            _schedule(16, 1)
            rocdl.sched_barrier(0)
            a0, a1, a2, a3, a4, a5, a6, a7 = acc_c
            p0, s0, p1, s1 = queue_c[0][0], queue_c[0][2], queue_c[1][0], queue_c[1][2]
        acc = [a0, a1, a2, a3, a4, a5, a6, a7]
        queue = [(p0, (7, 2), s0), (p1, (7, 3), s1)]
        rocdl.sched_barrier(0)
        _wait(vmcnt=0)
        if fx.const_expr(not (LAST and GROUP == 1)):
            # G1 skips its last barrier: G0 has no partner stage left.
            _stage_end()
        return (
            *acc,
            queue[0][0],
            queue[0][2],
            queue[1][0],
            queue[1][2],
            *xs_next,
            wsg_next,
            wsu_next,
        )

    # Prologue: G0 loads the gate rows of K-block 0, G1 X0(0) and X1(0); scales of K-block 0.
    if fx.const_expr(GROUP == 0):
        _dma_w13(w_res, storage, q, lane4, 0, fx.Int32(0), gate16, K)
    else:
        _dma_x_gather(x_res, storage, q, table, chunk, h, 0, fx.Int32(0))
        _dma_x_gather(x_res, storage, q, table, chunk, h, 1, fx.Int32(0))
    wsg_request = _read_scalar_g2r(WS, wsg_row)
    wsu_request = _read_scalar_g2r(WS, wsu_row)
    xs_first = _load_xs_gather(xs_res, xs_off, 0)
    _wait(vmcnt=0, lgkmcnt=0)
    wsg_first = _scalar_ready(wsg_request)
    wsu_first = _scalar_ready(wsu_request)
    _stage_end()
    if fx.const_expr(GROUP == 1):
        # G1 falls one stage behind.
        _stage_end()
    zero = fx.Vector.filled(16, 0.0, fx.Float32)
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
    state = (
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero4,
        fx.Float32(0.0),
        zero4,
        fx.Float32(0.0),
        *xs_first,
        wsg_first,
        wsu_first,
    )
    for kb in range(fx.Int32(0), fx.Int32(KB - 1), fx.Int32(1)):
        state = kblock(*state, kb, False)
    state = kblock(*state, fx.Int32(KB - 1), True)
    _act_epilogue(
        _retire_last(state),
        storage,
        shared,
        wave,
        lane,
        wm,
        wn,
        r16,
        g,
        SW,
        A2,
        A2S,
        R,
        row0,
        rows,
        jt,
        I,
    )


def _tile(TILES, tile):
    """(first row, rows, expert) of compact tile `tile` (see the route sort)."""
    row0 = _scalar_ready(_read_scalar_g2r(TILES, 2 * tile))
    info = _scalar_ready(_read_scalar_g2r(TILES, 2 * tile + 1))
    return row0, (info >> 16) * 64, info & 0xFFFF


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _gate_up_kernel(
    X: fx.Tensor,
    W: fx.Tensor,
    XS: fx.Tensor,
    WS: fx.Tensor,
    SID: fx.Tensor,
    TILES: fx.Tensor,
    COUNT: fx.Tensor,
    SW: fx.Tensor,
    A2: fx.Tensor,
    A2S: fx.Tensor,
    T: fx.Int32,
    R: fx.Int32,
    I: fx.Constexpr[int],
    K: fx.Constexpr[int],
):
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, LDS_BYTES, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES, 1))
    )
    body = _gate_up_body
    order = _xcd_order
    block = fx.Int32(gpu.block_id("x"))
    TN = I // 128
    tiles = _scalar_ready(_read_scalar_g2r(COUNT, 0)) * TN
    group = _uniform(fx.Int32(gpu.thread_id("x")) >> 8)
    if block < tiles:
        v = order(block, tiles)
        tile = v // TN
        jt = v - tile * TN
        row0, rows, expert = _tile(TILES, tile)
        if group != 0:
            body(
                X,
                W,
                XS,
                WS,
                SID,
                SW,
                A2,
                A2S,
                storage,
                T,
                R,
                row0,
                rows,
                jt,
                expert,
                I,
                K,
                1,
            )
        else:
            body(
                X,
                W,
                XS,
                WS,
                SID,
                SW,
                A2,
                A2S,
                storage,
                T,
                R,
                row0,
                rows,
                jt,
                expert,
                I,
                K,
                0,
            )


def _load_v4(resource, voffset, soffset=0):
    vt = ir.VectorType.get([4], fx.Int32.ir_type)
    return fx.Vector(
        rocdl.raw_ptr_buffer_load(
            vt,
            resource,
            fx.Int32(voffset).ir_value(),
            fx.Int32(soffset).ir_value(),
            aux=0,
        )
    )


def _down_lane(lane):
    """DMA lane offset inside one 256-byte piece (16 rows x 16 K bytes): LDS row position rho holds row
    rho ^ (4 (rho >> 3)), which keeps the permuted fragment reads of _read_unit bank-conflict free.
    """
    return (((lane >> 2) ^ ((lane >> 5) << 2)) << 4) + ((lane & 3) << 2)


def _dma_chunk(resource, storage, wave, lane_off, slot, chunk, K):
    """W2 rows 64 chunk .. + 63 of the task (contiguous 64 K bytes in the 16x16 shuffle) -> LDS slot."""
    pieces = DOWN_CHUNK * K // 1024
    for t in range(pieces):
        p = wave * pieces + t
        _dma(
            resource,
            storage,
            slot * (DOWN_CHUNK * K) + p * 256,
            lane_off,
            chunk * (DOWN_CHUNK * K) + p * 256,
        )


def _read_unit(w_read, slot, unit, K):
    """Fragments wf[v][kb] (two ds_read_b128 each) of a 32-row unit. MFMA v takes unit row
    8 (r16 >> 2) + 4 v + (r16 & 3) as its A row r16, so output lane (r16, g) holds columns 8 g + 4 v .. + 3.
    """
    base = slot * (DOWN_CHUNK * K) + unit * (32 * K)
    return [
        [
            [_read_address(w_read[v][j], base + kb * 2048) for j in range(2)]
            for kb in range(K // BK)
        ]
        for v in range(2)
    ]


def _fma4(a, b, c):
    vt = ir.VectorType.get([4], fx.Float32.ir_type)
    splat = fx.Vector.from_elements([fx.Float32(b)] * 4, fx.Float32)
    return fx.Vector(
        llvm.call_intrinsic(
            vt,
            "llvm.fma.v4f32",
            [fx.Vector(a).ir_value(), splat.ir_value(), fx.Vector(c).ir_value()],
            [],
            [],
        )
    )


def _retire(item):
    """out = p0 * s0 + p1 * s1 (FP32, K-block order) -> BF16 RNE: 4 words = the lane's 8 consecutive columns."""
    parts, s0, s1 = item[:3]
    words = []
    for v in range(2):
        packed = _pack_bf16(
            _fma4(parts[v][1], s1, fx.Vector(parts[v][0]) * fx.Float32(s0))
        ).bitcast(fx.Int32)
        words += [packed[0], packed[1]]
    return words


def _store_pair(first, second, y_res, voffset, soffset):
    """Columns 8 g .. + 7 of an even unit and of the odd unit after it, back to back: each 128-byte line
    of the row is complete within two adjacent stores (64-byte halves written apart cost ~40% bandwidth).
    """
    for words, offset in ((first, soffset - 64), (second, soffset)):
        rocdl.raw_ptr_buffer_store(
            fx.Vector.from_elements(words, fx.Int32).ir_value(),
            y_res,
            fx.Int32(voffset).ir_value(),
            fx.Int32(offset).ir_value(),
        )


def _unit(wf, x, scales, item, held, y_res, vo, soffset, odd, prefetch):
    """64 MFMAs of a 32-column unit (4 M-blocks x 2 v x 2 K-blocks x 4 steps). Each M-block's partials are
    retired during the next M-block's MFMAs (3 VALU slots per MFMA); the first retires `item` (the previous
    unit's last M-block), the last is returned as the next item. Even units' words wait in `held` and are
    stored with the odd unit's. prefetch() issues the next unit's fragment reads up front.
    """
    rocdl.sched_barrier(0)
    following = prefetch() if prefetch is not None else None
    held = list(held)
    for mb in range(4):
        acc = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)] for _ in range(2)
        ]
        for step in range(4):
            for v in range(2):
                for kb in range(2):
                    acc[v][kb] = _mfma(
                        _operand(wf[v][kb], step), _operand(x[mb][kb], step), acc[v][kb]
                    )
        words = _retire(item)
        slot = 3 if mb == 0 else mb - 1
        if (not odd) if mb == 0 else odd:
            _store_pair(held[slot], words, y_res, item[3], item[4])
        else:
            held[slot] = words
        item = (acc, scales[mb][0], scales[mb][1], vo[mb], soffset)
    if prefetch is not None:
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 8, 0)
    for _ in range(64):
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
        rocdl.sched_group_barrier(0x002, 3, 0)
    rocdl.sched_barrier(0)
    return item, held, following


def _maybe(flag, fn):
    return fn if flag else None


def _down_idle(w_res, storage, wave, lane_off, ITERS, K):
    """After the common prologue, a wave without rows in the tile: its quarter of every W2 chunk DMA and the
    barriers of the computing waves (after U0 of every pair, after U2 of every pair but the last).
    """
    for it in range(ITERS - 1):
        _wait(vmcnt=0)
        _stage_end()
        _dma_chunk(w_res, storage, wave, lane_off, 0, fx.Int32(2 * it + 2), K)
        _wait(vmcnt=0)
        _stage_end()
        _dma_chunk(w_res, storage, wave, lane_off, 1, fx.Int32(2 * it + 3), K)
    _wait(vmcnt=0)
    _stage_end()


@flyc.jit
def _down_direct_body(
    A2,
    W,
    A2S,
    WS,
    SID,
    Y,
    storage,
    R,
    T,
    tile_row,
    nblk,
    cg,
    expert,
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
):
    """Rows tile_row .. + 64 nblk - 1 (one expert) x columns cg * DOWN_COLS ..: wave w < nblk owns rows 64 w ..
    + 63, the other waves only take their part of the W2 DMA ring (no MFMAs)."""
    _helpers = (
        _load_v4,
        _load_i32,
        _load_f32,
        _dma,
        _down_lane,
        _dma_chunk,
        _read_unit,
        _read_address,
        _fma4,
        _retire,
        _store_pair,
        _unit,
        _maybe,
        _operand,
        _mfma,
        _pack_bf16,
        _as_f32,
        _buffer,
        _min,
        _pin_i32,
        _read_scalar_g2r,
        _scalar_ready,
        _stage_end,
        _uniform,
        _wait,
        _down_idle,
    )
    assert K == 256, "direct down: K = 256 (two K-blocks per unit)"
    KB = K // BK
    ITERS = DOWN_COLS // 128
    tid = fx.Int32(gpu.thread_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    r16, g = lane & 15, lane >> 4
    row0 = tile_row + wave * 64
    col0 = cg * DOWN_COLS
    active = wave < nblk
    a2_res = _buffer(
        fx.make_view(fx.get_iter(A2) + fx.Int64(row0) * K, fx.make_layout(64 * K, 1)),
        active.select(fx.Int32(64 * K), fx.Int32(0)),
    )
    a2s_res = _buffer(A2S, KB * R * 4)
    sid_res = _buffer(SID, R * 4)
    w_res = _buffer(
        fx.make_view(
            fx.get_iter(W) + (fx.Int64(expert) * N + fx.Int64(col0)) * K,
            fx.make_layout(DOWN_COLS * K, 1),
        ),
        DOWN_COLS * K,
    )
    y_res = _buffer(
        fx.make_view(
            fx.get_iter(Y) + fx.Int64(tile_row) * N, fx.make_layout(BM * N, 1)
        ),
        nblk * (64 * N * 2),
    )
    lane_off = _pin_i32(_down_lane(lane))
    ws_row = (expert * (N // 128) + col0 // 128) * KB

    # Prologue: W2 chunks 0 / 1 -> slots 0 / 1; the wave's 64 A2 rows (whole K) and scales stay in registers.
    _dma_chunk(w_res, storage, wave, lane_off, 0, fx.Int32(0), K)
    _dma_chunk(w_res, storage, wave, lane_off, 1, fx.Int32(1), K)
    x = [
        [
            [
                _load_v4(a2_res, (16 * mb + r16) * K + kb * 128 + 32 * g + 16 * j)
                for j in range(2)
            ]
            for kb in range(KB)
        ]
        for mb in range(4)
    ]
    xs = [
        [_load_f32(a2s_res, (kb * R + row0 + 16 * mb + r16) * 4) for kb in range(KB)]
        for mb in range(4)
    ]
    tokens = [
        _load_i32(sid_res, (row0 + 16 * mb + r16) * 4) & 0xFFFFFF for mb in range(4)
    ]
    requests = [_read_scalar_g2r(WS, ws_row + kb) for kb in range(KB)]
    _wait(vmcnt=0)
    ws_first = [_scalar_ready(r) for r in requests]
    # Lane (r16, g) stores columns 8 g .. + 7 of row 16 mb + r16; padding rows (token T) store nothing.
    vo = [
        _pin_i32(
            (tokens[mb] < T).select(
                ((wave * 64 + 16 * mb + r16) * N + 8 * g) * 2, fx.Int32(_DROP)
            )
        )
        for mb in range(4)
    ]
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    b = (r16 >> 2) & 1
    w_read = [
        [
            _pin_i32(
                shared
                + (r16 >> 3) * (16 * K)
                + (2 * g + j) * 256
                + (8 * b + 4 * (b ^ v) + (r16 & 3)) * 16
            )
            for j in range(2)
        ]
        for v in range(2)
    ]
    _stage_end()
    wf_first = _read_unit(w_read, 0, 0, K)
    _wait(lgkmcnt=0)
    rocdl.sched_barrier(0)

    @flyc.jit
    def pair(
        f0,
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        f7,
        p0,
        p1,
        p2,
        p3,
        s0,
        s1,
        item_vo,
        h0,
        h1,
        h2,
        h3,
        ws0,
        ws1,
        it,
        LAST: fx.Constexpr[bool],
    ):
        """Columns 128 it .. + 127 (one weight block scale): chunk 2 it in slot 0, 2 it + 1 in slot 1, as
        units U0-U3 of 32 columns. Entry: U0's fragments in f*, chunk 2 it + 1 in flight, the previous
        pair's last M-block unretired in p* / s* / item_vo, and U2's M-block 3 words in h*.
        """
        it = fx.Int32(it)
        wf = [
            [[fx.Vector(f0), fx.Vector(f1)], [fx.Vector(f2), fx.Vector(f3)]],
            [[fx.Vector(f4), fx.Vector(f5)], [fx.Vector(f6), fx.Vector(f7)]],
        ]
        cbase = (col0 + it * 128) * 2
        item = (
            [[fx.Vector(p0), fx.Vector(p1)], [fx.Vector(p2), fx.Vector(p3)]],
            fx.Float32(s0),
            fx.Float32(s1),
            fx.Int32(item_vo),
            cbase - 64,
        )
        held = [None, None, None, [fx.Int32(h) for h in (h0, h1, h2, h3)]]
        ws = (_as_f32(ws0), _as_f32(ws1))
        scales = [
            [fx.Float32(xs[mb][kb]) * ws[kb] for kb in range(KB)] for mb in range(4)
        ]
        following = _min(it + 1, fx.Int32(ITERS - 1))
        requests = [
            _read_scalar_g2r(WS, ws_row + following * KB + kb) for kb in range(KB)
        ]
        item, held, wf = _unit(
            wf,
            x,
            scales,
            item,
            held,
            y_res,
            vo,
            cbase,
            False,
            lambda: _read_unit(w_read, 0, 1, K),
        )
        _wait(lgkmcnt=0)
        # Chunk 2 it + 1: its DMA preceded the 8 stores of the previous U3 and this U0 (in-order vmcnt).
        # After the barrier every wave has slot 0 in registers: refill it with chunk 2 it + 2.
        _wait(vmcnt=8)
        _stage_end()
        if fx.const_expr(not LAST):
            _dma_chunk(w_res, storage, wave, lane_off, 0, it * 2 + 2, K)
        item, held, wf = _unit(
            wf,
            x,
            scales,
            item,
            held,
            y_res,
            vo,
            cbase + 64,
            True,
            lambda: _read_unit(w_read, 1, 0, K),
        )
        _wait(lgkmcnt=0)
        item, held, wf = _unit(
            wf,
            x,
            scales,
            item,
            held,
            y_res,
            vo,
            cbase + 128,
            False,
            lambda: _read_unit(w_read, 1, 1, K),
        )
        _wait(lgkmcnt=0)
        if fx.const_expr(not LAST):
            # Chunk 2 it + 2 (8 stores ago) is visible after this barrier; slot 1 is in registers.
            _wait(vmcnt=8)
            _stage_end()
            _dma_chunk(w_res, storage, wave, lane_off, 1, it * 2 + 3, K)
        item, held, following_wf = _unit(
            wf,
            x,
            scales,
            item,
            held,
            y_res,
            vo,
            cbase + 192,
            True,
            _maybe(not LAST, lambda: _read_unit(w_read, 0, 0, K)),
        )
        _wait(lgkmcnt=0)
        ws_next = [_scalar_ready(r) for r in requests]
        wf = wf if following_wf is None else following_wf
        parts = item[0]
        return (
            *[wf[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)],
            parts[0][0],
            parts[0][1],
            parts[1][0],
            parts[1][1],
            item[1],
            item[2],
            vo[3],
            *held[3],
            *ws_next,
        )

    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    state0 = (
        *[wf_first[v][kb][j] for v in range(2) for kb in range(2) for j in range(2)],
        zero,
        zero,
        zero,
        zero,
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Int32(_DROP),
        *[fx.Int32(0)] * 4,
        *ws_first,
    )
    if active:
        state = state0
        for it in range(fx.Int32(0), fx.Int32(ITERS - 1), fx.Int32(1)):
            state = pair(*state, it, False)
        state = pair(*state, fx.Int32(ITERS - 1), True)
        last = ([[state[8], state[9]], [state[10], state[11]]], state[12], state[13])
        _store_pair(
            [fx.Int32(h) for h in state[15:19]],
            _retire(last),
            y_res,
            vo[3],
            (col0 + DOWN_COLS - 32) * 2,
        )
    else:
        _down_idle(w_res, storage, wave, lane_off, ITERS, K)


@flyc.kernel(known_block_size=[DOWN_THREADS, 1, 1])
def _down_direct_kernel(
    A2: fx.Tensor,
    W: fx.Tensor,
    A2S: fx.Tensor,
    WS: fx.Tensor,
    TILES: fx.Tensor,
    COUNT: fx.Tensor,
    SID: fx.Tensor,
    Y: fx.Tensor,
    R: fx.Int32,
    T: fx.Int32,
    CG0: fx.Int32,
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    GROUPS: fx.Constexpr[int],
):
    """K = 256 down GEMM: task = (compact tile, 1024 columns of groups CG0 .. CG0 + GROUPS - 1)."""
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, 2 * DOWN_CHUNK * K, 16])
        .peek()
        .view(fx.make_layout(2 * DOWN_CHUNK * K, 1))
    )
    body = _down_direct_body
    order = _xcd_order
    block = fx.Int32(gpu.block_id("x"))
    tiles = _scalar_ready(_read_scalar_g2r(COUNT, 0)) * GROUPS
    if block < tiles:
        v = order(block, tiles)
        tile = v // GROUPS
        cg = CG0 + v - tile * GROUPS
        row0, rows, expert = _tile(TILES, tile)
        body(A2, W, A2S, WS, SID, Y, storage, R, T, row0, rows >> 6, cg, expert, N, K)


def _attrs():
    return {
        "rocdl.waves_per_eu": 2,
        "llvm.target_features": ir.Attribute.parse(
            '#llvm.target_features<["-packed-fp32-ops"]>'
        ),
    }


@flyc.jit
def launch_gate_up(
    X: fx.Tensor,
    W: fx.Tensor,
    XS: fx.Tensor,
    WS: fx.Tensor,
    SID: fx.Tensor,
    TILES: fx.Tensor,
    COUNT: fx.Tensor,
    SW: fx.Tensor,
    A2: fx.Tensor,
    A2S: fx.Tensor,
    T: fx.Int32,
    R: fx.Int32,
    GRID: fx.Int32,
    I: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stream: fx.Stream,
):
    _gate_up_kernel(
        X, W, XS, WS, SID, TILES, COUNT, SW, A2, A2S, T, R, I, K, value_attrs=_attrs()
    ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


@flyc.jit
def launch_down_direct(
    A2: fx.Tensor,
    W: fx.Tensor,
    A2S: fx.Tensor,
    WS: fx.Tensor,
    TILES: fx.Tensor,
    COUNT: fx.Tensor,
    SID: fx.Tensor,
    Y: fx.Tensor,
    R: fx.Int32,
    T: fx.Int32,
    CG0: fx.Int32,
    GRID: fx.Int32,
    N: fx.Constexpr[int],
    K: fx.Constexpr[int],
    GROUPS: fx.Constexpr[int],
    stream: fx.Stream,
):
    _down_direct_kernel(
        A2,
        W,
        A2S,
        WS,
        TILES,
        COUNT,
        SID,
        Y,
        R,
        T,
        CG0,
        N,
        K,
        GROUPS,
        value_attrs=_attrs(),
    ).launch(grid=(GRID, 1, 1), block=(DOWN_THREADS, 1, 1), stream=stream)


# ---- route sort: the rows of aiter's moe_sorting (block 64), its compact tiles and inv, in 3 kernels ------
#
# Expert runs in expert order, empty experts skipped, each padded to 64 rows (token T, slot TOPK, weight 0);
# an expert's routes in token order. Block b of count / scatter = tokens 256 b .. + 255, one per thread.
#   count:   COUNTS[b, e] = routes of block b to expert e (LDS atomics).
#   scan:    one block, thread e: COUNTS[b, e] -> routes of blocks before b to e; COUNTS[NBP + 0 / 1 / 2, e] =
#            first row / routes / first tile of e; COUNT[0] = tiles.
#   scatter: a route's row = first row of e + COUNTS[b, e] + routes of earlier tokens of block b to e (LDS
#            token bitmasks); SID / SW rows and inv[t, k]. Then one expert per wave: its padding rows and
#            tiles, one per lane.
SORT_THREADS = 256
_SORT_WORDS = SORT_THREADS // 32  # bitmask words per expert


def _lds_atomic(op, address, value):
    llvm.atomicrmw(
        op,
        _lds(address),
        fx.Int32(value).ir_value(),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
    )


def _popc(value):
    return fx.Int32(
        llvm.call_intrinsic(
            fx.Int32.ir_type, "llvm.ctpop.i32", [fx.Int32(value).ir_value()], [], []
        )
    )


@flyc.kernel(known_block_size=[SORT_THREADS, 1, 1])
def _route_count_kernel(
    IDS: fx.Tensor,
    COUNTS: fx.Tensor,
    T: fx.Int32,
    E: fx.Constexpr[int],
    TOPK: fx.Constexpr[int],
    NBP: fx.Constexpr[int],
):
    """COUNTS[b * E + e] = routes of block b's tokens to expert e."""
    _helpers = (_buffer, _load_v4, _store, _lds, _lds_atomic)
    hist = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, SORT_THREADS, 16])
        .peek()
        .view(fx.make_layout(SORT_THREADS, 1))
    )
    lds = fx.Int32(fx.ptrtoint(fx.get_iter(hist)))
    tid = fx.Int32(gpu.thread_id("x"))
    block = fx.Int32(gpu.block_id("x"))
    hist[tid] = fx.Int32(0)
    gpu.barrier()
    t = block * SORT_THREADS + tid
    if t < T:
        ids = _buffer(IDS, T * (TOPK * 4))
        for j in range_constexpr(TOPK // 4):
            quad = _load_v4(ids, t * (TOPK * 4) + j * 16)
            for i in range_constexpr(4):
                _lds_atomic(llvm.AtomicBinOp.add, lds + fx.Int32(quad[i]) * 4, 1)
    gpu.barrier()
    if tid < E:
        _store(fx.Int32(hist[tid]), _buffer(COUNTS, E * NBP * 4), (block * E + tid) * 4)


@flyc.kernel(known_block_size=[SORT_THREADS, 1, 1])
def _route_scan_kernel(
    COUNTS: fx.Tensor,
    COUNT: fx.Tensor,
    NB: fx.Int32,
    E: fx.Constexpr[int],
    NBP: fx.Constexpr[int],
    BLOCK: fx.Constexpr[int],
):
    """Thread e: COUNTS[b, e] -> routes of blocks before b to expert e; COUNTS[NBP + 0 / 1 / 2, e] = first
    sorted row / routes / first compact tile of e (rows padded to BLOCK, tiles of 256 rows and one tail, both
    in expert order); COUNT[0] = tiles."""
    _helpers = (_buffer, _load_i32, _store)
    shared = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, 2 * SORT_THREADS, 16])
        .peek()
        .view(fx.make_layout(2 * SORT_THREADS, 1))
    )
    e = fx.Int32(gpu.thread_id("x"))
    live = e < E
    counts = _buffer(COUNTS, (NBP + 3) * E * 4)
    total = fx.Int32(0)
    for c in range_constexpr(NBP // 64):
        # Blocks >= NB count 0; threads >= E do not store.
        values = [_load_i32(counts, ((c * 64 + i) * E + e) * 4) for i in range(64)]
        for i in range_constexpr(64):
            b = c * 64 + i
            _store(total, counts, live.select((b * E + e) * 4, fx.Int32(_DROP)))
            total = total + (live & (NB > b)).select(values[i], fx.Int32(0))
    padded = (total + BLOCK - 1) // BLOCK * BLOCK
    per = BM // BLOCK
    tiles = (padded // BLOCK + per - 1) // per
    shared[e] = padded
    shared[SORT_THREADS + e] = tiles
    gpu.barrier()
    # Inclusive scans of the padded rows and the tiles over the experts.
    for shift in range_constexpr(SORT_THREADS.bit_length() - 1):
        d = 1 << shift
        peer = (e >= d).select(e - d, e)
        rows_sum = shared[e] + (e >= d).select(shared[peer], fx.Int32(0))
        tiles_sum = shared[SORT_THREADS + e] + (e >= d).select(
            shared[SORT_THREADS + peer], fx.Int32(0)
        )
        gpu.barrier()
        shared[e] = rows_sum
        shared[SORT_THREADS + e] = tiles_sum
        gpu.barrier()
    first = fx.Int32(shared[e]) - padded
    tile0 = fx.Int32(shared[SORT_THREADS + e]) - tiles
    for i, value in enumerate((first, total, tile0)):
        _store(value, counts, live.select(((NBP + i) * E + e) * 4, fx.Int32(_DROP)))
    if e == 0:
        _store(fx.Int32(shared[2 * SORT_THREADS - 1]), _buffer(COUNT, 4), fx.Int32(0))


@flyc.kernel(known_block_size=[SORT_THREADS, 1, 1])
def _route_scatter_kernel(
    IDS: fx.Tensor,
    W: fx.Tensor,
    COUNTS: fx.Tensor,
    SID: fx.Tensor,
    SW: fx.Tensor,
    INV: fx.Tensor,
    TILES: fx.Tensor,
    T: fx.Int32,
    R: fx.Int32,
    CAP: fx.Int32,
    NB: fx.Int32,
    E: fx.Constexpr[int],
    TOPK: fx.Constexpr[int],
    NBP: fx.Constexpr[int],
    BLOCK: fx.Constexpr[int],
):
    """Route (t, k) of block b to expert e -> row first(e) + COUNTS[b, e] + tokens of block b before t routed
    to e: SID = k << 24 | t, SW = its weight, INV[t, k] = row. Wave w of block b then writes the padding rows
    (SID = TOPK << 24 | T, SW = 0) and the tiles (TILES[2 i] = first row, TILES[2 i + 1] = e | rows / 64 << 16)
    of experts e = 4 b + w + 4 NB j."""
    _helpers = (
        _buffer,
        _load_i32,
        _load_v4,
        _store,
        _lds,
        _lds_atomic,
        _lds_load_v4,
        _lds_store_v4,
        _popc,
        _min,
    )
    masks = SORT_THREADS * _SORT_WORDS
    shared = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, masks + SORT_THREADS, 16])
        .peek()
        .view(fx.make_layout(masks + SORT_THREADS, 1))
    )
    lds = fx.Int32(fx.ptrtoint(fx.get_iter(shared)))
    tid = fx.Int32(gpu.thread_id("x"))
    block = fx.Int32(gpu.block_id("x"))
    zero = fx.Vector.filled(4, 0, fx.Int32)
    for h in range_constexpr(_SORT_WORDS // 4):
        _lds_store_v4(lds + tid * (_SORT_WORDS * 4) + h * 16, zero)
    counts = _buffer(COUNTS, (NBP + 3) * E * 4)
    shared[masks + tid] = _load_i32(counts, (block * E + tid) * 4) + _load_i32(
        counts, (NBP * E + tid) * 4
    )
    gpu.barrier()
    t = block * SORT_THREADS + tid
    word, bit = tid >> 5, tid & 31
    if t < T:
        ids = _buffer(IDS, T * (TOPK * 4))
        for j in range_constexpr(TOPK // 4):
            quad = _load_v4(ids, t * (TOPK * 4) + j * 16)
            for i in range_constexpr(4):
                _lds_atomic(
                    llvm.AtomicBinOp._or,
                    lds + (fx.Int32(quad[i]) * _SORT_WORDS + word) * 4,
                    fx.Int32(1) << bit,
                )
    gpu.barrier()
    if t < T:
        ids = _buffer(IDS, T * (TOPK * 4))
        weights = _buffer(W, T * (TOPK * 4))
        sid = _buffer(SID, R * 4)
        sw = _buffer(SW, R * 4)
        rows = []
        for j in range_constexpr(TOPK // 4):
            quad = _load_v4(ids, t * (TOPK * 4) + j * 16)
            wquad = _load_v4(weights, t * (TOPK * 4) + j * 16)
            for i in range_constexpr(4):
                k = j * 4 + i
                expert = fx.Int32(quad[i])
                words = []
                for h in range_constexpr(_SORT_WORDS // 4):
                    quad_w = _lds_load_v4(lds + expert * (_SORT_WORDS * 4) + h * 16)
                    words += [fx.Int32(quad_w[v]) for v in range(4)]
                rank = fx.Int32(0)
                for w in range_constexpr(_SORT_WORDS):
                    below = (word > w).select(
                        fx.Int32(-1),
                        (word == w).select((fx.Int32(1) << bit) - 1, fx.Int32(0)),
                    )
                    rank = rank + _popc(words[w] & below)
                row = fx.Int32(shared[masks + expert]) + rank
                _store(fx.Int32(k << 24) | t, sid, row * 4)
                _store(fx.Int32(wquad[i]), sw, row * 4)
                rows.append(row)
        inv = _buffer(INV, T * (TOPK * 4))
        for j in range_constexpr(TOPK // 4):
            _store(
                fx.Vector.from_elements(rows[4 * j : 4 * j + 4], fx.Int32),
                inv,
                t * (TOPK * 4) + j * 16,
            )
    lane = tid & 63
    sid = _buffer(SID, R * 4)
    sw = _buffer(SW, R * 4)
    out = _buffer(TILES, CAP * 8)
    pad = fx.Int32(TOPK << 24) | T
    per = BM // BLOCK
    for e in range(block * 4 + (tid >> 6), fx.Int32(E), NB * 4):
        expert = fx.Int32(e)
        first = _load_i32(counts, (NBP * E + expert) * 4)
        total = _load_i32(counts, ((NBP + 1) * E + expert) * 4)
        tile0 = _load_i32(counts, ((NBP + 2) * E + expert) * 4)
        padded = (total + BLOCK - 1) // BLOCK * BLOCK
        r = first + total + lane
        offset = (r < first + padded).select(r * 4, fx.Int32(_DROP))
        _store(pad, sid, offset)
        _store(fx.Int32(0), sw, offset)
        tiles = (padded // BLOCK + per - 1) // per
        for i0 in range(fx.Int32(0), tiles, fx.Int32(64)):
            i = fx.Int32(i0) + lane
            rows = _min(fx.Int32(BM), padded - i * BM)
            offset = (i < tiles).select((tile0 + i) * 8, fx.Int32(_DROP))
            _store(fx.Int32(first + i * BM), out, offset)
            _store(fx.Int32(expert | ((rows >> 6) << 16)), out, offset + 4)


@flyc.jit
def launch_route_count(
    IDS: fx.Tensor,
    COUNTS: fx.Tensor,
    T: fx.Int32,
    GRID: fx.Int32,
    E: fx.Constexpr[int],
    TOPK: fx.Constexpr[int],
    NBP: fx.Constexpr[int],
    stream: fx.Stream,
):
    _route_count_kernel(IDS, COUNTS, T, E, TOPK, NBP).launch(
        grid=(GRID, 1, 1), block=(SORT_THREADS, 1, 1), stream=stream
    )


@flyc.jit
def launch_route_scan(
    COUNTS: fx.Tensor,
    COUNT: fx.Tensor,
    NB: fx.Int32,
    E: fx.Constexpr[int],
    NBP: fx.Constexpr[int],
    BLOCK: fx.Constexpr[int],
    stream: fx.Stream,
):
    _route_scan_kernel(COUNTS, COUNT, NB, E, NBP, BLOCK).launch(
        grid=(1, 1, 1), block=(SORT_THREADS, 1, 1), stream=stream
    )


@flyc.jit
def launch_route_scatter(
    IDS: fx.Tensor,
    W: fx.Tensor,
    COUNTS: fx.Tensor,
    SID: fx.Tensor,
    SW: fx.Tensor,
    INV: fx.Tensor,
    TILES: fx.Tensor,
    T: fx.Int32,
    R: fx.Int32,
    CAP: fx.Int32,
    NB: fx.Int32,
    E: fx.Constexpr[int],
    TOPK: fx.Constexpr[int],
    NBP: fx.Constexpr[int],
    BLOCK: fx.Constexpr[int],
    stream: fx.Stream,
):
    _route_scatter_kernel(
        IDS, W, COUNTS, SID, SW, INV, TILES, T, R, CAP, NB, E, TOPK, NBP, BLOCK
    ).launch(grid=(NB, 1, 1), block=(SORT_THREADS, 1, 1), stream=stream)


# ---- fused top-k sum + reduce-scatter (sequence-parallel layout, 8 ranks, HIP IPC) ----------------------
#
# Rank s owns token rows [off[s], off[s] + size[s]) of the gathered batch (sizes may differ). Every rank holds
# Y (its I-slice route rows, route weight folded in) and INV [T, 8] (row of route (t, k), -1: none).
# out (rank s, row u < size[s]) = bf16(sum over ranks in rank order of bf16(sum_k Y[INV[off[s] + u, k]])),
# sums in FP32. Unit u = row u of every shard. Persistent grid of RS_GRID 512-thread WGs (all resident),
# units u = block + j * grid below the largest shard; (u, s) with u >= size[s] are skipped.
#   RS: waves 0-3 sum the 8 route rows of token (unit u, destination s) and write a quarter row into an LDS
#      ring; waves 4-5 copy two quarters each to rank s's inbox[parity][rank][u] with write-through stores and,
#      after a unit, set flag[rank][u][quarter] = epoch on its destinations (monotonic, never reset).
#      s rotates by block // 8, so the WGs writing to one destination at a time write adjacent rows.
#   reduce: wait for this rank's rows' flags, sum the 8 inbox rows in rank order (L2-bypassing loads),
#      _RS_UNITS units per step.
# Push never waits on another rank (all WGs are resident). Every consumer,
# including an empty owner, waits for peer progress so no rank gets more than
# one epoch ahead; the inbox alternates by parity. Inbox rows use src * size[s]
# + u, but flags use src * RCAP + u so changing shard sizes cannot alias flags
# belonging to a different source in an overlapping epoch.

# Fewer pushing WGs congest xGMI less, more WGs sum faster (32K: 64 / 80 / 96 / 112 -> 1035 / 988 / 1036 / 1168 us).
RS_GRID = 80
_RS_THREADS = 512
_RS_SLOTS = 4  # LDS ring rows
_RS_UNITS = 4  # reduce units per step
_ROW = 4096 * 2  # bytes per hidden row
_QUARTER = 4096 // 8 // 4  # 16-byte vectors per row quarter (one sum wave)
# Buffer cache policy: nt; sc0 | sc1 (write-through stores, L2-bypassing loads); sc1 (system-scope atomic);
# sc0 | sc1 | nt (inbox stores).
_AUX_NT, _AUX_SYS, _AUX_SC1, _AUX_SYS_NT = 2, 17, 16, 19


def symmetric_buffer_bytes(rcap: int) -> int:
    """[2 parity][8 src][rcap][4096] bf16 inbox + [8 src][rcap][4 quarters] int32 flags."""
    return 2 * 8 * rcap * _ROW + 8 * rcap * 16


def _aux(bits):
    return ir.IntegerAttr.get(fx.Int32.ir_type, bits)


def _rsrc(address, size_bytes):
    """Raw buffer descriptor at a 64-bit address; loads at or past size_bytes return zero."""
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), fx.Int64(address).ir_value())
    return rocdl.make_buffer_rsrc(
        ir.Type.parse("!llvm.ptr<8>"),
        pointer,
        fx.Int16(0).ir_value(),
        fx.Int64(size_bytes).ir_value(),
        fx.Int32(0x27000).ir_value(),
    )


def _load_v4_aux(resource, voffset, bits):
    vt = ir.VectorType.get([4], fx.Int32.ir_type)
    return fx.Vector(
        rocdl.raw_ptr_buffer_load(
            vt,
            resource,
            fx.Int32(voffset).ir_value(),
            fx.Int32(0).ir_value(),
            aux=_aux(bits),
        )
    )


def _store_v4_aux(value, resource, voffset, bits):
    rocdl.raw_ptr_buffer_store(
        fx.Vector(value).ir_value(),
        resource,
        fx.Int32(voffset).ir_value(),
        fx.Int32(0).ir_value(),
        aux=_aux(bits),
    )


def _lds(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _global(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.Int64(address).ir_value())


def _lds_load_v4(address):
    vt = ir.VectorType.get([4], fx.Int32.ir_type)
    return fx.Vector(llvm.LoadOp(vt, _lds(address), alignment=16).result)


def _lds_store_v4(address, value):
    llvm.StoreOp(fx.Vector(value).ir_value(), _lds(address), alignment=16)


def _counter_load(address):
    # LDS ring counter; acquire keeps the ring reads after it (LDS only: lgkmcnt).
    return fx.Int32(
        llvm.LoadOp(
            fx.Int32.ir_type,
            _lds(address),
            alignment=4,
            ordering=llvm.AtomicOrdering.acquire,
            syncscope="workgroup-one-as",
        ).result
    )


def _counter_store(address, value):
    # release: this wave's earlier ring accesses complete first (LDS only: lgkmcnt).
    llvm.StoreOp(
        fx.Int32(value).ir_value(),
        _lds(address),
        alignment=4,
        ordering=llvm.AtomicOrdering.release,
        syncscope="workgroup-one-as",
    )


def _flag_load(resource, voffset):
    # buffer_load_dword sc0 sc1 of a flag written by a peer GPU. The atomic variant of the buffer load
    # intrinsic is not read-only to LLVM, so a spin loop re-reads it every iteration.
    return fx.Int32(
        llvm.call_intrinsic(
            fx.Int32.ir_type,
            "llvm.amdgcn.raw.ptr.atomic.buffer.load",
            [
                resource,
                fx.Int32(voffset).ir_value(),
                fx.Int32(0).ir_value(),
                fx.Int32(_AUX_SYS).ir_value(),
            ],
            [],
            [],
        )
    )


def _flag_swap(resource, voffset, value):
    # buffer_atomic_swap sc1 (dropped at or past the resource size).
    llvm.call_intrinsic(
        fx.Int32.ir_type,
        "llvm.amdgcn.raw.ptr.buffer.atomic.swap",
        [
            fx.Int32(value).ir_value(),
            resource,
            fx.Int32(voffset).ir_value(),
            fx.Int32(0).ir_value(),
            fx.Int32(_AUX_SC1).ir_value(),
        ],
        [],
        [],
    )


def _scalar_i32x8(address):
    # Uniform 32-byte aligned address, read-only during the kernel: s_load_dwordx8.
    vt = ir.VectorType.get([8], fx.Int32.ir_type)
    return fx.Vector(
        llvm.LoadOp(vt, _global(address), alignment=32, invariant=True).result
    )


def _pick(index, values):
    out = values[0]
    for i in range(1, len(values)):
        out = (index == i).select(values[i], out)
    return out


def _sum_bf16(vectors):
    """FP32 sum (list order, from +0.0) of bf16x8 vectors (4 x i32 each) -> bf16x8 (RNE)."""
    lo = fx.Vector.filled(4, 0.0, fx.Float32)
    hi = fx.Vector.filled(4, 0.0, fx.Float32)
    for v in vectors:
        bits = fx.Vector(v).bitcast(fx.Uint32)
        lo = lo + (bits << 16).bitcast(fx.Float32)
        hi = hi + (bits & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32)
    lo, hi = lo.bitcast(fx.Uint32), hi.bitcast(fx.Uint32)
    lo = lo + fx.Uint32(0x7FFF) + ((lo >> 16) & fx.Uint32(1))
    hi = hi + fx.Uint32(0x7FFF) + ((hi >> 16) & fx.Uint32(1))
    return ((lo >> 16) | (hi & fx.Uint32(0xFFFF0000))).bitcast(fx.Int32)


@flyc.kernel(known_block_size=[_RS_THREADS, 1, 1])
def _sum_rs_kernel(
    Y: fx.Tensor,
    INV: fx.Tensor,
    OUT: fx.Tensor,
    P0: fx.Int64,
    P1: fx.Int64,
    P2: fx.Int64,
    P3: fx.Int64,
    P4: fx.Int64,
    P5: fx.Int64,
    P6: fx.Int64,
    P7: fx.Int64,
    O0: fx.Int32,
    O1: fx.Int32,
    O2: fx.Int32,
    O3: fx.Int32,
    O4: fx.Int32,
    O5: fx.Int32,
    O6: fx.Int32,
    O7: fx.Int32,
    S0: fx.Int32,
    S1: fx.Int32,
    S2: fx.Int32,
    S3: fx.Int32,
    S4: fx.Int32,
    S5: fx.Int32,
    S6: fx.Int32,
    S7: fx.Int32,
    RMAX: fx.Int32,
    RCAP: fx.Int32,
    RANK: fx.Int32,
    EPOCH: fx.Int32,
    G: fx.Int32,
):
    _helpers = (
        _aux,
        _rsrc,
        _load_v4_aux,
        _store_v4_aux,
        _lds,
        _global,
        _lds_load_v4,
        _lds_store_v4,
        _counter_load,
        _counter_store,
        _flag_load,
        _flag_swap,
        _scalar_i32x8,
        _pick,
        _sum_bf16,
        _stage_end,
        _uniform,
        _wait,
        _buffer,
    )
    peers = [P0, P1, P2, P3, P4, P5, P6, P7]
    offs = [O0, O1, O2, O3, O4, O5, O6, O7]
    sizes = [S0, S1, S2, S3, S4, S5, S6, S7]
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, _RS_SLOTS * _ROW + 32, 16])
        .peek()
        .view(fx.make_layout(_RS_SLOTS * _ROW + 32, 1))
    )
    lds = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    tid = fx.Int32(gpu.thread_id("x"))
    block = fx.Int32(gpu.block_id("x"))
    wave = _uniform(tid >> 6)
    lane = tid & 63
    q = wave & 3
    prod = lds + _RS_SLOTS * _ROW + q * 4
    cons = prod + 16
    inbox_off = fx.Int64(EPOCH & 1) * fx.Int64(RCAP) * (8 * _ROW)
    flags_off = fx.Int64(RCAP) * (2 * 8 * _ROW)
    flags_bytes = RCAP * (8 * 16)
    rows = (RMAX - block + G - 1) // G * 8
    y_base = fx.Int64(fx.ptrtoint(fx.get_iter(Y)))
    inv_base = fx.Int64(fx.ptrtoint(fx.get_iter(INV)))
    lane_off = (q * _QUARTER + lane) * 16

    def dest(seq):
        # seq -> (unit, destination rank, its shard size); valid if u < size
        u = block + (seq >> 3) * G
        s = (RANK + (block >> 3) + (seq & 7)) & 7
        size = _pick(s, sizes)
        return u, s, size

    if tid < 8:
        _counter_store(lds + _RS_SLOTS * _ROW + tid * 4, fx.Int32(0))
    _wait(lgkmcnt=0)
    _stage_end()

    if wave < 4:
        # Sum role: route rows prefetched one row ahead -> FP32 sum -> bf16 -> LDS quarter.
        def fetch(seq):
            u, s, size = dest(seq)
            ok = (seq < rows) & (u < size)
            t = ok.select(_pick(s, offs) + u, fx.Int32(0))
            routes = _scalar_i32x8(inv_base + fx.Int64(t) * 32)
            loads = []
            for k in range_constexpr(8):
                row = fx.Int32(routes[k])
                live = ok & (row >= 0)
                r = _rsrc(
                    y_base + fx.Int64(live.select(row, fx.Int32(0))) * _ROW,
                    live.select(fx.Int32(0x7FFFFFFF), fx.Int32(0)),
                )
                loads += [
                    _load_v4_aux(r, lane_off + h * 1024, _AUX_NT) for h in range(2)
                ]
            return loads

        def produce(current, cur):
            u, s, size = dest(cur)
            if u < size:
                words = [_sum_bf16(current[h::2]) for h in range(2)]
                while _counter_load(cons) < cur - (_RS_SLOTS - 1):
                    rocdl.s_sleep(0)
                slot = lds + (cur & (_RS_SLOTS - 1)) * _ROW + lane_off
                for h in range_constexpr(2):
                    _lds_store_v4(slot + h * 1024, words[h])
            _counter_store(prod, cur + 1)

        # Two rows per iteration with ping-pong buffers: the loop-carried rows are reloaded
        # only after their last use, so no register copies wait for loads in flight.
        @flyc.jit
        def sum_pair(
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, cur
        ):
            current = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
            current += [a14, a15]
            cur = fx.Int32(cur)
            following = fetch(cur + 1)
            produce(current, cur)
            current = fetch(cur + 2)
            produce(following, cur + 1)
            return tuple(current)

        state = tuple(fetch(fx.Int32(0)))
        for cur in range(fx.Int32(0), rows, fx.Int32(2)):
            state = sum_pair(*state, cur)
    else:
        if wave < 6:
            # Push role: two LDS quarters -> rank s's inbox (write-through); per unit, flags on its destinations.
            q0 = (wave - 4) * 2
            for seq in range(fx.Int32(0), rows, fx.Int32(1)):
                u, s, size = dest(seq)
                if u < size:
                    v = []
                    for i in range_constexpr(2):
                        counter = lds + _RS_SLOTS * _ROW + (q0 + i) * 4
                        while _counter_load(counter) < seq + 1:
                            rocdl.s_sleep(0)
                        slot = (
                            lds
                            + (seq & (_RS_SLOTS - 1)) * _ROW
                            + ((q0 + i) * _QUARTER + lane) * 16
                        )
                        v += [_lds_load_v4(slot + h * 1024) for h in range(2)]
                    for i in range_constexpr(2):
                        _counter_store(
                            lds + _RS_SLOTS * _ROW + (q0 + i) * 4 + 16, seq + 1
                        )
                    r = _rsrc(
                        _pick(s, peers)
                        + inbox_off
                        + (fx.Int64(RANK) * fx.Int64(size) + fx.Int64(u)) * _ROW,
                        fx.Int32(_ROW),
                    )
                    for i in range_constexpr(2):
                        for h in range_constexpr(2):
                            _store_v4_aux(
                                v[2 * i + h],
                                r,
                                ((q0 + i) * _QUARTER + lane) * 16 + h * 1024,
                                _AUX_SYS_NT,
                            )
                else:
                    for i in range_constexpr(2):
                        _counter_store(
                            lds + _RS_SLOTS * _ROW + (q0 + i) * 4 + 16, seq + 1
                        )
                if (seq & 7) == 7:
                    # This wave's write-through inbox stores are acknowledged before its flags.
                    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="workgroup")
                    _wait(vmcnt=0)
                    if lane == 0:
                        for d in range_constexpr(8):
                            live = u < sizes[d]
                            flag_rsrc = _rsrc(
                                peers[d] + flags_off,
                                live.select(flags_bytes, fx.Int32(0)),
                            )
                            for i in range_constexpr(2):
                                _flag_swap(
                                    flag_rsrc,
                                    ((RANK * RCAP + u) * 4 + q0 + i) * 4,
                                    EPOCH,
                                )
    _stage_end()

    # Reduce this rank's rows, _RS_UNITS units per step (units of one step are G apart).
    own = _pick(RANK, sizes)
    own_base = _pick(RANK, peers)
    inbox = _rsrc(own_base + inbox_off, fx.Int32(0x7FFFFFFF))
    flags = _rsrc(own_base + flags_off, flags_bytes)
    out = _buffer(OUT, own * _ROW)
    if own == 0:
        # An empty owner still pushes rows for its peers. Wait for row zero at
        # a nonempty destination before advancing, otherwise it could lap a
        # slow consumer and overwrite that consumer's parity buffer.
        leader = fx.Int32(0)
        for d in range_constexpr(8):
            leader = (sizes[d] > 0).select(fx.Int32(d), leader)
        peer_flags = _rsrc(_pick(leader, peers) + flags_off, flags_bytes)
        if tid < 32:
            flag = (((tid >> 2) * RCAP) * 4 + (tid & 3)) * 4
            while _flag_load(peer_flags, flag) < EPOCH:
                rocdl.s_sleep(1)
        _stage_end()
    for u0 in range(block, own, G * _RS_UNITS):
        if tid < 32 * _RS_UNITS:
            uw = u0 + (tid >> 5) * G
            if uw < own:
                flag = ((((tid >> 2) & 7) * RCAP + uw) * 4 + (tid & 3)) * 4
                while _flag_load(flags, flag) < EPOCH:
                    rocdl.s_sleep(1)
            # Inbox reads stay after the flags (they bypass the caches: sc0 sc1).
            llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")
        _stage_end()
        rows_in = []
        for k in range_constexpr(_RS_UNITS):
            u = u0 + k * G
            row = (u < own).select(u, fx.Int32(0))
            rows_in.append(
                [
                    _load_v4_aux(inbox, (src * own + row) * _ROW + tid * 16, _AUX_SYS)
                    for src in range(8)
                ]
            )
        # Units past the shard store past the end of OUT (dropped).
        for k in range_constexpr(_RS_UNITS):
            _store_v4_aux(_sum_bf16(rows_in[k]), out, (u0 + k * G) * _ROW + tid * 16, 0)


@flyc.jit
def launch_sum_rs(
    Y: fx.Tensor,
    INV: fx.Tensor,
    OUT: fx.Tensor,
    P0: fx.Int64,
    P1: fx.Int64,
    P2: fx.Int64,
    P3: fx.Int64,
    P4: fx.Int64,
    P5: fx.Int64,
    P6: fx.Int64,
    P7: fx.Int64,
    O0: fx.Int32,
    O1: fx.Int32,
    O2: fx.Int32,
    O3: fx.Int32,
    O4: fx.Int32,
    O5: fx.Int32,
    O6: fx.Int32,
    O7: fx.Int32,
    S0: fx.Int32,
    S1: fx.Int32,
    S2: fx.Int32,
    S3: fx.Int32,
    S4: fx.Int32,
    S5: fx.Int32,
    S6: fx.Int32,
    S7: fx.Int32,
    RMAX: fx.Int32,
    RCAP: fx.Int32,
    RANK: fx.Int32,
    EPOCH: fx.Int32,
    G: fx.Int32,
    stream: fx.Stream,
):
    _sum_rs_kernel(
        Y,
        INV,
        OUT,
        P0,
        P1,
        P2,
        P3,
        P4,
        P5,
        P6,
        P7,
        O0,
        O1,
        O2,
        O3,
        O4,
        O5,
        O6,
        O7,
        S0,
        S1,
        S2,
        S3,
        S4,
        S5,
        S6,
        S7,
        RMAX,
        RCAP,
        RANK,
        EPOCH,
        G,
    ).launch(grid=(G, 1, 1), block=(_RS_THREADS, 1, 1), stream=stream)


# ---- host ------------------------------------------------------------------------------------------------

FP8 = torch.float8_e4m3fnuz
# Experts padded to 64 sorted rows: 256-row GEMM tiles plus one 64..192-row tail per expert.
COMPACT_BLOCK = 64
HIDDEN, INTER, TOPK, RANKS = 4096, 256, 8, 8
_compiled = {}
_buffers = {}


def _run(key, launcher, *args):
    compiled = _compiled.get(key)
    if compiled is None:
        # flyc.compile also launches this first call.
        with flydsl_disk_cache(__file__):
            _compiled[key] = flyc.compile(launcher, *args)
    else:
        compiled(*args)


def _workspace(device, rows, inter, hidden):
    """Flat A2 (FP8 bytes), A2S (FP32) and Y (BF16) views for `rows` sorted rows; grown, never shrunk."""
    need = (rows * inter, (inter // 128) * rows, rows * hidden)
    have = _buffers.get(device)
    if have is None or any(h.numel() < n for h, n in zip(have, need)):
        have = _buffers[device] = (
            torch.empty(need[0], dtype=torch.uint8, device=device),
            torch.empty(need[1], dtype=torch.float32, device=device),
            torch.empty(need[2], dtype=torch.bfloat16, device=device),
        )
    return tuple(h[:n] for h, n in zip(have, need))


def _route_sort(topk_ids, topk_weights, experts, rows, cap):
    """The rows of aiter's moe_sorting (block COMPACT_BLOCK), their compact tiles and inv: (SID [rows],
    SW [rows], INV [T, TOPK], TILES [2 cap], COUNT [1]); rows past the padded routes are not written.
    """
    tokens = topk_ids.shape[0]
    device = topk_ids.device
    stream = torch.cuda.current_stream(device)
    blocks = (tokens + SORT_THREADS - 1) // SORT_THREADS
    nbp = (blocks + 63) // 64 * 64
    i32 = torch.int32
    counts = torch.empty((nbp + 3) * experts, dtype=i32, device=device)
    sid = torch.empty(rows, dtype=i32, device=device)
    sw = torch.empty(rows, dtype=torch.float32, device=device)
    inv = torch.empty((tokens, TOPK), dtype=i32, device=device)
    tiles = torch.empty(2 * cap, dtype=i32, device=device)
    count = torch.empty(1, dtype=i32, device=device)
    ids, weights = topk_ids.view(-1), topk_weights.view(-1)
    _run(
        ("route_count", device, experts, nbp),
        launch_route_count,
        ids,
        counts,
        tokens,
        blocks,
        experts,
        TOPK,
        nbp,
        stream,
    )
    _run(
        ("route_scan", device, experts, nbp),
        launch_route_scan,
        counts,
        count,
        blocks,
        experts,
        nbp,
        COMPACT_BLOCK,
        stream,
    )
    _run(
        ("route_scatter", device, experts, nbp),
        launch_route_scatter,
        ids,
        weights,
        counts,
        sid,
        sw,
        inv.view(-1),
        tiles,
        tokens,
        rows,
        cap,
        blocks,
        experts,
        TOPK,
        nbp,
        COMPACT_BLOCK,
        stream,
    )
    return sid, sw, inv, tiles, count


def moe_route_rows(x8, xs, w13, w2, topk_weights, topk_ids, w13_scale, w2_scale):
    """Route rows of a TP MoE with I = 256 per rank: (Y [rows, H] BF16 in sorted order with the route weight
    folded in, inv [T, topk] int32 = row of route (t, k)), so out[t] = sum_k Y[inv[t, k]].

    x8: [T, H] FP8 E4M3FNUZ; xs: [H / 128, T] FP32 (aiter's transposed per-1x128 activation scale); w13:
    [E, 2 I, H] FP8 (gate rows first), w2: [E, H, I] FP8, both aiter shuffle_weight(16, 16); w13_scale:
    [E, 2 I / 128, H / 128], w2_scale: [E, H / 128, I / 128] FP32; topk_ids / topk_weights: [T, topk]
    int32 / FP32. Y is a workspace view, valid until the next call.
    """
    tokens, hidden = x8.shape
    experts, inter = w13.shape[0], w2.shape[2]
    # Row capacity of aiter's moe_sorting: every expert's run padded to COMPACT_BLOCK rows.
    rows = tokens * TOPK + experts * COMPACT_BLOCK - TOPK
    per = BM // COMPACT_BLOCK
    cap = ((rows + COMPACT_BLOCK - 1) // COMPACT_BLOCK + per - 1) // per + experts
    sid, sw, inv, tiles, count = _route_sort(topk_ids, topk_weights, experts, rows, cap)
    device = x8.device
    stream = torch.cuda.current_stream(device)
    a2, a2s, y = _workspace(device, rows, inter, hidden)
    u8 = torch.uint8
    _run(
        ("gate_up", device, inter, hidden),
        launch_gate_up,
        x8.view(u8).view(-1),
        w13.view(u8).view(-1),
        xs.reshape(-1),
        w13_scale.reshape(-1),
        sid,
        tiles,
        count,
        sw,
        a2,
        a2s,
        tokens,
        rows,
        cap * (inter // 128),
        inter,
        hidden,
        stream,
    )
    groups = hidden // DOWN_COLS
    _run(
        ("down_direct", device, hidden, inter),
        launch_down_direct,
        a2,
        w2.view(u8).view(-1),
        a2s,
        w2_scale.reshape(-1),
        tiles,
        count,
        sid,
        y,
        rows,
        tokens,
        0,
        cap * groups,
        hidden,
        inter,
        groups,
        stream,
    )
    return y.view(rows, hidden), inv


def supports_layer(layer, tp_size: int) -> bool:
    """FusedMoE `layer` matches the kernels: gfx942 FP8 128x128 block-scale experts (aiter-shuffled), H 4096,
    I 256 per rank, at most 256 experts, top-8 SiLU-gated, TP8 without EP / shared-expert fusion / clamps /
    biases.
    """
    cfg = getattr(layer, "moe_runner_config", None)
    w13, w2 = getattr(layer, "w13_weight", None), getattr(layer, "w2_weight", None)
    s13, s2 = getattr(layer, "w13_weight_scale_inv", None), getattr(
        layer, "w2_weight_scale_inv", None
    )
    if cfg is None or w13 is None or w2 is None or s13 is None or s2 is None:
        return False
    experts = w13.shape[0]
    return (
        is_gfx942_supported()
        and tp_size == RANKS
        and experts <= SORT_THREADS
        and cfg.top_k == TOPK
        and getattr(layer, "moe_ep_size", 1) == 1
        and cfg.activation == "silu"
        and cfg.is_gated
        and not cfg.apply_router_weight_on_input
        and not cfg.no_combine
        and cfg.routed_scaling_factor in (None, 1.0)
        and cfg.gemm1_alpha is None
        and cfg.gemm1_clamp_limit is None
        and not cfg.swiglu_limit
        and not cfg.num_fused_shared_experts
        and getattr(layer, "w13_weight_bias", None) is None
        and getattr(layer, "w2_weight_bias", None) is None
        and w13.dtype == FP8
        and w2.dtype == FP8
        and tuple(w13.shape) == (experts, 2 * INTER, HIDDEN)
        and tuple(w2.shape) == (experts, HIDDEN, INTER)
        and getattr(w13, "is_shuffled", False)
        and getattr(w2, "is_shuffled", False)
        and s13.dtype == torch.float32
        and s2.dtype == torch.float32
        and s13.numel() == experts * (2 * INTER // 128) * (HIDDEN // 128)
        and s2.numel() == experts * (HIDDEN // 128) * (INTER // 128)
        and not getattr(layer, "hidden_pad", 0)
        and not getattr(layer, "intermediate_pad", 0)
    )


class _HipIpcBuffer:
    """One hipMalloc'ed buffer per rank, mapped into every rank of the group through HIP IPC (process
    lifetime). Handles are exchanged with an all-gather on the group's device communicator, i.e. in stream
    order like the forward's other collectives."""

    class _Handle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_byte * 64)]

    def __init__(self, group, nbytes: int, device: torch.device):
        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            find_loaded_library,
        )

        # The HIP runtime torch already uses (a second copy would have its own device contexts).
        hip = ctypes.CDLL(find_loaded_library("libamdhip64") or "libamdhip64.so")
        hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        hip.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(self._Handle),
            ctypes.c_void_p,
        ]
        hip.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            self._Handle,
            ctypes.c_uint,
        ]

        def check(err, what):
            if err != 0:
                raise RuntimeError(f"{what} failed with HIP error {err}")

        ptr = ctypes.c_void_p()
        check(hip.hipMalloc(ctypes.byref(ptr), nbytes), "hipMalloc")
        check(
            hip.hipMemset(ptr, 0, nbytes), "hipMemset"
        )  # flags start below every epoch
        torch.cuda.synchronize(device)
        handle = self._Handle()
        check(hip.hipIpcGetMemHandle(ctypes.byref(handle), ptr), "hipIpcGetMemHandle")
        mine = torch.frombuffer(bytearray(bytes(handle.reserved)), dtype=torch.uint8)
        handles = torch.empty(group.world_size * 64, dtype=torch.uint8, device=device)
        group.all_gather_into_tensor(handles, mine.to(device))
        handles = handles.cpu().numpy().tobytes()
        self.addrs: List[int] = []
        for r in range(group.world_size):
            if r == group.rank_in_group:
                self.addrs.append(ptr.value)
                continue
            peer = self._Handle()
            ctypes.memmove(ctypes.byref(peer), handles[64 * r : 64 * (r + 1)], 64)
            mapped = ctypes.c_void_p()
            check(
                hip.hipIpcOpenMemHandle(ctypes.byref(mapped), peer, 1),
                "hipIpcOpenMemHandle",
            )
            self.addrs.append(mapped.value)


class _SumReduceScatter:
    """Fused top-k sum + reduce-scatter on one TP group (shared by every MoE layer): a symmetric buffer
    for shards of up to `max_rows` rows, allocated on first use."""

    def __init__(self, group, max_rows: int, device: torch.device):
        nbytes = symmetric_buffer_bytes(max_rows)
        self.group, self.max_rows, self.epoch = group, max_rows, 0
        self.peers = _HipIpcBuffer(group, nbytes, device).addrs
        logger.info(
            "PyHIP MoE fused sum + reduce-scatter: shards up to %d rows, %.0f MiB symmetric buffer",
            max_rows,
            nbytes / 2**20,
        )

    def __call__(self, y, inv, sizes: List[int]) -> torch.Tensor:
        if not any(sizes):
            return torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=y.device)
        self.epoch += 1
        rank = self.group.rank_in_group
        offsets = [sum(sizes[:i]) for i in range(RANKS)]
        out = torch.empty(sizes[rank], HIDDEN, dtype=torch.bfloat16, device=y.device)
        _run(
            ("sum_rs", y.device),
            launch_sum_rs,
            y.view(-1),
            inv.view(-1),
            out.view(-1),
            *self.peers,
            *offsets,
            *sizes,
            max(sizes),
            self.max_rows,
            rank,
            self.epoch,
            RS_GRID,
            torch.cuda.current_stream(y.device),
        )
        return out


_reduce_scatter: Optional[_SumReduceScatter] = None


def fits(sizes: List[int], experts: int, max_rows: int) -> bool:
    """Shards fit the symmetric buffer, and sorted rows x hidden one 4 GiB buffer (32-bit offsets)."""
    tokens = sum(sizes)
    return (
        max(sizes) <= max_rows
        and (tokens * TOPK + experts * COMPACT_BLOCK) * HIDDEN * 2 < 1 << 32
    )


def moe_reduce_scatter(
    layer, x8, xs, topk_weights, topk_ids, sizes: List[int], group, max_rows: int
) -> torch.Tensor:
    """Sequence-parallel TP MoE: tokens gathered from the 8 ranks (rank r's shard = sizes[r] rows, in rank
    order) in, this rank's shard of the sum over ranks of the experts' output out (needs fits()). Same
    arithmetic class as aiter fused_moe + reduce-scatter: FP8 GEMMs with FP32 accumulation, BF16 route rows
    and per-rank partial sums; the rank sum is FP32 in rank order with one BF16 rounding.
    """
    global _reduce_scatter
    if not any(sizes):
        return torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=x8.device)
    if _reduce_scatter is None:
        _reduce_scatter = _SumReduceScatter(group, max_rows, x8.device)
    assert _reduce_scatter.group is group and _reduce_scatter.max_rows == max_rows
    y, inv = moe_route_rows(
        x8,
        xs,
        layer.w13_weight,
        layer.w2_weight,
        topk_weights,
        topk_ids,
        layer.w13_weight_scale_inv,
        layer.w2_weight_scale_inv,
    )
    return _reduce_scatter(y, inv, sizes)
