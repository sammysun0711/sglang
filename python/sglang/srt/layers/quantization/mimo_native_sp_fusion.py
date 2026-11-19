"""Fuse native row quantization and the lossless SP packet copies on gfx942."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
import torch
import triton
import triton.language as tl

from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import _pack_activation, _pointer
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def _packet_words(hidden, topk):
    return ((hidden // 4 + 1 + 2 * topk + 3) // 4) * 4


@cache
def _compile_quant_pack(rows, hidden, topk):
    words = _packet_words(hidden, topk)
    data_words = hidden // 4
    padding = words - data_words - 1 - 2 * topk

    @flyc.kernel(known_block_size=(256, 1, 1))
    def kernel(source: fx.Int64, weights: fx.Int64, ids: fx.Int64, packet: fx.Int64):
        tid = fx.thread_idx.x
        row = fx.block_idx.x * 4 + tid // 64
        lane = tid % 64
        if row < rows:
            values = []
            amax = fx.Float32(0.0)
            # Preserve the native quantize_rows reduction and rounding order.
            for chunk in range_constexpr(hidden // 256):
                offset = fx.Int64(row) * hidden + lane * 4 + chunk * 256
                value = fx.ptr_load(
                    fx.add_offset(_pointer(source, fx.BFloat16, 8), offset),
                    result_type=fx.Vector.make_type(4, fx.BFloat16),
                ).to(fx.Float32)
                values.append(value)
                for component in range_constexpr(4):
                    amax = fx.max(amax, fx.max(value[component], -value[component]))
            for bit in range_constexpr(6):
                amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
            scale = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
            inverse = 1.0 / scale
            base = fx.Int64(row) * words
            if lane == 0:
                fx.ptr_store(scale, fx.add_offset(_pointer(packet, fx.Float32, 4), base + data_words))
            if lane < topk:
                weight = fx.ptr_load(fx.add_offset(_pointer(weights, fx.Float32, 4), fx.Int64(row) * topk + lane))
                expert = fx.ptr_load(fx.add_offset(_pointer(ids, fx.Int32, 4), fx.Int64(row) * topk + lane))
                fx.ptr_store(weight, fx.add_offset(_pointer(packet, fx.Float32, 4), base + data_words + 1 + lane))
                fx.ptr_store(expert, fx.add_offset(_pointer(packet, fx.Int32, 4), base + data_words + 1 + topk + lane))
            if lane < padding:
                fx.ptr_store(fx.Int32(0), fx.add_offset(_pointer(packet, fx.Int32, 4), base + data_words + 1 + 2 * topk + lane))
            for chunk in range_constexpr(hidden // 256):
                value = values[chunk] * inverse
                packed = _pack_activation(value[0], value[1], value[2], value[3])
                fx.ptr_store(packed, fx.add_offset(_pointer(packet, fx.Int32, 4), base + lane + chunk * 64))

    @flyc.jit
    def launch(source: fx.Int64, weights: fx.Int64, ids: fx.Int64, packet: fx.Int64, stream: fx.Stream):
        kernel(source, weights, ids, packet).launch(grid=((rows + 3) // 4,), block=(256,), stream=stream)

    return launch


def quantize_pack_native_sp(states, weights, ids):
    rows, hidden = states.shape
    topk = weights.shape[1]
    if (rows < 1 or hidden % 256 or not 0 < topk <= 64
        or states.dtype != torch.bfloat16 or weights.dtype != torch.float32
        or ids.dtype != torch.int32 or weights.shape != (rows, topk) or ids.shape != weights.shape
        or any(not tensor.is_contiguous() or tensor.device != states.device for tensor in (states, weights, ids))):
        raise ValueError("Native SP fusion requires contiguous BF16 rows and matching FP32/int32 routing")
    packet = torch.empty((rows, _packet_words(hidden, topk)), device=states.device, dtype=torch.float32)
    _run_compiled(_compile_quant_pack(rows, hidden, topk), states.data_ptr(), weights.data_ptr(),
                  ids.data_ptr(), packet.data_ptr(), torch.cuda.current_stream(states.device))
    return packet


@triton.jit
def _unpack(packet, quantized, scales, weights, ids, WORDS: tl.constexpr,
            DATA_WORDS: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    base = packet + row * WORDS
    value = tl.load(base + offset, mask=offset < DATA_WORDS, other=0)
    tl.store(quantized + row * DATA_WORDS + offset, value, mask=offset < DATA_WORDS)
    scale = tl.load(base + DATA_WORDS).to(tl.float32, bitcast=True)
    tl.store(scales + row, scale)
    weight = tl.load(base + DATA_WORDS + 1 + offset, mask=offset < TOPK, other=0).to(tl.float32, bitcast=True)
    expert = tl.load(base + DATA_WORDS + 1 + TOPK + offset, mask=offset < TOPK, other=0)
    tl.store(weights + row * TOPK + offset, weight, mask=offset < TOPK)
    tl.store(ids + row * TOPK + offset, expert, mask=offset < TOPK)


def unpack_native_sp(packet, hidden, topk):
    rows = packet.shape[0]
    if (packet.dtype != torch.float32 or not packet.is_contiguous()
        or packet.shape[1] != _packet_words(hidden, topk) or hidden % 256):
        raise ValueError("Unexpected native SP packet layout")
    values = torch.empty((rows, hidden), device=packet.device, dtype=torch.float8_e4m3fnuz)
    scales = torch.empty(rows, device=packet.device, dtype=torch.float32)
    weights = torch.empty((rows, topk), device=packet.device, dtype=torch.float32)
    ids = torch.empty((rows, topk), device=packet.device, dtype=torch.int32)
    _unpack[(rows,)](packet.view(torch.int32), values.view(torch.int32), scales, weights, ids,
                     WORDS=packet.shape[1], DATA_WORDS=hidden // 4, TOPK=topk,
                     BLOCK=triton.next_power_of_2(hidden // 4), num_warps=4)
    return values, scales, weights, ids
