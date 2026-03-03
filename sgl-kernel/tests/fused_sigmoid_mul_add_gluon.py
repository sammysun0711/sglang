"""Gluon implementation of the fused sigmoid-mul-add kernel (copy for unit tests).

Computes in-place:
    final_hidden_states[row, col] += sigmoid(gate[row, 0]) * shared_output[row, col]

Copied from python/sglang/jit_kernel/fused_sigmoid_mul_add_gluon.py so tests can run
without registering torch.ops.sgl_kernel. Uses backend-appropriate warp size (32 CUDA, 64 HIP).

Performance (why fused can be slower than PyTorch ref):
- PyTorch one-liner (out + sigmoid(gate)*shared) is often fused by the compiler into a single
  highly optimized element-wise kernel; our explicit kernel may still be slower.
- This kernel uses FP32 for sigmoid and accumulate (gl.exp requires fp32/fp64), which doubles
  effective bandwidth vs bf16/fp16; PyTorch may keep lower precision and be faster.
- Grid (num_tokens, num_col_blocks); BLOCK_SIZE=256 increases block count vs 1024 and can
  improve occupancy for small num_tokens.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def _make_layout(threads_per_warp: int):
    """BlockedLayout with given threads_per_warp (32 for CUDA, 64 for HIP)."""
    return gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[threads_per_warp],
        warps_per_cta=[4],
        order=[0],
    )

# def _make_layout(threads_per_warp: int):
#     """BlockedLayout with given threads_per_warp (32 for CUDA, 64 for HIP)."""
#     return gl.BlockedLayout(
#         size_per_thread=[1, 8],
#         threads_per_warp=[4, 16], #threads_per_warp=[threads_per_warp],
#         warps_per_cta=[2, 2], #warps_per_cta=[4],
#         order=[1, 0],
#     )

def _fused_sigmoid_mul_add_gluon_kernel_impl(threads_per_warp: int):
    """Build kernel with backend-appropriate layout (constexpr must be fixed at compile)."""
    layout = _make_layout(threads_per_warp)

    @gluon.jit
    def kernel(
        gate_ptr,  # [num_tokens] (flattened from [num_tokens, 1])
        shared_ptr,  # [num_tokens, hidden_size]
        out_ptr,  # [num_tokens, hidden_size] (in-place)
        hidden_size,  # number of columns
        shared_stride_row,  # stride of shared_output along row dim
        out_stride_row,  # stride of final_hidden_states along row dim
        BLOCK_SIZE: gl.constexpr,
    ):
        row = gl.program_id(0)
        col_block = gl.program_id(1)

        col_offsets = col_block * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
        mask = col_offsets < hidden_size

        # gl.exp requires fp32/fp64; sigmoid in fp32 then multiply with loaded values (fp32 for precision).
        gate_val = gl.load(gate_ptr + row).to(gl.float32)
        # Gluon does not provide a built-in sigmoid, so we compute it manually:
        #   sigmoid(x) = 1 / (1 + exp(-x))
        sig = 1.0 / (1.0 + gl.exp(-gate_val))

        # Compute row offsets for 2D tensors using strides.
        shared_offsets = row * shared_stride_row + col_offsets
        out_offsets = row * out_stride_row + col_offsets

        # Load tiles, cast to fp32 for arithmetic precision.
        shared_val = gl.load(shared_ptr + shared_offsets, mask=mask).to(gl.float32)
        out_val = gl.load(out_ptr + out_offsets, mask=mask).to(gl.float32)

        # Fused sigmoid * mul + add.
        result = out_val + sig * shared_val
        gl.store(out_ptr + out_offsets, result, mask=mask)

    return kernel


# Compile one kernel per backend so layout matches warp size (CUDA 32, HIP 64).
_fused_sigmoid_mul_add_gluon_kernel_32 = _fused_sigmoid_mul_add_gluon_kernel_impl(32)
_fused_sigmoid_mul_add_gluon_kernel_64 = _fused_sigmoid_mul_add_gluon_kernel_impl(64)


def _is_hip() -> bool:
    """True if PyTorch is built with HIP (AMD), where warp size is 64."""
    return getattr(torch.version, "hip", None) is not None


def fused_sigmoid_mul_add_gluon(
    gate: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Fused sigmoid-mul-add: final_hidden_states += sigmoid(gate) * shared_output.

    Args:
        gate: [num_tokens, 1] (or flattenable to 1D).
        shared_output: [num_tokens, hidden_size].
        final_hidden_states: [num_tokens, hidden_size], modified in-place.
    """
    num_tokens, hidden_size = shared_output.shape
    gate_flat = gate.view(-1)

    # Smaller BLOCK_SIZE increases grid size (num_col_blocks) and can improve occupancy
    # when num_tokens is small; 256 often beats 1024 for MoE shapes (e.g. 80x4096).
    BLOCK_SIZE = 256
    num_col_blocks = triton.cdiv(hidden_size, BLOCK_SIZE)
    grid = (num_tokens, num_col_blocks)

    kernel = (
        _fused_sigmoid_mul_add_gluon_kernel_64
        if _is_hip()
        else _fused_sigmoid_mul_add_gluon_kernel_32
    )
    kernel[grid](
        gate_flat,
        shared_output,
        final_hidden_states,
        hidden_size,
        shared_output.stride(0),
        final_hidden_states.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
