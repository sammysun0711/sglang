"""Mixed-input router GEMM optimized for large MiMo prefill batches on ROCm."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _mixed_router_gemm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    weight_ptrs = (
        weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (k_start + offs_k[None, :] < K),
            other=0.0,
        )
        weight = tl.load(
            weight_ptrs,
            mask=(offs_n[None, :] < N) & (k_start + offs_k[:, None] < K),
            other=0.0,
        )
        accumulator = tl.dot(
            x.to(tl.float16), weight, acc=accumulator, out_dtype=tl.float32
        )
        x_ptrs += BLOCK_K * stride_xk
        weight_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def mixed_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T`` with BF16 activations and FP16 weights.

    Accumulation and output remain FP32. Converting the BF16 activation tile to
    FP16 in registers avoids materializing the full FP32 router input.
    """
    assert x.ndim == 2 and weight.ndim == 2
    assert x.shape[1] == weight.shape[1]
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.float16
    assert x.is_contiguous() and weight.is_contiguous()

    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)

    block_m = 128
    block_n = 128
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _mixed_router_gemm_kernel[grid](
        x,
        weight,
        out,
        M=m,
        N=n,
        K=k,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        GROUP_M=8,
        num_warps=8,
        num_stages=2,
    )
    return out
