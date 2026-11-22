"""Keep PyTorch's full-batch FP32 scan order on a request-owned row subset."""

import functools

import torch

_DECLARATION = "torch::Tensor owner_cumsum(torch::Tensor input, int64_t reference_rows);"
_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/native/cuda/ScanUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <functional>
#include <climits>

torch::Tensor owner_cumsum(torch::Tensor input, int64_t reference_rows) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat,
              "owner cumsum requires a CUDA/HIP FP32 tensor");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
              "owner cumsum requires a contiguous matrix");
  const auto rows = input.size(0);
  const auto columns = input.size(1);
  TORCH_CHECK(reference_rows >= rows && reference_rows > 1 &&
              reference_rows <= UINT_MAX && rows <= UINT_MAX && columns <= UINT_MAX,
              "invalid reference or matrix dimensions");
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty_like(input);
  if (rows == 0 || columns == 0) return output;
  const uint32_t log_x = at::native::get_log_num_threads_x_inner_scan<uint32_t>(
      reference_rows, columns);
  const uint32_t threads_x = 1u << log_x;
  const dim3 threads(threads_x, 512u / threads_x);
  const dim3 grid((rows + threads.y - 1) / threads.y);
  // Reuse the installed PyTorch kernel, changing only the launch geometry's
  // reference row count. Arithmetic and the per-row Sklansky scan stay intact.
  at::native::tensor_kernel_scan_innermost_dim<float>
      <<<grid, threads, 1024 * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
          output.data_ptr<float>(), input.data_ptr<float>(), rows, columns,
          log_x, 0.0f, std::plus<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@functools.lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load_inline

    if torch.version.hip is None:
        raise RuntimeError("The request-owner scan extension is qualified for ROCm only")
    return load_inline(
        name="sglang_dflash_owner_scan",
        cpp_sources=_DECLARATION,
        cuda_sources=_CUDA_SOURCE,
        functions=["owner_cumsum"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def cumsum_like_full_batch(values: torch.Tensor, reference_rows: int) -> torch.Tensor:
    if values.ndim != 2 or values.dtype != torch.float32 or not values.is_contiguous():
        raise ValueError("Request-owner cumsum requires a contiguous FP32 matrix")
    if reference_rows < values.shape[0]:
        raise ValueError("The reference row count cannot be smaller than the owned rows")
    if reference_rows == values.shape[0] or values.numel() == 0:
        return values.cumsum(dim=-1)
    return _extension().owner_cumsum(values, int(reference_rows))
