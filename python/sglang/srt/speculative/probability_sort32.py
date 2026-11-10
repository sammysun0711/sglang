"""Experimental ROCm probability sort with 32-bit internal token indices."""

import functools
import logging

import torch

_DECLARATION = "std::vector<torch::Tensor> prob_sort32(torch::Tensor probabilities);"
_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/hip/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <climits>

std::vector<torch::Tensor> prob_sort32(torch::Tensor probabilities) {
  TORCH_CHECK(probabilities.is_cuda() && probabilities.scalar_type() == at::kFloat,
              "probability sort requires ROCm FP32 input");
  TORCH_CHECK(probabilities.dim() == 2 && probabilities.is_contiguous(),
              "probability sort requires a contiguous matrix");
  TORCH_CHECK(probabilities.numel() <= INT_MAX,
              "probability sort exceeds segmented radix-sort limits");
  const c10::cuda::CUDAGuard guard(probabilities.device());
  const int64_t rows = probabilities.size(0);
  const int64_t columns = probabilities.size(1);
  auto sorted = torch::empty_like(probabilities);
  auto indices = torch::empty(probabilities.sizes(), probabilities.options().dtype(at::kInt));
  if (rows == 0 || columns == 0) return {sorted, indices.to(at::kLong)};
  auto input_indices = torch::arange(columns, probabilities.options().dtype(at::kInt))
                           .repeat({rows, 1});
  auto offsets = torch::arange(0, probabilities.numel() + 1, columns,
                               probabilities.options().dtype(at::kInt));
  at::cuda::cub::segmented_sort_pairs<float, int32_t, const int32_t*>(
      probabilities.data_ptr<float>(), sorted.data_ptr<float>(),
      input_indices.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
      probabilities.numel(), rows, offsets.data_ptr<int32_t>(),
      offsets.data_ptr<int32_t>() + 1, true);
  // Preserve the existing Torch scatter/gather interface. This conversion
  // moves integer metadata only; no probability arithmetic changes.
  return {sorted, indices.to(at::kLong)};
}
"""


@functools.lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load_inline

    if torch.version.hip is None:
        raise RuntimeError("This sort prototype requires ROCm")
    module = load_inline(
        name="sglang_dflash_prob_sort32_v1",
        cpp_sources=_DECLARATION,
        cuda_sources=_CUDA_SOURCE,
        functions=["prob_sort32"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    logging.getLogger(__name__).info(
        "Using 32-bit internal indices for DFlash probability sort"
    )
    return module


def sort_probabilities(probabilities):
    """Return sorted FP32 probabilities and int64 indices, like Torch sort."""
    if (torch.version.hip is None or probabilities.ndim != 2 or probabilities.shape[1] != 152576
            or probabilities.dtype != torch.float32 or not probabilities.is_contiguous()
            or not probabilities.is_cuda or probabilities.requires_grad
            or probabilities.numel() > 2**31 - 1):
        return probabilities.sort(dim=-1, descending=True)
    values, indices = extension().prob_sort32(probabilities)
    return values, indices
