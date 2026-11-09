"""Experimental DFlash chain bookkeeping; probability policy/sampling stay intact."""

import ctypes
import functools
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_logged = False

@functools.lru_cache(maxsize=16)
def _scalar_reciprocal(threshold_acc):
    # Match ATen div_true_kernel_cuda's CPU-scalar path for FP32 tensors:
    # float inv_b = 1.0f / float(b), followed by a GPU multiplication.
    denominator = ctypes.c_float(max(float(threshold_acc), 1e-9)).value
    return ctypes.c_float(1.0 / denominator).value


@triton.jit
def _chain_prefix(probs, candidates, coins, counts, rejected_ids,
                  VOCAB: tl.constexpr, TOKENS: tl.constexpr, BLOCK: tl.constexpr,
                  THRESHOLD_SINGLE: tl.constexpr, INV_THRESHOLD_ACC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    depth = tl.arange(0, BLOCK)
    valid = depth < TOKENS - 1
    token = tl.load(candidates + row * TOKENS + depth + 1, valid, other=0)
    probability = tl.load(probs + (row * TOKENS + depth) * VOCAB + token, valid, other=0)
    coin = tl.load(coins + row * TOKENS + depth, valid, other=1)
    accepted = valid & ((coin <= probability * INV_THRESHOLD_ACC)
                        | (probability >= THRESHOLD_SINGLE))
    prefix = tl.cumprod(accepted.to(tl.int32), axis=0)
    count = tl.sum(prefix, axis=0)
    rejected = tl.load(candidates + row * TOKENS + count + 1, count < TOKENS - 1, other=-1)
    tl.store(counts + row, count)
    tl.store(rejected_ids + row, rejected)


@triton.jit
def _chain_final_weights(probs, counts, rejected_ids, weights,
                         VOCAB: tl.constexpr, TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(counts + row).to(tl.int64)
    rejected = tl.load(rejected_ids + row)
    probability = tl.load(probs + (row * TOKENS + count) * VOCAB + columns, columns < VOCAB, other=0)
    tl.store(weights + row * VOCAB + columns, tl.where(columns == rejected, 0., probability), columns < VOCAB)


def chain_weights(target_probs, candidates, uniform_samples, *, threshold_single=1., threshold_acc=1.):
    """Return accepted prefix and final weights for a finite, nonnegative chain.

    A rejection removes exactly the rejected candidate's mass. It needs one
    token index per request, instead of a dense [batch, block, vocab] tensor.
    """
    if target_probs.ndim != 3 or target_probs.dtype != torch.float32 or not target_probs.is_contiguous():
        raise ValueError("Chain probabilities must be contiguous FP32 [batch, block, vocab]")
    batch, tokens, vocab = target_probs.shape
    if batch < 1 or tokens not in (1, 8) or vocab < 1:
        raise ValueError("The prototype covers positive batches and DFlash block1/8")
    for tensor, dtype in ((candidates, torch.int64), (uniform_samples, torch.float32)):
        if tensor.shape != (batch, tokens) or tensor.dtype != dtype or not tensor.is_contiguous() or tensor.device != target_probs.device:
            raise ValueError("Invalid chain candidates or uniform samples")
    counts = torch.empty(batch, dtype=torch.int32, device=target_probs.device)
    rejected = torch.empty(batch, dtype=torch.int64, device=target_probs.device)
    weights = torch.empty(batch, vocab, dtype=torch.float32, device=target_probs.device)
    _chain_prefix[(batch,)](target_probs, candidates, uniform_samples, counts, rejected,
                           vocab, tokens, triton.next_power_of_2(tokens), float(threshold_single),
                           _scalar_reciprocal(threshold_acc), num_warps=1)
    _chain_final_weights[(batch, triton.cdiv(vocab, 1024))](
        target_probs, counts, rejected, weights, vocab, tokens, 1024, num_warps=4)
    return counts, weights


def verify_chain(target_probs, candidates, uniform_samples, final_uniforms, *,
                 threshold_single=1., threshold_acc=1., reference_batch_size=None):
    from sglang.srt.speculative.eagle_utils import _sample_from_weights_with_coin

    counts, weights = chain_weights(target_probs, candidates, uniform_samples,
                                    threshold_single=threshold_single, threshold_acc=threshold_acc)
    bonus = _sample_from_weights_with_coin(weights, final_uniforms, reference_rows=reference_batch_size)
    global _logged
    if not _logged:
        logger.info("Using fused DFlash chain verification: block8, FP32 probabilities")
        _logged = True
    return counts, bonus
