"""Opt-in DFlash verification on request-owned full-vocabulary logits."""

import logging
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_logged_modes = set()


@triton.jit
def _assemble_logits(recv, output, ROWS: tl.constexpr, SHARD: tl.constexpr,
                     VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source_rank = cols // SHARD
    local_col = cols % SHARD
    offsets = (source_rank.to(tl.int64) * ROWS + row) * SHARD + local_col
    values = tl.load(recv + offsets, cols < VOCAB, other=0).to(tl.float32)
    tl.store(output + row * VOCAB + cols, values, cols < VOCAB)


def distribute_logits(local_logits, group, *, block_size, vocab_size, output_buffer=None):
    """Collect full vocabulary columns only for this rank's request block."""
    world, rank = group.world_size, group.rank_in_group
    if world != 4 or block_size != 8 or local_logits.dtype != torch.bfloat16:
        raise ValueError("Request-owner logits require TP4, DFlash8 and BF16 LM-head output")
    rows, shard = local_logits.shape
    if rows <= 0 or rows % block_size or shard * world != vocab_size:
        raise ValueError("Unsupported request-owner logit shape")
    if not local_logits.is_contiguous():
        local_logits = local_logits.contiguous()
    chunk = (rows // block_size + world - 1) // world
    owned_rows = chunk * block_size
    padded_rows = owned_rows * world
    if padded_rows != rows:
        local_logits = torch.cat((local_logits, local_logits.new_zeros(padded_rows - rows, shard)))
    recv = local_logits.new_empty((world, owned_rows, shard))
    comm = group.pynccl_comm
    if comm is None or not comm.available:
        raise RuntimeError("Request-owner logits require the direct RCCL communicator")
    with comm.change_state(enable=True):
        comm.group_start()
        for peer in range(world):
            if peer != rank:
                comm.send(local_logits[peer * owned_rows:(peer + 1) * owned_rows], peer)
                comm.recv(recv[peer], peer)
        comm.group_end()
    recv[rank].copy_(local_logits[rank * owned_rows:(rank + 1) * owned_rows])
    if output_buffer is None:
        output = torch.empty(owned_rows, vocab_size, device=local_logits.device, dtype=torch.float32)
    else:
        if (output_buffer.dtype != torch.float32 or output_buffer.shape[0] < owned_rows
                or output_buffer.shape[1] != vocab_size or not output_buffer.is_contiguous()):
            raise ValueError("Invalid request-owner logits buffer")
        output = output_buffer[:owned_rows]
    _assemble_logits[(owned_rows, triton.cdiv(vocab_size, 1024))](
        recv, output, owned_rows, shard, vocab_size, 1024, num_warps=4)
    return output, chunk


def restore_full_logits(logits, group, *, batch_size, block_size, chunk):
    """Restore the original row layout for a consumer needing the dense path."""
    if logits.shape[0] != chunk * block_size or not logits.is_contiguous():
        raise ValueError("Invalid captured owner layout")
    full = logits.new_empty(group.world_size * logits.shape[0], logits.shape[1])
    with group.pynccl_comm.change_state(enable=True):
        group.pynccl_comm.all_gather(full, logits)
    return full[:batch_size * block_size]


def can_verify_owned(sampling_info):
    if sampling_info is None:
        return True
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    return not (
        sampling_info.has_custom_logit_processor
        or (penalizer is not None and penalizer.is_required)
        or getattr(sampling_info, "acc_linear_penalties", None) is not None
        or getattr(sampling_info, "vocab_mask", None) is not None
        or getattr(sampling_info, "grammar_mask", None) is not None
        or getattr(sampling_info, "logit_bias", None) is not None
        or getattr(sampling_info, "need_min_p_sampling", False)
    )


def verify_owned(logits, candidates, sampling_info, group, *, chunk,
                 max_top_k=None, uniform_top_k_value=None):
    """Return full-batch accepted lengths/bonus IDs, or request dense fallback."""
    from sglang.srt.speculative.dflash_utils import (
        compute_dflash_correct_drafts_and_bonus,
        compute_dflash_sampling_correct_drafts_and_bonus,
        is_dflash_sampling_verify_available,
    )

    if not can_verify_owned(sampling_info):
        return None
    greedy = sampling_info is None or sampling_info.is_all_greedy
    if not greedy and not is_dflash_sampling_verify_available():
        return None
    batch, block = candidates.shape
    if group.world_size != 4 or block != 8 or logits.shape[0] != chunk * block:
        raise ValueError("Invalid request-owner verification layout")
    start = group.rank_in_group * chunk
    end = min(start + chunk, batch)
    if not greedy:
        # Keep the original per-TP RNG call shapes/order. Only slice after both
        # full-batch draws, including on ranks with no real requests.
        coins = torch.rand((batch, block), dtype=torch.float32, device=logits.device)
        final_coins = torch.rand((batch,), dtype=torch.float32, device=logits.device)
    packed = torch.zeros(chunk, 2, device=logits.device, dtype=torch.int64)
    if end > start:
        local = logits[:(end - start) * block]
        if greedy:
            accepted, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates[start:end], target_predict=local.argmax(-1).view(end - start, block))
        else:
            info = SimpleNamespace(
                temperatures=sampling_info.temperatures[start:end],
                top_ps=sampling_info.top_ps[start:end], top_ks=sampling_info.top_ks[start:end],
                need_top_k_sampling=sampling_info.need_top_k_sampling,
                need_top_p_sampling=sampling_info.need_top_p_sampling,
            )
            accepted, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates[start:end], next_token_logits=local, sampling_info=info,
                uniform_samples=coins[start:end], uniform_samples_for_final_sampling=final_coins[start:end],
                max_top_k=max_top_k, uniform_top_k_value=uniform_top_k_value,
                reference_batch_size=batch,
            )
        packed[:end - start].copy_(torch.stack((accepted.to(torch.int64), bonus), dim=1))
    gathered = packed.new_empty((group.world_size * chunk, 2))
    with group.pynccl_comm.change_state(enable=True):
        group.pynccl_comm.all_gather(gathered, packed)
    mode = "greedy" if greedy else "sampling"
    if mode not in _logged_modes:
        logger.info("Using request-owner DFlash %s verification: TP4, block8", mode)
        _logged_modes.add(mode)
    return gathered[:batch, 0].to(torch.int32), gathered[:batch, 1]
