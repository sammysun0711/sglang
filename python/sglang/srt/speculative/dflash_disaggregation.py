from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def build_dflash_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    """Seed the first DFlash decode step after disaggregated prefill."""
    batch_size = int(last_tokens_tensor.numel())
    spec_info = DFlashDraftInputV2(
        topk_p=torch.empty(
            (batch_size, 0), device=batch.device, dtype=torch.float32
        ),
        topk_index=torch.empty(
            (batch_size, 0), device=batch.device, dtype=torch.int64
        ),
        verified_id=last_tokens_tensor.to(dtype=torch.int32),
        new_seq_lens=batch.seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty(
            (batch_size, 0), device=batch.device, dtype=torch.float16
        ),
        cur_allocated_seq_lens_cpu=batch.seq_lens_cpu,
    )

    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, spec_info.new_seq_lens)
        future_map.stash(spec_info.future_indices, spec_info)

    return spec_info
