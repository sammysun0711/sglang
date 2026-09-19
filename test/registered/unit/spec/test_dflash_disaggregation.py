from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.speculative.dflash_disaggregation import (
    build_dflash_disagg_draft_input,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def test_dflash_dispatches_disaggregation_builder(monkeypatch):
    sentinel = object()
    builder = Mock(return_value=sentinel)
    monkeypatch.setattr(
        "sglang.srt.speculative.dflash_disaggregation.build_dflash_disagg_draft_input",
        builder,
    )

    batch = object()
    last_tokens = object()
    future_map = object()
    result = SpeculativeAlgorithm.DFLASH.build_disagg_draft_input(
        batch, object(), last_tokens, future_map
    )

    assert result is sentinel
    builder.assert_called_once_with(batch, last_tokens, future_map)


def test_dflash_disaggregation_seeds_direct_and_relay_state():
    batch = SimpleNamespace(
        device=torch.device("cpu"),
        enable_overlap=True,
        seq_lens=torch.tensor([64, 96], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([64, 96], dtype=torch.int64),
        req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
    )
    last_tokens = torch.tensor([11, 13], dtype=torch.int64)
    future_map = Mock()

    spec_info = build_dflash_disagg_draft_input(batch, last_tokens, future_map)

    assert spec_info.verified_id.dtype == torch.int32
    assert torch.equal(spec_info.verified_id, last_tokens.to(torch.int32))
    assert torch.equal(spec_info.new_seq_lens, batch.seq_lens)
    assert spec_info.cur_allocated_seq_lens_cpu is batch.seq_lens_cpu
    assert spec_info.future_indices is batch.req_pool_indices
    assert spec_info.direct_carry_valid
    future_map.publish.assert_called_once_with(
        batch.req_pool_indices, spec_info.new_seq_lens
    )
    future_map.stash.assert_called_once_with(batch.req_pool_indices, spec_info)
