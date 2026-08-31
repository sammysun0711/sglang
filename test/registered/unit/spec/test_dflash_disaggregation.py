from unittest.mock import Mock

from sglang.srt.speculative import dflash_disaggregation
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def test_dflash_builds_disaggregation_draft_input(monkeypatch):
    sentinel = object()
    builder = Mock(return_value=sentinel)
    monkeypatch.setattr(
        dflash_disaggregation, "build_dflash_family_disagg_draft_input", builder
    )

    batch = object()
    last_tokens = object()
    future_map = object()
    result = SpeculativeAlgorithm.DFLASH.build_disagg_draft_input(
        batch, last_tokens, future_map
    )

    assert result is sentinel
    builder.assert_called_once_with(batch, last_tokens, future_map)
