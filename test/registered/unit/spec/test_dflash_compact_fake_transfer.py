"""Exercise transfer registration: target indices cannot name compact banks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.disaggregation import decode
from sglang.srt.disaggregation.utils import KVClassType
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


@pytest.mark.parametrize("compact", [False, True])
def test_decode_transfer_registration_excludes_bounded_draft(compact, monkeypatch):
    target_pool = SimpleNamespace(
        page_size=64,
        get_contiguous_buf_infos=lambda: ([101, 102], [4096, 4096], [64, 64]),
    )
    draft_buffers = Mock(return_value=([201, 202], [128, 128], [64, 64]))
    if compact:
        draft_buffers.side_effect = AssertionError(
            "compact draft buffers must never be registered"
        )
    draft_pool = SimpleNamespace(get_contiguous_buf_infos=draft_buffers)
    manager = Mock()
    args = SimpleNamespace(
        speculative_dflash_compact_cache=compact,
        disaggregation_mode="decode",
        disaggregation_transfer_backend="fake",
        disaggregation_ib_device=None,
    )
    queue = decode.DecodePreallocQueue.__new__(decode.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        server_args=args,
        ps=SimpleNamespace(dp_rank=0, gpu_id=0),
        enable_hisparse=False,
        model_config=SimpleNamespace(num_hidden_layers=70),
    )
    queue.token_to_kv_pool = target_pool
    queue.draft_token_to_kv_pool = draft_pool
    queue.metadata_buffers = SimpleNamespace(get_buf_infos=lambda: ([301], [64], [64]))
    queue.transfer_backend = "fake"
    queue.tp_rank = queue.pp_rank = 0
    queue.is_mla_backend = False
    queue.enable_staging = False
    queue.req_to_token_pool = object()
    monkeypatch.setattr(decode, "get_attention_tp_size", lambda: 8)
    monkeypatch.setattr(
        decode,
        "get_kv_class",
        lambda backend, kind: (
            SimpleNamespace if kind == KVClassType.KVARGS else manager
        ),
    )
    state_registration = Mock()
    monkeypatch.setattr(decode, "setup_state_kv_args", state_registration)

    queue._init_kv_manager()

    registered = manager.call_args.args[0]
    assert registered.kv_data_ptrs == ([101, 102] if compact else [101, 102, 201, 202])
    assert registered.kv_data_lens == (
        [4096, 4096] if compact else [4096, 4096, 128, 128]
    )
    assert registered.kv_item_lens == ([64, 64] if compact else [64] * 4)
    assert state_registration.call_args.args[2] is (None if compact else draft_pool)
    assert draft_buffers.call_count == (0 if compact else 1)


def test_real_decode_transfer_fails_before_collecting_any_buffers(monkeypatch):
    queue = decode.DecodePreallocQueue.__new__(decode.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            speculative_dflash_compact_cache=True,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
        )
    )
    queue.draft_token_to_kv_pool = Mock()
    collect = Mock(side_effect=AssertionError("no buffer collection before rejection"))
    monkeypatch.setattr(decode, "get_kv_class", collect)
    with pytest.raises(ValueError, match="does not support real"):
        queue._init_kv_manager()
    collect.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
