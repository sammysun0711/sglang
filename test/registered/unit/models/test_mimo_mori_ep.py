from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import topk
from sglang.srt.layers.moe.token_dispatcher import moriep
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


@pytest.mark.parametrize(
    "configured,recording,capturing,expected",
    [
        (False, False, False, False),
        (False, False, True, False),
        (True, False, False, False),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, True),
    ],
)
def test_mori_counts_only_for_configured_or_active_recorder(
    monkeypatch, configured, recording, capturing, expected
):
    recorder = (
        SimpleNamespace(recording=recording)
        if configured
        else moriep._ExpertDistributionRecorderNoop()
    )
    monkeypatch.setattr(
        moriep, "get_global_expert_distribution_recorder", lambda: recorder
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda: SimpleNamespace(is_current_stream_capturing=lambda: capturing),
    )
    assert moriep._should_record_expert_distribution() is expected


def test_hip_masks_padded_expert_ids_without_eplb(monkeypatch):
    monkeypatch.setattr(topk, "_is_cuda", False)
    monkeypatch.setattr(topk, "_is_hip", True)
    monkeypatch.setattr(topk, "get_global_experts_capturer", lambda: None)
    monkeypatch.setattr(topk, "_eplb_remap_enabled", lambda: False)
    monkeypatch.setattr(
        topk,
        "_mask_topk_ids_padded_region",
        lambda ids, n, fill_value: ids[int(n) :].fill_(fill_value),
    )
    monkeypatch.setattr(
        topk,
        "_zero_topk_weights_padded_region",
        lambda weights, n: weights[int(n) :].zero_(),
    )
    ids = torch.tensor([[2, 3], [-1, 999]], dtype=torch.int32)
    weights = torch.ones((2, 2))
    config = SimpleNamespace(
        num_fused_shared_experts=0, fused_shared_experts_scaling_factor=1.0
    )
    result, weights, _ = topk._post_process_topk_ids(
        ids,
        weights,
        config,
        torch.empty((2, 4)),
        layer_id=0,
        num_token_non_padded=torch.tensor(1),
    )
    assert result.tolist() == [[2, 3], [0, 0]]
    assert weights.tolist() == [[1.0, 1.0], [0.0, 0.0]]
