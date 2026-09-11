import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


@pytest.fixture(autouse=True)
def _clear_aiter_signature_caches():
    cached_probes = (
        aiter_runner._aiter_fused_moe_supports_no_combine,
        aiter_runner._aiter_fused_moe_supports_ep_route_convention,
        aiter_runner._aiter_fused_moe_supports_transposed_a1_scale,
    )
    for probe in cached_probes:
        probe.cache_clear()
    yield
    for probe in cached_probes:
        probe.cache_clear()


def _runner_input():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    return AiterRunnerInput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones(topk_ids.shape, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 8, 2)),
        "w2_weight": torch.empty((2, 4, 2)),
        "quant_type": AiterQuantType.PER_1X32,
    }
    kwargs.update(overrides)
    return AiterMoeQuantInfo(**kwargs)


def _install_fake_aiter(monkeypatch, fused_moe):
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="INTERLEAVE")
    )

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)


def test_aiter_runner_forwards_no_combine_and_extra_fused_moe_kwargs(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"}),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_aiter_runner_rejects_no_combine_when_fused_moe_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))

    with pytest.raises(NotImplementedError, match="no_combine=True"):
        runner.run(_runner_input(), _quant_info(), running_state={})


def test_aiter_runner_preserves_no_combine_rank_for_empty_input(monkeypatch):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))
    runner_input = _runner_input()
    runner_input.hidden_states = torch.zeros((0, 4), dtype=torch.bfloat16)
    runner_input.topk_ids = torch.zeros((0, 2), dtype=torch.int32)
    runner_input.topk_weights = torch.zeros((0, 2), dtype=torch.float32)

    output = runner.run(runner_input, _quant_info(), running_state={})

    assert output.hidden_states.shape == (0, 2, 4)


@pytest.mark.parametrize(
    "shared_experts,extra_fake_column,is_ep,expected_has_fake_route",
    [
        (0, False, True, False),
        (1, False, True, False),
        (2, False, True, False),
        (0, True, True, True),
        (1, True, True, True),
        (2, True, True, True),
        (0, False, False, True),
        (1, False, False, True),
    ],
)
def test_aiter_runner_preserves_shared_and_fake_route_conventions(
    monkeypatch, shared_experts, extra_fake_column, is_ep, expected_has_fake_route
):
    captured = {}

    def fused_moe(*, ep_has_fake_route=True, **kwargs):
        captured.update(kwargs)
        captured["ep_has_fake_route"] = ep_has_fake_route
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    # Shared columns count toward configured top_k; an extra fake column does not.
    runner = AiterRunnerCore(
        MoeRunnerConfig(
            activation="silu",
            top_k=2 + shared_experts,
            num_fused_shared_experts=shared_experts,
        )
    )
    runner_input = _runner_input()
    ids = [0, 1] + list(range(3, 3 + shared_experts))
    mask = [1, 1, 0] + [1] * shared_experts
    if extra_fake_column:
        ids.append(len(mask))
        mask.append(0)
    runner_input.topk_ids = torch.tensor([ids], dtype=torch.int32)
    runner_input.topk_weights = torch.ones((1, len(ids)), dtype=torch.float32)
    expert_mask = torch.tensor(mask, dtype=torch.int32) if is_ep else None

    runner.run(
        runner_input,
        _quant_info(expert_mask=expert_mask),
        running_state={},
    )

    assert captured["topk_ids"] is runner_input.topk_ids
    assert captured["topk_weight"] is runner_input.topk_weights
    assert captured["expert_mask"] is expert_mask
    assert captured["ep_has_fake_route"] is expected_has_fake_route


def test_aiter_runner_preserves_ep_inputs_for_legacy_aiter(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", top_k=2))
    runner_input = _runner_input()
    expert_mask = torch.tensor([1, 1, 0], dtype=torch.int32)

    runner.run(
        runner_input,
        _quant_info(expert_mask=expert_mask),
        running_state={},
    )

    assert captured["topk_ids"] is runner_input.topk_ids
    assert captured["topk_weight"] is runner_input.topk_weights
    assert captured["expert_mask"] is expert_mask
    assert "ep_has_fake_route" not in captured


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
