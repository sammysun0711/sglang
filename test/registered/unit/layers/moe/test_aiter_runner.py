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
    yield
    aiter_runner._aiter_fused_moe_supports_no_combine.cache_clear()
    aiter_runner._aiter_fused_moe_supports_transposed_a1_scale.cache_clear()
    aiter_runner._aiter_fused_moe_supports_ep_route_convention.cache_clear()
    aiter_runner._aiter_fused_moe_supports_opus_stage2_output_dtype.cache_clear()


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


def test_aiter_runner_forwards_gate_layout_for_native_mxfp4(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu"))

    runner.run(
        _runner_input(),
        _quant_info(is_fp4_experts=True),
        running_state={},
    )

    assert captured["gate_mode"] == "INTERLEAVE"
    assert "swiglu_limit" not in captured


@pytest.mark.parametrize("output_dtype", ["auto", "fp8", "bf16"])
def test_aiter_runner_selects_mxfp4_stage2_output(monkeypatch, output_dtype):
    captured = {}

    def fused_moe(*, opus_stage2_output_dtype="auto", **kwargs):
        captured.update(kwargs)
        captured["opus_stage2_output_dtype"] = opus_stage2_output_dtype
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_mxfp4_stage2_output_dtype", lambda: output_dtype
    )
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu"))

    runner.run(
        _runner_input(),
        _quant_info(is_fp4_experts=True),
        running_state={},
    )

    assert captured["opus_stage2_output_dtype"] == output_dtype


def test_aiter_runner_keeps_native_bf16_output_for_ep(monkeypatch):
    captured = {}

    def fused_moe(*, opus_stage2_output_dtype="auto", ep_has_fake_route=True, **kwargs):
        captured.update(kwargs)
        captured["opus_stage2_output_dtype"] = opus_stage2_output_dtype
        captured["ep_has_fake_route"] = ep_has_fake_route
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_mxfp4_stage2_output_dtype", lambda: "bf16"
    )
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", top_k=2))
    expert_mask = torch.tensor([1, 1, 0], dtype=torch.int32)

    runner.run(
        _runner_input(),
        _quant_info(is_fp4_experts=True, expert_mask=expert_mask),
        running_state={},
    )

    assert captured["opus_stage2_output_dtype"] == "auto"
    assert captured["ep_has_fake_route"] is False


def test_aiter_runner_adds_ep_tuning_sentinel_for_legacy_aiter(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", top_k=2))
    expert_mask = torch.tensor([1, 1, 0], dtype=torch.int32)

    runner.run(
        _runner_input(),
        _quant_info(expert_mask=expert_mask),
        running_state={},
    )

    assert captured["topk_ids"].shape[-1] == 3
    assert captured["topk_ids"][0, -1].item() == expert_mask.numel()
    assert captured["topk_weight"][0, -1].item() == 0
    assert captured["expert_mask"].tolist() == [1, 1, 0, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
