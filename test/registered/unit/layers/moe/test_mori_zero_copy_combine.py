import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    _pre_permute_deepep_to_aiter,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import moriep
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _dispatcher(monkeypatch, enabled=True):
    monkeypatch.setitem(
        sys.modules,
        "mori",
        SimpleNamespace(
            ops=SimpleNamespace(EpDispatchCombineKernelType=SimpleNamespace(IntraNode=0))
        ),
    )
    obj = moriep._MoriEPDispatcherImplNormal.__new__(moriep._MoriEPDispatcherImplNormal)
    obj.zero_copy_combine = enabled
    obj._combine_input_buffer = None
    obj._combine_kwargs = {"use_external_inp_buf": 0} if enabled else {}
    obj.dispatch_dtype = moriep.DispatchDtype.fp8
    obj.combine_dtype = moriep.CombineDtype.bf16
    obj.params_dtype = torch.bfloat16
    obj.hidden_size = 4
    obj._comm_stream = None
    buffer = torch.empty((8, 4), dtype=torch.bfloat16)
    obj._mori_op = SimpleNamespace(
        config=SimpleNamespace(kernel_type=0),
        get_registered_combine_input_buffer=Mock(return_value=buffer),
        combine=Mock(return_value=(buffer,)),
    )
    return obj, buffer


@pytest.mark.parametrize("enabled", [False, True])
def test_registered_buffer_is_cached_only_when_enabled(monkeypatch, enabled):
    obj, buffer = _dispatcher(monkeypatch, enabled)
    obj._prepare_combine_input_buffer()
    obj._prepare_combine_input_buffer()
    assert obj._mori_op.get_registered_combine_input_buffer.call_count == int(enabled)
    assert obj._combine_input_buffer is (buffer if enabled else None)


@pytest.mark.parametrize("enabled", [False, True])
def test_combine_mode_and_output_address(monkeypatch, enabled):
    obj, buffer = _dispatcher(monkeypatch, enabled)
    obj._prepare_combine_input_buffer()
    output = buffer[:3] if enabled else torch.empty_like(buffer)
    ids = torch.zeros((1, 2), dtype=torch.int32)
    obj._combine_core(output, ids, None, None)
    obj._mori_op.combine.assert_called_once_with(output, None, ids, **obj._combine_kwargs)
    if enabled:
        with pytest.raises(ValueError, match="registered buffer"):
            obj._combine_core(buffer[1:], ids, None, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dispatch_dtype", moriep.DispatchDtype.bf16),
        ("combine_dtype", moriep.CombineDtype.fp8),
        ("params_dtype", torch.float32),
    ],
)
def test_registered_buffer_rejects_unsupported_dtype(monkeypatch, field, value):
    obj, _ = _dispatcher(monkeypatch)
    setattr(obj, field, value)
    with pytest.raises(ValueError, match="intranode FP8"):
        obj._prepare_combine_input_buffer()
    obj._mori_op.get_registered_combine_input_buffer.assert_not_called()


@pytest.mark.parametrize("unsupported", ["auto", "tbo", "sdma", "runner"])
def test_shared_output_rejects_unsupported_execution(monkeypatch, unsupported):
    monkeypatch.setenv("SGLANG_MORI_ZERO_COPY_COMBINE", "1")
    monkeypatch.setenv("MORI_ENABLE_SDMA", "1" if unsupported == "sdma" else "0")
    monkeypatch.setattr(moriep, "is_tbo_enabled", lambda: unsupported == "tbo")
    monkeypatch.setattr(
        moriep,
        "get_moe_runner_backend",
        lambda: (
            moriep.MoeRunnerBackend.AUTO
            if unsupported == "runner"
            else moriep.MoeRunnerBackend.AITER
        ),
    )
    mode = moriep.DeepEPMode.AUTO if unsupported == "auto" else moriep.DeepEPMode.NORMAL
    with pytest.raises(ValueError, match="requires AITER"):
        moriep.MoriEPDispatcher(None, 8, deepep_mode=mode)


def test_shared_output_rejects_internode(monkeypatch):
    obj, _ = _dispatcher(monkeypatch)
    obj._mori_op.config.kernel_type = 1
    with pytest.raises(ValueError, match="intranode FP8"):
        obj._prepare_combine_input_buffer()


@pytest.mark.parametrize("cap", [0, 3])
def test_mori_pre_permute_keeps_shared_buffer_aligned_with_input(monkeypatch, cap):
    monkeypatch.setenv("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", str(cap))
    x = torch.empty((8, 4), dtype=torch.float8_e4m3fnuz)
    ids = torch.zeros((8, 2), dtype=torch.int32)
    weights = torch.ones((8, 2))
    output = torch.empty((8, 4), dtype=torch.bfloat16)
    dispatch = moriep.MoriEPNormalDispatchOutput(
        x,
        torch.ones((8, 1)),
        ids,
        weights,
        torch.tensor([8]),
        ids,
        weights,
        torch.bfloat16,
        output,
    )
    info = AiterMoeQuantInfo(torch.empty(0), torch.empty(0), AiterQuantType.PER_128X128)
    inp = _pre_permute_deepep_to_aiter(dispatch, info, MoeRunnerConfig(), {})
    assert inp.hidden_states.shape == inp.output_buffer.shape == (cap or 8, 4)
    assert inp.output_buffer.data_ptr() == output.data_ptr()
