from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import mimo_v2


def _decoder_stub(weight_dtype: torch.dtype):
    layer = mimo_v2.MiMoV2DecoderLayer.__new__(mimo_v2.MiMoV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(weight=torch.empty(1, dtype=weight_dtype))
    )
    return layer


@pytest.mark.parametrize(
    ("enabled", "gfx95", "gfx942", "weight_dtype", "expected"),
    [
        (False, True, False, torch.float8_e4m3fn, ""),
        (True, False, False, torch.float8_e4m3fn, ""),
        (True, False, True, torch.float8_e4m3fn, "fp8"),
        (True, True, False, torch.bfloat16, ""),
        (True, True, False, torch.float8_e4m3fn, "fp8"),
    ],
)
def test_mimo_fused_rms_qkv_quant_selector(
    monkeypatch, enabled, gfx95, gfx942, weight_dtype, expected
):
    monkeypatch.setattr(mimo_v2, "is_gfx95_supported", lambda: gfx95)
    monkeypatch.setattr(mimo_v2, "is_gfx942_supported", lambda: gfx942)
    layer = _decoder_stub(weight_dtype)

    with mimo_v2.envs.SGLANG_MIMO_FUSED_RMS_QKV_QUANT.override(enabled):
        assert layer._detect_fused_rms_qkv_quant_format() == expected


def test_mimo_prepare_attn_forwards_selected_qkv_quant_format():
    calls = []

    class State(SimpleNamespace):
        def update(self, values):
            self.__dict__.update(values)

    layer = _decoder_stub(torch.float8_e4m3fn)
    layer._fused_rms_qkv_quant_format = "fp8"
    layer.layer_communicator = SimpleNamespace(
        prepare_attn=lambda *args: (calls.append(args) or ("hidden", "residual"))
    )
    state = State()
    forward_batch = object()

    layer.op_comm_prepare_attn(
        state=state,
        positions="positions",
        hidden_states="hidden_states",
        forward_batch=forward_batch,
        residual="residual",
        tbo_subbatch_index=2,
    )

    assert calls == [("hidden_states", "residual", forward_batch, "fp8")]
    assert state.hidden_states_after_comm_pre_attn == "hidden"
    assert state.residual_after_input_ln == "residual"
    assert state.positions == "positions"
    assert state.tbo_subbatch_index == 2


@pytest.mark.parametrize("num_tokens", [0, 7])
def test_mimo_hidden_num_tokens_accepts_fused_qkv_tuple(num_tokens):
    quantized = torch.empty(num_tokens, 6144, dtype=torch.float8_e4m3fn)
    scale = torch.empty(num_tokens, 48, dtype=torch.float32)

    assert mimo_v2._mimo_hidden_num_tokens(quantized) == num_tokens
    assert mimo_v2._mimo_hidden_num_tokens((quantized, scale)) == num_tokens
