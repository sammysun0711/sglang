from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.communicator import _Fp8TransposedScaleInput
from sglang.srt.layers.quantization import fp8, fp8_utils
from sglang.srt.models import mimo_v2


def _decoder_stub(weight_dtype: torch.dtype):
    layer = mimo_v2.MiMoV2DecoderLayer.__new__(mimo_v2.MiMoV2DecoderLayer)
    nn.Module.__init__(layer)
    quant_method = mimo_v2.Fp8LinearMethod.__new__(mimo_v2.Fp8LinearMethod)
    quant_method.block_quant = True
    quant_method.use_mxfp8 = quant_method.use_marlin = False
    quant_method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    quant_method.w8a8_block_fp8_linear = fp8_utils.aiter_w8a8_block_fp8_linear
    layer.self_attn = SimpleNamespace(
        qkv_proj=SimpleNamespace(
            weight=torch.empty(1, dtype=weight_dtype), quant_method=quant_method
        )
    )
    return layer


@pytest.mark.parametrize(
    ("enabled", "gfx95", "gfx942", "weight_dtype", "expected"),
    [
        (False, True, False, torch.float8_e4m3fn, ""),
        (True, False, False, torch.float8_e4m3fn, ""),
        (True, False, True, torch.float8_e4m3fn, "fp8"),
        (True, False, True, torch.float8_e4m3fnuz, "fp8"),
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


@pytest.mark.parametrize(
    "unsupported",
    ["triton", "non_block", "mxfp8", "marlin", "block_size", "other_method"],
)
def test_mimo_fused_qkv_rejects_incompatible_consumers(monkeypatch, unsupported):
    monkeypatch.setattr(mimo_v2, "is_gfx95_supported", lambda: True)
    layer = _decoder_stub(torch.float8_e4m3fn)
    qkv = layer.self_attn.qkv_proj
    method = qkv.quant_method
    if unsupported == "triton":
        method.w8a8_block_fp8_linear = fp8_utils.triton_w8a8_block_fp8_linear
    elif unsupported == "non_block":
        method.block_quant = False
    elif unsupported == "mxfp8":
        method.use_mxfp8 = True
    elif unsupported == "marlin":
        method.use_marlin = True
    elif unsupported == "block_size":
        method.quant_config.weight_block_size = [128, 64]
    else:
        qkv.quant_method = None
    with mimo_v2.envs.SGLANG_MIMO_FUSED_RMS_QKV_QUANT.override(True):
        assert layer._detect_fused_rms_qkv_quant_format() == ""


@pytest.mark.parametrize("compile_cpu", [False, True])
@pytest.mark.parametrize("with_extra", [False, True])
def test_tagged_fp8_input_reaches_linear_consumer(monkeypatch, compile_cpu, with_extra):
    monkeypatch.setattr(fp8, "use_intel_amx_backend", lambda _: False)
    qkv = _decoder_stub(torch.float8_e4m3fn).self_attn.qkv_proj
    qkv.weight_scale_inv = torch.ones(1)
    # Exercise the real tuple dispatch without launching a GEMM kernel.
    qkv.quant_method.w8a8_block_fp8_linear = (
        lambda input, input_scale, **kwargs: input.float() * input_scale
    )

    def forward(activation, scale):
        tensors = (activation, scale)
        if with_extra:
            tensors += (activation.to(torch.bfloat16),)
        hidden = _Fp8TransposedScaleInput(tensors)
        assert mimo_v2._mimo_hidden_num_tokens(hidden) == activation.shape[0]
        return qkv.quant_method.apply(qkv, hidden)

    if compile_cpu:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    activation = torch.ones(2, 128, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [2.0]])
    torch.testing.assert_close(forward(activation, scale), activation.float() * scale)


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
