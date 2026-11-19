"""Unit tests for srt/layers/quantization/mxfp4_aiter_a8w4_moe.py (gfx942 A8W4 MoE)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")

# MiMo-V2.6-Flash TP8 experts: a8w4_moe only runs shapes of pyhip's tuned table.
E, TOPK, HIDDEN, INTER = 256, 8, 4096, 256
FP4_DTYPE = torch.float4_e2m1fn_x2


def _is_gfx942():
    if not (is_hip() and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


class TestMxfp4AiterA8W4MoEMethod(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _is_gfx942():
            raise unittest.SkipTest("A8W4 FlyDSL MoE kernels require gfx942 (MI300/MI325)")
        import aiter
        from aiter import QuantType, dtypes

        from sglang.srt.layers.quantization import mxfp4_aiter_a8w4_moe as mod

        cls.mod = mod
        torch.manual_seed(0)
        tq = aiter.get_torch_quant(QuantType.per_1x32)
        w1 = torch.randn((E, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device="cuda")
        w2 = torch.randn((E, HIDDEN, INTER), dtype=torch.bfloat16, device="cuda")
        w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
        # Checkpoint layout as loaded by Fp8MoEMethod (store_dtype="mxfp4"):
        # raw uint8 bytes, separated [gate; up] w13 rows, e8m0 scale bytes.
        cls.w13 = w1_qt.view(torch.uint8).view(E, 2 * INTER, HIDDEN // 2)
        cls.w2 = w2_qt.view(torch.uint8).view(E, HIDDEN, INTER // 2)
        cls.w13_scale = w1_scale.view(torch.uint8).view(E, 2 * INTER, HIDDEN // 32)
        cls.w2_scale = w2_scale.view(torch.uint8).view(E, HIDDEN, INTER // 32)

    def _make_layer(self):
        layer = torch.nn.Module()
        for name, t in (
            ("w13_weight", self.w13),
            ("w2_weight", self.w2),
            ("w13_weight_scale_inv", self.w13_scale),
            ("w2_weight_scale_inv", self.w2_scale),
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(t.clone(), requires_grad=False)
            )
        layer.dispatcher = MagicMock()
        method = self.mod.Mxfp4AiterA8W4MoEMethod(MagicMock(), prefix="experts")
        method.process_weights_after_loading(layer)
        return method, layer

    def test_process_weights_after_loading(self):
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
        from pyhip.contrib.flydsl.a8w4 import prepare_weights

        # aiter's a16w4 preshuffle, then pyhip's fast-decode rewrite.
        ref = prepare_weights(
            shuffle_weight_a16w4(self.w13.view(FP4_DTYPE), 16, False),
            shuffle_weight_a16w4(self.w2.view(FP4_DTYPE), 16, False),
            shuffle_scale_a16w4(self.w13_scale.view(-1, HIDDEN // 32), E, False),
            shuffle_scale_a16w4(self.w2_scale.view(-1, INTER // 32), E, False),
        )
        _, layer = self._make_layer()
        aw = layer.all_weights
        self.assertEqual(aw.keys(), ref.keys())
        for key, value in ref.items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(aw[key].dtype, value.dtype, key)
                self.assertTrue(torch.equal(aw[key], value), key)
            else:
                self.assertEqual(aw[key], value, key)
        # Per-output-channel max e8m0 code, from the unshuffled checkpoint scales.
        torch.testing.assert_close(
            aw["w1_exp"], self.w13_scale.amax(-1).int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            aw["w2_exp"], self.w2_scale.amax(-1).int(), rtol=0, atol=0
        )

        # The kernel buffers alias the layer storage (no second weight copy).
        for key, param in (
            ("w1", layer.w13_weight),
            ("w2", layer.w2_weight),
            ("w1_scale", layer.w13_weight_scale_inv),
            ("w2_scale", layer.w2_weight_scale_inv),
        ):
            self.assertEqual(aw[key].data_ptr(), param.data_ptr(), key)
        for w in (layer.w13_weight, layer.w2_weight):
            self.assertEqual(w.dtype, FP4_DTYPE)
            self.assertTrue(w.is_shuffled)
            self.assertFalse(w.gate_up_interleave)
        layer.dispatcher.set_quant_config.assert_called_once_with(
            {"weight_dtype": FP4_DTYPE, "use_mxfp8": False}
        )

    def test_apply_matches_torch_reference(self):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_topk, torch_moe_stage1, torch_moe_stage2

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        w1 = self.w13.view(dtypes.fp4x2)
        w2 = self.w2.view(dtypes.fp4x2)
        method, layer = self._make_layer()
        for tokens in (1, 37, 256):
            with self.subTest(tokens=tokens):
                x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
                score = torch.randn((tokens, E), dtype=torch.bfloat16, device="cuda")
                topk_weights, topk_ids = fused_topk(x, score, TOPK, True)

                o1 = torch_moe_stage1(
                    x,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    dtype=torch.bfloat16,
                    activation=ActivationType.Silu,
                    quant_type=QuantType.per_1x32,
                    a1_scale=None,
                    w1_scale=self.w13_scale.view(dtypes.fp8_e8m0),
                    doweight=False,
                )
                ref = torch_moe_stage2(
                    o1.view(tokens, TOPK, INTER),
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    dtype=torch.bfloat16,
                    quant_type=QuantType.per_1x32,
                    w2_scale=self.w2_scale.view(dtypes.fp8_e8m0),
                    a2_scale=None,
                    doweight=True,
                )

                dispatch_output = SimpleNamespace(
                    hidden_states=x,
                    topk_output=SimpleNamespace(
                        topk_weights=topk_weights, topk_ids=topk_ids
                    ),
                )
                combine = method.apply(layer, dispatch_output)
                self.assertIsInstance(combine, StandardCombineInput)
                out = combine.hidden_states
                self.assertEqual(tuple(out.shape), (tokens, HIDDEN))
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertFalse(out.isnan().any().item())
                self.assertLess(_cos_diff(ref, out), 1e-2)


if __name__ == "__main__":
    unittest.main()
