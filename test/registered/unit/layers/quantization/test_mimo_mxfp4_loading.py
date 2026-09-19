import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import is_mimo_v2_mxfp4
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMiMoMXFP4Loading(CustomTestCase):
    def test_detects_native_mxfp4_checkpoint(self):
        config = {
            "architectures": ["MiMoV2ForCausalLM"],
            "quantization_config": {
                "quant_method": "fp8",
                "store_dtype": "mxfp4",
            },
        }
        self.assertTrue(is_mimo_v2_mxfp4(config))

        config["quantization_config"]["store_dtype"] = "fp8"
        self.assertFalse(is_mimo_v2_mxfp4(config))

    def test_mxfp4_checkpoint_uses_raw_byte_storage(self):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            is_fp4_experts=True,
            store_dtype="mxfp4",
        )
        method = Fp8MoEMethod(config)
        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(is_gated=True)

        with patch(
            "sglang.srt.layers.quantization.fp8.get_tensor_model_parallel_world_size",
            return_value=1,
        ):
            method.create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=128,
                intermediate_size_per_partition=128,
                params_dtype=torch.bfloat16,
            )

        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale_inv.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
