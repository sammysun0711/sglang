"""Unit tests for srt/layers/quantization/mxfp4_aiter_ep_a8w4_moe.py (gfx942 EP A8W4
MoE)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd")

EP, E_LOCAL, TOPK, HIDDEN, INTER = 4, 32, 8, 2048, 2048
E_GLOBAL = EP * E_LOCAL
FP4_DTYPE = torch.float4_e2m1fn_x2


def _is_gfx942():
    if not (is_hip() and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


class TestMxfp4AiterEpA8W4MoEMethod(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _is_gfx942():
            raise unittest.SkipTest("EP A8W4 FlyDSL MoE kernels require gfx942")
        import aiter
        from aiter import QuantType, dtypes

        from sglang.srt.layers.quantization import mxfp4_aiter_ep_a8w4_moe as mod

        cls.mod = mod
        torch.manual_seed(0)
        tq = aiter.get_torch_quant(QuantType.per_1x32)
        w1 = torch.randn((E_LOCAL, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device="cuda")
        w2 = torch.randn((E_LOCAL, HIDDEN, INTER), dtype=torch.bfloat16, device="cuda")
        w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
        # Checkpoint layout as loaded by Fp8MoEMethod (store_dtype="mxfp4").
        cls.w13 = w1_qt.view(torch.uint8).view(E_LOCAL, 2 * INTER, HIDDEN // 2)
        cls.w2 = w2_qt.view(torch.uint8).view(E_LOCAL, HIDDEN, INTER // 2)
        cls.w13_scale = w1_scale.view(torch.uint8).view(E_LOCAL, 2 * INTER, HIDDEN // 32)
        cls.w2_scale = w2_scale.view(torch.uint8).view(E_LOCAL, HIDDEN, INTER // 32)

    def _make_layer(self, rank):
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
        layer.moe_ep_rank = rank
        method = self.mod.Mxfp4AiterEpA8W4MoEMethod(MagicMock(), prefix="experts")
        method.process_weights_after_loading(layer)
        return method, layer

    def test_process_weights_after_loading(self):
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        from sglang.srt.layers.quantization.flydsl_moe_ep_w4a8_gfx942 import ep8_moe

        _, layer = self._make_layer(0)
        ref = ep8_moe.prepare_weights(
            shuffle_weight_a16w4(self.w13.view(FP4_DTYPE), 16, False),
            shuffle_weight_a16w4(self.w2.view(FP4_DTYPE), 16, False),
            shuffle_scale_a16w4(self.w13_scale.view(-1, HIDDEN // 32), E_LOCAL, False),
            shuffle_scale_a16w4(self.w2_scale.view(-1, INTER // 32), E_LOCAL, False),
        )
        W = layer.ep8_weights
        for key in ("w1", "w2", "w1_scale", "w2_scale", "w1_exp", "w2_exp"):
            self.assertTrue(torch.equal(W[key], ref[key]), key)
        # Per-output-channel max e8m0 code, from the unshuffled checkpoint scales.
        self.assertTrue(torch.equal(W["w1_exp"].view(-1), self.w13_scale.amax(-1).int().view(-1)))
        self.assertTrue(torch.equal(W["w2_exp"].view(-1), self.w2_scale.amax(-1).int().view(-1)))
        # The kernel buffers alias the layer storage (no second weight copy).
        for key, param in (
            ("w1", layer.w13_weight),
            ("w2", layer.w2_weight),
            ("w1_scale", layer.w13_weight_scale_inv),
            ("w2_scale", layer.w2_weight_scale_inv),
        ):
            self.assertEqual(W[key].data_ptr(), param.data_ptr(), key)
        layer.dispatcher.set_quant_config.assert_called_once_with(
            {"weight_dtype": FP4_DTYPE, "use_mxfp8": False}
        )

    def _reference(self, x, topk_weights, topk_ids, rank):
        """bf16-activation reference of this rank's local experts."""
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

        tokens = x.shape[0]
        loc = topk_ids.long() - rank * E_LOCAL
        is_local = (loc >= 0) & (loc < E_LOCAL)
        local_ids = torch.where(is_local, loc, 0).to(torch.int32)
        local_w = torch.where(is_local, topk_weights, 0.0)
        w1 = self.w13.view(dtypes.fp4x2)
        w2 = self.w2.view(dtypes.fp4x2)
        o1 = torch_moe_stage1(
            x, w1, w2, local_w, local_ids,
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            a1_scale=None,
            w1_scale=self.w13_scale.view(dtypes.fp8_e8m0),
            doweight=False,
        )
        return torch_moe_stage2(
            o1.view(tokens, TOPK, INTER), w1, w2, local_w, local_ids,
            dtype=torch.bfloat16,
            quant_type=QuantType.per_1x32,
            w2_scale=self.w2_scale.view(dtypes.fp8_e8m0),
            a2_scale=None,
            doweight=True,
        )

    def test_apply_matches_torch_reference(self):
        from aiter.fused_moe import fused_topk

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        for rank in (0, EP - 1):
            method, layer = self._make_layer(rank)
            # token counts hit every auto config: t16 / s16 / s32 / m64 / l128
            for tokens in (1, 37, 700, 1500, 3000):
                with self.subTest(rank=rank, tokens=tokens):
                    x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
                    score = torch.randn((tokens, E_GLOBAL), dtype=torch.bfloat16, device="cuda")
                    topk_weights, topk_ids = fused_topk(x, score, TOPK, True)
                    ref = self._reference(x, topk_weights, topk_ids, rank)
                    dispatch_output = SimpleNamespace(
                        hidden_states=x,
                        hidden_states_scale=None,
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
                    # tokens without a local route must come out exactly 0
                    no_local = ~(
                        (topk_ids >= rank * E_LOCAL) & (topk_ids < (rank + 1) * E_LOCAL)
                    ).any(-1)
                    self.assertEqual(out[no_local].abs().max().item() if no_local.any() else 0, 0)
                    if ref.abs().max() > 0:
                        self.assertLess(_cos_diff(ref, out), 5e-3)

    def test_apply_chunks_large_batches(self):
        from aiter.fused_moe import fused_topk

        method, layer = self._make_layer(1)
        tokens = 300
        x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        score = torch.randn((tokens, E_GLOBAL), dtype=torch.bfloat16, device="cuda")
        tw, ti = fused_topk(x, score, TOPK, True)
        dispatch_output = SimpleNamespace(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=SimpleNamespace(topk_weights=tw, topk_ids=ti),
        )
        full = method.apply(layer, dispatch_output).hidden_states.float().clone()
        orig = self.mod.EP8_MAX_TOKENS
        try:
            self.mod.EP8_MAX_TOKENS = 128
            chunked = method.apply(layer, dispatch_output).hidden_states.float()
        finally:
            self.mod.EP8_MAX_TOKENS = orig
        self.assertLess(_cos_diff(full, chunked), 1e-4)

    def _mori_recv(self, rank, capacity, valid, seed):
        """A MORI receive buffer: rows [0, valid) are tokens with >= 1 local route
        (global top-k ids), the rest is stale garbage that must be ignored."""
        from aiter.fused_moe import fused_topk

        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((capacity, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=g)
        score = torch.randn(
            (capacity, E_GLOBAL), dtype=torch.bfloat16, device="cuda", generator=g
        )
        tw, ti = fused_topk(x, score, TOPK, True)
        ti = ti.to(torch.int32)
        lo, hi = rank * E_LOCAL, (rank + 1) * E_LOCAL
        has_local = ((ti >= lo) & (ti < hi)).any(-1)
        forced = torch.randint(lo, hi, (capacity,), device="cuda", generator=g)
        ti[:, 0] = torch.where(has_local, ti[:, 0], forced.to(torch.int32))
        return x, tw.float().contiguous(), ti.contiguous()

    def test_apply_mori_receive_buffer(self):
        from aiter import QuantType, get_hip_quant

        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPLLDispatchOutput,
            MoriEPNormalCombineInput,
            MoriEPNormalDispatchOutput,
        )

        quant_1x128 = get_hip_quant(QuantType.per_1x128)
        rank, capacity = 1, 1024
        method, layer = self._make_layer(rank)
        for dispatch_dtype in ("fp8", "bf16"):
            for valid, max_tokens in ((1, None), (300, None), (1000, 128)):
                with self.subTest(dtype=dispatch_dtype, valid=valid, chunk=max_tokens):
                    x, tw, ti = self._mori_recv(rank, capacity, valid, seed=valid)
                    if dispatch_dtype == "fp8":
                        # the MORI dispatcher's blockwise quant
                        xq, xs = quant_1x128(x, quant_dtype=torch.float8_e4m3fnuz)
                        x_ref = (
                            (xq.float().view(capacity, -1, 128) * xs.view(capacity, -1, 1))
                            .view(capacity, HIDDEN)
                            .to(torch.bfloat16)
                        )
                    else:
                        xq, xs, x_ref = x, None, x
                    origin_ids, origin_w = MagicMock(), MagicMock()
                    cls_in, cls_out = (
                        (MoriEPNormalDispatchOutput, MoriEPNormalCombineInput)
                        if valid != 300
                        else (MoriEPLLDispatchOutput, MoriEPLLCombineInput)
                    )
                    dispatch_output = cls_in(
                        hidden_states=xq,
                        hidden_states_scale=xs,
                        topk_ids=ti,
                        topk_weights=tw,
                        num_recv_tokens_per_expert=torch.tensor(
                            [valid], dtype=torch.int32, device="cuda"
                        ),
                        origin_topk_ids=origin_ids,
                        origin_topk_weights=origin_w,
                        out_dtype=torch.bfloat16,
                    )
                    orig = self.mod.EP8_MAX_TOKENS
                    try:
                        if max_tokens:
                            self.mod.EP8_MAX_TOKENS = max_tokens
                        combine = method.apply(layer, dispatch_output)
                    finally:
                        self.mod.EP8_MAX_TOKENS = orig
                    self.assertIsInstance(combine, cls_out)
                    self.assertIs(combine.topk_ids, origin_ids)
                    self.assertIs(combine.topk_weights, origin_w)
                    out = combine.hidden_states
                    self.assertEqual(tuple(out.shape), (capacity, HIDDEN))
                    self.assertEqual(out.dtype, torch.bfloat16)
                    out = out[:valid]
                    self.assertFalse(out.isnan().any().item())
                    ref = self._reference(x_ref[:valid], tw[:valid], ti[:valid], rank)
                    self.assertLess(_cos_diff(ref, out), 5e-3)

    def test_mori_dyn_cuda_graph_replay(self):
        """One capture serves any device-side valid count (no host sync)."""
        from aiter import QuantType, get_hip_quant

        from sglang.srt.layers.quantization.flydsl_moe_ep_w4a8_gfx942 import ep8_moe

        quant_1x128 = get_hip_quant(QuantType.per_1x128)
        rank, capacity = 2, 512
        _, layer = self._make_layer(rank)
        W = layer.ep8_weights
        x, tw, ti = self._mori_recv(rank, capacity, capacity, seed=1)
        xq, xs = quant_1x128(x, quant_dtype=torch.float8_e4m3fnuz)
        m_valid = torch.tensor([64], dtype=torch.int32, device="cuda")

        def run():
            # est below the valid count for some replays: WGs stride over tiles
            return ep8_moe.ep8_moe_dyn(xq, xs, W, tw, ti, m_valid, rank=rank, est_tokens=128)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = run()
        for valid in (64, 1, 400, 0, 512):
            with self.subTest(valid=valid):
                x2, tw2, ti2 = self._mori_recv(rank, capacity, capacity, seed=10 + valid)
                xq2, xs2 = quant_1x128(x2, quant_dtype=torch.float8_e4m3fnuz)
                xq.copy_(xq2)
                xs.copy_(xs2)
                tw.copy_(tw2)
                ti.copy_(ti2)
                m_valid.fill_(valid)
                graph.replay()
                eager = run()
                if valid:
                    self.assertLess(_cos_diff(graph_out[:valid], eager[:valid]), 1e-4)


if __name__ == "__main__":
    unittest.main()
