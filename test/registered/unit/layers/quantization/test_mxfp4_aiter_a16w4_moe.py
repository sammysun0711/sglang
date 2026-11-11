"""Unit tests for srt/layers/quantization/mxfp4_aiter_a16w4_moe.py: the gfx942 A8W4
MoE of pyhip.contrib.flydsl.a8w4 under TP, EP with global top-k ids and EP on a
MORI receive buffer, with SiLU or SiTUv2 experts."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=240, suite="stage-b-test-1-gpu-small-amd")

FP4_DTYPE = torch.float4_e2m1fn_x2
# TP: all experts local, a small inter_dim slice (TP-family kernels, which also
# serve EP with an expert mask when the EP kernels do not fit).
TP_E, TP_TOPK, TP_HIDDEN, TP_INTER = 32, 4, 1024, 256
# EP: E_LOCAL of the EP * E_LOCAL global experts per rank (EP-family kernels).
EP, E_LOCAL, EP_TOPK, EP_HIDDEN, EP_INTER = 4, 32, 8, 2048, 2048
# 1 - cosine similarity vs the torch reference (measured ~1e-3: activation quant)
COS_TOL = 5e-3
# SiTUv2 runner config (also pyhip's ``situ`` dict): the clamp sits well inside the
# gate/up range of the random data (std ~32-45), so a missing clamp fails COS_TOL.
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0, swiglu_limit=16.0)


def _is_gfx942():
    if not (is_hip() and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


class _Checkpoint:
    """MXFP4 expert weights as loaded by Fp8MoEMethod (store_dtype="mxfp4"): raw
    uint8 bytes, separated [gate; up] w13 rows, e8m0 scales."""

    def __init__(self, experts, hidden, inter, topk):
        import aiter
        from aiter import QuantType, dtypes

        self.experts, self.hidden, self.inter, self.topk = experts, hidden, inter, topk
        tq = aiter.get_torch_quant(QuantType.per_1x32)
        w1 = torch.randn(
            (experts, 2 * inter, hidden), dtype=torch.bfloat16, device="cuda"
        )
        w2 = torch.randn((experts, hidden, inter), dtype=torch.bfloat16, device="cuda")
        w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
        e8m0 = torch.float8_e8m0fnu
        self.w13 = w1_qt.view(torch.uint8).view(experts, 2 * inter, hidden // 2)
        self.w2 = w2_qt.view(torch.uint8).view(experts, hidden, inter // 2)
        self.w13_scale = w1_scale.view(e8m0).view(experts, 2 * inter, hidden // 32)
        self.w2_scale = w2_scale.view(e8m0).view(experts, hidden, inter // 32)

    def routing(self, tokens, num_experts, generator=None):
        """bf16 tokens and their fused_topk routing over ``num_experts``."""
        from aiter.fused_moe import fused_topk

        kw = dict(dtype=torch.bfloat16, device="cuda", generator=generator)
        x = torch.randn((tokens, self.hidden), **kw)
        score = torch.randn((tokens, num_experts), **kw)
        topk_weights, topk_ids = fused_topk(x, score, self.topk, True)
        return x, topk_weights, topk_ids

    def mori_recv(self, rank, capacity, seed):
        """A MORI receive buffer: global top-k ids, every row routed to >= 1 of the
        rank's experts; rows past the device-side count must be ignored."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        x, topk_weights, topk_ids = self.routing(capacity, EP * self.experts, g)
        lo, hi = rank * self.experts, (rank + 1) * self.experts
        has_local = ((topk_ids >= lo) & (topk_ids < hi)).any(-1)
        forced = torch.randint(lo, hi, (capacity,), device="cuda", generator=g)
        topk_ids[:, 0] = torch.where(
            has_local, topk_ids[:, 0], forced.to(topk_ids.dtype)
        )
        return x, topk_weights, topk_ids

    def reference(self, x, topk_weights, topk_ids, expert_offset=0):
        """bf16-activation torch MoE over the experts [expert_offset, + experts)."""
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2

        local = topk_ids.long() - expert_offset
        is_local = (local >= 0) & (local < self.experts)
        ids = torch.where(is_local, local, 0).to(torch.int32)
        weights = torch.where(is_local, topk_weights, 0.0)
        w1, w2 = self.w13.view(dtypes.fp4x2), self.w2.view(dtypes.fp4x2)
        o1 = torch_moe_stage1(
            x,
            w1,
            w2,
            weights,
            ids,
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            a1_scale=None,
            w1_scale=self.w13_scale,
            doweight=False,
        )
        return torch_moe_stage2(
            o1.view(x.shape[0], self.topk, self.inter),
            w1,
            w2,
            weights,
            ids,
            dtype=torch.bfloat16,
            quant_type=QuantType.per_1x32,
            w2_scale=self.w2_scale,
            a2_scale=None,
            doweight=True,
        )

    def situ_reference(
        self, x, topk_weights, topk_ids, expert_offset=0, quant_x=True, situ=SITU
    ):
        """fp32 SiTUv2 MoE with the kernels' clamp and fp8 activation emulation
        (aiter's torch_moe_stage1 does not clamp situv2)."""
        from pyhip.contrib.flydsl.a8w4 import ref

        raw = dict(w1q=self.w13, w1s=self.w13_scale, w2q=self.w2, w2s=self.w2_scale)
        return ref.ref_moe(
            x, topk_weights, topk_ids, raw, expert_offset, "situv2", situ, quant_x
        )


def _mori_fp8(x):
    """MORI's blockwise fp8 dispatch quant (fp8 FNUZ + an f32 scale per (token,
    128)) and its dequantized bf16 value, the reference input."""
    from aiter import QuantType, get_hip_quant

    xq, xs = get_hip_quant(QuantType.per_1x128)(x, quant_dtype=torch.float8_e4m3fnuz)
    m = x.shape[0]
    x_ref = (xq.float().view(m, -1, 128) * xs.view(m, -1, 1)).view(m, -1)
    return xq, xs, x_ref.to(torch.bfloat16)


def _make_layer(ckpt, rank=0, ep_size=1, a2a=MoeA2ABackend.NONE, **runner_config):
    """A loaded FusedMoE-like layer, after create_moe_runner (``runner_config``:
    MoeRunnerConfig fields) and process_weights_after_loading."""
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.quantization import mxfp4_aiter_a16w4_moe as mod

    layer = torch.nn.Module()
    for name, t in (
        ("w13_weight", ckpt.w13),
        ("w2_weight", ckpt.w2),
        ("w13_weight_scale_inv", ckpt.w13_scale),
        ("w2_weight_scale_inv", ckpt.w2_scale),
    ):
        layer.register_parameter(
            name, torch.nn.Parameter(t.clone(), requires_grad=False)
        )
    layer.moe_ep_rank = rank
    layer.num_local_experts = ckpt.experts
    layer.num_experts = ckpt.experts * ep_size
    layer.top_k = ckpt.topk
    layer.dispatcher = MagicMock()
    method = mod.Mxfp4AiterA16W4MoEMethod(MagicMock(), prefix="experts")
    with patch.object(mod, "get_moe_a2a_backend", return_value=a2a), patch.object(
        mod, "MoeRunner"
    ):
        method.create_moe_runner(layer, MoeRunnerConfig(**runner_config))
        method.process_weights_after_loading(layer)
    return method, layer


def _standard_dispatch(x, topk_weights, topk_ids):
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        ),
    )


class TestMxfp4AiterA16W4MoEMethod(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _is_gfx942():
            raise unittest.SkipTest(
                "A8W4 FlyDSL MoE kernels require gfx942 (MI300/MI325)"
            )
        from sglang.srt.layers.quantization import mxfp4_aiter_a16w4_moe as mod

        cls.mod = mod
        torch.manual_seed(0)
        cls.tp = _Checkpoint(TP_E, TP_HIDDEN, TP_INTER, TP_TOPK)
        cls.ep = _Checkpoint(E_LOCAL, EP_HIDDEN, EP_INTER, EP_TOPK)

    def _apply_standard(self, method, layer, x, topk_weights, topk_ids):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        dispatch_output = _standard_dispatch(x, topk_weights, topk_ids)
        combine = method.apply(layer, dispatch_output)
        self.assertIsInstance(combine, StandardCombineInput)
        out = combine.hidden_states
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(out.isnan().any().item())
        return out

    def test_create_weights_checks_kernel_shape(self):
        fp8_method = MagicMock()
        method = self.mod.Mxfp4AiterA16W4MoEMethod(fp8_method, prefix="experts")
        layer = torch.nn.Module()
        layer.num_experts, layer.top_k = EP * E_LOCAL, EP_TOPK
        method.create_weights(layer, E_LOCAL, EP_HIDDEN, EP_INTER, torch.bfloat16)
        fp8_method.create_weights.assert_called_once_with(
            layer,
            E_LOCAL,
            EP_HIDDEN,
            EP_INTER,
            torch.bfloat16,
            fp4_scale_dtype=torch.float8_e8m0fnu,
        )
        with self.assertRaises(ValueError):  # hidden_size % 512
            method.create_weights(layer, E_LOCAL, 1000, EP_INTER, torch.bfloat16)
        with self.assertRaises(NotImplementedError):
            method.create_weights(
                layer, E_LOCAL, EP_HIDDEN, EP_INTER, torch.bfloat16, with_bias=True
            )
        # 64 local experts: the TP-family kernels with an expert mask serve
        # --moe-a2a-backend none, but MORI needs the EP kernels (<= 32 local).
        layer.num_experts = EP * 64
        method.create_weights(layer, 64, EP_HIDDEN, EP_INTER, torch.bfloat16)
        with patch.object(
            self.mod, "get_moe_a2a_backend", return_value=MoeA2ABackend.MORI
        ):
            with self.assertRaises(ValueError):
                method.create_weights(layer, 64, EP_HIDDEN, EP_INTER, torch.bfloat16)
            layer.num_experts = EP * E_LOCAL
            method.create_weights(layer, E_LOCAL, EP_HIDDEN, EP_INTER, torch.bfloat16)

    def test_process_weights_after_loading(self):
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
        from pyhip.contrib.flydsl.a8w4 import prepare_weights

        for ckpt, ep_size, a2a, dispatch_weight_dtype in (
            (self.tp, 1, MoeA2ABackend.NONE, FP4_DTYPE),
            (self.ep, EP, MoeA2ABackend.NONE, FP4_DTYPE),
            # MORI dispatches blockwise fp8, consumed directly by the kernels.
            (self.ep, EP, MoeA2ABackend.MORI, torch.float8_e4m3fnuz),
        ):
            with self.subTest(ep_size=ep_size, a2a=a2a.value):
                _, layer = _make_layer(ckpt, ep_size - 1, ep_size, a2a)
                E = ckpt.experts
                ref = prepare_weights(
                    shuffle_weight_a16w4(ckpt.w13.view(FP4_DTYPE), 16, False),
                    shuffle_weight_a16w4(ckpt.w2.view(FP4_DTYPE), 16, False),
                    shuffle_scale_a16w4(
                        ckpt.w13_scale.view(-1, ckpt.hidden // 32), E, False
                    ),
                    shuffle_scale_a16w4(
                        ckpt.w2_scale.view(-1, ckpt.inter // 32), E, False
                    ),
                )
                W = layer.a8w4_weights
                self.assertEqual(set(W), set(ref))
                for key, val in ref.items():
                    if isinstance(val, torch.Tensor):
                        self.assertTrue(torch.equal(W[key], val), key)
                    else:
                        self.assertEqual(W[key], val, key)
                # Per-output-channel max e8m0 code of the unshuffled checkpoint.
                for key, scale in (
                    ("w1_exp", ckpt.w13_scale),
                    ("w2_exp", ckpt.w2_scale),
                ):
                    exp = scale.view(torch.uint8).amax(-1).int()
                    self.assertTrue(torch.equal(W[key].view(-1), exp.view(-1)), key)
                # The kernel buffers alias the layer storage (no second copy).
                for key, param in (
                    ("w1", layer.w13_weight),
                    ("w2", layer.w2_weight),
                    ("w1_scale", layer.w13_weight_scale_inv),
                    ("w2_scale", layer.w2_weight_scale_inv),
                ):
                    self.assertEqual(W[key].data_ptr(), param.data_ptr(), key)
                for w in (layer.w13_weight, layer.w2_weight):
                    self.assertEqual(w.dtype, FP4_DTYPE)
                    self.assertTrue(w.is_shuffled)
                    self.assertFalse(w.gate_up_interleave)
                layer.dispatcher.set_quant_config.assert_called_once_with(
                    {"weight_dtype": dispatch_weight_dtype, "use_mxfp8": False}
                )

    def test_apply_tp(self):
        method, layer = _make_layer(self.tp)
        for tokens in (1, 37, 256):
            with self.subTest(tokens=tokens):
                x, tw, ti = self.tp.routing(tokens, TP_E)
                out = self._apply_standard(method, layer, x, tw, ti)
                ref = self.tp.reference(x, tw, ti)
                self.assertLess(_cos_diff(ref, out), COS_TOL)
        with self.subTest(tokens=0):  # an idle rank
            x = torch.empty((0, TP_HIDDEN), dtype=torch.bfloat16, device="cuda")
            tw = torch.empty((0, TP_TOPK), dtype=torch.float32, device="cuda")
            ti = torch.empty((0, TP_TOPK), dtype=torch.int32, device="cuda")
            self._apply_standard(method, layer, x, tw, ti)

    def test_apply_ep_global_ids(self):
        """--moe-a2a-backend none: every rank gets all tokens with global top-k
        ids. The TP shape runs the TP-family kernels with an expert mask (inter_dim
        too small for the EP kernels); the EP shape runs the EP kernels at every
        auto config (t16 / s16 / s32 / m64 / l128)."""
        from pyhip.contrib.flydsl.a8w4 import lookup_config

        for ckpt, family, token_counts in (
            (self.tp, "tp", (1, 37, 256)),
            (self.ep, "ep", (1, 37, 700, 1500, 3000)),
        ):
            num_experts = EP * ckpt.experts
            shape = (ckpt.hidden, ckpt.inter, ckpt.experts, num_experts, ckpt.topk)
            for rank in (0, EP - 1):
                method, layer = _make_layer(ckpt, rank, EP)
                lo, hi = rank * ckpt.experts, (rank + 1) * ckpt.experts
                for tokens in token_counts:
                    with self.subTest(family=family, rank=rank, tokens=tokens):
                        kind = lookup_config(shape, tokens)["kind"]
                        self.assertTrue(kind.startswith(family), kind)
                        x, tw, ti = ckpt.routing(tokens, num_experts)
                        out = self._apply_standard(method, layer, x, tw, ti)
                        # tokens without a local route come out exactly 0
                        no_local = ~((ti >= lo) & (ti < hi)).any(-1)
                        if no_local.any():
                            self.assertEqual(out[no_local].abs().max().item(), 0)
                        ref = ckpt.reference(x, tw, ti, lo)
                        if ref.abs().max() > 0:
                            self.assertLess(_cos_diff(ref, out), COS_TOL)

    def _mori_dispatch(self, cls, xq, xs, ti, tw, count):
        return cls(
            hidden_states=xq,
            hidden_states_scale=xs,
            topk_ids=ti,
            topk_weights=tw,
            num_recv_tokens_per_expert=count,
            origin_topk_ids=MagicMock(),
            origin_topk_weights=MagicMock(),
            out_dtype=torch.bfloat16,
        )

    def test_apply_mori_receive_buffer(self):
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPLLDispatchOutput,
            MoriEPNormalCombineInput,
            MoriEPNormalDispatchOutput,
        )

        rank, capacity = 1, 1024
        method, layer = _make_layer(self.ep, rank, EP, MoeA2ABackend.MORI)
        normal = (MoriEPNormalDispatchOutput, MoriEPNormalCombineInput)
        low_latency = (MoriEPLLDispatchOutput, MoriEPLLCombineInput)
        # (received rows, A8W4_DYN_MAX_TOKENS override, dispatch mode)
        cases = (
            (0, None, normal),
            (1, None, low_latency),
            (300, None, normal),
            (1000, 128, low_latency),
        )
        for dispatch_dtype in ("fp8", "bf16"):
            for valid, chunk, (cls_in, cls_out) in cases:
                with self.subTest(dtype=dispatch_dtype, valid=valid, chunk=chunk):
                    x, tw, ti = self.ep.mori_recv(rank, capacity, seed=valid)
                    if dispatch_dtype == "fp8":
                        xq, xs, x_ref = _mori_fp8(x)
                    else:
                        xq, xs, x_ref = x, None, x
                    count = torch.tensor([valid], dtype=torch.int32, device="cuda")
                    dispatch_output = self._mori_dispatch(cls_in, xq, xs, ti, tw, count)
                    with patch.object(
                        self.mod,
                        "A8W4_DYN_MAX_TOKENS",
                        chunk or self.mod.A8W4_DYN_MAX_TOKENS,
                    ):
                        combine = method.apply(layer, dispatch_output)
                    self.assertIsInstance(combine, cls_out)
                    self.assertIs(combine.topk_ids, dispatch_output.origin_topk_ids)
                    self.assertIs(
                        combine.topk_weights, dispatch_output.origin_topk_weights
                    )
                    out = combine.hidden_states
                    # capacity sized: MORI combine reads the received rows
                    self.assertEqual(tuple(out.shape), (capacity, EP_HIDDEN))
                    self.assertEqual(out.dtype, torch.bfloat16)
                    if valid:
                        out = out[:valid]
                        self.assertFalse(out.isnan().any().item())
                        ref = self.ep.reference(
                            x_ref[:valid], tw[:valid], ti[:valid], rank * E_LOCAL
                        )
                        self.assertLess(_cos_diff(ref, out), COS_TOL)

        with self.subTest(dtype="mxfp4"), self.assertRaises(NotImplementedError):
            x = torch.empty(
                (capacity, EP_HIDDEN // 2), dtype=torch.uint8, device="cuda"
            )
            method.apply(
                layer,
                self._mori_dispatch(
                    MoriEPNormalDispatchOutput, x.view(FP4_DTYPE), None, ti, tw, count
                ),
            )

    def test_mori_cuda_graph_replay(self):
        """One capture serves any device-side receive count (no host sync), also
        counts above the DP-metadata estimate the grids are sized for."""
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPNormalDispatchOutput,
        )

        rank, capacity = 2, 512
        method, layer = _make_layer(self.ep, rank, EP, MoeA2ABackend.MORI)
        x, tw, ti = self.ep.mori_recv(rank, capacity, seed=1)
        xq, xs, _ = _mori_fp8(x)
        count = torch.tensor([64], dtype=torch.int32, device="cuda")
        dispatch_output = self._mori_dispatch(
            MoriEPNormalDispatchOutput, xq, xs, ti, tw, count
        )
        # 4 attention-DP ranks, only the local token count (16) known: the
        # capture-time estimate is 64 valid rows.
        with patch.multiple(
            "sglang.srt.layers.dp_attention",
            is_dp_attention_enabled=MagicMock(return_value=True),
            get_attention_dp_size=MagicMock(return_value=4),
            get_dp_global_num_tokens=MagicMock(return_value=[16]),
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                method.apply(layer, dispatch_output)  # eager warm-up
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = method.apply(layer, dispatch_output).hidden_states
        for valid in (64, 1, 400, 0, 512):
            with self.subTest(valid=valid):
                x2, tw2, ti2 = self.ep.mori_recv(rank, capacity, seed=10 + valid)
                xq2, xs2, x_ref = _mori_fp8(x2)
                xq.copy_(xq2)
                xs.copy_(xs2)
                tw.copy_(tw2)
                ti.copy_(ti2)
                count.fill_(valid)
                graph.replay()
                if valid:
                    ref = self.ep.reference(
                        x_ref[:valid], tw2[:valid], ti2[:valid], rank * E_LOCAL
                    )
                    self.assertLess(_cos_diff(ref, graph_out[:valid]), COS_TOL)

    def test_apply_situv2(self):
        """activation="situv2" on every path: TP; EP with global ids on the TP
        family (+ expert mask) and on the EP family (non-TR / TR epilogues); MORI
        with fp8 dispatch, chunked."""
        from pyhip.contrib.flydsl.a8w4 import lookup_config

        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPNormalDispatchOutput,
        )

        situ = dict(activation="situv2", **SITU)
        method, layer = _make_layer(self.tp, **situ)
        self.assertEqual(method._act_kwargs, dict(act="situv2", situ=SITU))
        with self.subTest(path="tp"):
            x, tw, ti = self.tp.routing(37, TP_E)
            out = self._apply_standard(method, layer, x, tw, ti)
            ref = self.tp.situ_reference(x, tw, ti)
            self.assertLess(_cos_diff(ref, out), COS_TOL)
            no_clamp = dict(SITU, swiglu_limit=float("inf"))
            ref_no_clamp = self.tp.situ_reference(x, tw, ti, situ=no_clamp)
            self.assertGreater(_cos_diff(ref, ref_no_clamp), 10 * COS_TOL)

        rank = EP - 1
        for ckpt, token_counts in ((self.tp, (37,)), (self.ep, (37, 1500))):
            num_experts = EP * ckpt.experts
            shape = (ckpt.hidden, ckpt.inter, ckpt.experts, num_experts, ckpt.topk)
            method, layer = _make_layer(ckpt, rank, EP, **situ)
            for tokens in token_counts:
                cfg = lookup_config(shape, tokens, act="situv2")
                with self.subTest(path="ep", kind=cfg["kind"], tokens=tokens):
                    x, tw, ti = ckpt.routing(tokens, num_experts)
                    out = self._apply_standard(method, layer, x, tw, ti)
                    ref = ckpt.situ_reference(x, tw, ti, rank * ckpt.experts)
                    self.assertLess(_cos_diff(ref, out), COS_TOL)

        with self.subTest(path="mori"):
            capacity, valid = 1024, 1000
            method, layer = _make_layer(self.ep, 1, EP, MoeA2ABackend.MORI, **situ)
            x, tw, ti = self.ep.mori_recv(1, capacity, seed=3)
            xq, xs, x_ref = _mori_fp8(x)
            count = torch.tensor([valid], dtype=torch.int32, device="cuda")
            dispatch_output = self._mori_dispatch(
                MoriEPNormalDispatchOutput, xq, xs, ti, tw, count
            )
            with patch.object(self.mod, "A8W4_DYN_MAX_TOKENS", 256):
                out = method.apply(layer, dispatch_output).hidden_states[:valid]
            ref = self.ep.situ_reference(
                x_ref[:valid], tw[:valid], ti[:valid], E_LOCAL, quant_x=False
            )
            self.assertLess(_cos_diff(ref, out), COS_TOL)


class TestMxfp4AiterA16W4MoERunnerConfig(CustomTestCase):
    def test_create_moe_runner(self):
        from sglang.srt.layers.moe import MoeRunnerConfig
        from sglang.srt.layers.quantization import mxfp4_aiter_a16w4_moe as mod

        method = mod.Mxfp4AiterA16W4MoEMethod(MagicMock(), prefix="experts")
        with patch.object(mod, "MoeRunner") as runner:
            for a2a in (MoeA2ABackend.NONE, MoeA2ABackend.MORI):
                with patch.object(mod, "get_moe_a2a_backend", return_value=a2a):
                    method.create_moe_runner(torch.nn.Module(), MoeRunnerConfig())
            self.assertEqual(runner.call_count, 2)
            for kwargs in (
                dict(activation="gelu"),
                dict(is_gated=False),
                dict(apply_router_weight_on_input=True),
                dict(num_fused_shared_experts=1),
            ):
                with self.subTest(**kwargs), self.assertRaises(ValueError):
                    method.create_moe_runner(
                        torch.nn.Module(), MoeRunnerConfig(**kwargs)
                    )
            with self.subTest(a2a="deepep"), patch.object(
                mod, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
            ):
                with self.assertRaises(ValueError):
                    method.create_moe_runner(torch.nn.Module(), MoeRunnerConfig())
            self.assertEqual(runner.call_count, 2)

    def test_create_moe_runner_situv2(self):
        from sglang.srt.layers.moe import MoeRunnerConfig
        from sglang.srt.layers.quantization import mxfp4_aiter_a16w4_moe as mod

        method = mod.Mxfp4AiterA16W4MoEMethod(MagicMock(), prefix="experts")
        betas = dict(situ_beta=4.0, situ_linear_beta=25.0)
        inf = float("inf")
        with patch.object(mod, "MoeRunner"):
            self.assertFalse(hasattr(method, "_act_kwargs"))
            method.create_moe_runner(torch.nn.Module(), MoeRunnerConfig())
            self.assertEqual(method._act_kwargs, dict(act="silu"))
            # the clamp limit comes from either spelling; unset / 0 = no clamp
            for limits, want in (
                (dict(), inf),
                (dict(swiglu_limit=0.0), inf),
                (dict(swiglu_limit=64.0), 64.0),
                (dict(gemm1_clamp_limit=32.0), 32.0),
                (dict(swiglu_limit=inf), inf),
            ):
                with self.subTest(**limits):
                    config = MoeRunnerConfig(activation="situv2", **betas, **limits)
                    method.create_moe_runner(torch.nn.Module(), config)
                    self.assertEqual(
                        method._act_kwargs,
                        dict(act="situv2", situ=dict(**betas, swiglu_limit=want)),
                    )
            for bad in (
                dict(),
                dict(situ_beta=4.0),
                dict(betas, situ_beta=0.0),
                dict(betas, situ_linear_beta=-1.0),
                dict(betas, swiglu_limit=-1.0),
                dict(betas, gemm1_clamp_limit=float("nan")),
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    method.create_moe_runner(
                        torch.nn.Module(), MoeRunnerConfig(activation="situv2", **bad)
                    )


if __name__ == "__main__":
    unittest.main()
