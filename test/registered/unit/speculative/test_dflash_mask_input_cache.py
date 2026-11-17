"""Draft mask inputs retain graph storage across steps and batch changes."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as graph_module
from sglang.srt.speculative.dflash_input_embedding import (
    DFlashGraphInputCache,
    DFlashInputEmbeddings,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


class TensorOperations(TorchDispatchMode):
    def __init__(self):
        self.copies = []
        self.allocations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.copy_.default:
            self.copies.append(tuple(args[0].shape))
        if func._schema.name in (
            "aten::empty",
            "aten::empty_like",
            "aten::empty_strided",
            "aten::clone",
            "aten::repeat",
        ):
            self.allocations.append(func._schema.name)
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block", [4, 8])
def test_repeated_patches_match_lookup_without_block_allocation(dtype, block):
    generator = torch.Generator().manual_seed(12345)
    weight = torch.randn(32, 16, generator=generator).to(dtype)
    mask_id, capacity = 31, 12
    mask = weight[mask_id : mask_id + 1].clone()
    buffer = torch.empty(capacity * block, 16, dtype=dtype)
    pointer = buffer.data_ptr()
    cache = DFlashGraphInputCache()
    for step, batch in enumerate([0, 1, 12, 3, 9, 0, 12]):
        ids = torch.randint(0, 32, (batch,), generator=generator)
        block_ids = torch.full((batch, block), mask_id, dtype=torch.long)
        block_ids[:, 0] = ids
        inputs = DFlashInputEmbeddings(F.embedding(ids, weight), mask, block)
        with TensorOperations() as ops:
            cache.update(buffer, inputs)
        expected = F.embedding(block_ids, weight).view(batch * block, 16)
        torch.testing.assert_close(buffer[: batch * block], expected, rtol=0, atol=0)
        assert buffer.data_ptr() == pointer
        assert not ops.allocations
        assert ops.copies == (
            [(capacity * block, 16), (batch, 16)] if step == 0 else [(batch, 16)]
        )
        # Inactive mask rows must also remain valid when the batch grows again.
        torch.testing.assert_close(
            buffer.view(capacity, block, 16)[:, 1:],
            mask.view(1, 1, 16).expand(capacity, block - 1, 16),
            rtol=0,
            atol=0,
        )


def test_storage_replacement_mask_replacement_and_invalidation():
    inputs = DFlashInputEmbeddings(torch.ones(2, 4), torch.full((1, 4), 3.0), 8)
    cache = DFlashGraphInputCache()
    first, second = torch.empty(32, 4), torch.empty(32, 4)
    for buffer in (first, second):
        cache.update(buffer, inputs)
        torch.testing.assert_close(buffer[:16], inputs.materialize())
    second.fill_(-100)  # A full-input copy or capture may overwrite mask rows.
    cache.invalidate()
    cache.update(second, inputs)
    torch.testing.assert_close(second[:16], inputs.materialize())
    inputs.mask = torch.full((1, 4), 7.0)
    cache.update(second, inputs)
    torch.testing.assert_close(second[:16], inputs.materialize())


@pytest.mark.parametrize("batch", [0, 3])
def test_eager_materialization_can_be_modified_without_corrupting_cache(batch):
    inputs = DFlashInputEmbeddings(torch.ones(batch, 4), torch.full((1, 4), 3.0), 8)
    first = inputs.materialize()
    expected = first.clone()
    first.add_(99)
    torch.testing.assert_close(inputs.materialize(), expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.mask, torch.full((1, 4), 3.0))


def make_runner(capacity=12, block=8, hidden=4):
    runner = graph_module.DecodeCudaGraphRunner.__new__(
        graph_module.DecodeCudaGraphRunner
    )
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
        is_draft_worker=True,
        hisparse_coordinator=None,
    )
    runner._dflash_input_cache = DFlashGraphInputCache()
    runner.buffers = SimpleNamespace(
        input_embeds=torch.empty(capacity * block, hidden),
        input_ids=torch.empty(capacity * block, dtype=torch.long),
        positions=torch.empty(capacity * block, dtype=torch.long),
    )
    runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
    runner.require_mlp_tp_gather = False
    runner.enable_tbo_cuda_graph = False
    runner.enable_pdmux = False
    runner.is_encoder_decoder = False
    runner.num_tokens_per_bs = block
    runner.capture_bs = [4, 8, 12]
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner.seq_len_fill_value = 0
    runner.recapture_if_needed = lambda batch: None
    runner._resolve_lora_variant = lambda batch: "nolora"
    runner._make_graph_key = lambda bs, stream, variant: bs
    runner._pad_to_bucket = lambda bs, sizes: next(n for n in sizes if n >= bs)
    runner.buffer_registry = SimpleNamespace(fill_from=lambda *args, **kwargs: None)
    runner.attn_backend = SimpleNamespace(
        init_forward_metadata_out_graph=lambda fb: None
    )
    return runner


@pytest.mark.parametrize("preplanned", [False, True])
def test_both_replay_prepare_paths_patch_graph_storage(monkeypatch, preplanned):
    monkeypatch.setattr(graph_module, "build_replay_fb_view", lambda **kw: None)
    runner = make_runner()
    mask = torch.full((1, 4), 3.0)
    for batch in [8, 3, 10]:
        inputs = DFlashInputEmbeddings(torch.full((batch, 4), float(batch)), mask, 8)
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=batch,
            input_ids=torch.arange(batch * 8),
            req_pool_indices=torch.arange(batch),
            seq_lens=torch.ones(batch, dtype=torch.int32),
            out_cache_loc=torch.arange(batch * 8),
            seq_lens_sum=batch,
            positions=torch.arange(batch * 8),
            dflash_input_embeds=inputs,
        )
        fb.needs_forward_metadata_init = lambda: not preplanned
        if preplanned:
            runner.bs = next(n for n in runner.capture_bs if n >= batch)
            runner.raw_num_token = batch * 8
        runner.replay_prepare(fb)
        torch.testing.assert_close(
            runner.buffers.input_embeds[: batch * 8], inputs.materialize()
        )
        assert fb.input_embeds is None  # No intermediate full block was created.


def test_full_input_copy_invalidates_graph_mask_rows():
    runner = make_runner()
    inputs = DFlashInputEmbeddings(torch.ones(3, 4), torch.full((1, 4), 3.0), 8)
    fb = SimpleNamespace(dflash_input_embeds=inputs, input_embeds=None)
    runner._prepare_dflash_input_embeds(fb, 24)
    raw = torch.full((24, 4), -1.0)
    runner._prepare_dflash_input_embeds(
        SimpleNamespace(dflash_input_embeds=None, input_embeds=raw), 24
    )
    torch.testing.assert_close(runner.buffers.input_embeds[:24], raw)
    runner._prepare_dflash_input_embeds(fb, 24)
    torch.testing.assert_close(runner.buffers.input_embeds[:24], inputs.materialize())


def test_model_runner_materializes_only_on_eager_fallback(monkeypatch):
    from sglang.srt.model_executor import model_runner as module

    monkeypatch.setattr(module, "has_forward_context", lambda: True)
    inputs = DFlashInputEmbeddings(torch.ones(3, 4), torch.full((1, 4), 3.0), 8)
    fb = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=3,
        input_ids=torch.arange(24),
        req_pool_indices=torch.arange(3),
        seq_lens=torch.ones(3, dtype=torch.int32),
        out_cache_loc=torch.arange(24),
        seq_lens_sum=3,
        dflash_input_embeds=inputs,
    )
    fb.prepare_attn_tp_scatter_input = lambda runner: None
    runner = SimpleNamespace(
        device="cuda",
        decode_cuda_graph_runner=None,
        hisparse_coordinator=None,
        forward_extend=lambda batch, **kwargs: (batch.input_embeds, False),
    )
    out = module.ModelRunner._forward_raw(runner, fb, None)
    assert not out.can_run_graph
    torch.testing.assert_close(out.logits_output, inputs.materialize())
    assert fb.dflash_input_embeds is None


def test_model_runner_graph_path_never_materializes_full_input(monkeypatch):
    from sglang.srt.model_executor import model_runner as module

    monkeypatch.setattr(module, "has_forward_context", lambda: True)
    inputs = DFlashInputEmbeddings(torch.ones(3, 4), torch.full((1, 4), 3.0), 8)
    fb = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        batch_size=3,
        input_ids=torch.arange(24),
        req_pool_indices=torch.arange(3),
        seq_lens=torch.ones(3, dtype=torch.int32),
        out_cache_loc=torch.arange(24),
        seq_lens_sum=3,
        dflash_input_embeds=inputs,
    )

    def forbidden_materialization():
        raise AssertionError("Graph path must not construct a full embedding block")

    inputs.materialize = forbidden_materialization
    graph_runner = make_runner()

    def replay(batch, **kwargs):
        graph_runner._prepare_dflash_input_embeds(batch, 24)
        return graph_runner.buffers.input_embeds[:24]

    runner = SimpleNamespace(
        device="cuda",
        decode_cuda_graph_runner=SimpleNamespace(
            can_run=lambda batch: True, replay=replay
        ),
        hisparse_coordinator=None,
    )
    out = module.ModelRunner._forward_raw(runner, fb, None)
    assert out.can_run_graph
    assert fb.input_embeds is None
    assert out.logits_output.data_ptr() == graph_runner.buffers.input_embeds.data_ptr()
