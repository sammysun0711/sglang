"""Exercise the real worker methods with CPU model/graph stand-ins.

Kernel correctness and graph replay are covered separately by test_spec_topk1.
These tests verify accepted-row selection, fallback policy and token mapping.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import eagle_worker_v2 as module
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cpu_postprocess(
    logits,
    positions=None,
    draft_tokens=None,
    draft_token_column=0,
    input_row_indices=None,
):
    selected = logits if input_row_indices is None else logits[input_row_indices]
    indices = selected.argmax(-1, keepdim=True)
    if positions is not None:
        positions.add_(1)
    if draft_tokens is not None:
        draft_tokens[:, draft_token_column].copy_(indices[:, 0])
    return torch.ones_like(indices, dtype=torch.float32), indices


@pytest.fixture
def worker_context(monkeypatch):
    spec = SimpleNamespace(speculative_use_rejection_sampling=False)
    monkeypatch.setattr(module, "get_spec", lambda: spec)
    monkeypatch.setattr(module, "_is_hip", True)
    monkeypatch.setattr(module, "_is_cuda", False)
    for name in ("maybe_detect_nan", "maybe_detect_inf", "maybe_detect_oob"):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    fused = Mock(side_effect=_cpu_postprocess)
    monkeypatch.setattr(module, "draft_topk1_postprocess", fused)
    monkeypatch.setattr(
        module, "forward_context", lambda *args: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        module.IndexTopKShareState,
        "mtp_iteration",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    return spec, fused


def _extend_worker(monkeypatch, enabled, topk, graph_mode):
    worker = module.EagleDraftWorker.__new__(module.EagleDraftWorker)
    worker._hip_topk1_fusion = enabled
    worker.topk = topk
    worker.device = "cpu"
    worker.speculative_num_draft_tokens = 4
    worker.seed_dsa_topk_from_draft_extend = False
    worker.plan_stream = None
    worker.plan_stream_ctx = contextlib.nullcontext()
    logits = torch.full((8, 16), -5.0)
    logits[torch.arange(8), torch.arange(8) + 2] = 5.0
    hidden = torch.arange(24).reshape(8, 3).float()

    def output():
        return SimpleNamespace(
            next_token_logits=logits.clone(), hidden_states=hidden.clone()
        )

    worker.draft_runner = SimpleNamespace(
        canary_manager=None,
        forward=Mock(side_effect=lambda batch: SimpleNamespace(logits_output=output())),
    )
    worker.cuda_graph_runner_for_draft_extend = SimpleNamespace(
        can_run_graph=lambda batch: graph_mode,
        execute=Mock(side_effect=lambda batch: output()),
    )
    monkeypatch.setattr(
        module,
        "prepare_for_draft_extend",
        lambda *args, **kwargs: SimpleNamespace(input_ids=torch.arange(8)),
    )
    return worker, logits, hidden


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize(
    "enabled,topk,rejection",
    [(True, 1, False), (False, 1, False), (True, 2, False), (True, 1, True)],
)
def test_decode_extend_selected_rows_and_fallback(
    monkeypatch, worker_context, graph_mode, enabled, topk, rejection
):
    spec, fused = worker_context
    spec.speculative_use_rejection_sampling = rejection
    worker, logits, hidden = _extend_worker(monkeypatch, enabled, topk, graph_mode)
    batch = SimpleNamespace(
        seq_lens=torch.tensor([10, 20]),
        sampling_info=SimpleNamespace(temperatures=torch.ones(2, 1)),
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        accept_lens=torch.tensor([3, 2]),
        next_token_ids=torch.arange(8),
        next_draft_input=EagleDraftInput(),
    )
    sample = Mock(
        side_effect=lambda values, temperatures: (
            values.softmax(-1),
            *torch.max(values.softmax(-1), -1, keepdim=True),
        )
    )
    monkeypatch.setattr(module, "sample_draft_proposal", sample)
    worker._draft_extend_for_decode(batch, result)
    selected = torch.tensor([2, 5])
    expected_probabilities = logits[selected].softmax(-1)
    expected_indices = expected_probabilities.topk(topk, -1).indices
    torch.testing.assert_close(result.next_draft_input.topk_index, expected_indices)
    torch.testing.assert_close(result.next_draft_input.hidden_states, hidden[selected])
    assert fused.call_count == int(enabled and topk == 1 and not rejection)
    if fused.called:
        assert fused.call_args.args[0].shape == (8, 16)
        torch.testing.assert_close(
            fused.call_args.kwargs["input_row_indices"], selected
        )
        torch.testing.assert_close(result.next_draft_input.topk_p, torch.ones(2, 1))
    assert sample.call_count == int(rejection)
    if rejection:
        torch.testing.assert_close(
            result.next_draft_input.draft_probs, expected_probabilities
        )
    assert worker.draft_runner.forward.call_count == int(not graph_mode)
    assert worker.cuda_graph_runner_for_draft_extend.execute.call_count == int(
        graph_mode
    )


@pytest.mark.parametrize(
    "enabled,rejection", [(True, False), (False, False), (True, True)]
)
def test_prefill_extend_greedy_and_rejection_fallback(
    monkeypatch, worker_context, enabled, rejection
):
    spec, fused = worker_context
    spec.speculative_use_rejection_sampling = rejection
    worker, _, _ = _extend_worker(monkeypatch, enabled, 1, False)
    worker.speculative_algorithm = SimpleNamespace(is_standalone=lambda: False)
    logits = torch.tensor([[0.0, 5.0, -2.0], [4.0, 0.0, -2.0]])
    worker.draft_runner.forward = Mock(
        return_value=SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=logits, hidden_states=torch.ones(2, 3)
            )
        )
    )
    monkeypatch.setattr(
        module.ForwardBatch,
        "init_new",
        lambda *args, **kwargs: SimpleNamespace(forward_mode=ForwardMode.EXTEND),
    )
    monkeypatch.setattr(module, "_eagle_prefill_tail_tokens", lambda batch, ids: ids)
    sample = Mock(
        side_effect=lambda probabilities, num_samples: torch.max(
            probabilities, -1, keepdim=True
        )
    )
    monkeypatch.setattr(module, "fast_sample", sample)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.tensor([1, 2, 3, 4]),
        extend_lens=[2, 2],
        sampling_info=SimpleNamespace(temperatures=torch.ones(2, 1)),
    )
    output = worker._draft_extend_for_prefill(
        batch, torch.ones(4, 3), torch.tensor([7, 8])
    )
    torch.testing.assert_close(batch.input_ids, torch.tensor([2, 7, 4, 8]))
    torch.testing.assert_close(output.topk_index, logits.argmax(-1, keepdim=True))
    assert fused.call_count == int(enabled and not rejection)
    assert sample.call_count == int(rejection)
    assert (output.draft_probs is not None) == rejection


@pytest.mark.parametrize(
    "enabled,token_map,rejection",
    [
        (True, False, False),
        (True, True, False),
        (False, False, False),
        (True, False, True),
    ],
)
def test_draft_loop_chain_positions_and_token_map(
    monkeypatch, worker_context, enabled, token_map, rejection
):
    spec, fused = worker_context
    spec.speculative_use_rejection_sampling = rejection
    worker = module.EagleDraftWorker.__new__(module.EagleDraftWorker)
    worker._hip_topk1_fusion = enabled
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker.index_share_for_mtp_iteration = False
    worker.seed_dsa_topk_from_draft_extend = False
    worker._topk1_parents_prealloc = torch.zeros(8, 3, dtype=torch.long)
    worker._topk1_score_indices_prealloc = torch.zeros(8, 2, dtype=torch.long)
    worker.hot_token_id = torch.tensor([10, 11, 12, 13]) if token_map else None
    logits = torch.tensor([[0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    inputs = []

    def forward(batch):
        inputs.append(batch.input_ids.clone())
        return SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=logits, hidden_states=torch.ones(2, 3)
            )
        )

    worker.draft_runner = SimpleNamespace(
        canary_manager=None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen4ExpMTP"])
        ),
        forward=forward,
    )
    worker.draft_attn_backend = SimpleNamespace(attn_backends=[None, None, None])

    # Use the uncompiled tree selector so the fallback exercises tree assembly
    # without triggering torch.compile in this CPU worker regression test.
    def select(step, probabilities, indices, hidden, scores, topk):
        return (
            indices.flatten(),
            hidden,
            probabilities,
            (probabilities[:, None], indices, torch.zeros(2, 2, dtype=torch.long)),
        )

    monkeypatch.setattr(module, "select_top_k_tokens", select)
    monkeypatch.setattr(
        module,
        "sample_draft_proposal",
        lambda values, temperatures: (
            values.softmax(-1),
            *torch.max(values.softmax(-1), -1, keepdim=True),
        ),
    )
    info = EagleDraftInput(
        topk_p=torch.ones(2, 1),
        topk_index=torch.tensor([[0], [1]]),
        hidden_states=torch.ones(2, 3),
        draft_probs=torch.ones(2, 4) / 4,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=info,
        batch_size=2,
        out_cache_loc=torch.arange(6),
        positions=torch.tensor([10, 20]),
        sampling_info=SimpleNamespace(temperatures=torch.ones(2, 1)),
    )
    _, _, chain, probabilities = worker.draft_forward(batch)
    expected = torch.tensor([[0, 2, 2], [1, 3, 3]])
    if token_map:
        expected = worker.hot_token_id[expected]
    torch.testing.assert_close(chain, expected)
    torch.testing.assert_close(batch.positions, torch.tensor([12, 22]))
    torch.testing.assert_close(inputs[0], expected[:, 0])
    torch.testing.assert_close(inputs[1], expected[:, 1])
    assert fused.call_count == (2 if enabled and not rejection else 0)
    if token_map and fused.called:
        assert fused.call_args.args[2] is None
    assert (probabilities is not None) == rejection
