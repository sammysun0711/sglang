from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")

import unittest

import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.test.test_utils import CustomTestCase


def _make_logits_with_unique_argmax(
    batch_size: int,
    vocab_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        (batch_size, vocab_size), dtype=dtype, device=device, generator=g
    )
    expected_index = (
        torch.arange(batch_size, dtype=torch.long, device=device) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, expected_index[:, None], 1000.0)
    return logits, expected_index[:, None]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestSpecTopk1Triton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_draft_topk1_postprocess_matches_argmax_and_position_add(self):
        configs = [
            (1, 127, torch.float32),
            (4, 8192, torch.float16),
            (7, 8193, torch.bfloat16),
            (3, 50000, torch.float32),
        ]
        for batch_size, vocab_size, dtype in configs:
            with self.subTest(
                batch_size=batch_size, vocab_size=vocab_size, dtype=dtype
            ):
                logits, expected_index = _make_logits_with_unique_argmax(
                    batch_size,
                    vocab_size,
                    dtype=dtype,
                    device=self.device,
                    seed=vocab_size,
                )
                positions = torch.arange(
                    batch_size, dtype=torch.long, device=self.device
                )
                expected_positions = positions + 1

                topk_p, topk_index = draft_topk1_postprocess(logits, positions)

                torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
                torch.testing.assert_close(
                    topk_p,
                    torch.ones(
                        (batch_size, 1), dtype=torch.float32, device=self.device
                    ),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    positions, expected_positions, rtol=0, atol=0
                )

    def test_draft_topk1_postprocess_can_write_draft_token_column(self):
        batch_size = 17
        # Multi-split vocab so the fused write composes with the split reduction.
        vocab_size = 50000
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=0,
        )
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        backing = torch.full((batch_size, 5), -1, dtype=torch.long, device=self.device)
        draft_tokens = backing[:, 1:4]

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=2
        )

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        # Exactly one backing column is written; both neighbors stay untouched.
        expected_backing = torch.full_like(backing, -1)
        expected_backing[:, 3] = expected_index[:, 0]
        torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )

    def test_row_strided_logits_view_matches_argmax(self):
        batch_size = 5
        vocab_size = 8193
        # Poison the padding columns: if the kernel used the dense vocab width
        # as the row stride it would read them and pick the wrong index.
        backing = torch.full(
            (batch_size, vocab_size + 64),
            2000.0,
            dtype=torch.float32,
            device=self.device,
        )
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=1,
        )
        backing[:, :vocab_size] = logits
        strided_logits = backing[:, :vocab_size]
        self.assertFalse(strided_logits.is_contiguous())
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(strided_logits, positions)

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )

    def test_empty_batch(self):
        logits = torch.empty((0, 1024), dtype=torch.float32, device=self.device)
        positions = torch.empty((0,), dtype=torch.long, device=self.device)
        draft_tokens = torch.empty((0, 3), dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=1
        )

        self.assertEqual(topk_p.shape, (0, 1))
        self.assertEqual(topk_index.shape, (0, 1))
        self.assertEqual(draft_tokens.numel(), 0)

    def test_selected_rows_preserve_first_index_ties_in_graph_replay(self):
        # Match Qwen's wide logits and poison unused backing columns. Ties
        # cross reduction splits, so finalization must pick the first token.
        vocab_size = 248320
        backing = torch.full((32, vocab_size + 64), 200.0, device=self.device)
        logits = backing[:, :vocab_size]
        logits.fill_(-3.0)
        logits[:, 17] = 7.0
        logits[:, 8197] = 7.0
        logits[:, vocab_size - 1] = 7.0
        rows = torch.tensor([31, 3, 19, 0, 4, 18, 7, 25], device=self.device)
        expected = torch.max(torch.softmax(logits[rows], dim=-1), dim=-1)[1][:, None]
        graph = torch.cuda.CUDAGraph()
        draft_topk1_postprocess(logits, input_row_indices=rows)
        with torch.cuda.graph(graph):
            probabilities, indices = draft_topk1_postprocess(
                logits, input_row_indices=rows
            )
        graph.replay()
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            probabilities, torch.ones_like(probabilities), rtol=0, atol=0
        )
        logits[:, 17] = 2.0
        rows.copy_(rows.flip(0))
        graph.replay()
        expected = torch.max(torch.softmax(logits[rows], dim=-1), dim=-1)[1][:, None]
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)

    def test_selected_rows_advance_only_selected_output_positions(self):
        logits, _ = _make_logits_with_unique_argmax(
            32, 248320, dtype=torch.float32, device=self.device, seed=42
        )
        rows = torch.tensor([31, 2, 7, 16, 8, 20, 1, 0], device=self.device)
        positions = torch.arange(8, device=self.device)
        chain = torch.full((8, 3), -1, dtype=torch.long, device=self.device)
        probabilities, indices = draft_topk1_postprocess(
            logits, positions, chain, 1, input_row_indices=rows
        )
        expected = torch.max(torch.softmax(logits[rows], dim=-1), dim=-1)[1][:, None]
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        torch.testing.assert_close(chain[:, 1:2], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            chain[:, [0, 2]], torch.full_like(chain[:, [0, 2]], -1), rtol=0, atol=0
        )
        torch.testing.assert_close(
            positions, torch.arange(1, 9, device=self.device), rtol=0, atol=0
        )

    def test_non_contiguous_inputs_raise(self):
        logits = torch.empty((16, 4), dtype=torch.float32, device=self.device).t()
        positions = torch.arange(8, dtype=torch.long, device=self.device)[::2]

        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(
                logits, torch.empty(4, dtype=torch.long, device=self.device)
            )
        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(torch.empty((4, 16), device=self.device), positions)

    def test_selected_rows_empty_repeated_and_int32(self):
        logits, _ = _make_logits_with_unique_argmax(
            4, 8193, dtype=torch.float32, device=self.device, seed=76
        )
        for dtype in (torch.int32, torch.int64):
            rows = torch.tensor([3, 0, 3, 1], dtype=dtype, device=self.device)
            probabilities, indices = draft_topk1_postprocess(
                logits, input_row_indices=rows
            )
            torch.testing.assert_close(indices, logits[rows].argmax(-1, keepdim=True))
            torch.testing.assert_close(probabilities, torch.ones_like(probabilities))
            empty = torch.empty(0, dtype=dtype, device=self.device)
            probabilities, indices = draft_topk1_postprocess(
                logits, input_row_indices=empty
            )
            self.assertEqual(indices.shape, (0, 1))
            self.assertEqual(probabilities.shape, (0, 1))

    def test_raw_logit_near_tie_contract(self):
        # Greedy selection uses raw logits. A rounded softmax may turn these
        # distinct FP32 values into a tie; that is not this helper's contract.
        logits = torch.tensor([[0.0, 1e-8]], device=self.device)
        _, indices = draft_topk1_postprocess(logits)
        torch.testing.assert_close(indices, torch.tensor([[1]], device=self.device))


if __name__ == "__main__":
    unittest.main()
