from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dflash_utils import (
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.utils import is_hip


@pytest.mark.skipif(not is_hip(), reason="requires ROCm/HIP")
def test_dflash_hip_non_greedy_verify_uses_torch_fallback(monkeypatch):
    monkeypatch.setenv("SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY", "1")
    assert is_dflash_sampling_verify_available()

    candidates = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int64, device="cuda"
    )
    logits = torch.full((8, 8), float("-inf"), dtype=torch.float32, device="cuda")
    logits[0, 1] = 0.0
    logits[1, 2] = 0.0
    logits[2, 3] = 0.0
    logits[3, 4] = 0.0
    logits[4, 1] = 0.0
    logits[5, 5] = 0.0
    logits[6, 6] = 0.0
    logits[7, 7] = 0.0
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((2, 1), dtype=torch.float32, device="cuda"),
        top_ks=torch.full((2,), 8, dtype=torch.int32, device="cuda"),
        top_ps=torch.ones((2,), dtype=torch.float32, device="cuda"),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
    )

    correct_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=sampling_info,
        uniform_samples=torch.full(
            (2, 4), 0.5, dtype=torch.float32, device="cuda"
        ),
        uniform_samples_for_final_sampling=torch.full(
            (2,), 0.5, dtype=torch.float32, device="cuda"
        ),
        threshold_single=1.0,
        threshold_acc=1.0,
    )

    assert correct_len.tolist() == [3, 1]
    assert bonus.tolist() == [4, 5]


@pytest.mark.skipif(not is_hip(), reason="requires ROCm/HIP")
def test_dflash_hip_non_greedy_verify_remains_opt_in(monkeypatch):
    monkeypatch.setenv("SGLANG_MIMO_EAGLE_HIP_NONGREEDY_VERIFY", "0")
    assert not is_dflash_sampling_verify_available()
