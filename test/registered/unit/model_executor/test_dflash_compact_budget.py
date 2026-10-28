from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.model_executor.pool_configurator import HybridSWAPoolConfigurator
from sglang.srt.speculative.dflash_compact_layout import CompactDFlashLayout
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


def runner(compact, ratio):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            full_attention_layer_ids=list(range(10)),
            swa_attention_layer_ids=list(range(60)),
            get_num_kv_heads=lambda _: 1,
            get_swa_num_kv_heads=lambda _: 1,
            head_dim=192,
            v_head_dim=128,
            swa_head_dim=192,
            swa_v_head_dim=128,
            context_len=1048576,
        ),
        kv_cache_dtype=torch.bfloat16,
        server_args=SimpleNamespace(
            speculative_dflash_compact_cache=compact,
            swa_full_tokens_ratio=ratio,
            max_speculative_num_draft_tokens=8,
            speculative_algorithm="DFLASH",
            page_size=64,
            speculative_eagle_topk=1,
        ),
        dflash_compact_layout=CompactDFlashLayout(54),
        dflash_draft_cell_size_per_token=2560,
        dflash_draft_num_layers=5,
    )


@pytest.mark.parametrize("ratio", [0.0012, 0.004, 0.016, 0.04])
@pytest.mark.parametrize("budget", [1 << 30, 64 << 30, 128 << 30])
def test_exact_allocation_fits_budget_and_next_page_does_not(ratio, budget):
    mr = runner(True, ratio)
    with patch(
        "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
        return_value=8,
    ):
        solver = HybridSWAPoolConfigurator(mr)
    config = solver.calculate_pool_sizes(budget, 64)
    layout = mr.dflash_compact_layout

    # Independently sum the actual tensor shapes; do not use the solver's cost helper.
    def used(f):
        s = int(f * ratio) // 64 * 64
        draft = (layout.physical_tokens + 64) * 2560 + layout.metadata_bytes(5)
        target = (f + 64) * 6400 + (s + 64) * 38400
        metadata = (
            55 * (1048576 + 12) * 4 + 70 * 48 + (f + 65) * 8 + ((f + s) // 64) * 8
        )
        return draft + target + metadata

    assert used(config.full_max_total_num_tokens) <= budget
    assert used(config.full_max_total_num_tokens + 64) > budget
    assert config.dflash_compact_fixed_bytes == layout.tensor_bytes(
        2560
    ) + layout.metadata_bytes(5)
    capped = solver.calculate_pool_sizes_from_max_tokens(65536, 64)
    assert capped.full_max_total_num_tokens == 65536
    assert capped.dflash_compact_layout == layout
    assert (
        capped.dflash_compact_target_accounted_bytes + capped.dflash_compact_fixed_bytes
        == used(65536)
    )


def test_legacy_pool_sizes_unchanged():
    mr = runner(False, 0.016)
    with patch(
        "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
        return_value=8,
    ):
        solver = HybridSWAPoolConfigurator(mr)
    capacity = solver.calculate_pool_sizes(100 << 30, 64)
    expected_f = int((100 << 30) // (6400 + 0.016 * 38400)) // 64 * 64
    assert capacity.full_max_total_num_tokens == expected_f
    assert capacity.swa_max_total_num_tokens == int(expected_f * 0.016) // 64 * 64
    assert capacity.dflash_compact_layout is None
    assert capacity.dflash_compact_fixed_bytes == 0


def test_insufficient_fixed_budget_fails():
    with patch(
        "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
        return_value=8,
    ):
        solver = HybridSWAPoolConfigurator(runner(True, 0.016))
    with pytest.raises(ValueError, match="no memory"):
        solver.calculate_pool_sizes(64 << 20, 64)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
