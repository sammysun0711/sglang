"""Address-contract tests independent of torch, kernels, and the model runner."""

import random

import pytest
from sglang.srt.speculative.dflash_compact_layout import (
    CompactDFlashLayout,
    draft_owner_count,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("prefix", [0, 1, 63, 64, 65, 1023, 1024, 1025, 1048576])
@pytest.mark.parametrize("residue", range(64))
@pytest.mark.parametrize("bank", [0, 1])
def test_page_table_resolves_every_prefix_and_verify_address(prefix, residue, bank):
    layout = CompactDFlashLayout(owner_count=54)
    prefix += residue
    length = layout.visible_length(prefix)
    assert 0 <= length <= 1087
    locs = [layout.physical_location(54, bank, i) for i in range(length + 8)]
    page_table = [locs[i] // 64 for i in range(0, len(locs), 64)]
    # This is how a paged attention kernel reconstructs individual addresses.
    reconstructed = [page_table[i // 64] * 64 + i % 64 for i in range(len(locs))]
    assert reconstructed == locs
    assert len(set(locs)) == len(locs)
    assert min(locs) >= layout.bank_base(54, bank)
    assert max(locs) < layout.bank_base(54, bank) + layout.bank_rows


def test_all_request_slots_and_dummy_banks_are_disjoint():
    owners = draft_owner_count(18, decode=True, preallocated_requests=18)
    assert owners == 54
    layout = CompactDFlashLayout(owners)
    occupied = set()
    for owner in range(owners + 1):
        for bank in range(2):
            rows = set(
                range(
                    layout.bank_base(owner, bank),
                    layout.bank_base(owner, bank) + layout.bank_rows,
                )
            )
            assert not rows.intersection(occupied)
            occupied.update(rows)
    assert occupied == set(range(layout.physical_tokens))
    assert draft_owner_count(18, decode=False) == 18
    assert draft_owner_count(64, decode=True, preallocated_requests=128) == 192


@pytest.mark.parametrize("accepted", range(1, 9))
def test_banks_preserve_committed_history_across_many_window_turnovers(accepted):
    layout = CompactDFlashLayout(2)
    # Distinct per-position identities make stale/duplicated/other-owner rows visible.
    memory = [None] * layout.physical_tokens
    rng = random.Random(accepted)
    for owner in (1, 2):
        prefix = 1048576 - 12000
        bank = 0
        start = layout.visible_start(prefix)
        for pos in range(start, prefix):
            memory[layout.physical_location(owner, bank, pos - start)] = (owner, pos)
        for _ in range(2048):
            next_start = layout.visible_start(prefix)
            length = prefix - next_start
            work = 1 - bank
            for i, pos in enumerate(range(next_start, prefix)):
                memory[layout.physical_location(owner, work, i)] = memory[
                    layout.physical_location(owner, bank, pos - start)
                ]
            for i in range(8):
                memory[layout.physical_location(owner, work, length + i)] = (
                    "noise",
                    rng.random(),
                )
            for i in range(accepted):
                memory[layout.physical_location(owner, work, length + i)] = (
                    owner,
                    prefix + i,
                )
            bank, start, prefix = work, next_start, prefix + accepted
            visible_start = layout.visible_start(prefix)
            actual = [
                memory[layout.physical_location(owner, bank, p - start)]
                for p in range(visible_start, prefix)
            ]
            assert actual == [(owner, p) for p in range(visible_start, prefix)]


def test_allocation_is_independent_of_full_context_and_prices_dummy_and_sentinel():
    layout = CompactDFlashLayout(54)
    assert layout.bank_rows == 1152
    assert layout.physical_tokens == 55 * 2304
    assert layout.tensor_bytes(2560) == (55 * 2304 + 64) * 2560
    assert layout.request_table_bytes == 55 * 1152 * 4
    assert layout.visible_length(1048576) == layout.visible_length(1024)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"owner_count": 0},
        {"owner_count": 1, "window_size": 1025},
        {"owner_count": 1, "block_size": 2048},
    ],
)
def test_invalid_geometry_fails(kwargs):
    with pytest.raises(ValueError):
        CompactDFlashLayout(**kwargs)


def test_invalid_owner_and_offset_fail():
    layout = CompactDFlashLayout(54)
    for owner, bank, offset in [(-1, 0, 0), (55, 0, 0), (1, 2, 0), (1, 0, 1152)]:
        with pytest.raises(ValueError):
            layout.physical_location(owner, bank, offset)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
