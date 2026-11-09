"""Token ownership and window boundaries for MiMo EP prefill attention."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import communicator
from sglang.srt.layers.attention import mimo_sp
from sglang.srt.model_executor import forward_context
from sglang.srt.models import mimo_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize(
    "lengths,prefixes",
    [
        ([409], [0]),
        ([4096], [409]),
        ([17, 450, 128], [0, 4096, 64]),
        ([1, 3, 5], [300, 0, 127]),
        ([4096, 4096], [32768, 19]),
        ([127, 1, 128], [127, 4096, 0]),
        ([0, 129, 0, 1], [4096, 64, 8192, 3]),
        ([0], [0]),
    ],
)
def test_swa_sp_preserves_request_windows(monkeypatch, rank, lengths, prefixes):
    monkeypatch.setattr(mimo_sp, "get_attention_tp_size", lambda: 4)
    monkeypatch.setattr(mimo_sp, "get_attention_tp_rank", lambda: rank)
    rows = (sum(lengths) + 3) // 4
    pool_width = max(p + n for p, n in zip(prefixes, lengths)) + 64
    req_slots = torch.arange(len(lengths) * pool_width).reshape(-1, pool_width)
    req_indices = torch.arange(len(lengths) - 1, -1, -1)
    monkeypatch.setattr(
        forward_context,
        "get_req_to_token_pool",
        lambda: SimpleNamespace(req_to_token=req_slots),
    )
    monkeypatch.setattr(
        forward_context,
        "get_token_to_kv_pool",
        lambda: SimpleNamespace(
            translate_loc_from_full_to_swa=lambda slots: slots * 3 + 5
        ),
    )
    batch = SimpleNamespace(
        positions=torch.arange(rows * 4),
        extend_seq_lens_cpu=lengths,
        # Match resumed chunks whose cache accounting reports zero.
        extend_prefix_lens_cpu=[0] * len(lengths),
        seq_lens_cpu=[p + n for p, n in zip(prefixes, lengths)],
        req_pool_indices=req_indices,
    )
    meta = mimo_sp._mimo_sp_swa_metadata(batch, 127, rows)
    assert mimo_sp._mimo_sp_swa_metadata(batch, 127, rows) is meta
    # Encode request ID as well as absolute position, so crossing a request
    # boundary cannot pass merely because the row counts agree.
    logical, cached, expected_slots = [], [], []
    for request, (length, prefix) in enumerate(zip(lengths, prefixes)):
        logical.extend(request * 100000 + t for t in range(prefix, prefix + length))
        if length:
            cached.extend(
                request * 100000 + t for t in range(max(0, prefix - 127), prefix)
            )
            expected_slots.extend(
                (
                    req_slots[req_indices[request], max(0, prefix - 127) : prefix]
                    * 3
                    + 5
                ).tolist()
            )
    logical += [-1] * (rows * 4 - len(logical))
    tail = min(127, rows)
    tails = [
        value
        for r in range(4)
        for value in logical[(r + 1) * rows - tail : (r + 1) * rows]
    ]
    sources = torch.tensor(
        cached + tails + logical[rank * rows : (rank + 1) * rows] + [-1]
    )
    actual = sources[meta.kv_src]
    assert meta.cache_rows == len(expected_slots)
    if expected_slots:
        assert torch.equal(
            meta.cache_page * 64 + meta.cache_offset, torch.tensor(expected_slots)
        )
        assert ((meta.cache_offset >= 0) & (meta.cache_offset < 64)).all()
    else:
        assert meta.cache_page is None and meta.cache_offset is None

    start, segment, local_q_count = 0, 0, 0
    max_query_rows, max_key_rows = 0, 0
    for request, (length, prefix) in enumerate(zip(lengths, prefixes)):
        begin, end = max(start, rank * rows), min(start + length, (rank + 1) * rows)
        if begin < end:
            q_begin, q_end = prefix + begin - start, prefix + end - start
            key_begin = max(0, q_begin - 127)
            expected = torch.arange(key_begin, q_end) + request * 100000
            assert meta.cu_seqlens_q[segment] == local_q_count
            assert meta.cu_seqlens_q[segment + 1] == local_q_count + end - begin
            max_query_rows = max(max_query_rows, end - begin)
            max_key_rows = max(max_key_rows, expected.numel())
            page_begin, page_end = meta.kv_indptr[segment : segment + 2].tolist()
            key_count = (page_end - page_begin - 1) * 64 + int(
                meta.last_page_lens[segment]
            )
            keys = actual[page_begin * 64 : page_begin * 64 + key_count]
            assert torch.equal(keys, expected)
            assert (
                actual[page_begin * 64 + key_count : page_end * 64] == -1
            ).all()
            # Bottom-right causal alignment + left-window mask, as in kernel.
            for q in range(q_begin, q_end):
                allowed = keys[
                    (keys >= request * 100000 + max(0, q - 127))
                    & (keys <= request * 100000 + q)
                ]
                assert torch.equal(
                    allowed, torch.arange(max(0, q - 127), q + 1) + request * 100000
                )
            local_q_count += end - begin
            segment += 1
        start += length
    assert meta.q_rows == local_q_count
    assert meta.cu_seqlens_q[-1] == local_q_count
    assert len(meta.cu_seqlens_q) == len(meta.kv_indptr) == segment + 1
    assert meta.max_q == max_query_rows and meta.max_kv == max_key_rows
    assert torch.equal(meta.page_ids, torch.arange(int(meta.kv_indptr[-1])))
    if not local_q_count:
        assert meta.max_q == meta.max_kv == meta.kv_src.numel() == 0


@pytest.mark.parametrize("dense_tp", [None, 1, 8])
@pytest.mark.parametrize("tbo", [False, True])
def test_sp_eligibility_keeps_dense_transition_constructible(
    monkeypatch, dense_tp, tbo
):
    args = SimpleNamespace(
        moe_dense_tp_size=dense_tp,
        enable_two_batch_overlap=tbo,
        attention_backend="aiter",
        page_size=64,
        kv_cache_dtype="bf16",
        attn_cp_size=1,
    )
    backend = SimpleNamespace(is_mori=lambda: True, is_none=lambda: False)
    for module in (mimo_v2, communicator):
        monkeypatch.setattr(module, "get_global_server_args", lambda: args)
        monkeypatch.setattr(module, "get_moe_a2a_backend", lambda: backend)
    monkeypatch.setattr(mimo_v2, "is_gfx942_supported", lambda: True)
    monkeypatch.setattr(mimo_v2, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(mimo_v2, "get_attention_tp_size", lambda: 4)

    with mimo_v2.envs.SGLANG_AITER_KV_CACHE_LAYOUT.override("vectorized_5d"):
        supported = mimo_v2._mimo_ep_sp_supported()
    assert supported is (dense_tp == 1 and not tbo)

    # The first SWA/MoE layer follows the initial dense layer. This layout is
    # selected while constructing its communicator, before any forward guard.
    modes = communicator.LayerScatterModes.init_new(
        layer_id=1,
        num_layers=48,
        is_layer_sparse=True,
        is_previous_layer_sparse=False,
        is_next_layer_sparse=True,
        sp_attn=supported,
    )
    context = SimpleNamespace(is_same_group_size=lambda a, b: a == b)
    fn = communicator.CommunicateSimpleFn.get_fn(
        modes.layer_input_mode, modes.attn_mode, context
    )
    assert fn is communicator.CommunicateSimpleFn._trivial


def test_scattered_attention_does_not_reduce_complete_rows():
    from sglang.srt.layers.communicator import (
        CommunicateWithAllReduceAndLayerNormFn,
        ScatterMode,
    )

    context = SimpleNamespace(is_same_group_size=lambda *args: True, attn_tp_size=4)
    fn = CommunicateWithAllReduceAndLayerNormFn.get_fn(
        hidden_states_input_mode=ScatterMode.SCATTERED,
        residual_input_mode=ScatterMode.SCATTERED,
        hidden_states_output_mode=ScatterMode.SCATTERED,
        residual_output_mode=ScatterMode.SCATTERED,
        context=context,
    )
    assert fn is CommunicateWithAllReduceAndLayerNormFn._simple


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
