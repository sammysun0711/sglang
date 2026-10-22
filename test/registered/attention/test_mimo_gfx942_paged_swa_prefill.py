"""gfx942 SWA prefill must consume page-64 cache data and preserve gfx950 routing."""

import math
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention import aiter_utils
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype


@pytest.fixture(autouse=True)
def enable_flypa(monkeypatch):
    monkeypatch.setenv(aiter_utils.FLYPA_MIMO_PREFILL_ENV, "1")


def _case(q_lens, kv_lens, *, device="cpu", padded=0, evict=False):
    torch.manual_seed(917)
    device = torch.device(device)
    live = sum(q_lens)
    q = torch.full(
        (live + padded, 16 * 192), float("nan"), dtype=torch.bfloat16, device=device
    )
    q[:live] = torch.randn((live, 16 * 192), dtype=torch.bfloat16, device=device) * 0.7
    keys = (
        torch.randn((sum(kv_lens), 1, 192), dtype=torch.bfloat16, device=device) * 0.7
    )
    values = (
        torch.randn((sum(kv_lens), 1, 128), dtype=torch.bfloat16, device=device) * 0.3
    )
    q_reference = q[:live].view(-1, 16, 192).float()
    k_scale = v_scale = None
    k_reference, v_reference = keys.float(), values.float()
    page_counts = [math.ceil(n / 64) for n in kv_lens]
    total_pages = sum(page_counts)
    # Different, shuffled FULL and SWA page assignments, plus an unused page 0.
    full_ids = torch.randperm(total_pages) + 1
    swa_ids = full_ids.roll(1)
    storage_dtype = keys.dtype
    poison = float("nan")
    k_pages = torch.full(
        (total_pages + 1, 64, 1, 192), poison, dtype=storage_dtype, device=device
    )
    v_pages = torch.full(
        (total_pages + 1, 64, 1, 128), poison, dtype=storage_dtype, device=device
    )
    full_to_swa = torch.zeros((total_pages + 1) * 64, dtype=torch.int64, device=device)
    slots = []
    offset = page_offset = 0
    for nq, nk, n_pages in zip(q_lens, kv_lens, page_counts):
        earliest = max(0, nk - nq - 128)
        for page in range(n_pages):
            full, swa = int(full_ids[page_offset + page]), int(
                swa_ids[page_offset + page]
            )
            length = min(64, nk - page * 64)
            k_pages[swa, :length] = keys[
                offset + page * 64 : offset + page * 64 + length
            ]
            v_pages[swa, :length] = values[
                offset + page * 64 : offset + page * 64 + length
            ]
            resident = not evict or (page + 1) * 64 > earliest
            if resident:
                full_to_swa[full * 64 : (full + 1) * 64] = (
                    torch.arange(64, device=device) + swa * 64
                )
            slots.append(
                torch.arange(length, device=device, dtype=torch.int32) + swa * 64
                if resident
                else torch.zeros(length, device=device, dtype=torch.int32)
            )
        offset += nk
        page_offset += n_pages
    pack = 8
    k_buf = (
        k_pages.view(-1, 64, 1, 192 // pack, pack).permute(0, 2, 3, 1, 4).contiguous()
    )
    v_buf = (
        v_pages.view(-1, 64 // pack, pack, 1, 128).permute(0, 3, 1, 4, 2).contiguous()
    )
    sub_pool = SimpleNamespace(
        dtype=torch.bfloat16,
        store_dtype=storage_dtype,
        start_layer=0,
        k_buffer=[k_buf],
        v_buffer=[v_buf],
    )
    translations = []

    def translate(indices):
        translations.append(indices)
        return full_to_swa[indices]

    pool = SimpleNamespace(
        layers_mapping={0: (0, True)},
        swa_kv_pool=sub_pool,
        translate_loc_from_full_to_swa=translate,
    )

    def indptr(lengths):
        return torch.tensor([0] + lengths, device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )

    metadata = SimpleNamespace(
        swa_page_table=torch.cat(slots),
        paged_kv_indptr=indptr(page_counts),
        paged_kv_indices=torch.nn.functional.pad(
            full_ids.to(device=device, dtype=torch.int32), (0, 256)
        ),
        paged_kv_last_page_len=torch.tensor(
            [(n - 1) % 64 + 1 for n in kv_lens], device=device, dtype=torch.int32
        ),
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
    )
    backend = SimpleNamespace(
        input_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        page_size=64,
        logits_soft_cap=0.0,
        forward_metadata=metadata,
        token_to_kv_pool=pool,
        qo_indptr=indptr(q_lens),
        k_scale=k_scale,
        v_scale=v_scale,
    )
    layer = SimpleNamespace(
        layer_id=0,
        sliding_window_size=128,
        tp_q_head_num=16,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=192,
        head_dim=192,
        v_head_dim=128,
        mimo_original_v_head_dim=128,
        scaling=0.75 * 192**-0.5,
        k_scale=k_scale,
        v_scale=None,
    )
    batch = SimpleNamespace(
        extend_seq_lens_cpu=q_lens,
        extend_prefix_lens_cpu=[k - n for n, k in zip(q_lens, kv_lens)],
        seq_lens_cpu=torch.tensor(kv_lens),
        seq_lens_sum=sum(kv_lens),
    )
    sinks = torch.linspace(-1, 1, 16, dtype=torch.float32, device=device)
    return SimpleNamespace(
        backend=backend,
        layer=layer,
        batch=batch,
        q=q,
        sinks=sinks,
        k_buf=k_buf,
        v_buf=v_buf,
        translations=translations,
        expected_pages=full_to_swa[metadata.paged_kv_indices.long() * 64]
        .div(64, rounding_mode="floor")
        .int(),
        q_reference=q_reference,
        k_reference=k_reference,
        v_reference=v_reference,
    )


def _run(case, **kwargs):
    # Fresh K/V deliberately contain zeros: the selected path must use the cache.
    return aiter_utils.forward_extend_vectorized_5d(
        case.backend,
        case.q,
        case.q.new_zeros((case.q.shape[0], 1, 192)),
        case.q.new_zeros((case.q.shape[0], 1, 128)),
        case.layer,
        case.batch,
        len(case.batch.extend_seq_lens_cpu) + 1,
        (128, -1),
        case.sinks,
        **kwargs,
    )


@pytest.mark.parametrize("fresh", [False, True])
def test_gfx942_swa_uses_5d_pages_and_caches_translation(monkeypatch, fresh):
    case = _case([7, 65], [7, 65] if fresh else [71, 193], padded=8)
    calls = []

    def flypa(q, k, v, *args, **kwargs):
        assert k.data_ptr() == case.k_buf.data_ptr() and k.ndim == 5
        assert v.data_ptr() == case.v_buf.data_ptr() and v.ndim == 5
        assert q.shape[0] == 72
        torch.testing.assert_close(args[2], case.expected_pages)
        assert kwargs["window_left"] == 128
        assert kwargs["sink_ptr"] is case.sinks
        assert kwargs["softmax_scale"] == case.layer.scaling
        calls.append(kwargs)
        return kwargs["out"].fill_(3)

    def reject(*args, **kwargs):
        raise AssertionError("gfx942 SWA must not gather LINEAR K/V")

    monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: False)
    import aiter.ops.flydsl as flydsl_ops

    monkeypatch.setattr(flydsl_ops, "flydsl_paged_attention_swa_bf16", flypa)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", reject)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject)
    for _ in range(2):
        result = _run(case)
        assert torch.all(result[:72] == 3)
        assert torch.all(result[72:] == 0)
    assert len(calls) == 2 and len(case.translations) == 1


def test_gfx950_does_not_enter_new_swa_path(monkeypatch):
    case = _case([7], [7])

    def reject(*args, **kwargs):
        raise AssertionError("gfx950 routing must not enter the gfx942 helper")

    def varlen(q, k, v, *args, **kwargs):
        return q.new_zeros((q.shape[0], 16, 128))

    monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: True)
    monkeypatch.setattr(aiter_utils, "_mimo_gfx942_paged_swa_prefill", reject)
    monkeypatch.setattr(aiter_utils, "flash_attn_varlen_func", varlen)
    assert _run(case).shape == (7, 16 * 128)


def test_swa_cache_not_written_keeps_fresh_input_path(monkeypatch):
    case = _case([7], [7])

    def reject(*args, **kwargs):
        raise AssertionError("cannot read pages when the cache write was skipped")

    def linear(q, k, v, *args, **kwargs):
        assert k.ndim == v.ndim == 3
        return q.new_zeros((q.shape[0], 16, 128))

    monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: False)
    monkeypatch.setattr(aiter_utils, "_mimo_gfx942_paged_swa_prefill", reject)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", linear)
    assert _run(case, kv_cache_written=False).shape == (7, 16 * 128)


def test_disabled_flypa_keeps_existing_path(monkeypatch):
    case = _case([7], [7])

    def reject(*args, **kwargs):
        raise AssertionError("disabled FlyPA must not enter the new route")

    monkeypatch.setenv(aiter_utils.FLYPA_MIMO_PREFILL_ENV, "0")
    monkeypatch.setattr(aiter_utils, "is_gfx942", lambda: True)
    monkeypatch.setattr(aiter_utils, "is_gfx950", lambda: False)
    monkeypatch.setattr(aiter_utils, "_mimo_gfx942_paged_swa_prefill", reject)
    monkeypatch.setattr(
        aiter_utils,
        "mha_batch_prefill_func",
        lambda q, *args, **kwargs: q.new_zeros((q.shape[0], 16, 128)),
    )
    assert _run(case).shape == (7, 16 * 128)


def test_fp8_swa_is_not_redirected():
    case = _case([7], [71])
    case.backend.kv_cache_dtype = fp8_dtype
    assert (
        aiter_utils._mimo_gfx942_paged_swa_prefill(
            case.backend, case.q, case.layer, case.batch, 2, (128, -1), case.sinks
        )
        is None
    )


@pytest.mark.parametrize(
    "missing", ["paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"]
)
def test_missing_paged_metadata_is_not_selected(missing):
    case = _case([7], [71])
    setattr(case.backend.forward_metadata, missing, None)
    assert (
        aiter_utils._mimo_gfx942_paged_swa_prefill(
            case.backend, case.q, case.layer, case.batch, 2, (128, -1), case.sinks
        )
        is None
    )


def _reference(case):
    pieces = []
    qo = ko = 0
    for nq, nk in zip(case.batch.extend_seq_lens_cpu, case.batch.seq_lens_cpu.tolist()):
        for start in range(0, nq, 128):
            stop = min(start + 128, nq)
            ks, ke = max(0, nk - nq + start - 128), nk - nq + stop
            q = case.q_reference[qo + start : qo + stop].transpose(0, 1)
            k = case.k_reference[ko + ks : ko + ke].transpose(0, 1)
            v = case.v_reference[ko + ks : ko + ke].transpose(0, 1)
            score = q @ k.transpose(-1, -2) * case.layer.scaling
            qp = nk - nq + torch.arange(start, stop, device=q.device)[:, None]
            kp = torch.arange(ks, ke, device=q.device)[None, :]
            score.masked_fill_(~((kp <= qp) & (kp >= qp - 128))[None], -torch.inf)
            logits = torch.cat(
                [score, case.sinks[:, None, None].expand(-1, stop - start, 1)], -1
            )
            pieces.append((logits.softmax(-1)[..., :-1] @ v).transpose(0, 1))
        qo += nq
        ko += nk
    return torch.cat(pieces)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires gfx942")
@pytest.mark.parametrize(
    "q_lens,kv_lens,padded,evict",
    [
        ([1], [1], 0, False),
        ([1], [257], 0, False),
        ([129], [129], 0, False),
        ([3968], [16384], 0, False),
        ([7, 63, 129], [73, 192, 257], 57, False),
        ([129], [1025], 0, True),
        ([129], [961], 0, True),
        ([32768], [65536], 0, False),
    ],
)
def test_gpu_paged_swa_matches_reference(monkeypatch, q_lens, kv_lens, padded, evict):
    if not aiter_utils.is_gfx942():
        pytest.skip("gfx942-only routing change")
    case = _case(q_lens, kv_lens, device="cuda", padded=padded, evict=evict)
    import aiter.ops.flydsl as flydsl_ops

    original = flydsl_ops.flydsl_paged_attention_swa_bf16

    def flypa(q, k, v, *args, **kwargs):
        assert k.ndim == v.ndim == 5
        assert k.data_ptr() == case.k_buf.data_ptr()
        assert v.data_ptr() == case.v_buf.data_ptr()
        return original(q, k, v, *args, **kwargs)

    def reject(*args, **kwargs):
        raise AssertionError("SWA prefill unexpectedly gathered LINEAR K/V")

    monkeypatch.setattr(flydsl_ops, "flydsl_paged_attention_swa_bf16", flypa)
    monkeypatch.setattr(aiter_utils, "launch_gather_shuffle_5d_to_linear", reject)
    monkeypatch.setattr(aiter_utils, "mha_batch_prefill_func", reject)
    output = _run(case).view(-1, 16, 128)
    live = sum(q_lens)
    expected = _reference(case)
    tolerance = 0.02
    torch.testing.assert_close(
        output[:live].float(), expected, atol=tolerance, rtol=tolerance
    )
    assert torch.all(output[live:] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires gfx942")
def test_gpu_paged_swa_replays_graph():
    if not aiter_utils.is_gfx942():
        pytest.skip("gfx942-only routing change")
    case = _case([7, 63, 129], [73, 192, 257], device="cuda", padded=57)
    _run(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _run(case).view(-1, 16, 128)
    for scale in [1.0, 0.5]:
        case.q[:199].mul_(scale)
        case.q_reference = case.q[:199].view(-1, 16, 192).float()
        graph.replay()
        torch.testing.assert_close(
            output[:199].float(), _reference(case), atol=0.02, rtol=0.02
        )
        assert torch.all(output[199:] == 0)
