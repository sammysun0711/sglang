from sglang.srt.batch_overlap.two_batch_overlap import _resolve_tbo_cu_split


def test_mi300x_304_keeps_requested_240():
    assert _resolve_tbo_cu_split(304, 240) == (240, 64)
    assert _resolve_tbo_cu_split(304, 228) == (228, 76)
    assert _resolve_tbo_cu_split(304, 0) is None


def test_gfx950_256_keeps_240_compute():
    assert _resolve_tbo_cu_split(256, 240) == (240, 16)
    assert _resolve_tbo_cu_split(256, 228) == (228, 28)
    assert _resolve_tbo_cu_split(256, 256) == (255, 1)


def test_mask_off_when_request_is_non_positive():
    assert _resolve_tbo_cu_split(304, 0) is None
    assert _resolve_tbo_cu_split(304, -1) is None
