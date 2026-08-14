import pytest

from enhancer.analyze import (
    FieldAnalysis,
    ScanType,
    _parse_idet,
    classify_scan,
)

PROGRESSIVE_OUTPUT = """
[Parsed_idet_0 @ 000] Repeated Fields: Neither:   200 Top:     0 Bottom:     0
[Parsed_idet_0 @ 000] Single frame detection: TFF:     0 BFF:     0 Progressive:   200 Undetermined:     0
[Parsed_idet_0 @ 000] Multi frame detection: TFF:     0 BFF:     0 Progressive:   200 Undetermined:     0
"""

TELECINE_OUTPUT = """
[Parsed_idet_0 @ 000] Repeated Fields: Neither:   120 Top:    40 Bottom:    40
[Parsed_idet_0 @ 000] Single frame detection: TFF:    80 BFF:     0 Progressive:   120 Undetermined:     0
[Parsed_idet_0 @ 000] Multi frame detection: TFF:    80 BFF:     0 Progressive:   120 Undetermined:     0
"""

INTERLACED_OUTPUT = """
[Parsed_idet_0 @ 000] Repeated Fields: Neither:   198 Top:     1 Bottom:     1
[Parsed_idet_0 @ 000] Single frame detection: TFF:   190 BFF:     2 Progressive:     8 Undetermined:     0
[Parsed_idet_0 @ 000] Multi frame detection: TFF:   192 BFF:     2 Progressive:     6 Undetermined:     0
"""


def test_parse_idet_extracts_counts():
    a = _parse_idet(INTERLACED_OUTPUT)
    assert a.tff == 192
    assert a.bff == 2
    assert a.progressive == 6
    assert a.repeated_top == 1
    assert a.repeated_bottom == 1


def test_parse_idet_prefers_multi_frame_detection():
    """Multi-frame detection is more reliable than single-frame."""
    a = _parse_idet(TELECINE_OUTPUT)
    assert a.tff == 80


def test_parse_idet_raises_on_missing_output():
    with pytest.raises(ValueError, match="no idet output"):
        _parse_idet("ffmpeg version N-1234\nnothing useful here\n")


def test_classify_progressive():
    assert classify_scan(_parse_idet(PROGRESSIVE_OUTPUT)) is ScanType.PROGRESSIVE


def test_classify_telecine_from_repeated_fields():
    """Repeated fields are the 3:2 pulldown signature."""
    assert classify_scan(_parse_idet(TELECINE_OUTPUT)) is ScanType.TELECINED


def test_classify_interlaced():
    assert classify_scan(_parse_idet(INTERLACED_OUTPUT)) is ScanType.INTERLACED


def test_classify_handles_all_zero_counts():
    a = FieldAnalysis(tff=0, bff=0, progressive=0, undetermined=0,
                      repeated_top=0, repeated_bottom=0)
    assert classify_scan(a) is ScanType.PROGRESSIVE


def test_field_order_reports_bff_when_bottom_dominant():
    a = FieldAnalysis(tff=2, bff=190, progressive=8, undetermined=0,
                      repeated_top=1, repeated_bottom=1)
    assert classify_scan(a) is ScanType.INTERLACED
    assert a.field_order == "bff"


def test_field_order_is_tff_when_top_dominant():
    assert _parse_idet(INTERLACED_OUTPUT).field_order == "tff"


import numpy as np

from enhancer.analyze import estimate_blockiness, estimate_grain


def test_grain_is_near_zero_for_a_flat_frame():
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert estimate_grain(flat) < 0.5


def test_grain_rises_with_added_noise():
    rng = np.random.default_rng(0)
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    noisy = np.clip(
        flat.astype(np.int16) + rng.normal(0, 8, flat.shape), 0, 255
    ).astype(np.uint8)
    assert estimate_grain(noisy) > estimate_grain(flat) + 2


def test_grain_ignores_smooth_gradients():
    """A gradient has energy but no grain; the estimator must not confuse them."""
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    frame = np.dstack([ramp] * 3)
    assert estimate_grain(frame) < 1.0


def test_grain_is_deterministic():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert estimate_grain(frame) == estimate_grain(frame)


def test_blockiness_is_low_for_a_smooth_frame():
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    assert estimate_blockiness(np.dstack([ramp] * 3)) < 1.0


def test_blockiness_rises_with_8x8_discontinuities():
    """Simulate DCT block edges every 8 pixels."""
    frame = np.full((128, 128, 3), 120, dtype=np.uint8)
    frame[:, ::8] = 160
    assert estimate_blockiness(frame) > 3.0
