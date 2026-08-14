import numpy as np
import pytest

from enhancer.scenes import DEFAULT_THRESHOLD, frame_difference, is_scene_change


def _frame(value, size=64):
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_identical_frames_have_zero_difference():
    f = _frame(120)
    assert frame_difference(f, f) == 0.0


def test_difference_is_symmetric():
    a, b = _frame(50), _frame(200)
    assert frame_difference(a, b) == pytest.approx(frame_difference(b, a))


def test_difference_is_normalised_to_unit_range():
    assert 0.0 <= frame_difference(_frame(0), _frame(255)) <= 1.0


def test_black_to_white_is_near_maximum():
    assert frame_difference(_frame(0), _frame(255)) > 0.9


def test_hard_cut_is_detected():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 60, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(195, 256, (64, 64, 3), dtype=np.uint8)
    assert is_scene_change(a, b)


def test_gentle_pan_is_not_a_scene_change():
    """A shifted image is high-motion but the same shot."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    a = base[:, :64]
    b = base[:, 4:68]
    assert not is_scene_change(a, b)


def test_small_brightness_change_is_not_a_scene_change():
    assert not is_scene_change(_frame(120), _frame(128))


def test_threshold_is_adjustable():
    a, b = _frame(120), _frame(150)
    assert not is_scene_change(a, b, threshold=0.9)
    assert is_scene_change(a, b, threshold=0.01)


def test_default_threshold_is_in_a_sane_range():
    assert 0.1 <= DEFAULT_THRESHOLD <= 0.6


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        frame_difference(_frame(0, 32), _frame(0, 64))
