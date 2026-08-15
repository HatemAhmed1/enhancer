import numpy as np
import pytest

from enhancer.interpolate import Interpolator, interpolate_stream


class FakeFlow:
    """Linear cross-fade standing in for a real flow model."""

    def __init__(self):
        self.calls = []

    def __call__(self, a, b, t):
        self.calls.append(t)
        return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


def _frames(n, size=32):
    return [np.full((size, size, 3), i * 10 % 256, dtype=np.uint8) for i in range(n)]


def test_same_fps_copies_every_frame_without_inference():
    model = FakeFlow()
    out = list(interpolate_stream(_frames(5), src_fps=24, dst_fps=24, model=model))
    assert len(out) == 5
    assert model.calls == [], "no synthesis should happen at 1:1"


def test_doubling_produces_twice_as_many_frames():
    out = list(interpolate_stream(_frames(5), src_fps=24, dst_fps=48, model=FakeFlow()))
    assert len(out) == 10


def test_non_integer_ratio_produces_the_planned_count():
    out = list(interpolate_stream(_frames(4), src_fps=24, dst_fps=60, model=FakeFlow()))
    assert len(out) == 10


def test_copied_frames_are_bit_identical_to_the_source():
    src = _frames(4)
    out = list(interpolate_stream(src, src_fps=24, dst_fps=48, model=FakeFlow()))
    assert np.array_equal(out[0], src[0])
    assert np.array_equal(out[2], src[1])


def test_synthesised_frames_lie_between_their_neighbours():
    """A 100 to 160 step is a plausible within-shot lighting change.

    Deliberately not 0 to 200: that is a cut by any distribution-based measure,
    so the detector would correctly duplicate rather than synthesise and this
    test would be asserting the wrong thing.
    """
    src = [np.full((8, 8, 3), 100, dtype=np.uint8), np.full((8, 8, 3), 160, dtype=np.uint8)]
    out = list(interpolate_stream(src, src_fps=24, dst_fps=48, model=FakeFlow()))
    assert 100 < int(out[1].mean()) < 160


def test_scene_change_duplicates_instead_of_synthesising():
    """A morph across a cut is glaring; a duplicated frame is invisible."""
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    model = FakeFlow()
    out = list(interpolate_stream([a, b], src_fps=24, dst_fps=48, model=model))
    assert model.calls == [], "must not synthesise across a cut"
    assert np.array_equal(out[1], a)


def test_scene_detection_can_be_disabled():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    model = FakeFlow()
    list(interpolate_stream([a, b], src_fps=24, dst_fps=48, model=model, scene_threshold=None))
    assert model.calls, "disabling detection must allow synthesis"


def test_output_dtype_and_shape_are_preserved():
    out = list(interpolate_stream(_frames(3), src_fps=24, dst_fps=48, model=FakeFlow()))
    assert out[0].dtype == np.uint8
    assert out[0].shape == (32, 32, 3)


def test_single_frame_source_yields_that_frame():
    src = _frames(1)
    out = list(interpolate_stream(src, src_fps=24, dst_fps=48, model=FakeFlow()))
    assert len(out) == 1
    assert np.array_equal(out[0], src[0])


def test_empty_source_yields_nothing():
    assert list(interpolate_stream([], src_fps=24, dst_fps=48, model=FakeFlow())) == []


def test_model_receives_timesteps_in_the_unit_interval():
    model = FakeFlow()
    list(interpolate_stream(_frames(6), src_fps=24, dst_fps=60, model=model))
    assert all(0.0 < t < 1.0 for t in model.calls)


def test_interpolator_target_fps_from_multiplier():
    assert Interpolator.target_fps(src_fps=24, multiplier=2) == 48
    assert Interpolator.target_fps(src_fps=25, multiplier=4) == 100


def test_interpolator_rejects_multiplier_below_one():
    with pytest.raises(ValueError):
        Interpolator.target_fps(src_fps=24, multiplier=0)
