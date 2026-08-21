import numpy as np
import pytest
from PIL import Image

from enhancer.compare import ComparePair, compare_frame, frame_at
from enhancer.requests import RenderRequest
from enhancer.video_io import SourceProfile


class _Doubler:
    """A fake upscaler: scale 2, nearest neighbour, no GPU and no model."""

    scale = 2
    device = "cpu"

    def __init__(self):
        self.seen = []

    def process(self, frame):
        self.seen.append(np.array(frame, copy=True))
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


class _RecordingTexture:
    """Stands in for TexturePost and records the order of its stages."""

    enabled = True
    detail_enabled = True
    grain_enabled = True

    def __init__(self):
        self.calls = []

    def apply_detail(self, output, source):
        self.calls.append("detail")
        return output

    def apply_grain(self, frame, index=0):
        self.calls.append("grain")
        return frame


def _request(tmp_path, source, **kwargs):
    return RenderRequest(
        model=tmp_path / "model.pth",
        source=source,
        output=tmp_path / "out.mkv",
        **kwargs,
    )


@pytest.fixture
def still(tmp_path, rng):
    path = tmp_path / "still.png"
    Image.fromarray(rng.integers(0, 256, (30, 50, 3), dtype=np.uint8)).save(path)
    return path


# --- geometry -------------------------------------------------------------


def test_before_is_returned_at_the_after_dimensions(tmp_path, synthetic_clip):
    pair = compare_frame(_request(tmp_path, synthetic_clip), _Doubler(), seconds=0.5)
    assert pair.before.shape == pair.after.shape
    assert pair.after.shape == (240 * 2, 320 * 2, 3)
    assert pair.before.dtype == np.uint8


def test_source_size_reports_the_true_original_not_the_resized_before(
    tmp_path, synthetic_clip
):
    pair = compare_frame(_request(tmp_path, synthetic_clip), _Doubler(), seconds=0.5)
    assert pair.source_size == (320, 240)
    # The before array is bigger than the size it reports, which is the whole
    # point: the reported size is the truth, the array is the fair display.
    assert pair.before.shape[:2] == (480, 640)
    assert pair.scale == 2


def test_pair_is_frozen(tmp_path, synthetic_clip):
    pair = compare_frame(_request(tmp_path, synthetic_clip), _Doubler())
    assert isinstance(pair, ComparePair)
    with pytest.raises(Exception):
        pair.scale = 4


# --- the before half must be raw ------------------------------------------


def test_before_is_not_filtered_even_when_degrain_is_high(tmp_path, synthetic_clip):
    heavy = compare_frame(
        _request(tmp_path, synthetic_clip, degrain=1.0, detail_retention=0.0, regrain=0.0),
        _Doubler(),
        seconds=0.4,
    )
    none = compare_frame(
        _request(tmp_path, synthetic_clip, no_restore=True),
        _Doubler(),
        seconds=0.4,
    )
    # Same raw source frame both times: degrain must not touch the before half.
    assert np.array_equal(heavy.before, none.before)


def test_before_matches_a_raw_decode_of_the_same_frame(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    raw, index = frame_at(profile, 0.4)

    pair = compare_frame(
        _request(tmp_path, synthetic_clip, degrain=0.9), _Doubler(), seconds=0.4
    )
    assert pair.frame_index == index
    # before is that raw frame, only enlarged.
    enlarged = np.asarray(
        Image.fromarray(raw).resize(
            (pair.after.shape[1], pair.after.shape[0]), Image.BICUBIC
        )
    )
    assert np.array_equal(pair.before, enlarged)


def test_heavy_degrain_does_change_the_after_half(tmp_path, synthetic_clip):
    heavy = compare_frame(
        _request(tmp_path, synthetic_clip, degrain=1.0, detail_retention=0.0, regrain=0.0),
        _Doubler(),
        seconds=0.4,
    )
    plain = compare_frame(
        _request(tmp_path, synthetic_clip, no_restore=True), _Doubler(), seconds=0.4
    )
    assert not np.array_equal(heavy.after, plain.after)


# --- filter chain ---------------------------------------------------------


def test_no_restore_produces_an_empty_video_filter(tmp_path, synthetic_clip, monkeypatch):
    import enhancer.compare as compare

    seen = {}
    real_decoder = compare.Decoder

    def spy(profile, start_frame=0, max_frames=None, video_filter=""):
        seen.setdefault("filters", []).append(video_filter)
        return real_decoder(profile, start_frame, max_frames, video_filter)

    monkeypatch.setattr(compare, "Decoder", spy)

    def boom(*a, **k):
        raise AssertionError("probe_scan must not run when no_restore is set")

    monkeypatch.setattr(compare, "probe_scan", boom)

    compare_frame(
        _request(tmp_path, synthetic_clip, no_restore=True), _Doubler(), seconds=0.2
    )
    assert seen["filters"] == ["", ""]


def test_restoring_run_builds_a_non_empty_filter_for_the_after_half(
    tmp_path, synthetic_clip, monkeypatch
):
    import enhancer.compare as compare

    filters = []
    real_decoder = compare.Decoder

    def spy(profile, start_frame=0, max_frames=None, video_filter=""):
        filters.append(video_filter)
        return real_decoder(profile, start_frame, max_frames, video_filter)

    monkeypatch.setattr(compare, "Decoder", spy)

    compare_frame(
        _request(tmp_path, synthetic_clip, degrain=0.5), _Doubler(), seconds=0.2
    )
    # The before decode is always raw; the after decode carries the chain.
    assert filters[0] == ""
    assert "hqdn3d" in filters[1]


def test_both_halves_decode_from_the_same_start_frame(
    tmp_path, synthetic_clip, monkeypatch
):
    import enhancer.compare as compare

    starts = []
    real_decoder = compare.Decoder

    def spy(profile, start_frame=0, max_frames=None, video_filter=""):
        starts.append(start_frame)
        return real_decoder(profile, start_frame, max_frames, video_filter)

    monkeypatch.setattr(compare, "Decoder", spy)

    pair = compare_frame(
        _request(tmp_path, synthetic_clip, degrain=0.5), _Doubler(), seconds=0.6
    )
    assert starts == [pair.frame_index, pair.frame_index]


# --- texture ordering -----------------------------------------------------


def test_detail_retention_runs_before_regrain(tmp_path, synthetic_clip):
    texture = _RecordingTexture()
    compare_frame(
        _request(tmp_path, synthetic_clip), _Doubler(), seconds=0.2, texture=texture
    )
    assert texture.calls == ["detail", "grain"]


def test_detail_retention_is_paired_with_the_filtered_frame(tmp_path, synthetic_clip):
    upscaler = _Doubler()
    captured = {}

    class _Capture(_RecordingTexture):
        def apply_detail(self, output, source):
            captured["source"] = np.array(source, copy=True)
            return super().apply_detail(output, source)

    compare_frame(
        _request(tmp_path, synthetic_clip, degrain=1.0),
        upscaler,
        seconds=0.4,
        texture=_Capture(),
    )
    # apply_detail sees exactly the frame that was fed to the upscaler, i.e.
    # the filtered one, not the raw before frame.
    assert np.array_equal(captured["source"], upscaler.seen[-1])


def test_texture_is_skipped_when_disabled(tmp_path, synthetic_clip):
    class _Off(_RecordingTexture):
        enabled = False

    texture = _Off()
    compare_frame(
        _request(tmp_path, synthetic_clip), _Doubler(), seconds=0.2, texture=texture
    )
    assert texture.calls == []


def test_texture_is_built_from_the_request_when_not_supplied(
    tmp_path, synthetic_clip, monkeypatch
):
    import enhancer.compare as compare

    built = {}
    real = compare.TexturePost

    def spy(**kwargs):
        built.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(compare, "TexturePost", spy)

    compare_frame(
        _request(tmp_path, synthetic_clip, detail_retention=0.4, regrain=0.3),
        _Doubler(),
        seconds=0.2,
    )
    assert built["detail_retention"] == 0.4
    assert built["regrain"] == 0.3
    assert built["device"] == "cpu"


# --- timing and clamping --------------------------------------------------


def test_seeking_past_the_end_clamps_rather_than_raising(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    pair = compare_frame(_request(tmp_path, synthetic_clip), _Doubler(), seconds=9999.0)
    assert pair.frame_index <= profile.frame_count - 1
    assert pair.seconds <= profile.duration + 1e-6
    assert pair.after.shape == (480, 640, 3)


def test_negative_seconds_clamps_to_the_first_frame(tmp_path, synthetic_clip):
    pair = compare_frame(_request(tmp_path, synthetic_clip), _Doubler(), seconds=-5.0)
    assert pair.frame_index == 0
    assert pair.seconds == 0.0


def test_frame_at_returns_the_index_it_used(synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    frame, index = frame_at(profile, 1.0)
    assert index == 25  # 25 fps
    assert frame.shape == (240, 320, 3)


def test_frame_at_clamps_past_the_end(synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    frame, index = frame_at(profile, 1_000.0)
    assert index == profile.frame_count - 1
    assert frame.shape == (240, 320, 3)


def test_an_empty_decode_raises_value_error_not_index_error(
    tmp_path, synthetic_clip, monkeypatch
):
    import enhancer.compare as compare

    class _Empty:
        def __init__(self, profile, *a, **k):
            self.profile = profile
            self.start_frame = 0

        def frames(self):
            return iter(())

    monkeypatch.setattr(compare, "Decoder", _Empty)

    with pytest.raises(ValueError, match="produced no frames"):
        compare_frame(_request(tmp_path, synthetic_clip), _Doubler())


# --- still images ---------------------------------------------------------


def test_a_still_goes_down_the_image_path(tmp_path, still, monkeypatch):
    import enhancer.compare as compare

    def boom(*a, **k):
        raise AssertionError("a still must not touch the video machinery")

    monkeypatch.setattr(compare, "Decoder", boom)
    monkeypatch.setattr(compare.SourceProfile, "probe", staticmethod(boom))
    monkeypatch.setattr(compare, "probe_scan", boom)

    pair = compare_frame(_request(tmp_path, still), _Doubler())

    assert pair.source_size == (50, 30)
    assert pair.after.shape == (60, 100, 3)
    assert pair.before.shape == pair.after.shape
    assert pair.scale == 2
    assert pair.seconds == 0.0
    assert pair.frame_index == 0


def test_a_still_still_applies_texture_in_order(tmp_path, still):
    texture = _RecordingTexture()
    compare_frame(_request(tmp_path, still), _Doubler(), texture=texture)
    assert texture.calls == ["detail", "grain"]


def test_a_still_before_is_the_raw_image_enlarged(tmp_path, still):
    from enhancer.images import load_image

    rgb, _ = load_image(still)
    pair = compare_frame(_request(tmp_path, still), _Doubler())
    expected = np.asarray(Image.fromarray(rgb).resize((100, 60), Image.BICUBIC))
    assert np.array_equal(pair.before, expected)
