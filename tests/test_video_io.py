import pytest

from enhancer.video_io import SourceProfile, _parse_probe, _parse_fraction


def test_parse_fraction_handles_rationals():
    assert _parse_fraction("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert _parse_fraction("25/1") == 25.0


def test_parse_fraction_handles_zero_denominator():
    assert _parse_fraction("0/0") == 0.0


def test_parse_probe_extracts_core_fields():
    raw = {
        "streams": [{
            "codec_type": "video", "width": 720, "height": 576,
            "r_frame_rate": "25/1", "avg_frame_rate": "25/1",
            "nb_frames": "500", "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "16:15", "field_order": "tt",
            "color_primaries": "bt470bg", "color_transfer": "bt709",
            "color_space": "bt470bg",
        }],
        "format": {"duration": "20.0"},
    }
    p = _parse_probe(raw)
    assert p.width == 720 and p.height == 576
    assert p.fps == 25.0
    assert p.frame_count == 500
    assert p.sar == "16:15"
    assert p.interlaced is True
    assert p.color_space == "bt470bg"


def test_parse_probe_marks_progressive():
    raw = {
        "streams": [{
            "codec_type": "video", "width": 1920, "height": 1080,
            "r_frame_rate": "24/1", "avg_frame_rate": "24/1",
            "nb_frames": "100", "pix_fmt": "yuv420p", "field_order": "progressive",
        }],
        "format": {"duration": "4.16"},
    }
    assert _parse_probe(raw).interlaced is False


def test_parse_probe_raises_without_video_stream():
    with pytest.raises(ValueError, match="no video stream"):
        _parse_probe({"streams": [{"codec_type": "audio"}], "format": {}})


def test_probe_real_clip(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    assert p.width == 320 and p.height == 240
    assert p.fps == pytest.approx(25.0)
    assert p.interlaced is False
