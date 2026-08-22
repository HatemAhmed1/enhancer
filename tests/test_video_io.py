import numpy as np
import pytest

from enhancer.video_io import Decoder, Encoder, SourceProfile, _parse_probe, _parse_fraction


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


def test_decoder_yields_correct_shape_and_count(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p).frames())
    assert len(frames) == 50, "2 seconds at 25 fps"
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8


def test_decoder_frames_are_not_all_identical(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p).frames())
    assert not np.array_equal(frames[0], frames[-1])


def test_encoder_writes_playable_file(tmp_path, synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mp4"
    with Encoder(out, width=320, height=240, fps=25.0, source=p) as enc:
        for frame in Decoder(p).frames():
            enc.write(frame)
    assert out.exists() and out.stat().st_size > 0
    result = SourceProfile.probe(out)
    assert result.width == 320 and result.height == 240


def test_encode_roundtrip_preserves_frame_count(tmp_path, synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mp4"
    with Encoder(out, width=320, height=240, fps=25.0, source=p) as enc:
        for frame in Decoder(p).frames():
            enc.write(frame)
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 50


def test_encoder_scales_output_dimensions(tmp_path, synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "big.mp4"
    with Encoder(out, width=640, height=480, fps=25.0, source=p) as enc:
        for frame in Decoder(p).frames():
            enc.write(np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1))
    result = SourceProfile.probe(out)
    assert result.width == 640 and result.height == 480


def test_decoder_start_frame_skips_leading_frames(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(p).frames())
    seeked = list(Decoder(p, start_frame=10).frames())
    assert len(seeked) == len(all_frames) - 10


def test_decoder_start_frame_lands_on_the_right_frame(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(p).frames())
    seeked = list(Decoder(p, start_frame=10).frames())
    assert np.array_equal(seeked[0], all_frames[10])


def test_decoder_max_frames_limits_output(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    assert len(list(Decoder(p, max_frames=7).frames())) == 7


def test_decoder_start_and_max_select_a_window(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(p).frames())
    window = list(Decoder(p, start_frame=20, max_frames=5).frames())
    assert len(window) == 5
    assert np.array_equal(window[0], all_frames[20])


def test_decoder_start_frame_zero_matches_no_seek(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    assert len(list(Decoder(p, start_frame=0).frames())) == 50


def test_decoder_accepts_a_filter_chain(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p, video_filter="hqdn3d=4:3:6:4").frames())
    assert len(frames) == 50
    assert frames[0].shape == (240, 320, 3)


def test_filtered_output_differs_from_unfiltered(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    plain = list(Decoder(p).frames())[0]
    blurred = list(Decoder(p, video_filter="boxblur=4").frames())[0]
    assert not np.array_equal(plain, blurred)


def test_empty_filter_chain_is_equivalent_to_none(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    a = list(Decoder(p, video_filter="").frames())
    b = list(Decoder(p).frames())
    assert len(a) == len(b)
    assert np.array_equal(a[0], b[0])


def test_decimating_filter_reduces_frame_count(synthetic_clip):
    """IVTC drops frames; the decoder must not assume a fixed count."""
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p, video_filter="select='not(mod(n,2))'").frames())
    assert len(frames) == 25


def test_filter_combines_with_seeking(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(
        Decoder(p, start_frame=10, max_frames=5, video_filter="hqdn3d=4:3:6:4").frames()
    )
    assert len(frames) == 5


def test_the_pipe_buffer_does_not_scale_with_frame_size():
    """Sizing it per frame made it 99 MB at 4K and decoding four times slower.

    Python's buffered reader fills the whole buffer before returning the first
    frame, so a huge buffer stalls the stream: measured 5.4 fps at 4K against
    22.0 fps with a fixed megabyte.
    """
    from enhancer.video_io import PIPE_BUFFER_BYTES

    uhd_frame_bytes = 3840 * 2160 * 3
    assert PIPE_BUFFER_BYTES < uhd_frame_bytes, "buffer would stall 4K decoding"
    assert PIPE_BUFFER_BYTES >= 1 << 16, "a tiny buffer costs a syscall per read"
