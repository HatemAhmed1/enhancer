import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest

from enhancer import system
from enhancer.video_io import Decoder, Encoder, SourceProfile, _parse_probe, _parse_fraction


def ffprobe_stream(path):
    """codec_name / pix_fmt / profile straight from the file, not from our args."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,profile,width,height", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return dict(
        line.split("=", 1) for line in out.stdout.splitlines() if "=" in line
    )


def encode_clip(dest, clip, **kwargs):
    """Push a real decoded clip through Encoder and return the finished path."""
    profile = SourceProfile.probe(clip)
    with Encoder(
        dest, width=profile.width, height=profile.height, fps=profile.fps,
        source=profile, **kwargs,
    ) as enc:
        for frame in Decoder(profile).frames():
            enc.write(frame)
    return dest


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


def test_a_resizing_filter_needs_its_output_size_declared(tmp_path):
    """Without it the read loop pulls source-sized chunks from a scaled stream.

    Every frame after the first is then torn, silently, because the reads and
    the frame boundaries have drifted apart. Asserting the getter alone would
    not notice: this decodes real frames and checks what comes out.
    """
    import subprocess

    from enhancer.video_io import Decoder, SourceProfile

    clip = tmp_path / "solid.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=c=red:size=640x480:rate=10:duration=1",
         "-c:v", "ffv1", str(clip)],
        check=True, capture_output=True,
    )
    profile = SourceProfile.probe(clip)

    frames = list(
        Decoder(profile, video_filter="scale=320:240", frame_size=(320, 240)).frames()
    )
    assert len(frames) >= 5, "the scaled stream produced almost nothing"
    for frame in frames:
        assert frame.shape == (240, 320, 3)
        # A solid red source stays solid unless the reads have slipped out of
        # step with the frame boundaries, which is exactly the tearing bug.
        assert frame[..., 0].min() > 200, "frame is torn: reads lost alignment"
        assert frame[..., 1].max() < 60


def test_the_frame_size_defaults_to_the_profile():
    from enhancer.video_io import Decoder, SourceProfile

    profile = SourceProfile(
        path=Path("x.mp4"), width=3840, height=2160, fps=24.0, frame_count=10,
        pix_fmt="yuv420p", sar="1:1", interlaced=False, field_order="tff",
        color_primaries="", color_transfer="", color_space="", duration=1.0,
    )
    assert Decoder(profile).output_size == (3840, 2160)
    scaled = Decoder(profile, video_filter="scale=-2:720", frame_size=(1280, 720))
    assert scaled.output_size == (1280, 720)


# --------------------------------------------------------------------------
# Encoder selection: this used to be hardcoded to hevc_nvenc with no fallback,
# so every render died at the encode step on any machine without an NVIDIA GPU.
# --------------------------------------------------------------------------


def _profile(tmp_path):
    return SourceProfile(
        path=tmp_path / "x.mp4", width=320, height=240, fps=25.0, frame_count=10,
        pix_fmt="yuv420p", sar="1:1", interlaced=False, field_order="progressive",
        color_primaries="bt709", color_transfer="bt709", color_space="bt709",
        duration=1.0,
    )


def test_the_default_codec_is_not_hardcoded(tmp_path):
    """It must come from system.py, so an AMD or Intel or CPU-only box works."""
    enc = Encoder(tmp_path / "o.mkv", 320, 240, 25.0, _profile(tmp_path))
    assert enc.codec == system.default_encoder()


def test_an_explicit_codec_still_wins(tmp_path, monkeypatch):
    def explode():
        raise AssertionError("an explicit codec must not trigger a probe")

    monkeypatch.setattr(system, "default_encoder", explode)
    enc = Encoder(tmp_path / "o.mkv", 320, 240, 25.0, _profile(tmp_path), codec="libx265")
    assert enc.codec == "libx265"


@pytest.mark.parametrize(
    "codec, flag",
    [("libx265", "-crf"), ("libx264", "-crf"), ("hevc_nvenc", "-cq"),
     ("hevc_qsv", "-global_quality")],
)
def test_the_quality_flag_follows_the_codec(tmp_path, codec, flag):
    """-cq is NVENC's alone. libx265 ignores it silently and encodes at its
    own default, so the wrong flag costs quality without any error at all.
    """
    cmd = Encoder(
        tmp_path / "o.mkv", 320, 240, 25.0, _profile(tmp_path), codec=codec, quality=18
    )._build_command()
    assert flag in cmd
    assert cmd[cmd.index(flag) + 1] == "18"
    assert "-cq" not in cmd or codec.endswith("_nvenc")


def test_libx265_writes_a_real_ten_bit_hevc_file(tmp_path, synthetic_clip):
    """The guaranteed floor, exercised for real rather than argument-checked."""
    out = encode_clip(tmp_path / "x265.mkv", synthetic_clip, codec="libx265", bit_depth=10)
    info = ffprobe_stream(out)
    assert info["codec_name"] == "hevc"
    assert info["pix_fmt"] == "yuv420p10le"
    assert info["profile"] == "Main 10"
    assert out.stat().st_size > 1000


def test_the_detected_encoder_writes_a_real_file(tmp_path, synthetic_clip):
    """Whatever system.py picked here has to survive an actual encode."""
    codec = system.default_encoder()
    out = encode_clip(tmp_path / "auto.mkv", synthetic_clip, codec=codec, bit_depth=10)
    info = ffprobe_stream(out)
    assert info["codec_name"] in {"hevc", "h264"}
    assert int(info["width"]) == 320 and int(info["height"]) == 240
    assert out.stat().st_size > 1000
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 50


def test_quality_actually_reaches_the_encoder(tmp_path, synthetic_clip):
    """The regression that argument assertions cannot catch.

    A quality flag the encoder does not recognise is not an error: ffmpeg warns
    "has not been used for any stream", exits 0, and encodes at its default. The
    only proof the flag landed is that the output size responds to it.
    """
    good = encode_clip(tmp_path / "q10.mkv", synthetic_clip, codec="libx265", quality=10)
    poor = encode_clip(tmp_path / "q40.mkv", synthetic_clip, codec="libx265", quality=40)
    assert good.stat().st_size > poor.stat().st_size * 1.5


def test_an_eight_bit_only_encoder_falls_back_instead_of_failing(
    tmp_path, synthetic_clip, caplog
):
    """libx264 is treated as 8-bit only; asking for 10 must degrade, not die."""
    with caplog.at_level(logging.WARNING, logger="enhancer.video_io"):
        out = encode_clip(
            tmp_path / "x264.mkv", synthetic_clip, codec="libx264", bit_depth=10
        )
    info = ffprobe_stream(out)
    assert info["pix_fmt"] == "yuv420p"
    assert info["codec_name"] == "h264"
    assert "10-bit" in caplog.text and "libx264" in caplog.text


def test_eight_bit_is_honoured_when_asked_for(tmp_path, synthetic_clip):
    out = encode_clip(tmp_path / "eight.mkv", synthetic_clip, codec="libx265", bit_depth=8)
    assert ffprobe_stream(out)["pix_fmt"] == "yuv420p"


def test_colour_metadata_still_rides_along(tmp_path):
    """Unchanged behaviour; guarded here because the command builder moved."""
    cmd = Encoder(
        tmp_path / "o.mkv", 320, 240, 25.0, _profile(tmp_path), codec="libx265"
    )._build_command()
    assert cmd[cmd.index("-color_primaries") + 1] == "bt709"
    assert cmd[cmd.index("-color_trc") + 1] == "bt709"
    assert cmd[cmd.index("-colorspace") + 1] == "bt709"
