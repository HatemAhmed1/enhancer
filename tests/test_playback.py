import subprocess

import numpy as np
import pytest
from PIL import Image

from enhancer.playback import ComparePlayer, PlaybackPair, source_index_for
from enhancer.video_io import SourceProfile

# The gray-coded clips below carry their own frame number in their pixel value:
# source frame i is a solid (2i, 2i, 2i). Encoded losslessly with ffv1, the
# value survives the round trip exactly, so the frame a pair is showing can be
# read straight back out of the array. That is what makes it possible to assert
# WHICH source frame landed under a given output frame, rather than only that
# some frame did.
SOURCE_FRAMES = 120  # 5 s at 24 fps
SOURCE_FPS = 24


def _gray_value(frame: np.ndarray) -> int:
    """Recover the source frame number encoded in a solid gray frame."""
    return int(round(float(frame.mean()))) // 2


def _encode_lossless(cmd: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cmd], check=True)


@pytest.fixture(scope="session")
def gray_source(tmp_path_factory):
    """5 s, 24 fps, 320x240, every frame a distinct solid gray."""
    directory = tmp_path_factory.mktemp("gray")
    frames = directory / "frames"
    frames.mkdir()
    for i in range(SOURCE_FRAMES):
        Image.fromarray(np.full((240, 320, 3), i * 2, np.uint8)).save(
            frames / f"f{i:04d}.png"
        )
    path = directory / "source24.mkv"
    _encode_lossless(
        ["-framerate", str(SOURCE_FPS), "-i", str(frames / "f%04d.png"),
         "-c:v", "ffv1", "-pix_fmt", "gbrp", str(path)]
    )
    return path


@pytest.fixture(scope="session")
def gray_output_60(gray_source, tmp_path_factory):
    """The same 5 s at 60 fps and 2x size: an interpolated render, in effect."""
    path = tmp_path_factory.mktemp("gray60") / "output60.mkv"
    _encode_lossless(
        ["-i", str(gray_source),
         "-vf", "scale=640:480:flags=neighbor,fps=60",
         "-c:v", "ffv1", "-pix_fmt", "gbrp", str(path)]
    )
    return path


@pytest.fixture(scope="session")
def gray_preview_60(gray_source, tmp_path_factory):
    """One second of 60 fps output against the five-second source."""
    path = tmp_path_factory.mktemp("preview") / "preview60.mkv"
    _encode_lossless(
        ["-i", str(gray_source), "-t", "1",
         "-vf", "scale=640:480:flags=neighbor,fps=60",
         "-c:v", "ffv1", "-pix_fmt", "gbrp", str(path)]
    )
    return path


@pytest.fixture
def doubled_clip(synthetic_clip, tmp_path):
    """A textured 25 fps clip scaled 2x, frame rate unchanged."""
    path = tmp_path / "doubled.mkv"
    _encode_lossless(
        ["-i", str(synthetic_clip), "-vf", "scale=640:480",
         "-c:v", "ffv1", "-pix_fmt", "gbrp", str(path)]
    )
    return path


def _spy_on_processes(monkeypatch):
    """Record every ffmpeg process any Decoder starts, for leak checking."""
    import enhancer.video_io as video_io

    real = video_io.proc.popen
    started = []

    def spy(cmd, **kwargs):
        process = real(cmd, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(video_io.proc, "popen", spy)
    return started


def _spy_on_decoders(monkeypatch):
    """Record every Decoder built and every frame generator it handed out."""
    import enhancer.playback as playback

    real = playback.Decoder
    built = []
    generators = []

    class Spy(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

        def frames(self):
            generator = super().frames()
            generators.append(generator)
            return generator

    monkeypatch.setattr(playback, "Decoder", Spy)
    return built, generators


# --- time alignment -------------------------------------------------------


def test_source_index_is_computed_from_time_not_from_frame_number():
    # 24 -> 60: two and a half output frames per source frame.
    assert source_index_for(0, 60.0, 24.0) == 0
    assert source_index_for(3, 60.0, 24.0) == 1
    assert source_index_for(150, 60.0, 24.0) == 60
    assert source_index_for(299, 60.0, 24.0) == 119
    # The real cinema case: 23.976 against 60.
    assert source_index_for(600, 60.0, 24000 / 1001) == 239
    # Inverse telecine runs the other way: the output is slower than the source.
    assert source_index_for(24, 24.0, 30.0) == 30


def test_alignment_still_holds_at_the_end_of_the_clip(gray_source, gray_output_60):
    """Index alignment passes at frame 0 and fails everywhere after it.

    Each assertion below reads the source frame number back out of the pixels,
    so this is the real decoded picture and not a restatement of the formula.
    Output frame 296 must be showing source frame 118; frame-number alignment
    would have run off the end of a 120-frame source three seconds earlier.
    """
    with ComparePlayer(gray_source, gray_output_60) as player:
        observed = {}
        while (pair := player.next_pair()) is not None:
            observed[pair.index] = _gray_value(pair.before)

    assert len(observed) == 300
    assert observed[0] == 0
    assert observed[2] == 0  # the first source frame is held for 2.5 output frames
    assert observed[3] == 1
    assert observed[150] == 60  # middle: half way through the output, half way in
    assert observed[296] == 118  # end: NOT 296, and not the clamped final frame
    assert observed[299] == 119

    # Every frame, not just the sampled ones.
    for index, source_frame in observed.items():
        assert source_frame == source_index_for(index, 60.0, float(SOURCE_FPS))


def test_equal_frame_rates_align_one_to_one(synthetic_clip, doubled_clip):
    with ComparePlayer(synthetic_clip, doubled_clip) as player:
        assert player.fps == 25.0
        for expected in range(10):
            pair = player.next_pair()
            assert pair.index == expected
            assert player.source_index == expected


def test_a_held_source_frame_is_not_decoded_again(gray_source, gray_output_60):
    """Output frames 0, 1 and 2 share one source frame — one decode, one resize."""
    with ComparePlayer(gray_source, gray_output_60) as player:
        first, second, third, fourth = (player.next_pair() for _ in range(4))

    assert first.before is second.before is third.before
    assert fourth.before is not third.before
    assert _gray_value(fourth.before) == 1


def test_only_two_decoders_are_ever_opened_for_a_whole_pass(
    gray_source, gray_output_60, monkeypatch
):
    built, _generators = _spy_on_decoders(monkeypatch)
    with ComparePlayer(gray_source, gray_output_60) as player:
        count = 0
        while player.next_pair() is not None:
            count += 1
    assert count == 300
    # One ffmpeg per file for the entire clip, not one per frame.
    assert len(built) == 2


# --- geometry -------------------------------------------------------------


def test_before_and_after_share_a_shape_for_every_pair(gray_source, gray_output_60):
    with ComparePlayer(gray_source, gray_output_60) as player:
        pairs = 0
        while (pair := player.next_pair()) is not None:
            assert pair.before.shape == pair.after.shape == (480, 640, 3)
            assert pair.before.dtype == np.uint8 == pair.after.dtype
            pairs += 1
    assert pairs == 300


def test_before_is_the_bicubic_enlargement_of_the_source_frame(
    synthetic_clip, doubled_clip
):
    """The honest baseline: a 1080p original must not be shown small."""
    from enhancer.compare import frame_at

    raw, index = frame_at(SourceProfile.probe(synthetic_clip), 0.0)
    assert index == 0
    expected = np.asarray(
        Image.fromarray(raw).resize((640, 480), Image.BICUBIC), dtype=np.uint8
    )

    with ComparePlayer(synthetic_clip, doubled_clip, prefer_gpu=False) as player:
        pair = player.next_pair()

    assert pair.before.shape == (480, 640, 3)
    assert np.array_equal(pair.before, expected)


def test_the_card_gives_the_same_enlargement_as_pillow(synthetic_clip, doubled_clip):
    """The fast path must stay the honest baseline, not merely a fast one.

    Pillow's cubic kernel uses a = -0.5 and torch's a = -0.75, so the two are
    not bit-identical. Both are plain bicubic, and on real footage the gap is
    invisible; this pins that it stays a rounding difference rather than
    drifting into a different filter.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA on this machine")

    with ComparePlayer(synthetic_clip, doubled_clip, prefer_gpu=False) as player:
        on_cpu = player.next_pair().before
    with ComparePlayer(synthetic_clip, doubled_clip, prefer_gpu=True) as player:
        on_gpu = player.next_pair().before

    assert on_gpu.shape == on_cpu.shape
    difference = np.abs(on_cpu.astype(int) - on_gpu.astype(int))
    assert difference.mean() < 2.0, "the two enlargements have diverged"


def test_pair_is_frozen(gray_source, gray_output_60):
    with ComparePlayer(gray_source, gray_output_60) as player:
        pair = player.next_pair()
    assert isinstance(pair, PlaybackPair)
    with pytest.raises(Exception):
        pair.index = 7


def test_timeline_properties_describe_the_output(gray_source, gray_output_60):
    player = ComparePlayer(gray_source, gray_output_60)
    assert player.fps == 60.0
    assert player.frame_count == 300
    assert player.duration == pytest.approx(5.0)
    assert player.covers == pytest.approx((0.0, 119 / 24))


# --- seeking --------------------------------------------------------------


def test_seek_returns_the_frame_at_that_time(gray_source, gray_output_60):
    with ComparePlayer(gray_source, gray_output_60) as player:
        player.seek(2.0)
        pair = player.next_pair()

    assert pair.index == 120
    assert pair.seconds == pytest.approx(2.0)
    # 2 s at 24 fps is source frame 48.
    assert _gray_value(pair.before) == 48


def test_seek_can_go_backwards_and_replays_the_same_frames(
    gray_source, gray_output_60
):
    with ComparePlayer(gray_source, gray_output_60) as player:
        player.seek(3.0)
        forward = player.next_pair()
        player.seek(1.0)
        back = player.next_pair()
        player.seek(3.0)
        again = player.next_pair()

    assert _gray_value(back.before) == 24
    assert back.index == 60
    assert again.index == forward.index
    assert np.array_equal(again.before, forward.before)
    assert np.array_equal(again.after, forward.after)


def test_seek_past_the_end_clamps_instead_of_raising(gray_source, gray_output_60):
    with ComparePlayer(gray_source, gray_output_60) as player:
        player.seek(9_999.0)
        pair = player.next_pair()
        assert pair.index == 299
        assert player.next_pair() is None


def test_repeated_seeks_leave_no_ffmpeg_process_running(
    gray_source, gray_output_60, monkeypatch
):
    """Two checks, because a leak shows up two ways.

    The first is direct: every ffmpeg process started by a Decoder is recorded,
    and after the seeks all but the two currently live ones must have exited
    (poll() returning a status rather than None). Decoder.frames() closes its
    pipe and waits for the child in a finally block, so an exited process is
    proof the teardown ran rather than being left to the garbage collector.
    The second is structural: the abandoned frame generators must be closed.
    """
    started = _spy_on_processes(monkeypatch)
    _built, generators = _spy_on_decoders(monkeypatch)

    player = ComparePlayer(gray_source, gray_output_60)
    for step in range(10):
        player.seek(step * 0.4)
        assert player.next_pair() is not None

    # Two files, ten seeks, one process each per seek. Nothing more.
    assert len(started) == 20
    assert len(generators) == 20
    assert all(process.poll() is not None for process in started[:-2])
    assert all(generator.gi_frame is None for generator in generators[:-2])

    player.close()
    assert all(process.poll() is not None for process in started)
    assert all(generator.gi_frame is None for generator in generators)


def test_close_shuts_both_pipes_down(gray_source, gray_output_60, monkeypatch):
    started = _spy_on_processes(monkeypatch)
    with ComparePlayer(gray_source, gray_output_60) as player:
        player.next_pair()
        assert len(started) == 2
        assert all(process.poll() is None for process in started)
    assert all(process.poll() is not None for process in started)


def test_reaching_the_end_releases_the_processes(
    gray_source, gray_preview_60, monkeypatch
):
    started = _spy_on_processes(monkeypatch)
    player = ComparePlayer(gray_source, gray_preview_60)
    player.open()
    while player.next_pair() is not None:
        pass
    # No explicit close: end of stream must let ffmpeg go on its own.
    assert started
    assert all(process.poll() is not None for process in started)


# --- mismatched material --------------------------------------------------


def test_a_short_preview_plays_its_overlap_and_reports_it(
    gray_source, gray_preview_60
):
    """A ten-second preview of a two-hour film is normal, not an error."""
    player = ComparePlayer(gray_source, gray_preview_60)

    assert player.frame_count == 60
    assert player.duration == pytest.approx(1.0)
    start, end = player.covers
    assert start == 0.0
    assert end == pytest.approx(59 / 60, abs=0.02)
    # The source runs five seconds; only the first second of it is covered.
    assert end < SOURCE_FRAMES / SOURCE_FPS

    with player:
        pairs = []
        while (pair := player.next_pair()) is not None:
            pairs.append(pair)

    assert len(pairs) == 60
    assert _gray_value(pairs[0].before) == 0
    assert _gray_value(pairs[-1].before) == source_index_for(59, 60.0, 24.0) == 23
    assert all(p.before.shape == p.after.shape for p in pairs)


def test_a_long_output_against_a_short_source_stops_at_the_overlap(
    gray_source, gray_output_60, tmp_path
):
    """The reverse mismatch: play what both files have, then stop."""
    short = tmp_path / "short_source.mkv"
    _encode_lossless(["-i", str(gray_source), "-t", "1", "-c:v", "ffv1",
                      "-pix_fmt", "gbrp", str(short)])

    player = ComparePlayer(short, gray_output_60)
    assert player.frame_count < 300
    assert player.covers[1] == pytest.approx(23 / 24, abs=0.05)

    with player:
        count = 0
        while player.next_pair() is not None:
            count += 1
    assert count == player.frame_count


# --- end of stream and failures -------------------------------------------


def test_end_of_stream_returns_none_and_keeps_returning_none(
    gray_source, gray_preview_60
):
    with ComparePlayer(gray_source, gray_preview_60) as player:
        while player.next_pair() is not None:
            pass
        for _ in range(5):
            assert player.next_pair() is None


def test_a_truncated_source_never_yields_half_a_pair(
    gray_source, gray_output_60, monkeypatch
):
    """If the source pipe dies early, the answer is None, not a partial pair."""
    import enhancer.playback as playback

    real = playback.Decoder

    class Stubborn(real):
        def frames(self):
            if self.profile.path == gray_source:
                yield from (f for f, _ in zip(super().frames(), range(3)))
            else:
                yield from super().frames()

    monkeypatch.setattr(playback, "Decoder", Stubborn)

    with ComparePlayer(gray_source, gray_output_60) as player:
        pairs = 0
        while (pair := player.next_pair()) is not None:
            assert pair.before is not None and pair.after is not None
            pairs += 1
    # Three source frames cover output frames 0..7 at 24 -> 60.
    assert pairs == 8


def test_probing_a_non_video_raises_value_error(tmp_path, gray_output_60):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a film", encoding="utf-8")
    with pytest.raises(ValueError, match="could not probe"):
        ComparePlayer(junk, gray_output_60)


def test_a_missing_file_raises_value_error(tmp_path, gray_source):
    with pytest.raises(ValueError):
        ComparePlayer(gray_source, tmp_path / "nothing_here.mkv")


def test_an_empty_video_raises_value_error(gray_source, gray_output_60, monkeypatch):
    import dataclasses

    import enhancer.playback as playback

    real = SourceProfile.probe.__func__

    def empty(cls, path):
        return dataclasses.replace(real(cls, path), frame_count=0)

    monkeypatch.setattr(playback.SourceProfile, "probe", classmethod(empty))
    with pytest.raises(ValueError, match="no frames"):
        ComparePlayer(gray_source, gray_output_60)
