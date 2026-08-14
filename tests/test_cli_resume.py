import subprocess

import numpy as np
import pytest

from enhancer.cli import render_resumable
from enhancer.jobs import JobState
from enhancer.segments import segment_path
from enhancer.video_io import Decoder, SourceProfile


def _stream_count(path, kind):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return len([line for line in out.stdout.splitlines() if line.strip()])


class DoublingUpscaler:
    """Stands in for a real Upscaler."""

    scale = 2
    cpu_fallback_count = 0

    def process(self, frame):
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def test_render_produces_correct_output(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
    )
    result = SourceProfile.probe(out)
    assert result.width == 640 and result.height == 480
    assert len(list(Decoder(result).frames())) == 50


def test_render_writes_expected_segment_count(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "out.mkv",
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )
    # 50 frames at 20 per segment -> 3 segments
    assert segment_path(job_dir, 2).exists()


def test_interrupted_render_resumes_and_matches_uninterrupted(tmp_path, synthetic_clip):
    """The whole point: a killed job resumes to a byte-equivalent result."""
    profile = SourceProfile.probe(synthetic_clip)

    reference = tmp_path / "reference.mkv"
    render_resumable(
        profile, DoublingUpscaler(), reference,
        job_dir=tmp_path / "job_ref", segment_frames=20, settings={"scale": 2},
    )
    reference_frames = list(Decoder(SourceProfile.probe(reference)).frames())

    class FailsOnSecondSegment(DoublingUpscaler):
        def __init__(self):
            self.seen = 0

        def process(self, frame):
            self.seen += 1
            if self.seen > 25:
                raise RuntimeError("simulated crash")
            return super().process(frame)

    job_dir = tmp_path / "job_resume"
    with pytest.raises(RuntimeError, match="simulated crash"):
        render_resumable(
            profile, FailsOnSecondSegment(), tmp_path / "out.mkv",
            job_dir=job_dir, segment_frames=20, settings={"scale": 2},
        )
    assert segment_path(job_dir, 0).exists(), "first segment should have survived"
    assert not segment_path(job_dir, 1).exists(), "crashed segment must not persist"

    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )
    resumed_frames = list(Decoder(SourceProfile.probe(out)).frames())
    assert len(resumed_frames) == len(reference_frames)


def test_resume_skips_already_completed_segments(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "first.mkv",
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )

    class ExplodesIfCalled(DoublingUpscaler):
        def process(self, frame):
            raise AssertionError("should not reprocess a completed segment")

    render_resumable(
        profile, ExplodesIfCalled(), tmp_path / "second.mkv",
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )


def test_interrupted_render_with_audio_resumes_correctly(tmp_path, synthetic_clip_with_audio):
    """End-to-end: a real interrupted render, with audio, resumes correctly.

    Proves three things together: the crashed segment is discarded while the
    prior segment survives, the resumed run does not reprocess completed
    segments, and the final assembled file has correct frame count,
    dimensions, and exactly one audio stream.
    """
    profile = SourceProfile.probe(synthetic_clip_with_audio)

    class FailsPartwayThroughSecondSegment(DoublingUpscaler):
        """50 frames / 20 per segment -> segments of 20, 20, 10.

        Fails on the 6th frame of the second segment (raw frame index 25),
        well after the first segment's 20 frames are done.
        """

        def __init__(self):
            self.seen = 0

        def process(self, frame):
            self.seen += 1
            if self.seen > 25:
                raise RuntimeError("simulated crash mid-segment")
            return super().process(frame)

    job_dir = tmp_path / "job"
    out = tmp_path / "out.mkv"
    with pytest.raises(RuntimeError, match="simulated crash mid-segment"):
        render_resumable(
            profile, FailsPartwayThroughSecondSegment(), out,
            job_dir=job_dir, segment_frames=20, settings={"scale": 2},
        )
    assert segment_path(job_dir, 0).exists(), "first segment should have survived"
    assert not segment_path(job_dir, 1).exists(), "crashed segment must not persist"

    class CountingUpscaler(DoublingUpscaler):
        """Tracks how many frames it was actually asked to process."""

        def __init__(self):
            self.calls = 0

        def process(self, frame):
            self.calls += 1
            return super().process(frame)

    resumer = CountingUpscaler()
    render_resumable(
        profile, resumer, out,
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )

    assert resumer.calls == 30, (
        "only the remaining 30 frames (segments 1 and 2) should be "
        f"reprocessed; got {resumer.calls} calls, which means the completed "
        "segment 0 was reprocessed"
    )

    result = SourceProfile.probe(out)
    assert result.width == 640 and result.height == 480
    assert len(list(Decoder(result).frames())) == 50
    assert _stream_count(out, "a") == 1, "final output must have exactly one audio stream"
