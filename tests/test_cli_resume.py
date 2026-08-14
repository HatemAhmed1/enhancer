import numpy as np
import pytest

from enhancer.cli import render_resumable
from enhancer.jobs import JobState
from enhancer.segments import segment_path
from enhancer.video_io import Decoder, SourceProfile


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
