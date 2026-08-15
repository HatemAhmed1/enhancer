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


def test_decimating_filter_forces_a_single_segment(tmp_path, synthetic_clip):
    """Decimation changes the output frame count after -ss has already seeked
    in input time, so segment boundaries computed from profile.frame_count
    would not line up with what ffmpeg actually decoded. Rendering as one
    segment sidesteps the misalignment entirely, at the cost of resumability
    for this job."""
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "out.mkv",
        job_dir=job_dir, segment_frames=10, settings={"scale": 2},
        video_filter="decimate",
    )
    assert segment_path(job_dir, 0).exists()
    for i in range(1, 5):
        assert not segment_path(job_dir, i).exists()


def test_non_decimating_filter_still_segments_normally(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "out.mkv",
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
        video_filter="hqdn3d=4:3:6:4",
    )
    # 50 frames at 20 per segment -> 3 segments
    assert segment_path(job_dir, 2).exists()


def test_zero_frame_count_raises_a_clear_error(tmp_path, synthetic_clip):
    """Some containers report neither frame count nor duration.

    Without this guard the failure surfaces much later as an opaque ffmpeg
    concat error against an empty file list.
    """
    from dataclasses import replace

    profile = replace(SourceProfile.probe(synthetic_clip), frame_count=0)
    with pytest.raises(ValueError, match="could not determine a frame count"):
        render_resumable(
            profile, DoublingUpscaler(), tmp_path / "out.mkv",
            job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
        )


class _CrossFade:
    """Linear blend standing in for RIFE."""

    def __call__(self, a, b, t):
        return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


def test_interpolated_render_produces_the_planned_frame_count(tmp_path, synthetic_clip):
    """50 source frames at 25 fps, doubled, is 100 output frames."""
    profile = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(),
    )
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 100


def test_interpolated_output_reports_the_target_frame_rate(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(),
    )
    assert SourceProfile.probe(out).fps == pytest.approx(50.0, abs=0.1)


def test_non_integer_ratio_renders_end_to_end(tmp_path, synthetic_clip):
    """25 -> 60 fps is 2.4x, which fixed-insertion schemes cannot express."""
    profile = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
        interpolate_to=60.0, flow_model=_CrossFade(),
    )
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 120


def test_interpolated_render_still_resumes(tmp_path, synthetic_clip):
    """Resume matters most here: interpolation is when renders get longest."""
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"

    class FailsLate(DoublingUpscaler):
        def __init__(self):
            self.seen = 0

        def process(self, frame):
            self.seen += 1
            if self.seen > 15:
                raise RuntimeError("simulated crash")
            return super().process(frame)

    with pytest.raises(RuntimeError, match="simulated crash"):
        render_resumable(
            profile, FailsLate(), tmp_path / "crashed.mkv",
            job_dir=job_dir, segment_frames=20, settings={"scale": 2},
            interpolate_to=50.0, flow_model=_CrossFade(),
        )
    assert segment_path(job_dir, 0).exists()

    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(),
    )
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 100


def test_interpolation_without_a_flow_model_is_rejected(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    with pytest.raises(ValueError, match="flow_model"):
        render_resumable(
            profile, DoublingUpscaler(), tmp_path / "out.mkv",
            job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
            interpolate_to=50.0,
        )


def test_interpolation_with_inverse_telecine_is_rejected(tmp_path, synthetic_clip):
    """Decimation changes the frame count interpolation timing is derived from.

    The plan would be built from the probed count while the decoder supplies
    roughly four fifths of it, so the plan indexes past the end of the source.
    """
    profile = SourceProfile.probe(synthetic_clip)
    with pytest.raises(ValueError, match="inverse telecine"):
        render_resumable(
            profile, DoublingUpscaler(), tmp_path / "out.mkv",
            job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
            interpolate_to=50.0, flow_model=_CrossFade(),
            video_filter="fieldmatch,decimate",
        )


def test_interpolation_caps_segment_size_for_memory(tmp_path, synthetic_clip):
    """Segments must shrink with output resolution, not stay at the default.

    A segment's upscaled frames are all resident at once during interpolation.
    """
    from enhancer.cli import INTERPOLATION_RAM_BUDGET

    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "out.mkv",
        job_dir=job_dir, segment_frames=100000, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(),
    )
    import json
    journal = json.loads((job_dir / "job.json").read_text())
    per_frame = (profile.width * 2) * (profile.height * 2) * 3
    held = (journal["segment_frames"] / 2.0) * per_frame
    assert held <= INTERPOLATION_RAM_BUDGET * 1.1, "segment size must respect the RAM budget"


class _GrainSpy:
    """Records which frame indices grain was applied to."""

    def __init__(self):
        self.detail_calls = []
        self.grain_calls = []
        self.detail_retention = 0.25
        self.regrain = 0.6

    enabled = True
    detail_enabled = True
    grain_enabled = True

    def apply_detail(self, output, source):
        self.detail_calls.append(None)
        return output

    def apply_grain(self, frame, index=0):
        self.grain_calls.append(index)
        return frame


def test_grain_is_applied_once_per_output_frame_when_interpolating(tmp_path, synthetic_clip):
    """Graining before interpolation would blend two noise fields together.

    That attenuates grain on synthesized frames by 1/sqrt(2); at 24 to 60 fps
    four frames in five are synthesized, producing visible grain pulsing.
    """
    profile = SourceProfile.probe(synthetic_clip)
    spy = _GrainSpy()
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "out.mkv",
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(), texture=spy,
    )
    assert len(spy.grain_calls) == 100, "one grain call per OUTPUT frame"
    assert sorted(spy.grain_calls) == list(range(100)), "grain seeds must be output indices"

    # Detail runs per SOURCE frame, plus one per segment boundary: the last
    # output frame of a segment brackets source j to j+1, and the next segment
    # starts at j+1, so boundary frames are upscaled twice. 50 sources across
    # 5 segments gives 4 overlaps.
    assert len(spy.detail_calls) == 54


# --- dual pass --------------------------------------------------------------


def test_dual_pass_produces_both_files(tmp_path, synthetic_clip):
    from enhancer.cli import render_dual_pass

    profile = SourceProfile.probe(synthetic_clip)
    mid, final = render_dual_pass(
        profile, DoublingUpscaler(), DoublingUpscaler(),
        output=tmp_path / "out.mkv", job_dir=tmp_path / "job",
        segment_frames=20, settings={"scale": 2},
    )
    assert mid.exists() and final.exists()


def test_dual_pass_applies_both_scale_factors(tmp_path, synthetic_clip):
    """Two 2x passes on 320x240 give 1280x960."""
    from enhancer.cli import render_dual_pass

    profile = SourceProfile.probe(synthetic_clip)
    mid, final = render_dual_pass(
        profile, DoublingUpscaler(), DoublingUpscaler(),
        output=tmp_path / "out.mkv", job_dir=tmp_path / "job",
        segment_frames=20, settings={"scale": 2},
    )
    assert (SourceProfile.probe(mid).width, SourceProfile.probe(mid).height) == (640, 480)
    assert (SourceProfile.probe(final).width, SourceProfile.probe(final).height) == (1280, 960)


def test_dual_pass_preserves_the_frame_count(tmp_path, synthetic_clip):
    from enhancer.cli import render_dual_pass

    profile = SourceProfile.probe(synthetic_clip)
    _mid, final = render_dual_pass(
        profile, DoublingUpscaler(), DoublingUpscaler(),
        output=tmp_path / "out.mkv", job_dir=tmp_path / "job",
        segment_frames=20, settings={"scale": 2},
    )
    assert len(list(Decoder(SourceProfile.probe(final)).frames())) == 50


def test_dual_pass_keeps_the_intermediate_for_inspection(tmp_path, synthetic_clip):
    """The whole point of stopping mid-way is having something to look at."""
    from enhancer.cli import render_dual_pass

    profile = SourceProfile.probe(synthetic_clip)
    mid, _final = render_dual_pass(
        profile, DoublingUpscaler(), DoublingUpscaler(),
        output=tmp_path / "out.mkv", job_dir=tmp_path / "job",
        segment_frames=20, settings={"scale": 2},
    )
    assert mid.exists()
    assert "pass1" in mid.name


def test_dual_pass_can_discard_the_intermediate(tmp_path, synthetic_clip):
    from enhancer.cli import render_dual_pass

    profile = SourceProfile.probe(synthetic_clip)
    mid, _final = render_dual_pass(
        profile, DoublingUpscaler(), DoublingUpscaler(),
        output=tmp_path / "out.mkv", job_dir=tmp_path / "job",
        segment_frames=20, settings={"scale": 2}, keep_intermediate=False,
    )
    assert not mid.exists()


def test_dual_pass_uses_separate_job_dirs_per_pass(tmp_path, synthetic_clip):
    """An interruption in pass two must not re-run pass one."""
    from enhancer.cli import render_dual_pass

    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_dual_pass(
        profile, DoublingUpscaler(), DoublingUpscaler(),
        output=tmp_path / "out.mkv", job_dir=job_dir,
        segment_frames=20, settings={"scale": 2},
    )
    assert (job_dir / "pass1" / "job.json").exists()
    assert (job_dir / "pass2" / "job.json").exists()


def test_dual_pass_progress_spans_both_passes(tmp_path, synthetic_clip):
    from enhancer.cli import render_dual_pass

    profile = SourceProfile.probe(synthetic_clip)
    seen = []
    render_dual_pass(
        profile, DoublingUpscaler(), DoublingUpscaler(),
        output=tmp_path / "out.mkv", job_dir=tmp_path / "job",
        segment_frames=20, settings={"scale": 2},
        on_progress=lambda d, t: seen.append((d, t)),
    )
    assert seen[-1][0] == seen[-1][1], "progress must reach 100% across both passes"
    assert seen[-1][1] == 100, "50 frames over two passes"


def test_dual_pass_can_use_different_models_per_pass(tmp_path, synthetic_clip):
    """Commonly a texture-preserving model first, a fast one second."""
    from enhancer.cli import render_dual_pass

    class _Tripler(DoublingUpscaler):
        scale = 3

        def process(self, frame):
            return np.repeat(np.repeat(frame, 3, axis=0), 3, axis=1)

    profile = SourceProfile.probe(synthetic_clip)
    _mid, final = render_dual_pass(
        profile, DoublingUpscaler(), _Tripler(),
        output=tmp_path / "out.mkv", job_dir=tmp_path / "job",
        segment_frames=20, settings={"scale": 2},
    )
    assert SourceProfile.probe(final).width == 320 * 2 * 3
