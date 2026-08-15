import numpy as np
import pytest

from enhancer.gui import CancelledError, RenderJob
from enhancer.requests import RenderRequest
from enhancer.video_io import Decoder, SourceProfile


class _Doubler:
    scale = 2
    device = "cpu"
    cpu_fallback_count = 0

    def process(self, frame):
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def _request(tmp_path, clip, **kw):
    kw.setdefault("no_restore", True)
    return RenderRequest(
        model=tmp_path / "unused.pth", source=clip,
        output=tmp_path / "out.mkv", **kw
    )


def test_job_reports_progress(tmp_path, synthetic_clip):
    seen = []
    job = RenderJob(_request(tmp_path, synthetic_clip), upscaler=_Doubler())
    job.run(on_progress=lambda d, t: seen.append((d, t)))
    assert seen
    assert seen[-1][0] == seen[-1][1] == 50


def test_job_produces_the_output_file(tmp_path, synthetic_clip):
    req = _request(tmp_path, synthetic_clip)
    RenderJob(req, upscaler=_Doubler()).run()
    assert req.output.exists()
    assert SourceProfile.probe(req.output).width == 640


def test_cancel_stops_the_render(tmp_path, synthetic_clip):
    job = RenderJob(_request(tmp_path, synthetic_clip), upscaler=_Doubler())

    def cancel_after_ten(done, total):
        if done >= 10:
            job.cancel()

    with pytest.raises(CancelledError):
        job.run(on_progress=cancel_after_ten)


def test_cancel_leaves_the_job_resumable(tmp_path, synthetic_clip):
    """Cancel is the crash path taken deliberately, so resume must work."""
    req = _request(tmp_path, synthetic_clip, segment_frames=10)
    job = RenderJob(req, upscaler=_Doubler())

    def cancel_late(done, total):
        if done >= 25:
            job.cancel()

    with pytest.raises(CancelledError):
        job.run(on_progress=cancel_late)

    resumed = RenderJob(req, upscaler=_Doubler())
    resumed.run()
    assert len(list(Decoder(SourceProfile.probe(req.output)).frames())) == 50


def test_cancel_does_not_discard_completed_segments(tmp_path, synthetic_clip):
    from enhancer.segments import segment_path

    req = _request(tmp_path, synthetic_clip, segment_frames=10)
    job = RenderJob(req, upscaler=_Doubler())

    def cancel_late(done, total):
        if done >= 25:
            job.cancel()

    with pytest.raises(CancelledError):
        job.run(on_progress=cancel_late)

    assert segment_path(req.job_dir, 0).exists()
    assert segment_path(req.job_dir, 1).exists()


def test_preview_renders_only_the_requested_frames(tmp_path, synthetic_clip):
    req = _request(tmp_path, synthetic_clip, preview_frames=10)
    RenderJob(req, upscaler=_Doubler()).run()
    assert len(list(Decoder(SourceProfile.probe(req.output)).frames())) == 10


def test_preview_does_not_touch_the_real_job_dir(tmp_path, synthetic_clip):
    real = _request(tmp_path, synthetic_clip)
    preview = _request(tmp_path, synthetic_clip, preview_frames=10)
    RenderJob(preview, upscaler=_Doubler()).run()
    assert not real.job_dir.exists()


def test_preview_output_is_named_distinctly(tmp_path, synthetic_clip):
    req = _request(tmp_path, synthetic_clip, preview_frames=10)
    assert "preview" in req.output.name


def test_not_cancelled_by_default(tmp_path, synthetic_clip):
    assert not RenderJob(_request(tmp_path, synthetic_clip), upscaler=_Doubler()).cancelled


def test_restoration_path_runs_end_to_end(tmp_path, synthetic_clip):
    """Exercise the analyse-and-filter branch, not just --no-restore."""
    req = _request(tmp_path, synthetic_clip, no_restore=False, preview_frames=10)
    RenderJob(req, upscaler=_Doubler()).run()
    assert req.output.exists()


# --- preview must never refuse to run --------------------------------------


def test_preview_reruns_after_settings_change(tmp_path, synthetic_clip):
    """A preview exists to try settings out.

    Refusing to resume it because the settings changed defeats its entire
    purpose, which is exactly what the resume guard did before this.
    """
    first = _request(tmp_path, synthetic_clip, preview_frames=10, degrain=0.0,
                     no_restore=False, detail_retention=0.1)
    RenderJob(first, upscaler=_Doubler()).run()

    second = _request(tmp_path, synthetic_clip, preview_frames=10, degrain=0.0,
                      no_restore=False, detail_retention=0.5)
    RenderJob(second, upscaler=_Doubler()).run()
    assert second.output.exists()


def test_preview_reruns_after_a_model_change(tmp_path, synthetic_clip):
    class _Tripler(_Doubler):
        scale = 3

        def process(self, frame):
            return np.repeat(np.repeat(frame, 3, axis=0), 3, axis=1)

    req = _request(tmp_path, synthetic_clip, preview_frames=10)
    RenderJob(req, upscaler=_Doubler()).run()
    assert SourceProfile.probe(req.output).width == 640

    req2 = _request(tmp_path, synthetic_clip, preview_frames=10)
    RenderJob(req2, upscaler=_Tripler()).run()
    assert SourceProfile.probe(req2.output).width == 960


def test_preview_discards_previous_job_state(tmp_path, synthetic_clip):
    req = _request(tmp_path, synthetic_clip, preview_frames=10, segment_frames=5)
    RenderJob(req, upscaler=_Doubler()).run()
    marker = req.job_dir / "stale_marker.txt"
    marker.write_text("left over from a previous preview")

    RenderJob(req, upscaler=_Doubler()).run()
    assert not marker.exists(), "a new preview must start from a clean directory"


def test_preview_can_be_run_many_times_in_a_row(tmp_path, synthetic_clip):
    for detail in (0.0, 0.3, 0.6, 0.9):
        req = _request(tmp_path, synthetic_clip, preview_frames=10,
                       no_restore=False, detail_retention=detail)
        RenderJob(req, upscaler=_Doubler()).run()
        assert req.output.exists()


def test_full_render_still_refuses_after_a_settings_change(tmp_path, synthetic_clip):
    """The guard must stay in place where it belongs.

    A real render spliced from two different settings has a visible seam.
    """
    from enhancer.jobs import SettingsMismatch

    first = _request(tmp_path, synthetic_clip, no_restore=False, detail_retention=0.1)
    RenderJob(first, upscaler=_Doubler()).run()

    second = _request(tmp_path, synthetic_clip, no_restore=False, detail_retention=0.9)
    with pytest.raises(SettingsMismatch):
        RenderJob(second, upscaler=_Doubler()).run()
