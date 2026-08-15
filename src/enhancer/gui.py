"""Desktop interface.

The Qt layer holds no policy. Every decision lives in `RenderRequest`, and the
render itself lives in `RenderJob`, which knows nothing about Qt so it can be
tested without a display.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from .analyze import ScanType, classify_scan, probe_scan
from .requests import RenderRequest
from .restore import RestoreSettings, TexturePost, build_filter_chain
from .video_io import SourceProfile

ProgressFn = Callable[[int, int], None]


class CancelledError(RuntimeError):
    """Raised inside the progress callback to abort a render.

    Cancelling is the crash path taken deliberately: `write_segment` deletes its
    partial file, the job journal keeps every completed segment, and the next
    run resumes. No separate teardown is needed, and cancelling cannot corrupt
    output.
    """


class RenderJob:
    """One render, cancellable, with no Qt dependency."""

    def __init__(self, request: RenderRequest, upscaler=None, flow_model=None) -> None:
        self.request = request
        self._upscaler = upscaler
        self._flow_model = flow_model
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _profile(self) -> SourceProfile:
        profile = SourceProfile.probe(self.request.source)
        if self.request.preview_frames is not None:
            # Clamp the source so the preview travels the identical code path
            # as a real render. Previewing anything else would preview something
            # other than what is about to be rendered.
            profile = dataclasses.replace(
                profile,
                frame_count=min(profile.frame_count, self.request.preview_frames),
            )
        return profile

    def run(self, on_progress: ProgressFn | None = None) -> Path:
        from .cli import render_resumable
        from .images import is_image, upscale_image

        req = self.request

        if is_image(req.source):
            # A still has no frame rate, no segments and nothing to resume.
            texture = TexturePost(
                detail_retention=req.detail_retention,
                regrain=req.regrain,
                device=str(getattr(self._upscaler, "device", "cpu")),
            )
            out = upscale_image(req.source, req.output, self._upscaler, texture=texture)
            if on_progress:
                on_progress(1, 1)
            return out

        profile = self._profile()
        req.validate_against(profile.fps)

        if req.no_restore:
            settings = RestoreSettings(
                deblock=0.0, degrain=0.0, detail_retention=0.0, regrain=0.0
            )
            scan, field_order, video_filter = ScanType.PROGRESSIVE, "tff", ""
        else:
            analysis = probe_scan(req.source)
            scan = classify_scan(analysis)
            field_order = analysis.field_order
            settings = RestoreSettings(
                deblock=req.deblock, degrain=req.degrain,
                detail_retention=req.detail_retention, regrain=req.regrain,
            )
            video_filter = build_filter_chain(scan, field_order, settings)

        upscaler = self._upscaler
        device = getattr(upscaler, "device", "cpu")
        texture = TexturePost(
            detail_retention=settings.detail_retention,
            regrain=settings.regrain,
            device=str(device),
        )

        def progress(done: int, total: int) -> None:
            if self._cancelled:
                raise CancelledError("render cancelled")
            if on_progress:
                on_progress(done, total)

        return render_resumable(
            profile, upscaler, req.output,
            job_dir=req.job_dir,
            segment_frames=req.segment_frames,
            settings=req.settings_dict(upscaler.scale, req.tile, video_filter),
            on_progress=progress,
            video_filter=video_filter,
            texture=texture,
            interpolate_to=req.target_fps,
            flow_model=self._flow_model,
            scene_threshold=req.scene_threshold,
        )
