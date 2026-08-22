"""Before/after for a single frame, so settings can be judged in a second.

Committing to a seventeen-hour render to find out whether skin went waxy is the
worst feedback loop in this project. This module produces one honest pair of
images for one instant of the source, travelling the same code path a real
render would, and nothing else. There is no Qt here on purpose: the engine must
be testable without a display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .analyze import classify_scan, probe_scan
from .images import is_image, load_image
from .requests import RenderRequest
from .restore import RestoreSettings, TexturePost, build_filter_chain
from .video_io import Decoder, SourceProfile


@dataclass(frozen=True)
class ComparePair:
    """One source instant, before and after, at identical pixel dimensions."""

    before: np.ndarray  # HWC uint8 RGB, already scaled up to the after size
    after: np.ndarray  # HWC uint8 RGB
    source_size: tuple[int, int]  # (w, h) of the true original frame
    scale: int
    seconds: float  # timestamp actually used, after clamping
    frame_index: int


def _match_size(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Scale `before` up to `after`'s exact dimensions with bicubic.

    THIS IS THE POINT OF THE WHOLE COMPARISON AND MUST NOT BE REMOVED AS A
    REDUNDANT RESIZE. Showing a 1080p original beside a 4K result flatters the
    model for free: the viewer reads "bigger" as "better" before looking at a
    single pixel of texture. The viewer displays both halves at identical pixel
    dimensions, so the original is brought up to the result's size with plain
    bicubic first — the cheapest possible upscale, and the honest baseline. What
    remains visible between the two is only what the model actually did.

    Bicubic specifically: it is the standard "no model" enlargement, neither
    softened like bilinear nor sharpened like Lanczos, so it neither flatters
    nor sandbags the comparison.

    Pillow rather than the graphics card. A CUDA version of this was written and
    removed: the transfers cost more than the kernel saves, so end to end it was
    about twice as fast rather than the nine times a kernel-only timing seemed
    to promise — and playback, the only caller that needed the speed, now scales
    inside ffmpeg instead and does not reach this function at all.
    """
    height, width = after.shape[:2]
    if before.shape[0] == height and before.shape[1] == width:
        return np.ascontiguousarray(before)
    resized = Image.fromarray(before, mode="RGB").resize(
        (width, height), Image.BICUBIC
    )
    return np.asarray(resized, dtype=np.uint8)


def _first_frame(decoder: Decoder, what: str) -> np.ndarray:
    """Pull exactly one frame, or explain clearly why there wasn't one.

    Decoder.frames() yields read-only views over the pipe buffer, so the frame
    is copied before the generator is closed and the buffer reused.
    """
    for frame in decoder.frames():
        return np.array(frame, dtype=np.uint8, copy=True, order="C")
    raise ValueError(
        f"could not decode the {what} frame at frame {decoder.start_frame} of "
        f"{decoder.profile.path}: the decoder produced no frames"
    )


def frame_at(profile: SourceProfile, seconds: float) -> tuple[np.ndarray, int]:
    """Decode the RAW source frame nearest `seconds`.

    Raw means raw: no restoration filter, no degrain, no deinterlace. This is
    the "before" image, and the before image is what the user actually has.

    `seconds` is clamped into the source's own extent. A seek past the end
    otherwise hands ffmpeg a start time with nothing after it and returns an
    empty buffer, which used to surface far away as an IndexError.
    """
    fps = profile.fps if profile.fps > 0 else 0.0
    last_index = max(0, profile.frame_count - 1)

    if fps > 0:
        # Prefer the frame count over the container duration: the duration is
        # often set by an audio track that runs slightly past the video.
        span = last_index / fps
        if profile.duration > 0:
            span = min(span, profile.duration)
        clamped = min(max(float(seconds), 0.0), max(span, 0.0))
        index = min(int(round(clamped * fps)), last_index)
        used = index / fps
    else:
        index = 0
        used = 0.0

    frame = _first_frame(
        Decoder(profile, start_frame=index, max_frames=1), "source"
    )
    return frame, index


def _texture_for(request: RenderRequest, upscaler, settings: RestoreSettings):
    """Mirror how RenderJob builds its TexturePost, device included."""
    return TexturePost(
        detail_retention=settings.detail_retention,
        regrain=settings.regrain,
        device=str(getattr(upscaler, "device", "cpu")),
    )


def _apply_texture(texture, output: np.ndarray, source: np.ndarray, index: int) -> np.ndarray:
    """Detail retention first, re-grain second — never the other way round.

    Detail retention lifts real high frequencies out of the source frame; if
    grain were added first it would be fed back through that step and counted
    twice. render_resumable applies them in exactly this order, and a preview
    that reordered them would be previewing a different picture.
    """
    if texture is None or not getattr(texture, "enabled", False):
        return output
    output = texture.apply_detail(output, source)
    return texture.apply_grain(output, index=index)


def _compare_image(request: RenderRequest, upscaler, texture) -> ComparePair:
    """The still path: no probe, no filter graph, no frame timing."""
    rgb, _alpha = load_image(request.source)

    settings = RestoreSettings(
        deblock=request.deblock,
        degrain=request.degrain,
        detail_retention=request.detail_retention,
        regrain=request.regrain,
    )
    if texture is None:
        texture = _texture_for(request, upscaler, settings)

    after = upscaler.process(rgb)
    after = _apply_texture(texture, after, rgb, 0)

    return ComparePair(
        before=_match_size(rgb, after),
        after=np.ascontiguousarray(after),
        source_size=(rgb.shape[1], rgb.shape[0]),
        scale=int(getattr(upscaler, "scale", 1)),
        seconds=0.0,
        frame_index=0,
    )


def compare_frame(
    request: RenderRequest,
    upscaler,
    *,
    seconds: float = 0.0,
    texture=None,
) -> ComparePair:
    """Produce the before/after pair for one frame of `request.source`.

    The "after" half travels the real render path — the same scan
    classification, the same ffmpeg filter chain, the same upscaler, the same
    texture stages in the same order — so what is previewed is what will be
    rendered. Frame interpolation is the one stage that cannot apply: a single
    still has no neighbour to blend towards, so `request.target_fps` is ignored.
    """
    if is_image(request.source):
        return _compare_image(request, upscaler, texture)

    profile = SourceProfile.probe(request.source)

    # The raw original, straight off the source, at the source's own size.
    before, index = frame_at(profile, seconds)

    if request.no_restore:
        settings = RestoreSettings(
            deblock=0.0, degrain=0.0, detail_retention=0.0, regrain=0.0
        )
        video_filter = ""
    else:
        analysis = probe_scan(request.source)
        scan = classify_scan(analysis)
        settings = RestoreSettings(
            deblock=request.deblock,
            degrain=request.degrain,
            detail_retention=request.detail_retention,
            regrain=request.regrain,
        )
        video_filter = build_filter_chain(scan, analysis.field_order, settings)

    if texture is None:
        texture = _texture_for(request, upscaler, settings)

    # Decode the "after" input from the SAME start_frame as the "before" input.
    # With a deinterlacing or inverse-telecine filter active the filtered stream
    # does not have the same frame count as the source — bwdif can double it,
    # fieldmatch+decimate cuts it by 4/5 — so the two indices are not
    # interchangeable in general. They are here because both decodes seek the
    # original stream by the same input timestamp: the filter runs after the
    # seek, so both halves show the same instant of the film even though
    # counting frames in the filtered stream would give a different number.
    filtered = _first_frame(
        Decoder(profile, start_frame=index, max_frames=1, video_filter=video_filter),
        "filtered",
    )

    after = upscaler.process(filtered)
    # Detail retention is paired with the frame that was actually upscaled, as
    # in render_resumable — not with the raw "before" frame.
    after = _apply_texture(texture, after, filtered, index)

    return ComparePair(
        before=_match_size(before, after),
        after=np.ascontiguousarray(after),
        source_size=(profile.width, profile.height),
        scale=int(getattr(upscaler, "scale", 1)),
        seconds=index / profile.fps if profile.fps > 0 else 0.0,
        frame_index=index,
    )
