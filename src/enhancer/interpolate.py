"""Frame-rate conversion driven by a flow model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .scenes import DEFAULT_THRESHOLD, is_scene_change
from .timing import OutputFrame, plan_output_frames


class Interpolator:
    """Namespace for frame-rate helpers."""

    @staticmethod
    def target_fps(src_fps: float, multiplier: float) -> float:
        if multiplier < 1:
            raise ValueError(f"multiplier must be at least 1, got {multiplier}")
        return src_fps * multiplier


def interpolate_stream(
    frames: Iterable[np.ndarray],
    src_fps: float,
    dst_fps: float,
    model,
    scene_threshold: float | None = DEFAULT_THRESHOLD,
) -> Iterator[np.ndarray]:
    """Convert a frame sequence to `dst_fps`.

    `model` is any callable taking `(frame_a, frame_b, t)` and returning the
    synthesized intermediate. Injecting it keeps this driver testable without
    network weights.

    Frames are materialized because the plan may reference a source frame after
    the stream has advanced past it. Callers process one segment at a time, so
    the working set stays bounded.
    """
    source = list(frames)
    if not source:
        return
    if len(source) == 1:
        # A single frame has no neighbour to interpolate toward; duration-based
        # planning would still want to fill more than one output slot for it.
        yield source[0]
        return

    plan = plan_output_frames(src_fps, dst_fps, len(source))
    yield from render_from_plan(source, 0, plan, model, scene_threshold)


def render_from_plan(
    frames: Iterable[np.ndarray],
    first_index: int,
    plan: Iterable[OutputFrame],
    model,
    scene_threshold: float | None = DEFAULT_THRESHOLD,
) -> Iterator[np.ndarray]:
    """Produce output frames for a pre-computed plan.

    `frames[i]` holds source frame `first_index + i`. The segmented renderer
    plans the whole job once and slices it per segment, so timesteps stay
    globally consistent no matter where a segment boundary falls — planning each
    segment independently would restart the phase and produce visible stutter at
    every boundary.
    """
    source = list(frames)

    # Cut detection is per source pair, not per output frame: the same pair
    # backs several output frames at high ratios, and the answer cannot change
    # between them.
    cuts: dict[tuple[int, int], bool] = {}

    for entry in plan:
        left = entry.left - first_index
        right = entry.right - first_index

        if entry.is_copy or left == right:
            yield source[left]
            continue

        key = (left, right)
        if key not in cuts:
            cuts[key] = (
                scene_threshold is not None
                and is_scene_change(source[left], source[right], scene_threshold)
            )

        if cuts[key]:
            # Duplicate the nearer real frame rather than morphing between shots.
            # An exact midpoint has no nearer side, so it favours the earlier frame.
            yield source[left if entry.t <= 0.5 else right]
        else:
            yield model(source[left], source[right], entry.t)
