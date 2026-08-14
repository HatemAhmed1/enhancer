"""Output-frame timing for frame-rate conversion."""

from __future__ import annotations

from dataclasses import dataclass

# Positions closer than this to a source frame are treated as exact copies.
COPY_EPSILON = 1e-6


@dataclass(frozen=True)
class OutputFrame:
    """One frame of the output, located between two source frames.

    `t` is the fractional position from `left` toward `right`. A `t` of zero
    means the output frame coincides with `left` and can be copied rather than
    synthesized.
    """

    index: int
    left: int
    right: int
    t: float

    @property
    def is_copy(self) -> bool:
        return self.t <= COPY_EPSILON


def output_frame_count(src_fps: float, dst_fps: float, src_count: int) -> int:
    """Number of output frames produced by a conversion."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"frame rates must be positive, got {src_fps} and {dst_fps}")
    if src_count <= 0:
        return 0
    duration = src_count / src_fps
    return max(1, int(round(duration * dst_fps)))


def plan_output_frames(
    src_fps: float, dst_fps: float, src_count: int
) -> list[OutputFrame]:
    """Map each output frame onto a bracketing pair of source frames.

    Works for any ratio, integer or not, because each output frame is placed by
    time rather than by counting insertions.
    """
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"frame rates must be positive, got {src_fps} and {dst_fps}")
    if dst_fps < src_fps:
        raise ValueError(
            f"target frame rate {dst_fps} is below the source rate {src_fps}; "
            f"this tool interpolates but does not decimate"
        )
    if src_count <= 0:
        return []

    total = output_frame_count(src_fps, dst_fps, src_count)
    ratio = src_fps / dst_fps
    last = src_count - 1

    plan: list[OutputFrame] = []
    for k in range(total):
        position = k * ratio
        left = min(int(position), last)
        t = position - left
        if t <= COPY_EPSILON:
            # Exact source frame: no synthesis, so it brackets to itself.
            t = 0.0
            right = left
        else:
            right = min(left + 1, last)
        plan.append(OutputFrame(index=k, left=left, right=right, t=t))
    return plan
