"""Cut detection, so interpolation never morphs across a shot change."""

from __future__ import annotations

import numpy as np

# Chosen so genuine cuts fire while fast pans and lighting changes do not.
DEFAULT_THRESHOLD = 0.30

HISTOGRAM_BINS = 32


def _channel_histograms(frame: np.ndarray) -> list[np.ndarray]:
    """Normalised intensity histogram for each channel, kept separate.

    Channels are kept apart (rather than concatenated into one vector) so the
    cumulative sum below stays a meaningful per-channel distribution instead of
    treating unrelated channel ranges as contiguous.
    """
    parts = []
    for c in range(frame.shape[2]):
        counts = np.histogram(frame[..., c], bins=HISTOGRAM_BINS, range=(0, 256))[0]
        counts = counts.astype(np.float64)
        total = counts.sum()
        parts.append(counts / total if total else counts)
    return parts


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Dissimilarity between two frames, in the range 0 to 1.

    Uses the area between cumulative colour histograms (a 1-D earth mover's
    distance) rather than per-pixel difference or raw histogram intersection.
    A camera pan moves every pixel while leaving the distribution of colours
    largely intact, so a pixel metric would flag it as a cut. A genuine shot
    change alters the distribution itself. The cumulative form also stays
    smooth for small colour shifts that land in different histogram bins,
    where a bin-intersection metric would swing straight from full overlap to
    none.
    """
    if a.shape != b.shape:
        raise ValueError(f"frames must have the same shape, got {a.shape} and {b.shape}")
    hist_a, hist_b = _channel_histograms(a), _channel_histograms(b)
    per_channel = [
        np.abs(np.cumsum(ca) - np.cumsum(cb)).sum() / HISTOGRAM_BINS
        for ca, cb in zip(hist_a, hist_b)
    ]
    return float(np.mean(per_channel))


def is_scene_change(
    a: np.ndarray, b: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> bool:
    """True when the two frames belong to different shots."""
    return frame_difference(a, b) > threshold
