"""Source characterisation: scan type, grain, and compression artifacts."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

# Fraction of frames showing repeated fields above which a source is treated as
# telecined rather than interlaced. 3:2 pulldown repeats one field in five.
TELECINE_REPEAT_RATIO = 0.10

# Fraction of frames that must be detected as interlaced before deinterlacing.
INTERLACE_RATIO = 0.20

IDET_FRAMES = 400


class ScanType(Enum):
    PROGRESSIVE = "progressive"
    INTERLACED = "interlaced"
    TELECINED = "telecined"


@dataclass(frozen=True)
class FieldAnalysis:
    tff: int
    bff: int
    progressive: int
    undetermined: int
    repeated_top: int
    repeated_bottom: int

    @property
    def total(self) -> int:
        return self.tff + self.bff + self.progressive + self.undetermined

    @property
    def repeated(self) -> int:
        return self.repeated_top + self.repeated_bottom

    @property
    def field_order(self) -> str:
        return "tff" if self.tff >= self.bff else "bff"


_MULTI_RE = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
    r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)
_REPEAT_RE = re.compile(
    r"Repeated Fields:\s*Neither:\s*(\d+)\s*Top:\s*(\d+)\s*Bottom:\s*(\d+)"
)


def _parse_idet(output: str) -> FieldAnalysis:
    """Parse ffmpeg's idet filter summary.

    Multi-frame detection is used in preference to single-frame: it considers
    neighbouring frames and is markedly more reliable on real footage. The
    *last* occurrence is used rather than the first: some ffmpeg builds print
    an initial, near-empty summary during format probing before the real,
    cumulative summary at the end of processing. Taking the first match would
    silently report all-zero counts on real sources.
    """
    multi_matches = list(_MULTI_RE.finditer(output))
    if not multi_matches:
        raise ValueError("no idet output found; the probe may have failed")
    multi = multi_matches[-1]
    repeat_matches = list(_REPEAT_RE.finditer(output))
    repeat = repeat_matches[-1] if repeat_matches else None

    return FieldAnalysis(
        tff=int(multi.group(1)),
        bff=int(multi.group(2)),
        progressive=int(multi.group(3)),
        undetermined=int(multi.group(4)),
        repeated_top=int(repeat.group(2)) if repeat else 0,
        repeated_bottom=int(repeat.group(3)) if repeat else 0,
    )


def classify_scan(analysis: FieldAnalysis) -> ScanType:
    """Decide which pre-pass a source needs.

    Order matters: a film-sourced 30i DVD reports as interlaced to a naive
    check, but needs inverse telecine rather than deinterlacing. Repeated
    fields are the discriminator, since 3:2 pulldown repeats one field in five.
    """
    if analysis.total == 0:
        return ScanType.PROGRESSIVE

    if analysis.repeated / analysis.total >= TELECINE_REPEAT_RATIO:
        return ScanType.TELECINED

    interlaced = analysis.tff + analysis.bff
    if interlaced / analysis.total >= INTERLACE_RATIO:
        return ScanType.INTERLACED

    return ScanType.PROGRESSIVE


def probe_scan(path: str | Path, frames: int = IDET_FRAMES) -> FieldAnalysis:
    """Run ffmpeg's idet filter over the opening frames of a source."""
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-i", str(path),
            "-vf", "idet", "-frames:v", str(frames),
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    return _parse_idet(result.stderr)


def _luma(frame: np.ndarray) -> np.ndarray:
    """Rec. 709 luma as float32."""
    f = frame.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def estimate_grain(frame: np.ndarray) -> float:
    """Estimate grain amplitude, in 0-255 units.

    Measured as the standard deviation of a high-pass residual restricted to
    locally flat regions. Restricting to flat regions is what separates grain
    from genuine image detail and from smooth gradients, both of which would
    otherwise inflate the estimate.
    """
    y = _luma(frame)

    # 3x3 box blur via cumulative sums along both axes.
    pad = np.pad(y, 1, mode="edge")
    blur = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
        + pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
        + pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0

    residual = y - blur

    # Local gradient magnitude, used to exclude edges and textured regions.
    gy = np.abs(np.diff(blur, axis=0, prepend=blur[:1]))
    gx = np.abs(np.diff(blur, axis=1, prepend=blur[:, :1]))
    gradient = gy + gx

    flat = gradient < np.percentile(gradient, 40)
    if not flat.any():
        return float(residual.std())
    return float(residual[flat].std())


def estimate_blockiness(frame: np.ndarray) -> float:
    """Estimate DCT block-edge strength, in 0-255 units.

    Compares the mean absolute difference across 8-pixel-aligned column
    boundaries against non-aligned boundaries. Compression leaves discontinuities
    on the block grid; natural detail does not prefer that grid.
    """
    y = _luma(frame)
    diffs = np.abs(np.diff(y, axis=1))
    if diffs.shape[1] < 16:
        return 0.0

    columns = np.arange(diffs.shape[1])
    on_grid = (columns + 1) % 8 == 0
    if not on_grid.any() or on_grid.all():
        return 0.0

    return float(diffs[:, on_grid].mean() - diffs[:, ~on_grid].mean())
