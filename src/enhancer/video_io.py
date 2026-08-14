"""ffmpeg-backed source analysis and rawvideo streaming."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

INTERLACED_FIELD_ORDERS = {"tt", "bb", "tb", "bt"}


def _parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    num, _, den = value.partition("/")
    try:
        d = float(den) if den else 1.0
        return float(num) / d if d else 0.0
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class SourceProfile:
    """Everything downstream stages need to know about a source."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    pix_fmt: str
    sar: str
    interlaced: bool
    field_order: str
    color_primaries: str
    color_transfer: str
    color_space: str
    duration: float

    @classmethod
    def probe(cls, path: str | Path) -> "SourceProfile":
        path = Path(path)
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return replace(_parse_probe(json.loads(out.stdout)), path=path)


def _parse_probe(raw: dict) -> SourceProfile:
    video = next(
        (s for s in raw.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video is None:
        raise ValueError("no video stream found in source")

    fps = _parse_fraction(video.get("avg_frame_rate")) or _parse_fraction(
        video.get("r_frame_rate")
    )
    duration = float(raw.get("format", {}).get("duration") or 0.0)
    count = int(video.get("nb_frames") or 0)
    if count == 0 and fps and duration:
        count = int(round(fps * duration))

    field_order = str(video.get("field_order") or "progressive")

    return SourceProfile(
        path=Path("."),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        frame_count=count,
        pix_fmt=str(video.get("pix_fmt") or "yuv420p"),
        sar=str(video.get("sample_aspect_ratio") or "1:1"),
        interlaced=field_order in INTERLACED_FIELD_ORDERS,
        field_order=field_order,
        color_primaries=str(video.get("color_primaries") or ""),
        color_transfer=str(video.get("color_transfer") or ""),
        color_space=str(video.get("color_space") or ""),
        duration=duration,
    )
