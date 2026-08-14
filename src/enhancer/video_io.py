"""ffmpeg-backed source analysis and rawvideo streaming."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

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


class Decoder:
    """Streams RGB24 frames from ffmpeg over a pipe. Never touches disk."""

    def __init__(self, profile: SourceProfile) -> None:
        self.profile = profile

    def frames(self) -> Iterator[np.ndarray]:
        p = self.profile
        frame_bytes = p.width * p.height * 3
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin",
            "-i", str(p.path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(p.height, p.width, 3)
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()


class Encoder:
    """Streams RGB24 frames into ffmpeg. Preserves color metadata and side streams."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        source: SourceProfile,
        codec: str = "hevc_nvenc",
        quality: int = 20,
        bit_depth: int = 10,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.source = source
        self.codec = codec
        self.quality = quality
        self.bit_depth = bit_depth
        self._proc: subprocess.Popen | None = None

    def _build_command(self) -> list[str]:
        s = self.source
        pix_fmt = "p010le" if self.bit_depth == 10 else "yuv420p"
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "-",
            # Second input carries the original audio/subtitle/chapter streams.
            "-i", str(s.path),
            "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
            "-c:a", "copy", "-c:s", "copy",
            "-c:v", self.codec, "-cq", str(self.quality),
            "-pix_fmt", pix_fmt,
        ]
        # Carry color metadata through explicitly.
        if s.color_primaries:
            cmd += ["-color_primaries", s.color_primaries]
        if s.color_transfer:
            cmd += ["-color_trc", s.color_transfer]
        if s.color_space:
            cmd += ["-colorspace", s.color_space]
        cmd.append(str(self.path))
        return cmd

    def __enter__(self) -> "Encoder":
        self._proc = subprocess.Popen(self._build_command(), stdin=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Encoder must be used as a context manager")
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def __exit__(self, *exc) -> None:
        if self._proc is None:
            return
        if self._proc.stdin:
            self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0 and exc[0] is None:
            raise RuntimeError(f"ffmpeg encoder exited with status {code}")
