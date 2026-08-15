"""Segmented output for resumable renders (spec §7.4)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from . import proc
from .video_io import Encoder, SourceProfile

SEGMENT_SUFFIX = ".mkv"


def segment_path(directory: str | Path, index: int) -> Path:
    """Zero-padded so lexical order matches index order."""
    return Path(directory) / f"seg_{index:05d}{SEGMENT_SUFFIX}"


def write_segment(
    path: str | Path,
    frames: Iterable[np.ndarray],
    width: int,
    height: int,
    fps: float,
    source: SourceProfile,
    codec: str = "hevc_nvenc",
    quality: int = 20,
    bit_depth: int = 10,
) -> int:
    """Encode `frames` to `path`, finalising atomically.

    Writes to a .part file and renames only on success, so an interrupted
    segment never leaves behind a file that looks complete. Returns the number
    of frames written.

    Audio and subtitles are deliberately not muxed here: `Encoder` normally
    attaches the source's audio/subtitle streams, but doing that per segment
    would attach the *entire* source audio track to every segment. Audio and
    subtitles are muxed once, from the original source, during `assemble`.
    """
    path = Path(path)
    partial = path.with_name(path.stem + ".part" + SEGMENT_SUFFIX)
    written = 0
    try:
        with Encoder(
            partial, width=width, height=height, fps=fps, source=source,
            codec=codec, quality=quality, bit_depth=bit_depth, mux_audio=False,
        ) as enc:
            for frame in frames:
                enc.write(frame)
                written += 1
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return written


def completed_segment_paths(directory: str | Path, count: int) -> list[Path]:
    """Return all `count` segment paths in index order, or raise on a gap."""
    paths = []
    for i in range(count):
        p = segment_path(directory, i)
        if not p.exists():
            raise FileNotFoundError(f"segment {i} is missing: {p}")
        paths.append(p)
    return paths


def assemble(
    directory: str | Path,
    count: int,
    output: str | Path,
    source: SourceProfile,
) -> Path:
    """Concatenate segments into the final file with a stream copy.

    No re-encode, so this takes seconds regardless of length and loses nothing.
    Audio and subtitles are muxed once here from the original source rather than
    per segment, which would accumulate sync drift at every boundary.
    """
    directory = Path(directory)
    output = Path(output)
    paths = completed_segment_paths(directory, count)

    listing = directory / "concat.txt"
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in paths),
        encoding="utf-8",
    )

    cmd = [
        "ffmpeg", "-v", "error", "-y", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-i", str(source.path),
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
        "-c", "copy",
        str(output),
    ]
    proc.run(cmd, check=True)
    return output
