"""Segmented output for resumable renders (spec §7.4)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from . import proc
from .video_io import Encoder, SourceProfile

SEGMENT_SUFFIX = ".mkv"


class OutputCollision(ValueError):
    """The chosen output path would destroy the file it reads from."""


class EmptySegment(ValueError):
    """A segment finished without a single frame in it."""


def segment_path(directory: str | Path, index: int) -> Path:
    """Zero-padded so lexical order matches index order."""
    return Path(directory) / f"seg_{index:05d}{SEGMENT_SUFFIX}"


def _normalised(path: str | Path) -> str:
    """A path in the form the filesystem itself would compare."""
    return os.path.normcase(str(Path(path).resolve()))


def same_file(a: str | Path, b: str | Path) -> bool:
    """True when two paths reach the same bytes on this filesystem.

    Comparing the strings is not enough, and ffmpeg's own guard does exactly
    that. Windows and macOS filesystems fold case, so `CASEDEMO.mp4` and
    `casedemo.mp4` are one file under two names; and a symlink or a hardlink
    reaches the same file through a name that shares nothing with the original.

    `os.path.samefile` settles all three, but only for files that already
    exist — and the output usually does not — so a normalised path comparison
    is the fallback. `normcase` is a no-op on POSIX, where the string compare
    is the right answer anyway.
    """
    a, b = Path(a), Path(b)
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return _normalised(a) == _normalised(b)


def ensure_distinct_output(source: str | Path, output: str | Path) -> None:
    """Refuse to write the result over the film it was made from.

    There is no recovering from this: the source is read as it is overwritten,
    so what is lost is lost. Checked before any work starts, and again at
    assembly, which is the moment the damage would actually be done.
    """
    if same_file(source, output):
        raise OutputCollision(
            f"the output path is the same file as the source:\n"
            f"  source: {Path(source)}\n"
            f"  output: {Path(output)}\n"
            f"Rendering there would overwrite the original with the result, and "
            f"the original could not be recovered. Two paths can be the same "
            f"file even when they look different: this filesystem ignores case, "
            f"and a shortcut or hard link reaches the same file under another "
            f"name. Choose an output path that is a genuinely different file."
        )


def write_segment(
    path: str | Path,
    frames: Iterable[np.ndarray],
    width: int,
    height: int,
    fps: float,
    source: SourceProfile,
    codec: str | None = None,
    quality: int = 20,
    bit_depth: int = 10,
) -> int:
    """Encode `frames` to `path`, finalising atomically.

    Writes to a .part file and renames only on success, so an interrupted
    segment never leaves behind a file that looks complete. Returns the number
    of frames written.

    A segment that received no frames at all is a failure, not a result: the
    encoder still emits a few hundred bytes of container header, `assemble`
    concatenates it happily, and the render reports success having quietly
    dropped that stretch of the film. Raises `EmptySegment` instead.

    `codec=None` lets `Encoder` pick whatever encoder works on this machine.

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
        if written == 0:
            raise EmptySegment(
                f"no frames were produced for {path.name}. The source supplied "
                f"nothing at this position, which usually means it is shorter "
                f"than its container claims. Try remuxing it first: "
                f"ffmpeg -i input -c copy remuxed.mkv"
            )
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
        if p.stat().st_size == 0:
            raise EmptySegment(f"segment {i} is empty: {p}")
        paths.append(p)
    return paths


def concat_line(path: str | Path) -> str:
    """One `file '...'` line for the concat demuxer, correctly escaped.

    Inside single quotes the demuxer treats everything literally until the next
    quote, so a path containing an apostrophe has to close the quote, escape
    the apostrophe, and reopen: `O'Brien` becomes `'O'\\''Brien'`. Without
    that, the demuxer silently drops the apostrophe and reports the resulting
    path as impossible to open — after the whole render has already been done.
    Every Windows account named O'Brien, D'Souza or N'Diaye has one in the
    default output path.
    """
    text = Path(path).resolve().as_posix()
    return "file '" + text.replace("'", "'\\''") + "'\n"


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

    Finalised atomically, like a segment: ffmpeg writes to a .part file beside
    the output and it is renamed into place only once the concat succeeds. In
    place, a crash here left a half-written file where a previous result used to
    be, and the existence of the output stopped meaning that assembly finished.
    The temporary keeps the real extension because ffmpeg picks its muxer from
    it.
    """
    directory = Path(directory)
    output = Path(output)
    # Last line of defence. Every caller is meant to have refused this long
    # before spending the GPU time, but this is the single point where the
    # source would actually be destroyed.
    ensure_distinct_output(source.path, output)
    paths = completed_segment_paths(directory, count)

    listing = directory / "concat.txt"
    listing.write_text("".join(concat_line(p) for p in paths), encoding="utf-8")

    partial = output.with_name(output.stem + ".part" + output.suffix)
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-i", str(source.path),
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
        "-c", "copy",
        str(partial),
    ]
    try:
        proc.run(cmd, check=True)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output
