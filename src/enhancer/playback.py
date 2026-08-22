"""Play the before/after comparison as moving footage, streamed from disk.

A single frame hides the two faults this project exists to avoid. Grain pulsing
only appears as grain changing between frames, and waxy skin only reads as waxy
once a face moves across it. So the comparison has to run, with the original and
the finished render advancing together under the swipe divider.

The clips are feature length, so nothing is buffered: two ffmpeg pipes are held
open and walked forward in lockstep, one frame at a time. There is no Qt here on
purpose — the engine must be testable without a display, exactly as compare.py
is. Pacing, painting and the divider belong to the window; this module only
answers "what are the two frames for output frame i".
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Imported, not copied. The before half must be enlarged the same way the still
# comparison enlarges it, and having two bicubic helpers drift apart would mean
# the still preview and the playing preview quietly judge the model by different
# rules. See compare._match_size for why the resize exists at all.
from .compare import _match_size
from .video_io import Decoder, SourceProfile

# How far the two clips' running times may disagree before they are treated as
# different material rather than the same material rounded differently.
_DURATION_SLOP = 1.0

# Frame boundaries land on exact ratios in theory and a hair under them in
# floating point (150 / 60.0 * 24.0 can come out 59.999999999999993). Nudging by
# a nanosecond of a frame before flooring keeps a legitimate boundary from
# falling back one frame; nothing real sits within 1e-9 of a boundary.
_FLOOR_EPSILON = 1e-9


@dataclass(frozen=True)
class PlaybackPair:
    """One instant of the film, before and after, at identical dimensions."""

    before: np.ndarray  # HWC uint8 RGB, bicubic-scaled up to the after size
    after: np.ndarray  # HWC uint8 RGB
    index: int  # output frame index
    seconds: float  # presentation time, on the output's clock


def source_index_for(output_index: int, output_fps: float, source_fps: float) -> int:
    """Which source frame belongs under output frame `output_index`.

    ALIGN BY TIME, NEVER BY FRAME NUMBER. The output does not have the source's
    frame count: interpolation turns 23.976 fps into 60, inverse telecine turns
    30 into 24, and a deinterlacing chain can double the count. Pairing frame i
    of one with frame i of the other looks right for the first second and then
    drifts without bound — at 23.976 to 60 the source is consumed 2.5x too fast,
    so by the end of a reel the "before" half is minutes ahead of the "after"
    half. Every frame index is therefore converted to a time on its own clock
    and back again on the other's:

        source_frame = floor(output_index / output_fps * source_fps)

    An "optimisation" back to `source_frame = output_index` will pass any test
    that only inspects the opening frames and desynchronise everything after.
    """
    if output_fps <= 0 or source_fps <= 0:
        return 0
    seconds = output_index / output_fps
    return max(0, int(math.floor(seconds * source_fps + _FLOOR_EPSILON)))


def _probe(path: Path, what: str) -> SourceProfile:
    """Probe one file, turning every failure into a ValueError.

    SourceProfile.probe raises CalledProcessError for a file ffprobe cannot read
    at all and ValueError for a container with no video stream. The caller is a
    viewer; it wants one exception type meaning "this file is not playable".
    """
    try:
        profile = SourceProfile.probe(path)
    except ValueError as exc:
        raise ValueError(f"the {what} {path} has no video stream: {exc}") from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ValueError(f"could not probe the {what} {path}: {exc}") from exc
    if profile.fps <= 0:
        raise ValueError(f"the {what} {path} reports no frame rate")
    if profile.frame_count <= 0:
        raise ValueError(f"the {what} {path} contains no frames")
    return profile


def _last_instant(profile: SourceProfile) -> float:
    """Presentation time of the final frame.

    Frame count over frame rate, not the container duration: the duration is
    routinely set by an audio track that runs past the picture, which would
    claim coverage the video does not have.
    """
    return (profile.frame_count - 1) / profile.fps


class ComparePlayer:
    """Two clips, walked forward together, aligned by time.

    Both decoders only ever move forward. When the output runs faster than the
    source — every interpolated render — the same source frame legitimately sits
    under several consecutive output frames, so it is held and repeated rather
    than decoded again. Going backwards, or to an arbitrary point, means `seek`,
    which tears both pipes down and starts two new ones.
    """

    def __init__(self, source: Path, output: Path) -> None:
        self.source = Path(source)
        self.output = Path(output)

        self._source_profile = _probe(self.source, "source")
        self._output_profile = _probe(self.output, "output")

        self._source_last = self._source_profile.frame_count - 1
        source_end = _last_instant(self._source_profile)
        output_end = _last_instant(self._output_profile)

        # A render that covers only part of its source is normal and expected:
        # previews are ten seconds of a two-hour film. So a large disagreement
        # is reported, not rejected — the overlapping range still plays, and
        # `covers` tells the window what it is actually looking at.
        if abs(source_end - output_end) <= _DURATION_SLOP:
            # Same material, timed slightly differently by the two frame rates.
            # Play all of the output and let the final source frame hold for the
            # odd trailing frame rather than truncating a matching pair.
            self._covered_end = source_end
            self._frames = self._output_profile.frame_count
        else:
            self._covered_end = min(source_end, output_end)
            reachable = int(math.floor(self._covered_end * self._output_profile.fps)) + 1
            self._frames = min(self._output_profile.frame_count, reachable)

        # Belt and braces: _probe already refuses a file with no frames, which
        # is the only way an overlap of nothing can be reached today. It stays
        # so that a future relaxation of that check cannot quietly produce a
        # player that opens successfully and then plays nothing.
        if self._covered_end < 0 or self._frames <= 0:
            raise ValueError(
                f"{self.source} and {self.output} do not overlap in time: "
                f"the source ends at {source_end:.3f}s and the output at "
                f"{output_end:.3f}s"
            )

        self._out_frames = None
        self._src_frames = None
        self._opened = False

        self._index = 0
        self._src_index = -1
        self._src_frame: np.ndarray | None = None
        self._before: np.ndarray | None = None
        self._before_shape: tuple[int, int] | None = None
        self._done = False

    # --- geometry of the timeline ----------------------------------------

    @property
    def fps(self) -> float:
        """The OUTPUT frame rate. Playback is paced by this, not the source's."""
        return self._output_profile.fps

    @property
    def frame_count(self) -> int:
        """Output frames this player will produce, clipped to the overlap."""
        return self._frames

    @property
    def duration(self) -> float:
        """Playable running time, in seconds, on the output's clock."""
        return self._frames / self._output_profile.fps

    @property
    def covers(self) -> tuple[float, float]:
        """(start, end) seconds of the SOURCE that the output accounts for.

        The start is 0.0: a rendered file carries no record of an offset into
        its source, so a preview is assumed to begin where the film does, which
        is how previews are actually produced. The end is the honest part — for
        a ten-second preview of a two-hour film this reads (0.0, 10.0), and the
        window can say so instead of implying the whole film was rendered.
        """
        return (0.0, max(0.0, self._covered_end))

    @property
    def source_index(self) -> int:
        """Source frame currently held, or -1 before anything is decoded."""
        return self._src_index

    # --- lifecycle --------------------------------------------------------

    def open(self) -> None:
        """Start both pipes at the beginning."""
        self._restart(0)

    def close(self) -> None:
        """Shut both pipes down now, without waiting for garbage collection."""
        self._release()

    def seek(self, seconds: float) -> None:
        """Restart both decoders at `seconds` on the output's clock.

        Dragging a slider calls this many times a second, so it must leave
        nothing behind: the two running ffmpeg processes are torn down before
        the two new ones start. Decoder.frames() closes its pipe and waits for
        its child in a finally block, which generator .close() triggers, so the
        teardown is deterministic rather than left to the collector.
        """
        index = int(round(seconds * self._output_profile.fps))
        self._restart(index)

    def __enter__(self) -> "ComparePlayer":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- streaming --------------------------------------------------------

    def next_pair(self) -> PlaybackPair | None:
        """The next aligned pair, or None once the stream is finished.

        None is the only end-of-stream signal: never an exception, and never a
        pair with one half missing. A viewer polling on a timer must be able to
        run off the end without special-casing anything.
        """
        if self._done:
            return None
        if self._out_frames is None:
            if self._opened:
                return None  # explicitly closed; nothing to resume
            self.open()
        if self._index >= self._frames:
            self._finish()
            return None

        after_view = next(self._out_frames, None)
        if after_view is None:
            self._finish()
            return None

        # Advance the source only as far as this output frame's time demands.
        # Interpolated output asks for the same source frame several times in a
        # row; that is a hold, not a re-decode.
        wanted = min(
            source_index_for(
                self._index, self._output_profile.fps, self._source_profile.fps
            ),
            self._source_last,
        )
        while self._src_index < wanted:
            frame = next(self._src_frames, None)
            if frame is None:
                # The source ran dry early. Half a pair is worse than no pair.
                self._finish()
                return None
            self._src_index += 1
            # frames() yields read-only views tied to the pipe read; this frame
            # is held across several output frames, so it is copied out.
            self._src_frame = np.array(frame, dtype=np.uint8, copy=True, order="C")
            self._before = None

        after = np.array(after_view, dtype=np.uint8, copy=True, order="C")
        before = self._scaled_before(after)

        pair = PlaybackPair(
            before=before,
            after=after,
            index=self._index,
            seconds=self._index / self._output_profile.fps,
        )
        self._index += 1
        return pair

    # --- internals --------------------------------------------------------

    def _scaled_before(self, after: np.ndarray) -> np.ndarray:
        """The held source frame, enlarged to the after frame's exact size.

        Cached for as long as the same source frame is held: at 24 to 60 fps
        this is 60% of frames, and repeating a 4K bicubic resize for a picture
        that has not changed is the difference between keeping up and not. The
        array is shared between those consecutive pairs and therefore handed out
        read-only, so a consumer cannot scribble on a frame another pair is
        still holding.
        """
        shape = (after.shape[0], after.shape[1])
        if self._before is None or self._before_shape != shape:
            scaled = _match_size(self._src_frame, after)
            if scaled is self._src_frame:
                # Same size already: _match_size hands the array straight back.
                # Take a copy so the read-only cache and the held source frame
                # are not the same object.
                scaled = self._src_frame.copy()
            scaled.setflags(write=False)
            self._before = scaled
            self._before_shape = shape
        return self._before

    def _restart(self, index: int) -> None:
        self._release()
        index = max(0, min(index, self._frames - 1))
        start = min(
            source_index_for(
                index, self._output_profile.fps, self._source_profile.fps
            ),
            self._source_last,
        )

        self._out_frames = Decoder(self._output_profile, start_frame=index).frames()
        self._src_frames = Decoder(self._source_profile, start_frame=start).frames()

        self._index = index
        # Nothing decoded yet: the first pull lands on `start`.
        self._src_index = start - 1
        self._src_frame = None
        self._before = None
        self._before_shape = None
        self._done = False
        self._opened = True

    def _release(self) -> None:
        """Close both generators, running Decoder.frames()'s finally block.

        A generator that was created but never advanced has not spawned ffmpeg
        yet — frames() opens the pipe on first next() — so closing one is free.
        """
        streams = (self._out_frames, self._src_frames)
        self._out_frames = None
        self._src_frames = None
        # Drop the references first: whatever one close() does, the player must
        # not be left holding a stream it believes is still live.
        for stream in streams:
            if stream is not None:
                stream.close()

    def _finish(self) -> None:
        """End of stream: mark it and let the two ffmpeg processes go."""
        self._done = True
        self._release()
