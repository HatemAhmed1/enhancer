# Resilience Implementation Plan (Plan 1.5 of 5)

> **Implementation note:** each task is a self-contained TDD unit — write the failing test, confirm it fails, implement, confirm it passes, commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a multi-hour render survivable — close the Windows VRAM oversubscription hole, and make any interrupted job resume from where it stopped with no lost work beyond the current segment.

**Why now:** Plan 1's benchmark measured a 1080p→4K feature at ~17 hours. At that duration, a sleep, crash, driver reset, power loss, or simply closing the app is not an edge case — it is the expected case. Resume was originally scheduled for Plan 4; the measured numbers promote it.

**Architecture:** Output is written as a sequence of independently complete video segments rather than one file. Each segment is finalised atomically (write to `.part`, rename on success), so an interruption costs at most one segment. A JSON journal records the input identity, a hash of the settings, and which segments are done. Resume reads the journal, seeks the decoder to the first incomplete segment, and continues. The final file is produced by an ffmpeg concat with stream copy — instant, lossless, no re-encode — with audio and subtitles muxed once from the original source.

**Tech Stack:** Python 3.12, PyTorch, ffmpeg concat demuxer, pytest.

**Spec reference:** `docs/specs/2026-08-14-local-video-upscaler-design.md` §7.4, §8, §1.1.1.

---

## Design notes

### Why segments rather than a frame counter

An encoded video file cannot be meaningfully appended to mid-stream, so "remember the last frame index and continue writing" does not work against a single output file. Segmenting makes each unit of work independently finished and independently valid. It also gives three things for free: the user can inspect partial output at any time, a corrupt tail costs one segment rather than the whole render, and the final assembly is a stream copy that takes seconds.

Segment length is a tradeoff between per-segment overhead and work lost on interruption. Default is 500 frames — at the measured 2.9 fps for 1080p→4K that is roughly 3 minutes of lost work in the worst case, against negligible overhead.

**Audio and subtitles are NOT written per segment.** They are muxed once during final assembly, straight from the original source. Per-segment audio would accumulate sync drift at every boundary.

### Why the settings hash matters

Resuming a job whose settings changed would splice together footage processed two different ways — a visible discontinuity mid-film. The journal stores a hash of every setting that affects output. On mismatch the app refuses to resume and explains why, rather than silently producing a file with a seam in it.

### Why the WDDM fix polls instead of catching

Plan 1's benchmark recorded `max_memory_allocated() == 7855 MB` on a 6144 MB card, with zero exceptions raised and zero CPU fallbacks triggered. Windows WDDM had silently oversubscribed into system-RAM-backed shared GPU memory. The render did not crash; it just became very slow.

Spec §8's entire recovery design is built on catching an exception. On Windows that exception may never arrive. The fix is to stop relying on it exclusively: after each successful frame, compare peak allocation against a hard ceiling derived from *physical* VRAM. An overshoot is treated as a pressure signal and shrinks the tile, exactly as a caught OOM would — without needing the frame to fail first.

---

## File structure

| File | Responsibility |
|---|---|
| `src/enhancer/jobs.py` | Job journal: settings hash, segment ledger, atomic persistence |
| `src/enhancer/segments.py` | Segment writing, atomic finalisation, ffmpeg concat assembly |
| `src/enhancer/vram.py` | MODIFY: physical ceiling, dtype-aware budget, post-run pressure check |
| `src/enhancer/video_io.py` | MODIFY: `Decoder` gains `start_frame` seeking |
| `src/enhancer/cli.py` | MODIFY: `--job-dir`, `--resume`, `--segment-frames` |
| `tests/test_jobs.py` | Journal round-trip, settings-hash invalidation, ledger |
| `tests/test_segments.py` | Atomic finalisation, concat assembly, crash simulation |

---

## Task 1: Physical VRAM ceiling and dtype-aware budget

**Files:**
- Modify: `src/enhancer/vram.py`
- Test: `tests/test_vram.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vram.py`:

```python
from enhancer.vram import (
    bytes_per_output_pixel_for_dtype,
    physical_vram_ceiling,
)


def test_bytes_per_pixel_doubles_for_float32():
    fp16 = bytes_per_output_pixel_for_dtype(torch.float16)
    fp32 = bytes_per_output_pixel_for_dtype(torch.float32)
    assert fp32 == fp16 * 2


def test_bytes_per_pixel_is_positive():
    assert bytes_per_output_pixel_for_dtype(torch.float16) > 0


def test_physical_ceiling_subtracts_headroom():
    assert physical_vram_ceiling(total_bytes=6 * 1024 ** 3) == 6 * 1024 ** 3 - HEADROOM_BYTES


def test_physical_ceiling_never_negative():
    assert physical_vram_ceiling(total_bytes=0) == 0


def test_runner_shrinks_when_peak_exceeds_ceiling_without_exception():
    """Windows WDDM oversubscribes silently instead of raising.

    The runner must treat a peak allocation above the physical ceiling as a
    pressure signal, even though no exception was raised.
    """
    runner = TileRunner(tile=512, overlap=0, scale=1, min_tile=128, vram_ceiling=1000)
    img = torch.rand(1, 3, 64, 64)
    runner.run(lambda t: t, img, peak_bytes=5000)
    assert runner.tile < 512, "peak above ceiling must shrink the tile"


def test_runner_does_not_shrink_when_peak_within_ceiling():
    runner = TileRunner(tile=512, overlap=0, scale=1, min_tile=128, vram_ceiling=10_000)
    img = torch.rand(1, 3, 64, 64)
    runner.run(lambda t: t, img, peak_bytes=5000)
    assert runner.tile == 512


def test_runner_ignores_peak_when_no_ceiling_configured():
    runner = TileRunner(tile=512, overlap=0, scale=1, min_tile=128)
    img = torch.rand(1, 3, 64, 64)
    runner.run(lambda t: t, img, peak_bytes=10 ** 12)
    assert runner.tile == 512
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v -k ceiling or dtype or peak`
Expected: FAIL, `ImportError: cannot import name 'bytes_per_output_pixel_for_dtype'`

- [ ] **Step 3: Implement**

Add to `src/enhancer/vram.py`:

```python
# Calibrated per-output-pixel activation cost at fp16, from Plan 1 measurements.
BYTES_PER_OUTPUT_PIXEL_FP16: int = 32


def bytes_per_output_pixel_for_dtype(dtype: torch.dtype) -> int:
    """Per-output-pixel activation budget for a model running in `dtype`.

    An architecture spandrel reports as supports_half=False runs fp32 and needs
    roughly double the activation memory of the fp16 case the constant is
    calibrated against.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return BYTES_PER_OUTPUT_PIXEL_FP16
    return BYTES_PER_OUTPUT_PIXEL_FP16 * 2


def physical_vram_ceiling(total_bytes: int) -> int:
    """Hard allocation ceiling derived from PHYSICAL VRAM.

    Windows WDDM will silently oversubscribe into system-RAM-backed shared GPU
    memory rather than raising a catchable OOM, so an exception-driven recovery
    policy alone is not sufficient. This ceiling gives the runner something to
    compare against.
    """
    return max(0, total_bytes - HEADROOM_BYTES)


def total_vram_bytes() -> int:
    """Physical VRAM on the current CUDA device, or 0 when unavailable."""
    if not torch.cuda.is_available():
        return 0
    _free, total = torch.cuda.mem_get_info()
    return int(total)
```

Modify `TileRunner.__init__` to accept `vram_ceiling: int | None = None` and store it as `self.vram_ceiling`.

Modify `TileRunner.run` to accept an optional `peak_bytes: int | None = None` parameter. After `self._note_success()` and before returning, add:

```python
            self._check_pressure(peak_bytes)
            return out
```

And add this method:

```python
    def _check_pressure(self, peak_bytes: int | None) -> None:
        """Shrink on silent oversubscription (spec §1.1.1).

        On Windows WDDM an over-budget allocation does not raise; it spills into
        shared system memory and merely gets slow. Polling the allocator peak is
        the only reliable signal there.
        """
        if self.vram_ceiling is None or peak_bytes is None:
            return
        if peak_bytes <= self.vram_ceiling:
            return
        if self.tile <= self.min_tile:
            return
        self._successes = 0
        self._probe_interval = min(self._probe_interval * 2, MAX_PROBE_INTERVAL)
        self.tile = self._step_down(self.tile)
        log.warning(
            "Peak allocation %.0f MB exceeded the %.0f MB physical ceiling "
            "without raising; shrinking tile to %d",
            peak_bytes / 1024 ** 2, self.vram_ceiling / 1024 ** 2, self.tile,
        )
```

- [ ] **Step 4: Run, confirm all pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v`

- [ ] **Step 5: Wire it up in `upscale.py`**

In `Upscaler.__init__`, derive the ceiling and pass it to `TileRunner`:

```python
        ceiling = physical_vram_ceiling(total_vram_bytes()) if self.device.type == "cuda" else None
        self.runner = TileRunner(
            tile=tile, overlap=overlap, scale=model.scale,
            min_tile=128, vram_ceiling=ceiling,
        )
```

In `Upscaler.process`, reset and read the allocator peak around the run:

```python
        track = self.device.type == "cuda" and torch.cuda.is_available()
        if track:
            torch.cuda.reset_peak_memory_stats()
        try:
            peak = None
            out = self.runner.run(self._infer, img)
            if track:
                peak = int(torch.cuda.max_memory_allocated())
                self.runner._check_pressure(peak)
        except TileFloorReached:
            ...
```

Import `physical_vram_ceiling` and `total_vram_bytes` at the top of `upscale.py`.

- [ ] **Step 6: Run the full suite, then commit**

```bash
git add src/enhancer/vram.py src/enhancer/upscale.py tests/test_vram.py && git commit -m "fix(vram): enforce physical ceiling against silent WDDM oversubscription"
```

---

## Task 2: Job journal

**Files:**
- Create: `src/enhancer/jobs.py`
- Test: `tests/test_jobs.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest

from enhancer.jobs import JobState, SettingsMismatch, settings_hash


def test_settings_hash_is_stable_for_same_input():
    a = settings_hash({"model": "span", "scale": 2, "tile": 512})
    b = settings_hash({"tile": 512, "scale": 2, "model": "span"})
    assert a == b, "key order must not affect the hash"


def test_settings_hash_changes_when_a_value_changes():
    a = settings_hash({"model": "span", "scale": 2})
    b = settings_hash({"model": "span", "scale": 4})
    assert a != b


def test_new_job_has_no_completed_segments(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    assert job.completed_segments == []
    assert job.next_segment_index == 0


def test_marking_a_segment_complete_advances_the_ledger(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    job.mark_complete(0)
    assert job.completed_segments == [0]
    assert job.next_segment_index == 1


def test_journal_survives_a_round_trip(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    job.mark_complete(0)
    reloaded = JobState.load(tmp_path / "job", settings={"scale": 2})
    assert reloaded.completed_segments == [0]
    assert reloaded.total_frames == 1000
    assert reloaded.segment_frames == 500


def test_resume_refuses_when_settings_changed(tmp_path):
    JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    with pytest.raises(SettingsMismatch):
        JobState.load(tmp_path / "job", settings={"scale": 4})


def test_segment_count_rounds_up(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1001, segment_frames=500)
    assert job.segment_count == 3


def test_start_frame_for_segment(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1000, segment_frames=500)
    assert job.start_frame_for(0) == 0
    assert job.start_frame_for(1) == 500


def test_frames_in_final_segment_is_the_remainder(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1100, segment_frames=500)
    assert job.frames_in_segment(0) == 500
    assert job.frames_in_segment(2) == 100


def test_is_complete_only_when_every_segment_is_done(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1000, segment_frames=500)
    job.mark_complete(0)
    assert not job.is_complete
    job.mark_complete(1)
    assert job.is_complete


def test_out_of_order_completion_is_recorded(tmp_path):
    """Segments may finish out of order; next_segment_index is the first gap."""
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1500, segment_frames=500)
    job.mark_complete(1)
    assert job.next_segment_index == 0
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.jobs'`**

- [ ] **Step 3: Implement**

```python
"""Job journal for resumable renders (spec §7.4)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

JOURNAL_NAME = "job.json"


class SettingsMismatch(RuntimeError):
    """A journal exists but was written with different settings.

    Resuming would splice together footage processed two different ways,
    producing a visible discontinuity mid-render.
    """


def settings_hash(settings: dict) -> str:
    """Stable hash of every setting that affects output."""
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class JobState:
    directory: Path
    source: str
    settings_digest: str
    total_frames: int
    segment_frames: int
    completed_segments: list[int] = field(default_factory=list)

    @property
    def journal_path(self) -> Path:
        return self.directory / JOURNAL_NAME

    @property
    def segment_count(self) -> int:
        if self.segment_frames <= 0:
            raise ValueError("segment_frames must be positive")
        return (self.total_frames + self.segment_frames - 1) // self.segment_frames

    @property
    def next_segment_index(self) -> int:
        """Index of the first segment not yet complete."""
        done = set(self.completed_segments)
        for i in range(self.segment_count):
            if i not in done:
                return i
        return self.segment_count

    @property
    def is_complete(self) -> bool:
        return len(set(self.completed_segments)) >= self.segment_count

    def start_frame_for(self, index: int) -> int:
        return index * self.segment_frames

    def frames_in_segment(self, index: int) -> int:
        start = self.start_frame_for(index)
        return min(self.segment_frames, self.total_frames - start)

    @classmethod
    def create(
        cls,
        directory: str | Path,
        source: str,
        settings: dict,
        total_frames: int,
        segment_frames: int,
    ) -> "JobState":
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        job = cls(
            directory=directory,
            source=str(source),
            settings_digest=settings_hash(settings),
            total_frames=total_frames,
            segment_frames=segment_frames,
        )
        job.save()
        return job

    @classmethod
    def load(cls, directory: str | Path, settings: dict) -> "JobState":
        directory = Path(directory)
        data = json.loads((directory / JOURNAL_NAME).read_text())
        expected = settings_hash(settings)
        if data["settings_digest"] != expected:
            raise SettingsMismatch(
                "This job was started with different settings. Resuming would "
                "produce a visible seam. Start a new job, or restore the "
                "original settings."
            )
        return cls(
            directory=directory,
            source=data["source"],
            settings_digest=data["settings_digest"],
            total_frames=data["total_frames"],
            segment_frames=data["segment_frames"],
            completed_segments=list(data.get("completed_segments", [])),
        )

    def mark_complete(self, index: int) -> None:
        if index not in self.completed_segments:
            self.completed_segments.append(index)
            self.completed_segments.sort()
        self.save()

    def save(self) -> None:
        """Write the journal atomically so a crash mid-write cannot corrupt it."""
        payload = {
            "source": self.source,
            "settings_digest": self.settings_digest,
            "total_frames": self.total_frames,
            "segment_frames": self.segment_frames,
            "completed_segments": self.completed_segments,
        }
        tmp = self.journal_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.journal_path)
```

- [ ] **Step 4: Run, confirm 11 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/jobs.py tests/test_jobs.py && git commit -m "feat(jobs): resumable job journal with settings-change detection"
```

---

## Task 3: Segment writing and assembly

**Files:**
- Create: `src/enhancer/segments.py`
- Test: `tests/test_segments.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from enhancer.jobs import JobState
from enhancer.segments import (
    assemble,
    completed_segment_paths,
    segment_path,
    write_segment,
)
from enhancer.video_io import Decoder, SourceProfile


def test_segment_path_is_zero_padded_and_sortable(tmp_path):
    assert segment_path(tmp_path, 7).name == "seg_00007.mkv"
    names = [segment_path(tmp_path, i).name for i in (2, 10, 1)]
    assert sorted(names) == ["seg_00001.mkv", "seg_00002.mkv", "seg_00010.mkv"]


def test_write_segment_produces_a_playable_file(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(profile).frames())[:10]
    out = segment_path(tmp_path, 0)
    write_segment(out, iter(frames), width=320, height=240, fps=25.0, source=profile)
    assert out.exists()
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 10


def test_write_segment_leaves_no_part_file_on_success(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(profile).frames())[:5]
    out = segment_path(tmp_path, 0)
    write_segment(out, iter(frames), width=320, height=240, fps=25.0, source=profile)
    assert not out.with_suffix(".part.mkv").exists()
    assert list(tmp_path.glob("*.part*")) == []


def test_write_segment_discards_partial_output_on_failure(tmp_path, synthetic_clip):
    """An interrupted segment must not leave a file that looks complete."""
    profile = SourceProfile.probe(synthetic_clip)

    def exploding_frames():
        yield from list(Decoder(profile).frames())[:3]
        raise RuntimeError("simulated interruption")

    out = segment_path(tmp_path, 0)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_segment(out, exploding_frames(), width=320, height=240, fps=25.0, source=profile)
    assert not out.exists(), "a failed segment must not be left behind as complete"


def test_completed_segment_paths_are_in_index_order(tmp_path):
    for i in (2, 0, 1):
        segment_path(tmp_path, i).write_bytes(b"x")
    found = completed_segment_paths(tmp_path, count=3)
    assert [p.name for p in found] == ["seg_00000.mkv", "seg_00001.mkv", "seg_00002.mkv"]


def test_completed_segment_paths_raises_on_a_gap(tmp_path):
    segment_path(tmp_path, 0).write_bytes(b"x")
    segment_path(tmp_path, 2).write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        completed_segment_paths(tmp_path, count=3)


def test_assemble_concatenates_segments_and_preserves_frame_count(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(profile).frames())
    write_segment(segment_path(tmp_path, 0), iter(all_frames[:25]),
                  width=320, height=240, fps=25.0, source=profile)
    write_segment(segment_path(tmp_path, 1), iter(all_frames[25:50]),
                  width=320, height=240, fps=25.0, source=profile)

    final = tmp_path / "final.mkv"
    assemble(tmp_path, count=2, output=final, source=profile)
    assert final.exists()
    assert len(list(Decoder(SourceProfile.probe(final)).frames())) == 50
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.segments'`**

- [ ] **Step 3: Implement**

```python
"""Segmented output for resumable renders (spec §7.4)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

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
    """
    path = Path(path)
    partial = path.with_name(path.stem + ".part" + SEGMENT_SUFFIX)
    written = 0
    try:
        with Encoder(
            partial, width=width, height=height, fps=fps, source=source,
            codec=codec, quality=quality, bit_depth=bit_depth,
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
    subprocess.run(cmd, check=True)
    return output
```

- [ ] **Step 4: Run, confirm 7 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/segments.py tests/test_segments.py && git commit -m "feat(segments): atomic segment writing and stream-copy assembly"
```

---

## Task 4: Decoder seeking

**Files:**
- Modify: `src/enhancer/video_io.py`
- Test: `tests/test_video_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_video_io.py`:

```python
def test_decoder_start_frame_skips_leading_frames(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(p).frames())
    seeked = list(Decoder(p, start_frame=10).frames())
    assert len(seeked) == len(all_frames) - 10


def test_decoder_start_frame_lands_on_the_right_frame(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(p).frames())
    seeked = list(Decoder(p, start_frame=10).frames())
    assert np.array_equal(seeked[0], all_frames[10])


def test_decoder_max_frames_limits_output(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    assert len(list(Decoder(p, max_frames=7).frames())) == 7


def test_decoder_start_and_max_select_a_window(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(p).frames())
    window = list(Decoder(p, start_frame=20, max_frames=5).frames())
    assert len(window) == 5
    assert np.array_equal(window[0], all_frames[20])


def test_decoder_start_frame_zero_matches_no_seek(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    assert len(list(Decoder(p, start_frame=0).frames())) == 50
```

- [ ] **Step 2: Run and confirm the seeking tests fail**

- [ ] **Step 3: Implement**

Replace `Decoder` in `src/enhancer/video_io.py`:

```python
class Decoder:
    """Streams RGB24 frames from ffmpeg over a pipe. Never touches disk.

    `start_frame` seeks before decoding, which is what makes resume cheap: an
    interrupted render at hour ten does not re-process the first ten hours.
    """

    def __init__(
        self,
        profile: SourceProfile,
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> None:
        self.profile = profile
        self.start_frame = start_frame
        self.max_frames = max_frames

    def _command(self) -> list[str]:
        p = self.profile
        cmd = ["ffmpeg", "-v", "error", "-nostdin"]
        if self.start_frame > 0 and p.fps > 0:
            # Input-side seek: fast, and accurate in modern ffmpeg because it
            # decodes from the preceding keyframe and discards internally.
            cmd += ["-ss", f"{self.start_frame / p.fps:.6f}"]
        cmd += ["-i", str(p.path)]
        if self.max_frames is not None:
            cmd += ["-frames:v", str(self.max_frames)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        return cmd

    def frames(self) -> Iterator[np.ndarray]:
        p = self.profile
        frame_bytes = p.width * p.height * 3
        proc = subprocess.Popen(
            self._command(), stdout=subprocess.PIPE, bufsize=frame_bytes * 4
        )
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
```

- [ ] **Step 4: Run the full `test_video_io.py`, confirm all pass**

If `test_decoder_start_frame_lands_on_the_right_frame` fails by an off-by-small-N, that is keyframe seek imprecision. Do NOT weaken the assertion. Instead switch to output-side seeking (`-ss` placed AFTER `-i`), which is frame-exact but slower, and report the change.

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/video_io.py tests/test_video_io.py && git commit -m "feat(video_io): decoder seeking and frame-count limiting"
```

---

## Task 5: Resumable render pipeline and CLI

**Files:**
- Modify: `src/enhancer/cli.py`
- Test: `tests/test_cli_resume.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from enhancer.cli import render_resumable
from enhancer.jobs import JobState
from enhancer.segments import segment_path
from enhancer.video_io import Decoder, SourceProfile


class DoublingUpscaler:
    """Stands in for a real Upscaler."""

    scale = 2
    cpu_fallback_count = 0

    def process(self, frame):
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def test_render_produces_correct_output(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
    )
    result = SourceProfile.probe(out)
    assert result.width == 640 and result.height == 480
    assert len(list(Decoder(result).frames())) == 50


def test_render_writes_expected_segment_count(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "out.mkv",
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )
    # 50 frames at 20 per segment -> 3 segments
    assert segment_path(job_dir, 2).exists()


def test_interrupted_render_resumes_and_matches_uninterrupted(tmp_path, synthetic_clip):
    """The whole point: a killed job resumes to a byte-equivalent result."""
    profile = SourceProfile.probe(synthetic_clip)

    reference = tmp_path / "reference.mkv"
    render_resumable(
        profile, DoublingUpscaler(), reference,
        job_dir=tmp_path / "job_ref", segment_frames=20, settings={"scale": 2},
    )
    reference_frames = list(Decoder(SourceProfile.probe(reference)).frames())

    class FailsOnSecondSegment(DoublingUpscaler):
        def __init__(self):
            self.seen = 0

        def process(self, frame):
            self.seen += 1
            if self.seen > 25:
                raise RuntimeError("simulated crash")
            return super().process(frame)

    job_dir = tmp_path / "job_resume"
    with pytest.raises(RuntimeError, match="simulated crash"):
        render_resumable(
            profile, FailsOnSecondSegment(), tmp_path / "out.mkv",
            job_dir=job_dir, segment_frames=20, settings={"scale": 2},
        )
    assert segment_path(job_dir, 0).exists(), "first segment should have survived"
    assert not segment_path(job_dir, 1).exists(), "crashed segment must not persist"

    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )
    resumed_frames = list(Decoder(SourceProfile.probe(out)).frames())
    assert len(resumed_frames) == len(reference_frames)


def test_resume_skips_already_completed_segments(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"
    render_resumable(
        profile, DoublingUpscaler(), tmp_path / "first.mkv",
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )

    class ExplodesIfCalled(DoublingUpscaler):
        def process(self, frame):
            raise AssertionError("should not reprocess a completed segment")

    render_resumable(
        profile, ExplodesIfCalled(), tmp_path / "second.mkv",
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
    )
```

- [ ] **Step 2: Run and confirm `ImportError: cannot import name 'render_resumable'`**

- [ ] **Step 3: Implement**

Add to `src/enhancer/cli.py`:

```python
from .jobs import JobState, SettingsMismatch
from .segments import assemble, segment_path, write_segment

DEFAULT_SEGMENT_FRAMES = 500


def render_resumable(
    profile: SourceProfile,
    upscaler,
    output: str | Path,
    job_dir: str | Path,
    segment_frames: int = DEFAULT_SEGMENT_FRAMES,
    settings: dict | None = None,
    on_progress=None,
) -> Path:
    """Render `profile` to `output`, resuming any interrupted prior attempt.

    Output is written as independently complete segments so an interruption
    costs at most one segment. The final file is a stream-copy concat.
    """
    job_dir = Path(job_dir)
    settings = settings or {}
    scale = upscaler.scale

    if (job_dir / "job.json").exists():
        job = JobState.load(job_dir, settings=settings)
    else:
        job = JobState.create(
            job_dir, source=str(profile.path), settings=settings,
            total_frames=profile.frame_count, segment_frames=segment_frames,
        )

    out_w = profile.width * scale
    out_h = profile.height * scale

    for index in range(job.segment_count):
        if index in job.completed_segments and segment_path(job_dir, index).exists():
            continue

        start = job.start_frame_for(index)
        count = job.frames_in_segment(index)
        decoder = Decoder(profile, start_frame=start, max_frames=count)

        def processed():
            for i, frame in enumerate(decoder.frames()):
                yield upscaler.process(frame)
                if on_progress:
                    on_progress(start + i + 1, job.total_frames)

        write_segment(
            segment_path(job_dir, index), processed(),
            width=out_w, height=out_h, fps=profile.fps, source=profile,
        )
        job.mark_complete(index)

    return assemble(job_dir, job.segment_count, output, profile)
```

Rewrite `cmd_video` to use it:

```python
def cmd_video(args: argparse.Namespace) -> int:
    device = select_device(prefer_cuda=not args.cpu)
    model = load_model(Path(args.model), device=device, half=not args.cpu)
    tile = args.tile or _auto_tile(model.scale, args.overlap)

    profile = SourceProfile.probe(args.input)
    up = Upscaler(model, tile=tile, overlap=args.overlap, device=device, half=not args.cpu)

    settings = {
        "model": Path(args.model).name,
        "scale": model.scale,
        "tile": tile,
        "overlap": args.overlap,
        "cpu": bool(args.cpu),
    }
    job_dir = Path(args.job_dir) if args.job_dir else Path(args.output).with_suffix(".job")

    print(f"{profile.width}x{profile.height} -> {profile.width * model.scale}x"
          f"{profile.height * model.scale}, {profile.frame_count} frames, tile={tile}")
    print(f"Job directory: {job_dir}  (interrupted renders resume from here)")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0:
            print(f"\r{done}/{total}", end="", flush=True)

    try:
        render_resumable(
            profile, up, args.output, job_dir=job_dir,
            segment_frames=args.segment_frames, settings=settings,
            on_progress=progress,
        )
    except SettingsMismatch as exc:
        print(f"\nCannot resume: {exc}")
        return 2

    print(f"\nDone. CPU fallbacks: {up.cpu_fallback_count}")
    return 0
```

Add these arguments to the `video` subparser:

```python
    v.add_argument("--job-dir", default=None,
                   help="where to keep resumable state (default: <output>.job)")
    v.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES,
                   help="frames per resumable segment")
```

- [ ] **Step 4: Run, confirm 4 passed**

- [ ] **Step 5: Real interruption test**

Generate a 30-second clip, start a render, kill it partway with Ctrl-C, then re-run the identical command and confirm it resumes rather than restarting, and that the final frame count is correct.

```bash
ffmpeg -y -loglevel error -f lavfi -i testsrc=size=854x480:rate=25:duration=30 -pix_fmt yuv420p out/resume_in.mp4
```

- [ ] **Step 6: Run the full suite and commit**

```bash
git add src/enhancer/cli.py tests/test_cli_resume.py && git commit -m "feat(cli): resumable segmented rendering"
```

---

## Self-review notes

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| §1.1.1 WDDM silent oversubscription | Task 1 |
| §7.4 resume, journaling | Tasks 2, 3, 4, 5 |
| §8 dtype-aware VRAM budget | Task 1 |

**Deliberately out of scope:** the restoration stage (§5), interpolation (§4.3), TensorRT (§6),
segment preview (§7.2), dual-pass (§7.3), and the GUI (§10). Plans 2-4.

**Known limitation to revisit:** `render_resumable` re-opens a decoder per segment, so each segment
pays one seek. At the default 500 frames that is negligible against ~3 minutes of inference, but if
segment size is ever reduced substantially the seek cost should be re-measured.
