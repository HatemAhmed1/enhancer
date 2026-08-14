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
