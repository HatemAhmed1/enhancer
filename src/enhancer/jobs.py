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


class SourceMismatch(SettingsMismatch):
    """A journal exists but was written for a different source file.

    A subclass because callers want to refuse both the same way: the journal
    describes work that has nothing to do with what is being asked for now,
    and finishing it would deliver the wrong film under the right name.
    """


def settings_hash(settings: dict) -> str:
    """Stable hash of every setting that affects output."""
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_identity(path: str | Path) -> dict:
    """What makes a source file *this* file, cheaply.

    The job directory defaults to `<output>.job`, so two renders that share an
    output name share a journal. Comparing settings alone then lets a finished
    job be handed straight back for an unrelated source: every segment is
    already marked complete, so nothing is rendered, the earlier film's
    segments are re-concatenated, and the run reports success.

    Identity here is path, size and modification time together, because none
    of the three is sufficient alone. The path misses a file replaced in place
    — download a better rip over the old one and the path is unchanged. Size
    and time miss two different films that happen to match. Content hashing
    would be definitive and is unaffordable: these sources are feature films,
    and reading twenty gigabytes to decide whether to resume would cost more
    than some of the renders. Modification time is kept to whole seconds
    because filesystems disagree below that.
    """
    p = Path(path)
    identity = {"path": os.path.normcase(str(p.resolve()))}
    try:
        stat = p.stat()
    except OSError:
        # An unreadable source is the render's problem, not the journal's; the
        # path alone still catches the common case.
        return identity
    identity["size"] = stat.st_size
    identity["mtime"] = int(stat.st_mtime)
    return identity


def source_digest(path: str | Path) -> str:
    return settings_hash(source_identity(path))


@dataclass
class JobState:
    directory: Path
    source: str
    settings_digest: str
    total_frames: int
    segment_frames: int
    source_fingerprint: str = ""
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
            source_fingerprint=source_digest(source),
        )
        job.save()
        return job

    @classmethod
    def load(cls, directory: str | Path, settings: dict, source: str | Path) -> "JobState":
        """Reopen a journal, refusing anything it does not actually describe.

        `source` is not optional: a journal that matches on settings but not on
        the film is the more dangerous of the two mismatches, because it
        finishes silently and delivers the wrong footage rather than failing.
        """
        directory = Path(directory)
        data = json.loads((directory / JOURNAL_NAME).read_text())
        expected = settings_hash(settings)
        if data["settings_digest"] != expected:
            raise SettingsMismatch(
                "This job was started with different settings. Resuming would "
                "produce a visible seam. Start a new job, or restore the "
                "original settings."
            )

        recorded = data.get("source_fingerprint") or ""
        if recorded:
            matches = recorded == source_digest(source)
        else:
            # A journal written before source identity was recorded. Its path
            # is all it has, which still catches a job directory reused for a
            # different file.
            matches = source_identity(data["source"])["path"] == \
                source_identity(source)["path"]
        if not matches:
            same_path = source_identity(data["source"])["path"] == \
                source_identity(source)["path"]
            if same_path:
                what = (
                    f"the source file has changed since this job was started:\n"
                    f"  {Path(source)}\n"
                    f"Its size or modification time is different, so it is a "
                    f"different file now — a re-download or a better rip. What "
                    f"is already rendered came from the old one, and splicing "
                    f"the two would show."
                )
            else:
                what = (
                    f"this job directory belongs to a render of a different "
                    f"source:\n"
                    f"  journal: {data['source']}\n"
                    f"  now:     {Path(source)}\n"
                    f"Resuming it would deliver that earlier film under this "
                    f"name."
                )
            raise SourceMismatch(
                f"{what}\nThe whole render has to be redone. Delete "
                f"{directory} to start fresh, or render to a different output."
            )

        return cls(
            directory=directory,
            source=data["source"],
            settings_digest=data["settings_digest"],
            total_frames=data["total_frames"],
            segment_frames=data["segment_frames"],
            source_fingerprint=recorded,
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
            "source_fingerprint": self.source_fingerprint,
            "settings_digest": self.settings_digest,
            "total_frames": self.total_frames,
            "segment_frames": self.segment_frames,
            "completed_segments": self.completed_segments,
        }
        tmp = self.journal_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.journal_path)
