"""Model manifest, acquisition, verification, and loading."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from spandrel import ImageModelDescriptor, ModelLoader

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024

ProgressFn = Callable[[int, int], None]

CUSTOM_SUFFIXES = {".pth", ".safetensors", ".ckpt"}


class VerificationError(RuntimeError):
    """A downloaded file did not match its expected digest."""


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file, safe for multi-gigabyte weights."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelEntry:
    id: str
    tier: str
    arch: str
    scale: int
    url: str
    sha256: str
    size: int

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        digest = str(d["sha256"]).lower()
        if not _SHA256_RE.match(digest):
            raise ValueError(
                f"model {d.get('id')!r} has a malformed sha256: {d['sha256']!r}"
            )
        return cls(
            id=str(d["id"]),
            tier=str(d["tier"]),
            arch=str(d["arch"]),
            scale=int(d["scale"]),
            url=str(d["url"]),
            sha256=digest,
            size=int(d["size"]),
        )

    def verify(self, path: Path) -> None:
        actual = sha256_file(path)
        if actual != self.sha256:
            raise VerificationError(
                f"{self.id}: expected sha256 {self.sha256}, got {actual}. "
                f"The file at {path} is corrupt or the source changed."
            )


class ModelRegistry:
    """Manifest-backed model catalogue plus a custom drop-in directory."""

    def __init__(self, manifest_path: Path, cache_dir: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        data = json.loads(self.manifest_path.read_text())
        entries = [ModelEntry.from_dict(d) for d in data["models"]]

        seen: set[str] = set()
        for e in entries:
            if e.id in seen:
                raise ValueError(f"duplicate model id in manifest: {e.id!r}")
            seen.add(e.id)

        self._entries = entries

    def list(self) -> list[ModelEntry]:
        return list(self._entries)

    def get(self, model_id: str) -> ModelEntry:
        for e in self._entries:
            if e.id == model_id:
                return e
        raise KeyError(f"unknown model id: {model_id!r}")

    def path_for(self, entry: ModelEntry) -> Path:
        return self.cache_dir / f"{entry.id}.pth"
