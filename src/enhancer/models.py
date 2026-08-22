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

    def ensure(self, entry: ModelEntry, on_progress: ProgressFn | None = None) -> Path:
        """Return a verified local path, downloading if necessary."""
        dest = self.path_for(entry)
        if dest.exists():
            try:
                entry.verify(dest)
                return dest
            except VerificationError:
                dest.unlink()

        dest.parent.mkdir(parents=True, exist_ok=True)
        _fetch(entry.url, dest, on_progress)
        try:
            entry.verify(dest)
        except VerificationError:
            dest.unlink(missing_ok=True)
            raise
        return dest


def scan_custom_dir(directory: Path) -> list[Path]:
    """Find drop-in weight files (spec §4.1). Missing directory is not an error."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in CUSTOM_SUFFIXES
    )


def _fetch(url: str, dest: Path, on_progress: ProgressFn | None = None) -> None:
    """Stream a URL to disk, reporting (bytes_done, bytes_total)."""
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as fh:
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)


class LoadedModel:
    """A spandrel-loaded upscaler ready for inference."""

    def __init__(self, descriptor: ImageModelDescriptor, device: torch.device) -> None:
        self._d = descriptor
        self.scale = int(descriptor.scale)
        self.arch = descriptor.architecture.name
        self.supports_half = bool(descriptor.supports_half)
        # `device` is accepted for the caller's convenience and applied here so
        # that constructing a LoadedModel is enough; it is not remembered,
        # because the .device property reads the parameters themselves.
        descriptor.to(torch.device(device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self._d(x)

    @property
    def device(self) -> torch.device:
        """Where the weights actually are, read from the parameters.

        A property rather than a remembered constructor argument, so that it
        cannot drift out of step with reality after a to() call — the CPU
        fallback in upscale.py moves this model and must be able to trust what
        it reads back.
        """
        return next(self._d.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """The actual dtype of the model's parameters.

        This is ground truth, independent of what a caller *asked* load_model()
        for: when descriptor.supports_half is False, load_model() leaves the
        weights in float32 even if half=True was requested (spec: only cast
        when spandrel reports the architecture actually supports it).
        """
        return next(self._d.model.parameters()).dtype

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LoadedModel":
        """Move and/or cast the weights in place, returning self.

        The per-frame CPU fallback needs both halves of this: the processor has
        no real float16 kernels, so a frame that drops off the accelerator has
        to be run in float32 there and then cast back.

        `scale`, `arch` and `supports_half` are properties of the architecture
        and are unaffected; `device` and `dtype` are read from the parameters,
        so they report the new state without any bookkeeping here. Casting goes
        through descriptor.model rather than descriptor.to() deliberately:
        load_model() already decided whether half is allowed, and spandrel's
        own to() would refuse to cast *back* to a dtype it had let us start in.
        """
        if device is not None:
            self._d.to(torch.device(device))
        if dtype is not None:
            self._d.model.to(dtype=dtype)
        if device is not None or dtype is not None:
            # The layout is chosen once in load_model(); a cast can drop it, so
            # restate it and a returning model is exactly as it left.
            self._d.model.to(memory_format=torch.channels_last)
        return self


def load_model(path: Path, device: str | torch.device = "cuda", half: bool = True) -> LoadedModel:
    """Load any supported architecture from a bare state dict.

    spandrel deduces the architecture and hyperparameters, so no per-model code
    is required (spec §4.1).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model weights not found: {path}")

    device = torch.device(device)
    descriptor = ModelLoader().load_from_file(str(path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(f"{path.name} is not an image-to-image model")

    descriptor.to(device)
    if half and device.type == "cuda" and descriptor.supports_half:
        descriptor.model.half()
    descriptor.model.to(memory_format=torch.channels_last)
    descriptor.eval()
    return LoadedModel(descriptor, device)
