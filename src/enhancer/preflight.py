"""What this machine needs before a render can run, and how to get it.

The application assumes ffmpeg is on PATH, that a model has been downloaded, and
that there is somewhere to write seventeen hours of output. On the machine it
was built on all three were true and none was ever checked. On anybody else's,
the first of them fails as a FileNotFoundError from deep inside a subprocess
call, which tells the user nothing they can act on.

So every prerequisite is checked up front and reported in the user's own terms,
with the command that fixes it on their operating system. Nothing here raises:
a failed check is a result, not an error.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import proc

# A render needs somewhere to put a feature-length 4K file. Below this there is
# no point starting: the failure would arrive hours in, with the journal half
# written and nothing to show for the electricity.
COMFORTABLE_FREE_BYTES = 20 * 1024 ** 3
TIGHT_FREE_BYTES = 5 * 1024 ** 3


@dataclass(frozen=True)
class Requirement:
    """One thing the application needs, and what to do when it is absent."""

    name: str
    ok: bool
    detail: str
    fix: str = ""
    # Essential means the application cannot render at all. Everything else
    # only removes a feature, and saying so honestly matters: telling somebody
    # with no graphics card that they cannot proceed would be a lie.
    essential: bool = True

    @property
    def blocking(self) -> bool:
        return self.essential and not self.ok


def _version_line(program: str) -> str:
    try:
        out = proc.run(
            [program, "-version"], capture_output=True, text=True, timeout=15
        )
        first = (out.stdout or out.stderr or "").splitlines()
        return first[0].strip() if first else program
    except (OSError, subprocess.SubprocessError):
        return program


def _install_hint(package: str) -> str:
    """The command that installs `package` on the running platform."""
    system = platform.system()
    if system == "Windows":
        return (
            f"Install it with:  winget install {'Gyan.FFmpeg' if package == 'ffmpeg' else package}\n"
            "Then close and reopen this application so it picks up the change.\n"
            "If winget is unavailable, download a build from ffmpeg.org and add "
            "its bin folder to PATH."
        )
    if system == "Darwin":
        return f"Install it with:  brew install {package}"
    return (
        f"Install it with your package manager, for example:\n"
        f"  sudo apt install {package}\n"
        f"  sudo dnf install {package}"
    )


def check_ffmpeg() -> Requirement:
    path = shutil.which("ffmpeg")
    if path:
        return Requirement("ffmpeg", True, _version_line("ffmpeg"))
    return Requirement(
        "ffmpeg", False,
        "Not found on PATH. Every video stage reads and writes through it.",
        _install_hint("ffmpeg"),
    )


def check_ffprobe() -> Requirement:
    path = shutil.which("ffprobe")
    if path:
        return Requirement("ffprobe", True, _version_line("ffprobe"))
    return Requirement(
        "ffprobe", False,
        "Not found on PATH. It ships with ffmpeg and is used to inspect sources.",
        _install_hint("ffmpeg"),
    )


def check_accelerator() -> Requirement:
    """A graphics card is a large speed difference, not a requirement."""
    try:
        from .system import detect

        hw = detect(probe_encoders=False)
    except Exception:  # noqa: BLE001 - detection must never block startup
        return Requirement(
            "Graphics acceleration", True,
            "Could not be determined; the processor will be used.",
            essential=False,
        )

    if hw.accelerator != "cpu":
        vram = f", {hw.vram_bytes / 1024 ** 3:.1f} GB" if hw.vram_bytes else ""
        return Requirement(
            "Graphics acceleration", True,
            f"{hw.gpu_name or hw.accelerator}{vram}", essential=False,
        )
    return Requirement(
        "Graphics acceleration", False,
        "None detected. Rendering will use the processor, which is tens of "
        "times slower — minutes of video rather than hours of film.",
        "An NVIDIA card with current drivers gives the largest gain. Apple "
        "Silicon and Intel Arc are also used when present.\n"
        "If you do have an NVIDIA card, this usually means the processor-only "
        "build of PyTorch is installed; reinstall it with CUDA support.",
        essential=False,
    )


def check_models(models_dir: Path | str) -> Requirement:
    from .models import scan_custom_dir

    directory = Path(models_dir)
    found = scan_custom_dir(directory)
    if found:
        names = ", ".join(p.stem for p in found[:3])
        more = f" and {len(found) - 3} more" if len(found) > 3 else ""
        return Requirement("Models", True, f"{len(found)} available: {names}{more}")
    return Requirement(
        "Models", False,
        f"No model files in {directory}. One is needed to enlarge anything.",
        "Download a recommended model from the Models panel, or drop any "
        "OpenModelDB .pth file into that folder — the architecture is "
        "detected automatically.",
    )


def check_disk_space(where: Path | str) -> Requirement:
    """Free space where the output will be written."""
    target = Path(where)
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        free = shutil.disk_usage(target).free
    except OSError as exc:
        return Requirement(
            "Disk space", True, f"Could not be determined: {exc}", essential=False
        )

    gigabytes = free / 1024 ** 3
    if free >= COMFORTABLE_FREE_BYTES:
        return Requirement("Disk space", True, f"{gigabytes:.0f} GB free", essential=False)
    if free >= TIGHT_FREE_BYTES:
        return Requirement(
            "Disk space", True,
            f"Only {gigabytes:.0f} GB free. Enough for a short clip, not a feature.",
            "A 4K feature can run to tens of gigabytes. Choose an output "
            "folder on a larger drive before starting a long render.",
            essential=False,
        )
    return Requirement(
        "Disk space", False,
        f"{gigabytes:.1f} GB free, which a render will exhaust.",
        "Free some space or choose an output folder on another drive. A render "
        "that fills the disk stops partway through and keeps only what it had "
        "already finished.",
        essential=False,
    )


def check_interpolation(weights_dir: Path | str | None = None) -> Requirement:
    """Frame interpolation needs its own weights; everything else works without."""
    try:
        from .cli import rife_weights

        found = rife_weights() if weights_dir is None else list(Path(weights_dir).glob("*.pkl"))
    except Exception:  # noqa: BLE001
        found = []
    if found:
        return Requirement(
            "Smoother motion", True, f"{Path(found[0]).name} available", essential=False
        )
    return Requirement(
        "Smoother motion", False,
        "RIFE weights not found. Everything else works; only frame-rate "
        "conversion is unavailable.",
        "Place a RIFE .pkl weights file in the models folder to enable it.",
        essential=False,
    )


def check_all(models_dir: Path | str = "models/custom",
              output_dir: Path | str | None = None) -> list[Requirement]:
    """Every prerequisite, essential ones first."""
    return [
        check_ffmpeg(),
        check_ffprobe(),
        check_models(models_dir),
        check_accelerator(),
        check_disk_space(output_dir or Path.home()),
        check_interpolation(),
    ]


def blocking(requirements: list[Requirement]) -> list[Requirement]:
    """Only those that stop the application working at all."""
    return [r for r in requirements if r.blocking]


def summary(requirements: list[Requirement]) -> str:
    """One line per requirement, for a terminal."""
    lines = []
    for r in requirements:
        mark = "ok  " if r.ok else ("MISSING" if r.essential else "note")
        lines.append(f"{mark:>8}  {r.name}: {r.detail}")
        if not r.ok and r.fix:
            lines.extend(f"          {line}" for line in r.fix.splitlines())
    return "\n".join(lines)
