"""What this machine is, and which video encoder actually works on it.

The app used to hardcode `hevc_nvenc` with no fallback, so it ran only on an
NVIDIA machine and died at the encode step everywhere else. Nothing here asks
the user a question: the machine is inspected and a working encoder is chosen.

No Qt, no hard dependency on torch, and `detect()` never raises. A machine we
cannot read is a machine that gets conservative defaults, not a crash.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass

from . import proc

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hardware:
    """A single snapshot of the machine, as far as we can read it."""

    os_name: str
    machine: str
    cpu_name: str
    cpu_cores: int
    ram_bytes: int
    accelerator: str
    gpu_name: str
    vram_bytes: int
    encoder: str
    encoder_is_hardware: bool


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------

# Best first. `libx265` is the floor: always compiled in, always works, just
# slower than any of the hardware encoders above it.
CANDIDATE_ENCODERS: tuple[str, ...] = (
    "hevc_nvenc",
    "hevc_qsv",
    "hevc_amf",
    "hevc_videotoolbox",
    "libx265",
    "libx264",
)

FALLBACK_ENCODER = "libx265"

HARDWARE_ENCODERS = frozenset(
    {"hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_videotoolbox", "hevc_vaapi"}
)

# NVENC's HEVC encoder refuses anything at or below 128 in either dimension:
# "Frame dimensions are less than the minimum supported value". Measured on an
# RTX 3060 Laptop, 128x128 fails and 129x129 succeeds. A 64x64 probe therefore
# reports NVENC as broken on a machine where it is the best encoder available,
# which is the exact failure this module exists to prevent.
PROBE_SIZE = 256
PROBE_TIMEOUT_SECONDS = 30.0


def _quality_args(encoder: str, quality: int) -> list[str]:
    if encoder.endswith("_nvenc"):
        return ["-cq", str(quality)]
    if encoder.endswith("_qsv"):
        return ["-global_quality", str(quality)]
    if encoder.endswith("_amf"):
        # The qp values are only honoured in constant-QP rate control; the
        # default mode is a variable bitrate that ignores them.
        return ["-rc", "cqp", "-qp_i", str(quality), "-qp_p", str(quality)]
    if encoder.endswith("_vaapi"):
        return ["-qp", str(quality)]
    if encoder.endswith("_videotoolbox"):
        return ["-q:v", str(_videotoolbox_quality(quality))]
    # libx264, libx265 and every other software encoder worth using.
    return ["-crf", str(quality)]


def _videotoolbox_quality(quality: int) -> int:
    """Map a CRF-style value onto videotoolbox's inverted 1..100 scale.

    Every other encoder here takes "lower is better". videotoolbox takes
    "higher is better", so passing a CRF number straight through would ask for
    the worst quality it can produce whenever the user asked for the best.
    """
    scaled = round(100.0 * (1.0 - quality / 51.0))
    return max(1, min(100, scaled))


def quality_args(encoder: str, quality: int) -> list[str]:
    """The constant-quality arguments `encoder` actually reads.

    A wrong flag here does not fail loudly. ffmpeg logs "Codec AVOption cq ...
    has not been used for any stream" at warning level, exits 0, and encodes at
    the encoder's own default quality. Measured: `-cq` changes NVENC's output
    size by 4.4x across the quality range and does nothing at all to libx265,
    where only `-crf` moves it.
    """
    return _quality_args(encoder, quality)


def supports_10bit(encoder: str) -> bool:
    """Whether `encoder` can be trusted to produce a 10-bit stream.

    Anything H.264 is treated as 8-bit only. `h264_nvenc`, `h264_qsv` and
    `h264_amf` genuinely cannot do 10-bit, and while the libx264 in this
    particular ffmpeg build does advertise `yuv420p10le` and produces a High 10
    profile, most builds are 8-bit only and there is no cheap way to tell them
    apart. libx264 is the last-resort floor anyway.
    """
    name = encoder.lower()
    return "264" not in name and "avc" not in name


def pixel_format(encoder: str, bit_depth: int) -> str:
    """The pixel format to hand `encoder` for a given bit depth.

    Hardware encoders take the semi-planar `p010le`; libx265 takes planar
    `yuv420p10le` and does not list `p010le` at all. ffmpeg will insert a
    conversion either way, but naming the format the encoder actually wants
    keeps that conversion out of the filter graph.
    """
    if bit_depth < 10 or not supports_10bit(encoder):
        return "yuv420p"
    if encoder.startswith("lib") or "_" not in encoder:
        return "yuv420p10le"
    return "p010le"


def available_encoders() -> set[str]:
    """Encoder names this ffmpeg was compiled with.

    Necessary but nowhere near sufficient: this build advertises `hevc_qsv`,
    `hevc_amf` and `hevc_vaapi` regardless of whether the matching hardware or
    driver is present. Only `probe_encoder` can tell.
    """
    try:
        out = proc.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        log.warning("could not list ffmpeg encoders", exc_info=True)
        return set()

    names: set[str] = set()
    for line in (out.stdout or "").splitlines():
        parts = line.split()
        # Encoder rows look like " V....D libx265  libx265 H.265 / HEVC ...".
        if len(parts) >= 2 and parts[0].startswith("V") and len(parts[0]) == 6:
            names.add(parts[1])
    return names


def probe_encoder(encoder: str, quality: int = 20) -> bool:
    """Run a real encode and report whether ffmpeg survived it.

    A listing entry only means the encoder was compiled in. `hevc_amf` on this
    machine fails with "DLL amfrt64.dll failed to open" and `hevc_vaapi` cannot
    build a filter graph at all, yet both are listed.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-f", "lavfi",
        "-i", f"color=c=black:size={PROBE_SIZE}x{PROBE_SIZE}:rate=25:duration=0.2",
        "-c:v", encoder,
        *quality_args(encoder, quality),
        "-pix_fmt", pixel_format(encoder, 8),
        "-f", "null", "-",
    ]
    try:
        result = proc.run(
            cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.SubprocessError):
        log.debug("encoder probe for %s could not run", encoder, exc_info=True)
        return False
    return result.returncode == 0


# Probing costs a few hundred milliseconds per candidate, so the answer is
# remembered for the life of the process. `reset_encoder_cache` exists for
# tests; hardware does not change under a running render.
_ENCODER_CACHE: str | None = None


def reset_encoder_cache() -> None:
    """Forget the probed encoder. For tests, and for nothing else."""
    global _ENCODER_CACHE
    _ENCODER_CACHE = None


def choose_encoder(probe: bool = True, use_cache: bool = True) -> str:
    """The best encoder that works here, in `CANDIDATE_ENCODERS` order.

    With `probe=False` the choice comes from the ffmpeg listing alone, which is
    fast but credulous. Use it only where a wrong answer is harmless.
    """
    global _ENCODER_CACHE
    if probe and use_cache and _ENCODER_CACHE is not None:
        return _ENCODER_CACHE

    listed = available_encoders()
    if not listed and not probe:
        # No listing and no probe leaves nothing to decide on. Guessing the
        # top candidate here is how the NVIDIA-only bug happened in the first
        # place; take the encoder that is always present instead.
        return FALLBACK_ENCODER

    chosen = FALLBACK_ENCODER
    for name in CANDIDATE_ENCODERS:
        if listed and name not in listed:
            continue
        if not probe:
            chosen = name
            break
        if probe_encoder(name):
            chosen = name
            break
    else:
        log.warning(
            "no candidate encoder passed its probe; falling back to %s",
            FALLBACK_ENCODER,
        )

    if probe and use_cache:
        _ENCODER_CACHE = chosen
    return chosen


def default_encoder() -> str:
    """The probed, cached encoder for this machine. What `Encoder` asks for."""
    return choose_encoder(probe=True, use_cache=True)


# --------------------------------------------------------------------------
# CPU, RAM, accelerator
# --------------------------------------------------------------------------


def _ram_bytes() -> int:
    """Physical RAM, or 0 if this machine will not say. `psutil` is not a dependency."""
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return 0

        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5.0,
            )
            return int(out.stdout.strip() or 0)

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except Exception:
        log.debug("could not determine RAM size", exc_info=True)
        return 0


def _cpu_name() -> str:
    """A readable CPU model. `platform.processor()` is empty on most Linux."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5.0,
            )
            name = out.stdout.strip()
            if name:
                return name
        elif system == "Windows":
            try:
                import winreg

                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                with key:
                    name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
                if name.strip():
                    return name.strip()
            except OSError:
                pass
            env_name = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
            if env_name:
                return env_name
    except Exception:
        log.debug("could not determine CPU name", exc_info=True)

    return (platform.processor() or platform.machine() or "unknown").strip()


def _accelerator() -> tuple[str, str, int]:
    """(kind, gpu name, VRAM bytes). Every branch is guarded.

    Older torch builds have no `torch.xpu` and no `torch.backends.mps` at all,
    so every attribute is reached for defensively rather than assumed.
    """
    try:
        import torch
    except Exception:
        log.debug("torch is not importable; assuming CPU", exc_info=True)
        return ("cpu", "", 0)

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return ("cuda", str(props.name), int(props.total_memory))
    except Exception:
        log.debug("CUDA probe failed", exc_info=True)

    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            vram = 0
            recommended = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
            if callable(recommended):
                vram = int(recommended())
            return ("mps", "Apple Silicon GPU", vram)
    except Exception:
        log.debug("MPS probe failed", exc_info=True)

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            props = xpu.get_device_properties(0)
            return (
                "xpu",
                str(getattr(props, "name", "Intel GPU")),
                int(getattr(props, "total_memory", 0)),
            )
    except Exception:
        log.debug("XPU probe failed", exc_info=True)

    return ("cpu", "", 0)


def detect(probe_encoders: bool = True) -> Hardware:
    """Read this machine. Never raises: a failed read means a poorer default.

    `probe_encoders=False` skips the ffmpeg probes entirely, for callers that
    only want the CPU and RAM facts and do not want to pay for a real encode.
    """
    try:
        os_name = platform.system() or "unknown"
    except Exception:
        os_name = "unknown"

    try:
        machine = platform.machine() or "unknown"
    except Exception:
        machine = "unknown"

    try:
        cores = os.cpu_count() or 0
    except Exception:
        cores = 0

    accelerator, gpu_name, vram = _accelerator()

    try:
        encoder = choose_encoder(probe=probe_encoders)
    except Exception:
        log.warning("encoder selection failed; using %s", FALLBACK_ENCODER, exc_info=True)
        encoder = FALLBACK_ENCODER

    return Hardware(
        os_name=os_name,
        machine=machine,
        cpu_name=_cpu_name(),
        cpu_cores=cores,
        ram_bytes=_ram_bytes(),
        accelerator=accelerator,
        gpu_name=gpu_name,
        vram_bytes=vram,
        encoder=encoder,
        encoder_is_hardware=encoder in HARDWARE_ENCODERS,
    )


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.1f} GiB" if value else "unknown"


def describe(hw: Hardware) -> str:
    """A few short lines for a human. Not a format anything should parse."""
    if hw.accelerator == "cpu":
        gpu_line = "GPU:      none detected (CPU only)"
    else:
        name = hw.gpu_name or hw.accelerator
        vram = f", {_gib(hw.vram_bytes)}" if hw.vram_bytes else ""
        gpu_line = f"GPU:      {name} ({hw.accelerator}{vram})"

    kind = "hardware" if hw.encoder_is_hardware else "software"
    return "\n".join([
        f"System:   {hw.os_name} {hw.machine}",
        f"CPU:      {hw.cpu_name} ({hw.cpu_cores} threads)",
        f"Memory:   {_gib(hw.ram_bytes)}",
        gpu_line,
        f"Encoder:  {hw.encoder} ({kind})",
    ])
