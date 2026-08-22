"""What the current settings will produce, worked out before anything runs.

The source panel says what came in. This says what will come out: size, frame
rate, the processing that will be applied, roughly how long it will take and
roughly how large it will be.

The point is that a full feature can take many hours. Finding out the output
was 4x larger than intended, or that the chosen model needs a day, is worth
knowing before starting rather than after.

The built-in numbers were measured on an RTX 3060 Laptop, which is the wrong
machine for almost everybody. So the estimate calibrates itself: every finished
render reports what it actually achieved, that is remembered per model, and
later estimates use the real figure. A model never run here is scaled by how
this machine compares to the reference on the models it has run.

Until anything has been measured the reference numbers are used unchanged and
the estimate says so. They are honest estimates either way, not promises: trust
the live frames-per-second readout once a render is actually going.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Measured input megapixels per second, per model. Input pixels rather than
# output, because most of the work in these networks happens at input
# resolution and only the final upsample scales with the output.
MEASURED_MPPS: dict[str, float] = {
    "2xParimgCompact": 8.0,
    "2xModernSpanimationV1": 5.6,
    "RealESRGAN_x2plus": 0.82,
    "realesr-general-x4v3": 4.35,
    "4xPurePhoto-span": 6.9,
    "4xNomos2_realplksr_dysample": 0.25,
}

# Fallbacks by architecture, for a model the table has never seen.
ARCH_MPPS: dict[str, float] = {
    "compact": 6.0,
    "span": 5.5,
    "plksr": 0.3,
    "esrgan": 0.8,
    "dat": 0.2,
    "swinir": 0.2,
}

DEFAULT_MPPS = 2.0

# The table above is measured at 854x480 input.
REFERENCE_MEGAPIXELS = 854 * 480 / 1_000_000

# Throughput is not flat in input megapixels: going from 480p to 1080p cost
# about a quarter of it, consistently across every architecture measured.
# Larger frames mean larger tiles and more memory traffic per pixel. This
# exponent reproduces that 25% drop.
RESOLUTION_FALLOFF = 0.178

# Restoration and texture work measured at roughly a third off throughput.
RESTORATION_COST = 1.35

# Interpolation runs the flow network on top of the upscaler.
INTERPOLATION_COST = 2.0

# Rough bits per output pixel for 10-bit HEVC at the default quality. Real
# footage varies enormously with grain and motion, so this is a ballpark.
BITS_PER_PIXEL = 0.10


@dataclass
class Forecast:
    width: int
    height: int
    fps: float
    frames: int
    duration_seconds: float
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    megabytes: float = 0.0

    @property
    def resolution(self) -> str:
        return f"{self.width} x {self.height}"

    @property
    def label(self) -> str:
        """A familiar name for the output size, only when it earns one.

        Named by width, and only near a real standard. Calling 1708x960 "720p"
        would be worse than saying nothing: 720p is 1280x720, and a wrong
        familiar name is more misleading than an unfamiliar exact one.
        """
        for low, high, name in (
            (7000, 99999, "8K"),
            (3400, 4400, "4K"),
            (2400, 2800, "2K"),
            (1800, 2100, "1080p"),
            (1200, 1400, "720p"),
        ):
            if low <= self.width < high:
                return name
        return ""

    @property
    def time_estimate(self) -> str:
        if self.seconds <= 0:
            return "unknown"
        if self.seconds < 90:
            return f"{self.seconds:.0f} seconds"
        if self.seconds < 5400:
            return f"{self.seconds / 60:.0f} minutes"
        hours = self.seconds / 3600
        return f"{hours:.1f} hours" if hours < 48 else f"{hours / 24:.1f} days"

    @property
    def size_estimate(self) -> str:
        if self.megabytes <= 0:
            return "unknown"
        if self.megabytes < 1024:
            return f"{self.megabytes:.0f} MB"
        return f"{self.megabytes / 1024:.1f} GB"


def _state_dir() -> Path:
    """Where this machine's own measurements live.

    Not beside the models: those travel with the application, and a figure
    measured on one machine would be wrong on the next one to open the folder.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "Enhancer"


def measurements_path() -> Path:
    return _state_dir() / "throughput.json"


def load_measurements() -> dict[str, float]:
    """Megapixels a second actually achieved on this machine, by model.

    Never raises. A missing, unreadable or corrupt file simply means nothing
    has been measured yet, which is the normal state on a first run.
    """
    try:
        raw = json.loads(measurements_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(k): float(v) for k, v in raw.items()
        if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0
    }


def record_measurement(model_name: str, megapixels_per_second: float) -> None:
    """Remember what a finished render actually achieved.

    Averaged with whatever is already stored rather than replacing it, so one
    render that fought a game for the card does not become the new truth.
    """
    if not (megapixels_per_second > 0 and math.isfinite(megapixels_per_second)):
        return
    stem = model_name.rsplit(".", 1)[0]
    known = load_measurements()
    previous = known.get(stem)
    known[stem] = (
        megapixels_per_second if previous is None
        else (previous + megapixels_per_second) / 2.0
    )
    try:
        path = measurements_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(known, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:  # a read-only home is not worth failing a render over
        log.info("could not save the throughput measurement: %s", exc)


def machine_ratio(measurements: dict[str, float] | None = None) -> float:
    """How this machine compares with the reference, from what it has run.

    The geometric mean over models measured here that the reference table also
    knows. Geometric because these are ratios: a machine twice as fast on one
    model and half as fast on another is, on balance, neither.

    1.0 when nothing comparable has been measured — meaning "assume the
    reference machine", which is the only honest default.
    """
    known = load_measurements() if measurements is None else measurements
    ratios = [
        known[stem] / MEASURED_MPPS[key]
        for stem in known
        for key in MEASURED_MPPS
        if key.lower() == stem.lower() and MEASURED_MPPS[key] > 0
    ]
    if not ratios:
        return 1.0
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def throughput_for(
    model_name: str, arch: str = "", measurements: dict[str, float] | None = None
) -> float:
    """Input megapixels a second for a model on THIS machine.

    A figure measured here wins outright. Otherwise the reference figure is
    scaled by how this machine has compared on everything else it has run,
    which is a far better guess than the reference number on a card that is
    not an RTX 3060 Laptop.
    """
    stem = model_name.rsplit(".", 1)[0]
    known = load_measurements() if measurements is None else measurements

    for key, value in known.items():
        if key.lower() == stem.lower():
            return value

    ratio = machine_ratio(known)
    for key, value in MEASURED_MPPS.items():
        if key.lower() == stem.lower():
            return value * ratio

    haystack = f"{stem} {arch}".lower()
    for key, value in ARCH_MPPS.items():
        if key in haystack:
            return value * ratio
    return DEFAULT_MPPS * ratio


def forecast(
    width: int,
    height: int,
    fps: float,
    frames: int,
    scale: int,
    model_name: str,
    arch: str = "",
    scan: str = "progressive",
    deblock: float = 0.0,
    degrain: float = 0.0,
    detail_retention: float = 0.0,
    regrain: float = 0.0,
    target_fps: float | None = None,
    cpu: bool = False,
    is_image: bool = False,
) -> Forecast:
    """Work out what these settings will produce."""
    out_w, out_h = width * scale, height * scale
    out_fps = target_fps or fps
    duration = frames / fps if fps else 0.0
    out_frames = 1 if is_image else int(round(duration * out_fps))

    result = Forecast(
        width=out_w, height=out_h, fps=out_fps,
        frames=out_frames, duration_seconds=duration,
    )

    # What will actually happen to the picture, in order.
    if not is_image:
        if scan == "telecined":
            result.steps.append("Recover original film frames (inverse telecine)")
        elif scan == "interlaced":
            result.steps.append("Deinterlace")
    if deblock > 0:
        result.steps.append(f"Remove compression artifacts ({deblock:.2f})")
    if degrain > 0:
        result.steps.append(f"Reduce noise and grain ({degrain:.2f})")
    result.steps.append(f"Enlarge {scale}x with {model_name}")
    if detail_retention > 0:
        result.steps.append(f"Restore original detail ({detail_retention:.2f})")
    if regrain > 0:
        result.steps.append(f"Add film grain ({regrain:.2f})")
    if target_fps and not is_image:
        result.steps.append(f"Smooth motion to {out_fps:g} fps")

    # Time.
    megapixels = (width * height) / 1_000_000
    rate = throughput_for(model_name, arch)
    if megapixels > 0:
        rate *= (REFERENCE_MEGAPIXELS / megapixels) ** RESOLUTION_FALLOFF
    if cpu:
        rate /= 40.0  # measured order of magnitude, not a precise figure
    per_frame = megapixels / rate if rate else 0.0
    if degrain or deblock or detail_retention or regrain:
        per_frame *= RESTORATION_COST
    if target_fps and not is_image:
        per_frame *= INTERPOLATION_COST
    source_frames = 1 if is_image else frames
    result.seconds = per_frame * source_frames

    # Size.
    if is_image:
        result.megabytes = (out_w * out_h * 3) / 1024 ** 2
    else:
        bits = out_w * out_h * BITS_PER_PIXEL * out_fps * duration
        result.megabytes = bits / 8 / 1024 ** 2

    result.warnings = _warnings(
        width, height, scale, out_w, result.seconds, scan, target_fps, cpu, is_image
    )
    return result


def _warnings(
    width: int, height: int, scale: int, out_w: int,
    seconds: float, scan: str, target_fps: float | None,
    cpu: bool, is_image: bool,
) -> list[str]:
    notes: list[str] = []

    if scale >= 4 and width >= 1600:
        notes.append(
            f"A {scale}x model on a {width}-wide source gives {out_w} wide and "
            f"does {scale * scale // 4}x the work of a 2x one. 2x already "
            f"reaches 4K from 1080p."
        )
    if seconds > 6 * 3600 and not is_image:
        notes.append(
            "This is an overnight job. Preview ten seconds first — the render "
            "can be stopped and resumed at any point."
        )
    if scan == "telecined" and target_fps:
        notes.append(
            "Frame rate conversion cannot be combined with inverse telecine in "
            "one pass. Convert this source first, then smooth the result."
        )
    if scan == "telecined":
        notes.append("Inverse telecine renders in one piece, so this job cannot resume.")
    if cpu:
        notes.append("Running on the processor is dramatically slower than the graphics card.")
    return notes
