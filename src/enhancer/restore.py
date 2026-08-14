"""Restoration: pre-upscale filtering and post-upscale texture work."""

from __future__ import annotations

from dataclasses import dataclass

from .analyze import ScanType

# Maximum hqdn3d strengths at degrain=1.0. Temporal is allowed to go
# considerably higher than spatial: temporal denoising removes frame-to-frame
# noise while leaving within-frame detail comparatively intact, which is the
# correct bias when skin texture is the thing being protected.
MAX_LUMA_SPATIAL = 4.0
MAX_CHROMA_SPATIAL = 3.0
MAX_LUMA_TEMPORAL = 9.0
MAX_CHROMA_TEMPORAL = 7.0


@dataclass(frozen=True)
class RestoreSettings:
    """Restoration strengths, each in the range 0.0 to 1.0.

    `degrain` defaults low on purpose. Grain and skin micro-texture occupy the
    same spatial frequencies, so an aggressive degrain removes exactly the
    detail this project exists to preserve.
    """

    deblock: float = 0.0
    degrain: float = 0.25
    detail_retention: float = 0.25
    regrain: float = 0.6


def _check(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


def build_filter_chain(
    scan: ScanType,
    field_order: str,
    settings: RestoreSettings,
) -> str:
    """Build the ffmpeg -vf chain applied before upscaling.

    These run inside the decoder at native C speed. Reimplementing them per
    frame in Python would cost more than the upscaler itself.
    """
    _check("deblock", settings.deblock)
    _check("degrain", settings.degrain)

    filters: list[str] = []

    if scan is ScanType.TELECINED:
        # Recovers the original progressive frames exactly. A deinterlacer
        # would instead discard half the real information in every frame.
        filters.append("fieldmatch")
        filters.append("decimate")
    elif scan is ScanType.INTERLACED:
        parity = 1 if field_order == "bff" else 0
        filters.append(f"bwdif=mode=0:parity={parity}:deint=0")

    if settings.deblock > 0:
        strength = 0.02 + 0.08 * settings.deblock
        filters.append(f"deblock=filter=weak:block=8:alpha={strength:.3f}")

    if settings.degrain > 0:
        d = settings.degrain
        filters.append(
            "hqdn3d="
            f"{MAX_LUMA_SPATIAL * d:.2f}:"
            f"{MAX_CHROMA_SPATIAL * d:.2f}:"
            f"{MAX_LUMA_TEMPORAL * d:.2f}:"
            f"{MAX_CHROMA_TEMPORAL * d:.2f}"
        )

    return ",".join(filters)
