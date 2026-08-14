"""Restoration: pre-upscale filtering and post-upscale texture work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .analyze import ScanType
from .upscale import to_frame, to_tensor

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


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on an (N, C, H, W) tensor."""
    radius = max(1, int(round(3.0 * sigma)))
    taps = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(taps ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, horizontal, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, vertical, groups=channels)


def apply_detail_retention(
    output: torch.Tensor,
    source: torch.Tensor,
    alpha: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Blend the source's real high-frequency detail over the model output.

    Every other stage in this pipeline either preserves information or invents
    it. This one restores genuinely photographed micro-texture: the residual is
    derived from the source, so it can only contain detail that was actually
    captured. That is precisely why it cannot fabricate pores, and precisely why
    it is the most direct answer to the requirement that skin not look polished.

    `output` is (N, C, H, W) in [0, 1]. `source` is the pre-upscale frame in the
    same layout at lower resolution.
    """
    _check("alpha", alpha)
    if alpha == 0.0:
        return output

    reference = F.interpolate(
        source, size=output.shape[-2:], mode="bicubic", align_corners=False
    )
    residual = reference - gaussian_blur(reference, sigma)
    return (output + alpha * residual).clamp_(0.0, 1.0)


# Grain amplitude at amount=1.0, in normalised [0, 1] units. Roughly 5/255,
# which reads as noticeable but not heavy film grain.
MAX_GRAIN_SIGMA = 0.02


def apply_regrain(
    x: torch.Tensor,
    amount: float,
    seed: int = 0,
    grain_size: float = 0.8,
) -> torch.Tensor:
    """Add synthetic grain to an upscaled frame.

    Super-resolution models classify low-contrast skin texture as noise and
    remove it. Putting grain back is the strongest available counter to the
    resulting plastic appearance: it restores the high-frequency content that
    makes skin read as photographed rather than rendered.

    Grain is attenuated towards black and white, mirroring real film, where
    grain is most visible in the midtones.
    """
    _check("amount", amount)
    if amount == 0.0:
        return x

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device)

    if grain_size > 0:
        noise = gaussian_blur(noise, sigma=grain_size)
        # Blurring reduces variance; renormalise so `amount` stays meaningful.
        std = noise.std()
        if std > 0:
            noise = noise / std

    # Triangular midtone weighting: peaks at 0.5, falls to zero at 0 and 1.
    luminance = x.mean(dim=1, keepdim=True)
    weight = 1.0 - (2.0 * luminance - 1.0).abs()

    return (x + noise * weight * (MAX_GRAIN_SIGMA * amount)).clamp_(0.0, 1.0)


class TexturePost:
    """Post-upscale texture work: detail retention then re-graining.

    Ordering matters. Detail retention restores real high frequencies from the
    source; re-graining then adds synthetic texture on top. Reversing them would
    high-pass the synthetic grain into the retention step and double-count it.
    """

    def __init__(
        self,
        detail_retention: float = 0.25,
        regrain: float = 0.6,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        _check("detail_retention", detail_retention)
        _check("regrain", regrain)
        self.detail_retention = detail_retention
        self.regrain = regrain
        self.device = torch.device(device)
        self.seed = seed

    @property
    def enabled(self) -> bool:
        return self.detail_retention > 0 or self.regrain > 0

    def apply(self, output: np.ndarray, source: np.ndarray, index: int = 0) -> np.ndarray:
        """Apply texture work to one upscaled frame."""
        if not self.enabled:
            return output

        out_t = to_tensor(output).to(self.device)
        if self.detail_retention > 0:
            src_t = to_tensor(source).to(self.device)
            out_t = apply_detail_retention(out_t, src_t, self.detail_retention)
        if self.regrain > 0:
            # Seed varies per frame so grain moves like film rather than sitting
            # static like dirt on the lens.
            out_t = apply_regrain(out_t, self.regrain, seed=self.seed + index)
        return to_frame(out_t)
