"""Single-image upscaling.

Images are not one-frame videos. A still has no frame rate, no audio, no
segments and nothing to resume, and it must come out as a PNG or JPEG rather
than a Matroska file. Routing them through the video path produced either a
hard failure on `frame_count == 0` or a video container holding one frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Formats that cannot store alpha; an RGBA source is flattened onto black.
OPAQUE_SUFFIXES = {".jpg", ".jpeg", ".bmp"}


def is_image(path: str | Path) -> bool:
    """True when `path` should be treated as a still rather than a video."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def load_image(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an image as HWC uint8 RGB, plus its alpha channel if it has one.

    Alpha is separated rather than fed to the model: super-resolution networks
    expect three channels, and an alpha channel run through one would come back
    as hallucinated colour.
    """
    with Image.open(path) as handle:
        image = handle.convert("RGBA") if handle.mode in ("RGBA", "LA", "PA") else handle.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)

    if array.shape[2] == 4:
        return np.ascontiguousarray(array[:, :, :3]), np.ascontiguousarray(array[:, :, 3])
    return np.ascontiguousarray(array), None


def save_image(
    path: str | Path,
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    quality: int = 95,
) -> Path:
    """Write HWC uint8 RGB, reattaching alpha where the format allows it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if alpha is not None and path.suffix.lower() not in OPAQUE_SUFFIXES:
        merged = np.dstack([rgb, alpha])
        Image.fromarray(merged, mode="RGBA").save(path)
        return path

    image = Image.fromarray(rgb, mode="RGB")
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(path, quality=quality, subsampling=0)
    else:
        image.save(path)
    return path


def estimate_still_grain(rgb: np.ndarray) -> tuple[float, float]:
    """Grain and blockiness for a still, reusing the video estimators."""
    from .analyze import estimate_blockiness, estimate_grain

    return estimate_grain(rgb), estimate_blockiness(rgb)


def upscale_image(
    source: str | Path,
    output: str | Path,
    upscaler,
    texture=None,
    quality: int = 95,
) -> Path:
    """Upscale one still image.

    Alpha is resized separately with a plain nearest-neighbour expansion, which
    keeps hard edges hard and avoids inventing translucency the source did not
    have.
    """
    rgb, alpha = load_image(source)
    result = upscaler.process(rgb)

    if texture is not None and getattr(texture, "enabled", False):
        result = texture.apply(result, rgb)

    if alpha is not None:
        scale = upscaler.scale
        alpha = np.repeat(np.repeat(alpha, scale, axis=0), scale, axis=1)
        alpha = alpha[: result.shape[0], : result.shape[1]]

    return save_image(output, result, alpha, quality=quality)
