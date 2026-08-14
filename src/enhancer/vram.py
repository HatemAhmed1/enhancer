"""Tiling, tiled execution, and out-of-memory recovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """One tile of an image.

    The *core* rectangle (y0..y1, x0..x1) is the region this tile is responsible
    for in the output. Cores partition the image exactly.

    The *padded* rectangle (py0..py1, px0..px1) extends the core by the overlap
    amount, clamped to the image bounds. This is what gets fed to the model, so
    that core pixels are computed with proper receptive-field context.
    """

    y0: int
    y1: int
    x0: int
    x1: int
    py0: int
    py1: int
    px0: int
    px1: int


def plan_tiles(h: int, w: int, tile: int, overlap: int) -> list[Tile]:
    """Partition an h x w image into tiles of at most `tile` pixels per side."""
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")

    tiles: list[Tile] = []
    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            tiles.append(
                Tile(
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    py0=max(0, y0 - overlap),
                    py1=min(h, y1 + overlap),
                    px0=max(0, x0 - overlap),
                    px1=min(w, x1 + overlap),
                )
            )
    return tiles


from collections.abc import Callable

import torch

InferFn = Callable[[torch.Tensor], torch.Tensor]


def run_tiled(
    fn: InferFn,
    img: torch.Tensor,
    tile: int,
    overlap: int,
    scale: int,
) -> torch.Tensor:
    """Run `fn` over `img` in tiles and reassemble the output.

    `img` is (B, C, H, W). `fn` must upscale by exactly `scale`. Only the core
    region of each tile's output is written, so reassembly is bit-exact.
    """
    if img.ndim != 4:
        raise ValueError(f"expected a 4-D (B, C, H, W) tensor, got shape {tuple(img.shape)}")

    _, _, h, w = img.shape
    tiles = plan_tiles(h, w, tile, overlap)

    if len(tiles) == 1:
        return fn(img)

    out: torch.Tensor | None = None
    for t in tiles:
        patch = img[:, :, t.py0:t.py1, t.px0:t.px1]
        up = fn(patch)

        if out is None:
            b, c = up.shape[0], up.shape[1]
            out = torch.empty(
                (b, c, h * scale, w * scale), dtype=up.dtype, device=up.device
            )

        # Offset of the core within the padded patch, in output pixels.
        oy = (t.y0 - t.py0) * scale
        ox = (t.x0 - t.px0) * scale
        ch = (t.y1 - t.y0) * scale
        cw = (t.x1 - t.x0) * scale

        out[:, :, t.y0 * scale:t.y1 * scale, t.x0 * scale:t.x1 * scale] = (
            up[:, :, oy:oy + ch, ox:ox + cw]
        )

    assert out is not None
    return out
