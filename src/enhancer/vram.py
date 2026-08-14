"""Tiling, tiled execution, and out-of-memory recovery."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)

InferFn = Callable[[torch.Tensor], torch.Tensor]

# Descending ladder of candidate tile sizes. Used for stepping in BOTH
# directions so that shrink and recover are symmetric.
TILE_LADDER: tuple[int, ...] = (1024, 768, 512, 384, 256, 192, 128)

# Reserved VRAM. A laptop GPU also drives the desktop compositor under WDDM,
# so this is deliberately generous.
HEADROOM_BYTES: int = 1024 ** 3

# Upper bound on the success-probe backoff, so probes stay rare but never stop.
MAX_PROBE_INTERVAL: int = 8192

# Substrings identifying a RECOVERABLE allocation failure. Deliberately narrow:
# "CUDA error: an illegal memory access" is unrecoverable and must NOT match,
# because retrying it forever would hide a real bug.
_OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "alloc_failed",
    "cuda_error_out_of_memory",
)


def is_oom_error(exc: BaseException) -> bool:
    """True if `exc` is a recoverable allocation failure.

    CUDA exhaustion does not always arrive as torch.cuda.OutOfMemoryError.
    cuDNN and cuBLAS workspace allocation and driver-level failures surface as
    a plain RuntimeError, and the CPU fallback path raises MemoryError. Catching
    only the typed error lets a real OOM kill a multi-hour render.
    """
    if isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(marker in message for marker in _OOM_MARKERS)
    return False


class TileFloorReached(Exception):
    """Tiling cannot make this frame fit. The caller should fall back to CPU.

    Deliberately NOT a RuntimeError subclass: TileRunner widens its except
    clause to RuntimeError, and this signal must pass through it uncaught.
    """


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


def _check_scale(out: torch.Tensor, inp: torch.Tensor, scale: int) -> None:
    """Verify the model actually upscaled by `scale`.

    Without this, a mismatch (e.g. a 4x model driven with scale=2) slices
    successfully and writes scrambled pixels with no error at all.
    """
    expected = (inp.shape[-2] * scale, inp.shape[-1] * scale)
    actual = tuple(out.shape[-2:])
    if actual != expected:
        raise ValueError(
            f"model returned {actual} for a {tuple(inp.shape[-2:])} input; "
            f"expected {expected} at scale={scale}"
        )


def _alloc_output(sample: torch.Tensor, h: int, w: int, scale: int) -> torch.Tensor:
    """Allocate the full output buffer, attributing its failure correctly.

    This buffer's size depends only on h, w and scale, so shrinking tiles cannot
    make it fit. Reporting it as TileFloorReached sends the caller straight to
    the CPU fallback instead of burning several pointless halving rounds and
    then blaming the tile size.
    """
    try:
        return torch.empty(
            (sample.shape[0], sample.shape[1], h * scale, w * scale),
            dtype=sample.dtype,
            device=sample.device,
        )
    except Exception as exc:
        if is_oom_error(exc):
            raise TileFloorReached(
                f"output buffer {h * scale}x{w * scale} does not fit in memory; "
                f"tiling cannot reduce it"
            ) from exc
        raise


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
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    _, _, h, w = img.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"image has a zero dimension: {h}x{w}")

    tiles = plan_tiles(h, w, tile, overlap)

    if len(tiles) == 1:
        out = fn(img)
        _check_scale(out, img, scale)
        return out

    out: torch.Tensor | None = None
    for t in tiles:
        patch = img[:, :, t.py0:t.py1, t.px0:t.px1]
        up = fn(patch)
        _check_scale(up, patch, scale)

        if out is None:
            out = _alloc_output(up, h, w, scale)

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


class TileRunner:
    """Runs tiled inference, shrinking tiles on OOM and probing back upward.

    On an allocation failure the tile steps down the ladder and the frame is
    retried. After sustained success the runner probes a larger tile again, with
    the probe interval doubling after each failure. That converges: a size that
    genuinely does not fit is retried a handful of times over a long render
    rather than every `recover_after` frames forever, while a transient spike
    from another application still recovers.
    """

    def __init__(
        self,
        tile: int,
        overlap: int,
        scale: int,
        min_tile: int = 128,
        recover_after: int = 64,
    ) -> None:
        self.tile = tile
        self.max_tile = tile
        self.overlap = overlap
        self.scale = scale
        self.min_tile = min_tile
        self.recover_after = recover_after
        self._probe_interval = recover_after
        self._successes = 0

    def _step_down(self, tile: int) -> int:
        for rung in TILE_LADDER:
            if rung < tile:
                return max(self.min_tile, rung)
        return self.min_tile

    def _step_up(self, tile: int) -> int:
        for rung in reversed(TILE_LADDER):
            if rung > tile:
                return rung
        return self.max_tile

    def run(self, fn: InferFn, img: torch.Tensor) -> torch.Tensor:
        # Doubles at most once per call: an OOM'd frame can retry through
        # several ladder rungs before it fits, and that is still a single
        # transient event, not one probe-interval doubling per retry.
        doubled_this_call = False
        while True:
            try:
                out = run_tiled(fn, img, self.tile, self.overlap, self.scale)
            except TileFloorReached:
                raise
            except Exception as exc:
                if not is_oom_error(exc):
                    raise
                self._successes = 0
                if not doubled_this_call:
                    self._probe_interval = min(self._probe_interval * 2, MAX_PROBE_INTERVAL)
                    doubled_this_call = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if self.tile <= self.min_tile:
                    raise TileFloorReached(
                        f"allocation failed at minimum tile size {self.min_tile}"
                    ) from exc
                self.tile = self._step_down(self.tile)
                log.warning("Allocation failure; retrying frame at tile=%d", self.tile)
                continue

            self._note_success()
            return out

    def _note_success(self) -> None:
        if self.tile >= self.max_tile:
            return
        self._successes += 1
        if self._successes >= self._probe_interval:
            self.tile = min(self.max_tile, self._step_up(self.tile))
            self._successes = 0
            log.info("VRAM pressure eased; probing tile size %d", self.tile)


def choose_tile(
    free_bytes: int,
    bytes_per_output_pixel: int,
    overlap: int = 0,
    scale: int = 1,
) -> int:
    """Pick the largest ladder tile whose working set fits the VRAM budget.

    The cost model uses the PADDED tile area, because that is what actually gets
    fed to the model, and scales by `scale**2`, because a 4x model produces 16x
    the pixels. `bytes_per_output_pixel` is a per-output-pixel constant obtained
    by calibration; it must NOT already include the scale factor.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    budget = max(0, free_bytes - HEADROOM_BYTES)
    for tile in TILE_LADDER:
        padded = tile + 2 * overlap
        if padded * padded * bytes_per_output_pixel * scale * scale <= budget:
            return tile
    return TILE_LADDER[-1]


def free_vram_bytes() -> int:
    """Free VRAM on the current CUDA device, or 0 when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
