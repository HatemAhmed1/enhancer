"""Headless benchmark harness validating the project's throughput targets."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .upscale import Upscaler

# A 2-hour feature at 24 fps.
FEATURE_FRAMES = 172_800


@dataclass(frozen=True)
class BenchResult:
    arch: str
    frames: int
    seconds: float
    fps: float
    in_width: int
    in_height: int
    out_width: int
    out_height: int
    peak_vram_mb: float
    cpu_fallbacks: int
    feature_hours: float

    def format(self) -> str:
        return (
            f"{self.arch}: {self.in_width}x{self.in_height} -> "
            f"{self.out_width}x{self.out_height} | {self.fps:.1f} fps | "
            f"peak VRAM {self.peak_vram_mb:.0f} MB | "
            f"2h feature ~ {self.feature_hours:.1f} h | "
            f"CPU fallbacks: {self.cpu_fallbacks}"
        )


def benchmark(
    model,
    width: int,
    height: int,
    frames: int = 30,
    tile: int = 512,
    overlap: int = 16,
    device: str = "cuda",
    half: bool = True,
    warmup: int = 3,
) -> BenchResult:
    """Time `frames` synthetic frames through the upscaler."""
    up = Upscaler(model, tile=tile, overlap=overlap, device=device, half=half)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    for _ in range(warmup):
        up.process(frame)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(frames):
        out = up.process(frame)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak = (
        torch.cuda.max_memory_allocated() / 1024 ** 2
        if device == "cuda" and torch.cuda.is_available()
        else 0.0
    )
    fps = frames / elapsed if elapsed > 0 else float("inf")

    return BenchResult(
        arch=getattr(model, "arch", type(model).__name__),
        frames=frames,
        seconds=elapsed,
        fps=fps,
        in_width=width,
        in_height=height,
        out_width=out.shape[1],
        out_height=out.shape[0],
        peak_vram_mb=peak,
        cpu_fallbacks=up.cpu_fallback_count,
        feature_hours=(FEATURE_FRAMES / fps) / 3600 if fps else float("inf"),
    )
