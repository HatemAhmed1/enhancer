"""Frame upscaling with tiling and a per-frame CPU fallback."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .vram import TileFloorReached, TileRunner, physical_vram_ceiling, total_vram_bytes

log = logging.getLogger(__name__)


def to_tensor(frame: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> (1, C, H, W) float32 in [0, 1].

    Copies rather than viewing: Decoder yields read-only arrays from
    np.frombuffer, and torch.from_numpy rejects non-writable buffers.
    """
    array = np.array(frame, dtype=np.uint8, copy=True, order="C")
    t = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return t.float().div_(255.0)


def to_frame(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) float -> HWC uint8."""
    t = t.detach().float().clamp_(0.0, 1.0).mul_(255.0).round_()
    return t.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()


class Upscaler:
    """Upscales single frames, degrading gracefully rather than crashing."""

    def __init__(
        self,
        model,
        tile: int,
        overlap: int,
        device: str | torch.device = "cuda",
        half: bool = True,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"
        if self.half:
            # Derive the half decision from the model's ACTUAL parameter
            # dtype, not from the caller's flag alone. load_model() only casts
            # to half when spandrel reports descriptor.supports_half is True
            # (e.g. PLKSR/RealPLKSR does not support it), so trusting `half`
            # blindly here fed half-precision input into float32 weights and
            # crashed with a dtype mismatch. When the model exposes no dtype
            # (e.g. a bare test double), we cannot verify it, so we fall back
            # to trusting the caller's flag.
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is not None and model_dtype != torch.float16:
                self.half = False
                log.info(
                    "half precision requested but %s weights are %s, not "
                    "float16 (architecture does not support half); running "
                    "this tier in fp32, which will be slower",
                    getattr(model, "arch", type(model).__name__), model_dtype,
                )
        ceiling = physical_vram_ceiling(total_vram_bytes()) if self.device.type == "cuda" else None
        self.runner = TileRunner(
            tile=tile, overlap=overlap, scale=model.scale,
            min_tile=128, vram_ceiling=ceiling,
        )
        self.cpu_fallback_count = 0

    def _infer(self, patch: torch.Tensor) -> torch.Tensor:
        if self.half:
            patch = patch.half()
        patch = patch.to(memory_format=torch.channels_last)
        return self.model(patch).float()

    def process(self, frame: np.ndarray) -> np.ndarray:
        img = to_tensor(frame).to(self.device)
        track = self.device.type == "cuda" and torch.cuda.is_available()
        if track:
            torch.cuda.reset_peak_memory_stats()
        try:
            out = self.runner.run(self._infer, img)
            if track:
                # The true peak is only known once the run has finished, so it
                # cannot be passed into run() itself; this is the single live
                # pressure check for this frame (run()'s own internal check is
                # a no-op here since it is called without peak_bytes).
                peak = int(torch.cuda.max_memory_allocated())
                self.runner._check_pressure(peak)
        except TileFloorReached:
            self.cpu_fallback_count += 1
            log.warning(
                "Frame did not fit at minimum tile size; processing on CPU "
                "(fallback #%d)", self.cpu_fallback_count,
            )
            out = self._process_on_cpu(frame)
        return to_frame(out)

    def _process_on_cpu(self, frame: np.ndarray) -> torch.Tensor:
        """Last-resort path. Slow, but per-frame and it never crashes."""
        self.model.to("cpu")
        try:
            img = to_tensor(frame)
            with torch.inference_mode():
                return self.model(img).float()
        finally:
            self.model.to(self.device)
