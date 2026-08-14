import numpy as np
import pytest
import torch

from enhancer.upscale import Upscaler, to_tensor, to_frame
from enhancer.vram import TileFloorReached


class FakeModel:
    """Stands in for a LoadedModel."""

    def __init__(self, scale=2, fail_on_device=None):
        self.scale = scale
        self.arch = "Fake"
        self.device = torch.device("cpu")
        self.fail_on_device = fail_on_device
        self.calls = []

    def __call__(self, x):
        self.calls.append(x.device.type)
        if self.fail_on_device and x.device.type == self.fail_on_device:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")

    def to(self, device):
        self.device = torch.device(device)
        return self


def test_to_tensor_roundtrip_preserves_pixels(rng):
    frame = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    assert np.array_equal(to_frame(to_tensor(frame)), frame)


def test_to_tensor_shape_and_range(rng):
    frame = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    t = to_tensor(frame)
    assert t.shape == (1, 3, 16, 24)
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_to_tensor_accepts_readonly_frame(rng):
    """Decoder yields read-only arrays from np.frombuffer."""
    frame = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    frame.flags.writeable = False
    t = to_tensor(frame)
    assert t.shape == (1, 3, 16, 24)


def test_upscaler_doubles_dimensions(rng):
    frame = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    up = Upscaler(FakeModel(scale=2), tile=16, overlap=4, device="cpu")
    assert up.process(frame).shape == (64, 96, 3)


def test_upscaler_output_dtype_is_uint8(rng):
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    up = Upscaler(FakeModel(scale=2), tile=16, overlap=4, device="cpu")
    assert up.process(frame).dtype == np.uint8


def test_falls_back_to_cpu_when_tile_floor_reached(rng, monkeypatch):
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = FakeModel(scale=2)
    up = Upscaler(model, tile=32, overlap=0, device="cpu")

    calls = {"n": 0}

    def fake_run(fn, img):
        calls["n"] += 1
        raise TileFloorReached("simulated")

    monkeypatch.setattr(up.runner, "run", fake_run)
    out = up.process(frame)
    assert out.shape == (64, 64, 3)
    assert up.cpu_fallback_count == 1
    assert calls["n"] == 1


def test_cpu_fallback_counter_starts_at_zero():
    up = Upscaler(FakeModel(), tile=32, overlap=0, device="cpu")
    assert up.cpu_fallback_count == 0
