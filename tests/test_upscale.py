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


def test_upscaler_exposes_model_scale():
    """render_resumable and other callers read upscaler.scale directly;

    it must not be necessary to reach through to upscaler.model.scale.
    """
    up = Upscaler(FakeModel(scale=4), tile=16, overlap=4, device="cpu")
    assert up.scale == 4


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


class FakeFloat32Model:
    """Stands in for a LoadedModel whose spandrel descriptor reported
    supports_half=False (e.g. PLKSR/RealPLKSR), so its weights are still
    float32. Raises the real-world error if ever handed a half input, so a
    failure to gate `half` correctly shows up as a crash, not silently."""

    def __init__(self, scale=2):
        self.scale = scale
        self.arch = "FakeFloat32"
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def __call__(self, x):
        if x.dtype == torch.float16:
            raise RuntimeError(
                "Input type (struct c10::Half) and bias type (float) "
                "should be the same"
            )
        return torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")

    def to(self, device):
        self.device = torch.device(device)
        return self


class FakeHalfModel:
    """Stands in for a LoadedModel whose weights were successfully cast to
    half (supports_half=True), mirroring load_model()'s descriptor.model.half()."""

    def __init__(self, scale=2):
        self.scale = scale
        self.arch = "FakeHalf"
        self.dtype = torch.float16
        self.device = torch.device("cpu")

    def __call__(self, x):
        assert x.dtype == torch.float16, f"expected a half input, got {x.dtype}"
        out = torch.nn.functional.interpolate(x.float(), scale_factor=self.scale, mode="nearest")
        return out.half()

    def to(self, device):
        self.device = torch.device(device)
        return self


def test_half_disabled_when_model_weights_are_float32(rng):
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = FakeFloat32Model(scale=2)
    up = Upscaler(model, tile=16, overlap=4, device="cpu", half=True)
    assert up.half is False
    out = up.process(frame)  # must not raise
    assert out.shape == (64, 64, 3)


@pytest.mark.gpu
def test_half_enabled_when_model_weights_are_half(rng):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = FakeHalfModel(scale=2)
    up = Upscaler(model, tile=16, overlap=4, device="cuda", half=True)
    assert up.half is True
    out = up.process(frame)
    assert out.shape == (64, 64, 3)


@pytest.mark.gpu
def test_half_requested_but_float32_model_does_not_crash_on_cuda(rng):
    """Reproduces the reported bug directly: half=True + cuda device used to
    be sufficient to set self.half=True regardless of what the model's
    weights actually are. For an architecture like RealPLKSR where
    supports_half is False, that forced a half input into float32 weights and
    crashed with 'Input type (struct c10::Half) and bias type (float) should
    be the same'. This is the scenario that must fail before the fix."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = FakeFloat32Model(scale=2)
    up = Upscaler(model, tile=16, overlap=4, device="cuda", half=True)
    assert up.half is False
    out = up.process(frame)  # must not raise
    assert out.shape == (64, 64, 3)


def test_vram_budget_caps_the_ceiling():
    """An explicit cap keeps the machine usable for other work."""
    up = Upscaler(FakeModel(), tile=256, overlap=0, device="cpu",
                  vram_budget=512 * 1024 ** 2)
    # On CPU there is no ceiling at all; the budget only applies to CUDA.
    assert up.runner.vram_ceiling is None


@pytest.mark.gpu
def test_vram_budget_is_used_instead_of_the_physical_ceiling():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    budget = 512 * 1024 ** 2
    up = Upscaler(FakeModel(), tile=256, overlap=0, device="cuda",
                  half=False, vram_budget=budget)
    assert up.runner.vram_ceiling == budget


@pytest.mark.gpu
def test_no_budget_falls_back_to_the_physical_ceiling():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from enhancer.vram import physical_vram_ceiling, total_vram_bytes

    up = Upscaler(FakeModel(), tile=256, overlap=0, device="cuda", half=False)
    assert up.runner.vram_ceiling == physical_vram_ceiling(total_vram_bytes())
