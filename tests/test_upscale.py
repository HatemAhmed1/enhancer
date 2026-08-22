from types import SimpleNamespace

import numpy as np
import pytest
import torch

from enhancer.models import LoadedModel
from enhancer.upscale import Upscaler, to_tensor, to_frame
from enhancer.vram import TileFloorReached


# ---------------------------------------------------------------------------
# The test double.
#
# These used to be hand-written classes, and one of them had grown a `to()`
# method that the real LoadedModel did not have. That kept this file green for
# months while Upscaler._process_on_cpu — the CPU fallback that exists so a
# small card never crashes — raised AttributeError the instant it fired, on
# exactly the machines it was written for.
#
# So the double is no longer a look-alike: it IS a LoadedModel, wrapping a
# stub network instead of real weights. It cannot be more capable than the
# class it stands in for, because every method under test is that class's own.
# Only the convolution is fake.
# ---------------------------------------------------------------------------


class _StubNet(torch.nn.Module):
    """A real nn.Module, so device, dtype and memory-format moves behave."""

    def __init__(self, scale):
        super().__init__()
        self.upscale_factor = scale
        # A real parameter: LoadedModel reads .device and .dtype off it, and a
        # stub without one would make both of those lie.
        self.weight = torch.nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x):
        w = self.weight
        if x.dtype != w.dtype:
            # The error a real architecture gives, so a dtype the model was
            # never cast to shows up as a crash rather than silently working.
            raise RuntimeError(
                f"Input type ({x.dtype}) and bias type ({w.dtype}) "
                "should be the same"
            )
        out = torch.nn.functional.interpolate(
            x.float(), scale_factor=self.upscale_factor, mode="nearest"
        )
        return (out * w.float()).to(w.dtype)


class _StubDescriptor:
    """The slice of spandrel's ImageModelDescriptor that LoadedModel uses."""

    def __init__(self, scale, supports_half, arch):
        self.model = _StubNet(scale)
        self.scale = scale
        self.supports_half = supports_half
        self.architecture = SimpleNamespace(name=arch)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, x):
        return self.model(x)


def fake_model(scale=2, half=False, device="cpu", arch="Stub"):
    """A genuine LoadedModel around a stub network.

    `half=True` mirrors what load_model() does for an architecture spandrel
    reports as supporting it; `half=False` mirrors one it does not (PLKSR,
    RealPLKSR), whose weights stay float32 even when half was requested.
    """
    model = LoadedModel(
        _StubDescriptor(scale, supports_half=half, arch=arch),
        torch.device(device),
    )
    if half:
        model.to(dtype=torch.float16)
    return model


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
    up = Upscaler(fake_model(scale=2), tile=16, overlap=4, device="cpu")
    assert up.process(frame).shape == (64, 96, 3)


def test_upscaler_output_dtype_is_uint8(rng):
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    up = Upscaler(fake_model(scale=2), tile=16, overlap=4, device="cpu")
    assert up.process(frame).dtype == np.uint8


def test_upscaler_exposes_model_scale():
    """render_resumable and other callers read upscaler.scale directly;

    it must not be necessary to reach through to upscaler.model.scale.
    """
    up = Upscaler(fake_model(scale=4), tile=16, overlap=4, device="cpu")
    assert up.scale == 4


def _floor_out(up, monkeypatch):
    """Make the tiled path report that it has nowhere left to shrink to."""
    calls = {"n": 0}

    def fake_run(fn, img):
        calls["n"] += 1
        raise TileFloorReached("simulated")

    monkeypatch.setattr(up.runner, "run", fake_run)
    return calls


def test_falls_back_to_cpu_when_tile_floor_reached(rng, monkeypatch):
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = fake_model(scale=2)
    up = Upscaler(model, tile=32, overlap=0, device="cpu")
    calls = _floor_out(up, monkeypatch)
    out = up.process(frame)
    assert out.shape == (64, 64, 3)
    assert up.cpu_fallback_count == 1
    assert calls["n"] == 1


def test_cpu_fallback_returns_the_same_pixels_as_the_tiled_path(rng, monkeypatch):
    """The fallback must produce a real result, not merely fail to crash.

    Counting cpu_fallback_count and checking the shape was all this file used
    to do, and a shape is satisfied by any garbage of the right size.
    """
    frame = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    up = Upscaler(fake_model(scale=2), tile=16, overlap=4, device="cpu")
    expected = up.process(frame)

    _floor_out(up, monkeypatch)
    fallback = up.process(frame)
    assert up.cpu_fallback_count == 1
    assert np.array_equal(fallback, expected)


@pytest.mark.gpu
def test_cpu_fallback_runs_a_half_model_and_gives_it_back(rng, monkeypatch):
    """The bug this whole path shipped with, in one test.

    LoadedModel had no .to() at all, so the first frame that would not fit at
    the minimum tile died with AttributeError — on precisely the small card
    the fallback exists to protect. And once .to() existed, a half model still
    had to be cast: the processor has no real float16 kernels and to_tensor()
    hands it float32 either way. Finally the model has to come back, or every
    later frame silently runs on the CPU for the rest of the render.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = fake_model(scale=2, half=True, device="cuda")
    up = Upscaler(model, tile=16, overlap=4, device="cuda", half=True)
    assert up.half is True

    _floor_out(up, monkeypatch)
    out = up.process(frame)

    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert up.cpu_fallback_count == 1
    # Borrowed for one frame, then handed straight back.
    assert model.device.type == "cuda"
    assert model.dtype == torch.float16


@pytest.mark.gpu
def test_the_frame_after_a_fallback_runs_on_the_accelerator_again(rng, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = fake_model(scale=2, half=True, device="cuda")
    up = Upscaler(model, tile=16, overlap=4, device="cuda", half=True)

    with monkeypatch.context() as ctx:
        _floor_out(up, ctx)
        up.process(frame)

    seen = []
    real_infer = up._infer
    monkeypatch.setattr(up, "_infer", lambda p: seen.append(p.device.type) or real_infer(p))
    up.process(frame)
    assert seen and all(kind == "cuda" for kind in seen)
    assert up.cpu_fallback_count == 1


def test_the_double_is_the_real_class_not_a_look_alike():
    """Guards the failure mode that hid the bug rather than the bug itself.

    The old double was a bespoke class with a to() method LoadedModel did not
    have, so it was strictly *more* capable than the object it stood for and
    the suite passed on a path that could never run. Anything the Upscaler
    reaches for must therefore exist on LoadedModel itself.
    """
    model = fake_model()
    assert isinstance(model, LoadedModel)
    for name in ("scale", "arch", "supports_half", "dtype", "device", "to"):
        assert hasattr(LoadedModel, name) or hasattr(model, name), name
    # Nothing public on the double that the real class does not define.
    extra = {n for n in vars(model) if not n.startswith("_")} - set(dir(LoadedModel))
    assert extra <= {"scale", "arch", "supports_half"}, extra


def test_cpu_fallback_counter_starts_at_zero():
    up = Upscaler(fake_model(), tile=32, overlap=0, device="cpu")
    assert up.cpu_fallback_count == 0


def test_half_disabled_when_model_weights_are_float32(rng):
    """An architecture spandrel reports as supports_half=False (PLKSR,
    RealPLKSR) keeps float32 weights even when half was requested."""
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = fake_model(scale=2, half=False)
    up = Upscaler(model, tile=16, overlap=4, device="cpu", half=True)
    assert up.half is False
    out = up.process(frame)  # must not raise
    assert out.shape == (64, 64, 3)


@pytest.mark.gpu
def test_half_enabled_when_model_weights_are_half(rng):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    model = fake_model(scale=2, half=True, device="cuda")
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
    model = fake_model(scale=2, half=False, device="cuda")
    up = Upscaler(model, tile=16, overlap=4, device="cuda", half=True)
    assert up.half is False
    out = up.process(frame)  # must not raise
    assert out.shape == (64, 64, 3)


def test_vram_budget_caps_the_ceiling():
    """An explicit cap keeps the machine usable for other work."""
    up = Upscaler(fake_model(), tile=256, overlap=0, device="cpu",
                  vram_budget=512 * 1024 ** 2)
    # On CPU there is no ceiling at all; the budget only applies to CUDA.
    assert up.runner.vram_ceiling is None


@pytest.mark.gpu
def test_vram_budget_is_used_instead_of_the_physical_ceiling():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    budget = 512 * 1024 ** 2
    up = Upscaler(fake_model(device="cuda"), tile=256, overlap=0, device="cuda",
                  half=False, vram_budget=budget)
    assert up.runner.vram_ceiling == budget


@pytest.mark.gpu
def test_no_budget_falls_back_to_the_physical_ceiling():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from enhancer.vram import physical_vram_ceiling, total_vram_bytes

    up = Upscaler(fake_model(device="cuda"), tile=256, overlap=0, device="cuda",
                  half=False)
    assert up.runner.vram_ceiling == physical_vram_ceiling(total_vram_bytes())


# ---------------------------------------------------------------------------
# Non-CUDA accelerators. This machine is Windows + NVIDIA, so MPS and XPU are
# simulated: the model and tensors stay on CPU while the Upscaler is told it is
# on another backend, which exercises every decision __init__ makes from the
# device type. Actually running kernels on those backends is not testable here.
# ---------------------------------------------------------------------------


def _pretend_backend(monkeypatch, kind, total):
    """Make `kind` look present, with `total` bytes, without moving tensors."""
    from enhancer import vram as vram_module

    monkeypatch.setitem(vram_module._AVAILABILITY, kind, lambda: True)
    monkeypatch.setitem(vram_module._MEMORY_PROBES, kind, lambda: (total, total))


@pytest.mark.parametrize("kind", ["mps", "xpu"])
def test_accelerators_get_a_pressure_ceiling_like_cuda(monkeypatch, kind):
    from enhancer.vram import HEADROOM_BYTES

    _pretend_backend(monkeypatch, kind, 16 * 1024 ** 3)
    up = Upscaler(fake_model(), tile=256, overlap=0, device=kind, half=False)
    assert up.runner.vram_ceiling == 16 * 1024 ** 3 - HEADROOM_BYTES


@pytest.mark.parametrize("kind", ["mps", "xpu"])
def test_half_is_enabled_on_accelerators_other_than_cuda(monkeypatch, kind):
    _pretend_backend(monkeypatch, kind, 16 * 1024 ** 3)
    up = Upscaler(fake_model(half=True), tile=256, overlap=0, device=kind, half=True)
    assert up.half is True


def test_half_is_never_enabled_on_cpu():
    up = Upscaler(fake_model(half=True), tile=256, overlap=0, device="cpu", half=True)
    assert up.half is False


@pytest.mark.parametrize("kind", ["mps", "xpu"])
def test_vram_budget_applies_on_accelerators_other_than_cuda(monkeypatch, kind):
    _pretend_backend(monkeypatch, kind, 16 * 1024 ** 3)
    budget = 512 * 1024 ** 2
    up = Upscaler(fake_model(), tile=256, overlap=0, device=kind, half=False,
                  vram_budget=budget)
    assert up.runner.vram_ceiling == budget


def test_process_skips_the_pressure_check_when_no_peak_is_reportable(monkeypatch, rng):
    """A backend with no peak counter must not be read as 'peak = 0'."""
    from enhancer import upscale as upscale_module

    monkeypatch.setattr(upscale_module, "peak_memory_bytes", lambda device: None)
    checked = []
    frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    up = Upscaler(fake_model(scale=2), tile=16, overlap=4, device="cpu")
    monkeypatch.setattr(up.runner, "_check_pressure", lambda peak: checked.append(peak))
    up.process(frame)
    # run() always makes its own no-op call with peak_bytes=None; what must
    # not happen is a live check reporting a peak of zero.
    assert all(peak is None for peak in checked)
