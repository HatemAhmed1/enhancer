# Frame Interpolation Implementation Plan (Plan 3 of 5)

> **Implementation note:** each task is a self-contained TDD unit — write the failing test, confirm it fails, implement, confirm it passes, commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase frame rate — 24→48, 24→60, 30→60 and arbitrary targets — without ghosting across the hard cuts that fast-choreography footage is full of.

**Architecture:** The frame-timing arithmetic, the scene-change detector, and the stream driver are pure functions of frame indices and pixel data, with no dependency on any neural network. They are built and tested first. The RIFE network is vendored separately and injected, so a weight-acquisition failure cannot block the rest of the plan.

**Tech Stack:** Python 3.12, PyTorch, vendored RIFE IFNet (MIT), pytest.

**Spec reference:** `docs/specs/2026-08-14-local-video-upscaler-design.md` §4.3.

---

## Design notes

### Resampling, not "insert N frames"

Naively inserting a fixed number of frames between each pair only handles integer ratios. 24→60 is 2.5x, which that approach cannot express.

Instead, treat it as resampling along a timeline. Output frame *k* exists at time `k / dst_fps`. Source frame *j* exists at `j / src_fps`. For each output frame, find the bracketing source frames and the fractional position `t` between them. RIFE 4.x accepts an arbitrary timestep, so any `t` in [0, 1] is synthesizable.

This handles every ratio uniformly, including non-integer ones, and it falls out that `t == 0` means "copy the source frame" — no inference needed, and no quality loss on frames that already exist.

### Scene-change detection is mandatory, not optional

Interpolating across a hard cut produces a visible ghost-morph between two unrelated shots. Indian cinema choreography is heavily cut, so this is not a rare edge case — it would be visible many times per song sequence.

When the frame-pair difference exceeds the threshold, the nearest source frame is duplicated instead of synthesizing. A duplicated frame at a cut is invisible; a morph across a cut is glaring.

### Why the frame count is knowable in advance

Unlike inverse telecine (Plan 2), where decimation happens inside ffmpeg and the output length cannot be predicted, interpolation's output length is fully determined by `src_fps`, `dst_fps` and the source frame count. That means segmented resume can be preserved: an output segment maps deterministically back to the source frames it needs.

Preserving resume matters most here, because interpolation is precisely when renders get longest.

---

## File structure

| File | Responsibility |
|---|---|
| `src/enhancer/timing.py` | Output-frame planning: timesteps, source brackets, segment mapping |
| `src/enhancer/scenes.py` | Frame-pair difference metric and cut detection |
| `src/enhancer/rife/__init__.py` | Vendored RIFE IFNet architecture and loader |
| `src/enhancer/interpolate.py` | `Interpolator` wrapping a flow model; stream driver |
| `src/enhancer/cli.py` | MODIFY: `--fps`, `--interpolate`, `--scene-threshold` |
| `tests/test_timing.py` | Ratio arithmetic, including non-integer |
| `tests/test_scenes.py` | Cut detection versus fast motion |
| `tests/test_interpolate.py` | Stream driver against an injected fake model |

---

## Task 1: Output frame planning

**Files:**
- Create: `src/enhancer/timing.py`
- Test: `tests/test_timing.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from enhancer.timing import OutputFrame, output_frame_count, plan_output_frames


def test_same_fps_is_a_pure_passthrough():
    plan = plan_output_frames(src_fps=24, dst_fps=24, src_count=10)
    assert len(plan) == 10
    assert all(f.t == 0.0 for f in plan)
    assert [f.left for f in plan] == list(range(10))


def test_doubling_produces_alternating_real_and_synthetic_frames():
    plan = plan_output_frames(src_fps=24, dst_fps=48, src_count=4)
    assert [f.t for f in plan[:4]] == [0.0, 0.5, 0.0, 0.5]


def test_doubling_brackets_are_correct():
    plan = plan_output_frames(src_fps=24, dst_fps=48, src_count=4)
    assert (plan[1].left, plan[1].right) == (0, 1)
    assert (plan[2].left, plan[2].right) == (1, 1)


def test_non_integer_ratio_24_to_60():
    """2.5x cannot be expressed as 'insert N frames'."""
    plan = plan_output_frames(src_fps=24, dst_fps=60, src_count=4)
    assert len(plan) == 10
    expected = [0.0, 0.4, 0.8, 0.2, 0.6, 0.0, 0.4, 0.8, 0.2, 0.6]
    assert [round(f.t, 6) for f in plan] == expected


def test_ntsc_rates_do_not_drift():
    plan = plan_output_frames(src_fps=30000 / 1001, dst_fps=60000 / 1001, src_count=100)
    assert len(plan) == 200
    assert all(round(f.t, 6) in (0.0, 0.5) for f in plan)


def test_every_frame_references_valid_source_indices():
    plan = plan_output_frames(src_fps=25, dst_fps=60, src_count=50)
    for f in plan:
        assert 0 <= f.left < 50
        assert 0 <= f.right < 50
        assert f.right >= f.left


def test_t_is_always_within_unit_interval():
    plan = plan_output_frames(src_fps=23.976, dst_fps=59.94, src_count=30)
    assert all(0.0 <= f.t < 1.0 for f in plan)


def test_exact_source_frames_are_marked_as_copies():
    plan = plan_output_frames(src_fps=24, dst_fps=48, src_count=3)
    assert plan[0].is_copy
    assert not plan[1].is_copy


def test_last_source_frame_never_brackets_past_the_end():
    plan = plan_output_frames(src_fps=24, dst_fps=60, src_count=3)
    assert plan[-1].right <= 2


def test_output_frame_count_matches_the_plan_length():
    for dst in (24, 30, 48, 50, 60, 120):
        assert output_frame_count(24, dst, 100) == len(
            plan_output_frames(24, dst, 100)
        )


def test_slowing_down_is_rejected():
    with pytest.raises(ValueError, match="target frame rate"):
        plan_output_frames(src_fps=60, dst_fps=30, src_count=10)


def test_nonpositive_rates_are_rejected():
    with pytest.raises(ValueError):
        plan_output_frames(src_fps=0, dst_fps=60, src_count=10)


def test_empty_source_produces_an_empty_plan():
    assert plan_output_frames(src_fps=24, dst_fps=48, src_count=0) == []
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.timing'`**

- [ ] **Step 3: Implement**

```python
"""Output-frame timing for frame-rate conversion."""

from __future__ import annotations

from dataclasses import dataclass

# Positions closer than this to a source frame are treated as exact copies.
COPY_EPSILON = 1e-6


@dataclass(frozen=True)
class OutputFrame:
    """One frame of the output, located between two source frames.

    `t` is the fractional position from `left` toward `right`. A `t` of zero
    means the output frame coincides with `left` and can be copied rather than
    synthesized.
    """

    index: int
    left: int
    right: int
    t: float

    @property
    def is_copy(self) -> bool:
        return self.t <= COPY_EPSILON


def output_frame_count(src_fps: float, dst_fps: float, src_count: int) -> int:
    """Number of output frames produced by a conversion."""
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"frame rates must be positive, got {src_fps} and {dst_fps}")
    if src_count <= 0:
        return 0
    duration = src_count / src_fps
    return max(1, int(round(duration * dst_fps)))


def plan_output_frames(
    src_fps: float, dst_fps: float, src_count: int
) -> list[OutputFrame]:
    """Map each output frame onto a bracketing pair of source frames.

    Works for any ratio, integer or not, because each output frame is placed by
    time rather than by counting insertions.
    """
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"frame rates must be positive, got {src_fps} and {dst_fps}")
    if dst_fps < src_fps:
        raise ValueError(
            f"target frame rate {dst_fps} is below the source rate {src_fps}; "
            f"this tool interpolates but does not decimate"
        )
    if src_count <= 0:
        return []

    total = output_frame_count(src_fps, dst_fps, src_count)
    ratio = src_fps / dst_fps
    last = src_count - 1

    plan: list[OutputFrame] = []
    for k in range(total):
        position = k * ratio
        left = min(int(position), last)
        t = position - left
        if t <= COPY_EPSILON:
            t = 0.0
        right = min(left + 1, last)
        if right == left:
            t = 0.0
        plan.append(OutputFrame(index=k, left=left, right=right, t=t))
    return plan
```

- [ ] **Step 4: Run, confirm 13 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/timing.py tests/test_timing.py && git commit -m "feat(timing): output frame planning for arbitrary frame-rate ratios"
```

---

## Task 2: Scene-change detection

**Files:**
- Create: `src/enhancer/scenes.py`
- Test: `tests/test_scenes.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from enhancer.scenes import DEFAULT_THRESHOLD, frame_difference, is_scene_change


def _frame(value, size=64):
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_identical_frames_have_zero_difference():
    f = _frame(120)
    assert frame_difference(f, f) == 0.0


def test_difference_is_symmetric():
    a, b = _frame(50), _frame(200)
    assert frame_difference(a, b) == pytest.approx(frame_difference(b, a))


def test_difference_is_normalised_to_unit_range():
    assert 0.0 <= frame_difference(_frame(0), _frame(255)) <= 1.0


def test_black_to_white_is_near_maximum():
    assert frame_difference(_frame(0), _frame(255)) > 0.9


def test_hard_cut_is_detected():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 60, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(195, 256, (64, 64, 3), dtype=np.uint8)
    assert is_scene_change(a, b)


def test_gentle_pan_is_not_a_scene_change():
    """A shifted image is high-motion but the same shot."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    a = base[:, :64]
    b = base[:, 4:68]
    assert not is_scene_change(a, b)


def test_small_brightness_change_is_not_a_scene_change():
    assert not is_scene_change(_frame(120), _frame(128))


def test_threshold_is_adjustable():
    a, b = _frame(120), _frame(150)
    assert not is_scene_change(a, b, threshold=0.9)
    assert is_scene_change(a, b, threshold=0.01)


def test_default_threshold_is_in_a_sane_range():
    assert 0.1 <= DEFAULT_THRESHOLD <= 0.6


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        frame_difference(_frame(0, 32), _frame(0, 64))
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.scenes'`**

- [ ] **Step 3: Implement**

```python
"""Cut detection, so interpolation never morphs across a shot change."""

from __future__ import annotations

import numpy as np

# Chosen so genuine cuts fire while fast pans and lighting changes do not.
DEFAULT_THRESHOLD = 0.30

HISTOGRAM_BINS = 32


def _histogram(frame: np.ndarray) -> np.ndarray:
    """Normalised per-channel intensity histogram."""
    parts = [
        np.histogram(frame[..., c], bins=HISTOGRAM_BINS, range=(0, 256))[0]
        for c in range(frame.shape[2])
    ]
    hist = np.concatenate(parts).astype(np.float64)
    total = hist.sum()
    return hist / total if total else hist


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Dissimilarity between two frames, in the range 0 to 1.

    Uses histogram intersection rather than per-pixel difference. A camera pan
    moves every pixel while leaving the distribution of colours largely intact,
    so a pixel metric would flag it as a cut. A genuine shot change alters the
    distribution itself.
    """
    if a.shape != b.shape:
        raise ValueError(f"frames must have the same shape, got {a.shape} and {b.shape}")
    intersection = np.minimum(_histogram(a), _histogram(b)).sum()
    return float(1.0 - intersection)


def is_scene_change(
    a: np.ndarray, b: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> bool:
    """True when the two frames belong to different shots."""
    return frame_difference(a, b) > threshold
```

- [ ] **Step 4: Run, confirm 10 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/scenes.py tests/test_scenes.py && git commit -m "feat(scenes): histogram-based cut detection"
```

---

## Task 3: Interpolation stream driver

**Files:**
- Create: `src/enhancer/interpolate.py`
- Test: `tests/test_interpolate.py`

Built and tested against an injected fake flow model, so it does not depend on RIFE weights.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from enhancer.interpolate import Interpolator, interpolate_stream


class FakeFlow:
    """Linear cross-fade standing in for a real flow model."""

    def __init__(self):
        self.calls = []

    def __call__(self, a, b, t):
        self.calls.append(t)
        return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)


def _frames(n, size=32):
    return [np.full((size, size, 3), i * 10 % 256, dtype=np.uint8) for i in range(n)]


def test_same_fps_copies_every_frame_without_inference():
    model = FakeFlow()
    out = list(interpolate_stream(_frames(5), src_fps=24, dst_fps=24, model=model))
    assert len(out) == 5
    assert model.calls == [], "no synthesis should happen at 1:1"


def test_doubling_produces_twice_as_many_frames():
    out = list(interpolate_stream(_frames(5), src_fps=24, dst_fps=48, model=FakeFlow()))
    assert len(out) == 10


def test_non_integer_ratio_produces_the_planned_count():
    out = list(interpolate_stream(_frames(4), src_fps=24, dst_fps=60, model=FakeFlow()))
    assert len(out) == 10


def test_copied_frames_are_bit_identical_to_the_source():
    src = _frames(4)
    out = list(interpolate_stream(src, src_fps=24, dst_fps=48, model=FakeFlow()))
    assert np.array_equal(out[0], src[0])
    assert np.array_equal(out[2], src[1])


def test_synthesised_frames_lie_between_their_neighbours():
    src = [np.full((8, 8, 3), 0, dtype=np.uint8), np.full((8, 8, 3), 200, dtype=np.uint8)]
    out = list(interpolate_stream(src, src_fps=24, dst_fps=48, model=FakeFlow()))
    assert 0 < int(out[1].mean()) < 200


def test_scene_change_duplicates_instead_of_synthesising():
    """A morph across a cut is glaring; a duplicated frame is invisible."""
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    model = FakeFlow()
    out = list(interpolate_stream([a, b], src_fps=24, dst_fps=48, model=model))
    assert model.calls == [], "must not synthesise across a cut"
    assert np.array_equal(out[1], a)


def test_scene_detection_can_be_disabled():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    model = FakeFlow()
    list(interpolate_stream([a, b], src_fps=24, dst_fps=48, model=model, scene_threshold=None))
    assert model.calls, "disabling detection must allow synthesis"


def test_output_dtype_and_shape_are_preserved():
    out = list(interpolate_stream(_frames(3), src_fps=24, dst_fps=48, model=FakeFlow()))
    assert out[0].dtype == np.uint8
    assert out[0].shape == (32, 32, 3)


def test_single_frame_source_yields_that_frame():
    src = _frames(1)
    out = list(interpolate_stream(src, src_fps=24, dst_fps=48, model=FakeFlow()))
    assert len(out) == 1
    assert np.array_equal(out[0], src[0])


def test_empty_source_yields_nothing():
    assert list(interpolate_stream([], src_fps=24, dst_fps=48, model=FakeFlow())) == []


def test_model_receives_timesteps_in_the_unit_interval():
    model = FakeFlow()
    list(interpolate_stream(_frames(6), src_fps=24, dst_fps=60, model=model))
    assert all(0.0 < t < 1.0 for t in model.calls)


def test_interpolator_target_fps_from_multiplier():
    assert Interpolator.target_fps(src_fps=24, multiplier=2) == 48
    assert Interpolator.target_fps(src_fps=25, multiplier=4) == 100


def test_interpolator_rejects_multiplier_below_one():
    with pytest.raises(ValueError):
        Interpolator.target_fps(src_fps=24, multiplier=0)
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.interpolate'`**

- [ ] **Step 3: Implement**

```python
"""Frame-rate conversion driven by a flow model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .scenes import DEFAULT_THRESHOLD, is_scene_change
from .timing import plan_output_frames


class Interpolator:
    """Namespace for frame-rate helpers."""

    @staticmethod
    def target_fps(src_fps: float, multiplier: float) -> float:
        if multiplier < 1:
            raise ValueError(f"multiplier must be at least 1, got {multiplier}")
        return src_fps * multiplier


def interpolate_stream(
    frames: Iterable[np.ndarray],
    src_fps: float,
    dst_fps: float,
    model,
    scene_threshold: float | None = DEFAULT_THRESHOLD,
) -> Iterator[np.ndarray]:
    """Convert a frame sequence to `dst_fps`.

    `model` is any callable taking `(frame_a, frame_b, t)` and returning the
    synthesized intermediate. Injecting it keeps this driver testable without
    network weights.

    Frames are materialized because the plan may reference a source frame after
    the stream has advanced past it. Callers process one segment at a time, so
    the working set stays bounded.
    """
    source = list(frames)
    if not source:
        return

    plan = plan_output_frames(src_fps, dst_fps, len(source))

    # Cut detection is per source pair, not per output frame: the same pair
    # backs several output frames at high ratios, and the answer cannot change
    # between them.
    cuts: dict[tuple[int, int], bool] = {}

    for entry in plan:
        if entry.is_copy or entry.left == entry.right:
            yield source[entry.left]
            continue

        key = (entry.left, entry.right)
        if key not in cuts:
            cuts[key] = (
                scene_threshold is not None
                and is_scene_change(source[entry.left], source[entry.right], scene_threshold)
            )

        if cuts[key]:
            # Duplicate the nearer real frame rather than morphing between shots.
            yield source[entry.left if entry.t < 0.5 else entry.right]
        else:
            yield model(source[entry.left], source[entry.right], entry.t)
```

- [ ] **Step 4: Run, confirm 13 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/interpolate.py tests/test_interpolate.py && git commit -m "feat(interpolate): frame-rate conversion stream driver"
```

---

## Task 4: RIFE network

**Files:**
- Create: `src/enhancer/rife/__init__.py`
- Create: `src/enhancer/rife/ifnet.py`
- Create: `src/enhancer/rife/NOTICE`
- Test: `tests/test_rife.py`

RIFE is not an image-to-image model, so spandrel cannot load it. The IFNet architecture must be vendored.

- [ ] **Step 1: Obtain the architecture and weights**

Practical-RIFE (https://github.com/hzwer/Practical-RIFE) is MIT licensed, so vendoring the network definition with attribution is permitted.

1. Fetch the IFNet definition for RIFE **4.25** (or 4.26) from Practical-RIFE's `train_log/RIFE_HDv3.py` / `IFNet_HDv3.py`. Copy the network classes only — not the training or inference harness.
2. Place them in `src/enhancer/rife/ifnet.py`, unmodified apart from import fixes.
3. Create `src/enhancer/rife/NOTICE` recording the upstream project, its MIT licence, the commit or release the code came from, and the date.
4. Find a **direct** download URL for the matching `flownet.pkl` weights. Try Practical-RIFE releases and HuggingFace mirrors. A Google Drive interstitial is not usable.

**If no direct weight URL can be found, that is an acceptable outcome.** Record exactly what was tried, document manual placement into `models/rife/`, and let the weight-dependent tests skip. Do not invent a URL and do not fabricate a SHA-256.

- [ ] **Step 2: Write the failing tests**

```python
from pathlib import Path

import numpy as np
import pytest
import torch

from enhancer.rife import RIFE_DIR, RifeModel, load_rife


def _weights():
    return sorted(RIFE_DIR.glob("*.pkl")) + sorted(RIFE_DIR.glob("*.pth"))


def test_load_rife_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rife(tmp_path / "absent.pkl", device="cpu")


@pytest.mark.weights
def test_rife_loads_and_reports_its_device():
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    assert isinstance(model, RifeModel)
    assert model.device.type == "cpu"


@pytest.mark.weights
def test_rife_synthesises_a_frame_of_the_right_shape():
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    out = model(a, b, 0.5)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


@pytest.mark.weights
def test_rife_midpoint_lies_between_its_neighbours():
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 200, dtype=np.uint8)
    assert 0 < int(model(a, b, 0.5).mean()) < 200


@pytest.mark.weights
def test_rife_handles_dimensions_not_divisible_by_32():
    """RIFE pads internally; the caller must get its original size back."""
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    a = np.zeros((70, 130, 3), dtype=np.uint8)
    b = np.full((70, 130, 3), 128, dtype=np.uint8)
    assert model(a, b, 0.5).shape == (70, 130, 3)
```

- [ ] **Step 3: Implement the loader**

`src/enhancer/rife/__init__.py`:

```python
"""RIFE frame interpolation.

The network definition in `ifnet.py` is vendored from Practical-RIFE
(https://github.com/hzwer/Practical-RIFE), MIT licensed. See NOTICE.

RIFE is not an image-to-image model, so spandrel cannot load it and the
architecture has to travel with the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ifnet import IFNet

RIFE_DIR = Path("models/rife")

# IFNet downsamples by 32, so inputs are padded up to a multiple of it.
ALIGNMENT = 32


class RifeModel:
    """Callable adapter matching the flow-model contract in `interpolate.py`."""

    def __init__(self, net: IFNet, device: torch.device) -> None:
        self.net = net
        self.device = device

    def __call__(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape[:2]
        ph = (-h) % ALIGNMENT
        pw = (-w) % ALIGNMENT

        def prepare(frame: np.ndarray) -> torch.Tensor:
            x = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            x = x.unsqueeze(0).float().div_(255.0).to(self.device)
            return F.pad(x, (0, pw, 0, ph), mode="replicate")

        with torch.inference_mode():
            out = self.net(prepare(a), prepare(b), timestep=t)

        out = out[:, :, :h, :w].clamp_(0.0, 1.0).mul_(255.0).round_()
        return out.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def load_rife(path: str | Path, device: str | torch.device = "cuda") -> RifeModel:
    """Load RIFE weights into the vendored IFNet."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RIFE weights not found: {path}")

    device = torch.device(device)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    net = IFNet()
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    return RifeModel(net, device)
```

- [ ] **Step 4: Run, confirm the loader test passes and the weight-gated tests either pass or skip**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/rife tests/test_rife.py && git commit -m "feat(rife): vendored IFNet architecture and weight loader"
```

---

## Task 5: Pipeline and CLI integration

**Files:**
- Modify: `src/enhancer/cli.py`
- Test: `tests/test_cli_resume.py`

Interpolation's output length is fully determined in advance, so segmented resume is preserved — unlike inverse telecine.

- [ ] **Step 1: Write the failing tests**

```python
def test_interpolated_render_produces_the_planned_frame_count(tmp_path, synthetic_clip):
    """50 source frames at 25 fps, doubled, is 100 output frames."""
    profile = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(),
    )
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 100


def test_interpolated_output_reports_the_target_frame_rate(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=tmp_path / "job", segment_frames=20, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(),
    )
    assert SourceProfile.probe(out).fps == pytest.approx(50.0, abs=0.1)


def test_interpolated_render_still_resumes(tmp_path, synthetic_clip):
    """Resume matters most here: interpolation is when renders get longest."""
    profile = SourceProfile.probe(synthetic_clip)
    job_dir = tmp_path / "job"

    class FailsLate(DoublingUpscaler):
        def __init__(self):
            self.seen = 0

        def process(self, frame):
            self.seen += 1
            if self.seen > 15:
                raise RuntimeError("simulated crash")
            return super().process(frame)

    with pytest.raises(RuntimeError, match="simulated crash"):
        render_resumable(
            profile, FailsLate(), tmp_path / "out.mkv",
            job_dir=job_dir, segment_frames=20, settings={"scale": 2},
            interpolate_to=50.0, flow_model=_CrossFade(),
        )
    assert segment_path(job_dir, 0).exists()

    out = tmp_path / "out.mkv"
    render_resumable(
        profile, DoublingUpscaler(), out,
        job_dir=job_dir, segment_frames=20, settings={"scale": 2},
        interpolate_to=50.0, flow_model=_CrossFade(),
    )
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 100
```

Add this helper near the top of the file:

```python
class _CrossFade:
    def __call__(self, a, b, t):
        return (a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t).astype(np.uint8)
```

- [ ] **Step 2: Run and confirm `TypeError: render_resumable() got an unexpected keyword argument 'interpolate_to'`**

- [ ] **Step 3: Implement**

Add `interpolate_to: float | None = None` and `flow_model=None` parameters to `render_resumable`.

When `interpolate_to` is set:

- The job's `total_frames` becomes `output_frame_count(profile.fps, interpolate_to, profile.frame_count)`, so segments are sized in OUTPUT frames.
- For each segment, take the plan slice `plan[start : start + count]`, then decode source frames from `min(left)` through `max(right)` inclusive.
- Feed those through the upscaler, then `interpolate_stream`, then texture work.
- Pass `fps=interpolate_to` to `write_segment`.

**Upscale before interpolating.** RIFE then blends already-detailed frames, and the upscaler never runs on synthesized content — the order in spec §4.3.

Add to the CLI:

```python
    v.add_argument("--fps", type=float, default=None,
                   help="target output frame rate, e.g. 60")
    v.add_argument("--interpolate", type=float, default=None,
                   help="frame rate multiplier, e.g. 2 (alternative to --fps)")
    v.add_argument("--scene-threshold", type=float, default=None,
                   help="cut sensitivity, 0-1; lower detects more cuts")
```

`--fps` and `--interpolate` are mutually exclusive; reject both together with a clear message. Include the resolved target rate and the scene threshold in the job settings hash.

Load RIFE only when interpolation is requested, so a missing weight file never blocks a plain upscale.

- [ ] **Step 4: Run the full suite**

- [ ] **Step 5: Real verification**

```bash
ffmpeg -y -loglevel error -f lavfi -i testsrc=size=854x480:rate=24:duration=5 -pix_fmt yuv420p out/interp_in.mp4
.venv/Scripts/python.exe -m enhancer.cli video models/custom/2xParimgCompact.pth out/interp_in.mp4 out/interp_out.mkv --fps 60
```

Confirm with ffprobe: 60 fps, 300 frames, correct dimensions. Report the actual output and the throughput cost of interpolation.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(cli): frame interpolation and target frame rate"
```

---

## Self-review notes

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| §4.3 RIFE 4.25 default | Task 4 |
| §4.3 multiplier and target-FPS modes | Tasks 1, 5 |
| §4.3 non-integer ratios via arbitrary timestep | Tasks 1, 3 |
| §4.3 mandatory scene-change detection | Tasks 2, 3 |
| §4.3 upscale-before-interpolate ordering | Task 5 |

**Known gaps to address later:**
- `interpolate_stream` materializes a segment's frames in memory. At 500-frame segments and 4K output that is significant; segment size may need lowering when interpolating, or the driver reworked to a sliding window.
- The configurable stage order in §4.3 (interpolate-before-upscale for speed on the Turbo tier) is not implemented; only the quality-first order is.
- Scene detection uses a global histogram. A cut between two similarly-lit shots of the same scene may be missed. A per-tile histogram would be more sensitive at some cost.
