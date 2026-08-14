# Core Engine Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, OOM-proof, zero-disk-cache image and video upscaling engine plus a benchmark harness that validates the throughput targets in spec §1.1 on the real RTX 3060 Laptop.

**Architecture:** A pure-Python package with no GUI dependency. `vram.py` owns tiling and out-of-memory recovery; `models.py` owns weight acquisition and spandrel-based loading; `video_io.py` wraps ffmpeg subprocesses over rawvideo pipes so frames never touch disk; `upscale.py` composes them into a frame processor with a CPU fallback; `bench.py` measures it. Every module is importable and testable without a GPU — CUDA-only paths are dependency-injected so they can be faked in tests.

**Tech Stack:** Python 3.12, PyTorch 2.x + CUDA 12.x, spandrel, numpy, ffmpeg/ffprobe (external binaries), pytest.

**Spec reference:** `docs/superpowers/specs/2026-08-14-local-video-upscaler-design.md` (revision 2).

---

## Design note: refinement of spec §8 tiling

Spec §8 specifies "feathered blending" between tiles. This plan implements a stricter and simpler
scheme: **context-padding with exact-crop reassembly.**

Each tile has two rectangles — a *core* rectangle and a *padded* rectangle. Core rectangles
partition the image exactly with no overlap. Padded rectangles extend the core by the overlap
amount (clamped at image edges) and are what gets fed to the model. After inference, only the
region corresponding to the core is cropped out and written to the output.

This is better than feathering on three counts: reassembly is bit-exact for an identity function at
*any* overlap value (which makes it directly unit-testable), there is no blend-induced softening at
tile boundaries, and no weight-map allocation is needed. The overlap still serves its real purpose —
giving the model receptive-field context so core pixels are computed as if the tile boundary were
not there.

### Why `is_oom_error` is wider than `torch.cuda.OutOfMemoryError`

The engine's contract is that it must never die from an out-of-memory error. Catching only
`torch.cuda.OutOfMemoryError` does not honor that contract: CUDA exhaustion routinely surfaces
through cuDNN/cuBLAS workspace allocation or driver-level failures as a plain `RuntimeError` with
a message like `"CUDNN_STATUS_ALLOC_FAILED"`, and the CPU fallback path can raise a plain
`MemoryError`. A `TileRunner` that only catches the typed exception will let any of these kill a
multi-hour render. `is_oom_error` therefore also pattern-matches known-recoverable `RuntimeError`
messages — but deliberately does **not** match `"illegal memory access"` or other corruption-class
CUDA errors, because those are not safe to retry: retrying them forever would silently hide a real
bug instead of surfacing it. Do not "simplify" this back to `except torch.cuda.OutOfMemoryError`;
that reintroduces the exact crash the module exists to prevent.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest config |
| `src/enhancer/__init__.py` | Package marker, version |
| `src/enhancer/vram.py` | Tile planning, tiled execution, OOM retry policy, device probing |
| `src/enhancer/models.py` | Model manifest, download + SHA-256 verification, spandrel loading, custom-dir scan |
| `src/enhancer/video_io.py` | ffprobe source analysis, ffmpeg decode/encode rawvideo pipes |
| `src/enhancer/upscale.py` | Frame upscaling with CPU fallback; composes vram + models |
| `src/enhancer/sources.py` | YouTube search and download via yt-dlp |
| `src/enhancer/bench.py` | Headless benchmark harness |
| `src/enhancer/cli.py` | Command-line entrypoint |
| `tests/test_vram.py` | Tiling, reassembly, OOM policy, device probe |
| `tests/test_models.py` | Manifest parsing, hash verification, custom scan |
| `tests/test_video_io.py` | ffprobe parsing, decode/encode round-trip |
| `tests/test_upscale.py` | Frame processing, CPU fallback |
| `tests/test_sources.py` | YouTube search parsing, format selection |
| `tests/conftest.py` | Shared fixtures, synthetic clip generation |

`analyze.py`, `restore.py`, `interpolate.py`, `pipeline.py`, and `gui.py` are out of scope for this
plan and are delivered by plans 2–4.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/enhancer/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

- [ ] **Step 1: Create the Python 3.12 virtual environment**

The system Python is 3.14.3, which has no PyTorch wheels. Install Python 3.12 first if absent
(`winget install Python.Python.3.12`), then:

```bash
py -3.12 -m venv .venv
```

- [ ] **Step 2: Activate and install dependencies**

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
```

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```bash
.venv/Scripts/python.exe -m pip install spandrel numpy opencv-python Pillow yt-dlp pytest pytest-cov
```

- [ ] **Step 3: Verify CUDA is visible**

Run:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: prints a torch version, `True`, and `NVIDIA GeForce RTX 3060 Laptop GPU`.
If it prints `False`, stop and resolve the CUDA install before continuing — every later task depends on it.

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "enhancer"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch",
    "spandrel",
    "numpy",
    "opencv-python",
    "Pillow",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: requires a CUDA device",
    "weights: requires downloaded model weights",
]
```

- [ ] **Step 5: Write `src/enhancer/__init__.py`**

```python
"""Local AI video and image enhancer."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
models/
out/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 7: Write `tests/conftest.py`**

```python
import subprocess
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_clip(tmp_path):
    """Generate a 2-second 320x240 25fps test clip with ffmpeg."""
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path
```

- [ ] **Step 8: Install the package in editable mode**

```bash
.venv/Scripts/python.exe -m pip install -e .
```

- [ ] **Step 9: Verify pytest collects**

Run:

```bash
.venv/Scripts/python.exe -m pytest -v
```

Expected: `no tests ran` with exit code 5. That is success — collection worked and there are no tests yet.

- [ ] **Step 10: Initialise git and commit**

```bash
git init && git add -A && git commit -m "chore: project scaffolding for enhancer core engine"
```

---

## Task 2: Tile planning

**Files:**
- Create: `src/enhancer/vram.py`
- Test: `tests/test_vram.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from enhancer.vram import Tile, plan_tiles


def test_single_tile_when_image_fits():
    tiles = plan_tiles(h=100, w=100, tile=256, overlap=16)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.y0, t.y1, t.x0, t.x1) == (0, 100, 0, 100)
    assert (t.py0, t.py1, t.px0, t.px1) == (0, 100, 0, 100)


def test_cores_partition_image_exactly():
    tiles = plan_tiles(h=300, w=500, tile=128, overlap=16)
    covered = set()
    for t in tiles:
        for y in range(t.y0, t.y1):
            for x in range(t.x0, t.x1):
                assert (y, x) not in covered, "cores must not overlap"
                covered.add((y, x))
    assert len(covered) == 300 * 500, "cores must cover every pixel"


def test_padding_extends_core_and_clamps_at_edges():
    tiles = plan_tiles(h=300, w=300, tile=128, overlap=16)
    for t in tiles:
        assert t.py0 == max(0, t.y0 - 16)
        assert t.py1 == min(300, t.y1 + 16)
        assert t.px0 == max(0, t.x0 - 16)
        assert t.px1 == min(300, t.x1 + 16)


def test_zero_overlap_means_padded_equals_core():
    tiles = plan_tiles(h=200, w=200, tile=64, overlap=0)
    for t in tiles:
        assert (t.py0, t.py1, t.px0, t.px1) == (t.y0, t.y1, t.x0, t.x1)


def test_rejects_nonpositive_tile():
    with pytest.raises(ValueError):
        plan_tiles(h=10, w=10, tile=0, overlap=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhancer.vram'`

- [ ] **Step 3: Write minimal implementation**

All of `vram.py`'s imports are declared here, up front, even though `logging` and `torch` are not
yet used by this task's code. Tasks 3–5 only ever append new module-level definitions below
`plan_tiles`; they never add new `import` lines mid-file. (An earlier draft of this plan let later
tasks splice imports in wherever their new code landed, which produced a file with import
statements scattered after function definitions — a real lint violation. Front-loading the header
here avoids that permanently.)

```python
"""Tiling, tiled execution, and out-of-memory recovery."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)

InferFn = Callable[[torch.Tensor], torch.Tensor]


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/vram.py tests/test_vram.py && git commit -m "feat(vram): tile planning with context padding"
```

---

## Task 3: Tiled execution with exact-crop reassembly

**Files:**
- Modify: `src/enhancer/vram.py`
- Test: `tests/test_vram.py`

This task also introduces `is_oom_error` and `TileFloorReached`, ahead of the OOM retry policy
they're named for (Task 4). They have to land here because the output-buffer allocation below
needs both: attributing an allocation failure on the *full-size* output buffer to `TileFloorReached`
(instead of retrying at ever-smaller tile sizes that can never help, since that buffer's size
doesn't depend on tile size) is part of what makes this task's tiled execution correct. `run_tiled`
also validates that `fn` actually upscaled by `scale`, in both the multi-tile and single-tile paths
— without that, a scale mismatch (e.g. driving a 4x model with `scale=2`) slices successfully and
writes scrambled pixels with no error at all.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vram.py`:

```python
import torch
from enhancer.vram import is_oom_error, run_tiled


def test_identity_roundtrip_is_exact_at_zero_overlap():
    img = torch.rand(1, 3, 200, 300)
    out = run_tiled(lambda t: t, img, tile=64, overlap=0, scale=1)
    assert torch.equal(out, img)


def test_identity_roundtrip_is_exact_with_overlap():
    """Exact-crop reassembly must be lossless at any overlap."""
    img = torch.rand(1, 3, 200, 300)
    out = run_tiled(lambda t: t, img, tile=64, overlap=16, scale=1)
    assert torch.equal(out, img)


def test_scale_2x_produces_correct_shape():
    img = torch.rand(1, 3, 100, 150)
    out = run_tiled(
        lambda t: torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest"),
        img, tile=32, overlap=8, scale=2,
    )
    assert out.shape == (1, 3, 200, 300)


def test_nearest_upscale_matches_untiled_result():
    img = torch.rand(1, 3, 64, 64)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")
    tiled = run_tiled(fn, img, tile=16, overlap=8, scale=2)
    assert torch.allclose(tiled, fn(img))


def test_single_tile_path_matches_direct_call():
    img = torch.rand(1, 3, 32, 32)
    out = run_tiled(lambda t: t * 2, img, tile=256, overlap=16, scale=1)
    assert torch.equal(out, img * 2)


def test_reassembly_is_exact_for_non_square_non_divisible_at_scale_4():
    """Bit-exact, not allclose — nearest-neighbor has no float accumulation,
    so any crop-offset bug shows up as a hard mismatch. Non-square,
    non-divisible-by-tile-size, scale > 1: the combination that actually
    exercises the offset arithmetic."""
    img = torch.rand(1, 3, 213, 377)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")
    tiled = run_tiled(fn, img, tile=64, overlap=16, scale=4)
    assert torch.equal(tiled, fn(img))


def test_scale_mismatch_raises_instead_of_corrupting():
    img = torch.rand(1, 3, 64, 64)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")
    with pytest.raises(ValueError):
        run_tiled(fn, img, tile=16, overlap=4, scale=2)


def test_single_tile_path_also_validates_scale():
    """The single-tile fast path must agree with the multi-tile path."""
    img = torch.rand(1, 3, 32, 32)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")
    with pytest.raises(ValueError):
        run_tiled(fn, img, tile=256, overlap=16, scale=2)


def test_scale_mismatch_on_edge_tile_raises():
    """A model that pads its input to a stride multiple returns a different
    effective scale on small edge tiles than on full-size interior tiles.
    _check_scale must run on every tile, not just the first, or this
    silently returns a wrong-shape, wrong-pixel result with no error."""
    def padding_model(t):
        h, w = t.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        t2 = torch.nn.functional.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return torch.nn.functional.interpolate(t2, scale_factor=2, mode="nearest")

    with pytest.raises(ValueError):
        run_tiled(padding_model, torch.rand(1, 3, 200, 200), tile=64, overlap=0, scale=2)


def test_zero_dimension_image_raises_value_error():
    img = torch.rand(1, 3, 0, 64)
    with pytest.raises(ValueError):
        run_tiled(lambda t: t, img, tile=32, overlap=0, scale=1)


def test_nonpositive_scale_raises_value_error():
    img = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError):
        run_tiled(lambda t: t, img, tile=32, overlap=0, scale=0)


def test_is_oom_error_accepts_plain_runtime_error_variants():
    assert is_oom_error(RuntimeError("CUDA error: out of memory"))
    assert is_oom_error(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"))
    assert is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert is_oom_error(MemoryError("cannot allocate"))
    assert is_oom_error(torch.cuda.OutOfMemoryError("CUDA out of memory"))


def test_is_oom_error_rejects_unrelated_runtime_error():
    """Retrying an illegal memory access forever would mask a real bug."""
    assert not is_oom_error(
        RuntimeError("CUDA error: an illegal memory access was encountered")
    )
    assert not is_oom_error(ValueError("nope"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v -k run_tiled or roundtrip or scale or nearest or single_tile or oom_error`
Expected: FAIL with `ImportError: cannot import name 'run_tiled'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/enhancer/vram.py` (below `plan_tiles`; no new import lines — they're all in the header
from Task 2):

```python
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

    Deliberately NOT a RuntimeError subclass: TileRunner (Task 4) widens its
    except clause to RuntimeError, and this signal must pass through it
    uncaught.
    """


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
        # Validated on EVERY tile, not just the first: edge tiles can be a
        # different size than interior tiles, and a model that pads its
        # input to a stride multiple can return a different effective scale
        # on them. Checking only tile #1 lets that slip through silently
        # (see test_scale_mismatch_on_edge_tile_raises).
        _check_scale(up, patch, scale)

        if out is None:
            out = _alloc_output(up, h, w, scale)

        # Offset of the core within the padded patch, in output pixels.
        # Do not change this arithmetic — it is verified correct at
        # non-square, non-divisible sizes and scale > 1 (see the
        # non_square_non_divisible test above).
        oy = (t.y0 - t.py0) * scale
        ox = (t.x0 - t.px0) * scale
        ch = (t.y1 - t.y0) * scale
        cw = (t.x1 - t.x0) * scale

        out[:, :, t.y0 * scale:t.y1 * scale, t.x0 * scale:t.x1 * scale] = (
            up[:, :, oy:oy + ch, ox:ox + cw]
        )

    assert out is not None
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v`
Expected: 18 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/vram.py tests/test_vram.py && git commit -m "feat(vram): tiled execution with exact-crop reassembly"
```

---

## Task 4: OOM retry policy

**Files:**
- Modify: `src/enhancer/vram.py`
- Test: `tests/test_vram.py`

Implements spec §8 steps 2–4: on any recoverable allocation failure (as classified by
`is_oom_error`, Task 3 — not just the typed `torch.cuda.OutOfMemoryError`), step the tile down the
ladder, retry the same frame, stop at a floor, and probe back up after sustained success. The
recovery probe interval doubles after every failure at that tile size, capped at
`MAX_PROBE_INTERVAL`. A fixed interval would re-probe (and re-fail against) a tile size that
genuinely never fits every `recover_after` frames, forever, over a multi-hour render; the doubling
interval converges instead.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vram.py`:

```python
from enhancer.vram import TileFloorReached, TileRunner


def _oom_for_first(n_failures):
    """Return an infer fn that raises CUDA OOM for the first n calls."""
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        if calls["n"] <= n_failures:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return t

    fn.calls = calls
    return fn


def test_steps_down_ladder_on_oom_and_succeeds():
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    out = runner.run(_oom_for_first(1), img)
    assert torch.equal(out, img)
    assert runner.tile == 192, "the next rung down from 256 is 192, not 128"


def test_steps_down_repeatedly_until_success():
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    runner.run(_oom_for_first(3), img)
    assert runner.tile == 32


def test_raises_tile_floor_reached_when_floor_still_ooms():
    runner = TileRunner(tile=64, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    with pytest.raises(TileFloorReached):
        runner.run(_oom_for_first(99), img)


def test_recovers_tile_size_after_sustained_success():
    """recover_after=3 sets the initial probe interval to 3, but the one OOM
    below doubles it to 6, so it takes 5 more successes (6 total) — not 3 —
    before the tile steps back up."""
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32, recover_after=3)
    img = torch.rand(1, 3, 64, 64)
    runner.run(_oom_for_first(1), img)
    assert runner.tile == 192
    for _ in range(5):
        runner.run(lambda t: t, img)
    assert runner.tile == 256, "tile should step back up after sustained success"


def test_recovery_never_exceeds_starting_tile():
    runner = TileRunner(tile=128, overlap=0, scale=1, min_tile=32, recover_after=1)
    img = torch.rand(1, 3, 64, 64)
    for _ in range(10):
        runner.run(lambda t: t, img)
    assert runner.tile == 128


def test_step_up_reaches_max_tile_above_ladder_top():
    """max_tile above the top ladder rung must not strand recovery below it.

    _step_up only ever returns a ladder rung; when max_tile (2048) is above
    the top rung (1024), stepping up from 1024 must still land on max_tile,
    not get stuck returning 1024 unchanged forever."""
    runner = TileRunner(tile=2048, overlap=0, scale=1, min_tile=32, recover_after=1)
    img = torch.rand(1, 3, 64, 64)
    runner.run(_oom_for_first(1), img)
    assert runner.tile == 1024, "the top ladder rung below 2048"

    for _ in range(20):
        runner.run(lambda t: t, img)

    assert runner.tile == 2048, "should climb all the way back to max_tile, not stick at 1024"


def test_probe_interval_doubles_once_per_frame_not_per_retry():
    """One frame's descent through several ladder rungs is one probe-interval
    doubling, not one per retry. Otherwise a single transient spike that
    forces several step-downs inflates the recovery interval by 2**n instead
    of 2**1, making the runner recover orders of magnitude slower than
    recover_after implies."""
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        if calls["n"] <= 4:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return t

    runner = TileRunner(tile=1024, overlap=0, scale=1, min_tile=32, recover_after=5)
    img = torch.rand(1, 3, 64, 64)
    runner.run(fn, img)

    assert calls["n"] == 5, "sanity: 4 OOM retries then a success, all within one run() call"
    assert runner._probe_interval == 10, "doubled exactly once, not once per retry (2**4=80)"


def test_runner_recovers_from_plain_runtime_error_oom():
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: out of memory")
        return t

    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    out = runner.run(fn, img)
    assert torch.equal(out, img)
    assert calls["n"] == 2


def test_runner_propagates_non_oom_errors():
    def fn(t):
        raise ValueError("boom")

    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    with pytest.raises(ValueError):
        runner.run(fn, img)


def test_probe_interval_backs_off_so_oom_does_not_repeat_forever():
    threshold = 256
    recover_after = 5
    n_frames = 500
    runner = TileRunner(tile=1024, overlap=0, scale=1, min_tile=128, recover_after=recover_after)
    oom_count = {"n": 0}

    def fn(patch):
        if runner.tile > threshold:
            oom_count["n"] += 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return patch

    img = torch.rand(1, 3, 64, 64)  # smaller than any ladder rung: always single-tile
    for _ in range(n_frames):
        runner.run(fn, img)

    linear_bound = n_frames // recover_after
    assert oom_count["n"] < linear_bound, (
        f"{oom_count['n']} OOMs over {n_frames} frames is not far fewer than "
        f"the fixed-interval bound of {linear_bound}"
    )
    assert oom_count["n"] < 20, "expected logarithmic growth, not linear"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v -k TileRunner or oom or floor or recover`
Expected: FAIL with `ImportError: cannot import name 'TileRunner'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/enhancer/vram.py` (no new import lines — `TileFloorReached` and `is_oom_error` already
exist from Task 3):

```python
# Descending ladder of candidate tile sizes. Used for stepping in BOTH
# directions so that shrink and recover are symmetric.
TILE_LADDER: tuple[int, ...] = (1024, 768, 512, 384, 256, 192, 128)

# Upper bound on the success-probe backoff, so probes stay rare but never stop.
MAX_PROBE_INTERVAL: int = 8192


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
        # No ladder rung is above `tile`, but max_tile can still be — e.g.
        # TileRunner(tile=2048, ...) after one OOM drops to the ladder top
        # (1024). Returning `tile` unchanged here would strand recovery at
        # 1024 forever instead of climbing back to 2048.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v`
Expected: 28 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/vram.py tests/test_vram.py && git commit -m "feat(vram): OOM retry policy with ladder stepping and backing-off recovery"
```

---

## Task 5: Device probing and initial tile selection

**Files:**
- Modify: `src/enhancer/vram.py`
- Test: `tests/test_vram.py`

Implements spec §8 step 1, with the 1 GB default headroom. `TILE_LADDER` already exists (Task 4);
this task only adds `HEADROOM_BYTES` and the functions below.

`choose_tile`'s cost model uses the PADDED tile area (`tile + 2*overlap`), because that's what
actually gets allocated for inference, and scales by `scale**2`, because a 4x model produces 16x
the output pixels of a 1x model for the same input tile. Ignoring either factor under-provisions
the initial tile size and pushes more of the OOM-retry burden onto `TileRunner` than necessary.
`bytes_per_output_pixel` must therefore be a calibration constant that does NOT already bake in the
scale factor — `choose_tile` applies it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vram.py`:

```python
from enhancer.vram import HEADROOM_BYTES, TILE_LADDER, choose_tile, select_device


def test_headroom_default_is_one_gigabyte():
    assert HEADROOM_BYTES == 1024 ** 3


def test_ladder_is_descending():
    assert list(TILE_LADDER) == sorted(TILE_LADDER, reverse=True)


def test_picks_largest_tile_that_fits():
    # 3 GB usable after headroom, 64 bytes per output pixel.
    free = 4 * 1024 ** 3
    tile = choose_tile(free_bytes=free, bytes_per_output_pixel=64)
    budget = free - HEADROOM_BYTES
    assert tile * tile * 64 <= budget
    bigger = [t for t in TILE_LADDER if t > tile]
    for b in bigger:
        assert b * b * 64 > budget, "should have picked the largest fitting tile"


def test_falls_back_to_smallest_tile_when_nothing_fits():
    tile = choose_tile(free_bytes=HEADROOM_BYTES + 1, bytes_per_output_pixel=10 ** 9)
    assert tile == TILE_LADDER[-1]


def test_zero_free_memory_returns_smallest_tile():
    assert choose_tile(free_bytes=0, bytes_per_output_pixel=64) == TILE_LADDER[-1]


def test_choose_tile_accounts_for_overlap():
    """The padded (tile + 2*overlap) area is what's actually allocated."""
    bytes_per_output_pixel = 1
    budget = 100_000
    free = budget + HEADROOM_BYTES

    no_overlap = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, overlap=0)
    with_overlap = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, overlap=32)

    assert no_overlap == 256
    assert with_overlap < no_overlap

    padded = with_overlap + 2 * 32
    assert padded * padded * bytes_per_output_pixel <= budget


def test_choose_tile_accounts_for_scale():
    """A 4x model produces 16x the pixels, so it must pick a smaller tile."""
    free = HEADROOM_BYTES + 50 * 1024 * 1024
    bytes_per_output_pixel = 64

    t1 = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, scale=1)
    t4 = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, scale=4)

    assert t4 < t1


def test_select_device_returns_cpu_when_not_preferring_cuda():
    assert select_device(prefer_cuda=False) == torch.device("cpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v -k headroom or choose or fits or select_device`
Expected: FAIL with `ImportError: cannot import name 'HEADROOM_BYTES'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/enhancer/vram.py` (no new import lines):

```python
# Reserved VRAM. A laptop GPU also drives the desktop compositor under WDDM,
# so this is deliberately generous.
HEADROOM_BYTES: int = 1024 ** 3


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vram.py -v`
Expected: 36 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/vram.py tests/test_vram.py && git commit -m "feat(vram): device probing and initial tile selection"
```

---

## Task 6: Model manifest and hash verification

**Files:**
- Create: `src/enhancer/models.py`
- Create: `src/enhancer/manifest.json`
- Test: `tests/test_models.py`

Implements spec §12. Note the spec's hard requirement: **SHA-256 digests must never be invented.**
The manifest created here ships with an empty model list; Task 7 populates it from real downloads.

- [ ] **Step 1: Write the failing test**

```python
import hashlib
import json

import pytest

from enhancer.models import ModelEntry, ModelRegistry, VerificationError, sha256_file


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"indian cinema" * 1000)
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_sha256_file_handles_empty_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_entry_parses_from_dict():
    entry = ModelEntry.from_dict(
        {
            "id": "span-2x",
            "tier": "fast",
            "arch": "SPAN",
            "scale": 2,
            "url": "https://example.invalid/span.pth",
            "sha256": "a" * 64,
            "size": 8_000_000,
        }
    )
    assert entry.id == "span-2x"
    assert entry.scale == 2
    assert entry.tier == "fast"


def test_entry_rejects_malformed_sha256():
    with pytest.raises(ValueError, match="sha256"):
        ModelEntry.from_dict(
            {
                "id": "bad",
                "tier": "fast",
                "arch": "SPAN",
                "scale": 2,
                "url": "https://example.invalid/x.pth",
                "sha256": "tooshort",
                "size": 1,
            }
        )


def test_verify_accepts_matching_file(tmp_path):
    p = tmp_path / "m.pth"
    p.write_bytes(b"weights")
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(b"weights").hexdigest(), size=7,
    )
    entry.verify(p)  # must not raise


def test_verify_rejects_corrupted_file(tmp_path):
    p = tmp_path / "m.pth"
    p.write_bytes(b"corrupted")
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(b"weights").hexdigest(), size=7,
    )
    with pytest.raises(VerificationError):
        entry.verify(p)


def test_registry_loads_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [
        {
            "id": "span-2x", "tier": "fast", "arch": "SPAN", "scale": 2,
            "url": "https://example.invalid/span.pth",
            "sha256": "a" * 64, "size": 8_000_000,
        }
    ]}))
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")
    assert [e.id for e in reg.list()] == ["span-2x"]


def test_registry_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    dup = {
        "id": "same", "tier": "fast", "arch": "SPAN", "scale": 2,
        "url": "https://example.invalid/a.pth", "sha256": "a" * 64, "size": 1,
    }
    manifest.write_text(json.dumps({"models": [dup, dup]}))
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhancer.models'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Model manifest, acquisition, verification, and loading."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1024 * 1024


class VerificationError(RuntimeError):
    """A downloaded file did not match its expected digest."""


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file, safe for multi-gigabyte weights."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelEntry:
    id: str
    tier: str
    arch: str
    scale: int
    url: str
    sha256: str
    size: int

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        digest = str(d["sha256"]).lower()
        if not _SHA256_RE.match(digest):
            raise ValueError(
                f"model {d.get('id')!r} has a malformed sha256: {d['sha256']!r}"
            )
        return cls(
            id=str(d["id"]),
            tier=str(d["tier"]),
            arch=str(d["arch"]),
            scale=int(d["scale"]),
            url=str(d["url"]),
            sha256=digest,
            size=int(d["size"]),
        )

    def verify(self, path: Path) -> None:
        actual = sha256_file(path)
        if actual != self.sha256:
            raise VerificationError(
                f"{self.id}: expected sha256 {self.sha256}, got {actual}. "
                f"The file at {path} is corrupt or the source changed."
            )


class ModelRegistry:
    """Manifest-backed model catalogue plus a custom drop-in directory."""

    def __init__(self, manifest_path: Path, cache_dir: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        data = json.loads(self.manifest_path.read_text())
        entries = [ModelEntry.from_dict(d) for d in data["models"]]

        seen: set[str] = set()
        for e in entries:
            if e.id in seen:
                raise ValueError(f"duplicate model id in manifest: {e.id!r}")
            seen.add(e.id)

        self._entries = entries

    def list(self) -> list[ModelEntry]:
        return list(self._entries)

    def get(self, model_id: str) -> ModelEntry:
        for e in self._entries:
            if e.id == model_id:
                return e
        raise KeyError(f"unknown model id: {model_id!r}")

    def path_for(self, entry: ModelEntry) -> Path:
        return self.cache_dir / f"{entry.id}.pth"
```

- [ ] **Step 4: Create the empty manifest**

`src/enhancer/manifest.json`:

```json
{
  "_comment": "SHA-256 digests MUST be computed from real downloads (spec §12). Never invent them.",
  "models": []
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/enhancer/models.py src/enhancer/manifest.json tests/test_models.py && git commit -m "feat(models): manifest schema and SHA-256 verification"
```

---

## Task 7: Model download and custom-directory scan

**Files:**
- Modify: `src/enhancer/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
import hashlib

from enhancer.models import ModelEntry, ModelRegistry, VerificationError, scan_custom_dir


def test_scan_custom_dir_finds_pth_files(tmp_path):
    d = tmp_path / "custom"
    d.mkdir()
    (d / "my_finetune.pth").write_bytes(b"x")
    (d / "other.safetensors").write_bytes(b"x")
    (d / "notes.txt").write_bytes(b"x")
    found = {p.name for p in scan_custom_dir(d)}
    assert found == {"my_finetune.pth", "other.safetensors"}


def test_scan_custom_dir_returns_empty_when_missing(tmp_path):
    assert scan_custom_dir(tmp_path / "nope") == []


def test_download_verifies_and_keeps_file(tmp_path, monkeypatch):
    payload = b"model-weights-payload"
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"models": []}')
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")

    monkeypatch.setattr(
        "enhancer.models._fetch", lambda url, dest, on_progress=None: dest.write_bytes(payload)
    )
    path = reg.ensure(entry)
    assert path.read_bytes() == payload


def test_download_deletes_file_on_hash_mismatch(tmp_path, monkeypatch):
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256="b" * 64, size=5,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"models": []}')
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")

    monkeypatch.setattr(
        "enhancer.models._fetch", lambda url, dest, on_progress=None: dest.write_bytes(b"wrong")
    )
    with pytest.raises(VerificationError):
        reg.ensure(entry)
    assert not reg.path_for(entry).exists(), "corrupt download must not be left behind"


def test_ensure_skips_download_when_file_already_valid(tmp_path, monkeypatch):
    payload = b"already-here"
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"models": []}')
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")
    reg.cache_dir.mkdir(parents=True, exist_ok=True)
    reg.path_for(entry).write_bytes(payload)

    def boom(*a, **k):
        raise AssertionError("should not re-download a valid cached file")

    monkeypatch.setattr("enhancer.models._fetch", boom)
    assert reg.ensure(entry).read_bytes() == payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v -k scan_custom or download or ensure`
Expected: FAIL with `ImportError: cannot import name 'scan_custom_dir'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/enhancer/models.py`:

```python
import urllib.request
from collections.abc import Callable, Iterable

ProgressFn = Callable[[int, int], None]

CUSTOM_SUFFIXES = {".pth", ".safetensors", ".ckpt"}


def scan_custom_dir(directory: Path) -> list[Path]:
    """Find drop-in weight files (spec §4.1). Missing directory is not an error."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in CUSTOM_SUFFIXES
    )


def _fetch(url: str, dest: Path, on_progress: ProgressFn | None = None) -> None:
    """Stream a URL to disk, reporting (bytes_done, bytes_total)."""
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as fh:
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
```

Add these methods to `ModelRegistry`:

```python
    def ensure(self, entry: ModelEntry, on_progress: ProgressFn | None = None) -> Path:
        """Return a verified local path, downloading if necessary."""
        dest = self.path_for(entry)
        if dest.exists():
            try:
                entry.verify(dest)
                return dest
            except VerificationError:
                dest.unlink()

        dest.parent.mkdir(parents=True, exist_ok=True)
        _fetch(entry.url, dest, on_progress)
        try:
            entry.verify(dest)
        except VerificationError:
            dest.unlink(missing_ok=True)
            raise
        return dest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/models.py tests/test_models.py && git commit -m "feat(models): verified download and custom-dir scan"
```

---

## Task 8: Spandrel model loading

**Files:**
- Modify: `src/enhancer/models.py`
- Test: `tests/test_models.py`

Implements spec §4.1 — architecture auto-detection so any OpenModelDB weight or user fine-tune
loads with no code change.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
from pathlib import Path

from enhancer.models import LoadedModel, load_model

WEIGHTS_DIR = Path(__file__).parent.parent / "models"


@pytest.mark.weights
def test_load_model_detects_scale_and_arch():
    candidates = list(WEIGHTS_DIR.glob("*.pth"))
    if not candidates:
        pytest.skip("no weights downloaded; run the model manager first")
    m = load_model(candidates[0], device="cpu", half=False)
    assert isinstance(m, LoadedModel)
    assert m.scale in (1, 2, 4, 8)
    assert m.arch


@pytest.mark.weights
def test_loaded_model_upscales_by_declared_scale():
    candidates = list(WEIGHTS_DIR.glob("*.pth"))
    if not candidates:
        pytest.skip("no weights downloaded; run the model manager first")
    m = load_model(candidates[0], device="cpu", half=False)
    out = m(torch.rand(1, 3, 32, 32))
    assert out.shape[-2:] == (32 * m.scale, 32 * m.scale)


def test_load_model_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pth", device="cpu", half=False)
```

Add `import torch` at the top of `tests/test_models.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v -k load_model`
Expected: FAIL with `ImportError: cannot import name 'LoadedModel'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/enhancer/models.py`:

```python
import torch
from spandrel import ImageModelDescriptor, ModelLoader


class LoadedModel:
    """A spandrel-loaded upscaler ready for inference."""

    def __init__(self, descriptor: ImageModelDescriptor, device: torch.device) -> None:
        self._d = descriptor
        self.device = device
        self.scale = int(descriptor.scale)
        self.arch = descriptor.architecture.name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self._d(x)


def load_model(path: Path, device: str | torch.device = "cuda", half: bool = True) -> LoadedModel:
    """Load any supported architecture from a bare state dict.

    spandrel deduces the architecture and hyperparameters, so no per-model code
    is required (spec §4.1).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model weights not found: {path}")

    device = torch.device(device)
    descriptor = ModelLoader().load_from_file(str(path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(f"{path.name} is not an image-to-image model")

    descriptor.to(device)
    if half and device.type == "cuda" and descriptor.supports_half:
        descriptor.model.half()
    descriptor.model.to(memory_format=torch.channels_last)
    descriptor.eval()
    return LoadedModel(descriptor, device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_models.py -v`
Expected: 14 passed, 2 skipped (the `weights` tests skip until Task 13 downloads real weights)

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/models.py tests/test_models.py && git commit -m "feat(models): spandrel loading with fp16 and channels_last"
```

---

## Task 9: Source analysis via ffprobe

**Files:**
- Create: `src/enhancer/video_io.py`
- Test: `tests/test_video_io.py`

Captures the fields spec §5.1 and §5.4 depend on. Plans 2–4 consume this; Plan 1 only needs it to
size buffers and preserve color metadata.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from enhancer.video_io import SourceProfile, _parse_probe, _parse_fraction


def test_parse_fraction_handles_rationals():
    assert _parse_fraction("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert _parse_fraction("25/1") == 25.0


def test_parse_fraction_handles_zero_denominator():
    assert _parse_fraction("0/0") == 0.0


def test_parse_probe_extracts_core_fields():
    raw = {
        "streams": [{
            "codec_type": "video", "width": 720, "height": 576,
            "r_frame_rate": "25/1", "avg_frame_rate": "25/1",
            "nb_frames": "500", "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "16:15", "field_order": "tt",
            "color_primaries": "bt470bg", "color_transfer": "bt709",
            "color_space": "bt470bg",
        }],
        "format": {"duration": "20.0"},
    }
    p = _parse_probe(raw)
    assert p.width == 720 and p.height == 576
    assert p.fps == 25.0
    assert p.frame_count == 500
    assert p.sar == "16:15"
    assert p.interlaced is True
    assert p.color_space == "bt470bg"


def test_parse_probe_marks_progressive():
    raw = {
        "streams": [{
            "codec_type": "video", "width": 1920, "height": 1080,
            "r_frame_rate": "24/1", "avg_frame_rate": "24/1",
            "nb_frames": "100", "pix_fmt": "yuv420p", "field_order": "progressive",
        }],
        "format": {"duration": "4.16"},
    }
    assert _parse_probe(raw).interlaced is False


def test_parse_probe_raises_without_video_stream():
    with pytest.raises(ValueError, match="no video stream"):
        _parse_probe({"streams": [{"codec_type": "audio"}], "format": {}})


def test_probe_real_clip(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    assert p.width == 320 and p.height == 240
    assert p.fps == pytest.approx(25.0)
    assert p.interlaced is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_video_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhancer.video_io'`

- [ ] **Step 3: Write minimal implementation**

```python
"""ffmpeg-backed source analysis and rawvideo streaming (spec §7.1)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

INTERLACED_FIELD_ORDERS = {"tt", "bb", "tb", "bt"}


def _parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    num, _, den = value.partition("/")
    try:
        d = float(den) if den else 1.0
        return float(num) / d if d else 0.0
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class SourceProfile:
    """Everything downstream stages need to know about a source."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    pix_fmt: str
    sar: str
    interlaced: bool
    field_order: str
    color_primaries: str
    color_transfer: str
    color_space: str
    duration: float

    @classmethod
    def probe(cls, path: str | Path) -> "SourceProfile":
        path = Path(path)
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        profile = _parse_probe(json.loads(out.stdout))
        return replace_path(profile, path)


def replace_path(profile: SourceProfile, path: Path) -> SourceProfile:
    from dataclasses import replace
    return replace(profile, path=path)


def _parse_probe(raw: dict) -> SourceProfile:
    video = next(
        (s for s in raw.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video is None:
        raise ValueError("no video stream found in source")

    fps = _parse_fraction(video.get("avg_frame_rate")) or _parse_fraction(
        video.get("r_frame_rate")
    )
    duration = float(raw.get("format", {}).get("duration") or 0.0)
    count = int(video.get("nb_frames") or 0)
    if count == 0 and fps and duration:
        count = int(round(fps * duration))

    field_order = str(video.get("field_order") or "progressive")

    return SourceProfile(
        path=Path("."),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        frame_count=count,
        pix_fmt=str(video.get("pix_fmt") or "yuv420p"),
        sar=str(video.get("sample_aspect_ratio") or "1:1"),
        interlaced=field_order in INTERLACED_FIELD_ORDERS,
        field_order=field_order,
        color_primaries=str(video.get("color_primaries") or ""),
        color_transfer=str(video.get("color_transfer") or ""),
        color_space=str(video.get("color_space") or ""),
        duration=duration,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_video_io.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/video_io.py tests/test_video_io.py && git commit -m "feat(video_io): ffprobe source analysis with color and field metadata"
```

---

## Task 10: Rawvideo decode and encode pipes

**Files:**
- Modify: `src/enhancer/video_io.py`
- Test: `tests/test_video_io.py`

Implements the zero-disk-cache requirement of spec §7.1 and the color/stream passthrough of §5.4.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_video_io.py`:

```python
import numpy as np

from enhancer.video_io import Decoder, Encoder


def test_decoder_yields_correct_shape_and_count(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p).frames())
    assert len(frames) == 50, "2 seconds at 25 fps"
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8


def test_decoder_frames_are_not_all_identical(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p).frames())
    assert not np.array_equal(frames[0], frames[-1])


def test_encoder_writes_playable_file(tmp_path, synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mp4"
    with Encoder(out, width=320, height=240, fps=25.0, source=p) as enc:
        for frame in Decoder(p).frames():
            enc.write(frame)
    assert out.exists() and out.stat().st_size > 0
    result = SourceProfile.probe(out)
    assert result.width == 320 and result.height == 240


def test_encode_roundtrip_preserves_frame_count(tmp_path, synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "out.mp4"
    with Encoder(out, width=320, height=240, fps=25.0, source=p) as enc:
        for frame in Decoder(p).frames():
            enc.write(frame)
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 50


def test_encoder_scales_output_dimensions(tmp_path, synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    out = tmp_path / "big.mp4"
    with Encoder(out, width=640, height=480, fps=25.0, source=p) as enc:
        for frame in Decoder(p).frames():
            enc.write(np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1))
    result = SourceProfile.probe(out)
    assert result.width == 640 and result.height == 480
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_video_io.py -v -k Decoder or Encoder or roundtrip`
Expected: FAIL with `ImportError: cannot import name 'Decoder'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/enhancer/video_io.py`:

```python
from collections.abc import Iterator

import numpy as np


class Decoder:
    """Streams RGB24 frames from ffmpeg over a pipe. Never touches disk."""

    def __init__(self, profile: SourceProfile) -> None:
        self.profile = profile

    def frames(self) -> Iterator[np.ndarray]:
        p = self.profile
        frame_bytes = p.width * p.height * 3
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin",
            "-i", str(p.path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(p.height, p.width, 3)
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()


class Encoder:
    """Streams RGB24 frames into ffmpeg. Preserves color metadata and side streams."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        source: SourceProfile,
        codec: str = "hevc_nvenc",
        quality: int = 20,
        bit_depth: int = 10,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.source = source
        self.codec = codec
        self.quality = quality
        self.bit_depth = bit_depth
        self._proc: subprocess.Popen | None = None

    def _build_command(self) -> list[str]:
        s = self.source
        pix_fmt = "p010le" if self.bit_depth == 10 else "yuv420p"
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "-",
            # Second input carries the original audio/subtitle/chapter streams.
            "-i", str(s.path),
            "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
            "-c:a", "copy", "-c:s", "copy",
            "-c:v", self.codec, "-cq", str(self.quality),
            "-pix_fmt", pix_fmt,
        ]
        # Carry color metadata through explicitly (spec §5.4).
        if s.color_primaries:
            cmd += ["-color_primaries", s.color_primaries]
        if s.color_transfer:
            cmd += ["-color_trc", s.color_transfer]
        if s.color_space:
            cmd += ["-colorspace", s.color_space]
        cmd.append(str(self.path))
        return cmd

    def __enter__(self) -> "Encoder":
        self._proc = subprocess.Popen(self._build_command(), stdin=subprocess.PIPE)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Encoder must be used as a context manager")
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def __exit__(self, *exc) -> None:
        if self._proc is None:
            return
        if self._proc.stdin:
            self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0 and exc[0] is None:
            raise RuntimeError(f"ffmpeg encoder exited with status {code}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_video_io.py -v`
Expected: 11 passed

If the NVENC tests fail with an encoder-not-found error, re-run with `codec="libx264"` to confirm
the pipe logic is correct, then investigate the NVENC build separately.

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/video_io.py tests/test_video_io.py && git commit -m "feat(video_io): rawvideo decode and encode pipes with metadata passthrough"
```

---

## Task 11: Frame upscaler with CPU fallback

**Files:**
- Create: `src/enhancer/upscale.py`
- Test: `tests/test_upscale.py`

Completes spec §8 step 3 — the per-frame CPU fallback that makes an OOM crash impossible.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_upscale.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhancer.upscale'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Frame upscaling with tiling and a per-frame CPU fallback (spec §8)."""

from __future__ import annotations

import logging

import numpy as np
import torch

from .vram import TileFloorReached, TileRunner

log = logging.getLogger(__name__)


def to_tensor(frame: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> (1, C, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).unsqueeze(0)
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
        self.runner = TileRunner(
            tile=tile, overlap=overlap, scale=model.scale, min_tile=128
        )
        self.cpu_fallback_count = 0

    def _infer(self, patch: torch.Tensor) -> torch.Tensor:
        if self.half:
            patch = patch.half()
        patch = patch.to(memory_format=torch.channels_last)
        return self.model(patch).float()

    def process(self, frame: np.ndarray) -> np.ndarray:
        img = to_tensor(frame).to(self.device)
        try:
            out = self.runner.run(self._infer, img)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_upscale.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/upscale.py tests/test_upscale.py && git commit -m "feat(upscale): frame processing with per-frame CPU fallback"
```

---

## Task 12: Benchmark harness

**Files:**
- Create: `src/enhancer/bench.py`
- Test: `tests/test_bench.py`

This is the deliverable that validates spec §1.1 on real hardware.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch

from enhancer.bench import BenchResult, benchmark


class FakeModel:
    scale = 2
    arch = "Fake"
    device = torch.device("cpu")

    def __call__(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    def to(self, device):
        return self


def test_benchmark_returns_result_with_positive_fps():
    r = benchmark(FakeModel(), width=64, height=64, frames=5, tile=64, overlap=0, device="cpu")
    assert isinstance(r, BenchResult)
    assert r.fps > 0
    assert r.frames == 5


def test_benchmark_records_output_dimensions():
    r = benchmark(FakeModel(), width=64, height=48, frames=2, tile=64, overlap=0, device="cpu")
    assert r.out_width == 128 and r.out_height == 96


def test_benchmark_estimates_feature_length_hours():
    r = benchmark(FakeModel(), width=32, height=32, frames=3, tile=32, overlap=0, device="cpu")
    expected = (172_800 / r.fps) / 3600
    assert r.feature_hours == pytest.approx(expected, rel=1e-6)
```

Add `import pytest` at the top of `tests/test_bench.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhancer.bench'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Headless benchmark harness validating the spec §1.1 throughput targets."""

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
            f"2h feature ≈ {self.feature_hours:.1f} h | "
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/bench.py tests/test_bench.py && git commit -m "feat(bench): headless throughput and VRAM benchmark harness"
```

---

## Task 13: CLI entrypoint and first real measurement

**Files:**
- Create: `src/enhancer/cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the CLI**

```python
"""Command-line entrypoint for the headless engine."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .bench import benchmark
from .models import load_model, scan_custom_dir
from .upscale import Upscaler
from .video_io import Decoder, Encoder, SourceProfile
from .vram import choose_tile, free_vram_bytes, select_device

DEFAULT_BYTES_PER_OUTPUT_PIXEL = 64


def _auto_tile(scale: int, overlap: int) -> int:
    free = free_vram_bytes()
    if free == 0:
        return 256
    return choose_tile(free, DEFAULT_BYTES_PER_OUTPUT_PIXEL, overlap=overlap, scale=scale)


def cmd_bench(args: argparse.Namespace) -> int:
    device = select_device(prefer_cuda=not args.cpu)
    model = load_model(Path(args.model), device=device, half=not args.cpu)
    tile = args.tile or _auto_tile(model.scale, args.overlap)
    result = benchmark(
        model, width=args.width, height=args.height, frames=args.frames,
        tile=tile, overlap=args.overlap, device=str(device), half=not args.cpu,
    )
    print(result.format())
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    device = select_device(prefer_cuda=not args.cpu)
    model = load_model(Path(args.model), device=device, half=not args.cpu)
    tile = args.tile or _auto_tile(model.scale, args.overlap)

    profile = SourceProfile.probe(args.input)
    up = Upscaler(model, tile=tile, overlap=args.overlap, device=device, half=not args.cpu)
    out_w = profile.width * model.scale
    out_h = profile.height * model.scale

    print(f"{profile.width}x{profile.height} -> {out_w}x{out_h}, "
          f"{profile.frame_count} frames, tile={tile}")

    with Encoder(args.output, out_w, out_h, profile.fps, profile) as enc:
        for i, frame in enumerate(Decoder(profile).frames(), 1):
            enc.write(up.process(frame))
            if i % 50 == 0:
                print(f"\r{i}/{profile.frame_count}", end="", flush=True)
    print(f"\nDone. CPU fallbacks: {up.cpu_fallback_count}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    found = scan_custom_dir(Path(args.dir))
    if not found:
        print(f"No weights found in {args.dir}")
        return 1
    for p in found:
        print(f"  {p.name}  ({p.stat().st_size / 1024 ** 2:.0f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="enhancer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="measure throughput and peak VRAM")
    b.add_argument("model")
    b.add_argument("--width", type=int, default=854)
    b.add_argument("--height", type=int, default=480)
    b.add_argument("--frames", type=int, default=30)
    b.add_argument("--tile", type=int, default=0)
    b.add_argument("--overlap", type=int, default=16)
    b.add_argument("--cpu", action="store_true")
    b.set_defaults(func=cmd_bench)

    v = sub.add_parser("video", help="upscale a video file")
    v.add_argument("model")
    v.add_argument("input")
    v.add_argument("output")
    v.add_argument("--tile", type=int, default=0)
    v.add_argument("--overlap", type=int, default=16)
    v.add_argument("--cpu", action="store_true")
    v.set_defaults(func=cmd_video)

    m = sub.add_parser("models", help="list drop-in weights")
    m.add_argument("--dir", default="models/custom")
    m.set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Register the console script**

Add to `pyproject.toml`:

```toml
[project.scripts]
enhancer = "enhancer.cli:main"
```

- [ ] **Step 3: Reinstall so the entrypoint registers**

```bash
.venv/Scripts/python.exe -m pip install -e .
```

- [ ] **Step 4: Download real weights and record their digests**

Create `models/custom/`, then download at least one Compact and one SPAN model from OpenModelDB
or the Real-ESRGAN releases page into it. Compute each digest:

```bash
.venv/Scripts/python.exe -c "from enhancer.models import sha256_file; from pathlib import Path; [print(p.name, sha256_file(p)) for p in Path('models/custom').glob('*.pth')]"
```

Record each `(id, url, sha256, size)` into `src/enhancer/manifest.json`. **Only digests computed
from a real download may be entered** (spec §12).

- [ ] **Step 5: Run the full test suite including the weights-gated tests**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass; the two `@pytest.mark.weights` tests now run instead of skipping.

- [ ] **Step 6: Measure the real numbers**

```bash
.venv/Scripts/python.exe -m enhancer.cli bench models/custom/<compact_model>.pth --width 854 --height 480 --frames 50
```

Record the reported fps and peak VRAM for each downloaded tier. Compare against spec §1.1:
Turbo ≥ 40 fps, Fast ≥ 25 fps, Balanced ≥ 12 fps.

**If a tier misses its target, stop and report the measured numbers before starting Plan 2.**
The remaining plans assume these targets are approximately achievable; if they are not, the tier
defaults in spec §4.2 need revising first.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(cli): headless bench and video commands"
```

---

## Task 14: YouTube search and download

**Files:**
- Create: `src/enhancer/sources.py`
- Modify: `src/enhancer/cli.py`
- Test: `tests/test_sources.py`

Implements spec §5.0. Uses yt-dlp as a **library**, not a subprocess, so search results are
structured data. No API key or Google account is required — `ytsearchN:` is a built-in extractor.

Network access is mocked in all tests; nothing here contacts YouTube during the test suite.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from enhancer.sources import (
    FORMAT_SELECTOR,
    SearchResult,
    _parse_entry,
    build_search_query,
    download,
    search,
)


def test_build_search_query_uses_ytsearch_extractor():
    assert build_search_query("kathakali dance", 5) == "ytsearch5:kathakali dance"


def test_build_search_query_rejects_nonpositive_count():
    with pytest.raises(ValueError):
        build_search_query("x", 0)


def test_parse_entry_extracts_fields():
    r = _parse_entry({
        "id": "abc123",
        "title": "Test Video",
        "duration": 754,
        "uploader": "Some Channel",
        "view_count": 12345,
        "url": "https://www.youtube.com/watch?v=abc123",
    })
    assert isinstance(r, SearchResult)
    assert r.video_id == "abc123"
    assert r.title == "Test Video"
    assert r.duration == 754
    assert r.uploader == "Some Channel"


def test_parse_entry_tolerates_missing_optional_fields():
    r = _parse_entry({"id": "xyz", "title": "Bare"})
    assert r.duration == 0
    assert r.uploader == ""
    assert r.view_count == 0
    assert r.url == "https://www.youtube.com/watch?v=xyz"


def test_parse_entry_requires_id():
    with pytest.raises(KeyError):
        _parse_entry({"title": "No ID"})


def test_duration_hms_formats_readably():
    assert _parse_entry({"id": "a", "title": "t", "duration": 754}).duration_hms == "12:34"
    assert _parse_entry({"id": "a", "title": "t", "duration": 3725}).duration_hms == "1:02:05"
    assert _parse_entry({"id": "a", "title": "t", "duration": 0}).duration_hms == "?"


def test_format_selector_prefers_vp9_av1_over_h264():
    """YouTube gives VP9/AV1 more bitrate at equal resolution (spec §5.0)."""
    assert "vp9" in FORMAT_SELECTOR.lower() or "av01" in FORMAT_SELECTOR.lower()
    assert "bestaudio" in FORMAT_SELECTOR


def test_search_returns_parsed_results(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, query, download):
            assert download is False, "search must never download"
            assert query == "ytsearch2:test"
            return {"entries": [
                {"id": "a", "title": "First", "duration": 60},
                {"id": "b", "title": "Second", "duration": 120},
            ]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    results = search("test", limit=2)
    assert [r.video_id for r in results] == ["a", "b"]


def test_search_skips_null_entries(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, query, download):
            return {"entries": [None, {"id": "a", "title": "Only"}, None]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    assert len(search("test", limit=3)) == 1


def test_download_returns_resolved_path(monkeypatch, tmp_path):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            assert download is True
            return {"id": "abc", "title": "Vid", "ext": "mp4",
                    "requested_downloads": [{"filepath": str(tmp_path / "Vid.mp4")}]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    out = download("https://www.youtube.com/watch?v=abc", tmp_path)
    assert out == tmp_path / "Vid.mp4"


def test_download_reports_progress(monkeypatch, tmp_path):
    seen = []

    class FakeYDL:
        def __init__(self, opts):
            self.hook = opts["progress_hooks"][0]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            self.hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
            return {"id": "abc", "ext": "mp4",
                    "requested_downloads": [{"filepath": str(tmp_path / "v.mp4")}]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    download("u", tmp_path, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(50, 100)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhancer.sources'`

- [ ] **Step 3: Write minimal implementation**

```python
"""YouTube search and download via yt-dlp (spec §5.0).

yt-dlp is used as a library rather than a subprocess so that search results are
structured data. The `ytsearchN:` extractor is built in, so no API key or Google
account is needed.

Downloaded files are ordinary sources: they flow into SourceProfile.probe()
unchanged and get no special-case handling downstream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL

ProgressFn = Callable[[int, int], None]

# Prefer VP9/AV1 over H.264 at equal resolution: YouTube allocates them more
# bitrate, which matters when the result is about to be upscaled.
FORMAT_SELECTOR = (
    "bestvideo[vcodec^=av01]+bestaudio/"
    "bestvideo[vcodec^=vp9]+bestaudio/"
    "bestvideo+bestaudio/best"
)

WATCH_URL = "https://www.youtube.com/watch?v={}"


@dataclass(frozen=True)
class SearchResult:
    video_id: str
    title: str
    duration: int
    uploader: str
    view_count: int
    url: str

    @property
    def duration_hms(self) -> str:
        if not self.duration:
            return "?"
        h, rem = divmod(self.duration, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_search_query(query: str, limit: int) -> str:
    """Build a yt-dlp ytsearch pseudo-URL."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return f"ytsearch{limit}:{query}"


def _parse_entry(entry: dict) -> SearchResult:
    video_id = entry["id"]
    return SearchResult(
        video_id=video_id,
        title=str(entry.get("title") or ""),
        duration=int(entry.get("duration") or 0),
        uploader=str(entry.get("uploader") or ""),
        view_count=int(entry.get("view_count") or 0),
        url=str(entry.get("url") or WATCH_URL.format(video_id)),
    )


def search(query: str, limit: int = 10) -> list[SearchResult]:
    """Search YouTube. Returns metadata only — never downloads."""
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(build_search_query(query, limit), download=False)
    return [_parse_entry(e) for e in (info.get("entries") or []) if e]


def download(
    url: str,
    dest_dir: str | Path,
    on_progress: ProgressFn | None = None,
    format_selector: str = FORMAT_SELECTOR,
) -> Path:
    """Download a video and return the local path to the muxed file."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            on_progress(int(d.get("downloaded_bytes") or 0), int(total))

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": format_selector,
        "merge_output_format": "mkv",
        "outtmpl": str(dest_dir / "%(title)s.%(ext)s"),
        "progress_hooks": [hook],
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    downloads = info.get("requested_downloads") or []
    if downloads and downloads[0].get("filepath"):
        return Path(downloads[0]["filepath"])
    return dest_dir / f"{info.get('title', info['id'])}.{info.get('ext', 'mkv')}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_sources.py -v`
Expected: 11 passed

- [ ] **Step 5: Add the CLI subcommands**

Add these two functions to `src/enhancer/cli.py`:

```python
def cmd_search(args: argparse.Namespace) -> int:
    from .sources import search

    results = search(args.query, limit=args.limit)
    if not results:
        print("No results.")
        return 1
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r.title[:60]:60s}  {r.duration_hms:>9s}  {r.uploader[:24]}")
        print(f"    {r.url}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .sources import download

    def progress(done: int, total: int) -> None:
        if total:
            print(f"\r{done * 100 // total}%", end="", flush=True)

    path = download(args.url, args.dir, on_progress=progress)
    print(f"\nSaved to {path}")
    return 0
```

Register them inside `main()`, just before `args = parser.parse_args(argv)`:

```python
    s = sub.add_parser("search", help="search YouTube")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_search)

    f = sub.add_parser("fetch", help="download a YouTube video")
    f.add_argument("url")
    f.add_argument("--dir", default="downloads")
    f.set_defaults(func=cmd_fetch)
```

- [ ] **Step 6: Verify the CLI works end to end**

Run:

```bash
.venv/Scripts/python.exe -m enhancer.cli search "old tamil movie song" --limit 3
```

Expected: three numbered results with titles, durations, and URLs. This is the only step in the
plan that contacts the network; if it fails with an extractor error, update yt-dlp
(`pip install -U yt-dlp`) — YouTube changes break extractors regularly.

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/enhancer/sources.py tests/test_sources.py src/enhancer/cli.py && git commit -m "feat(sources): YouTube search and download via yt-dlp"
```

---

## Self-review notes

**Spec coverage for Plan 1's scope:**

| Spec section | Covered by |
|---|---|
| §1.1 throughput targets | Task 12 (harness), Task 13 step 6 (measurement) |
| §4.1 spandrel loading, custom dir | Tasks 7, 8 |
| §4.2 roster tiers | Task 13 step 4 (manifest population) |
| §5.1 source analysis fields | Task 9 |
| §5.4 color management, 10-bit, SAR | Tasks 9, 10 |
| §6 fp16, channels_last, inference_mode | Tasks 8, 11 |
| §7.1 zero-disk-cache streaming | Task 10 |
| §8 tiling, OOM retry, CPU fallback, 1 GB headroom | Tasks 2–5, 11 |
| §5.0 YouTube search and download | Task 14 |
| §12 manifest, verification, no invented hashes | Tasks 6, 7, 13 |

**Deferred to later plans (intentionally out of scope):** §3 texture doctrine, §4.3 interpolation,
§5.2/§5.3 restoration stages, §6 TensorRT opt-in, §7.2 segment preview, §7.3 dual-pass, §7.4
resume, §10 GUI, §13 fine-tuning docs.

**Known simplifications to revisit in later plans:**
- `DEFAULT_BYTES_PER_PIXEL = 64` in `cli.py` is a placeholder constant. Plan 4 should replace it
  with a measured per-model value taken from a calibration run.
- Two CUDA streams with pinned staging buffers (spec §7.1) are not implemented here; the current
  path is synchronous. Add in Plan 4 alongside the pipeline, and re-measure with the Task 12 harness.
