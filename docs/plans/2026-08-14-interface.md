# Interface Implementation Plan (Plan 4 of 5)

> **Implementation note:** each task is a self-contained TDD unit — write the failing test, confirm it fails, implement, confirm it passes, commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A desktop window that makes the engine usable without the command line, and that makes a bad setting cost two minutes instead of six hours.

**Architecture:** Everything that can be decided without Qt is decided without Qt. A `RenderRequest` dataclass holds every user choice, validates itself, and knows how to become keyword arguments for `render_resumable` and a settings dict for the job hash. That object is pure data and carries the entire test burden. The Qt layer builds one, hands it to a worker thread, and renders progress signals — it contains no policy.

The engine never imports Qt; the GUI never touches CUDA directly. That boundary already exists and this plan preserves it.

**Tech Stack:** Python 3.12, PySide6 6.11, pytest, pytest-qt.

**Spec reference:** `docs/specs/2026-08-14-local-video-upscaler-design.md` §7.2, §10.

---

## Scope

**In:** drag-and-drop input, automatic source analysis, model selection, restoration and texture controls, interpolation controls, live progress with fps and ETA, cancel, resume detection, and **segment preview**.

**Out, deferred to Plan 5:** dual-pass 2K→4K and the A/B compare view. Both are worth having; neither is worth delaying a usable window for. Segment preview is kept because it is the feature that actually protects against wasted hours, and the dual-pass workflow was only ever a clumsier route to the same goal.

**No face-restoration GAN controls**, consistent with Plan 2. There is nothing to expose.

---

## Design notes

### Why cancellation works through the progress callback

`render_resumable` already calls `on_progress` once per frame. Raising from inside that callback aborts the render exactly where the existing machinery expects a failure: `write_segment` deletes its `.part` file, the job journal keeps every completed segment, and the next run resumes. Cancel therefore needs no new mechanism and cannot corrupt output — it is the crash path, deliberately taken.

### Why segment preview matters more than dual-pass

A 1080p→4K feature is ~17 hours. The expensive mistake is not a slow render, it is a slow render with the wrong settings. Rendering ten seconds takes under a minute and answers the only question that matters. Dual-pass 2K→4K, as originally specified, still requires a full 2K pass before you can judge anything.

Preview reuses the render path exactly — same upscaler, same restoration, same interpolation — over a frame range. Anything else would preview something other than what you are about to render.

---

## File structure

| File | Responsibility |
|---|---|
| `src/enhancer/requests.py` | `RenderRequest`: user choices, validation, conversion to engine arguments |
| `src/enhancer/gui.py` | Qt window, worker thread, progress rendering |
| `src/enhancer/cli.py` | MODIFY: `gui` subcommand, `preview` subcommand |
| `tests/test_requests.py` | Validation and argument mapping |
| `tests/test_gui.py` | Worker signals and cancellation, offscreen |

---

## Task 1: RenderRequest

**Files:**
- Create: `src/enhancer/requests.py`
- Test: `tests/test_requests.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from enhancer.requests import RenderRequest


def _req(**kw):
    base = dict(
        model=Path("models/custom/m.pth"),
        source=Path("in.mp4"),
        output=Path("out.mkv"),
    )
    base.update(kw)
    return RenderRequest(**base)


def test_defaults_match_the_command_line_defaults():
    r = _req()
    assert r.degrain == 0.25
    assert r.detail_retention == 0.25
    assert r.regrain == 0.6
    assert r.deblock == 0.0
    assert r.target_fps is None


def test_job_dir_defaults_beside_the_output():
    assert _req(output=Path("/tmp/a.mkv")).job_dir == Path("/tmp/a.job")


def test_explicit_job_dir_is_respected():
    r = _req(job_dir=Path("/tmp/elsewhere"))
    assert r.job_dir == Path("/tmp/elsewhere")


def test_settings_dict_includes_every_pixel_affecting_choice():
    keys = set(_req().settings_dict(scale=2, tile=512, video_filter="hqdn3d=1:1:1:1"))
    for expected in (
        "model", "scale", "tile", "deblock", "degrain",
        "detail_retention", "regrain", "target_fps", "video_filter",
    ):
        assert expected in keys


def test_settings_dict_changes_when_a_slider_moves():
    a = _req().settings_dict(scale=2, tile=512, video_filter="")
    b = _req(regrain=0.9).settings_dict(scale=2, tile=512, video_filter="")
    assert a != b


def test_out_of_range_strength_is_rejected():
    with pytest.raises(ValueError):
        _req(degrain=1.5)


def test_negative_strength_is_rejected():
    with pytest.raises(ValueError):
        _req(detail_retention=-0.1)


def test_no_restore_zeroes_every_strength():
    r = _req(no_restore=True)
    assert (r.degrain, r.deblock, r.detail_retention, r.regrain) == (0.0, 0.0, 0.0, 0.0)


def test_preview_range_is_none_by_default():
    assert _req().preview_frames is None


def test_preview_request_is_marked_as_such():
    assert _req(preview_frames=250).is_preview


def test_preview_output_is_distinct_from_the_real_output():
    r = _req(output=Path("/tmp/a.mkv"), preview_frames=250)
    assert r.output != Path("/tmp/a.mkv")
    assert "preview" in r.output.name


def test_preview_uses_a_separate_job_dir():
    """A preview must never be mistaken for progress on the real render."""
    r = _req(output=Path("/tmp/a.mkv"), preview_frames=250)
    assert r.job_dir != Path("/tmp/a.job")


def test_target_fps_below_source_is_rejected_at_validation():
    with pytest.raises(ValueError, match="below"):
        _req(target_fps=24.0).validate_against(src_fps=60.0)


def test_target_fps_equal_to_source_is_accepted():
    _req(target_fps=24.0).validate_against(src_fps=24.0)


def test_validation_passes_when_no_interpolation_requested():
    _req().validate_against(src_fps=60.0)
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.requests'`**

- [ ] **Step 3: Implement**

```python
"""User render choices, independent of any interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PREVIEW_SUFFIX = ".preview"


def _check(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


@dataclass
class RenderRequest:
    """Every choice a render depends on.

    Deliberately free of Qt and of the engine: the GUI builds one, the CLI could
    build one, and it converts itself into engine arguments. Keeping the policy
    here rather than in widget callbacks is what makes it testable.
    """

    model: Path
    source: Path
    output: Path

    deblock: float = 0.0
    degrain: float = 0.25
    detail_retention: float = 0.25
    regrain: float = 0.6
    no_restore: bool = False

    target_fps: float | None = None
    scene_threshold: float = 0.30

    tile: int = 0
    overlap: int = 16
    cpu: bool = False
    segment_frames: int = 500

    preview_frames: int | None = None
    job_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.no_restore:
            self.deblock = self.degrain = 0.0
            self.detail_retention = self.regrain = 0.0
        for name in ("deblock", "degrain", "detail_retention", "regrain"):
            _check(name, getattr(self, name))

        self.model = Path(self.model)
        self.source = Path(self.source)
        self.output = Path(self.output)

        if self.preview_frames is not None:
            # A preview must be visibly separate from the real render, and must
            # never leave journal state the real render would try to resume.
            self.output = self.output.with_name(
                self.output.stem + PREVIEW_SUFFIX + self.output.suffix
            )

        if self.job_dir is None:
            self.job_dir = self.output.with_suffix(".job")
        else:
            self.job_dir = Path(self.job_dir)

    @property
    def is_preview(self) -> bool:
        return self.preview_frames is not None

    def validate_against(self, src_fps: float) -> None:
        """Check choices that need to know something about the source."""
        if self.target_fps is not None and self.target_fps < src_fps:
            raise ValueError(
                f"target rate {self.target_fps:g} is below the source rate "
                f"{src_fps:.3f}; this tool interpolates but does not decimate"
            )

    def settings_dict(self, scale: int, tile: int, video_filter: str) -> dict:
        """The job hash input: everything that can change output pixels."""
        return {
            "model": self.model.name,
            "scale": scale,
            "tile": tile,
            "overlap": self.overlap,
            "cpu": self.cpu,
            "no_restore": self.no_restore,
            "deblock": self.deblock,
            "degrain": self.degrain,
            "detail_retention": self.detail_retention,
            "regrain": self.regrain,
            "target_fps": self.target_fps,
            "scene_threshold": self.scene_threshold,
            "video_filter": video_filter,
        }
```

- [ ] **Step 4: Run, confirm 15 passed. Commit**

```bash
git add src/enhancer/requests.py tests/test_requests.py && git commit -m "feat(requests): interface-independent render request"
```

---

## Task 2: Render worker and preview

**Files:**
- Create: `src/enhancer/gui.py`
- Test: `tests/test_gui.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from enhancer.gui import CancelledError, RenderJob
from enhancer.requests import RenderRequest
from enhancer.video_io import Decoder, SourceProfile


def _request(tmp_path, clip, **kw):
    return RenderRequest(
        model=tmp_path / "unused.pth", source=clip,
        output=tmp_path / "out.mkv", **kw
    )


class _Doubler:
    scale = 2
    cpu_fallback_count = 0

    def process(self, frame):
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def test_job_reports_progress(tmp_path, synthetic_clip):
    seen = []
    job = RenderJob(_request(tmp_path, synthetic_clip), upscaler=_Doubler())
    job.run(on_progress=lambda d, t: seen.append((d, t)))
    assert seen
    assert seen[-1][0] == seen[-1][1]


def test_job_produces_the_output_file(tmp_path, synthetic_clip):
    req = _request(tmp_path, synthetic_clip)
    RenderJob(req, upscaler=_Doubler()).run()
    assert req.output.exists()
    assert SourceProfile.probe(req.output).width == 640


def test_cancel_stops_the_render(tmp_path, synthetic_clip):
    job = RenderJob(_request(tmp_path, synthetic_clip), upscaler=_Doubler())

    def cancel_after_ten(done, total):
        if done >= 10:
            job.cancel()

    with pytest.raises(CancelledError):
        job.run(on_progress=cancel_after_ten)


def test_cancel_leaves_the_job_resumable(tmp_path, synthetic_clip):
    """Cancel is the crash path, taken deliberately, so resume must work."""
    req = _request(tmp_path, synthetic_clip, segment_frames=10)
    job = RenderJob(req, upscaler=_Doubler())

    def cancel_late(done, total):
        if done >= 25:
            job.cancel()

    with pytest.raises(CancelledError):
        job.run(on_progress=cancel_late)

    resumed = RenderJob(req, upscaler=_Doubler())
    resumed.run()
    assert len(list(Decoder(SourceProfile.probe(req.output)).frames())) == 50


def test_preview_renders_only_the_requested_frames(tmp_path, synthetic_clip):
    req = _request(tmp_path, synthetic_clip, preview_frames=10)
    RenderJob(req, upscaler=_Doubler()).run()
    assert len(list(Decoder(SourceProfile.probe(req.output)).frames())) == 10


def test_preview_does_not_touch_the_real_job_dir(tmp_path, synthetic_clip):
    real = _request(tmp_path, synthetic_clip)
    preview = _request(tmp_path, synthetic_clip, preview_frames=10)
    RenderJob(preview, upscaler=_Doubler()).run()
    assert not real.job_dir.exists()


def test_preview_output_is_named_distinctly(tmp_path, synthetic_clip):
    req = _request(tmp_path, synthetic_clip, preview_frames=10)
    assert "preview" in req.output.name
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.gui'`**

- [ ] **Step 3: Implement `RenderJob`** (the non-Qt half of `gui.py`)

`RenderJob` wraps `render_resumable`, owns a cancel flag, and raises `CancelledError` from inside the progress callback. Previews clamp the source to `preview_frames` by shrinking the probed profile, so the preview travels the identical code path as the real render.

- [ ] **Step 4: Run, confirm 7 passed. Commit**

```bash
git add src/enhancer/gui.py tests/test_gui.py && git commit -m "feat(gui): cancellable render job with segment preview"
```

---

## Task 3: The window

**Files:**
- Modify: `src/enhancer/gui.py`
- Modify: `src/enhancer/cli.py`

Widgets carry no policy; every decision belongs to `RenderRequest`.

Layout, top to bottom:

- **Source** — drop zone and file picker. On load, run `analyze` in a thread and show resolution, frame rate, colour space, scan type, grain and blockiness, plus the recommended pre-pass.
- **Model** — combo populated from `models/custom/`, with the detected scale shown.
- **Texture** — sliders for detail retention, re-grain, degrain and deblock, each labelled with what raising it costs. A "no restoration" checkbox.
- **Frame rate** — off / multiplier / target, with 48, 50, 60 and 120 presets. Disabled with an explanation when RIFE weights are absent.
- **Output** — path picker, segment size, CPU checkbox.
- **Actions** — Preview 10s, Render, Cancel. Render shows a resume notice when a matching job directory already exists.
- **Progress** — bar, frames done and total, fps, ETA, and a log pane.

- [ ] **Step 1: Implement the window and worker thread**

The worker runs `RenderJob` on a `QThread` and emits `progress`, `log`, `finished` and `failed`. The UI thread never blocks.

- [ ] **Step 2: Add `gui` and `preview` subcommands to the CLI**

- [ ] **Step 3: Launch it and drive it by hand**

Load a clip, confirm the analysis panel fills in, run a preview, then a full render, then cancel one mid-way and confirm re-running resumes. Report what actually happened.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat(gui): desktop window for rendering and preview"
```

---

## Self-review notes

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| §7.2 segment preview | Tasks 1, 2 |
| §10 drag-and-drop, panels, progress, cancel | Task 3 |
| §10 UI thread never blocks | Task 3 |

**Deferred to Plan 5:** dual-pass 2K→4K (§7.3) and the A/B compare view (§10).

**Known gaps:**
- Preview clamps to the opening frames. Previewing a chosen timestamp would be more useful, and needs a seek offset threaded through.
- ETA is a running average and will mislead early in a render where tile size is still settling.
