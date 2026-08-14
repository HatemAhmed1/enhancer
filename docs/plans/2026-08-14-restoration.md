# Restoration Implementation Plan (Plan 2 of 5)

> **Implementation note:** each task is a self-contained TDD unit — write the failing test, confirm it fails, implement, confirm it passes, commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the texture-fidelity requirement — skin that reads as photographed rather than polished — and make old telecined and heavily compressed sources usable at all.

**Architecture:** Two halves, split by where the work is cheapest.

*Before the upscaler*, deinterlacing, inverse telecine, deblocking and degraining are expressed as an **ffmpeg filter graph inside the decoder**. These are mature C filters running at native speed; reimplementing them per-frame in numpy would cost more than the upscaler itself. `analyze.py` inspects the source and decides which filters are warranted; `restore.py` turns that decision into a filter string.

*After the upscaler*, detail retention and re-graining run **on the GPU in torch**, because they need the source frame and the upscaled frame together and must not round-trip through another encode.

**Tech Stack:** Python 3.12, PyTorch, ffmpeg filter graph (`idet`, `fieldmatch`, `decimate`, `bwdif`, `deblock`, `hqdn3d`), pytest.

**Spec reference:** `docs/specs/2026-08-14-local-video-upscaler-design.md` §3 (texture doctrine), §5.1–5.4.

---

## Scope decision: no face-restoration GANs

Spec §3.1 offers GFPGAN/CodeFormer behind a blend slider. This plan **omits them entirely**.

They are the largest single contributor to the waxy appearance the user has twice asked to avoid: they replace an actor's real micro-texture with a generic face prior, and they flicker across the hard cuts that Indian cinema choreography is full of. Every other item in §3 *preserves* real information; a face GAN *substitutes* invented information.

They remain easy to add later behind the already-designed blend slider if a genuinely soft print needs rescuing. Nothing in this plan forecloses that.

---

## Design notes

### Why detail retention is the centrepiece

Spec §3.4 describes Original Detail Retention as the only stage that adds back *photographed* texture rather than *plausible* texture. Concretely: the source frame is resampled to the output size with bicubic, high-pass filtered, and that residual is added back over the model's output.

```
reference = bicubic(source → output size)
residual  = reference − blur(reference)
final     = model_output + alpha · residual
```

The residual carries only detail that genuinely exists in the source, at the scale it exists. It cannot invent pores, and that is exactly the point. Default alpha is 0.25.

### Why degrain must be conservative

Degraining before upscaling is necessary — otherwise the model amplifies grain into digital noise. It is also the stage most likely to destroy skin texture, because grain and skin micro-texture occupy the same spatial frequencies. Over-degraining *is* the polishing agent this project exists to avoid.

Default is deliberately light: `hqdn3d` with low spatial and moderate temporal strength. Temporal denoising removes frame-to-frame noise while leaving within-frame detail comparatively intact, which is the correct bias here. It also suppresses the texture crawl that single-image super-resolution produces on jewellery specular highlights.

### Why re-graining is on by default

Grain is what makes skin read as organic. Removing it and never putting it back produces exactly the plastic appearance being avoided. Re-grain is applied after upscaling, at an amount estimated from the source, so a grainy print stays grainy and a clean digital source stays clean.

### Why the correct pre-pass is source-dependent

A film-sourced 30i DVD is **not** interlaced video — it is 24p carried as 29.97i via 3:2 pulldown. It needs inverse telecine (`fieldmatch` + `decimate`), which recovers the original progressive frames exactly. Running a deinterlacer on it instead throws away real information and softens every frame.

True interlaced video needs an actual deinterlacer (`bwdif`). Applying `fieldmatch` to it produces combing.

Choosing wrong is not a subtle quality difference; it damages every frame of a multi-hour render irreversibly. Hence `analyze.py` detects rather than assumes, and reports its finding for confirmation.

---

## File structure

| File | Responsibility |
|---|---|
| `src/enhancer/analyze.py` | ffmpeg `idet` probing, interlace/telecine classification, grain and blockiness estimation |
| `src/enhancer/restore.py` | Filter-graph construction; torch detail retention and re-graining |
| `src/enhancer/video_io.py` | MODIFY: `Decoder` accepts a filter chain |
| `src/enhancer/cli.py` | MODIFY: `analyze` subcommand, restoration flags on `video` |
| `tests/test_analyze.py` | idet parsing, classification thresholds, estimators |
| `tests/test_restore.py` | Filter strings, detail retention, re-grain |

---

## Task 1: Source field-order analysis

**Files:**
- Create: `src/enhancer/analyze.py`
- Test: `tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from enhancer.analyze import (
    FieldAnalysis,
    ScanType,
    _parse_idet,
    classify_scan,
)

PROGRESSIVE_OUTPUT = """
[Parsed_idet_0 @ 000] Repeated Fields: Neither:   200 Top:     0 Bottom:     0
[Parsed_idet_0 @ 000] Single frame detection: TFF:     0 BFF:     0 Progressive:   200 Undetermined:     0
[Parsed_idet_0 @ 000] Multi frame detection: TFF:     0 BFF:     0 Progressive:   200 Undetermined:     0
"""

TELECINE_OUTPUT = """
[Parsed_idet_0 @ 000] Repeated Fields: Neither:   120 Top:    40 Bottom:    40
[Parsed_idet_0 @ 000] Single frame detection: TFF:    80 BFF:     0 Progressive:   120 Undetermined:     0
[Parsed_idet_0 @ 000] Multi frame detection: TFF:    80 BFF:     0 Progressive:   120 Undetermined:     0
"""

INTERLACED_OUTPUT = """
[Parsed_idet_0 @ 000] Repeated Fields: Neither:   198 Top:     1 Bottom:     1
[Parsed_idet_0 @ 000] Single frame detection: TFF:   190 BFF:     2 Progressive:     8 Undetermined:     0
[Parsed_idet_0 @ 000] Multi frame detection: TFF:   192 BFF:     2 Progressive:     6 Undetermined:     0
"""


def test_parse_idet_extracts_counts():
    a = _parse_idet(INTERLACED_OUTPUT)
    assert a.tff == 192
    assert a.bff == 2
    assert a.progressive == 6
    assert a.repeated_top == 1
    assert a.repeated_bottom == 1


def test_parse_idet_prefers_multi_frame_detection():
    """Multi-frame detection is more reliable than single-frame."""
    a = _parse_idet(TELECINE_OUTPUT)
    assert a.tff == 80


def test_parse_idet_raises_on_missing_output():
    with pytest.raises(ValueError, match="no idet output"):
        _parse_idet("ffmpeg version N-1234\nnothing useful here\n")


def test_classify_progressive():
    assert classify_scan(_parse_idet(PROGRESSIVE_OUTPUT)) is ScanType.PROGRESSIVE


def test_classify_telecine_from_repeated_fields():
    """Repeated fields are the 3:2 pulldown signature."""
    assert classify_scan(_parse_idet(TELECINE_OUTPUT)) is ScanType.TELECINED


def test_classify_interlaced():
    assert classify_scan(_parse_idet(INTERLACED_OUTPUT)) is ScanType.INTERLACED


def test_classify_handles_all_zero_counts():
    a = FieldAnalysis(tff=0, bff=0, progressive=0, undetermined=0,
                      repeated_top=0, repeated_bottom=0)
    assert classify_scan(a) is ScanType.PROGRESSIVE


def test_field_order_reports_bff_when_bottom_dominant():
    a = FieldAnalysis(tff=2, bff=190, progressive=8, undetermined=0,
                      repeated_top=1, repeated_bottom=1)
    assert classify_scan(a) is ScanType.INTERLACED
    assert a.field_order == "bff"


def test_field_order_is_tff_when_top_dominant():
    assert _parse_idet(INTERLACED_OUTPUT).field_order == "tff"
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.analyze'`**

- [ ] **Step 3: Implement**

```python
"""Source characterisation: scan type, grain, and compression artifacts."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Fraction of frames showing repeated fields above which a source is treated as
# telecined rather than interlaced. 3:2 pulldown repeats one field in five.
TELECINE_REPEAT_RATIO = 0.10

# Fraction of frames that must be detected as interlaced before deinterlacing.
INTERLACE_RATIO = 0.20

IDET_FRAMES = 400


class ScanType(Enum):
    PROGRESSIVE = "progressive"
    INTERLACED = "interlaced"
    TELECINED = "telecined"


@dataclass(frozen=True)
class FieldAnalysis:
    tff: int
    bff: int
    progressive: int
    undetermined: int
    repeated_top: int
    repeated_bottom: int

    @property
    def total(self) -> int:
        return self.tff + self.bff + self.progressive + self.undetermined

    @property
    def repeated(self) -> int:
        return self.repeated_top + self.repeated_bottom

    @property
    def field_order(self) -> str:
        return "tff" if self.tff >= self.bff else "bff"


_MULTI_RE = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
    r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)
_REPEAT_RE = re.compile(
    r"Repeated Fields:\s*Neither:\s*(\d+)\s*Top:\s*(\d+)\s*Bottom:\s*(\d+)"
)


def _parse_idet(output: str) -> FieldAnalysis:
    """Parse ffmpeg's idet filter summary.

    Multi-frame detection is used in preference to single-frame: it considers
    neighbouring frames and is markedly more reliable on real footage.
    """
    multi = _MULTI_RE.search(output)
    if multi is None:
        raise ValueError("no idet output found; the probe may have failed")
    repeat = _REPEAT_RE.search(output)

    return FieldAnalysis(
        tff=int(multi.group(1)),
        bff=int(multi.group(2)),
        progressive=int(multi.group(3)),
        undetermined=int(multi.group(4)),
        repeated_top=int(repeat.group(2)) if repeat else 0,
        repeated_bottom=int(repeat.group(3)) if repeat else 0,
    )


def classify_scan(analysis: FieldAnalysis) -> ScanType:
    """Decide which pre-pass a source needs.

    Order matters: a film-sourced 30i DVD reports as interlaced to a naive
    check, but needs inverse telecine rather than deinterlacing. Repeated
    fields are the discriminator, since 3:2 pulldown repeats one field in five.
    """
    if analysis.total == 0:
        return ScanType.PROGRESSIVE

    if analysis.repeated / analysis.total >= TELECINE_REPEAT_RATIO:
        return ScanType.TELECINED

    interlaced = analysis.tff + analysis.bff
    if interlaced / analysis.total >= INTERLACE_RATIO:
        return ScanType.INTERLACED

    return ScanType.PROGRESSIVE


def probe_scan(path: str | Path, frames: int = IDET_FRAMES) -> FieldAnalysis:
    """Run ffmpeg's idet filter over the opening frames of a source."""
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-i", str(path),
            "-vf", "idet", "-frames:v", str(frames),
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    return _parse_idet(result.stderr)
```

- [ ] **Step 4: Run, confirm 9 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/analyze.py tests/test_analyze.py && git commit -m "feat(analyze): field-order probing and scan-type classification"
```

---

## Task 2: Grain and blockiness estimation

**Files:**
- Modify: `src/enhancer/analyze.py`
- Test: `tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analyze.py`:

```python
import numpy as np

from enhancer.analyze import estimate_blockiness, estimate_grain


def test_grain_is_near_zero_for_a_flat_frame():
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert estimate_grain(flat) < 0.5


def test_grain_rises_with_added_noise():
    rng = np.random.default_rng(0)
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    noisy = np.clip(
        flat.astype(np.int16) + rng.normal(0, 8, flat.shape), 0, 255
    ).astype(np.uint8)
    assert estimate_grain(noisy) > estimate_grain(flat) + 2


def test_grain_ignores_smooth_gradients():
    """A gradient has energy but no grain; the estimator must not confuse them."""
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    frame = np.dstack([ramp] * 3)
    assert estimate_grain(frame) < 1.0


def test_grain_is_deterministic():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert estimate_grain(frame) == estimate_grain(frame)


def test_blockiness_is_low_for_a_smooth_frame():
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    assert estimate_blockiness(np.dstack([ramp] * 3)) < 1.0


def test_blockiness_rises_with_8x8_discontinuities():
    """Simulate DCT block edges every 8 pixels."""
    frame = np.full((128, 128, 3), 120, dtype=np.uint8)
    frame[:, ::8] = 160
    assert estimate_blockiness(frame) > 3.0
```

- [ ] **Step 2: Run and confirm `ImportError: cannot import name 'estimate_grain'`**

- [ ] **Step 3: Implement**

Add to `src/enhancer/analyze.py` (add `import numpy as np` to the imports at the top):

```python
def _luma(frame: np.ndarray) -> np.ndarray:
    """Rec. 709 luma as float32."""
    f = frame.astype(np.float32)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def estimate_grain(frame: np.ndarray) -> float:
    """Estimate grain amplitude, in 0-255 units.

    Measured as the standard deviation of a high-pass residual restricted to
    locally flat regions. Restricting to flat regions is what separates grain
    from genuine image detail and from smooth gradients, both of which would
    otherwise inflate the estimate.
    """
    y = _luma(frame)

    # 3x3 box blur via cumulative sums along both axes.
    pad = np.pad(y, 1, mode="edge")
    blur = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
        + pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
        + pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0

    residual = y - blur

    # Local gradient magnitude, used to exclude edges and textured regions.
    gy = np.abs(np.diff(blur, axis=0, prepend=blur[:1]))
    gx = np.abs(np.diff(blur, axis=1, prepend=blur[:, :1]))
    gradient = gy + gx

    flat = gradient < np.percentile(gradient, 40)
    if not flat.any():
        return float(residual.std())
    return float(residual[flat].std())


def estimate_blockiness(frame: np.ndarray) -> float:
    """Estimate DCT block-edge strength, in 0-255 units.

    Compares the mean absolute difference across 8-pixel-aligned column
    boundaries against non-aligned boundaries. Compression leaves discontinuities
    on the block grid; natural detail does not prefer that grid.
    """
    y = _luma(frame)
    diffs = np.abs(np.diff(y, axis=1))
    if diffs.shape[1] < 16:
        return 0.0

    columns = np.arange(diffs.shape[1])
    on_grid = (columns + 1) % 8 == 0
    if not on_grid.any() or on_grid.all():
        return 0.0

    return float(diffs[:, on_grid].mean() - diffs[:, ~on_grid].mean())
```

- [ ] **Step 4: Run, confirm 15 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/analyze.py tests/test_analyze.py && git commit -m "feat(analyze): grain and blockiness estimation"
```

---

## Task 3: Restoration filter graph

**Files:**
- Create: `src/enhancer/restore.py`
- Test: `tests/test_restore.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from enhancer.analyze import ScanType
from enhancer.restore import RestoreSettings, build_filter_chain


def test_progressive_clean_source_needs_no_filters():
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert chain == ""


def test_telecined_source_uses_inverse_telecine_not_deinterlace():
    """Film-sourced 30i must be IVTC'd; deinterlacing would discard real frames."""
    chain = build_filter_chain(
        ScanType.TELECINED, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert "fieldmatch" in chain
    assert "decimate" in chain
    assert "bwdif" not in chain


def test_interlaced_source_uses_deinterlacer_not_ivtc():
    chain = build_filter_chain(
        ScanType.INTERLACED, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert "bwdif" in chain
    assert "fieldmatch" not in chain


def test_bottom_field_first_is_passed_to_the_deinterlacer():
    chain = build_filter_chain(
        ScanType.INTERLACED, field_order="bff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert "parity=1" in chain


def test_deblock_is_included_when_requested():
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.5, degrain=0.0),
    )
    assert "deblock" in chain


def test_degrain_is_included_when_requested():
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.5),
    )
    assert "hqdn3d" in chain


def test_degrain_biases_temporal_over_spatial():
    """Spatial denoising destroys skin texture; temporal mostly does not."""
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=1.0),
    )
    params = chain.split("hqdn3d=")[1].split(",")[0]
    luma_spatial, _chroma_spatial, luma_temporal, _chroma_temporal = (
        float(v) for v in params.split(":")
    )
    assert luma_temporal > luma_spatial


def test_default_degrain_is_light():
    """Over-degraining is itself the polishing agent this project avoids."""
    assert RestoreSettings().degrain <= 0.35


def test_filters_are_ordered_scan_then_deblock_then_degrain():
    chain = build_filter_chain(
        ScanType.INTERLACED, field_order="tff",
        settings=RestoreSettings(deblock=0.5, degrain=0.5),
    )
    assert chain.index("bwdif") < chain.index("deblock") < chain.index("hqdn3d")


def test_strength_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        build_filter_chain(
            ScanType.PROGRESSIVE, field_order="tff",
            settings=RestoreSettings(deblock=2.0, degrain=0.0),
        )
```

- [ ] **Step 2: Run and confirm `ModuleNotFoundError: No module named 'enhancer.restore'`**

- [ ] **Step 3: Implement**

```python
"""Restoration: pre-upscale filtering and post-upscale texture work."""

from __future__ import annotations

from dataclasses import dataclass

from .analyze import ScanType

# Maximum hqdn3d strengths at degrain=1.0. Temporal is allowed to go
# considerably higher than spatial: temporal denoising removes frame-to-frame
# noise while leaving within-frame detail comparatively intact, which is the
# correct bias when skin texture is the thing being protected.
MAX_LUMA_SPATIAL = 4.0
MAX_CHROMA_SPATIAL = 3.0
MAX_LUMA_TEMPORAL = 9.0
MAX_CHROMA_TEMPORAL = 7.0


@dataclass(frozen=True)
class RestoreSettings:
    """Restoration strengths, each in the range 0.0 to 1.0.

    `degrain` defaults low on purpose. Grain and skin micro-texture occupy the
    same spatial frequencies, so an aggressive degrain removes exactly the
    detail this project exists to preserve.
    """

    deblock: float = 0.0
    degrain: float = 0.25
    detail_retention: float = 0.25
    regrain: float = 0.6


def _check(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


def build_filter_chain(
    scan: ScanType,
    field_order: str,
    settings: RestoreSettings,
) -> str:
    """Build the ffmpeg -vf chain applied before upscaling.

    These run inside the decoder at native C speed. Reimplementing them per
    frame in Python would cost more than the upscaler itself.
    """
    _check("deblock", settings.deblock)
    _check("degrain", settings.degrain)

    filters: list[str] = []

    if scan is ScanType.TELECINED:
        # Recovers the original progressive frames exactly. A deinterlacer
        # would instead discard half the real information in every frame.
        filters.append("fieldmatch")
        filters.append("decimate")
    elif scan is ScanType.INTERLACED:
        parity = 1 if field_order == "bff" else 0
        filters.append(f"bwdif=mode=0:parity={parity}:deint=0")

    if settings.deblock > 0:
        strength = 0.02 + 0.08 * settings.deblock
        filters.append(f"deblock=filter=weak:block=8:alpha={strength:.3f}")

    if settings.degrain > 0:
        d = settings.degrain
        filters.append(
            "hqdn3d="
            f"{MAX_LUMA_SPATIAL * d:.2f}:"
            f"{MAX_CHROMA_SPATIAL * d:.2f}:"
            f"{MAX_LUMA_TEMPORAL * d:.2f}:"
            f"{MAX_CHROMA_TEMPORAL * d:.2f}"
        )

    return ",".join(filters)
```

- [ ] **Step 4: Run, confirm 10 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/restore.py tests/test_restore.py && git commit -m "feat(restore): pre-upscale filter graph for scan, deblock and degrain"
```

---

## Task 4: Decoder filter support

**Files:**
- Modify: `src/enhancer/video_io.py`
- Test: `tests/test_video_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_video_io.py`:

```python
def test_decoder_accepts_a_filter_chain(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p, video_filter="hqdn3d=4:3:6:4").frames())
    assert len(frames) == 50
    assert frames[0].shape == (240, 320, 3)


def test_filtered_output_differs_from_unfiltered(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    plain = list(Decoder(p).frames())[0]
    blurred = list(Decoder(p, video_filter="boxblur=4").frames())[0]
    assert not np.array_equal(plain, blurred)


def test_empty_filter_chain_is_equivalent_to_none(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    a = list(Decoder(p, video_filter="").frames())
    b = list(Decoder(p).frames())
    assert len(a) == len(b)
    assert np.array_equal(a[0], b[0])


def test_decimating_filter_reduces_frame_count(synthetic_clip):
    """IVTC drops frames; the decoder must not assume a fixed count."""
    p = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(p, video_filter="select='not(mod(n,2))'").frames())
    assert len(frames) == 25


def test_filter_combines_with_seeking(synthetic_clip):
    p = SourceProfile.probe(synthetic_clip)
    frames = list(
        Decoder(p, start_frame=10, max_frames=5, video_filter="hqdn3d=4:3:6:4").frames()
    )
    assert len(frames) == 5
```

- [ ] **Step 2: Run and confirm `TypeError: Decoder.__init__() got an unexpected keyword argument 'video_filter'`**

- [ ] **Step 3: Implement**

In `src/enhancer/video_io.py`, add `video_filter: str = ""` to `Decoder.__init__` (store as `self.video_filter`), and insert into `_command()` immediately before the `-f rawvideo` arguments:

```python
        if self.video_filter:
            cmd += ["-vf", self.video_filter]
```

Place it after the `-frames:v` handling so that frame limiting applies to filtered output.

- [ ] **Step 4: Run the full `test_video_io.py`, confirm all pass**

Note: `test_decimating_filter_reduces_frame_count` proves that a decimating filter changes the output frame count. This matters for Plan 1.5's segmenting, which sizes segments from `profile.frame_count`. Record the consequence in the report — inverse telecine reduces 30i to 24p, so a job's true output length is roughly four fifths of the probed count.

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/video_io.py tests/test_video_io.py && git commit -m "feat(video_io): decoder video filter chain support"
```

---

## Task 5: Detail retention

**Files:**
- Modify: `src/enhancer/restore.py`
- Test: `tests/test_restore.py`

This is spec §3.4 and the centrepiece of the texture requirement.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_restore.py`:

```python
import numpy as np
import torch

from enhancer.restore import apply_detail_retention, gaussian_blur


def test_gaussian_blur_preserves_shape():
    x = torch.rand(1, 3, 32, 48)
    assert gaussian_blur(x, sigma=1.5).shape == x.shape


def test_gaussian_blur_reduces_high_frequency_energy():
    x = torch.rand(1, 3, 64, 64)
    blurred = gaussian_blur(x, sigma=2.0)
    assert blurred.var() < x.var()


def test_gaussian_blur_leaves_a_constant_image_unchanged():
    x = torch.full((1, 3, 32, 32), 0.5)
    assert torch.allclose(gaussian_blur(x, sigma=2.0), x, atol=1e-5)


def test_alpha_zero_returns_model_output_unchanged():
    source = torch.rand(1, 3, 16, 16)
    output = torch.rand(1, 3, 32, 32)
    result = apply_detail_retention(output, source, alpha=0.0)
    assert torch.equal(result, output)


def test_detail_retention_adds_high_frequency_energy():
    """The residual must actually reach the output."""
    rng = torch.Generator().manual_seed(0)
    source = torch.rand(1, 3, 32, 32, generator=rng)
    output = gaussian_blur(
        torch.nn.functional.interpolate(source, scale_factor=2, mode="bicubic"),
        sigma=2.0,
    )
    enhanced = apply_detail_retention(output, source, alpha=0.5)
    assert enhanced.var() > output.var()


def test_detail_retention_output_stays_in_range():
    source = torch.rand(1, 3, 16, 16)
    output = torch.rand(1, 3, 32, 32)
    result = apply_detail_retention(output, source, alpha=1.0)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_detail_retention_is_deterministic():
    source = torch.rand(1, 3, 16, 16)
    output = torch.rand(1, 3, 32, 32)
    a = apply_detail_retention(output, source, alpha=0.4)
    b = apply_detail_retention(output, source, alpha=0.4)
    assert torch.equal(a, b)


def test_detail_retention_rejects_alpha_out_of_range():
    with pytest.raises(ValueError):
        apply_detail_retention(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 4, 4), alpha=1.5)


def test_detail_retention_handles_non_integer_scale():
    """Source and output need not differ by a whole-number factor."""
    source = torch.rand(1, 3, 17, 23)
    output = torch.rand(1, 3, 40, 55)
    assert apply_detail_retention(output, source, alpha=0.3).shape == output.shape
```

- [ ] **Step 2: Run and confirm `ImportError: cannot import name 'apply_detail_retention'`**

- [ ] **Step 3: Implement**

Add to `src/enhancer/restore.py` (add `import torch` and `import torch.nn.functional as F` at the top):

```python
def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on an (N, C, H, W) tensor."""
    radius = max(1, int(round(3.0 * sigma)))
    taps = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(taps ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, horizontal, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, vertical, groups=channels)


def apply_detail_retention(
    output: torch.Tensor,
    source: torch.Tensor,
    alpha: float,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Blend the source's real high-frequency detail over the model output.

    Every other stage in this pipeline either preserves information or invents
    it. This one restores genuinely photographed micro-texture: the residual is
    derived from the source, so it can only contain detail that was actually
    captured. That is precisely why it cannot fabricate pores, and precisely why
    it is the most direct answer to the requirement that skin not look polished.

    `output` is (N, C, H, W) in [0, 1]. `source` is the pre-upscale frame in the
    same layout at lower resolution.
    """
    _check("alpha", alpha)
    if alpha == 0.0:
        return output

    reference = F.interpolate(
        source, size=output.shape[-2:], mode="bicubic", align_corners=False
    )
    residual = reference - gaussian_blur(reference, sigma)
    return (output + alpha * residual).clamp_(0.0, 1.0)
```

- [ ] **Step 4: Run, confirm 19 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/restore.py tests/test_restore.py && git commit -m "feat(restore): original detail retention from source high frequencies"
```

---

## Task 6: Re-graining

**Files:**
- Modify: `src/enhancer/restore.py`
- Test: `tests/test_restore.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_restore.py`:

```python
from enhancer.restore import apply_regrain


def test_amount_zero_returns_the_input_unchanged():
    x = torch.rand(1, 3, 32, 32)
    assert torch.equal(apply_regrain(x, amount=0.0, seed=0), x)


def test_regrain_increases_local_variance():
    x = torch.full((1, 3, 64, 64), 0.5)
    grained = apply_regrain(x, amount=0.5, seed=0)
    assert grained.var() > x.var()


def test_regrain_is_deterministic_for_a_given_seed():
    x = torch.full((1, 3, 32, 32), 0.5)
    assert torch.equal(
        apply_regrain(x, amount=0.5, seed=42), apply_regrain(x, amount=0.5, seed=42)
    )


def test_different_seeds_produce_different_grain():
    x = torch.full((1, 3, 32, 32), 0.5)
    assert not torch.equal(
        apply_regrain(x, amount=0.5, seed=1), apply_regrain(x, amount=0.5, seed=2)
    )


def test_regrain_output_stays_in_range():
    x = torch.rand(1, 3, 32, 32)
    result = apply_regrain(x, amount=1.0, seed=0)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_stronger_amount_produces_more_grain():
    x = torch.full((1, 3, 64, 64), 0.5)
    light = apply_regrain(x, amount=0.2, seed=0)
    heavy = apply_regrain(x, amount=0.8, seed=0)
    assert heavy.var() > light.var()


def test_grain_is_suppressed_in_highlights_and_shadows():
    """Film grain is most visible in midtones; clipped areas show little."""
    mid = apply_regrain(torch.full((1, 3, 64, 64), 0.5), amount=0.8, seed=0)
    bright = apply_regrain(torch.full((1, 3, 64, 64), 0.99), amount=0.8, seed=0)
    assert mid.var() > bright.var()


def test_regrain_rejects_amount_out_of_range():
    with pytest.raises(ValueError):
        apply_regrain(torch.rand(1, 3, 8, 8), amount=-0.1, seed=0)
```

- [ ] **Step 2: Run and confirm `ImportError: cannot import name 'apply_regrain'`**

- [ ] **Step 3: Implement**

Add to `src/enhancer/restore.py`:

```python
# Grain amplitude at amount=1.0, in normalised [0, 1] units. Roughly 5/255,
# which reads as noticeable but not heavy film grain.
MAX_GRAIN_SIGMA = 0.02


def apply_regrain(
    x: torch.Tensor,
    amount: float,
    seed: int = 0,
    grain_size: float = 0.8,
) -> torch.Tensor:
    """Add synthetic grain to an upscaled frame.

    Super-resolution models classify low-contrast skin texture as noise and
    remove it. Putting grain back is the strongest available counter to the
    resulting plastic appearance: it restores the high-frequency content that
    makes skin read as photographed rather than rendered.

    Grain is attenuated towards black and white, mirroring real film, where
    grain is most visible in the midtones.
    """
    _check("amount", amount)
    if amount == 0.0:
        return x

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device)

    if grain_size > 0:
        noise = gaussian_blur(noise, sigma=grain_size)
        # Blurring reduces variance; renormalise so `amount` stays meaningful.
        std = noise.std()
        if std > 0:
            noise = noise / std

    # Triangular midtone weighting: peaks at 0.5, falls to zero at 0 and 1.
    luminance = x.mean(dim=1, keepdim=True)
    weight = 1.0 - (2.0 * luminance - 1.0).abs()

    return (x + noise * weight * (MAX_GRAIN_SIGMA * amount)).clamp_(0.0, 1.0)
```

- [ ] **Step 4: Run, confirm 27 passed**

- [ ] **Step 5: Commit**

```bash
git add src/enhancer/restore.py tests/test_restore.py && git commit -m "feat(restore): midtone-weighted synthetic re-graining"
```

---

## Task 7: Pipeline integration and CLI

**Files:**
- Modify: `src/enhancer/upscale.py`
- Modify: `src/enhancer/cli.py`
- Test: `tests/test_restore.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_restore.py`:

```python
from enhancer.restore import TexturePost


class _Passthrough:
    scale = 2

    def process(self, frame):
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def test_texture_post_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    output = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    post = TexturePost(detail_retention=0.3, regrain=0.5, device="cpu")
    result = post.apply(output, source)
    assert result.shape == output.shape
    assert result.dtype == np.uint8


def test_texture_post_disabled_returns_input_unchanged():
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    output = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    post = TexturePost(detail_retention=0.0, regrain=0.0, device="cpu")
    assert np.array_equal(post.apply(output, source), output)


def test_texture_post_changes_the_frame_when_enabled():
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    output = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    post = TexturePost(detail_retention=0.5, regrain=0.5, device="cpu")
    assert not np.array_equal(post.apply(output, source), output)


def test_texture_post_is_deterministic_per_frame_index():
    """Grain must not flicker: the same frame index yields the same grain."""
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    output = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    post = TexturePost(detail_retention=0.0, regrain=0.6, device="cpu")
    assert np.array_equal(post.apply(output, source, index=7),
                          post.apply(output, source, index=7))


def test_grain_differs_between_frames():
    """Static grain across frames reads as a dirty lens, not as film."""
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    output = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    post = TexturePost(detail_retention=0.0, regrain=0.6, device="cpu")
    assert not np.array_equal(post.apply(output, source, index=1),
                              post.apply(output, source, index=2))
```

- [ ] **Step 2: Run and confirm `ImportError: cannot import name 'TexturePost'`**

- [ ] **Step 3: Implement**

Add to `src/enhancer/restore.py` (import `numpy as np`, and `to_tensor`/`to_frame` from `.upscale`):

```python
class TexturePost:
    """Post-upscale texture work: detail retention then re-graining.

    Ordering matters. Detail retention restores real high frequencies from the
    source; re-graining then adds synthetic texture on top. Reversing them would
    high-pass the synthetic grain into the retention step and double-count it.
    """

    def __init__(
        self,
        detail_retention: float = 0.25,
        regrain: float = 0.6,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        _check("detail_retention", detail_retention)
        _check("regrain", regrain)
        self.detail_retention = detail_retention
        self.regrain = regrain
        self.device = torch.device(device)
        self.seed = seed

    @property
    def enabled(self) -> bool:
        return self.detail_retention > 0 or self.regrain > 0

    def apply(self, output: np.ndarray, source: np.ndarray, index: int = 0) -> np.ndarray:
        """Apply texture work to one upscaled frame."""
        if not self.enabled:
            return output

        out_t = to_tensor(output).to(self.device)
        if self.detail_retention > 0:
            src_t = to_tensor(source).to(self.device)
            out_t = apply_detail_retention(out_t, src_t, self.detail_retention)
        if self.regrain > 0:
            # Seed varies per frame so grain moves like film rather than sitting
            # static like dirt on the lens.
            out_t = apply_regrain(out_t, self.regrain, seed=self.seed + index)
        return to_frame(out_t)
```

Then wire it into `render_resumable` in `src/enhancer/cli.py`. Change the `processed()` closure to keep the source frame and pass it through:

```python
        def processed(decoder=decoder, start=start):
            for i, frame in enumerate(decoder.frames()):
                upscaled = upscaler.process(frame)
                if texture is not None and texture.enabled:
                    upscaled = texture.apply(upscaled, frame, index=start + i)
                yield upscaled
                if on_progress:
                    on_progress(start + i + 1, job.total_frames)
```

Add a `texture=None` keyword parameter to `render_resumable`.

- [ ] **Step 4: Add the `analyze` subcommand and restoration flags**

In `cli.py`:

```python
def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import classify_scan, estimate_blockiness, estimate_grain, probe_scan

    profile = SourceProfile.probe(args.input)
    analysis = probe_scan(args.input)
    scan = classify_scan(analysis)

    frame = next(iter(Decoder(profile, max_frames=1).frames()))
    grain = estimate_grain(frame)
    blockiness = estimate_blockiness(frame)

    print(f"Source:      {profile.width}x{profile.height} @ {profile.fps:.3f} fps")
    print(f"Colour:      {profile.color_space or 'unspecified'} / SAR {profile.sar}")
    print(f"Scan:        {scan.value} (field order {analysis.field_order})")
    print(f"Grain:       {grain:.2f}")
    print(f"Blockiness:  {blockiness:.2f}")
    print()
    if scan is ScanType.TELECINED:
        print("Recommended: inverse telecine. This is film carried as 30i;")
        print("             deinterlacing it would discard real detail.")
    elif scan is ScanType.INTERLACED:
        print("Recommended: deinterlace before upscaling.")
    else:
        print("Recommended: no scan correction needed.")
    if blockiness > 2.0:
        print("             Compression artifacts present; enable --deblock.")
    return 0
```

Register it, and add these flags to the `video` subparser:

```python
    v.add_argument("--deblock", type=float, default=0.0)
    v.add_argument("--degrain", type=float, default=0.25)
    v.add_argument("--detail-retention", type=float, default=0.25)
    v.add_argument("--regrain", type=float, default=0.6)
    v.add_argument("--no-restore", action="store_true",
                   help="skip all restoration and texture work")
```

In `cmd_video`, build the filter chain from the detected scan type, pass it to the `Decoder` used inside `render_resumable`, and construct a `TexturePost`. **Every restoration setting must be included in the `settings` dict** used for the job hash, so that changing one correctly refuses to resume.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`

- [ ] **Step 6: Visual verification on real footage**

Render the same 10-second clip three ways and compare stills:

1. `--no-restore`
2. defaults
3. `--detail-retention 0.0 --regrain 0.0` with degrain still on

Confirm that (2) shows more skin texture than (3), and that (3) looks smoother and more plastic than (2). This is the requirement being tested; it is not automatable and must be checked by eye.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(cli): restoration pipeline and source analysis command"
```

---

## Self-review notes

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| §3.2 conservative degrain | Task 3 |
| §3.3 re-grain on by default | Task 6 |
| §3.4 original detail retention | Task 5 |
| §3.5 no unsharp masking | Not implemented anywhere, by design |
| §5.1 source analysis | Tasks 1, 2, 7 |
| §5.2 deinterlace vs inverse telecine | Tasks 1, 3 |
| §5.3 deblock and degrain | Task 3 |

**Deliberately omitted:** §3.1 face-restoration GANs, for the reasons at the top of this document.

**Known gaps to address later:**
- Inverse telecine reduces frame count by roughly one fifth, but Plan 1.5's segmenting sizes segments from `profile.frame_count`, which is the pre-filter count. Resumable IVTC renders will mis-size their final segment. Task 4 surfaces this; it needs a follow-up that derives the true output length after filtering.
- `hqdn3d` is a spatial-temporal denoiser but not motion-compensated. Spec §5.3 asks for motion compensation, which needs VapourSynth MVTools. Deferred, consistent with the QTGMC decision to auto-detect rather than require VapourSynth.
- Grain and blockiness are estimated from a single frame in `cmd_analyze`. Sampling several frames spread through the source would be more robust.
