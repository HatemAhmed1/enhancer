# Local AI Video/Image Enhancer — Design Spec

**Date:** 2026-08-14
**Revision:** 2 (post-audit)
**Target hardware:** RTX 3060 Laptop (6 GB VRAM, driver 610.47), i7-11th gen, 64 GB DDR4, Windows 11
**Status:** Approved for planning

---

## 1. Goal

A local, free, GPU-accelerated desktop application that restores, spatially upscales, and
temporally interpolates video and images, tuned for Indian cinema footage: rich skin tones,
high-specularity jewelry, fast-cut choreography, and mixed-quality sources ranging from
interlaced DVD rips to modern digital masters.

Two constraints govern every decision below.

**Constraint 1 — throughput.** A 2-hour feature at 24 fps is ~172,800 frames. Any per-frame cost
above ~60 ms turns the job into an overnight render.

**Constraint 2 — texture fidelity.** Skin must not be polished. See §3, which is a design doctrine
rather than a feature list, and which overrides default choices elsewhere in this document.

### 1.1 Performance targets

Measured on target hardware, 480p→960p (2x), fp16, upscale stage only:

| Tier | Target throughput | 2h feature (est.) |
|---|---|---|
| Turbo (Compact) | ≥ 40 fps | ~1.2 h |
| Fast (SPAN) | ≥ 25 fps | ~1.9 h |
| Balanced (RealPLKSR) | ≥ 12 fps | ~4 h |
| Quality (RRDB-ESRGAN) | ≥ 1.5 fps | ~32 h (stills / short clips only) |

Acceptance thresholds, not guarantees. The benchmark harness (§11) measures actuals; the GUI shows
measured fps so a bad configuration is abandoned in minute two rather than discovered at hour six.

### 1.2 Non-goals

- No cloud inference, no paid APIs, no telemetry. Fully offline after model download.
- No training UI. Fine-tuning is a documented external workflow (§13).
- No NLE features: no editing, grading, or audio work beyond stream passthrough.
- No diffusion-based VSR (§4.4).

---

## 2. Why the current approach is slow

The existing Real-ESRGAN 2x workflow takes hours per video. Four compounding causes, by impact:

1. **Architecture.** RRDBNet is 16.7 M parameters, ~500 GFLOPs per 480p frame. At ~1.5 fps,
   172,800 frames is ~32 hours. This alone dominates.
2. **Disk frame-caching.** Most wrappers decode to PNG, upscale, re-encode — three full passes over
   hundreds of GB, stalling the GPU on I/O.
3. **fp32 inference.** Forfeits roughly half of Ampere's Tensor-Core throughput.
4. **Naive tiling.** Large overlaps recompute 20–40 % of pixels.

Item 1 is worth 10–20x; items 2–4 together roughly another 2–3x. Model *quality* selection is not
where the speed is. Additionally, **running a 4x model when 2x is needed wastes ~4x the compute** —
for a 1080p source, 2x already lands on 4K.

---

## 3. Texture fidelity doctrine

> **Requirement, stated by the user: skin textures must not be polished.**

Waxy skin is not one bad setting; it is the cumulative result of four independent stages each
removing a little high-frequency information. The mechanism is well documented: SR models
*"classify low-contrast skin texture as noise and remove it during denoising,"* after which
sharpening and face enhancement *"exaggerate highlights and edges, creating waxy cheeks and a
doll-like finish."*

This design therefore treats texture as a budget that every stage spends from, and constrains all
four spenders:

**3.1 Face-restoration GANs — default OFF for video.**
GFPGAN and CodeFormer hallucinate a generic face prior over the actor's real micro-texture, and
they flicker across cuts. Available with a **0–100 % blend slider**; defaults are **OFF for video**,
**40 % for stills**. CodeFormer's fidelity weight is exposed separately.

**3.2 Degrain — default LIGHT, and deliberately so.**
Degraining is necessary before SR (§5.3) but it is also the most likely stage to destroy skin
texture. Over-degraining *is* the polishing agent. Default strength is light — just enough to stop
the upscaler amplifying grain into digital noise, never enough to flatten pores. Motion-compensated
so it removes temporal noise rather than spatial detail.

**3.3 Re-grain — default ON.**
Grain is reintroduced after upscaling, matched to the source's estimated grain characteristics.
This is the strongest available anti-waxy lever: grain *"reintroduces the high-frequency texture
that generative models suppress"* and is *"the single biggest step toward skin reading as organic
rather than rendered."* It also improves perceived sharpness and masks residual SR artifacts.

**3.4 Original Detail Retention — a first-class control.**
A high-frequency residual is extracted from the source (source minus its low-pass) and
alpha-blended back over the SR output. This returns *genuine* photographed micro-texture rather
than model-inferred texture. Slider, default ~25 %. This is the single most direct answer to the
requirement, because it is the only stage that adds back real information rather than plausible
information.

**3.5 No unsharp masking anywhere in the default path.** Sharpening is what converts a slightly
soft result into a plastic one. Available, off by default.

**3.6 Tier default follows the doctrine.** RealPLKSR (Balanced) is the recommended default for
faces despite being slower than SPAN, because it retains real-world texture better.

---

## 4. Model decisions

### 4.1 Build our own model, or use existing?

**Decision: use existing community weights now; support local fine-tuning as a documented Phase 2.
Do not train from scratch.**

- *From scratch* requires multi-GPU A100-class compute for weeks plus a curated HR dataset. On a
  6 GB laptop this is months of effort to land below freely available community weights. Rejected.
- *Fine-tuning* is feasible on a 3060: SR trains on small crops, so a SPAN or Compact fine-tune fits
  6 GB at fp16, batch 4–8, over 1–3 days. Real-time-SR benchmarking finds SPAN and RLFN benefit
  significantly from domain-specific fine-tuning — precisely the Indian-cinema case.

The roster is therefore **data, not code**. Models load through
[spandrel](https://pypi.org/project/spandrel/), which auto-detects architecture and hyperparameters
from a bare `.pth`. A `models/custom/` directory is scanned at startup, so any OpenModelDB model or
user fine-tune appears in the GUI with zero code changes.

### 4.2 Upscaler roster

Compact and SPAN are the fastest video-viable architectures, RealPLKSR slightly behind,
ESRGAN/RRDB substantially slower.

| Tier | Architecture | Params | Role |
|---|---|---|---|
| Turbo | SRVGGNetCompact | ~1.2 M | Full-length passes, previews |
| Fast | SPAN | ~2 M | Feature-length work where speed dominates |
| **Balanced** | **RealPLKSR** (Nomos2 dysample) | ~7 M | **Default.** Best texture retention that stays video-viable |
| Quality | RRDB-ESRGAN (Remacri / UltraSharp / Nomos8kSC) | 16.7 M | Stills and short hero clips only |

**2x is the default scale factor.**

### 4.3 Frame interpolation and FPS upscaling

**RIFE 4.25** default (upstream Practical-RIFE's own recommendation), with **4.26** and
**4.25-lite** selectable.

Two user-facing modes:

- **Multiplier:** 2x, 3x, 4x.
- **Target FPS:** explicit output frame rate with presets 24→48, 24→60, 25→50, 30→60, 24→120.
  Non-integer ratios (24→60 is 2.5x) are supported because RIFE 4.x accepts an arbitrary
  **timestep** parameter, so intermediate frames are synthesized at fractional positions rather
  than restricted to integer multiples. Frame timing is resampled against the target rate.

**Scene-change detection is mandatory.** Indian cinema choreography is cut-heavy; interpolating
across a hard cut produces visible ghost-morphing. When the frame-pair difference metric exceeds a
threshold, the source frame is duplicated rather than synthesized. Threshold is user-adjustable.

**Stage order is configurable.** Default is upscale→interpolate: RIFE then blends already-detailed
frames, and SR never runs on synthesized content. Interpolate→upscale is offered because on the
Turbo tier it can be meaningfully cheaper. Documented trade-off, not a hardcode.

### 4.4 Rejected

- **Diffusion VSR (FlashVSR, CVPR 2026).** VAE plus sparse attention will not fit 6 GB at 1080p and
  would be slower than RRDB — the opposite of Constraint 1.
- **True recurrent VSR (BasicVSR++, RVRT).** The textbook fix for temporal flicker, but exceeds
  6 GB and is slower than RRDB. Flicker is addressed instead by motion-compensated temporal
  degraining in the pre-pass (§5.3), which attacks it at the source for near-zero cost.
- **Skipping near-identical frames.** Risks visible pops.
- **`torch.compile` as a promised tier.** Triton on Windows remains unreliable; opt-in best-effort.

---

## 5. Source acquisition and restoration pre-processing

This stage did not exist in revision 1 and is the most consequential addition. For older Indian
cinema it determines whether the render is usable at all.

### 5.0 YouTube acquisition (optional source)

Sources may be supplied as a local file or fetched from YouTube via
[yt-dlp](https://github.com/yt-dlp/yt-dlp), used as a Python library rather than a subprocess so
results are structured data.

- **Search** uses yt-dlp's built-in `ytsearchN:` extractor. No API key, no Google account, no
  additional dependency beyond yt-dlp itself. Results are listed flat (no download) with title,
  duration, uploader, view count, and available resolutions.
- **Download** selects the highest-quality source available, preferring VP9/AV1 over H.264 at equal
  resolution because YouTube allocates them more bitrate. ffmpeg merges the separate video and
  audio streams. The format selector is user-overridable.
- **Downloaded files are ordinary sources.** They flow into §5.1 analysis unchanged, and carry no
  special-case handling anywhere downstream.

**YouTube sources make the deblock pre-pass (§5.3) more important, not less.** YouTube re-encodes
aggressively at low bitrates, so blocking and mosquito noise are typically worse than on a disc
rip. The GUI biases the recommended deblock strength upward when the source is a YouTube download.

Legal note: downloading may conflict with YouTube's terms of service, and much of the material is
uploaded without the rights-holder's permission. The application does not attempt to assess this;
responsibility rests with the user.

### 5.1 Source analysis (automatic)

On load, the app probes the source with ffmpeg `idet` plus container metadata and reports:
field order and interlacing, telecine pulldown pattern, sample aspect ratio, color primaries /
transfer / matrix, bit depth, and an estimated compression-artifact and grain level. Findings are
shown to the user with a recommended pre-pass, which can be overridden.

**This auto-detection is the highest-value single feature in the application for this footage type.**

### 5.2 Deinterlace / inverse telecine

Restoration consensus is unambiguous: *"always deinterlace and inverse telecine the footage first
before upscaling,"* because *"when you upscale an interlaced source the interlacing becomes more
pronounced."* Comb artifacts baked in by the upscaler cannot be removed later.

Critically, **the correct operation is source-dependent**:

- **Film-sourced 30i (telecined).** Requires **inverse telecine**, 30i→24p, via ffmpeg
  `fieldmatch` + `decimate`. QTGMC is the *wrong* tool here — most films are 24p progressive and
  early DVDs merely carry them as 29.97i.
- **True interlaced video.** Requires deinterlacing. ffmpeg `bwdif` ships as the built-in default.
- **Progressive.** Skipped.

**QTGMC is auto-detected, not required.** If VapourSynth and QTGMC are present on the system the
GUI unlocks them as a higher-quality option for true-interlaced sources; otherwise `bwdif` is used.
No hard dependency, since a fragile Windows VapourSynth install must never block the app running.

### 5.3 Deblock and temporal degrain

- **Deblock** for compressed sources, since SR amplifies blocking and mosquito noise.
- **Temporal degrain**, motion-compensated, **default light** per §3.2. This also suppresses the
  frame-to-frame texture crawl that single-image SR produces on jewelry specular highlights.

### 5.4 Color management

Direct impact on the stated skin-tone priority. SD sources are BT.601, HD is BT.709; decoding to
RGB and re-encoding without carrying `color_primaries`, `color_trc`, and `colorspace` through
applies an unmanaged matrix shift whose visible symptom is wrong skin tones.

- Color metadata is read at decode and written explicitly at encode.
- Any required conversion is explicit and tagged, never implicit.
- **Sample/display aspect ratio is honored** — anamorphic DVD content is unsqueezed correctly
  rather than delivering stretched faces.
- **10-bit output** by default where the encoder supports it, eliminating banding in smoke and
  gradient lighting at negligible cost given fp16 internals.

---

## 6. Execution backend

**Hybrid — PyTorch fp16 core, opt-in TensorRT for the upscaler only.**

- **Core:** PyTorch 2.x + CUDA 12.x, `torch.float16`, `channels_last`, `torch.inference_mode()`,
  cuDNN benchmark on. Works for every roster model with zero conversion. Must always work.
- **Opt-in:** "Compile with TensorRT" for the upscaler, ~1.4–1.8x. Engines cached per
  (model, tile size, precision). First compile costs minutes; the GUI says so before starting.
- **RIFE stays on PyTorch** — its warping and grid-sample ops export to ONNX/TRT poorly.
- **Batch inference** is a tunable; small models at low resolution under-utilize the GPU at batch 1.

Fallback chain: TensorRT → CUDA fp16 → CUDA fp32 → CPU. Every downgrade is logged and surfaced.

---

## 7. Pipeline architecture

### 7.1 Zero-disk-cache streaming

No PNG frame dumps, ever.

```
decode ──▶ [source analysis: interlaced? telecined? SAR? colorspace?]
       ──▶ IVTC / deinterlace ──▶ deblock ──▶ temporal degrain
       ──▶ TILED UPSCALE ──▶ face blend (opt) ──▶ detail retention ──▶ re-grain
       ──▶ interpolate (opt) ──▶ 10-bit encode, color metadata preserved
```

- Decode and encode run as ffmpeg subprocesses over rawvideo pipes; Python moves bytes, never
  loops over pixels.
- Two CUDA streams with pinned host staging buffers overlap H2D / compute / D2H.
- Bounded queues apply backpressure, so RAM stays flat regardless of clip length.
- Audio, subtitle, and chapter streams pass through with `-c:a copy -c:s copy`.

### 7.2 Segment preview

Renders a short user-chosen segment (default 10 s) under up to three setting variants and presents
them side by side. This is the primary defense against wasting hours on wrong settings, and it is
cheaper and more direct than evaluating a full 2K pass.

### 7.3 Dual-pass 2K → 4K

- **Pass A → 2K.** Upscale to ~2560-wide, write a high-bitrate intermediate (NVENC HEVC at
  near-visually-lossless CQ, or FFV1 for true lossless).
- **Evaluate** via A/B compare.
- **Pass B → 4K.** Consumes the intermediate, applies the second upscale, encodes final output.

Passes are independently resumable and may use different tiers — commonly Balanced for A and Turbo
for B, since the second doubling carries less recoverable detail.

### 7.4 Resume

Job state (input hash, settings, last completed frame, intermediate path) is journaled every N
frames. An interrupted job offers resume-from-frame. This matters when jobs run for hours.

---

## 8. VRAM safety

**Adaptive auto-tiling with OOM retry and CPU fallback. The application must never terminate with
an out-of-memory error.**

1. **Probe.** Query free VRAM via `torch.cuda.mem_get_info`, subtract configurable headroom
   (**default 1 GB** — a laptop GPU also drives the desktop under WDDM, so 512 MB is too
   optimistic), select the largest tile from a candidate ladder that fits measured per-pixel cost.
2. **Escalate.** On `torch.cuda.OutOfMemoryError`: empty cache, halve the tile, retry the same
   frame, down to a floor size.
3. **Fall back.** If the floor tile still OOMs, process that frame on CPU. 64 GB RAM makes this
   viable — slow, but per-frame rather than per-job, and it does not crash.
4. **Recover.** After N consecutive successes, step the tile back up. A transient VRAM spike from
   another application must not permanently degrade a multi-hour run.

Tiles use configurable overlap (default 16 px) with feathered blending. Overlap is kept small
deliberately — it is pure recomputed waste.

An explicit **CPU-only mode** exists for machines without CUDA, using fp32 and all cores. It is
slow and the GUI says so plainly.

---

## 9. Module structure

Single entrypoint, small focused modules, each independently testable.

| Module | Responsibility | Key interface |
|---|---|---|
| `main.py` | Entrypoint, arg parsing, GUI bootstrap | — |
| `sources.py` | YouTube search and download via yt-dlp; yields a local file path | `search(query, n)`, `download(url, dest)` |
| `analyze.py` | Source probing: interlacing, pulldown, SAR, colorspace, grain estimate | `SourceProfile.probe(path)` |
| `restore.py` | IVTC/deinterlace, deblock, temporal degrain, re-grain, detail retention | `Restorer.pre(frame)`, `.post(frame)` |
| `models.py` | Roster manifest, download + verify, spandrel loading, custom-dir scan | `ModelRegistry.list()`, `.load(id)` |
| `vram.py` | Probing, tile ladder, OOM retry policy, device selection | `TilePlanner.plan()`, `run_tiled()` |
| `video_io.py` | ffmpeg decode/encode subprocesses, ring buffers, color + stream passthrough | `Decoder.frames()`, `Encoder.write()` |
| `upscale.py` | Spatial upscale, face pass + blending, dual-pass orchestration | `Upscaler.process(frame)` |
| `interpolate.py` | RIFE inference, timestep sampling, scene-change detection, target-fps logic | `Interpolator.between(a, b, t)` |
| `pipeline.py` | Job model, stage graph, journaling/resume, segment preview, progress events | `Job.run()` |
| `gui.py` | PySide6 window, drag-drop, settings, progress, A/B compare | — |

The engine layer emits progress events and never imports Qt. The GUI never calls CUDA directly.
This boundary enables headless benchmarking (§11) and keeps the render loop off the UI thread.

---

## 10. GUI specification

**Framework: PySide6** (LGPL, compatible with a free local build; mature, good threading story).

- **Input:** drag-and-drop plus file browser. `.mp4 .mkv .mov .avi .webm` and
  `.png .jpg .jpeg .webp .tif`. Batch queue supported.
- **Source panel:** auto-detected profile (interlacing, pulldown, SAR, colorspace, grain) with the
  recommended pre-pass and an override.
- **Restore panel:** IVTC/deinterlace method (QTGMC shown only if detected), deblock strength,
  degrain strength (default light), re-grain toggle and amount.
- **Texture panel:** Original Detail Retention slider (default 25 %), face-restore toggle + blend
  (default off for video), CodeFormer fidelity, sharpening (default off).
- **Model panel:** tier selector with plain-language speed/quality notes, live measured fps, scale
  factor, custom-model dropdown from `models/custom/`.
- **FPS panel:** mode (multiplier / target FPS), presets, scene-change sensitivity, stage order.
- **Pipeline panel:** single-pass vs. dual-pass 2K→4K, intermediate format, per-pass tier,
  segment-preview launcher.
- **Performance panel:** backend (Auto / TensorRT / CUDA fp16 / CUDA fp32 / CPU), tile override,
  VRAM headroom, batch size, NVENC vs. x264/x265, encoder quality, bit depth.
- **Progress:** overall and per-stage bars, fps, ETA, frames done/total, live VRAM gauge, current
  tile size, log pane. Pause / Resume / Cancel.
- **Compare view:** side-by-side and swipe-wipe A/B at a chosen frame.
- **Model manager:** presence, size, verification status, download action.

All long work runs on `QThread` workers communicating via signals. The UI thread never blocks.

---

## 11. Testing

- **Unit:** tile/detile round-trips losslessly at overlap 0; scene-change detector fires on
  synthetic cuts but not fast pans; VRAM planner picks expected tile for a mocked budget; OOM retry
  halves and recovers under injected fault; manifest verification rejects a corrupted file; pulldown
  detection identifies a synthetic 3:2 telecine pattern; RIFE timestep sampling produces correct
  frame counts for non-integer ratios such as 24→60.
- **Integration:** 5-second synthetic clip end-to-end per tier; audio/subtitle passthrough
  preserved; color metadata round-trips unchanged; anamorphic source produces correct display
  dimensions; dual-pass yields correct final dimensions; resume after kill matches an uninterrupted
  run.
- **Benchmark harness:** headless CLI reporting fps and peak VRAM per tier, validating §1.1 on the
  real machine and catching regressions.
- **Manual/visual:** skin-texture and jewelry-specularity checklist on real footage. Not automatable.

---

## 12. Model manifest and acquisition

**Hybrid delivery.** A JSON manifest defines each model: id, tier, architecture, scale, source URL,
SHA-256, size. On first selection the app downloads with a progress bar and verifies the hash; a
mismatch is a hard failure with a clear message. `models/custom/` is additionally scanned for
drop-in weights loaded via spandrel without manifest entries.

Canonical sources: Real-ESRGAN and GFPGAN GitHub releases, CodeFormer GitHub releases,
Practical-RIFE weights, and OpenModelDB-listed community weights for SPAN/RealPLKSR/ESRGAN.

**Implementation requirement:** exact URLs and SHA-256 values must be resolved by actually fetching
each file during implementation and recording the computed digest. Hashes must not be invented. Any
model whose source cannot be verified is omitted from the shipped manifest and documented as a
manual drop-in.

---

## 13. Phase 2 — domain fine-tuning (documented, not built)

Setup guide includes a neosr recipe for fine-tuning the SPAN or Compact tier on Indian-cinema
frames: dataset extraction, degradation modelling matched to target prints, fp16 training at
batch 4–8 within 6 GB, learning rate 1e-4, and where to drop the resulting `.pth`. No code required
— the `models/custom/` scan already handles it.

---

## 14. Environment

- **Python 3.12 venv.** System Python is 3.14.3, for which PyTorch has no stable wheels. The guide
  installs 3.12 alongside; the system install is not modified.
- PyTorch 2.x + CUDA 12.x wheels, spandrel, PySide6, numpy, opencv-python, Pillow, yt-dlp.
- FFmpeg with NVENC — already present (build N-123741).
- Optional, auto-detected: VapourSynth + QTGMC; torch-tensorrt.

---

## 15. Risks

| Risk | Mitigation |
|---|---|
| Wrong pre-pass choice bakes in comb artifacts | Auto-detection with explicit user confirmation before a long run; segment preview |
| Community model URLs move or vanish | Manifest verification fails loudly; manual drop-in documented |
| TensorRT compile fails on some model/tile combos | Opt-in only; automatic fallback to CUDA fp16 |
| Over-degraining flattens skin texture | Default light; re-grain and detail-retention counteract; §3 doctrine |
| NVENC quality insufficient for archival work | x264/x265 CRF selectable; FFV1 lossless intermediate |
| RIFE ghosting on rapid choreography | Mandatory scene-change detection, adjustable sensitivity |
| Laptop thermal throttling over multi-hour renders | Live fps readout makes degradation visible; pause/resume |
