# Enhancer

Local GPU video and image upscaler, tuned for restoring Indian cinema footage — skin texture, jewellery detail, and high-motion choreography.

Runs entirely offline. No cloud services, no API keys, no telemetry.

**Status:** working end to end — engine, restoration, frame interpolation, resumable rendering, and a desktop window. 290 tests. See [Roadmap](#roadmap) for what remains.

---

## Features

- **Desktop window** — drag-and-drop, automatic source analysis, live progress, safe cancel, and a ten-second preview so a bad setting costs a minute rather than a night
- **Frame interpolation** — any target frame rate, including non-integer ratios like 24→60, with cut detection so fast-cut footage never ghosts between shots
- **Texture-preserving** — degrain, upscale, then restore the source's real high-frequency detail and re-grain, so skin reads as photographed rather than polished
- **Source-aware restoration** — detects interlacing versus 3:2 telecine and applies the correct correction, since getting that wrong damages every frame irreversibly
- **Resumable** — renders are written as independently complete segments, so an interruption costs at most one segment, not the whole job
- **Streaming pipeline** — frames move through ffmpeg pipes and never touch disk, avoiding the three-pass PNG cache most tools use
- **OOM-proof** — adaptive tiling shrinks on memory pressure and falls back to CPU per frame rather than crashing, including on Windows, where the driver oversubscribes silently instead of raising
- **Any model** — drop any [OpenModelDB](https://openmodeldb.info/) `.pth` into `models/custom/` and it appears automatically; architecture is auto-detected
- **Colour-correct** — BT.601/709 primaries, transfer, matrix, and sample aspect ratio are preserved end to end
- **10-bit NVENC output** with audio, subtitle, and chapter passthrough
- **YouTube source** — search and download directly, no API key required

---

## Requirements

Windows only at present.

| | |
|---|---|
| OS | Windows 10/11 |
| GPU | NVIDIA, 6 GB VRAM or more (CUDA 12.x driver) |
| Python | 3.12 — **not** 3.13+, which has no PyTorch wheels |
| GUI | PySide6 (optional; the CLI works without it) |
| ffmpeg | on `PATH`, with NVENC support |

Verify ffmpeg:

```powershell
ffmpeg -version
```

---

## Setup

**1. Install Python 3.12**

```powershell
winget install Python.Python.3.12
```

**2. Create the virtual environment**

```powershell
py -3.12 -m venv .venv
```

**3. Install PyTorch with CUDA**

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**4. Install remaining dependencies**

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

**5. Confirm the GPU is visible**

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True NVIDIA GeForce RTX 3060 Laptop GPU` (or your card). If this prints `False`, resolve the CUDA install before continuing.

**6. Add a model**

Create `models\custom\` and download at least one upscaling model into it:

```powershell
mkdir models\custom
curl.exe -L -o models\custom\realesr-general-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
```

More models are listed in `src/enhancer/manifest.json`. Any architecture supported by [spandrel](https://github.com/chaiNNer-org/spandrel) will load.

---

## Usage

**Open the desktop window**

```powershell
.venv\Scripts\python.exe -m enhancer.cli gui
```

Drop a video in and it reports resolution, frame rate, colour space, scan type, grain and compression artifacts, then recommends a pre-pass. Sliders for degrain, detail retention, re-grain and deblock; frame-rate conversion; live progress with fps and ETA; cancel.

**Preview before committing.** The *Preview 10s* button renders ten seconds through the identical pipeline. A 1080p→4K feature is ~17 hours, so the expensive mistake is not a slow render but a slow render with the wrong settings. Ten seconds takes under a minute and answers the same question.

Cancel is safe: it stops after the current frame, keeps every completed segment, and re-running resumes.

**List available models**

```powershell
.venv\Scripts\python.exe -m enhancer.cli models
```

**Benchmark before committing to a long render**

```powershell
.venv\Scripts\python.exe -m enhancer.cli bench models\custom\realesr-general-x4v3.pth --width 1920 --height 1080 --frames 20
```

Reports frames per second, peak VRAM, selected tile size, and an estimated time for a two-hour feature.

**Inspect a source before rendering**

```powershell
.venv\Scripts\python.exe -m enhancer.cli analyze input.mp4
```

Reports resolution, colour space, sample aspect ratio, scan type, grain level and compression blockiness, then recommends a pre-pass. Worth running on anything from a disc rip — a film-sourced 30i DVD needs inverse telecine, not deinterlacing, and choosing wrong softens every frame permanently.

**Upscale a video**

```powershell
.venv\Scripts\python.exe -m enhancer.cli video models\custom\realesr-general-x4v3.pth input.mp4 output.mkv
```

Restoration is on by default. Scan correction is chosen automatically from the detected source type.

| Flag | Default | Effect |
|---|---|---|
| `--degrain` | 0.25 | Noise reduction before upscaling. Deliberately light — grain and skin micro-texture share the same frequencies, so heavy degraining is what produces waxy skin |
| `--detail-retention` | 0.25 | Blends the source's real high-frequency detail back over the model output. The only stage that restores photographed texture rather than inventing it |
| `--regrain` | 0.6 | Adds midtone-weighted film grain after upscaling, with a per-frame seed so it moves like film rather than sitting static |
| `--deblock` | 0.0 | Compression artifact removal. Raise it for YouTube sources and low-bitrate rips |
| `--no-restore` | off | Skip all restoration and texture work |

Restoration costs roughly 35% throughput. Use `--no-restore` for a fast preview, then render properly.

**Increase the frame rate**

```powershell
.venv\Scripts\python.exe -m enhancer.cli video models\custom\2xParimgCompact.pth input.mp4 output.mkv --fps 60
```

`--fps` sets an absolute target; `--interpolate 2` sets a multiplier instead. Non-integer ratios work — 24→60 is 2.5x, synthesized at fractional timesteps rather than by inserting whole frames.

Cut detection is always on. Interpolating across a hard cut produces a ghost-morph between two unrelated shots, which in cut-heavy choreography would be visible many times per song. At a cut the nearest real frame is duplicated instead. Tune with `--scene-threshold` (0–1, lower detects more cuts).

Requires RIFE weights in `models\rife\`:

```powershell
mkdir models\rife
curl.exe -L -o models\rife\flownet.pkl https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl
```

**Resume an interrupted render**

Re-run the identical command. Completed segments are skipped and the render picks up where it stopped.

```powershell
.venv\Scripts\python.exe -m enhancer.cli video models\custom\realesr-general-x4v3.pth input.mp4 output.mkv
```

State lives in `output.job\` beside the output, or wherever `--job-dir` points. `--segment-frames` controls how much work an interruption can cost — the default of 500 is roughly three minutes at 1080p→4K.

Changing settings between runs is refused rather than silently producing a file with a visible seam mid-render. Start a new job, or restore the original settings.

**Search YouTube**

```powershell
.venv\Scripts\python.exe -m enhancer.cli search "old tamil movie song" --limit 5
```

**Download from YouTube**

```powershell
.venv\Scripts\python.exe -m enhancer.cli fetch "https://www.youtube.com/watch?v=VIDEO_ID"
```

Downloading may conflict with YouTube's terms of service, and much material is uploaded without the rights-holder's permission. Responsibility rests with the user.

**Force CPU** — add `--cpu` to any command. Slow, but works without a GPU.

---

## Performance

Measured on an RTX 3060 Laptop (6 GB), 2x models, fp16.

**480p → 960p**

| Model | fps | 2-hour feature |
|---|---|---|
| Compact | 19.5 | ~2.5 h |
| SPAN | 13.7 | ~3.5 h |
| RRDB-ESRGAN | 2.0 | ~24 h |

**1080p → 4K**

| Model | fps | 2-hour feature |
|---|---|---|
| Compact | 2.9 | ~17 h |
| SPAN | 2.1 | ~22 h |
| RRDB-ESRGAN | 0.3 | ~155 h |

Roughly real time at 480p; roughly 8x slower than real time at 4K. A one-minute clip at 1080p→4K takes about eight minutes.

Benchmark your own hardware with the `bench` command before starting a long job. Model choice matters far more than any other setting: Compact and SPAN are roughly ten times faster than RRDB-ESRGAN.

---

## Project structure

```
src/enhancer/
  vram.py       tiling, OOM recovery, device probing
  models.py     model manifest, download, verification, loading
  video_io.py   ffprobe analysis, ffmpeg decode/encode pipes
  upscale.py    frame processing with CPU fallback
  bench.py      throughput and VRAM measurement
  sources.py    YouTube search and download
  cli.py        command-line entrypoint
docs/
  specs/        design specification
  plans/        implementation plans
```

Run the test suite:

```powershell
.venv\Scripts\python.exe -m pytest
```

---

## Roadmap

| Stage | Contents | Status |
|---|---|---|
| Core engine | Tiling, OOM safety, streaming I/O, model loading, benchmarking, YouTube | Complete |
| Resilience | Resumable renders, physical VRAM ceiling | Complete |
| Restoration | Deinterlace/IVTC, deblock, degrain, re-grain, detail retention | Complete |
| Interpolation | RIFE frame interpolation, scene-change detection, target FPS | Complete |
| Interface | Desktop GUI, segment preview | Complete |
| Comparison | Dual-pass 2K→4K, A/B compare view | Planned |

---

## License

MIT
