# Enhancer

Local GPU video and image upscaler, tuned for restoring Indian cinema footage — skin texture, jewellery detail, and high-motion choreography.

Runs entirely offline. No cloud services, no API keys, no telemetry.

**Status:** working end to end — engine, restoration, frame interpolation, resumable rendering, and a desktop window with before/after comparison. 1031 tests. See [Roadmap](#roadmap) for what remains.

---

## Features

- **Desktop window** — drag-and-drop, automatic source analysis, live progress, safe cancel, and a ten-second preview so a bad setting costs a minute rather than a night
- **Before and after** — put one frame through the current settings in about a second and compare it with the original by swipe, split or in-place toggle, zoomed to 100% or beyond; the original is enlarged to match, so you judge quality rather than size
- **Playback** — stream a finished result against its source in step, because grain that pulses and skin that slides in and out of focus only show in motion
- **Light or dark** — follows Windows, overridable, remembered
- **Frame interpolation** — any target frame rate, including non-integer ratios like 24→60, with cut detection so fast-cut footage never ghosts between shots
- **Texture-preserving** — degrain, upscale, then restore the source's real high-frequency detail and re-grain, so skin reads as photographed rather than polished
- **Source-aware restoration** — detects interlacing versus 3:2 telecine and applies the correct correction, since getting that wrong damages every frame irreversibly
- **Resumable** — renders are written as independently complete segments, so an interruption costs at most one segment, not the whole job
- **Streaming pipeline** — frames move through ffmpeg pipes and never touch disk, avoiding the three-pass PNG cache most tools use
- **OOM-proof** — adaptive tiling shrinks on memory pressure and falls back to CPU per frame rather than crashing, including on Windows, where the driver oversubscribes silently instead of raising
- **Your files are safe** — a render refuses to write over its own source, including when two paths differ only in capitalisation, and never resumes an unfinished job that belongs to a different file
- **Any model** — drop any [OpenModelDB](https://openmodeldb.info/) `.pth` into `models/custom/` and it appears automatically; architecture is auto-detected
- **Colour-correct** — BT.601/709 primaries, transfer, matrix, and sample aspect ratio are preserved end to end
- **Runs on what you have** — the graphics card, encoder and memory ladder are detected, and time estimates calibrate themselves from your own finished renders rather than assuming the machine this was built on
- **10-bit output** with audio, subtitle, and chapter passthrough, through whichever hardware encoder the machine has, or `libx265` when it has none
- **Images and video** — `.png`, `.jpg`, `.webp`, `.bmp`, `.tif` stills alongside `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, with alpha preserved
- **YouTube source** — search and download directly, no API key required

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11. macOS and Linux are supported in code but not tested here |
| GPU | Optional. NVIDIA (CUDA), Apple Silicon and Intel Arc are used when present; otherwise the processor, which works but is tens of times slower |
| Python | 3.12 — **not** 3.13+, which has no PyTorch wheels |
| GUI | PySide6 (optional; the CLI works without it) |
| ffmpeg | on `PATH`. Any build — a hardware encoder is used when there is one, and `libx265` otherwise |

Check everything at once, and get the command that installs whatever is missing:

```powershell
.venv\Scripts\python.exe -m enhancer.cli check
```

It reports the processor, memory, graphics card and the encoder chosen for it, and exits non-zero when something essential is absent. The desktop window shows the same under **System**, and says so by itself at startup if a render could not run.

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

**3. Install PyTorch**

With an NVIDIA card:

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

On Apple Silicon, or with no graphics card at all, the plain build is the right one:

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision
```

Take care not to install the plain build on an NVIDIA machine by accident. It is the default, it works, and it silently runs everything on the processor at a fraction of the speed. Step 5 catches that.

**4. Install remaining dependencies**

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

**5. Confirm what will actually be used**

```powershell
.venv\Scripts\python.exe -m enhancer.cli check
```

This names the processor, memory, graphics card and the encoder chosen for it, and lists anything missing with the command that installs it. If it reports no graphics acceleration on a machine that has a card, the processor-only PyTorch went in — go back to step 3.

**6. Add a model**

See what is available, then download one. Every catalogue entry is verified against a SHA-256 recorded from a real download:

```powershell
.venv\Scripts\python.exe -m enhancer.cli models
```

```powershell
.venv\Scripts\python.exe -m enhancer.cli models --get 2xParimgCompact
```

Good starting points: `2xParimgCompact` (fastest), `2xModernSpanimationV1` (balanced), `RealESRGAN_x2plus` (slowest, stills only).

You can also drop any [OpenModelDB](https://openmodeldb.info/) `.pth` straight into `models\custom\` — anything [spandrel](https://github.com/chaiNNer-org/spandrel) supports loads with no configuration.

---

## Building a standalone executable

Optional. Produces `dist\Enhancer\Enhancer.exe`, which runs without Python installed.

```powershell
.\build_exe.bat
```

Takes 10–20 minutes and produces roughly 5 GB, nearly all of it PyTorch and the CUDA runtime — the graphics code is the application, so there is no trimming it meaningfully.

It is a folder, not a single file, deliberately: a one-file build of this size unpacks itself to a temporary directory on every launch and takes half a minute to appear. Copy the whole `dist\Enhancer` folder to move it. Models are read from `models\custom` and `models\rife` beside the executable, wherever it is launched from, and a `models` folder in the current directory takes precedence so a project can keep its own.

ffmpeg is still required on `PATH`; it is not bundled.

---

## Usage

**Open the desktop window**

```powershell
.venv\Scripts\python.exe -m enhancer.cli gui
```

Drop a video in and it reports resolution, frame rate, colour space, scan type, grain and compression artifacts, then recommends a pre-pass. Sliders for degrain, detail retention, re-grain and deblock; frame-rate conversion; live progress with fps and ETA; cancel.

**Check one frame first.** *Compare this frame* puts a single frame through the settings currently set and shows it against the original, in about a second. Pick a lit close-up rather than a wide shot: waxy skin is invisible in a crowd scene and obvious on a face.

Drag the divider across the picture, or switch to *Split* for side by side, or *Toggle* to flip between the two in place — flipping is the most sensitive of the three, because the eye catches change more readily than difference. Judge at 100% or above; fitted to the window, a waxy face and a detailed one look the same.

The original is enlarged to the result's size before either is drawn, so what you see is the difference in quality and not the difference in size. That matters: a 1080p original shown next to a 4K result flatters any model for free.

**Preview before committing.** The *Preview 10s* button renders ten seconds through the identical pipeline. A 1080p→4K feature is ~17 hours, so the expensive mistake is not a slow render but a slow render with the wrong settings. Ten seconds takes under a minute and answers the same question.

**Then watch it move.** When a preview or render finishes it attaches itself to the transport under the picture; *Compare with* opens any earlier result instead. Press *Play* and both clips run in step under the divider.

Two faults only appear in motion and are invisible in a still: grain that pulses from frame to frame, and skin that slides between detailed and waxy as the light changes. *Loop* is worth using on a difficult shot.

Playback decodes at the size the pane shows, which is what keeps a 4K comparison watchable — the full picture is 24.9 MB a frame, and sixty of those a second is more than a pipe will carry. Zoom is therefore not the tool here: to inspect texture, pause and use *Compare this frame*, which always works at full resolution.

Cancel is safe: it stops after the current frame, keeps every completed segment, and re-running resumes.

**Your files are not overwritten by accident.** A render refuses to start when the output would be the same file as the source — including when the two differ only in capitalisation, which on Windows is the same file and which ffmpeg's own check does not catch. An existing result is kept too: replacing one needs `--force`, unless the unfinished render beside it belongs to this job, which is an ordinary resume.

An unfinished render is only continued when it came from the same file. Re-render after replacing a source with a better rip and it starts again rather than splicing the old footage into the new.

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

**Upscale an image**

```powershell
.venv\Scripts\python.exe -m enhancer.cli video models\custom\2xParimgCompact.pth photo.jpg photo_4k.jpg
```

Stills take a separate path: no frame rate, no segments, no resume. The output format follows the extension you give it, and PNG alpha is preserved.

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
| `--force`, `-f` | off | Replace an existing output file. Not needed to resume an unfinished render, which is recognised by its own journal |

Restoration costs roughly 35% throughput. Use `--no-restore` for a fast preview, then render properly.

**Two-stage upscaling**

```powershell
.venv\Scripts\python.exe -m enhancer.cli video models\custom\2xParimgCompact.pth input.mp4 output.mkv --dual-pass
```

Upscales in two stages and keeps the halfway file so you can look at it before the second stage runs. `--pass2-model` lets each stage use a different model — commonly a texture-preserving one for the first doubling and a fast one for the second, where there is less recoverable detail left to find. Each stage has its own job directory, so an interruption in the second never re-runs the first.

Restoration and interpolation apply to the first pass only: degraining an already-degrained frame flattens it further, and interpolating twice compounds synthesis errors.

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
  vram.py        tiling, OOM recovery, device probing
  models.py      model manifest, download, verification, loading
  video_io.py    ffprobe analysis, ffmpeg decode/encode pipes
  upscale.py     frame processing with CPU fallback
  restore.py     filter graph, detail retention, re-grain
  analyze.py     scan type, grain, blockiness
  timing.py      output frame planning for any frame-rate ratio
  scenes.py      cut detection
  interpolate.py RIFE stream driver
  forecast.py    predicted output, from measured throughput
  segments.py    atomic segment writing, concat assembly
  jobs.py        resume journal
  queue.py       render queue
  images.py      still-image path
  compare.py     one frame before and after
  playback.py    two streams aligned by time
  viewer.py      swipe, split and toggle comparison widget
  window.py      desktop window
  gui.py         one render, cancellable, with no Qt in it
  requests.py    the settings a render depends on
  theme.py       palette, light and dark
  help_text.py   every user-facing explanation
  splash.py      startup feedback while the libraries load
  single.py      one instance at a time
  proc.py        subprocess launch without console windows
  rife/          vendored RIFE interpolation network
  bench.py       throughput and VRAM measurement
  sources.py     YouTube search and download
  cli.py         command-line entrypoint
docs/
  specs/         design specification
  plans/         implementation plans
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
| Interface | Desktop GUI, segment preview, render queue | Complete |
| Comparison | Single-frame before/after, synced playback | Complete |
| Acceleration | TensorRT engine build | Planned |

---

## License

MIT
