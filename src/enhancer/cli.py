"""Command-line entrypoint for the headless engine."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .bench import benchmark
from .interpolate import render_from_plan
from .jobs import JobState, SettingsMismatch
from .models import load_model, scan_custom_dir
from .segments import assemble, segment_path, write_segment
from .timing import plan_output_frames
from .upscale import Upscaler
from .video_io import Decoder, SourceProfile
from .vram import choose_tile, free_vram_bytes, select_device

DEFAULT_BYTES_PER_OUTPUT_PIXEL = 64
DEFAULT_SEGMENT_FRAMES = 500
DEFAULT_SCENE_THRESHOLD = 0.30

# Interpolation holds a segment's upscaled frames in memory at once. This caps
# that working set so segment size adapts to output resolution.
INTERPOLATION_RAM_BUDGET = 1024 ** 3


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


def render_resumable(
    profile: SourceProfile,
    upscaler,
    output: str | Path,
    job_dir: str | Path,
    segment_frames: int = DEFAULT_SEGMENT_FRAMES,
    settings: dict | None = None,
    on_progress=None,
    video_filter: str = "",
    texture=None,
    interpolate_to: float | None = None,
    flow_model=None,
    scene_threshold: float | None = None,
) -> Path:
    """Render `profile` to `output`, resuming any interrupted prior attempt.

    Output is written as independently complete segments so an interruption
    costs at most one segment. The final file is a stream-copy concat.

    `video_filter` is applied inside the decoder, before upscaling. If it
    contains a decimating filter (inverse telecine's `decimate`), the true
    output frame count is unknowable from `profile.frame_count` alone and
    segment boundaries would not line up with what ffmpeg actually decoded,
    since `-ss` seeks in *input* time while decimation happens after that
    seek. Rather than risk a misaligned resume, the whole source is rendered
    as a single segment in that case, which simply forfeits resumability for
    this job.
    """
    job_dir = Path(job_dir)
    settings = settings or {}
    scale = upscaler.scale

    if profile.frame_count <= 0:
        # Some containers report neither a frame count nor a duration. Without
        # one there is nothing to divide into segments, and the failure would
        # otherwise surface as an opaque ffmpeg concat error much later.
        raise ValueError(
            f"could not determine a frame count for {profile.path}. The file may "
            f"be corrupt, or the container may not report duration. Try "
            f"remuxing it first: ffmpeg -i input -c copy remuxed.mkv"
        )

    if "decimate" in video_filter:
        segment_frames = profile.frame_count

    # Unlike inverse telecine, interpolation's output length is fully determined
    # in advance, so segments can be sized in OUTPUT frames and mapped back to
    # the source frames each one needs. Resume therefore survives interpolation,
    # which matters because interpolation is when renders get longest.
    plan = None
    out_fps = profile.fps
    total_frames = profile.frame_count
    if interpolate_to is not None:
        if flow_model is None:
            raise ValueError("interpolate_to was given without a flow_model")
        if "decimate" in video_filter:
            # The plan is built from the probed frame count, but decimation
            # drops roughly one frame in five inside ffmpeg, so the decoder
            # supplies fewer frames than the plan indexes. Knowing the true
            # post-filter count would take a separate counting pass.
            raise ValueError(
                "frame interpolation cannot be combined with inverse telecine "
                "in one pass, because decimation changes the frame count that "
                "interpolation timing is derived from. Run inverse telecine "
                "first, then interpolate the result:\n"
                "  enhancer video MODEL in.mkv ivtc.mkv\n"
                "  enhancer video MODEL ivtc.mkv out.mkv --fps 60 --no-restore"
            )
        plan = plan_output_frames(profile.fps, interpolate_to, profile.frame_count)
        total_frames = len(plan)
        out_fps = interpolate_to

        # Interpolation materialises a segment's upscaled frames in memory, so
        # segment size has to be bounded by output resolution rather than left
        # at the default. At 1080p to 4K the default 500 would hold ~4.6 GB.
        per_frame = profile.width * scale * profile.height * scale * 3
        affordable_src = max(24, INTERPOLATION_RAM_BUDGET // max(1, per_frame))
        cap = max(30, int(affordable_src * interpolate_to / profile.fps))
        segment_frames = min(segment_frames, cap)

    if (job_dir / "job.json").exists():
        job = JobState.load(job_dir, settings=settings)
    else:
        job = JobState.create(
            job_dir, source=str(profile.path), settings=settings,
            total_frames=total_frames, segment_frames=segment_frames,
        )

    out_w = profile.width * scale
    out_h = profile.height * scale

    for index in range(job.segment_count):
        if index in job.completed_segments and segment_path(job_dir, index).exists():
            continue

        start = job.start_frame_for(index)
        count = job.frames_in_segment(index)

        if plan is None:
            src_start, src_count = start, count
            segment_plan = None
        else:
            # Decode exactly the source frames this output segment brackets.
            segment_plan = plan[start:start + count]
            src_start = min(f.left for f in segment_plan)
            src_count = max(f.right for f in segment_plan) - src_start + 1

        decoder = Decoder(
            profile, start_frame=src_start, max_frames=src_count,
            video_filter=video_filter,
        )

        def upscaled_frames(decoder=decoder, src_start=src_start, grain=True):
            """Upscale each source frame and restore its real detail.

            Grain is applied here only when nothing downstream will blend these
            frames together. When interpolating it is deferred to the output
            stage, so every output frame gets full-strength grain.
            """
            for i, frame in enumerate(decoder.frames()):
                out = upscaler.process(frame)
                if texture is not None and texture.enabled:
                    out = texture.apply_detail(out, frame)
                    if grain:
                        out = texture.apply_grain(out, index=src_start + i)
                yield out

        def processed(
            decoder=decoder, start=start, src_start=src_start, segment_plan=segment_plan
        ):
            if segment_plan is None:
                frames = upscaled_frames(decoder, src_start, grain=True)
            else:
                # Upscale first, then interpolate: the flow model blends
                # already-detailed frames, and the upscaler never runs on
                # synthesized content.
                frames = render_from_plan(
                    upscaled_frames(decoder, src_start, grain=False), src_start,
                    segment_plan, flow_model, scene_threshold,
                )
            for i, frame in enumerate(frames):
                if segment_plan is not None and texture is not None and texture.grain_enabled:
                    # Grain after interpolation, keyed on the OUTPUT index.
                    frame = texture.apply_grain(frame, index=start + i)
                yield frame
                if on_progress:
                    on_progress(start + i + 1, job.total_frames)

        write_segment(
            segment_path(job_dir, index), processed(),
            width=out_w, height=out_h, fps=out_fps, source=profile,
        )
        job.mark_complete(index)

    return assemble(job_dir, job.segment_count, output, profile)


def rife_weights() -> list[Path]:
    """Available RIFE weight files, newest-sorted."""
    from .rife import RIFE_DIR

    return sorted(RIFE_DIR.glob("*.pkl")) + sorted(RIFE_DIR.glob("*.pth"))


def resolve_target_fps(
    fps: float | None,
    multiplier: float | None,
    src_fps: float,
    have_weights: bool,
) -> tuple[float | None, str | None]:
    """Work out the interpolation target, or explain why it is invalid.

    Separated from `cmd_video` so bad flags are rejected before a model is
    loaded, and so this logic is testable without a GPU or weights on disk.
    Returns `(target_fps, error)`; exactly one is ever non-None.
    """
    if fps is not None and multiplier is not None:
        return None, "Use either --fps or --interpolate, not both."

    if fps is not None:
        target = float(fps)
    elif multiplier is not None:
        from .interpolate import Interpolator

        try:
            target = Interpolator.target_fps(src_fps, float(multiplier))
        except ValueError as exc:
            return None, str(exc)
    else:
        return None, None

    if target < src_fps:
        return None, (
            f"Target rate {target:g} is below the source rate {src_fps:.3f}; "
            f"this tool interpolates but does not decimate."
        )
    if not have_weights:
        from .rife import RIFE_DIR

        return None, (
            f"Frame interpolation needs RIFE weights in {RIFE_DIR}. "
            f"See the setup guide."
        )
    return target, None


def _run_image(args: argparse.Namespace) -> int:
    """Upscale a still. Images have no frame rate, segments or audio."""
    from .images import upscale_image
    from .restore import RestoreSettings, TexturePost

    device = select_device(prefer_cuda=not args.cpu)
    model = load_model(Path(args.model), device=device, half=not args.cpu)
    tile = args.tile or _auto_tile(model.scale, args.overlap)
    up = Upscaler(model, tile=tile, overlap=args.overlap, device=device, half=not args.cpu)

    settings = RestoreSettings(
        deblock=0.0, degrain=0.0,
        detail_retention=0.0 if args.no_restore else args.detail_retention,
        regrain=0.0 if args.no_restore else args.regrain,
    )
    texture = TexturePost(
        detail_retention=settings.detail_retention,
        regrain=settings.regrain,
        device=str(device),
    )

    out = upscale_image(args.input, args.output, up, texture=texture)
    print(f"{out}  ({model.scale}x, tile={tile}, CPU fallbacks: {up.cpu_fallback_count})")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    from .analyze import ScanType, classify_scan, probe_scan
    from .images import is_image
    from .restore import RestoreSettings, TexturePost, build_filter_chain

    if is_image(args.input):
        if args.fps is not None or args.interpolate is not None:
            print("Frame rate conversion does not apply to a still image.")
            return 3
        return _run_image(args)

    profile = SourceProfile.probe(args.input)

    # Validate before loading a model: a typo should not cost a 60 MB load.
    target_fps, error = resolve_target_fps(
        args.fps, args.interpolate, profile.fps, bool(rife_weights())
    )
    if error:
        print(error)
        return 3

    device = select_device(prefer_cuda=not args.cpu)
    model = load_model(Path(args.model), device=device, half=not args.cpu)
    tile = args.tile or _auto_tile(model.scale, args.overlap)
    up = Upscaler(model, tile=tile, overlap=args.overlap, device=device, half=not args.cpu)

    if args.no_restore:
        restore_settings = RestoreSettings(
            deblock=0.0, degrain=0.0, detail_retention=0.0, regrain=0.0,
        )
        scan = ScanType.PROGRESSIVE
        field_order = "tff"
        video_filter = ""
    else:
        analysis = probe_scan(args.input)
        scan = classify_scan(analysis)
        field_order = analysis.field_order
        restore_settings = RestoreSettings(
            deblock=args.deblock, degrain=args.degrain,
            detail_retention=args.detail_retention, regrain=args.regrain,
        )
        video_filter = build_filter_chain(scan, field_order, restore_settings)

    texture = TexturePost(
        detail_retention=restore_settings.detail_retention,
        regrain=restore_settings.regrain,
        device=str(device),
    )

    flow_model = None
    if target_fps is not None:
        # Loaded only when interpolation is requested, so a missing RIFE weight
        # file never blocks a plain upscale.
        from .rife import load_rife

        flow_model = load_rife(rife_weights()[0], device=device)

    # Every restoration setting that can change the pixels goes into the job
    # hash, including the derived filter string itself: changing any of these
    # between runs against the same job directory must refuse to resume
    # rather than splice together footage processed two different ways.
    settings = {
        "model": Path(args.model).name,
        "scale": model.scale,
        "tile": tile,
        "overlap": args.overlap,
        "cpu": bool(args.cpu),
        "no_restore": bool(args.no_restore),
        "deblock": restore_settings.deblock,
        "degrain": restore_settings.degrain,
        "detail_retention": restore_settings.detail_retention,
        "regrain": restore_settings.regrain,
        "scan": scan.value,
        "field_order": field_order,
        "video_filter": video_filter,
        "target_fps": target_fps,
        "scene_threshold": args.scene_threshold,
    }
    job_dir = Path(args.job_dir) if args.job_dir else Path(args.output).with_suffix(".job")

    print(f"{profile.width}x{profile.height} -> {profile.width * model.scale}x"
          f"{profile.height * model.scale}, {profile.frame_count} frames, tile={tile}")
    print(f"Job directory: {job_dir}  (interrupted renders resume from here)")
    if "decimate" in video_filter:
        print("Inverse telecine detected: this source will render as a single "
              "segment, with resume disabled for this job.")
    if target_fps is not None:
        print(f"Interpolating {profile.fps:.3f} -> {target_fps:.3f} fps")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0:
            print(f"\r{done}/{total}", end="", flush=True)

    scene_threshold = (
        DEFAULT_SCENE_THRESHOLD if args.scene_threshold is None else args.scene_threshold
    )

    try:
        render_resumable(
            profile, up, args.output, job_dir=job_dir,
            segment_frames=args.segment_frames, settings=settings,
            on_progress=progress, video_filter=video_filter, texture=texture,
            interpolate_to=target_fps, flow_model=flow_model,
            scene_threshold=scene_threshold,
        )
    except SettingsMismatch as exc:
        print(f"\nCannot resume: {exc}")
        return 2
    except ValueError as exc:
        print(f"\n{exc}")
        return 3

    print(f"\nDone. CPU fallbacks: {up.cpu_fallback_count}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import ScanType, classify_scan, estimate_blockiness, estimate_grain, probe_scan

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


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .window import launch
    except ImportError:
        print("The desktop window needs PySide6:\n"
              "  python -m pip install PySide6")
        return 5
    return launch([])


def _registry(cache_dir: Path) -> "ModelRegistry":
    from .models import ModelRegistry

    manifest = Path(__file__).resolve().parent / "manifest.json"
    return ModelRegistry(manifest_path=manifest, cache_dir=cache_dir)


def cmd_models(args: argparse.Namespace) -> int:
    """List drop-in weights and the curated, hash-verified catalogue."""
    cache = Path(args.dir)

    if args.get:
        reg = _registry(cache)
        try:
            entry = reg.get(args.get)
        except KeyError:
            print(f"Unknown model id: {args.get!r}. Run 'models' to see the catalogue.")
            return 1

        def progress(done: int, total: int) -> None:
            if total:
                print(f"\r{entry.id}: {done * 100 // total}%", end="", flush=True)

        print(f"Downloading {entry.id} ({entry.size / 1024 ** 2:.0f} MB) from {entry.url}")
        path = reg.ensure(entry, on_progress=progress)
        print(f"\nVerified and saved to {path}")
        return 0

    found = scan_custom_dir(cache)
    print(f"Installed in {cache}:")
    if found:
        for p in found:
            print(f"  {p.name}  ({p.stat().st_size / 1024 ** 2:.0f} MB)")
    else:
        print("  (none)")

    print("\nCatalogue (download with: models --get ID):")
    for entry in _registry(cache).list():
        present = "installed" if reg_installed(cache, entry) else "not installed"
        print(f"  {entry.id:34s} {entry.tier:9s} {entry.scale}x  {entry.arch:12s} {present}")
    return 0 if found else 1


def reg_installed(cache: Path, entry) -> bool:
    """True when a catalogue model is already present under `cache`."""
    return (cache / f"{entry.id}.pth").exists() or (cache / f"{entry.id}.pkl").exists()


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


def _force_utf8_stdout() -> None:
    """Make stdout UTF-8 regardless of the Windows console code page.

    Search results routinely carry Tamil, Hindi, and Telugu titles. The default
    cp1252 console raises UnicodeEncodeError on them, which for this project is
    the normal case rather than an edge case.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
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
    v.add_argument("--job-dir", default=None,
                    help="where to keep resumable state (default: <output>.job)")
    v.add_argument("--segment-frames", type=int, default=DEFAULT_SEGMENT_FRAMES,
                    help="frames per resumable segment")
    v.add_argument("--deblock", type=float, default=0.0,
                    help="deblocking strength, 0.0-1.0")
    v.add_argument("--degrain", type=float, default=0.25,
                    help="pre-upscale denoise strength, 0.0-1.0")
    v.add_argument("--detail-retention", type=float, default=0.25,
                    help="blend of real source high-frequency detail into the output, 0.0-1.0")
    v.add_argument("--regrain", type=float, default=0.6,
                    help="synthetic film grain added after upscaling, 0.0-1.0")
    v.add_argument("--fps", type=float, default=None,
                    help="target output frame rate, e.g. 60")
    v.add_argument("--interpolate", type=float, default=None,
                    help="frame rate multiplier, e.g. 2 (alternative to --fps)")
    v.add_argument("--scene-threshold", type=float, default=None,
                    help="cut sensitivity 0-1; lower detects more cuts")
    v.add_argument("--no-restore", action="store_true",
                    help="skip all restoration and texture work")
    v.set_defaults(func=cmd_video)

    a = sub.add_parser("analyze", help="inspect a source's scan type, grain, and compression")
    a.add_argument("input")
    a.set_defaults(func=cmd_analyze)

    g = sub.add_parser("gui", help="open the desktop window")
    g.set_defaults(func=cmd_gui)

    m = sub.add_parser("models", help="list installed and downloadable weights")
    m.add_argument("--dir", default="models/custom")
    m.add_argument("--get", default=None, metavar="ID",
                    help="download a catalogue model and verify its SHA-256")
    m.set_defaults(func=cmd_models)

    s = sub.add_parser("search", help="search YouTube")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_search)

    f = sub.add_parser("fetch", help="download a YouTube video")
    f.add_argument("url")
    f.add_argument("--dir", default="downloads")
    f.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
