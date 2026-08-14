"""Command-line entrypoint for the headless engine."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .bench import benchmark
from .jobs import JobState, SettingsMismatch
from .models import load_model, scan_custom_dir
from .segments import assemble, segment_path, write_segment
from .upscale import Upscaler
from .video_io import Decoder, SourceProfile
from .vram import choose_tile, free_vram_bytes, select_device

DEFAULT_BYTES_PER_OUTPUT_PIXEL = 64
DEFAULT_SEGMENT_FRAMES = 500


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
) -> Path:
    """Render `profile` to `output`, resuming any interrupted prior attempt.

    Output is written as independently complete segments so an interruption
    costs at most one segment. The final file is a stream-copy concat.
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

    if (job_dir / "job.json").exists():
        job = JobState.load(job_dir, settings=settings)
    else:
        job = JobState.create(
            job_dir, source=str(profile.path), settings=settings,
            total_frames=profile.frame_count, segment_frames=segment_frames,
        )

    out_w = profile.width * scale
    out_h = profile.height * scale

    for index in range(job.segment_count):
        if index in job.completed_segments and segment_path(job_dir, index).exists():
            continue

        start = job.start_frame_for(index)
        count = job.frames_in_segment(index)
        decoder = Decoder(profile, start_frame=start, max_frames=count)

        def processed(decoder=decoder, start=start):
            for i, frame in enumerate(decoder.frames()):
                yield upscaler.process(frame)
                if on_progress:
                    on_progress(start + i + 1, job.total_frames)

        write_segment(
            segment_path(job_dir, index), processed(),
            width=out_w, height=out_h, fps=profile.fps, source=profile,
        )
        job.mark_complete(index)

    return assemble(job_dir, job.segment_count, output, profile)


def cmd_video(args: argparse.Namespace) -> int:
    device = select_device(prefer_cuda=not args.cpu)
    model = load_model(Path(args.model), device=device, half=not args.cpu)
    tile = args.tile or _auto_tile(model.scale, args.overlap)

    profile = SourceProfile.probe(args.input)
    up = Upscaler(model, tile=tile, overlap=args.overlap, device=device, half=not args.cpu)

    settings = {
        "model": Path(args.model).name,
        "scale": model.scale,
        "tile": tile,
        "overlap": args.overlap,
        "cpu": bool(args.cpu),
    }
    job_dir = Path(args.job_dir) if args.job_dir else Path(args.output).with_suffix(".job")

    print(f"{profile.width}x{profile.height} -> {profile.width * model.scale}x"
          f"{profile.height * model.scale}, {profile.frame_count} frames, tile={tile}")
    print(f"Job directory: {job_dir}  (interrupted renders resume from here)")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0:
            print(f"\r{done}/{total}", end="", flush=True)

    try:
        render_resumable(
            profile, up, args.output, job_dir=job_dir,
            segment_frames=args.segment_frames, settings=settings,
            on_progress=progress,
        )
    except SettingsMismatch as exc:
        print(f"\nCannot resume: {exc}")
        return 2
    except ValueError as exc:
        print(f"\n{exc}")
        return 3

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
    v.set_defaults(func=cmd_video)

    m = sub.add_parser("models", help="list drop-in weights")
    m.add_argument("--dir", default="models/custom")
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
