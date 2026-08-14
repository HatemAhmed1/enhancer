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
