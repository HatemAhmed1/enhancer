"""Tiling, tiled execution, out-of-memory recovery, and device selection."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)

InferFn = Callable[[torch.Tensor], torch.Tensor]

# Descending ladder of candidate tile sizes. Used for stepping in BOTH
# directions so that shrink and recover are symmetric.
TILE_LADDER: tuple[int, ...] = (1024, 768, 512, 384, 256, 192, 128)

# Reserved VRAM. A laptop GPU also drives the desktop compositor under WDDM,
# so this is deliberately generous.
HEADROOM_BYTES: int = 1024 ** 3

# Upper bound on the success-probe backoff, so probes stay rare but never stop.
MAX_PROBE_INTERVAL: int = 8192

# Substrings identifying a RECOVERABLE allocation failure. Deliberately narrow:
# "CUDA error: an illegal memory access" is unrecoverable and must NOT match,
# because retrying it forever would hide a real bug.
#
# Every backend's own allocator says "out of memory" somewhere in its message
# ("CUDA out of memory", "MPS backend out of memory (MPS allocated: ...)",
# "XPU out of memory. Tried to allocate ..."), so that one marker carries most
# of the weight. The rest exist because the layer UNDER the allocator - Metal
# on macOS, Level Zero/UR on Intel - reports the same condition in its own
# words, and those never contain the phrase.
_OOM_MARKERS: tuple[str, ...] = (
    # CUDA, and the common wording of torch's allocator on every backend.
    "out of memory",
    "alloc_failed",
    "cuda_error_out_of_memory",
    # MPS: Metal rejects an oversized single buffer with its own wording,
    # e.g. "Invalid buffer size: 12.00 GB" or an MPSNDArray complaint that the
    # array exceeds the 2**32-byte limit. Both are fixed by a smaller tile.
    "invalid buffer size",
    "total bytes of ndarray",
    # XPU: Level Zero / Unified Runtime failures reach Python as a plain
    # RuntimeError carrying the runtime's enum name rather than prose, e.g.
    # "Native API failed. ... UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY".
    # Underscored, so "out of memory" above does not match them.
    "out_of_device_memory",
    "out_of_host_memory",
    "out_of_resources",
)

# torch>=2.4 exposes a single torch.OutOfMemoryError shared by every backend,
# and torch.cuda.OutOfMemoryError is an alias for it. Older builds have only
# the cuda one. Resolve whatever exists, de-duplicated, so isinstance() works
# on any version without a version check.
_OOM_TYPES: tuple[type[BaseException], ...] = tuple(
    dict.fromkeys(
        candidate
        for candidate in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
            MemoryError,
        )
        if isinstance(candidate, type) and issubclass(candidate, BaseException)
    )
)


def is_oom_error(exc: BaseException) -> bool:
    """True if `exc` is a recoverable allocation failure.

    Accelerator exhaustion does not always arrive as a typed OutOfMemoryError.
    cuDNN and cuBLAS workspace allocation and driver-level failures surface as
    a plain RuntimeError, MPS reports Metal's own buffer complaints, XPU can
    surface a Level Zero error instead of the typed one, and the CPU fallback
    path raises MemoryError. Catching only the typed error lets a real OOM kill
    a multi-hour render.
    """
    if isinstance(exc, _OOM_TYPES):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(marker in message for marker in _OOM_MARKERS)
    return False


class TileFloorReached(Exception):
    """Tiling cannot make this frame fit. The caller should fall back to CPU.

    Deliberately NOT a RuntimeError subclass: TileRunner widens its except
    clause to RuntimeError, and this signal must pass through it uncaught.
    """


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
        _check_scale(up, patch, scale)

        if out is None:
            out = _alloc_output(up, h, w, scale)

        # Offset of the core within the padded patch, in output pixels.
        oy = (t.y0 - t.py0) * scale
        ox = (t.x0 - t.px0) * scale
        ch = (t.y1 - t.y0) * scale
        cw = (t.x1 - t.x0) * scale

        out[:, :, t.y0 * scale:t.y1 * scale, t.x0 * scale:t.x1 * scale] = (
            up[:, :, oy:oy + ch, ox:ox + cw]
        )

    assert out is not None
    return out


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
        vram_ceiling: int | None = None,
    ) -> None:
        self.tile = tile
        self.max_tile = tile
        self.overlap = overlap
        self.scale = scale
        self.min_tile = min_tile
        self.recover_after = recover_after
        self._probe_interval = recover_after
        self._successes = 0
        self.vram_ceiling = vram_ceiling

    def _step_down(self, tile: int) -> int:
        for rung in TILE_LADDER:
            if rung < tile:
                return max(self.min_tile, rung)
        return self.min_tile

    def _step_up(self, tile: int) -> int:
        for rung in reversed(TILE_LADDER):
            if rung > tile:
                return rung
        return self.max_tile

    def run(
        self, fn: InferFn, img: torch.Tensor, peak_bytes: int | None = None
    ) -> torch.Tensor:
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
                # Whatever device the frame is on: releasing the caching
                # allocator's pools is what makes the retry at a smaller tile
                # more likely to succeed, and every backend spells it the same.
                empty_device_cache(img.device)
                if self.tile <= self.min_tile:
                    raise TileFloorReached(
                        f"allocation failed at minimum tile size {self.min_tile}"
                    ) from exc
                self.tile = self._step_down(self.tile)
                log.warning("Allocation failure; retrying frame at tile=%d", self.tile)
                continue

            self._note_success()
            self._check_pressure(peak_bytes)
            return out

    def _check_pressure(self, peak_bytes: int | None) -> None:
        """Shrink on silent oversubscription (spec §1.1.1).

        On Windows WDDM an over-budget allocation does not raise; it spills into
        shared system memory and merely gets slow. Polling the allocator peak is
        the only reliable signal there.
        """
        if self.vram_ceiling is None or peak_bytes is None:
            return
        if peak_bytes <= self.vram_ceiling:
            return
        if self.tile <= self.min_tile:
            return
        self._successes = 0
        self._probe_interval = min(self._probe_interval * 2, MAX_PROBE_INTERVAL)
        self.tile = self._step_down(self.tile)
        log.warning(
            "Peak allocation %.0f MB exceeded the %.0f MB physical ceiling "
            "without raising; shrinking tile to %d",
            peak_bytes / 1024 ** 2, self.vram_ceiling / 1024 ** 2, self.tile,
        )

    def _note_success(self) -> None:
        if self.tile >= self.max_tile:
            return
        self._successes += 1
        if self._successes >= self._probe_interval:
            self.tile = min(self.max_tile, self._step_up(self.tile))
            self._successes = 0
            log.info("VRAM pressure eased; probing tile size %d", self.tile)


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


# Calibrated per-output-pixel activation cost at fp16, from Plan 1 measurements.
BYTES_PER_OUTPUT_PIXEL_FP16: int = 32


def bytes_per_output_pixel_for_dtype(dtype: torch.dtype) -> int:
    """Per-output-pixel activation budget for a model running in `dtype`.

    An architecture spandrel reports as supports_half=False runs fp32 and needs
    roughly double the activation memory of the fp16 case the constant is
    calibrated against.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return BYTES_PER_OUTPUT_PIXEL_FP16
    return BYTES_PER_OUTPUT_PIXEL_FP16 * 2


def physical_vram_ceiling(total_bytes: int) -> int:
    """Hard allocation ceiling derived from PHYSICAL VRAM.

    Windows WDDM will silently oversubscribe into system-RAM-backed shared GPU
    memory rather than raising a catchable OOM, so an exception-driven recovery
    policy alone is not sufficient. This ceiling gives the runner something to
    compare against.
    """
    return max(0, total_bytes - HEADROOM_BYTES)


# ---------------------------------------------------------------------------
# Backends
#
# Every accelerator is optional AT RUNTIME, not just at import time: a wheel
# built without XPU support has no `torch.xpu` attribute at all, `torch.backends
# .mps` is absent on non-Apple builds, and any of these probes can still raise
# on a machine with a half-installed driver. So each one is guarded twice, by
# hasattr and by try. A missing backend is a normal condition, never an error.
# ---------------------------------------------------------------------------

# Preference order. CUDA first because it is the fastest and best supported
# path; MPS before XPU only because Apple's backend is the more mature of the
# two. CPU is not listed: it is the fallback when none of these answer.
ACCELERATOR_PRIORITY: tuple[str, ...] = ("cuda", "mps", "xpu")


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not hasattr(backend, "is_available"):
        return False
    try:
        # is_built() is False on a non-Apple wheel that still ships the module.
        if hasattr(backend, "is_built") and not backend.is_built():
            return False
        return bool(backend.is_available())
    except Exception:
        return False


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not hasattr(xpu, "is_available"):
        return False
    try:
        return bool(xpu.is_available())
    except Exception:
        return False


_AVAILABILITY: dict[str, Callable[[], bool]] = {
    "cuda": _cuda_available,
    "mps": _mps_available,
    "xpu": _xpu_available,
}


def is_accelerator_available(kind: str) -> bool:
    """True if torch can actually use the named accelerator right now."""
    probe = _AVAILABILITY.get(kind)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        # A half-installed driver can make is_available() itself throw. Not
        # having a backend is never a reason to fail.
        log.debug("availability probe for %s failed", kind, exc_info=True)
        return False


def available_accelerators() -> tuple[str, ...]:
    """Every usable accelerator on this machine, best first."""
    return tuple(k for k in ACCELERATOR_PRIORITY if is_accelerator_available(k))


def select_accelerator(prefer_accelerator: bool = True) -> torch.device:
    """Return the best available torch device: CUDA, MPS, XPU, then CPU.

    With `prefer_accelerator=False` the answer is always CPU, which is what the
    user asked for with --cpu.
    """
    if prefer_accelerator:
        for kind in ACCELERATOR_PRIORITY:
            if is_accelerator_available(kind):
                return torch.device(kind)
    return torch.device("cpu")


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device.

    The parameter name predates MPS and XPU support and now reads as a lie: it
    has always meant "use an accelerator if there is one", so it still does.
    Callers pass it by keyword, so it cannot simply be renamed. Prefer
    `select_accelerator` in new code.
    """
    return select_accelerator(prefer_accelerator=prefer_cuda)


# ---------------------------------------------------------------------------
# Memory reporting
#
# These numbers size real work: choose_tile() turns `free` into a tile size and
# Upscaler turns `total` into the peak-allocation ceiling. Before this, both
# returned 0 on anything but CUDA, which meant an Apple or Intel machine got
# cli._auto_tile's blind 256 fallback and NO pressure ceiling at all.
# ---------------------------------------------------------------------------


def _system_memory_bytes() -> tuple[int, int]:
    """(free, total) system RAM in bytes, or (0, 0) if it cannot be found.

    psutil is deliberately not a dependency here, so this uses only what the
    standard library exposes on each platform.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys), int(status.ullTotalPhys)
        except Exception:
            pass

    try:
        # Linux. MemAvailable is the honest "could be handed to a process
        # without swapping" figure; MemFree ignores reclaimable page cache.
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    values[key] = int(parts[0]) * 1024
        total = values.get("MemTotal", 0)
        free = values.get("MemAvailable", values.get("MemFree", 0))
        if total:
            return free, total
    except Exception:
        pass

    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
        try:
            free = page * os.sysconf("SC_AVPHYS_PAGES")
        except (OSError, ValueError):
            # macOS has no SC_AVPHYS_PAGES. Reporting half of physical RAM as
            # free is a deliberate under-estimate: too small a tile is merely
            # slower, too large a one is an OOM.
            free = total // 2
        if total > 0:
            return int(free), int(total)
    except (OSError, ValueError, AttributeError):
        pass

    return 0, 0


def _cuda_memory_bytes() -> tuple[int, int]:
    if not _cuda_available():
        return 0, 0
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return 0, 0
    return int(free), int(total)


def _mps_memory_bytes() -> tuple[int, int]:
    """(free, total) for Apple unified memory.

    There is no dedicated VRAM to report, so "total" has to be defined rather
    than measured. Metal's recommendedMaxWorkingSetSize - what
    torch.mps.recommended_max_memory() returns - is the honest answer to "how
    much of this machine's RAM may the GPU use", so that is `total`, and `free`
    is what remains of it after the Metal driver's own allocations for this
    process (which include the caching allocator's pools).

    The memory is shared with the OS and every other application, so
    HEADROOM_BYTES matters MORE here than on a discrete card, not less.
    """
    mps = getattr(torch, "mps", None)
    total = 0
    if mps is not None and hasattr(mps, "recommended_max_memory"):
        try:
            total = int(mps.recommended_max_memory())
        except Exception:
            total = 0
    if total <= 0:
        # Old torch without the query: fall back to physical RAM.
        _free, total = _system_memory_bytes()
    if total <= 0:
        return 0, 0
    used = 0
    if mps is not None and hasattr(mps, "driver_allocated_memory"):
        try:
            used = int(mps.driver_allocated_memory())
        except Exception:
            used = 0
    return max(0, total - used), total


def _xpu_memory_bytes() -> tuple[int, int]:
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not _xpu_available():
        return 0, 0
    if hasattr(xpu, "mem_get_info"):
        try:
            free, total = xpu.mem_get_info()
            return int(free), int(total)
        except Exception:
            # Some Arc parts cannot report free memory at all; fall through to
            # the device properties rather than giving up and returning 0.
            pass
    try:
        total = int(xpu.get_device_properties().total_memory)
    except Exception:
        return 0, 0
    try:
        used = int(xpu.memory_reserved())
    except Exception:
        used = 0
    return max(0, total - used), total


_MEMORY_PROBES: dict[str, Callable[[], tuple[int, int]]] = {
    "cuda": _cuda_memory_bytes,
    "mps": _mps_memory_bytes,
    "xpu": _xpu_memory_bytes,
    "cpu": _system_memory_bytes,
}


def _device_kind(device: str | torch.device | None) -> str:
    """The backend name for `device`, defaulting to the one we would pick."""
    if device is None:
        return select_accelerator().type
    return torch.device(device).type


def device_memory_bytes(device: str | torch.device | None = None) -> tuple[int, int]:
    """(free, total) bytes of the memory `device` draws on. (0, 0) if unknown.

    On CPU that is system RAM, not zero: the tile planner sizes work from these
    numbers, so answering 0 there would silently pin every non-CUDA machine to
    the smallest tile on the ladder.
    """
    probe = _MEMORY_PROBES.get(_device_kind(device))
    if probe is None:
        return 0, 0
    try:
        free, total = probe()
    except Exception:
        return 0, 0
    return max(0, int(free)), max(0, int(total))


def total_vram_bytes(device: str | torch.device | None = None) -> int:
    """Total memory available to `device`, or 0 when it cannot be determined."""
    return device_memory_bytes(device)[1]


def free_vram_bytes(device: str | torch.device | None = None) -> int:
    """Free memory available to `device`, or 0 when it cannot be determined."""
    return device_memory_bytes(device)[0]


def device_memory_ceiling(device: str | torch.device | None = None) -> int | None:
    """Peak-allocation ceiling for `device`, or None when none applies.

    Both hazards this guards against are the same shape. On Windows WDDM an
    over-budget allocation spills into system-RAM-backed shared GPU memory
    instead of raising. On MPS torch's high-watermark ratio defaults to 1.7, so
    allocations between 1.0x and 1.7x of the recommended working set are also
    allowed through silently, and the machine simply starts swapping. In both
    cases an exception-driven policy alone never fires; the ceiling is what
    gives the runner something to compare a measured peak against.

    CPU returns None: its allocations are the system allocator's problem, and a
    genuine exhaustion there raises MemoryError, which is_oom_error catches.
    """
    kind = _device_kind(device)
    if kind == "cpu":
        return None
    total = total_vram_bytes(kind)
    if total <= 0:
        return None
    return physical_vram_ceiling(total)


def supports_half(device: str | torch.device | None = None) -> bool:
    """Whether fp16 inference is worth doing on `device`.

    cuda: yes, every card this targets runs fp16 at least at fp32 rate.
    mps:  yes, Metal is fp16-native and Apple GPUs prefer it.
    xpu:  yes, Arc/Xe fp16 throughput is at least fp32's.
    cpu:  NO. torch's fp16 CPU kernels are a compatibility shim rather than an
          optimisation - several ops have no fp16 CPU path at all and the ones
          that do are slower than fp32. The CPU fallback is already the slow
          path; making it slower and less compatible helps nobody.
    """
    return _device_kind(device) in ACCELERATOR_PRIORITY


def empty_device_cache(device: str | torch.device | None = None) -> None:
    """Release cached allocator blocks on `device`. A no-op where unsupported."""
    module = getattr(torch, _device_kind(device), None)
    release = getattr(module, "empty_cache", None)
    if release is None:
        return
    try:
        release()
    except Exception:  # pragma: no cover - a failed cache flush is not fatal
        log.debug("empty_cache failed on %s", device, exc_info=True)


def reset_peak_memory(device: str | torch.device | None = None) -> None:
    """Start a fresh peak-allocation measurement. A no-op where unsupported."""
    module = getattr(torch, _device_kind(device), None)
    reset = getattr(module, "reset_peak_memory_stats", None)
    if reset is None:
        return
    try:
        reset()
    except Exception:  # pragma: no cover
        log.debug("reset_peak_memory_stats failed on %s", device, exc_info=True)


def peak_memory_bytes(device: str | torch.device | None = None) -> int | None:
    """Peak bytes allocated since `reset_peak_memory`, or None if unknowable.

    None means "no pressure signal", which is NOT the same as zero: zero would
    read as "comfortably under the ceiling" and suppress the check entirely.
    """
    kind = _device_kind(device)
    if kind == "mps":
        # torch.mps exposes no peak counter. current_allocated_memory() is a
        # deliberate under-estimate of the peak - it excludes the allocator's
        # cached pools and is read after the run rather than at the high-water
        # mark - so it can only ever MISS pressure, never invent it. That is
        # the safe direction for a signal whose only effect is to shrink tiles.
        module = getattr(torch, "mps", None)
        probe = getattr(module, "current_allocated_memory", None)
    else:
        module = getattr(torch, kind, None)
        probe = getattr(module, "max_memory_allocated", None)
    if probe is None:
        return None
    try:
        return int(probe())
    except Exception:
        return None
