"""User render choices, independent of any interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PREVIEW_SUFFIX = ".preview"

# Nothing plays above this, and the cost is linear in it: a two-second clip at
# 100000 fps is two hundred thousand synthesized frames, which is a longer
# render than the feature film this tool exists for. Well above every real
# target (240 covers high-frame-rate displays and slow-motion masters).
MAX_OUTPUT_FPS = 240.0


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

    # Cap on graphics memory, in bytes. None means use whatever is free,
    # leaving the standard headroom for the desktop.
    vram_budget: int | None = None

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
            # never leave journal state that the real render would try to resume.
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
        """Check the choices that need to know something about the source."""
        if self.target_fps is not None and self.target_fps < src_fps:
            raise ValueError(
                f"target rate {self.target_fps:g} is below the source rate "
                f"{src_fps:.3f}; this tool interpolates but does not decimate"
            )
        if self.target_fps is not None and self.target_fps > MAX_OUTPUT_FPS:
            raise ValueError(
                f"target rate {self.target_fps:g} is above the {MAX_OUTPUT_FPS:g} "
                f"fps limit; every frame above the source rate has to be "
                f"synthesized, so the render time rises with it"
            )

    def settings_dict(self, scale: int, tile: int, video_filter: str) -> dict:
        """The job hash input: everything that can change output pixels.

        `tile` is accepted for call compatibility and deliberately left out.
        It is chosen from whatever graphics memory happens to be free, so the
        same command picks a different tile with a browser open — and refusing
        to resume for that reason costs hours. It is also not a stable input to
        begin with: the out-of-memory runner already steps the tile up and down
        *within* a single render, so pinning it in the hash would guarantee
        nothing anyway. `overlap`, which is what actually governs how much
        context each tile sees, stays in.

        The source is not in here either. It is checked separately, by the
        journal itself, so that every caller gets the check and the refusal can
        say which film the job directory belongs to.
        """
        return {
            "model": self.model.name,
            "scale": scale,
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
