"""User render choices, independent of any interface."""

from __future__ import annotations

from dataclasses import dataclass
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
