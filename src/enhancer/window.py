"""Qt window. Contains no policy — every decision lives in RenderRequest."""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .analyze import classify_scan, estimate_blockiness, estimate_grain, probe_scan
from .gui import CancelledError, RenderJob
from .help_text import GUIDE_SECTIONS, HELP, MODEL_NOTES, RECIPES, describe_model
from .jobs import SettingsMismatch
from .models import scan_custom_dir
from .queue import RenderQueue, Task, TaskState
from .requests import RenderRequest
from .video_io import Decoder, SourceProfile

MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".png", ".jpg", ".jpeg"}
PREVIEW_SECONDS = 10

# A cap deliberately set above any consumer card, so the graphics driver spills
# into system memory instead of the tile planner shrinking to fit. That is the
# only way to run a model whose weights alone exceed the card, and it is much
# slower, so it is never automatic.
SYSTEM_MEMORY_MB = 32768

# Graphics-memory caps offered in the Output panel, in megabytes.
VRAM_CHOICES = [
    ("Automatic", None),
    ("2 GB — leave the machine free", 2048),
    ("3 GB", 3072),
    ("4 GB", 4096),
    ("5 GB — maximum speed", 5120),
    ("Use system memory — for oversized models (slow)", SYSTEM_MEMORY_MB),
]


class Worker(QObject):
    """Runs a render on a background thread."""

    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(str)
    failed = Signal(str)
    settings_changed = Signal()

    def __init__(self, request: RenderRequest) -> None:
        super().__init__()
        self.request = request
        self.job: RenderJob | None = None

    def cancel(self) -> None:
        if self.job is not None:
            self.job.cancel()

    def run(self) -> None:
        try:
            from .cli import _auto_tile, rife_weights
            from .models import load_model
            from .upscale import Upscaler
            from .vram import select_device

            req = self.request
            device = select_device(prefer_cuda=not req.cpu)
            self.log.emit(f"Loading {req.model.name} on {device}...")
            model = load_model(req.model, device=device, half=not req.cpu)
            tile = req.tile or _auto_tile(model.scale, req.overlap)
            self.log.emit(f"{model.arch}, scale {model.scale}x, tile {tile}")

            upscaler = Upscaler(
                model, tile=tile, overlap=req.overlap, device=device,
                half=not req.cpu, vram_budget=req.vram_budget,
            )

            flow = None
            if req.target_fps is not None:
                from .rife import load_rife

                weights = rife_weights()
                if not weights:
                    self.failed.emit("Frame interpolation needs RIFE weights.")
                    return
                self.log.emit(f"Loading {weights[0].name}...")
                flow = load_rife(weights[0], device=device)

            self.job = RenderJob(req, upscaler=upscaler, flow_model=flow)
            out = self.job.run(on_progress=self.progress.emit)
            self.log.emit(f"CPU fallbacks: {upscaler.cpu_fallback_count}")
            self.finished.emit(str(out))
        except SettingsMismatch:
            self.settings_changed.emit()
        except CancelledError:
            self.failed.emit("Cancelled. Re-run to resume from where it stopped.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            self.log.emit(traceback.format_exc())


def _slider(minimum: int, maximum: int, value: int) -> QSlider:
    s = QSlider(Qt.Horizontal)
    s.setRange(minimum, maximum)
    s.setValue(value)
    return s


def help_button(key: str) -> QLabel:
    """A small '?' that explains a control on hover.

    Tooltip text lives in `help_text.py` so the wording can be reviewed and
    tested without touching widget code. Qt wraps plain text at a sensible
    width once the tooltip is long enough, so the newlines in the source read
    as paragraph breaks.
    """
    label = QLabel("?")
    label.setObjectName("help")
    label.setToolTip(HELP.get(key, "No description available."))
    label.setAlignment(Qt.AlignCenter)
    label.setFixedSize(16, 16)
    label.setCursor(Qt.WhatsThisCursor)
    return label


def with_help(widget: QWidget, key: str, top: bool = False) -> QWidget:
    """Pair a control with its '?' on one row.

    Centred against single-line controls and pinned to the top against blocks
    of text, so the marker sits on the first line either way.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(theme.GAP_TIGHT)
    row.addWidget(widget, 1)
    row.addWidget(help_button(key), 0, Qt.AlignTop if top else Qt.AlignVCenter)
    return holder


def form(parent: QWidget) -> QFormLayout:
    """A form laid out the same way everywhere."""
    layout = QFormLayout(parent)
    layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setHorizontalSpacing(theme.GAP)
    layout.setVerticalSpacing(theme.GAP)
    layout.setContentsMargins(0, 0, 0, 0)
    return layout


FIELD_LABELS = (
    "Weights", "Degrain", "Detail retention", "Re-grain", "Deblock",
    "Mode", "Target", "Multiplier", "Cut sensitivity",
    "File", "Segment frames", "Graphics memory",
)


# Upper bound on the label column. Measuring alone is not enough: under a wide
# fallback font the column grew until the two settings columns together no
# longer fitted the window and the right one was clipped.
LABEL_WIDTH_MAX = 150


def label_width() -> int:
    """Width of the widest form label, measured in the font actually in use.

    Hardcoding this clipped labels under a different font or display scaling;
    measuring without a cap pushed the layout wider than the window. Both.
    """
    metrics = QFontMetrics(QApplication.font())
    widest = max(metrics.horizontalAdvance(text) for text in FIELD_LABELS)
    return min(widest + theme.GAP, LABEL_WIDTH_MAX)


def scale_from_name(name: str) -> int:
    """Guess the scale factor from a model's file name.

    Community models are named by convention: 2x..., 4x... . This is only for
    the forecast; the real value is read from the file when it loads, and the
    two are checked against each other during the render.
    """
    match = re.search(r"(?:^|[^0-9])([248])\s*[xX]", name)
    return int(match.group(1)) if match else 2


def field_label(text: str) -> QLabel:
    """A form label of uniform width, so every field column lines up."""
    label = QLabel(text)
    width = label_width()
    label.setMinimumWidth(width)
    label.setMaximumWidth(width)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    label.setToolTip(text)
    return label


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Enhancer")
        self.setMinimumSize(900, 560)
        self.resize(1280, 780)
        self.setAcceptDrops(True)

        self.source: Path | None = None
        self.profile: SourceProfile | None = None
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self._started_at = 0.0
        self.queue = RenderQueue()
        self.active_task: Task | None = None
        self._scan_type = "progressive"
        self._image_size: tuple[int, int] | None = None

        # Landscape: settings in two columns side by side, progress spanning the
        # full width beneath. The splitter lets the columns be rebalanced, and
        # every group expands with the window rather than sitting at a fixed size.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(theme.GAP_WIDE)
        left_layout.addWidget(self._source_group())
        left_layout.addWidget(self._model_group())
        left_layout.addWidget(self._output_group())
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(theme.GAP_WIDE)
        right_layout.addWidget(self._texture_group())
        right_layout.addWidget(self._fps_group())
        right_layout.addWidget(self._queue_group())
        right_layout.addStretch(1)

        self.columns = QSplitter(Qt.Horizontal)
        self.columns.addWidget(left)
        self.columns.addWidget(right)
        self.columns.setStretchFactor(0, 3)
        self.columns.setStretchFactor(1, 2)
        self.columns.setChildrenCollapsible(False)

        # Without this the settings crush into each other and overlap as soon
        # as the window is shorter than their natural height — groups lose
        # their padding, then their borders touch, then content spills across
        # them. Scrolling keeps every group at its proper size instead.
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.columns)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left.setMinimumWidth(320)
        right.setMinimumWidth(300)
        self.columns.setSizes([720, 520])

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(theme.GAP_WIDE, theme.GAP_WIDE, theme.GAP_WIDE, theme.GAP_WIDE)
        layout.setSpacing(theme.GAP_WIDE)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(self._forecast_group())
        layout.addLayout(self._actions())
        layout.addWidget(self._progress_group(), 0)
        self.setCentralWidget(root)

        self._watch_settings()
        self._update_forecast()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    # --- construction -------------------------------------------------------

    def _source_group(self) -> QGroupBox:
        box = QGroupBox("Source")
        v = QVBoxLayout(box)
        v.setSpacing(theme.GAP)

        self.drop_label = QLabel("Drop a video or image here, or use Browse")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setWordWrap(True)
        self.drop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.drop_label.setMinimumHeight(64)
        self.drop_label.setObjectName("dropzone")
        v.addWidget(with_help(self.drop_label, "source", top=True))

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_source)
        v.addWidget(browse)

        self.analysis = QLabel("No source loaded.")
        self.analysis.setWordWrap(True)
        self.analysis.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.analysis.setAlignment(Qt.AlignTop)
        self.analysis.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.analysis.setObjectName("analysis")
        v.addWidget(with_help(self.analysis, "analysis", top=True))
        return box

    def _model_group(self) -> QGroupBox:
        box = QGroupBox("Model")
        f = form(box)
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._reload_models()
        self.model_combo.currentIndexChanged.connect(self._show_model_note)
        f.addRow(field_label("Weights"), with_help(self.model_combo, "model"))

        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.model_note.setObjectName("hint")
        f.addRow("", self.model_note)
        self._show_model_note()

        refresh = QPushButton("Rescan models/custom")
        refresh.clicked.connect(self._reload_models)
        f.addRow("", with_help(refresh, "rescan"))
        return box

    def _texture_group(self) -> QGroupBox:
        box = QGroupBox("Restoration and texture")
        f = form(box)

        self.no_restore = QCheckBox("Skip all restoration (fastest, for previews)")
        f.addRow("", with_help(self.no_restore, "no_restore"))

        self.degrain = _slider(0, 100, 25)
        f.addRow(field_label("Degrain"), self._labelled(
            self.degrain, "degrain",
            "higher removes more grain — and more skin texture"))

        self.detail = _slider(0, 100, 25)
        f.addRow(field_label("Detail retention"), self._labelled(
            self.detail, "detail",
            "restores real detail from your source file"))

        self.regrain = _slider(0, 100, 60)
        f.addRow(field_label("Re-grain"), self._labelled(
            self.regrain, "regrain",
            "the strongest fix for plastic-looking skin"))

        self.deblock = _slider(0, 100, 0)
        f.addRow(field_label("Deblock"), self._labelled(
            self.deblock, "deblock",
            "raise for YouTube and low-bitrate sources"))
        return box

    def _labelled(self, slider: QSlider, key: str, hint: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        value = QLabel(f"{slider.value() / 100:.2f}")
        value.setObjectName("value")
        value.setFixedWidth(36)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider.valueChanged.connect(lambda n: value.setText(f"{n / 100:.2f}"))
        slider.setToolTip(HELP.get(key, ""))
        row.addWidget(slider, 1)
        row.addWidget(value)
        row.addWidget(help_button(key))
        v.addLayout(row)
        note = QLabel(hint)
        note.setWordWrap(True)
        note.setObjectName("hint")
        note.setMinimumWidth(1)
        note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        v.addWidget(note)
        return w

    def _fps_group(self) -> QGroupBox:
        box = QGroupBox("Frame rate (make motion smoother)")
        f = form(box)

        self.fps_mode = QComboBox()
        self.fps_mode.addItems(["Off", "Target FPS", "Multiplier"])
        for i, note in enumerate([
            "Keep the source frame rate exactly as it is.",
            "Name the frame rate you want out, such as 60.",
            "Multiply whatever the source happens to be, such as 2x.",
        ]):
            self.fps_mode.setItemData(i, note, Qt.ToolTipRole)
        f.addRow(field_label("Mode"), with_help(self.fps_mode, "fps_mode"))

        self.fps_target = QComboBox()
        self.fps_target.setEditable(True)
        self.fps_target.addItems(["48", "50", "60", "120"])
        for i, note in enumerate([
            "48 — a gentle lift from 24, keeps some of the cinema feel.",
            "50 — for PAL sources and European televisions.",
            "60 — smooth playback on a normal screen. The usual choice.",
            "120 — high refresh-rate screens, or for slowing down later.",
        ]):
            self.fps_target.setItemData(i, note, Qt.ToolTipRole)
        self.fps_target.setCurrentText("60")
        f.addRow(field_label("Target"), with_help(self.fps_target, "fps_target"))

        self.fps_multiplier = QDoubleSpinBox()
        self.fps_multiplier.setRange(1.0, 8.0)
        self.fps_multiplier.setValue(2.0)
        f.addRow(field_label("Multiplier"), with_help(self.fps_multiplier, "fps_multiplier"))

        self.scene_threshold = QDoubleSpinBox()
        self.scene_threshold.setRange(0.0, 1.0)
        self.scene_threshold.setSingleStep(0.05)
        self.scene_threshold.setValue(0.30)
        f.addRow(field_label("Cut sensitivity"), with_help(self.scene_threshold, "scene_threshold"))

        self.fps_note = QLabel("")
        self.fps_note.setWordWrap(True)
        self.fps_note.setObjectName("hint")
        f.addRow("", self.fps_note)
        self._check_rife()
        return box

    def _output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        f = form(box)

        self.output_label = QLabel("(chosen automatically beside the source)")
        self.output_label.setWordWrap(True)
        f.addRow(field_label("File"), with_help(self.output_label, "output", top=True))

        pick = QPushButton("Choose output...")
        pick.clicked.connect(self._browse_output)
        f.addRow("", pick)

        self.segment_frames = QSpinBox()
        self.segment_frames.setRange(30, 10000)
        self.segment_frames.setValue(500)
        f.addRow(field_label("Segment frames"), with_help(self.segment_frames, "segment_frames"))

        self.vram_budget = QComboBox()
        for label, mb in VRAM_CHOICES:
            self.vram_budget.addItem(label, mb)
        self.vram_budget.setItemData(
            0, "Use whatever is free, leaving room for the desktop.", Qt.ToolTipRole)
        for i in range(1, self.vram_budget.count()):
            self.vram_budget.setItemData(
                i,
                "Cap the render at this much graphics memory so the machine "
                "stays usable. Slower, but never fails.",
                Qt.ToolTipRole,
            )
        f.addRow(field_label("Graphics memory"), with_help(self.vram_budget, "vram_budget"))

        self.cpu = QCheckBox("Force CPU (very slow)")
        f.addRow("", with_help(self.cpu, "cpu"))
        return box

    def _queue_group(self) -> QGroupBox:
        box = QGroupBox("Queue")
        v = QVBoxLayout(box)
        v.setSpacing(theme.GAP)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.queue_view = QTableWidget(0, 4)
        self.queue_view.setHorizontalHeaderLabels(["File", "Status", "Progress", "Note"])
        self.queue_view.horizontalHeader().setStretchLastSection(True)
        self.queue_view.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_view.setSelectionMode(QTableWidget.SingleSelection)
        self.queue_view.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue_view.setMinimumHeight(120)
        self.queue_view.setToolTip(HELP["queue"])
        top.addWidget(self.queue_view, 1)
        top.addWidget(help_button("queue"), 0, Qt.AlignTop)
        v.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(theme.GAP_TIGHT)
        for label, key, slot in [
            ("Start", "queue_start", self._queue_start),
            ("Stop", "queue_stop", self._queue_stop),
            ("Remove", "queue_remove", self._queue_remove),
            ("Clear", "queue_clear", self._queue_clear),
        ]:
            button = QPushButton(label)
            button.setToolTip(HELP[key])
            button.clicked.connect(slot)
            setattr(self, f"btn_{key}", button)
            # Equal widths so the row reads as one control group rather than
            # four buttons of arbitrary size. Each carries its own tooltip, so
            # a separate marker per button would only add clutter.
            button.setMinimumWidth(72)
            row.addWidget(button, 1)
        v.addLayout(row)
        return box

    def _guide_html(self) -> str:
        """Every explanation in one readable page, no hovering required."""
        parts = [
            "<h2>Enhancer guide</h2>",
            "<p>Every setting explained. The same text appears when you hover "
            "a <b>?</b> in the window.</p>",
            "<h3>If something looks wrong</h3><table cellpadding='5'>",
        ]
        for problem, fix in RECIPES:
            parts.append(f"<tr><td valign='top'><b>{problem}</b></td><td>{fix}</td></tr>")
        parts.append("</table>")

        for title, entries in GUIDE_SECTIONS:
            parts.append(f"<h3>{title}</h3>")
            for label, key in entries:
                body = HELP.get(key, "").replace("\n\n", "<br><br>").replace("\n", "<br>")
                parts.append(f"<p><b>{label}</b><br>{body}</p>")

        parts.append("<h3>Which model for which footage</h3>")
        for name in sorted(MODEL_NOTES):
            body = MODEL_NOTES[name].replace("\n\n", "<br>").replace("\n", "<br>")
            parts.append(f"<p><b>{name}</b><br>{body}</p>")
        return "".join(parts)

    def _open_guide(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Enhancer guide")
        dialog.resize(760, 720)
        layout = QVBoxLayout(dialog)
        view = QTextBrowser()
        view.setHtml(self._guide_html())
        layout.addWidget(view)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def _forecast_group(self) -> QGroupBox:
        box = QGroupBox("You will get")
        v = QVBoxLayout(box)
        v.setSpacing(theme.GAP_TIGHT)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.forecast_headline = QLabel("Load a source to see what will be produced.")
        self.forecast_headline.setObjectName("headline")
        self.forecast_headline.setWordWrap(True)
        top.addWidget(self.forecast_headline, 1)
        top.addWidget(help_button("forecast"), 0, Qt.AlignTop)
        v.addLayout(top)

        self.forecast_detail = QLabel("")
        self.forecast_detail.setObjectName("analysis")
        self.forecast_detail.setWordWrap(True)
        self.forecast_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.forecast_detail)

        self.forecast_warning = QLabel("")
        self.forecast_warning.setObjectName("warning")
        self.forecast_warning.setWordWrap(True)
        self.forecast_warning.hide()
        v.addWidget(self.forecast_warning)
        return box

    def _watch_settings(self) -> None:
        """Recompute the forecast whenever any control that affects it moves."""
        for widget in (self.degrain, self.detail, self.regrain, self.deblock):
            widget.valueChanged.connect(self._update_forecast)
        for widget in (self.model_combo, self.fps_mode, self.fps_target):
            widget.currentIndexChanged.connect(self._update_forecast)
        self.fps_target.editTextChanged.connect(self._update_forecast)
        self.fps_multiplier.valueChanged.connect(self._update_forecast)
        for widget in (self.no_restore, self.cpu):
            widget.toggled.connect(self._update_forecast)

    def _update_forecast(self) -> None:
        from .forecast import forecast
        from .images import is_image

        name = self.model_combo.currentText()
        path = self.model_combo.currentData()
        if self.source is None or not path:
            self.forecast_headline.setText("Load a source and pick a model.")
            self.forecast_detail.setText("")
            self.forecast_warning.hide()
            return

        still = is_image(self.source)
        if still and self._image_size is not None:
            width, height = self._image_size
            fps, frames = 1.0, 1
        elif self.profile is not None:
            width, height = self.profile.width, self.profile.height
            fps, frames = self.profile.fps, self.profile.frame_count
        else:
            return

        target = None
        if not still:
            mode = self.fps_mode.currentText()
            if mode == "Target FPS":
                try:
                    target = float(self.fps_target.currentText())
                except ValueError:
                    target = None
            elif mode == "Multiplier":
                target = fps * self.fps_multiplier.value()

        off = self.no_restore.isChecked()
        result = forecast(
            width=width, height=height, fps=fps, frames=frames,
            scale=scale_from_name(name), model_name=name,
            scan=self._scan_type,
            deblock=0.0 if off else self.deblock.value() / 100,
            degrain=0.0 if off else self.degrain.value() / 100,
            detail_retention=0.0 if off else self.detail.value() / 100,
            regrain=0.0 if off else self.regrain.value() / 100,
            target_fps=target,
            cpu=self.cpu.isChecked(),
            is_image=still,
        )

        size_label = f" ({result.label})" if result.label else ""
        if still:
            self.forecast_headline.setText(f"{result.resolution}{size_label} image")
        else:
            self.forecast_headline.setText(
                f"{result.resolution}{size_label} at {result.fps:g} fps · "
                f"{result.frames:,} frames · about {result.time_estimate} · "
                f"around {result.size_estimate}"
            )

        self.forecast_detail.setText(
            "\n".join(f"{i}. {step}" for i, step in enumerate(result.steps, 1))
        )
        if result.warnings:
            self.forecast_warning.setText("\n".join(f"— {w}" for w in result.warnings))
            self.forecast_warning.show()
        else:
            self.forecast_warning.hide()

    def _actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.GAP_TIGHT)
        self.guide_button = QPushButton("Guide")
        self.guide_button.setToolTip(HELP["guide_button"])
        self.guide_button.clicked.connect(self._open_guide)
        self.preview_button = QPushButton(f"Preview {PREVIEW_SECONDS}s")
        self.preview_button.setToolTip(HELP["preview_button"])
        self.preview_button.clicked.connect(lambda: self._start(preview=True))
        self.render_button = QPushButton("Render")
        self.render_button.setObjectName("primary")
        self.render_button.setToolTip(HELP["render_button"])
        self.render_button.clicked.connect(lambda: self._start(preview=False))
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip(HELP["cancel_button"])
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)

        # Guide sits apart on the left; the three actions group on the right in
        # the order they are used. Every button carries its own tooltip, so no
        # separate markers here — interleaving them made the row unreadable.
        for button in (self.preview_button, self.render_button, self.cancel_button):
            button.setMinimumWidth(110)

        row.addWidget(self.guide_button)
        row.addStretch(1)
        row.addWidget(self.preview_button)
        row.addWidget(self.render_button)
        row.addWidget(self.cancel_button)
        return row

    def _progress_group(self) -> QGroupBox:
        box = QGroupBox("Progress")
        v = QVBoxLayout(box)
        v.setSpacing(theme.GAP)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        top.addWidget(self.bar, 1)
        top.addWidget(help_button("progress"))
        v.addLayout(top)

        self.status = QLabel("Idle.")
        self.status.setObjectName("status")
        v.addWidget(self.status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setMinimumHeight(70)
        # Capped, or it grows without limit and squeezes the settings out
        # of the window when the window is short.
        self.log_view.setMaximumHeight(110)
        v.addWidget(self.log_view)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        return box

    # --- queue --------------------------------------------------------------

    def _selected_task(self) -> Task | None:
        row = self.queue_view.currentRow()
        if 0 <= row < len(self.queue.tasks):
            return self.queue.tasks[row]
        return None

    def _refresh_queue(self) -> None:
        self.queue_view.setRowCount(len(self.queue.tasks))
        for row, task in enumerate(self.queue.tasks):
            cells = [task.name, task.state.value, f"{task.percent}%", task.message]
            for column, text in enumerate(cells):
                self.queue_view.setItem(row, column, QTableWidgetItem(text))

        running = self.queue.running is not None
        self.btn_queue_start.setEnabled(not running and bool(self.queue.next_waiting()
                                                             or self._selected_task()))
        self.btn_queue_stop.setEnabled(running)
        selected = self._selected_task()
        self.btn_queue_remove.setEnabled(bool(selected and selected.can_remove))

    def _queue_start(self) -> None:
        if self.queue.running is not None:
            return
        task = self._selected_task()
        if task is None or task.state is TaskState.RUNNING:
            task = self.queue.next_waiting()
        if task is None:
            self._append("Nothing waiting in the queue.")
            return
        if task.state is not TaskState.WAITING:
            self.queue.requeue(task)
        self._run_task(task)

    def _queue_stop(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self._append("Stopping after the current frame...")

    def _queue_remove(self) -> None:
        task = self._selected_task()
        if task is None:
            return
        if not self.queue.remove(task):
            QMessageBox.information(
                self, "Still running",
                "Stop this job before removing it.")
        self._refresh_queue()

    def _queue_clear(self) -> None:
        self._append(f"Cleared {self.queue.clear_finished()} finished job(s).")
        self._refresh_queue()

    def _show_model_note(self) -> None:
        name = self.model_combo.currentText()
        self.model_note.setText(describe_model(name).split("\n\n")[0] if name else "")

    # --- behaviour ----------------------------------------------------------

    def _reload_models(self) -> None:
        self.model_combo.clear()
        found = scan_custom_dir(Path("models/custom"))
        if not found:
            self.model_combo.addItem("No models in models/custom", None)
            self.model_combo.setItemData(
                0,
                "Download one with:  enhancer models --get 2xParimgCompact\n"
                "Or drop any .pth into the models\\custom folder.",
                Qt.ToolTipRole,
            )
        for p in found:
            self.model_combo.addItem(p.name, str(p))
            # Per-entry tooltip: which kind of footage this one suits.
            self.model_combo.setItemData(
                self.model_combo.count() - 1, describe_model(p.name), Qt.ToolTipRole
            )
        if hasattr(self, "model_note"):
            self._show_model_note()

    def _check_rife(self) -> None:
        from .cli import rife_weights

        if rife_weights():
            self.fps_note.setText("RIFE weights found.")
        else:
            self.fps_note.setText(
                "No RIFE weights in models/rife — frame rate conversion unavailable."
            )

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt naming
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in MEDIA_SUFFIXES:
                self._load_source(path)
                break

    def _browse_source(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Choose a source", "", "Video (*.mp4 *.mkv *.mov *.avi *.webm)"
        )
        if name:
            self._load_source(Path(name))

    def _browse_output(self) -> None:
        name, _ = QFileDialog.getSaveFileName(self, "Choose output", "", "Matroska (*.mkv)")
        if name:
            self.output_label.setText(name)

    def _load_source(self, path: Path) -> None:
        from .images import is_image

        self.source = path
        self.drop_label.setText(path.name)
        self.analysis.setText("Analysing...")
        QApplication.processEvents()

        if is_image(path):
            self._load_image_source(path)
            return

        # Re-enable controls a previously loaded still may have switched off.
        self.fps_mode.setEnabled(True)
        self.preview_button.setEnabled(True)

        try:
            self.profile = SourceProfile.probe(path)
            analysis = probe_scan(path)
            scan = classify_scan(analysis)
            frame = next(iter(Decoder(self.profile, max_frames=1).frames()))
            grain = estimate_grain(frame)
            blockiness = estimate_blockiness(frame)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.analysis.setText(f"Could not analyse this file: {exc}")
            return

        p = self.profile
        lines = [
            f"{p.width}x{p.height} at {p.fps:.3f} fps, {p.frame_count} frames",
            f"Colour {p.color_space or 'unspecified'}, SAR {p.sar}, {p.pix_fmt}",
            f"Scan {scan.value} ({analysis.field_order}), "
            f"grain {grain:.2f}, blockiness {blockiness:.2f}",
        ]
        self._scan_type = scan.value
        if scan.value == "telecined":
            lines.append(
                "Film carried as interlaced. Inverse telecine will be applied; "
                "this disables resume and cannot be combined with frame rate conversion."
            )
        if blockiness > 2.0:
            lines.append("Compression artifacts present — consider raising Deblock.")
        self.analysis.setText("\n".join(lines))

        if self.output_label.text().startswith("("):
            self.output_label.setText(str(path.with_name(path.stem + "_enhanced.mkv")))
        self._update_forecast()

    def _load_image_source(self, path: Path) -> None:
        """Stills take a different path: no frame rate, no scan type, no resume."""
        from .images import estimate_still_grain, load_image

        self.profile = None
        try:
            rgb, alpha = load_image(path)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.analysis.setText(f"Could not read this image: {exc}")
            return

        grain, blockiness = estimate_still_grain(rgb)
        h, w = rgb.shape[:2]
        self._image_size = (w, h)
        self._scan_type = "progressive"
        lines = [
            f"Still image, {w}x{h}{', with alpha' if alpha is not None else ''}",
            f"Grain {grain:.2f}, blockiness {blockiness:.2f}",
        ]
        if blockiness > 2.0:
            lines.append("Compression artifacts present.")
        lines.append("Frame rate conversion and preview do not apply to a still.")
        self.analysis.setText("\n".join(lines))

        self.fps_mode.setCurrentText("Off")
        self.fps_mode.setEnabled(False)
        self.preview_button.setEnabled(False)

        if self.output_label.text().startswith("("):
            self.output_label.setText(str(path.with_name(path.stem + "_enhanced" + path.suffix)))
        self._update_forecast()

    def _build_request(self, preview: bool) -> RenderRequest | None:
        model = self.model_combo.currentData()
        if not model:
            QMessageBox.warning(self, "No model", "Put a .pth into models/custom first.")
            return None
        if self.source is None:
            QMessageBox.warning(self, "No source", "Load a video or image first.")
            return None

        # A still has no frame rate, so neither interpolation nor preview apply.
        target = None
        preview_frames = None
        if self.profile is not None:
            mode = self.fps_mode.currentText()
            if mode == "Target FPS":
                target = float(self.fps_target.currentText())
            elif mode == "Multiplier":
                target = self.profile.fps * self.fps_multiplier.value()
            if preview:
                preview_frames = int(self.profile.fps * PREVIEW_SECONDS)

        try:
            return RenderRequest(
                model=Path(model),
                source=self.source,
                output=Path(self.output_label.text()),
                deblock=self.deblock.value() / 100,
                degrain=self.degrain.value() / 100,
                detail_retention=self.detail.value() / 100,
                regrain=self.regrain.value() / 100,
                no_restore=self.no_restore.isChecked(),
                target_fps=target,
                scene_threshold=self.scene_threshold.value(),
                cpu=self.cpu.isChecked(),
                vram_budget=self._selected_vram_budget(),
                segment_frames=self.segment_frames.value(),
                preview_frames=preview_frames,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return None

    def _start(self, preview: bool) -> None:
        request = self._build_request(preview)
        if request is None:
            return
        if self.profile is not None:
            try:
                request.validate_against(self.profile.fps)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid settings", str(exc))
                return

        if not preview and request.job_dir.exists():
            self._append(f"Resuming existing job in {request.job_dir}")

        # Previews jump the queue: they are short and exist to answer a
        # question now. Full renders join the queue and run in turn.
        if preview:
            self._run_task(Task(request=request))
            return

        task = self.queue.add(request)
        self._refresh_queue()
        if self.queue.running is None:
            self._run_task(task)
        else:
            self._append(f"Queued {task.name} — will start when the current job ends.")

    def _run_task(self, task: Task) -> None:
        request = task.request
        if task in self.queue.tasks and not self.queue.start(task):
            return
        self.active_task = task
        self._refresh_queue()

        self._set_running(True)
        self._started_at = time.perf_counter()
        self.bar.setValue(0)
        self._append(f"{'Preview' if request.is_preview else 'Render'} -> {request.output}")

        self.thread = QThread()
        self.worker = Worker(request)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.settings_changed.connect(self._on_settings_changed)
        self.thread.start()

    def _on_settings_changed(self) -> None:
        """Offer a way forward instead of a dead end.

        Refusing to resume is right — splicing footage processed two different
        ways leaves a visible seam mid-render — but the user still has to be
        able to act on it without hunting for a folder to delete.
        """
        request = self.worker.request if self.worker else None
        self._teardown()
        if request is None:
            return

        choice = QMessageBox.question(
            self,
            "Settings changed",
            "This output already has a part-finished render made with "
            "different settings.\n\n"
            "Continuing it would leave a visible seam partway through the "
            "video, so it cannot simply carry on.\n\n"
            "Start again from the beginning with your current settings?\n"
            "(The part-finished render is discarded.)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if choice != QMessageBox.Yes:
            self.status.setText("Stopped — settings differ from the unfinished render.")
            self._append("Restore the previous settings to continue it instead.")
            return

        import shutil

        shutil.rmtree(request.job_dir, ignore_errors=True)
        self._append(f"Discarded {request.job_dir}. Starting fresh.")
        self._start(preview=request.is_preview)

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self._append("Cancelling after the current frame...")

    def _on_progress(self, done: int, total: int) -> None:
        if self.active_task is not None:
            self.active_task.done_frames, self.active_task.total_frames = done, total
            if done % 50 == 0:
                self._refresh_queue()
        self.bar.setMaximum(total)
        self.bar.setValue(done)
        elapsed = time.perf_counter() - self._started_at
        fps = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / fps if fps > 0 else 0.0
        self.status.setText(
            f"{done}/{total} frames | {fps:.1f} fps | "
            f"{remaining / 60:.1f} min remaining"
        )

    def _on_finished(self, path: str) -> None:
        self._append(f"Done: {path}")
        self.status.setText("Finished.")
        if self.active_task is not None:
            self.queue.finish(self.active_task)
        self._teardown()
        self._start_next()

    def _on_failed(self, message: str) -> None:
        self._append(message)
        self.status.setText("Stopped.")
        if self.active_task is not None:
            if "Cancelled" in message or "Stopping" in message:
                self.queue.stop(self.active_task)
            else:
                self.queue.fail(self.active_task, message[:80])
        self._teardown()
        self._start_next()

    def _start_next(self) -> None:
        """Move on to the next waiting job, if any."""
        self.active_task = None
        self._refresh_queue()
        following = self.queue.next_waiting()
        if following is not None:
            self._run_task(following)

    def _selected_vram_budget(self) -> int | None:
        megabytes = self.vram_budget.currentData()
        return None if megabytes is None else int(megabytes) * 1024 ** 2

    def _teardown(self) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.worker = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.render_button.setEnabled(not running)
        self.preview_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _append(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def bring_to_front(self) -> None:
        """Raise and focus, for when the shortcut is used a second time."""
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()


def launch(argv: list[str] | None = None) -> int:
    from .single import InstanceServer, another_instance_running

    app = QApplication(argv or [])

    if another_instance_running():
        # The running copy has been asked to come forward; nothing more to do.
        return 0

    theme.apply(app)
    guard = InstanceServer()
    window = MainWindow()
    guard.raise_requested.connect(window.bring_to_front)
    window.show()
    try:
        return app.exec()
    finally:
        guard.close()
