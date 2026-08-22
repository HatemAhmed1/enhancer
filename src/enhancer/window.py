"""Qt window. Contains no policy — every decision lives in RenderRequest."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .analyze import classify_scan, estimate_blockiness, estimate_grain, probe_scan
from .gui import CancelledError, RenderJob
from .images import IMAGE_SUFFIXES
from .help_text import (
    GUIDE_SECTIONS,
    HELP,
    MODEL_NOTES,
    RECIPES,
    describe_model,
    display_name,
    model_rank,
    model_scale as scale_from_name,
)
from .jobs import SettingsMismatch
from .models import scan_custom_dir
from .queue import RenderQueue, Task, TaskState
from .requests import RenderRequest
from .video_io import Decoder, SourceProfile

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".wmv"}

# One list, derived from the engine's own. Three had drifted apart: the Browse
# dialog offered video only, so a still could not be opened through it at all
# even though the whole image path existed; drag-and-drop took three image
# types; and images.py handled seven.
MEDIA_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES

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


class CompareWorker(QObject):
    """Renders one frame with the current settings, on a background thread.

    Separate from `Worker` because it must never touch the job journal, write
    an output file, or disturb a render that is already running. Its whole
    purpose is to answer "will faces look waxy?" in about a second rather than
    after seventeen hours.
    """

    ready = Signal(object)
    log = Signal(str)
    failed = Signal(str)

    def __init__(self, request: RenderRequest, seconds: float) -> None:
        super().__init__()
        self.request = request
        self.seconds = seconds

    def run(self) -> None:
        try:
            from .cli import _auto_tile
            from .compare import compare_frame
            from .models import load_model
            from .upscale import Upscaler
            from .vram import select_device

            req = self.request
            device = select_device(prefer_cuda=not req.cpu)
            self.log.emit(f"Comparing one frame with {req.model.name} on {device}...")
            model = load_model(req.model, device=device, half=not req.cpu)
            tile = req.tile or _auto_tile(model.scale, req.overlap)
            upscaler = Upscaler(
                model, tile=tile, overlap=req.overlap, device=device,
                half=not req.cpu, vram_budget=req.vram_budget,
            )
            pair = compare_frame(req, upscaler, seconds=self.seconds)
            self.ready.emit(pair)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            self.log.emit(traceback.format_exc())


class PlaybackWorker(QObject):
    """Owns the paired decoders on a background thread.

    Decoding two frames and rescaling one of them costs tens of milliseconds
    at 4K, which is most of a frame interval — done on the GUI thread the
    window would stop responding for the length of the clip.
    """

    opened = Signal(float, int, float, float)  # fps, frames, start, end
    frame = Signal(object)
    ended = Signal()
    failed = Signal(str)

    def __init__(self, source: Path, output: Path) -> None:
        super().__init__()
        self._source = source
        self._output = output
        self.player = None

    def open(self) -> None:
        try:
            from .playback import ComparePlayer

            self.player = ComparePlayer(self._source, self._output)
            self.player.open()
            start, end = self.player.covers
            self.opened.emit(
                self.player.fps, self.player.frame_count, start, end
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def deliver_next(self) -> None:
        if self.player is None:
            return
        try:
            pair = self.player.next_pair()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if pair is None:
            self.ended.emit()
        else:
            self.frame.emit(pair)

    def seek(self, seconds: float) -> None:
        if self.player is None:
            return
        try:
            self.player.seek(seconds)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.deliver_next()

    def shutdown(self) -> None:
        if self.player is not None:
            self.player.close()
        self.player = None


def _clock(seconds: float) -> str:
    """m:ss, or h:mm:ss once a clip runs past an hour."""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def source_filter() -> str:
    """File-dialog filter covering everything the engine can actually open."""
    def patterns(suffixes: set[str]) -> str:
        return " ".join(f"*{s}" for s in sorted(suffixes))

    return ";;".join([
        f"All supported ({patterns(MEDIA_SUFFIXES)})",
        f"Video ({patterns(VIDEO_SUFFIXES)})",
        f"Images ({patterns(IMAGE_SUFFIXES)})",
        "All files (*)",
    ])


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
    # Emitted from the GUI thread, received on the playback thread. Signals
    # rather than direct calls because Qt then queues them across the thread
    # boundary for us.
    request_open = Signal()
    request_frame = Signal()
    request_seek = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Enhancer")
        # The floor is what the panels genuinely need: settings scroll plus a
        # pinned forecast beside the picture, over the progress strip. Claiming
        # a smaller minimum only means clipping something.
        self.setMinimumSize(940, 680)
        self.resize(1320, 840)
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
        self.compare_thread: QThread | None = None
        self.compare_worker: CompareWorker | None = None
        self.play_thread: QThread | None = None
        self.play_worker: PlaybackWorker | None = None
        self.comparison: Path | None = None
        self._play_fps = 24.0
        self._play_duration = 0.0
        self._frame_in_flight = False
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._tick)
        self.theme_mode = theme.load_mode()

        # The picture takes the main area; the controls sit beside it.
        #
        # Everything the theme aims at — restraint, no colour for its own sake,
        # nothing competing with the image — was written for a window that had
        # no image in it. Thirty controls in two scrolling columns is a wall
        # however carefully it is spaced, and the one thing the user actually
        # needs to look at, the frame, was not on screen at all.
        settings = self._settings_panel()
        viewer = self._viewer_panel()

        self.columns = QSplitter(Qt.Horizontal)
        self.columns.addWidget(viewer)
        self.columns.addWidget(settings)
        self.columns.setStretchFactor(0, 3)
        self.columns.setStretchFactor(1, 2)
        self.columns.setChildrenCollapsible(False)
        # Air between the panes. Without it the viewer's zoom buttons and the
        # settings tab bar sit on the same line a few pixels apart and read as
        # one toolbar belonging to neither.
        self.columns.setHandleWidth(theme.GAP_WIDE)
        viewer.setMinimumWidth(300)
        settings.setMinimumWidth(340)
        self.columns.setSizes([700, 540])

        # Progress and the log get whatever height the user gives them, and no
        # more. Fixed, they took nearly a third of the window to show one bar
        # and an empty box, squeezing the picture — which is the one thing
        # worth looking at — into a strip.
        strip = self._status_strip()
        self.rows = QSplitter(Qt.Vertical)
        self.rows.addWidget(self.columns)
        self.rows.addWidget(strip)
        self.rows.setStretchFactor(0, 1)
        self.rows.setStretchFactor(1, 0)
        self.rows.setChildrenCollapsible(False)
        self.rows.setHandleWidth(theme.GAP)
        self.rows.setSizes([580, 200])

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(theme.GAP_WIDE, theme.GAP, theme.GAP_WIDE, theme.GAP)
        layout.setSpacing(theme.GAP)
        layout.addLayout(self._header())
        layout.addWidget(self.rows, 1)
        self.setCentralWidget(root)

        self._watch_settings()
        self._update_forecast()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    # --- construction -------------------------------------------------------

    def _header(self) -> QHBoxLayout:
        """One line naming what is loaded, and the two window-level actions."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.GAP_TIGHT)

        self.title_label = QLabel("Drop a video or image here")
        self.title_label.setObjectName("headline")
        row.addWidget(self.title_label, 1)

        self.theme_button = QToolButton()
        self.theme_button.setObjectName("segment")
        self.theme_button.setToolTip(HELP["theme_toggle"])
        self.theme_button.clicked.connect(self._toggle_theme)
        self._sync_theme_button()
        row.addWidget(self.theme_button)

        self.guide_button = QPushButton("Guide")
        self.guide_button.setToolTip(HELP["guide_button"])
        self.guide_button.clicked.connect(self._open_guide)
        row.addWidget(self.guide_button)
        return row

    def _viewer_panel(self) -> QWidget:
        """Before and after for a single frame, with the controls that drive it."""
        from .viewer import CompareView

        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(theme.GAP_TIGHT)

        self.view = CompareView()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.view, 1)

        # Which moment to test. A face in even light shows texture loss; a wide
        # shot or a dark scene hides exactly what this view exists to catch.
        row = QHBoxLayout()
        row.setSpacing(theme.GAP_TIGHT)
        row.addWidget(field_label("Frame at"))
        self.compare_time = QDoubleSpinBox()
        self.compare_time.setDecimals(1)
        self.compare_time.setSuffix(" s")
        self.compare_time.setRange(0.0, 0.0)
        self.compare_time.setSingleStep(1.0)
        self.compare_time.setToolTip(HELP["compare_time"])
        self.compare_time.setEnabled(False)
        row.addWidget(self.compare_time)
        row.addWidget(help_button("compare_time"))

        self.compare_button = QPushButton("Compare this frame")
        self.compare_button.setObjectName("primary")
        self.compare_button.setToolTip(HELP["compare_button"])
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self._run_compare)
        row.addWidget(self.compare_button, 1)
        row.addWidget(help_button("compare_button"))
        v.addLayout(row)
        v.addLayout(self._transport())
        return panel

    def _transport(self) -> QHBoxLayout:
        """Playback of a finished result against the original.

        A single frame hides the two faults that only show in motion: grain
        that pulses between frames, and skin that slides between detailed and
        waxy as the light changes.
        """
        row = QHBoxLayout()
        row.setSpacing(theme.GAP_TIGHT)

        self.play_button = QPushButton("Play")
        self.play_button.setToolTip(HELP["play_button"])
        self.play_button.setEnabled(False)
        self.play_button.setMinimumWidth(80)
        self.play_button.clicked.connect(self._toggle_play)
        row.addWidget(self.play_button)
        row.addWidget(help_button("play_button"))

        self.position = QSlider(Qt.Horizontal)
        self.position.setRange(0, 0)
        self.position.setEnabled(False)
        self.position.setToolTip(HELP["position"])
        self.position.sliderReleased.connect(self._seek_to_slider)
        row.addWidget(self.position, 1)

        self.time_label = QLabel("--:-- / --:--")
        self.time_label.setObjectName("caption")
        row.addWidget(self.time_label)

        self.loop_check = QCheckBox("Loop")
        self.loop_check.setToolTip(HELP["loop"])
        row.addWidget(self.loop_check)
        row.addWidget(help_button("loop"))

        self.compare_with_button = QPushButton("Compare with...")
        self.compare_with_button.setToolTip(HELP["compare_with"])
        self.compare_with_button.clicked.connect(self._browse_comparison)
        row.addWidget(self.compare_with_button)
        row.addWidget(help_button("compare_with"))
        return row

    def _settings_panel(self) -> QWidget:
        """Source above, everything else behind tabs, forecast below.

        Tabs rather than one long column because only one group is ever being
        adjusted at a time, while all six were competing for attention at once.
        Source and the forecast stay visible in every tab: what is loaded and
        what will come out are the context for every other decision.
        """
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, theme.GAP_TIGHT, 0)
        v.setSpacing(theme.GAP)

        # One group per tab, Source included. Stacking Source above the tabs
        # left the tab pages roughly a hundred and fifty pixels tall, which put
        # Degrain, Detail retention and Re-grain — the controls that decide
        # whether faces come out waxy — below the fold behind a scrollbar.
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._tab(self._source_group()), "Source")
        self.tabs.addTab(self._tab(self._model_group()), "Model")
        self.tabs.addTab(self._tab(self._texture_group()), "Texture")
        self.tabs.addTab(self._tab(self._fps_group()), "Motion")
        self.tabs.addTab(self._tab(self._output_group()), "Performance")
        self.tabs.addTab(self._tab(self._queue_group()), "Queue")
        v.addWidget(self.tabs)
        v.addStretch(1)

        # Without this the groups crush into each other and overlap as soon as
        # the window is shorter than their natural height — they lose their
        # padding, then their borders touch, then content spills across them.
        self.scroll = QScrollArea()
        self.scroll.setWidget(inner)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # The forecast sits outside the scroll area deliberately. Its whole
        # purpose is to be read immediately before pressing Render, and it
        # cannot do that from below the fold.
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.GAP)
        # A floor, or pinning the forecast starves the scroll area down to a
        # sliver showing one clipped label.
        self.scroll.setMinimumHeight(300)
        outer.addWidget(self.scroll, 1)
        outer.addWidget(self._forecast_group(), 0)
        return panel

    def _tab(self, *groups: QWidget) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, theme.GAP, 0, 0)
        v.setSpacing(theme.GAP_WIDE)
        for group in groups:
            v.addWidget(group)
        v.addStretch(1)
        return page

    def _status_strip(self) -> QWidget:
        """Progress, the render actions and the log, in one block."""
        strip = QWidget()
        v = QVBoxLayout(strip)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(theme.GAP_TIGHT)
        v.addWidget(self._progress_group(), 1)
        v.addLayout(self._actions())
        # Enough for the bar, the status line, two lines of log and the
        # action row. Below this the log is clipped mid-line, which looks
        # broken rather than compact.
        strip.setMinimumHeight(196)
        return strip

    def _sync_theme_button(self) -> None:
        dark = theme.resolve(QApplication.instance(), self.theme_mode) is theme.DARK
        self.theme_button.setText("Light" if dark else "Dark")

    def _toggle_theme(self) -> None:
        """Flip to the opposite of what is showing, and remember it.

        Deliberately two-state rather than cycling through Follow system: a
        toggle that sometimes appears to do nothing, because the system
        setting already matched, reads as broken.
        """
        app = QApplication.instance()
        dark = theme.resolve(app, self.theme_mode) is theme.DARK
        self.theme_mode = theme.Mode.LIGHT if dark else theme.Mode.DARK
        theme.apply(app, self.theme_mode)
        self._sync_theme_button()

    def _source_group(self) -> QGroupBox:
        box = QGroupBox("Source")
        v = QVBoxLayout(box)
        v.setSpacing(theme.GAP)

        self.drop_label = QLabel("Drop a video or image here, or use Browse")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setWordWrap(True)
        self.drop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.drop_label.setMinimumHeight(44)
        self.drop_label.setObjectName("dropzone")

        # Browse sits beside the drop target rather than under it. The header
        # already names the loaded file, so a tall box repeating that name was
        # spending sixty pixels of the settings column to say nothing.
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_source)
        picker = QHBoxLayout()
        picker.setContentsMargins(0, 0, 0, 0)
        picker.setSpacing(theme.GAP_TIGHT)
        picker.addWidget(self.drop_label, 1)
        picker.addWidget(browse, 0)
        picker.addWidget(help_button("source"), 0, Qt.AlignTop)
        v.addLayout(picker)

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

        path = self.model_combo.currentData()
        name = Path(path).name if path else self.model_combo.currentText()
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

        # The three actions group on the right in the order they are used.
        # Every button carries its own tooltip, so no separate markers here —
        # interleaving them made the row unreadable.
        for button in (self.preview_button, self.render_button, self.cancel_button):
            button.setMinimumWidth(110)

        row.addStretch(1)
        row.addWidget(self.preview_button)
        row.addWidget(self.render_button)
        row.addWidget(self.cancel_button)
        return row

    def _progress_group(self) -> QGroupBox:
        box = QGroupBox("Progress")
        v = QVBoxLayout(box)
        v.setSpacing(theme.GAP_TIGHT)
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
        self.log_view.setMinimumHeight(44)
        # Capped, or it grows without limit and squeezes the settings out
        # of the window when the window is short. The strip it sits in is a
        # splitter pane, so anyone who wants a taller log can drag for it.
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
        path = self.model_combo.currentData()
        name = Path(path).name if path else self.model_combo.currentText()
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
        width = self.profile.width if self.profile else (
            self._image_size[0] if self._image_size else None)
        for p in sorted(found, key=lambda q: (model_rank(q.name), q.name)):
            self.model_combo.addItem(display_name(p.name, width), str(p))
            index = self.model_combo.count() - 1
            # The file name is still what identifies it, so keep it visible in
            # the tooltip alongside what the model actually suits.
            self.model_combo.setItemData(
                index,
                p.name + "\n\n" + describe_model(p.name),
                Qt.ToolTipRole,
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
            self, "Choose a source", "", source_filter()
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
        self.title_label.setText(path.name)
        self.analysis.setText("Analysing...")
        self.view.clear()
        # Detach whatever result was attached to the previous source. Left
        # alone, Play would stream the old film's original against the old
        # film's render while the window named the new one.
        self._stop_playback()
        self.compare_button.setEnabled(True)
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
        duration = p.frame_count / p.fps if p.fps > 0 else 0.0
        self.compare_time.setRange(0.0, max(0.0, duration - 0.1))
        self.compare_time.setEnabled(duration > 0)
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
        self.compare_time.setRange(0.0, 0.0)
        self.compare_time.setEnabled(False)

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

    # --- playback -----------------------------------------------------------

    def _browse_comparison(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Choose a finished result", "",
            f"Video ({' '.join('*' + s for s in sorted(VIDEO_SUFFIXES))});;All files (*)",
        )
        if name:
            self._load_comparison(Path(name))

    def _load_comparison(self, path: Path) -> None:
        """Attach a finished result to the original for playback."""
        if self.source is None:
            return
        self._stop_playback()

        self.play_thread = QThread(self)
        self.play_worker = PlaybackWorker(self.source, path)
        self.play_worker.moveToThread(self.play_thread)
        self.play_worker.opened.connect(self._on_playback_opened)
        self.play_worker.frame.connect(self._on_playback_frame)
        self.play_worker.ended.connect(self._on_playback_ended)
        self.play_worker.failed.connect(self._on_playback_failed)
        self.request_open.connect(self.play_worker.open)
        self.request_frame.connect(self.play_worker.deliver_next)
        self.request_seek.connect(self.play_worker.seek)
        self.play_thread.start()
        self.comparison = path
        self.compare_with_button.setText(path.name)
        self.request_open.emit()

    def _on_playback_opened(self, fps: float, frames: int, start: float, end: float) -> None:
        self._play_fps = fps if fps > 0 else 24.0
        self.position.setRange(0, max(0, frames - 1))
        self.position.setEnabled(frames > 0)
        self.play_button.setEnabled(frames > 0)
        self._play_duration = frames / self._play_fps if self._play_fps else 0.0
        self._append(
            f"Ready to play {self.comparison.name}: {frames} frames at "
            f"{self._play_fps:.3f} fps, covering {start:.1f}-{end:.1f}s of the source."
        )
        # Show the first frame straight away, so attaching a result is visibly
        # different from attaching nothing.
        self._frame_in_flight = True
        self.request_frame.emit()

    def _on_playback_frame(self, pair) -> None:
        self._frame_in_flight = False
        self.view.set_frame(pair.before, pair.after)
        if not self.position.isSliderDown():
            self.position.blockSignals(True)
            self.position.setValue(pair.index)
            self.position.blockSignals(False)
        self.time_label.setText(
            f"{_clock(pair.seconds)} / {_clock(self._play_duration)}"
        )

    def _on_playback_ended(self) -> None:
        self._frame_in_flight = False
        if self.loop_check.isChecked() and self.play_timer.isActive():
            self.request_seek.emit(0.0)
            return
        self._pause()

    def _on_playback_failed(self, message: str) -> None:
        self._frame_in_flight = False
        self._pause()
        self._append(f"Playback failed: {message}")
        QMessageBox.warning(self, "Could not play", message)

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self._pause()
        else:
            self.view.set_playing(True)
            self.play_button.setText("Pause")
            self.play_timer.start(max(1, round(1000.0 / self._play_fps)))

    def _pause(self) -> None:
        self.play_timer.stop()
        self.view.set_playing(False)
        self.play_button.setText("Play")

    def _tick(self) -> None:
        """Ask for the next frame, unless the last one has not arrived.

        Dropping a frame is the right failure here. Queueing another request
        behind a slow decode would put playback further behind real time with
        every tick, and it would never recover.
        """
        if self._frame_in_flight:
            return
        self._frame_in_flight = True
        self.request_frame.emit()

    def _seek_to_slider(self) -> None:
        if self._play_fps:
            self._frame_in_flight = True
            self.request_seek.emit(self.position.value() / self._play_fps)

    def _stop_playback(self) -> None:
        self._pause()
        if self.play_thread is not None:
            # Quit and wait FIRST. shutdown() closes the ffmpeg pipes the
            # worker thread is reading from, so calling it while that thread
            # is still live races a decode already in progress.
            self.play_thread.quit()
            self.play_thread.wait()
            if self.play_worker is not None:
                self.play_worker.shutdown()
            self.play_thread.deleteLater()
        self.play_thread = None
        self.play_worker = None
        self._frame_in_flight = False
        self.comparison = None
        self.compare_with_button.setText("Compare with...")
        self.play_button.setEnabled(False)
        self.position.setEnabled(False)
        self.position.setRange(0, 0)
        self.time_label.setText("--:-- / --:--")

    # --- single-frame comparison --------------------------------------------

    def _run_compare(self) -> None:
        """Put one frame through the current settings and show it beside the source."""
        if self.compare_thread is not None:
            return
        request = self._build_request(preview=False)
        if request is None:
            return

        self.compare_button.setEnabled(False)
        self.compare_button.setText("Comparing...")

        self.compare_thread = QThread(self)
        self.compare_worker = CompareWorker(request, self.compare_time.value())
        self.compare_worker.moveToThread(self.compare_thread)
        self.compare_thread.started.connect(self.compare_worker.run)
        self.compare_worker.log.connect(self._append)
        self.compare_worker.ready.connect(self._on_compare_ready)
        self.compare_worker.failed.connect(self._on_compare_failed)
        self.compare_thread.start()

    def _on_compare_ready(self, pair) -> None:
        self.view.set_pair(pair.before, pair.after)
        w, h = pair.source_size
        after_h, after_w = pair.after.shape[:2]
        self._append(
            f"Compared {w}x{h} against {after_w}x{after_h} at {pair.seconds:.1f}s "
            f"(frame {pair.frame_index})."
        )
        self._end_compare()

    def _on_compare_failed(self, message: str) -> None:
        self._append(f"Comparison failed: {message}")
        QMessageBox.warning(self, "Could not compare", message)
        self._end_compare()

    def _end_compare(self) -> None:
        if self.compare_thread is not None:
            self.compare_thread.quit()
            self.compare_thread.wait()
            self.compare_thread.deleteLater()
        self.compare_thread = None
        self.compare_worker = None
        self.compare_button.setText("Compare this frame")
        self.compare_button.setEnabled(self.source is not None)

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

        # Attach what was just produced, so Play works without hunting for the
        # file. A still has nothing to play.
        from .images import is_image

        result = Path(path)
        if self.source is not None and not is_image(result) and result.exists():
            self._load_comparison(result)

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
        # Comparing loads a second copy of the model onto the same card. On
        # Windows the driver oversubscribes graphics memory silently rather
        # than failing, so this would not crash — it would quietly slow a
        # render that has hours left to run, with nothing on screen to say why.
        self.compare_button.setEnabled(not running and self.source is not None)
        self.compare_button.setToolTip(
            "Not while a render is using the graphics card."
            if running else HELP["compare_button"]
        )

    def _append(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the background threads before Qt tears their objects down.

        A QThread deleted while still running aborts the process, so closing
        the window mid-comparison — or mid-render — would take the application
        down with it rather than shutting it cleanly. A render that is stopped
        this way resumes from its journal on the next run; a comparison writes
        nothing and is simply abandoned.
        """
        for thread, worker in (
            (self.compare_thread, self.compare_worker),
            (self.thread, self.worker),
        ):
            if thread is None:
                continue
            if worker is not None and hasattr(worker, "cancel"):
                worker.cancel()
            thread.quit()
            thread.wait()
        self.compare_thread = self.thread = None
        self.compare_worker = self.worker = None
        self._stop_playback()
        super().closeEvent(event)

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
