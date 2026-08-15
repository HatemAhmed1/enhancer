"""Qt window. Contains no policy — every decision lives in RenderRequest."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .analyze import classify_scan, estimate_blockiness, estimate_grain, probe_scan
from .gui import CancelledError, RenderJob
from .help_text import HELP, describe_model
from .models import scan_custom_dir
from .requests import RenderRequest
from .video_io import Decoder, SourceProfile

MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".png", ".jpg", ".jpeg"}
PREVIEW_SECONDS = 10


class Worker(QObject):
    """Runs a render on a background thread."""

    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

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
                model, tile=tile, overlap=req.overlap, device=device, half=not req.cpu
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
    label.setToolTip(HELP.get(key, "No description available."))
    label.setAlignment(Qt.AlignCenter)
    label.setFixedSize(18, 18)
    label.setCursor(Qt.WhatsThisCursor)
    label.setStyleSheet(
        "QLabel { border: 1px solid palette(mid); border-radius: 9px;"
        " color: palette(mid); font-size: 11px; font-weight: bold; }"
        "QLabel:hover { color: palette(highlight); border-color: palette(highlight); }"
    )
    return label


def with_help(widget: QWidget, key: str) -> QWidget:
    """Pair a control with its '?' button on one row."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addWidget(widget, 1)
    row.addWidget(help_button(key), 0, Qt.AlignTop)
    return holder


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

        # Landscape: settings in two columns side by side, progress spanning the
        # full width beneath. The splitter lets the columns be rebalanced, and
        # every group expands with the window rather than sitting at a fixed size.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.addWidget(self._source_group(), 1)
        left_layout.addWidget(self._model_group())
        left_layout.addWidget(self._output_group())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.addWidget(self._texture_group())
        right_layout.addWidget(self._fps_group())
        right_layout.addStretch(1)

        self.columns = QSplitter(Qt.Horizontal)
        self.columns.addWidget(left)
        self.columns.addWidget(right)
        self.columns.setStretchFactor(0, 3)
        self.columns.setStretchFactor(1, 2)
        self.columns.setChildrenCollapsible(False)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(self.columns, 1)
        layout.addLayout(self._actions())
        layout.addWidget(self._progress_group())
        self.setCentralWidget(root)

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    # --- construction -------------------------------------------------------

    def _source_group(self) -> QGroupBox:
        box = QGroupBox("Source")
        v = QVBoxLayout(box)

        self.drop_label = QLabel("Drop a video or image here, or use Browse")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setWordWrap(True)
        self.drop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.drop_label.setMinimumHeight(70)
        self.drop_label.setStyleSheet(
            "border: 2px dashed palette(mid); padding: 16px; border-radius: 6px;"
        )
        v.addWidget(with_help(self.drop_label, "source"))

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_source)
        v.addWidget(browse)

        self.analysis = QLabel("No source loaded.")
        self.analysis.setWordWrap(True)
        self.analysis.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.analysis.setAlignment(Qt.AlignTop)
        self.analysis.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(with_help(self.analysis, "analysis"), 1)
        return box

    def _model_group(self) -> QGroupBox:
        box = QGroupBox("Model")
        f = QFormLayout(box)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._reload_models()
        self.model_combo.currentIndexChanged.connect(self._show_model_note)
        f.addRow("Weights", with_help(self.model_combo, "model"))

        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet("color: palette(mid); font-size: 11px;")
        f.addRow("", self.model_note)
        self._show_model_note()

        refresh = QPushButton("Rescan models/custom")
        refresh.clicked.connect(self._reload_models)
        f.addRow("", with_help(refresh, "rescan"))
        return box

    def _texture_group(self) -> QGroupBox:
        box = QGroupBox("Restoration and texture")
        f = QFormLayout(box)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.no_restore = QCheckBox("Skip all restoration (fastest, for previews)")
        f.addRow("", with_help(self.no_restore, "no_restore"))

        self.degrain = _slider(0, 100, 25)
        f.addRow("Degrain", self._labelled(
            self.degrain, "degrain",
            "higher removes more grain — and more skin texture"))

        self.detail = _slider(0, 100, 25)
        f.addRow("Detail retention", self._labelled(
            self.detail, "detail",
            "restores real detail from your source file"))

        self.regrain = _slider(0, 100, 60)
        f.addRow("Re-grain", self._labelled(
            self.regrain, "regrain",
            "the strongest fix for plastic-looking skin"))

        self.deblock = _slider(0, 100, 0)
        f.addRow("Deblock", self._labelled(
            self.deblock, "deblock",
            "raise for YouTube and low-bitrate sources"))
        return box

    def _labelled(self, slider: QSlider, key: str, hint: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        value = QLabel(f"{slider.value() / 100:.2f}")
        value.setFixedWidth(34)
        slider.valueChanged.connect(lambda n: value.setText(f"{n / 100:.2f}"))
        slider.setToolTip(HELP.get(key, ""))
        row.addWidget(slider, 1)
        row.addWidget(value)
        row.addWidget(help_button(key))
        v.addLayout(row)
        note = QLabel(hint)
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid); font-size: 11px;")
        v.addWidget(note)
        return w

    def _fps_group(self) -> QGroupBox:
        box = QGroupBox("Frame rate (make motion smoother)")
        f = QFormLayout(box)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.fps_mode = QComboBox()
        self.fps_mode.addItems(["Off", "Target FPS", "Multiplier"])
        for i, note in enumerate([
            "Keep the source frame rate exactly as it is.",
            "Name the frame rate you want out, such as 60.",
            "Multiply whatever the source happens to be, such as 2x.",
        ]):
            self.fps_mode.setItemData(i, note, Qt.ToolTipRole)
        f.addRow("Mode", with_help(self.fps_mode, "fps_mode"))

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
        f.addRow("Target", with_help(self.fps_target, "fps_target"))

        self.fps_multiplier = QDoubleSpinBox()
        self.fps_multiplier.setRange(1.0, 8.0)
        self.fps_multiplier.setValue(2.0)
        f.addRow("Multiplier", with_help(self.fps_multiplier, "fps_multiplier"))

        self.scene_threshold = QDoubleSpinBox()
        self.scene_threshold.setRange(0.0, 1.0)
        self.scene_threshold.setSingleStep(0.05)
        self.scene_threshold.setValue(0.30)
        f.addRow("Cut sensitivity", with_help(self.scene_threshold, "scene_threshold"))

        self.fps_note = QLabel("")
        self.fps_note.setWordWrap(True)
        self.fps_note.setStyleSheet("color: palette(mid); font-size: 11px;")
        f.addRow("", self.fps_note)
        self._check_rife()
        return box

    def _output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        f = QFormLayout(box)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.output_label = QLabel("(chosen automatically beside the source)")
        self.output_label.setWordWrap(True)
        f.addRow("File", with_help(self.output_label, "output"))

        pick = QPushButton("Choose output...")
        pick.clicked.connect(self._browse_output)
        f.addRow("", pick)

        self.segment_frames = QSpinBox()
        self.segment_frames.setRange(30, 10000)
        self.segment_frames.setValue(500)
        f.addRow("Segment frames", with_help(self.segment_frames, "segment_frames"))

        self.cpu = QCheckBox("Force CPU (very slow)")
        f.addRow("", with_help(self.cpu, "cpu"))
        return box

    def _actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.preview_button = QPushButton(f"Preview {PREVIEW_SECONDS}s")
        self.preview_button.setToolTip(HELP["preview_button"])
        self.preview_button.clicked.connect(lambda: self._start(preview=True))
        self.render_button = QPushButton("Render")
        self.render_button.setToolTip(HELP["render_button"])
        self.render_button.clicked.connect(lambda: self._start(preview=False))
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip(HELP["cancel_button"])
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)

        row.addWidget(self.preview_button, 1)
        row.addWidget(help_button("preview_button"))
        row.addWidget(self.render_button, 1)
        row.addWidget(help_button("render_button"))
        row.addWidget(self.cancel_button, 1)
        row.addWidget(help_button("cancel_button"))
        return row

    def _progress_group(self) -> QGroupBox:
        box = QGroupBox("Progress")
        v = QVBoxLayout(box)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        top.addWidget(self.bar, 1)
        top.addWidget(help_button("progress"))
        v.addLayout(top)

        self.status = QLabel("Idle.")
        v.addWidget(self.status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setMinimumHeight(90)
        v.addWidget(self.log_view)
        return box

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

        self._set_running(True)
        self._started_at = time.perf_counter()
        self.bar.setValue(0)
        self._append(f"{'Preview' if preview else 'Render'} -> {request.output}")

        self.thread = QThread()
        self.worker = Worker(request)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.thread.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self._append("Cancelling after the current frame...")

    def _on_progress(self, done: int, total: int) -> None:
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
        self._teardown()

    def _on_failed(self, message: str) -> None:
        self._append(message)
        self.status.setText("Stopped.")
        self._teardown()

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


def launch(argv: list[str] | None = None) -> int:
    app = QApplication(argv or [])
    window = MainWindow()
    window.show()
    return app.exec()
