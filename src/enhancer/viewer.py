"""Before/after image comparison.

The point of this widget is one question: did human skin keep its texture, or
did the upscaler turn it into wax? That question cannot be answered at
fit-to-window — the artefact lives at the scale of a pore. So everything here
exists to get the eye to real pixels, side by side, at the same place in both
frames:

* three ways to compare — a draggable swipe divider, a side-by-side split, and
  an in-place toggle (the eye catches *change* far better than *difference*,
  so flipping one image over the other is the most sensitive test of the
  three);
* zoom and pan shared by both images, so they can never drift out of register;
* nearest-neighbour sampling once magnified past 100%, because a smoothing
  filter invents detail that is not in the file and would paper over exactly
  the artefact being hunted.

Input controls
--------------
====================  ========================================================
Wheel / Ctrl+wheel    Zoom about the cursor (not about the centre).
Left drag             Swipe mode: move the divider.
                      Split/toggle mode: pan.
Middle drag           Pan, in every mode.
Space (held)          Toggle mode: show BEFORE while held, AFTER on release.
Ctrl+0 / Ctrl+1       Fit to window / 100%.
Ctrl+2                200%.
====================  ========================================================

Why QImage and not QPixmap
--------------------------
This started as a still-image widget, so each frame was converted once into a
QPixmap and that cost did not matter. Playback pushes a new pair 24-60 times a
second, and ``QPixmap.fromImage`` is a full-frame copy *and* a 24bpp->32bpp
widening: measured at 13.0 ms for one 3840x2160 frame on this machine, so 26 ms
per pair before a single pixel is drawn. That alone capped a 4K swipe at 30 fps.

Wrapping the numpy buffer in a ``QImage`` is free — it does not copy — and
``QPainter.drawImage`` blits straight from it. The blit itself is dearer than
the pixmap blit (6.1 ms against 1.6 ms for a half-clipped 4K pane fitted to a
1600x900 viewport), but it replaces 14.6 ms of convert-then-blit with 6.1 ms of
blit, and 4K swipe playback roughly doubles. Pre-scaling with ``QImage.scaled``
(33 ms) and widening to a 32bpp layout in numpy (82 ms) were both measured and
are both far worse than letting the blitter do it.

The price is that the numpy buffers are now load-bearing rather than defensive:
see ``_to_image``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFontMetricsF,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .theme import GAP_TIGHT, RADIUS

MODES = ("swipe", "split", "toggle")

MIN_ZOOM = 0.02
MAX_ZOOM = 64.0

# Gutter between the two panes in split mode.
SPLIT_GAP = 10


def _to_image(array: np.ndarray, name: str) -> tuple[QImage, np.ndarray]:
    """Wrap one HWC uint8 RGB array in a QImage, without copying it.

    Returns the image *and* the C-contiguous array it is a view of. **The
    caller must keep that array alive for exactly as long as it keeps the
    image.** ``QImage`` does not take ownership of the buffer it is handed and
    does not copy it, so the numpy array is the only owner of those pixels. Let
    it be collected and Qt paints happily from freed memory — this is a real
    crash, not a theoretical one.

    The widget used to convert to a QPixmap here, which deep-copied the pixels
    and made the retained array merely defensive. It no longer does (see the
    module docstring for the measurements), so the two attributes that hold
    these buffers are now the thing keeping the picture valid. They must be
    replaced as a pair with the images they back, and never cleared while an
    image that views them survives.

    The caller must also not mutate or reuse the array afterwards: there is no
    snapshot, so an overwrite lands on screen at the next repaint, half a frame
    at a time. Playback must hand over a fresh array per frame, not refill one.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"{name} must be an HWC RGB array with 3 channels, got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {array.dtype}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {array.shape}")

    # A non-contiguous or oddly strided array read with the wrong bytesPerLine
    # produces a sheared image, so make contiguity explicit and then take the
    # stride Qt should use straight from the array we actually pass.
    buffer = np.ascontiguousarray(array)
    height, width = buffer.shape[:2]
    image = QImage(
        buffer.data,
        width,
        height,
        buffer.strides[0],
        QImage.Format.Format_RGB888,
    )
    return image, buffer


class CompareView(QWidget):
    """A before/after pair with shared zoom and pan."""

    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._before: QImage | None = None
        self._after: QImage | None = None
        # NOT defensive: these arrays own the pixels the two QImages view, and
        # dropping one while its image survives is a use-after-free. See
        # _to_image. Always replaced in lockstep with _before / _after.
        self._before_buffer: np.ndarray | None = None
        self._after_buffer: np.ndarray | None = None

        self._mode = "swipe"
        self._playing = False
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)   # image top-left, in viewport coords
        self._fit_mode = True
        self._divider = 0.5                # fraction of the viewport width
        self._toggled = False              # toggle mode: True shows BEFORE

        self._dragging_divider = False
        self._panning = False
        self._pan_anchor = QPointF(0.0, 0.0)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._build_bar()
        self._update_caption()

    # ------------------------------------------------------------------ setup

    def _build_bar(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(GAP_TIGHT)

        self._bar = QWidget(self)
        row = QHBoxLayout(self._bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(GAP_TIGHT)

        self._segments = QButtonGroup(self)
        self._segments.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        for mode, label in (("swipe", "Swipe"), ("split", "Split"), ("toggle", "Toggle")):
            button = QPushButton(label, self._bar)
            button.setObjectName("segment")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, m=mode: self._segment_clicked(m))
            self._segments.addButton(button)
            self._mode_buttons[mode] = button
            row.addWidget(button)
        self._mode_buttons[self._mode].setChecked(True)

        row.addStretch(1)

        self.caption = QLabel("", self._bar)
        self.caption.setObjectName("caption")
        row.addWidget(self.caption)

        row.addSpacing(GAP_TIGHT)

        for label, action in (
            ("Fit", self.fit),
            ("100%", lambda: self.set_zoom(1.0)),
            ("200%", lambda: self.set_zoom(2.0)),
        ):
            button = QPushButton(label, self._bar)
            button.clicked.connect(action)
            row.addWidget(button)

        outer.addWidget(self._bar)
        outer.addStretch(1)

        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.fit)
        QShortcut(QKeySequence("Ctrl+1"), self, activated=lambda: self.set_zoom(1.0))
        QShortcut(QKeySequence("Ctrl+2"), self, activated=lambda: self.set_zoom(2.0))

    # -------------------------------------------------------------- public API

    def set_pair(self, before: np.ndarray, after: np.ndarray) -> None:
        """Show a before/after pair of HWC uint8 RGB arrays of identical size."""
        previous = self._accept(before, after)

        # A new frame of the same size keeps the current zoom and pan, so
        # stepping through preview frames does not throw the eye back to
        # fit-to-window every time. A different size has no shared frame of
        # reference, so start over.
        if previous != self.image_size():
            self.fit()
        else:
            self._clamp_offset()
        self._update_caption()
        self.update()

    def set_frame(self, before: np.ndarray, after: np.ndarray) -> None:
        """A pair during playback: same contract as set_pair, but on the hot path.

        It delegates, and that is the finding rather than an oversight. Once the
        two QPixmap conversions were gone there was nothing left in set_pair
        that a still image could afford and playback could not: pushing a pair
        costs 0.17 ms at 4K, against 26.8 ms before, and everything else on the
        path is arithmetic on a handful of floats.

        Two further divergences were written, measured and removed: invalidating
        only the picture rect instead of the whole widget (3.8% fewer pixels
        repainted, fps difference inside run-to-run noise) and skipping the
        caption when its text had not changed (0.8 us of a 16.7 ms frame). Both
        were complexity on the most delicate path in the widget for nothing.

        The behaviour playback depends on it therefore inherits exactly: zoom,
        pan, divider and mode all survive a new frame, because set_pair only
        refits when the image size changes — and mid-playback it does not. A
        genuine size change, at a reel boundary say, still refits, which is
        right: a pan into a frame that no longer exists cannot be held.

        What must not be forgotten is the buffer rule in _to_image. Nothing
        copies the pixels any more, so each call must be handed a fresh array
        the caller then leaves alone; a decode buffer refilled in place will
        tear on screen.
        """
        self.set_pair(before, after)

    def _accept(self, before: np.ndarray, after: np.ndarray) -> QSize | None:
        """Validate and install a pair. Returns the size that was on screen.

        Both images are built before either is installed, so a bad `after`
        cannot leave a new `before` paired with the previous `after` at a
        different size. The two (image, buffer) assignments then happen with no
        repaint in between, which is what keeps the buffers and the images that
        view them in step — see _to_image.
        """
        if before.shape != after.shape:
            raise ValueError(
                "before and after must have identical dimensions, got "
                f"{before.shape} and {after.shape}"
            )
        new_before = _to_image(before, "before")
        new_after = _to_image(after, "after")

        previous = self.image_size()
        self._before, self._before_buffer = new_before
        self._after, self._after_buffer = new_after
        return previous

    def clear(self) -> None:
        """Drop the pair. The widget stays paintable."""
        # Images first, then the buffers they view: never the other way round.
        self._before = self._after = None
        self._before_buffer = self._after_buffer = None
        self._toggled = False
        self._fit_mode = True
        self._playing = False
        self._update_caption()
        self.update()

    def is_playing(self) -> bool:
        """Whether the window is currently driving frames through set_frame."""
        return self._playing

    def set_playing(self, playing: bool) -> None:
        """Told to the widget by the window; the widget does not drive playback.

        It exists so the window can ask about playback state without reaching
        into internals, and so anything the widget paints can depend on it
        without a second source of truth.
        """
        playing = bool(playing)
        if playing == self._playing:
            return
        self._playing = playing
        self.update()

    def has_pair(self) -> bool:
        return self._before is not None and self._after is not None

    def image_size(self) -> QSize | None:
        return self._before.size() if self._before is not None else None

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        if mode == self._mode:
            return
        self._mode = mode
        self._toggled = False
        self._dragging_divider = False
        self._panning = False
        button = self._mode_buttons[mode]
        if not button.isChecked():
            button.setChecked(True)
        # Split halves the viewport, so a fitted view has to be refitted.
        if self._fit_mode:
            self.fit()
        else:
            self._clamp_offset()
        self._apply_cursor()
        self.update()

    def zoom(self) -> float:
        return self._scale

    def set_zoom(self, factor: float) -> None:
        """Zoom about the centre of the viewport."""
        viewport = self._viewports()[0]
        centre = QPointF(viewport.center()) if not viewport.isEmpty() else QPointF(0, 0)
        self.zoom_at(factor, centre)

    def zoom_at(self, factor: float, widget_pos: QPointF) -> None:
        """Zoom to `factor` keeping the image point under `widget_pos` still."""
        factor = max(MIN_ZOOM, min(MAX_ZOOM, float(factor)))
        index, viewport = self._viewport_at(widget_pos)
        if viewport.isEmpty() or not self.has_pair():
            self._set_scale(factor)
            self._update_caption()
            self.update()
            return

        local = QPointF(
            widget_pos.x() - viewport.x(),
            widget_pos.y() - viewport.y(),
        )
        # The image point currently under the cursor, which must not move.
        image_point = QPointF(
            (local.x() - self._offset.x()) / self._scale,
            (local.y() - self._offset.y()) / self._scale,
        )
        self._set_scale(factor)
        self._offset = QPointF(
            local.x() - image_point.x() * self._scale,
            local.y() - image_point.y() * self._scale,
        )
        self._fit_mode = False
        self._clamp_offset()
        self._update_caption()
        self.update()

    def fit(self) -> None:
        """Scale the pair to fill the viewport, and centre it."""
        self._fit_mode = True
        viewport = self._viewports()[0]
        size = self.image_size()
        if size is None or viewport.isEmpty() or size.width() == 0 or size.height() == 0:
            return
        scale = min(
            viewport.width() / size.width(),
            viewport.height() / size.height(),
        )
        self._set_scale(scale)
        self._centre()
        self._update_caption()
        self.update()

    def image_at(self, widget_pos: QPointF) -> QPointF | None:
        """Map a widget point to image coordinates, for tests and callers."""
        if not self.has_pair():
            return None
        _index, viewport = self._viewport_at(widget_pos)
        if viewport.isEmpty():
            return None
        return QPointF(
            (widget_pos.x() - viewport.x() - self._offset.x()) / self._scale,
            (widget_pos.y() - viewport.y() - self._offset.y()) / self._scale,
        )

    def transformation_mode(self) -> Qt.TransformationMode:
        """Nearest-neighbour once magnified, smooth when minified.

        Below 100% we are throwing pixels away, and smooth (area-averaged)
        sampling is the honest way to do that — nearest-neighbour would alias
        fine texture into noise that is not in the source. At or above 100% we
        are magnifying, and smoothing would blend neighbouring pixels into
        gradients that exist nowhere in the file. That is precisely the "waxy
        skin" look this widget was built to detect, so a filter that fabricates
        it defeats the whole comparison. Show the real pixels instead.
        """
        if self._scale > 1.0:
            return Qt.TransformationMode.FastTransformation
        return Qt.TransformationMode.SmoothTransformation

    def divider(self) -> float:
        return self._divider

    def set_divider(self, fraction: float) -> None:
        self._divider = max(0.0, min(1.0, float(fraction)))
        self.update()

    def showing_before(self) -> bool:
        """Toggle mode only: which of the two is currently on screen."""
        return self._toggled

    # -------------------------------------------------------------- geometry

    def _canvas_rect(self) -> QRect:
        """The picture area: everything below the control bar."""
        bar = self._bar.geometry()
        top = bar.bottom() + 1 + GAP_TIGHT if bar.isValid() else 0
        height = self.height() - top
        if height <= 0 or self.width() <= 0:
            return QRect()
        return QRect(0, top, self.width(), height)

    def _viewports(self) -> list[QRect]:
        """One rect per pane. Split has two; the other modes have one."""
        rect = self._canvas_rect()
        if rect.isEmpty():
            return [QRect()]
        if self._mode == "split":
            half = (rect.width() - SPLIT_GAP) // 2
            if half <= 0:
                return [rect]
            return [
                QRect(rect.x(), rect.y(), half, rect.height()),
                QRect(
                    rect.x() + half + SPLIT_GAP,
                    rect.y(),
                    rect.width() - half - SPLIT_GAP,
                    rect.height(),
                ),
            ]
        return [rect]

    def _viewport_at(self, pos: QPointF) -> tuple[int, QRect]:
        viewports = self._viewports()
        for index, viewport in enumerate(viewports):
            if viewport.contains(int(pos.x()), int(pos.y())):
                return index, viewport
        return 0, viewports[0]

    def _set_scale(self, scale: float) -> None:
        scale = max(MIN_ZOOM, min(MAX_ZOOM, float(scale)))
        if scale == self._scale:
            return
        self._scale = scale
        self.zoom_changed.emit(scale)

    def _centre(self) -> None:
        viewport = self._viewports()[0]
        size = self.image_size()
        if size is None or viewport.isEmpty():
            self._offset = QPointF(0.0, 0.0)
            return
        self._offset = QPointF(
            (viewport.width() - size.width() * self._scale) / 2.0,
            (viewport.height() - size.height() * self._scale) / 2.0,
        )

    def _clamp_offset(self) -> None:
        """Keep the image anchored: centred when smaller, in-bounds when larger."""
        viewport = self._viewports()[0]
        size = self.image_size()
        if size is None or viewport.isEmpty():
            return
        x = self._offset.x()
        y = self._offset.y()
        drawn_w = size.width() * self._scale
        drawn_h = size.height() * self._scale
        if drawn_w <= viewport.width():
            x = (viewport.width() - drawn_w) / 2.0
        else:
            x = min(0.0, max(viewport.width() - drawn_w, x))
        if drawn_h <= viewport.height():
            y = (viewport.height() - drawn_h) / 2.0
        else:
            y = min(0.0, max(viewport.height() - drawn_h, y))
        self._offset = QPointF(x, y)

    def _source_and_target(self, viewport: QRect) -> tuple[QRectF, QRectF] | None:
        """Only the visible slice of the source, so 4K stays interactive."""
        size = self.image_size()
        if size is None or viewport.isEmpty() or self._scale <= 0:
            return None
        left = -self._offset.x() / self._scale
        top = -self._offset.y() / self._scale
        source = QRectF(
            left,
            top,
            viewport.width() / self._scale,
            viewport.height() / self._scale,
        ).intersected(QRectF(0, 0, size.width(), size.height()))
        if source.isEmpty():
            return None
        target = QRectF(
            viewport.x() + self._offset.x() + source.x() * self._scale,
            viewport.y() + self._offset.y() + source.y() * self._scale,
            source.width() * self._scale,
            source.height() * self._scale,
        )
        return source, target

    def _picture_rect(self, viewport: QRect) -> QRectF:
        """Where the picture actually lands inside a pane.

        Captions hang off this and not off the pane, so that a letterboxed
        image does not end up labelled by a chip floating in empty background
        half a screen above it.
        """
        pair = self._source_and_target(viewport)
        if pair is None:
            return QRectF(viewport)
        return pair[1].intersected(QRectF(viewport))

    # -------------------------------------------------------------- painting

    def backdrop_colour(self) -> QColor:
        """The letterbox fill behind the picture, resolved on every repaint.

        Read through the module rather than bound once at construction: the
        header's light/dark toggle calls ``theme.apply`` while the window is
        open, so anything cached here would be a frame behind — or, worse,
        permanently stale.

        It was previously a translucent black wash, which meant the colour was
        whatever happened to be underneath: black on black in the dark palette,
        an unexplained grey band in the light one. See ``theme.VIEWER_BACKDROP``
        for why the replacement is one fixed neutral in both palettes and not a
        palette field.
        """
        return QColor(theme.VIEWER_BACKDROP)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        rect = self._canvas_rect()
        if rect.isEmpty():
            return
        painter.fillRect(rect, self.backdrop_colour())

        if not self.has_pair():
            painter.setPen(QColor(theme.VIEWER_BACKDROP_TEXT))
            painter.drawText(
                rect, Qt.AlignmentFlag.AlignCenter, "No comparison loaded yet"
            )
            return

        smooth = self.transformation_mode() == Qt.TransformationMode.SmoothTransformation
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)

        if self._mode == "split":
            self._paint_split(painter)
        elif self._mode == "toggle":
            self._paint_toggle(painter, rect)
        else:
            self._paint_swipe(painter, rect)

    def _paint_pane(self, painter: QPainter, viewport: QRect, image: QImage) -> None:
        pair = self._source_and_target(viewport)
        if pair is None:
            return
        source, target = pair
        painter.save()
        # Intersect, never replace: swipe mode has already clipped the painter
        # to one side of the divider, and Qt's default ReplaceClip would throw
        # that away and let the second image paint over the first. The clip is
        # also what keeps playback affordable — the blitter only touches the
        # destination pixels inside it, so a half-covered swipe pane costs half.
        painter.setClipRect(viewport, Qt.ClipOperation.IntersectClip)
        painter.drawImage(target, image, source)
        painter.restore()

    def _paint_split(self, painter: QPainter) -> None:
        viewports = self._viewports()
        self._paint_pane(painter, viewports[0], self._before)
        if len(viewports) > 1:
            self._paint_pane(painter, viewports[1], self._after)
            self._draw_chip(
                painter, self._picture_rect(viewports[1]), "After", left=True
            )
        self._draw_chip(painter, self._picture_rect(viewports[0]), "Before", left=True)

    def _paint_toggle(self, painter: QPainter, rect: QRect) -> None:
        image = self._before if self._toggled else self._after
        self._paint_pane(painter, rect, image)
        self._draw_chip(
            painter,
            self._picture_rect(rect),
            "Before" if self._toggled else "After",
            left=True,
        )

    def _paint_swipe(self, painter: QPainter, rect: QRect) -> None:
        split_x = rect.x() + int(round(rect.width() * self._divider))

        painter.save()
        painter.setClipRect(QRect(rect.x(), rect.y(), split_x - rect.x(), rect.height()))
        self._paint_pane(painter, rect, self._before)
        painter.restore()

        painter.save()
        painter.setClipRect(
            QRect(split_x, rect.y(), rect.right() - split_x + 1, rect.height())
        )
        self._paint_pane(painter, rect, self._after)
        painter.restore()

        # A bare 1px line disappears over a bright frame and a bare white line
        # disappears over a blown-out one, so the line is flanked by a dark
        # shadow. Legible over anything.
        painter.setPen(QPen(QColor(0, 0, 0, 120), 1))
        painter.drawLine(split_x - 1, rect.top(), split_x - 1, rect.bottom())
        painter.drawLine(split_x + 1, rect.top(), split_x + 1, rect.bottom())
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1))
        painter.drawLine(split_x, rect.top(), split_x, rect.bottom())

        # Grab handle, so the divider reads as draggable.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(0, 0, 0, 120), 1))
        painter.setBrush(QColor(250, 250, 250, 230))
        painter.drawEllipse(QPointF(split_x, rect.center().y()), 6.0, 6.0)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        picture = self._picture_rect(rect)
        self._draw_chip(painter, picture, "Before", left=True, limit=split_x)
        self._draw_chip(painter, picture, "After", left=False, limit=split_x)

    def _draw_chip(
        self,
        painter: QPainter,
        rect: QRectF,
        text: str,
        *,
        left: bool,
        limit: int | None = None,
    ) -> None:
        """Caption on a semi-opaque rounded chip, legible over any picture."""
        metrics = QFontMetricsF(painter.font())
        pad_x, pad_y = 8.0, 3.0
        width = metrics.horizontalAdvance(text) + 2 * pad_x
        height = metrics.height() + 2 * pad_y
        margin = 10.0

        if left:
            x = rect.x() + margin
        else:
            x = rect.right() - margin - width
        y = rect.y() + margin

        if limit is not None:
            # Do not let a caption spill across the divider onto the other half.
            if left and x + width > limit - 4:
                return
            if not left and x < limit + 4:
                return

        box = QRectF(x, y, width, height)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1))
        painter.setBrush(QColor(10, 10, 10, 175))
        painter.drawRoundedRect(box, RADIUS, RADIUS)
        painter.setPen(QColor(250, 250, 250))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def _update_caption(self) -> None:
        # Called once per frame during playback. Measured at 4.1 us with the
        # size and zoom unchanged, because QLabel.setText already short-circuits
        # an identical string and never reaches the layout. Guarding it here as
        # well was tried and saved 0.8 us of a 16.7 ms frame, so it was removed.
        size = self.image_size()
        if size is None:
            self.caption.setText("no image")
            return
        percent = self._scale * 100.0
        shown = f"{percent:.0f}%" if percent >= 10 else f"{percent:.1f}%"
        self.caption.setText(f"{size.width()} × {size.height()}  ·  {shown}")

    def _apply_cursor(self) -> None:
        if self._mode == "swipe" and self.has_pair():
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.unsetCursor()

    # ---------------------------------------------------------------- events

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()
        else:
            self._clamp_offset()
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Plain wheel and Ctrl+wheel both zoom; there is nothing to scroll here,
        # and zooming about the cursor is the whole point.
        delta = event.angleDelta().y()
        if not delta or not self.has_pair():
            event.ignore()
            return
        steps = delta / 120.0
        self.zoom_at(self._scale * (1.25 ** steps), event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        pos = event.position()
        if not self._canvas_rect().contains(int(pos.x()), int(pos.y())):
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_anchor = pos
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == "swipe":
                self._dragging_divider = True
                self._move_divider(pos)
            else:
                self._panning = True
                self._pan_anchor = pos
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        pos = event.position()
        if self._dragging_divider:
            self._move_divider(pos)
            event.accept()
            return
        if self._panning:
            self._offset += QPointF(
                pos.x() - self._pan_anchor.x(),
                pos.y() - self._pan_anchor.y(),
            )
            self._pan_anchor = pos
            self._fit_mode = False
            self._clamp_offset()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._dragging_divider = False
        self._panning = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if (
            event.key() == Qt.Key.Key_Space
            and self._mode == "toggle"
            and not event.isAutoRepeat()
        ):
            self._toggled = True
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if (
            event.key() == Qt.Key.Key_Space
            and self._mode == "toggle"
            and not event.isAutoRepeat()
        ):
            self._toggled = False
            self.update()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _move_divider(self, pos: QPointF) -> None:
        rect = self._canvas_rect()
        if rect.isEmpty():
            return
        self.set_divider((pos.x() - rect.x()) / rect.width())

    def _segment_clicked(self, mode: str) -> None:
        # Clicking the already-active Toggle segment flips the image, which is
        # the mouse equivalent of holding space.
        if mode == "toggle" and self._mode == "toggle":
            self._toggled = not self._toggled
            self.update()
            return
        self.set_mode(mode)
