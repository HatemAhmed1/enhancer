"""Offscreen tests for the before/after comparison widget.

These check the arithmetic that keeps the two images in register — zoom about
a point, the sampling filter, the pair surviving a mode change — plus the
crash-avoidance cases: no pair, empty widget, mismatched shapes.
"""

import gc
import os
import sys
import weakref

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402

from enhancer import theme  # noqa: E402
from enhancer.viewer import CompareView  # noqa: E402


def make_pair(width=64, height=48, seed=0):
    """A sharp noise field and a blurred copy of it — a plausible before/after."""
    rng = np.random.default_rng(seed)
    before = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    after = before.astype(np.float32)
    after[1:] = (after[1:] + after[:-1]) / 2.0
    after[:, 1:] = (after[:, 1:] + after[:, :-1]) / 2.0
    return before, np.ascontiguousarray(after.astype(np.uint8))


@pytest.fixture
def view(qtbot):
    widget = CompareView()
    qtbot.addWidget(widget)
    widget.resize(300, 200)
    return widget


def render(widget):
    """Render the widget into a pixmap, as the compositor would."""
    pixmap = QPixmap(widget.size())
    pixmap.fill(Qt.GlobalColor.transparent)
    widget.render(pixmap)
    return pixmap


def test_mismatched_dimensions_are_refused(view):
    before, _ = make_pair(64, 48)
    other, _ = make_pair(32, 24)
    with pytest.raises(ValueError, match="identical dimensions"):
        view.set_pair(before, other)


def test_non_uint8_input_is_refused(view):
    before, after = make_pair()
    with pytest.raises(ValueError, match="uint8"):
        view.set_pair(before.astype(np.float32), after.astype(np.float32))


def test_paints_before_any_pair_is_set(view):
    render(view)  # must not raise


def test_paints_after_clear(view):
    before, after = make_pair()
    view.set_pair(before, after)
    view.clear()
    assert not view.has_pair()
    render(view)  # must not raise


def test_zero_size_widget_does_not_divide_by_zero(qtbot):
    widget = CompareView()
    qtbot.addWidget(widget)
    widget.resize(0, 0)
    before, after = make_pair()
    widget.set_pair(before, after)
    widget.fit()
    widget.set_zoom(2.0)
    widget.set_mode("split")
    render(widget)


def test_zoom_about_a_point_keeps_that_point_stationary(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*make_pair(400, 300))
    view.set_zoom(2.0)

    canvas = view._canvas_rect()
    point = QPointF(canvas.x() + canvas.width() * 0.6, canvas.y() + canvas.height() * 0.4)
    before_point = view.image_at(point)

    view.zoom_at(4.0, point)
    after_point = view.image_at(point)

    assert view.zoom() == pytest.approx(4.0)
    assert after_point.x() == pytest.approx(before_point.x(), abs=1e-6)
    assert after_point.y() == pytest.approx(before_point.y(), abs=1e-6)


def test_switching_modes_keeps_the_pair(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    before, after = make_pair(120, 90)
    view.set_pair(before, after)
    size = view.image_size()

    for mode in ("split", "toggle", "swipe"):
        view.set_mode(mode)
        assert view.mode() == mode
        assert view.has_pair()
        assert view.image_size() == size
        render(view)


def test_unknown_mode_is_refused(view):
    with pytest.raises(ValueError, match="unknown mode"):
        view.set_mode("crossfade")


def test_sampling_is_nearest_when_magnified_and_smooth_when_minified(view):
    view.set_pair(*make_pair(400, 300))

    view.set_zoom(2.0)
    assert view.transformation_mode() == Qt.TransformationMode.FastTransformation

    view.set_zoom(1.01)
    assert view.transformation_mode() == Qt.TransformationMode.FastTransformation

    view.set_zoom(1.0)
    assert view.transformation_mode() == Qt.TransformationMode.SmoothTransformation

    view.set_zoom(0.25)
    assert view.transformation_mode() == Qt.TransformationMode.SmoothTransformation


def test_a_real_render_is_not_blank(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    before, after = make_pair(200, 150, seed=7)
    view.set_pair(before, after)
    view.fit()

    image = render(view).toImage()
    assert image.width() == 300 and image.height() == 200

    colours = {image.pixel(x, y) for y in range(0, 200, 4) for x in range(0, 300, 4)}
    # A blank widget yields one or two colours; a rendered noise field yields
    # hundreds.
    assert len(colours) > 100


def test_both_halves_of_the_swipe_are_drawn(view, qtbot):
    """A vertical stripe pattern before, a flat field after: each half differs."""
    view.show()
    qtbot.waitExposed(view)
    height, width = 150, 200
    before = np.zeros((height, width, 3), np.uint8)
    before[:, ::2] = 255
    after = np.full((height, width, 3), 128, np.uint8)
    view.set_pair(before, after)
    view.fit()
    view.set_divider(0.5)

    image = render(view).toImage()
    canvas = view._canvas_rect()
    row = canvas.y() + canvas.height() // 2
    # Sample either side of the divider but well inside the fitted picture, so
    # the letterbox background around it is not what is being measured.
    middle = canvas.x() + canvas.width() // 2
    left = {image.pixel(x, row) for x in range(middle - 70, middle - 30)}
    right = {image.pixel(x, row) for x in range(middle + 30, middle + 70)}

    assert len(left) > 1, "before half should show the stripes"
    assert len(right) == 1, "after half should show the flat field"
    assert left != right


def test_divider_is_clamped_to_the_widget(view):
    view.set_pair(*make_pair())
    view.set_divider(-3.0)
    assert view.divider() == 0.0
    view.set_divider(9.0)
    assert view.divider() == 1.0


def test_zoom_changed_is_emitted(view, qtbot):
    view.set_pair(*make_pair())
    with qtbot.waitSignal(view.zoom_changed, timeout=1000) as blocker:
        view.set_zoom(3.0)
    assert blocker.args[0] == pytest.approx(3.0)


def test_zoom_is_clamped(view):
    view.set_pair(*make_pair())
    view.set_zoom(10_000.0)
    assert view.zoom() <= 64.0
    view.set_zoom(0.0001)
    assert view.zoom() >= 0.02


def test_toggle_flips_between_the_two_images(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*make_pair(120, 90))
    view.set_mode("toggle")
    assert not view.showing_before()

    view._segment_clicked("toggle")
    assert view.showing_before()
    view._segment_clicked("toggle")
    assert not view.showing_before()


def test_space_shows_before_while_held(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*make_pair(120, 90))
    view.set_mode("toggle")

    qtbot.keyPress(view, Qt.Key.Key_Space)
    assert view.showing_before()
    qtbot.keyRelease(view, Qt.Key.Key_Space)
    assert not view.showing_before()


def test_a_new_frame_of_the_same_size_keeps_the_zoom(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*make_pair(200, 150, seed=1))
    view.set_zoom(3.0)
    view.set_pair(*make_pair(200, 150, seed=2))
    assert view.zoom() == pytest.approx(3.0)


def test_a_new_frame_of_a_different_size_refits(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*make_pair(200, 150))
    view.set_zoom(3.0)
    view.set_pair(*make_pair(400, 300))
    assert view.zoom() < 3.0


# --- the letterbox behind the picture ---------------------------------------
#
# It used to be a translucent black wash, so the colour was whatever the
# stylesheet happened to leave underneath: invisible in the dark palette and an
# unexplained grey band in the light one. It is now one fixed neutral, read
# from theme on every repaint.

def letterband(view, qtbot):
    """Fit a wide, short pair into a taller pane and sample the band above it.

    Returns the distinct colours across the middle of that band. Toggle mode,
    because the swipe divider runs the full height of the pane and would put
    its own three pixels in the middle of the sample.
    """
    view.show()
    qtbot.waitExposed(view)
    view.set_mode("toggle")
    view.set_pair(*make_pair(240, 40, seed=5))
    view.fit()

    canvas = view._canvas_rect()
    picture = view._picture_rect(canvas)
    row = (canvas.y() + int(picture.top())) // 2
    assert row > canvas.y(), "the picture should be letterboxed, not filling the pane"

    image = render(view).toImage()
    quarter = canvas.width() // 4
    return {
        image.pixelColor(x, row).name()
        for x in range(canvas.x() + quarter, canvas.right() - quarter)
    }


def test_the_backdrop_comes_from_the_theme(view):
    assert view.backdrop_colour().name() == theme.VIEWER_BACKDROP


def test_the_backdrop_is_read_afresh_and_not_cached_at_construction(
    view, qtbot, monkeypatch
):
    """theme.apply runs while the window is open, so nothing may be bound once."""
    monkeypatch.setattr(theme, "VIEWER_BACKDROP", "#ff00ff")
    assert view.backdrop_colour().name() == "#ff00ff"
    assert letterband(view, qtbot) == {"#ff00ff"}


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_the_letterbox_is_the_same_in_both_palettes(view, qtbot, palette):
    """Whichever palette is live, the band is the constant and not the ground."""
    view.setStyleSheet(theme.stylesheet(palette))
    band = letterband(view, qtbot)
    assert band == {theme.VIEWER_BACKDROP}
    assert palette.background not in band, "the band must not be the window ground"


def test_the_placeholder_text_is_legible_on_the_backdrop():
    """Both are fixed, so the pairing can be checked once, here."""
    def luma(name):
        c = QColor(name)
        return 0.2126 * c.red() + 0.7152 * c.green() + 0.0722 * c.blue()

    assert abs(luma(theme.VIEWER_BACKDROP_TEXT) - luma(theme.VIEWER_BACKDROP)) > 60


def test_a_non_contiguous_array_is_accepted(view):
    """A sliced array has an awkward stride; it must not shear the picture."""
    before, after = make_pair(80, 60)
    view.set_pair(before[:, ::2], after[:, ::2])
    assert view.image_size().width() == 40
    render(view)


# --- playback ----------------------------------------------------------------
#
# set_frame is the same contract as set_pair, pushed 24-60 times a second. The
# widget no longer copies the pixels into a QPixmap, so two things have to hold
# per frame: the view must not move, and the previous frame's numpy buffer must
# be released — while the current one must not be, since the QImage on screen is
# only a view of it.

def moving_pair(width=200, height=150, phase=0):
    """A bar that walks across the frame, so consecutive frames really differ."""
    before = np.zeros((height, width, 3), np.uint8)
    after = np.zeros((height, width, 3), np.uint8)
    x = (phase * 17) % (width - 20)
    before[:, x:x + 20] = 255
    after[:, x:x + 20] = 128
    return before, after


def test_set_frame_preserves_zoom_pan_and_divider(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*moving_pair(phase=0))
    view.set_zoom(3.0)
    view.set_divider(0.3)

    canvas = view._canvas_rect()
    probe = QPointF(canvas.x() + canvas.width() * 0.6, canvas.y() + canvas.height() * 0.4)
    view.zoom_at(3.0, probe)          # leaves fit mode, fixes a pan
    anchor = view.image_at(probe)

    for phase in range(1, 25):
        view.set_frame(*moving_pair(phase=phase))
        assert view.zoom() == pytest.approx(3.0), f"zoom moved on frame {phase}"
        assert view.divider() == pytest.approx(0.3), f"divider moved on frame {phase}"
        here = view.image_at(probe)
        assert here.x() == pytest.approx(anchor.x(), abs=1e-6), f"panned on {phase}"
        assert here.y() == pytest.approx(anchor.y(), abs=1e-6), f"panned on {phase}"


def test_set_frame_keeps_the_mode(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*moving_pair())
    for mode in ("split", "toggle", "swipe"):
        view.set_mode(mode)
        for phase in range(5):
            view.set_frame(*moving_pair(phase=phase))
            assert view.mode() == mode
        render(view)


def test_set_frame_refuses_mismatched_dimensions(view):
    view.set_pair(*make_pair(64, 48))
    before, _ = make_pair(64, 48)
    other, _ = make_pair(32, 24)
    with pytest.raises(ValueError, match="identical dimensions"):
        view.set_frame(before, other)


def test_set_frame_refuses_non_uint8(view):
    view.set_pair(*make_pair())
    before, after = make_pair()
    with pytest.raises(ValueError, match="uint8"):
        view.set_frame(before.astype(np.float32), after.astype(np.float32))


def test_a_rejected_frame_leaves_the_previous_pair_intact(view, qtbot):
    """Validation happens before either image is installed, so nothing is torn."""
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*make_pair(120, 90))
    size = view.image_size()

    good, _ = make_pair(64, 48)
    _, bad = make_pair(32, 24)
    with pytest.raises(ValueError):
        view.set_frame(good, bad)

    assert view.image_size() == size, "a refused frame must not replace the pair"
    render(view)  # the two images must still agree, or this would paint garbage


def test_pushing_many_frames_does_not_retain_them(view, qtbot):
    """Only the pair on screen is held; every earlier frame is released.

    The QImages are views over these arrays and copy nothing, so the widget has
    to hold the current pair (drop it and Qt paints freed memory) and must hold
    nothing else (hold them all and playback leaks 50 MB a second at 4K).
    """
    view.show()
    qtbot.waitExposed(view)

    # Weak references rather than getrefcount arithmetic: the question is
    # simply whether the array is still alive once this test lets go of it.
    seen = []
    for phase in range(30):
        before, after = moving_pair(phase=phase)
        seen.append((weakref.ref(before), weakref.ref(after)))
        view.set_frame(before, after)
        del before, after
    gc.collect()

    alive = [
        phase
        for phase, (before, after) in enumerate(seen)
        if before() is not None or after() is not None
    ]
    assert alive == [29], f"only the frame on screen may be retained, got {alive}"

    # And it is the array that was handed over, not a copy of it: that is what
    # makes the retention a correctness requirement rather than a leak.
    last_before, last_after = seen[-1][0](), seen[-1][1]()
    assert view._before_buffer is last_before
    assert view._after_buffer is last_after
    assert sys.getrefcount(last_before) > 2

    # The attribute is replaced, never appended to.
    view.set_frame(*moving_pair(phase=99))
    assert view._before_buffer is not last_before
    del last_before, last_after
    gc.collect()
    assert seen[-1][0]() is None, "the superseded frame must be released"


def test_set_frame_survives_a_size_change(view, qtbot):
    """A reel boundary is the one time a refit is right, and must not crash."""
    view.show()
    qtbot.waitExposed(view)
    view.set_pair(*moving_pair(200, 150))
    view.set_zoom(3.0)
    view.set_frame(*moving_pair(400, 300))
    assert view.image_size().width() == 400
    assert view.zoom() < 3.0, "a genuinely different size has no pan to keep"
    render(view)


def test_playback_flag_round_trips(view):
    assert not view.is_playing()
    view.set_playing(True)
    assert view.is_playing()
    view.set_playing(True)      # idempotent
    assert view.is_playing()
    view.set_playing(False)
    assert not view.is_playing()


def test_clear_stops_playback(view):
    view.set_pair(*make_pair())
    view.set_playing(True)
    view.clear()
    assert not view.is_playing()
    assert not view.has_pair()


def test_the_chips_and_caption_stay_visible_while_playing(view, qtbot):
    view.show()
    qtbot.waitExposed(view)
    view.set_mode("toggle")
    view.set_pair(*moving_pair(200, 150))
    view.fit()
    view.set_playing(True)

    for phase in range(10):
        view.set_frame(*moving_pair(200, 150, phase=phase))
    assert "200" in view.caption.text() and "150" in view.caption.text()

    # The chip is drawn onto the picture, so look for its light text on the
    # dark rounded plate in the top-left of where the picture landed.
    image = render(view).toImage()
    picture = view._picture_rect(view._canvas_rect())
    chip = {
        image.pixel(x, y)
        for y in range(int(picture.y()) + 12, int(picture.y()) + 30)
        for x in range(int(picture.x()) + 12, int(picture.x()) + 70)
    }
    assert len(chip) > 2, "the After chip should still be drawn during playback"


def test_a_read_only_array_is_still_taken_without_copying(view, qtbot):
    """Held source frames arrive with write=False, and must stay zero-copy.

    The playback driver caches the enlarged 'before' across the 60% of pairs
    that share a source frame and marks it read-only so no consumer can scribble
    on it. If that made the widget copy — or raise — the whole point of dropping
    the QPixmap would be lost on the commonest frame of all.
    """
    view.show()
    qtbot.waitExposed(view)
    before, after = make_pair(200, 150, seed=11)
    before.setflags(write=False)

    view.set_pair(before, after)
    assert view._before_buffer is before, "a read-only frame must not be copied"

    for _ in range(5):
        view.set_frame(before, after)
        assert view._before_buffer is before
    render(view)


def test_a_pushed_frame_actually_reaches_the_screen(view, qtbot):
    """Consecutive frames must paint differently, or playback is a still image."""
    view.show()
    qtbot.waitExposed(view)
    view.set_mode("toggle")
    view.set_pair(*moving_pair(200, 150, phase=0))
    view.fit()

    renders = []
    for phase in range(4):
        view.set_frame(*moving_pair(200, 150, phase=phase))
        renders.append(render(view).toImage())

    for phase in range(1, 4):
        assert renders[phase] != renders[phase - 1], f"frame {phase} did not repaint"
