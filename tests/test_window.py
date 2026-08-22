"""Offscreen tests for the Qt window.

The window holds no policy, so these only check that it constructs, that
controls map onto a RenderRequest correctly, and that guard rails fire.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from enhancer.window import MainWindow, PREVIEW_SECONDS  # noqa: E402


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_window_constructs_with_a_title(window):
    assert window.windowTitle() == "Enhancer"


def test_starts_idle_with_cancel_disabled(window):
    assert not window.cancel_button.isEnabled()
    assert window.render_button.isEnabled()


def test_defaults_match_the_engine_defaults(window):
    assert window.degrain.value() == 25
    assert window.detail.value() == 25
    assert window.regrain.value() == 60
    assert window.deblock.value() == 0


def test_frame_rate_is_off_by_default(window):
    assert window.fps_mode.currentText() == "Off"


def test_render_without_a_source_is_refused(window, monkeypatch):
    warned = []
    monkeypatch.setattr(
        "enhancer.window.QMessageBox.warning",
        lambda *a, **k: warned.append(a[1]),
    )
    assert window._build_request(preview=False) is None
    assert warned


def test_request_is_built_from_the_controls(window, synthetic_clip, tmp_path):
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.degrain.setValue(40)
    window.regrain.setValue(80)

    req = window._build_request(preview=False)
    assert req is not None
    assert req.degrain == pytest.approx(0.40)
    assert req.regrain == pytest.approx(0.80)
    assert req.target_fps is None


def test_preview_request_covers_the_preview_duration(window, synthetic_clip, tmp_path):
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))

    req = window._build_request(preview=True)
    assert req.is_preview
    assert req.preview_frames == int(25 * PREVIEW_SECONDS)


def test_target_fps_mode_sets_the_target(window, synthetic_clip, tmp_path):
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.fps_mode.setCurrentText("Target FPS")
    window.fps_target.setCurrentText("60")

    assert window._build_request(preview=False).target_fps == 60.0


def test_multiplier_mode_multiplies_the_source_rate(window, synthetic_clip, tmp_path):
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.fps_mode.setCurrentText("Multiplier")
    window.fps_multiplier.setValue(2.0)

    assert window._build_request(preview=False).target_fps == pytest.approx(50.0)


def test_no_restore_checkbox_zeroes_the_sliders(window, synthetic_clip, tmp_path):
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.no_restore.setChecked(True)
    window.regrain.setValue(90)

    req = window._build_request(preview=False)
    assert req.regrain == 0.0 and req.degrain == 0.0


def test_loading_a_source_fills_in_the_analysis(window, synthetic_clip):
    window._load_source(synthetic_clip)
    text = window.analysis.text()
    assert "320x240" in text
    assert "Scan" in text


def test_loading_a_source_proposes_an_output_path(window, synthetic_clip):
    window._load_source(synthetic_clip)
    assert window.output_label.text().endswith("_enhanced.mkv")


def test_unreadable_source_reports_instead_of_raising(window, tmp_path):
    bad = tmp_path / "not_a_video.mkv"
    bad.write_bytes(b"garbage")
    window._load_source(bad)
    assert "Could not analyse" in window.analysis.text()


def test_progress_updates_the_bar_and_status(window):
    window._started_at = 0.0
    window._on_progress(50, 200)
    assert window.bar.value() == 50
    assert window.bar.maximum() == 200
    assert "50/200" in window.status.text()


# --- still images -----------------------------------------------------------


@pytest.fixture
def png(tmp_path):
    import numpy as np
    from PIL import Image

    p = tmp_path / "still.png"
    Image.fromarray(np.full((40, 60, 3), 128, dtype=np.uint8)).save(p)
    return p


def test_loading_an_image_reports_it_as_a_still(window, png):
    window._load_source(png)
    assert "Still image" in window.analysis.text()
    assert "60x40" in window.analysis.text()


def test_loading_an_image_disables_frame_rate_conversion(window, png):
    window._load_source(png)
    assert not window.fps_mode.isEnabled()
    assert not window.preview_button.isEnabled()


def test_loading_a_video_after_an_image_re_enables_the_controls(window, png, synthetic_clip):
    window._load_source(png)
    window._load_source(synthetic_clip)
    assert window.fps_mode.isEnabled()
    assert window.preview_button.isEnabled()


def test_image_output_keeps_the_source_extension(window, png):
    window._load_source(png)
    assert window.output_label.text().endswith("_enhanced.png")


def test_image_request_has_no_target_fps_or_preview(window, png, tmp_path):
    window._load_source(png)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    req = window._build_request(preview=False)
    assert req is not None
    assert req.target_fps is None
    assert req.preview_frames is None


# --- help and layout --------------------------------------------------------


def test_window_opens_landscape_not_portrait(window):
    assert window.width() > window.height(), "the window must be wider than it is tall"


def test_window_is_resizable_and_has_no_fixed_size(window):
    assert window.minimumWidth() < window.width()
    assert window.maximumWidth() > 10000, "a maximum would stop it growing"


def test_the_picture_sits_beside_the_settings(window):
    """Viewer on the left, settings on the right, rebalanceable."""
    assert window.columns.count() == 2
    assert window.columns.widget(0) is window.view.parent()


def test_columns_cannot_be_collapsed_to_nothing(window):
    assert not window.columns.childrenCollapsible()


def test_progress_can_be_resized_but_not_collapsed(window):
    """Fixed, it took a third of the window to show one bar and an empty box."""
    assert window.rows.count() == 2
    assert not window.rows.childrenCollapsible()


def test_every_slider_carries_its_explanation(window):
    for slider in (window.degrain, window.detail, window.regrain, window.deblock):
        assert len(slider.toolTip()) > 60


def test_action_buttons_carry_explanations(window):
    for button in (window.preview_button, window.render_button, window.cancel_button):
        assert len(button.toolTip()) > 60


def test_every_model_entry_explains_what_it_suits(window):
    from PySide6.QtCore import Qt

    for i in range(window.model_combo.count()):
        note = window.model_combo.itemData(i, Qt.ToolTipRole)
        assert note, f"model entry {i} has no tooltip"
        assert len(note) > 40


def test_frame_rate_mode_entries_are_each_explained(window):
    from PySide6.QtCore import Qt

    for i in range(window.fps_mode.count()):
        assert window.fps_mode.itemData(i, Qt.ToolTipRole)


def test_frame_rate_presets_are_each_explained(window):
    from PySide6.QtCore import Qt

    for i in range(window.fps_target.count()):
        note = window.fps_target.itemData(i, Qt.ToolTipRole)
        assert note and window.fps_target.itemText(i) in note


def test_help_button_carries_the_matching_text(window):
    from enhancer.help_text import HELP
    from enhancer.window import help_button

    assert help_button("degrain").toolTip() == HELP["degrain"]


def test_help_button_for_an_unknown_key_does_not_crash(window):
    from enhancer.window import help_button

    assert help_button("no_such_control").toolTip()


def test_model_note_appears_under_the_dropdown(window):
    if window.model_combo.currentData():
        assert window.model_note.text()


def test_guide_covers_every_control(window):
    from enhancer.help_text import HELP

    html = window._guide_html()
    for key, text in HELP.items():
        first = text.split("\n")[0][:40]
        assert first in html, f"guide is missing the entry for {key!r}"


def test_guide_lists_every_catalogue_model(window):
    from enhancer.help_text import MODEL_NOTES

    html = window._guide_html()
    for name in MODEL_NOTES:
        assert name in html


def test_guide_leads_with_problem_fixes(window):
    html = window._guide_html()
    assert "plastic" in html
    assert "blocky" in html
    assert "too slow" in html


def test_guide_button_exists(window):
    assert window.guide_button.isEnabled()


# --- queue and memory control ----------------------------------------------


def test_queue_starts_empty(window):
    assert window.queue_view.rowCount() == 0
    assert len(window.queue) == 0


def test_queue_has_start_stop_remove_clear(window):
    for name in ("btn_queue_start", "btn_queue_stop",
                 "btn_queue_remove", "btn_queue_clear"):
        assert hasattr(window, name), name


def test_queue_buttons_all_explain_themselves(window):
    for name in ("btn_queue_start", "btn_queue_stop",
                 "btn_queue_remove", "btn_queue_clear"):
        assert len(getattr(window, name).toolTip()) > 30


def test_stop_and_remove_are_disabled_with_an_empty_queue(window):
    window._refresh_queue()
    assert not window.btn_queue_stop.isEnabled()
    assert not window.btn_queue_remove.isEnabled()


def test_queued_jobs_appear_as_rows(window, synthetic_clip, tmp_path):
    from enhancer.requests import RenderRequest

    for name in ("a.mkv", "b.mkv"):
        window.queue.add(RenderRequest(
            model=tmp_path / "m.pth", source=tmp_path / name,
            output=tmp_path / "out.mkv"))
    window._refresh_queue()
    assert window.queue_view.rowCount() == 2
    assert window.queue_view.item(0, 0).text() == "a.mkv"
    assert window.queue_view.item(0, 1).text() == "Waiting"


def test_clear_finished_removes_only_finished_rows(window, tmp_path):
    from enhancer.requests import RenderRequest

    def add(name):
        return window.queue.add(RenderRequest(
            model=tmp_path / "m.pth", source=tmp_path / name,
            output=tmp_path / "out.mkv"))

    waiting, done = add("wait.mkv"), add("done.mkv")
    window.queue.finish(done)
    window._queue_clear()
    assert window.queue_view.rowCount() == 1
    assert window.queue_view.item(0, 0).text() == "wait.mkv"


def test_memory_cap_offers_automatic_and_explicit_limits(window):
    labels = [window.vram_budget.itemText(i) for i in range(window.vram_budget.count())]
    assert labels[0] == "Automatic"
    assert len(labels) > 2


def test_automatic_memory_cap_means_no_budget(window):
    window.vram_budget.setCurrentIndex(0)
    assert window._selected_vram_budget() is None


def test_explicit_memory_cap_converts_to_bytes(window):
    """The ladder is built from the detected card, so read what it offers."""
    window.vram_budget.setCurrentIndex(1)
    megabytes = window.vram_budget.currentData()

    assert megabytes is not None, "the second entry should be a real cap"
    assert window._selected_vram_budget() == megabytes * 1024 ** 2


def test_memory_cap_reaches_the_request(window, synthetic_clip, tmp_path):
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.vram_budget.setCurrentIndex(1)
    megabytes = window.vram_budget.currentData()
    assert window._build_request(preview=False).vram_budget == megabytes * 1024 ** 2


def test_every_memory_option_is_explained(window):
    from PySide6.QtCore import Qt

    for i in range(window.vram_budget.count()):
        assert window.vram_budget.itemData(i, Qt.ToolTipRole)


# --- theme and alignment ----------------------------------------------------


def test_no_inline_styles_remain_in_the_window():
    """Styling belongs in theme.py, or it drifts control by control."""
    import pathlib

    source = pathlib.Path("src/enhancer/window.py").read_text(encoding="utf-8")
    assert "setStyleSheet" not in source


def test_form_labels_are_all_the_same_width(window):
    from enhancer.window import field_label

    widths = {field_label(t).maximumWidth() for t in
              ("Weights", "Degrain", "Cut sensitivity", "Graphics memory")}
    assert len(widths) == 1, "field columns would not line up"


def test_label_column_sizes_to_the_font_up_to_the_cap(window):
    """Two competing failures, so the rule is the smaller of the two.

    Hardcoding a width clipped labels under a wide font. Measuring without a
    cap grew the column until the two settings columns no longer fitted the
    window and the right one was clipped instead. A clipped label keeps its
    full text on hover.
    """
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QApplication

    from enhancer.window import FIELD_LABELS, LABEL_WIDTH_MAX, field_label, label_width

    metrics = QFontMetrics(QApplication.font())
    widest = max(metrics.horizontalAdvance(t) for t in FIELD_LABELS)
    assert label_width() == min(widest + 14, LABEL_WIDTH_MAX)
    assert field_label("Detail retention").toolTip() == "Detail retention"


def test_render_is_the_primary_action(window):
    assert window.render_button.objectName() == "primary"
    assert window.preview_button.objectName() != "primary"


def test_queue_buttons_are_equal_width(window):
    widths = {getattr(window, f"btn_queue_{k}").minimumWidth()
              for k in ("start", "stop", "remove", "clear")}
    assert len(widths) == 1


def test_action_buttons_are_equal_width(window):
    widths = {b.minimumWidth() for b in
              (window.preview_button, window.render_button, window.cancel_button)}
    assert len(widths) == 1


def test_bring_to_front_does_not_raise(window):
    window.bring_to_front()


# --- density and fit --------------------------------------------------------


def test_settings_scroll_rather_than_crush(window):
    from PySide6.QtCore import Qt

    """Without scrolling, groups lose their padding, then overlap."""
    window.resize(920, 580)
    assert window.scroll.widgetResizable()
    assert window.scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


def test_columns_fit_inside_the_window(window):
    """Minimum widths that together exceed the window clip the right column."""
    for width in (1280, 1100, 960):
        window.resize(width, 800)
        window.show()
        total = sum(window.columns.sizes())
        assert total <= width, f"columns need {total}px in a {width}px window"


def test_columns_can_compress(window):
    left, right = window.columns.widget(0), window.columns.widget(1)
    assert left.minimumWidth() < 400
    assert right.minimumWidth() < 400


def test_label_column_is_capped(window):
    """Measuring alone let the column grow until the layout no longer fitted."""
    from enhancer.window import LABEL_WIDTH_MAX, label_width

    assert label_width() <= LABEL_WIDTH_MAX


def test_the_log_pane_cannot_swallow_the_window(window):
    assert window.log_view.maximumHeight() < 200


def test_group_titles_have_clear_air_above_them(window):
    """Titles sitting on the border above made the groups read as one block."""
    from enhancer import theme

    assert theme.GROUP_TITLE_SPACE >= theme.GAP_WIDE


# --- before and after -------------------------------------------------------


def test_every_settings_group_is_reachable_as_a_tab(window):
    """Six groups, six tabs, one group each.

    Stacking Source above the tabs squeezed the pages to about a hundred and
    fifty pixels, which hid Degrain, Detail retention and Re-grain behind a
    scrollbar — the three controls that decide whether faces come out waxy.
    """
    names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert names == ["Source", "Model", "Texture", "Motion", "Performance", "Queue"]


def test_the_forecast_is_not_hidden_behind_a_scrollbar(window):
    """It exists to be read immediately before pressing Render."""
    parent = window.forecast_headline
    while parent is not None and parent is not window.scroll:
        parent = parent.parent()
    assert parent is not window.scroll, "the forecast scrolled out of sight"


def test_compare_is_refused_until_a_source_is_loaded(window):
    assert not window.compare_button.isEnabled()
    assert not window.compare_time.isEnabled()


def test_the_frame_picker_spans_the_clip(window, monkeypatch, tmp_path):
    """Offering a timestamp past the end would decode nothing."""
    from enhancer.video_io import SourceProfile

    src = tmp_path / "clip.mkv"
    src.write_bytes(b"")
    profile = SourceProfile(
        path=src, width=320, height=240, fps=25.0, frame_count=250,
        pix_fmt="yuv420p", sar="1:1", interlaced=False, field_order="tff",
        color_primaries="", color_transfer="", color_space="", duration=10.0,
    )
    monkeypatch.setattr(SourceProfile, "probe", staticmethod(lambda p: profile))
    monkeypatch.setattr("enhancer.window.probe_scan", lambda p: _flat_scan())
    monkeypatch.setattr("enhancer.window.classify_scan", lambda a: _progressive())
    monkeypatch.setattr("enhancer.window.estimate_grain", lambda f: 1.0)
    monkeypatch.setattr("enhancer.window.estimate_blockiness", lambda f: 1.0)
    monkeypatch.setattr(
        "enhancer.window.Decoder",
        lambda *a, **k: type("D", (), {"frames": lambda self: iter([_frame()])})(),
    )
    window._load_source(src)

    assert window.compare_button.isEnabled()
    assert window.compare_time.isEnabled()
    assert window.compare_time.maximum() == pytest.approx(9.9)


def test_a_still_has_no_timeline_to_pick_from(window, tmp_path):
    from PIL import Image

    src = tmp_path / "still.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(src)
    window._load_source(src)

    assert window.compare_button.isEnabled(), "a still can still be compared"
    assert not window.compare_time.isEnabled()
    assert window.compare_time.maximum() == 0.0


def test_the_theme_toggle_flips_and_is_remembered(window, monkeypatch):
    from enhancer import theme

    saved = []
    monkeypatch.setattr(theme, "save_mode", lambda m: saved.append(m))

    before = window.theme_button.text()
    window._toggle_theme()
    after = window.theme_button.text()

    assert before != after, "the toggle did not change what it offers"
    assert window.theme_mode in (theme.Mode.LIGHT, theme.Mode.DARK)
    assert saved and saved[-1] is window.theme_mode


def _frame():
    import numpy as np

    return np.zeros((240, 320, 3), dtype=np.uint8)


def _flat_scan():
    from enhancer.analyze import FieldAnalysis

    return FieldAnalysis(tff=0, bff=0, progressive=100, undetermined=0,
                         repeated_top=0, repeated_bottom=0)


def _progressive():
    from enhancer.analyze import ScanType

    return ScanType.PROGRESSIVE


def test_closing_mid_comparison_does_not_abort_the_process(window):
    """A QThread deleted while running takes the whole application down."""
    from PySide6.QtCore import QThread

    thread = QThread()
    thread.start()
    window.compare_thread = thread
    window.close()

    assert window.compare_thread is None
    assert not thread.isRunning()


def test_browse_offers_images_as_well_as_video(window):
    """The dialog listed video only, so a still could not be opened at all."""
    from enhancer.window import source_filter

    text = source_filter()
    for pattern in ("*.png", "*.jpg", "*.webp", "*.mp4", "*.mkv"):
        assert pattern in text, f"Browse cannot open {pattern}"
    assert text.startswith("All supported (")


def test_one_list_decides_what_can_be_opened(window):
    """Three lists had drifted: dialog, drag-and-drop and the image engine."""
    from enhancer.images import IMAGE_SUFFIXES
    from enhancer.window import MEDIA_SUFFIXES, VIDEO_SUFFIXES

    assert IMAGE_SUFFIXES <= MEDIA_SUFFIXES, "drag-and-drop rejects supported stills"
    assert MEDIA_SUFFIXES == VIDEO_SUFFIXES | IMAGE_SUFFIXES


# --- playback ---------------------------------------------------------------


def test_playback_is_refused_until_a_result_is_attached(window):
    assert not window.play_button.isEnabled()
    assert not window.position.isEnabled()
    assert window.time_label.text() == "--:-- / --:--"


def test_the_clock_grows_an_hours_field_only_when_needed():
    from enhancer.window import _clock

    assert _clock(0) == "0:00"
    assert _clock(62.7) == "1:02"
    assert _clock(3600) == "1:00:00"
    assert _clock(-5) == "0:00"


def test_a_slow_decode_drops_a_frame_rather_than_queueing_it(window):
    """Queueing behind a slow decode puts playback further behind every tick."""
    asked = []
    window.request_frame.connect(lambda: asked.append(1))

    window._frame_in_flight = False
    window._tick()
    window._tick()
    window._tick()

    assert len(asked) == 1, "requests piled up behind an unanswered one"


def test_reaching_the_end_stops_unless_looping(window):
    window.loop_check.setChecked(False)
    window.play_worker = object()
    window.play_timer.start(1000)
    window._on_playback_ended(window._play_token)
    assert not window.play_timer.isActive()
    assert window.play_button.text() == "Play"


def test_looping_rewinds_instead_of_stopping(window):
    sought = []
    window.request_seek.connect(sought.append)
    window.loop_check.setChecked(True)
    window.play_worker = object()
    window.play_timer.start(1000)
    window._on_playback_ended(window._play_token)

    assert sought == [0.0]
    assert window.play_timer.isActive(), "looping stopped playback"
    window._pause()


def test_closing_mid_playback_stops_the_stream(window):
    from PySide6.QtCore import QThread

    thread = QThread()
    thread.start()
    window.play_thread = thread
    window.close()

    assert window.play_thread is None
    assert not thread.isRunning()


def test_a_new_source_detaches_the_old_result(window, tmp_path):
    """Otherwise Play streams the previous film under the new film's name."""
    from PIL import Image

    window.comparison = tmp_path / "old_render.mkv"
    window.compare_with_button.setText("old_render.mkv")
    window.play_button.setEnabled(True)

    src = tmp_path / "next.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(src)
    window._load_source(src)

    assert window.comparison is None
    assert not window.play_button.isEnabled()
    assert window.compare_with_button.text() == "Compare with..."


def test_comparing_is_blocked_while_a_render_holds_the_card(window, tmp_path):
    """A second copy of the model on a 6 GB card silently slows the render."""
    from PIL import Image

    src = tmp_path / "s.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(src)
    window._load_source(src)
    assert window.compare_button.isEnabled()

    window._set_running(True)
    assert not window.compare_button.isEnabled()
    assert "graphics card" in window.compare_button.toolTip()

    window._set_running(False)
    assert window.compare_button.isEnabled()


def test_playback_decodes_only_what_the_pane_can_show(window):
    """A 4K frame is 24.9 MB; sixty a second is 1.5 GB/s and manages about ten.

    Fitted into a small pane, most of those rows are discarded before they are
    seen, so they are never decoded. Measured: 9.8 to 74.8 frames a second.
    """
    window.view.resize(800, 472)
    wanted = window._needed_decode_height()

    assert wanted == 472, "should decode exactly what the pane shows"
    assert wanted < 2160, "playing a fitted 4K clip at full resolution"


def test_the_played_height_ignores_the_zoom(window):
    """A magnified proxy would make the zoom readout mean something false.

    At "100%" the user is entitled to one output pixel per screen pixel. A
    stream decoded small and blown up cannot offer that, so playback stays a
    fitted preview and Compare this frame remains the full-resolution tool.
    """
    import numpy as np

    window.view.resize(800, 472)
    frame = np.zeros((600, 900, 3), dtype=np.uint8)
    window.view.set_pair(frame, frame.copy())

    window.view.fit()
    fitted = window._needed_decode_height()
    window.view.set_zoom(4.0)

    assert window._needed_decode_height() == fitted


def test_a_tiny_pane_still_asks_for_a_usable_picture(window):
    """A collapsed pane must not ask ffmpeg for a two-pixel-tall stream."""
    window.view.resize(800, 10)
    assert window._needed_decode_height() >= 240


# --- states the audit found ------------------------------------------------


def test_a_result_from_an_abandoned_player_is_ignored(window):
    """Probing a fresh 4K file is slow enough to finish after the user moves on.

    Acting on it enabled Play and filled the seek bar for a player that no
    longer existed: the timer then ran forever over a blank picture.
    """
    window.play_worker = None  # as _stop_playback leaves it
    window.comparison = None

    window._on_playback_opened(window._play_token, 60.0, 240, 0.0, 4.0)

    assert not window.play_button.isEnabled()
    assert window.position.maximum() == 0


def test_a_finished_render_is_paired_with_its_own_source(window, tmp_path, monkeypatch):
    """A queue holding two sources otherwise plays one film against another."""
    from enhancer.queue import Task
    from enhancer.requests import RenderRequest

    other = tmp_path / "other.mkv"
    other.write_bytes(b"")
    result = tmp_path / "done.mkv"
    result.write_bytes(b"")

    attached = []
    monkeypatch.setattr(window, "_load_comparison", attached.append)
    monkeypatch.setattr(window, "_start_next", lambda: None)

    window.source = tmp_path / "currently_open.mkv"
    window.active_task = Task(request=RenderRequest(
        model=tmp_path / "m.pth", source=other, output=result,
    ))
    window._on_finished(str(result))

    assert attached == [], "attached a render belonging to a different source"


def test_play_at_the_end_rewinds_instead_of_stalling(window):
    """Otherwise the button flickers Pause then Play and nothing moves."""
    sought = []
    window.request_seek.connect(sought.append)
    window._at_end = True
    window._play_fps = 60.0

    window._toggle_play()

    assert sought == [0.0]
    assert not window._at_end
    window._pause()


def test_compare_with_is_dead_until_a_source_is_open(window, tmp_path):
    """It opened a file dialog, took a file, and silently did nothing."""
    from PIL import Image

    assert not window.compare_with_button.isEnabled()
    src = tmp_path / "s.png"
    Image.new("RGB", (16, 16), (7, 7, 7)).save(src)
    window._load_source(src)
    assert window.compare_with_button.isEnabled()


def test_a_frame_from_a_superseded_player_never_reaches_the_screen(window):
    """Two attaches in quick succession: the first must not paint over the second."""
    import numpy as np

    class Pair:
        before = after = np.zeros((8, 8, 3), dtype=np.uint8)
        index, seconds = 5, 0.5

    window.play_worker = object()          # something is attached
    window._play_token = 7
    window._frame_in_flight = True

    window._on_playback_frame(6, Pair())   # from the previous player

    assert not window.view.has_pair(), "a stale frame was painted"
    assert window._frame_in_flight, "a stale frame cleared the in-flight guard"

    window._on_playback_frame(7, Pair())   # from the current one
    assert window.view.has_pair()
    assert not window._frame_in_flight


# --- built for the machine it is running on ---------------------------------


def test_the_memory_ladder_is_built_from_the_detected_card():
    """A fixed 2/3/4/5 GB ladder described exactly one card.

    On a 24 GB desktop card every option capped far below what was there; on a
    4 GB laptop card every option but the first asked for more than existed.
    """
    from enhancer.window import vram_choices

    big = [mb for _label, mb in vram_choices(24 * 1024 ** 3, 64 * 1024 ** 3)]
    small = [mb for _label, mb in vram_choices(4 * 1024 ** 3, 16 * 1024 ** 3)]

    assert big[0] is None and small[0] is None, "Automatic must stay first"
    assert max(m for m in big[1:-1]) == 24 * 1024
    assert max(m for m in small[1:-1]) == 4 * 1024
    assert all(m <= 4 * 1024 for m in small[1:-1]), "offered more than the card has"


def test_a_machine_with_no_card_is_not_offered_a_ladder():
    from enhancer.window import vram_choices

    labels = [label for label, _mb in vram_choices(0, 16 * 1024 ** 3)]
    assert labels[0] == "Automatic"
    assert len(labels) == 2, "nothing to divide up without a card"


def test_the_window_knows_what_it_is_running_on(window):
    assert window.hardware.accelerator in ("cuda", "mps", "xpu", "cpu")
    assert window.hardware.cpu_cores >= 0


def test_the_system_dialog_reports_every_requirement(window):
    html = window._requirements_html()
    for name in ("ffmpeg", "Models", "Disk space"):
        assert name in html


# --- what the safety review demonstrated ------------------------------------


def test_a_settings_mismatch_releases_the_queue(window, tmp_path, monkeypatch):
    """It used to jam the queue permanently, recoverable only by restarting.

    _teardown stopped the thread but left the task marked running, so
    queue.running never cleared. Every later Render queued behind a job with
    nothing behind it, and the phantom row could not be removed.
    """
    from PySide6.QtWidgets import QMessageBox

    from enhancer.queue import Task
    from enhancer.requests import RenderRequest

    request = RenderRequest(
        model=tmp_path / "m.pth", source=tmp_path / "s.mkv",
        output=tmp_path / "o.mkv",
    )
    task = window.queue.add(request)
    window.queue.start(task)
    window.active_task = task
    window.worker = None
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)

    window._on_settings_changed()

    assert window.queue.running is None, "the queue is still jammed"
    assert window.active_task is None
    assert task.can_remove, "the stalled row cannot be cleared"


def test_a_typed_frame_rate_that_is_not_a_number_is_refused(window, synthetic_clip,
                                                            tmp_path, monkeypatch):
    """The parse sat outside the try, so Render raised out of the slot.

    A packaged build has no console, so the button simply did nothing.
    """
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.fps_mode.setCurrentText("Target FPS")
    window.fps_target.setCurrentText("60 fps")

    assert window._build_request(preview=False) is None
    assert warned, "refused it silently"


def test_an_absurd_frame_rate_is_refused(window, synthetic_clip, tmp_path, monkeypatch):
    """100000 fps on a two-second clip grinds for hours before anyone notices."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.fps_mode.setCurrentText("Target FPS")
    window.fps_target.setCurrentText("100000")

    assert window._build_request(preview=False) is None


def test_a_processor_only_install_is_forecast_as_one(window, monkeypatch):
    """`pip install torch` gets the processor-only wheel by default.

    Those users were shown an estimate derived from a graphics card they do
    not have: seventeen hours for something that would really take weeks.
    """
    import dataclasses

    window.cpu.setChecked(False)
    monkeypatch.setattr(
        window, "hardware", dataclasses.replace(window.hardware, accelerator="cpu")
    )
    assert window._running_on_cpu()

    monkeypatch.setattr(
        window, "hardware", dataclasses.replace(window.hardware, accelerator="cuda")
    )
    assert not window._running_on_cpu()
    window.cpu.setChecked(True)
    assert window._running_on_cpu(), "the checkbox must still be honoured"


def test_models_are_found_beside_the_application_not_the_shell(tmp_path, monkeypatch):
    """Run the packaged build from anywhere else and it reported no models."""
    from enhancer import paths

    monkeypatch.chdir(tmp_path)
    assert paths.custom_models_dir().is_absolute()
    assert "models" in str(paths.custom_models_dir())
    assert paths.app_dir() != tmp_path


def test_a_models_folder_in_the_working_directory_still_wins(tmp_path, monkeypatch):
    """So a project can keep its own models beside it."""
    from enhancer import paths

    (tmp_path / "models" / "custom").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert paths.custom_models_dir() == tmp_path / "models" / "custom"
