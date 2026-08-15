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


def test_settings_are_laid_out_in_two_columns(window):
    assert window.columns.count() == 2


def test_columns_cannot_be_collapsed_to_nothing(window):
    assert not window.columns.childrenCollapsible()


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
    window.vram_budget.setCurrentIndex(1)
    assert window._selected_vram_budget() == 2048 * 1024 ** 2


def test_memory_cap_reaches_the_request(window, synthetic_clip, tmp_path):
    window._load_source(synthetic_clip)
    window.model_combo.clear()
    window.model_combo.addItem("m.pth", str(tmp_path / "m.pth"))
    window.output_label.setText(str(tmp_path / "out.mkv"))
    window.vram_budget.setCurrentIndex(1)
    assert window._build_request(preview=False).vram_budget == 2048 * 1024 ** 2


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


def test_label_column_fits_the_longest_label(window):
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QApplication

    from enhancer.window import FIELD_LABELS, label_width

    metrics = QFontMetrics(QApplication.font())
    widest = max(metrics.horizontalAdvance(t) for t in FIELD_LABELS)
    assert label_width() > widest, "labels would be clipped"


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
