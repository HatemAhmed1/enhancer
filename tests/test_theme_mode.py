"""Light or dark is the user's call.

The default defers to Windows, an explicit choice overrides it, and the choice
survives a restart. These tests keep the settings out of the real registry so
running them never touches the user's own preference.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from enhancer import theme  # noqa: E402


@pytest.fixture
def store(tmp_path, qapp, monkeypatch):
    """A throwaway ini file standing in for the user's real settings.

    QSettings.setDefaultFormat is not enough: PySide's two-string constructor
    still resolves to the native store, so a test that relied on it would write
    into the user's actual registry. Replacing the seam is the honest way.
    """
    from PySide6.QtCore import QSettings

    path = str(tmp_path / "enhancer.ini")

    def fresh():
        return QSettings(path, QSettings.Format.IniFormat)

    monkeypatch.setattr(theme, "_settings", fresh)
    return fresh


def test_the_preference_lives_where_the_application_can_find_it():
    """Constructing does not write; this only checks the names never drift."""
    assert (theme._ORG, theme._APP, theme._KEY) == (
        "Enhancer", "Enhancer", "theme/mode")
    assert "Enhancer" in theme._settings().fileName()


# --- resolving a mode to a palette ------------------------------------------

@pytest.mark.parametrize("system_is_dark", [True, False])
def test_explicit_modes_ignore_the_system_setting(monkeypatch, system_is_dark):
    monkeypatch.setattr(theme, "is_dark", lambda app: system_is_dark)
    assert theme.resolve(None, theme.Mode.LIGHT) is theme.LIGHT
    assert theme.resolve(None, theme.Mode.DARK) is theme.DARK


@pytest.mark.parametrize("system_is_dark, expected",
                         [(True, theme.DARK), (False, theme.LIGHT)])
def test_system_mode_follows_is_dark(monkeypatch, system_is_dark, expected):
    monkeypatch.setattr(theme, "is_dark", lambda app: system_is_dark)
    assert theme.resolve(None, theme.Mode.SYSTEM) is expected


def test_system_is_resolved_afresh_each_time(monkeypatch):
    """The OS can change while the application is running."""
    monkeypatch.setattr(theme, "is_dark", lambda app: True)
    first = theme.resolve(None, theme.Mode.SYSTEM)
    monkeypatch.setattr(theme, "is_dark", lambda app: False)
    assert first is not theme.resolve(None, theme.Mode.SYSTEM)


# --- remembering the choice -------------------------------------------------

def test_nothing_saved_means_system(store):
    assert theme.load_mode() is theme.Mode.SYSTEM


@pytest.mark.parametrize("mode", list(theme.Mode))
def test_save_then_load_round_trips(store, mode):
    theme.save_mode(mode)
    assert theme.load_mode() is mode


def test_the_choice_survives_a_restart(store):
    """A fresh QSettings object must see it, not just the one that wrote it."""
    theme.save_mode(theme.Mode.LIGHT)
    assert store().value("theme/mode") == "light"


@pytest.mark.parametrize("junk", ["", "midnight", "42", "None", "system;"])
def test_a_junk_saved_value_falls_back_to_system(store, junk):
    settings = store()
    settings.setValue("theme/mode", junk)
    settings.sync()
    assert theme.load_mode() is theme.Mode.SYSTEM


def test_a_saved_value_is_read_forgivingly(store):
    """Case and stray whitespace are not corruption."""
    settings = store()
    settings.setValue("theme/mode", " Dark ")
    settings.sync()
    assert theme.load_mode() is theme.Mode.DARK


# --- applying ---------------------------------------------------------------

def test_apply_with_no_mode_still_works(store, qapp):
    """splash.py and window.py both call theme.apply(app) with one argument."""
    palette = theme.apply(qapp)
    assert isinstance(palette, theme.Palette)
    assert qapp.styleSheet().strip()


def test_apply_with_no_mode_does_not_write_a_preference(store, qapp):
    theme.apply(qapp)
    assert store().value("theme/mode") is None


def test_apply_with_an_explicit_mode_saves_it(store, qapp):
    palette = theme.apply(qapp, theme.Mode.DARK)
    assert palette is theme.DARK
    assert theme.load_mode() is theme.Mode.DARK
    assert qapp.styleSheet() == theme.stylesheet(theme.DARK)


def test_apply_with_no_mode_uses_the_saved_preference(store, qapp):
    theme.save_mode(theme.Mode.LIGHT)
    assert theme.apply(qapp) is theme.LIGHT
    theme.save_mode(theme.Mode.DARK)
    assert theme.apply(qapp) is theme.DARK


# --- the new widgets the viewer needs ---------------------------------------

@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_stylesheet_styles_the_card_and_segment_widgets(palette):
    css = theme.stylesheet(palette)
    assert "#card" in css
    assert "#segment" in css
    assert "#segment:checked" in css
    assert "#caption" in css
    assert palette.accent_text in css


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_focus_is_visible_rather_than_a_dotted_outline(palette):
    css = theme.stylesheet(palette)
    assert ":focus" in css
    assert f"border: 1px solid {palette.accent}" in css


# --- the settings tabs ------------------------------------------------------
#
# Left unstyled, a QTabBar inherited QWidget's colour on QWidget's background.
# In the light palette that was white on white: only the selected tab could be
# seen and the other three looked like they did not exist, which hid three
# quarters of the settings.

TAB_NAMES = ("Picture", "Motion", "Performance", "Queue")


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_stylesheet_styles_every_tab_state(palette):
    css = theme.stylesheet(palette)
    for rule in ("QTabWidget::pane", "QTabBar::tab", "QTabBar::tab:selected",
                 "QTabBar::tab:!selected", "QTabBar::tab:focus"):
        assert rule in css, f"{rule} is missing"
    assert "QTabBar::tab:!selected:hover" in css


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_an_unselected_tab_is_muted_text_and_a_selected_one_is_filled(palette):
    """Readable either way, and never the same treatment as the other."""
    tab_css = theme.stylesheet(palette).split("QTabWidget::pane")[1]
    unselected = tab_css.split("QTabBar::tab:!selected {")[1].split("}")[0]
    selected = tab_css.split("QTabBar::tab:selected {")[1].split("}")[0]

    assert f"color: {palette.text_muted}" in unselected
    assert f"background-color: {palette.surface}" in unselected
    # Filled, like QPushButton#segment:checked — not merely outlined.
    assert f"background-color: {palette.accent}" in selected
    assert f"color: {palette.accent_text}" in selected


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_the_tab_bar_really_paints_those_colours(qtbot, palette):
    """Rendered, not reasoned about: a rule that never reaches the bar is a bug.

    Fills only — this machine has no font families under the offscreen
    platform, so glyphs cannot be measured here. Legibility of the labels was
    checked by rendering under the real Windows platform plugin.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QLabel, QTabWidget

    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.setDocumentMode(True)
    for name in TAB_NAMES:
        tabs.addTab(QLabel(name), name)
    tabs.setStyleSheet(theme.stylesheet(palette))
    tabs.resize(560, 240)
    # Shown, because an unshown QTabWidget keeps its default 100px bar and
    # clips three of the four tabs away — the render would prove nothing.
    tabs.show()
    qtbot.waitExposed(tabs)

    pixmap = QPixmap(tabs.size())
    pixmap.fill(Qt.GlobalColor.transparent)
    tabs.render(pixmap)
    image = pixmap.toImage()

    bar = tabs.tabBar().height()
    seen = {image.pixelColor(x, y).name()
            for y in range(min(bar, image.height()))
            for x in range(image.width())}

    assert palette.surface in seen, "an unselected tab has no ground of its own"
    assert palette.border in seen, "an unselected tab has no border"
    assert palette.text_muted in seen, "unselected labels are not muted ink"
    # The pointer sits at the origin under the offscreen platform, which is
    # over the first tab, so the selected fill may be either the accent or its
    # hover shade. Both are the filled treatment; neither is the bare ground.
    assert seen & {palette.accent, palette.text}, "the selected tab is not filled"
