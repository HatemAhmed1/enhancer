"""Visual theme.

A restrained neutral palette in the shadcn/ui idiom: greys only, one subtle
accent, thin borders, generous but consistent spacing. No gradients, no glow,
no colour for its own sake — this is a tool, and the picture being worked on is
the only thing on screen that should draw the eye.

Colours are the shadcn "neutral" scale converted from HSL to hex.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Palette:
    background: str
    surface: str
    surface_raised: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_faint: str
    accent: str
    accent_text: str
    selection: str
    danger: str
    success: str


DARK = Palette(
    background="#0a0a0a",       # neutral-950
    surface="#141414",
    surface_raised="#1c1c1c",
    border="#262626",           # neutral-800
    border_strong="#3f3f3f",    # neutral-700
    text="#fafafa",             # neutral-50
    text_muted="#a3a3a3",       # neutral-400
    text_faint="#737373",       # neutral-500
    accent="#e5e5e5",           # neutral-200
    accent_text="#0a0a0a",
    selection="#262626",
    danger="#f87171",
    success="#86efac",
)

LIGHT = Palette(
    background="#ffffff",
    surface="#fafafa",          # neutral-50
    surface_raised="#f5f5f5",   # neutral-100
    border="#e5e5e5",           # neutral-200
    border_strong="#d4d4d4",    # neutral-300
    text="#0a0a0a",             # neutral-950
    text_muted="#525252",       # neutral-600
    text_faint="#737373",       # neutral-500
    accent="#171717",           # neutral-900
    accent_text="#fafafa",
    selection="#f5f5f5",
    danger="#dc2626",
    success="#16a34a",
)

# Spacing scale, in pixels. Everything in the window uses one of these, so
# nothing drifts a pixel or two out of line with its neighbour.
#
# Deliberately loose. A tighter scale looked fine at a large window size and
# ran together once the window was smaller or the display scaled down, because
# a group's title sat almost on the border of the group above it. Separation
# between blocks has to survive being shrunk.
GAP_TIGHT = 8
GAP = 14
GAP_WIDE = 24
RADIUS = 6
CONTROL_HEIGHT = 30

# Clear air above a group so its title never crowds the box above.
GROUP_TITLE_SPACE = 26

# The letterbox behind the picture in the comparison viewer, and the text drawn
# on it before a pair is loaded.
#
# These two deliberately do NOT follow the palette, and are the only colours in
# the application that do not. The viewer exists to answer one question — did
# skin keep its texture, or did it turn to wax — and that judgement is made by
# eye. A surround shifts the tones you perceive in the picture it surrounds, so
# a backdrop that flipped from near-black to near-white with the interface
# theme would change what the user thinks they are seeing in the frame without
# anything in the frame having changed. A fixed mid-grey is the darkroom
# convention for exactly this reason: neutral, and biased neither way.
#
# Mid-grey also reads as deliberate against both palettes, which near-black
# (invisible in dark) and near-white (invisible in light) would not.
VIEWER_BACKDROP = "#525252"       # neutral-600
VIEWER_BACKDROP_TEXT = "#d4d4d4"  # neutral-300, legible on the above


class Mode(str, Enum):
    """What the user asked for, which is not the same as which palette wins.

    SYSTEM is a deferral: it means "whatever Windows is set to right now", and
    is resolved afresh every time the theme is applied.
    """

    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


# Where the preference lives. QSettings picks the per-user registry key on
# Windows; the organisation and application names must not drift, or a user's
# saved choice quietly disappears.
_ORG = "Enhancer"
_APP = "Enhancer"
_KEY = "theme/mode"


def is_dark(app) -> bool:
    """Follow the system light or dark setting."""
    colour = app.palette().window().color()
    return colour.value() < 128


def resolve(app, mode: Mode) -> Palette:
    """The palette a mode means right now. SYSTEM consults the OS."""
    if mode is Mode.LIGHT:
        return LIGHT
    if mode is Mode.DARK:
        return DARK
    return DARK if is_dark(app) else LIGHT


def _settings():
    """QSettings, imported here so the module stays importable without Qt."""
    from PySide6.QtCore import QSettings

    return QSettings(_ORG, _APP)


def load_mode() -> Mode:
    """The saved preference, SYSTEM when nothing is saved or the value is junk."""
    try:
        saved = _settings().value(_KEY)
    except Exception:
        return Mode.SYSTEM
    if saved is None:
        return Mode.SYSTEM
    try:
        return Mode(str(saved).strip().lower())
    except ValueError:
        return Mode.SYSTEM


def save_mode(mode: Mode) -> None:
    """Remember the preference for the next launch."""
    settings = _settings()
    settings.setValue(_KEY, Mode(mode).value)
    settings.sync()


def stylesheet(p: Palette) -> str:
    """Qt stylesheet for the whole application."""
    return f"""
    QWidget {{
        background-color: {p.background};
        color: {p.text};
        font-family: "Segoe UI", "Inter", system-ui, sans-serif;
        font-size: 13px;
    }}

    QGroupBox {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        margin-top: {GROUP_TITLE_SPACE}px;
        padding: {GAP_WIDE}px {GAP}px {GAP}px {GAP}px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: {GAP}px;
        top: 2px;
        padding: 0 6px;
        color: {p.text_faint};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    QLabel {{ background: transparent; }}

    QPushButton {{
        background-color: {p.surface_raised};
        color: {p.text};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS}px;
        padding: 6px 14px;
        min-height: {CONTROL_HEIGHT - 12}px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {p.selection};
        border-color: {p.text_faint};
    }}
    QPushButton:pressed {{
        background-color: {p.border};
        border-color: {p.text_faint};
    }}
    QPushButton:focus {{
        border: 1px solid {p.accent};
        outline: none;
    }}
    QPushButton:disabled {{
        color: {p.text_faint};
        border-color: {p.border};
        background-color: {p.surface};
    }}
    QPushButton#primary {{
        background-color: {p.accent};
        color: {p.accent_text};
        border-color: {p.accent};
        font-weight: 600;
    }}
    QPushButton#primary:hover {{ background-color: {p.text}; }}
    QPushButton#primary:pressed {{
        background-color: {p.text_muted};
        border-color: {p.text_muted};
    }}
    QPushButton#primary:focus {{ border: 1px solid {p.text_muted}; }}
    QPushButton#primary:disabled {{
        background-color: {p.surface_raised};
        color: {p.text_faint};
        border-color: {p.border};
    }}

    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS}px;
        padding: 4px 8px;
        min-height: {CONTROL_HEIGHT - 10}px;
        selection-background-color: {p.selection};
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
        border-color: {p.text_faint};
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {p.accent};
        outline: none;
    }}
    QComboBox:on {{ border-color: {p.accent}; }}
    QComboBox:disabled, QSpinBox:disabled,
    QDoubleSpinBox:disabled, QLineEdit:disabled {{
        color: {p.text_faint};
        background-color: {p.surface};
        border-color: {p.border};
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS}px;
        selection-background-color: {p.selection};
        selection-color: {p.text};
        padding: 4px;
        outline: none;
    }}

    QCheckBox {{ spacing: 8px; padding: 2px 0; }}
    QCheckBox::indicator {{
        width: 15px; height: 15px;
        border: 1px solid {p.border_strong};
        border-radius: 3px;
        background-color: {p.surface_raised};
    }}
    QCheckBox::indicator:hover {{ border-color: {p.text_faint}; }}
    QCheckBox::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
    }}
    QCheckBox::indicator:checked:hover {{ background-color: {p.text}; }}
    QCheckBox:focus {{ outline: none; }}
    QCheckBox::indicator:focus {{ border: 1px solid {p.accent}; }}
    QCheckBox:disabled {{ color: {p.text_faint}; }}
    QCheckBox::indicator:disabled {{
        background-color: {p.surface};
        border-color: {p.border};
    }}

    QSlider::groove:horizontal {{
        height: 3px;
        background: {p.border};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {p.text_muted};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {p.accent};
        border: none;
        width: 13px; height: 13px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{ background: {p.text}; }}
    QSlider::handle:horizontal:pressed {{ background: {p.text_muted}; }}
    QSlider::handle:horizontal:disabled {{ background: {p.border_strong}; }}
    QSlider::sub-page:horizontal:disabled {{ background: {p.border_strong}; }}
    QSlider:focus {{ outline: none; }}
    QSlider:focus::handle:horizontal {{
        border: 1px solid {p.accent};
        margin: -6px 0;
    }}

    QProgressBar {{
        background-color: {p.border};
        border: none;
        border-radius: 3px;
        height: 6px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{
        background-color: {p.text_muted};
        border-radius: 3px;
    }}

    QTableWidget {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        gridline-color: transparent;
        selection-background-color: {p.selection};
        selection-color: {p.text};
        outline: none;
    }}
    QTableWidget::item {{ padding: 5px 8px; border: none; }}
    QHeaderView::section {{
        background-color: {p.surface};
        color: {p.text_faint};
        border: none;
        border-bottom: 1px solid {p.border};
        padding: 5px 8px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    QTableCornerButton::section {{ background-color: {p.surface}; border: none; }}

    QPlainTextEdit, QTextBrowser {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        padding: {GAP_TIGHT}px;
        font-family: "Cascadia Mono", "Consolas", monospace;
        font-size: 12px;
        color: {p.text_muted};
        selection-background-color: {p.selection};
    }}

    QSplitter::handle {{ background: transparent; width: {GAP}px; }}

    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border_strong};
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {p.text_faint}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; }}
    QScrollBar::handle:horizontal {{
        background: {p.border_strong};
        border-radius: 5px;
        min-width: 28px;
    }}

    QToolTip {{
        background-color: {p.surface_raised};
        color: {p.text};
        border: 1px solid {p.border_strong};
        border-radius: {RADIUS}px;
        padding: 7px 9px;
    }}

    QLabel#hint {{ color: {p.text_faint}; font-size: 11px; }}
    QLabel#value {{ color: {p.text_muted}; font-size: 12px; }}
    QLabel#dropzone {{
        border: 1px dashed {p.border_strong};
        border-radius: {RADIUS}px;
        color: {p.text_faint};
        padding: 20px;
        background-color: {p.background};
    }}
    QLabel#help {{
        color: {p.text_faint};
        border: 1px solid {p.border_strong};
        border-radius: 8px;
        font-size: 10px;
        font-weight: 600;
    }}
    QLabel#help:hover {{ color: {p.text}; border-color: {p.text_faint}; }}
    QLabel#caption {{
        color: {p.text_faint};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}

    QFrame#card {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
    }}

    /* Segmented A/B switch: buttons sit in a row and read as one control,
       so the selected one is filled rather than merely outlined. */
    QPushButton#segment {{
        background-color: {p.surface};
        color: {p.text_muted};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        padding: 4px {GAP_TIGHT}px;
        min-height: {CONTROL_HEIGHT - 14}px;
        font-size: 12px;
        font-weight: 500;
    }}
    QPushButton#segment:hover {{
        background-color: {p.surface_raised};
        color: {p.text};
    }}
    QPushButton#segment:focus {{ border: 1px solid {p.accent}; }}
    QPushButton#segment:checked {{
        background-color: {p.accent};
        color: {p.accent_text};
        border-color: {p.accent};
        font-weight: 600;
    }}
    QPushButton#segment:checked:hover {{
        background-color: {p.text};
        border-color: {p.text};
        color: {p.accent_text};
    }}
    QPushButton#segment:disabled {{
        color: {p.text_faint};
        background-color: {p.surface};
        border-color: {p.border};
    }}

    /* The settings tabs are the primary navigation of the panel, so they are
       styled as the same segmented control as #segment above rather than as
       chrome-style tabs: one filled, the rest outlined. Every tab carries an
       explicit colour and border — left unstyled, an unselected tab inherited
       the window's text colour on the window's background and vanished
       outright in the light palette, which hid three quarters of the settings.
       The pane is a hairline only: the groups inside already have borders, and
       a second frame around them would double up. */
    QTabWidget::pane {{
        background: transparent;
        border: none;
        border-top: 1px solid {p.border};
        top: -1px;
    }}
    QTabWidget::tab-bar {{ left: 0; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{
        background-color: {p.surface};
        color: {p.text_muted};
        border: 1px solid {p.border};
        border-radius: {RADIUS}px;
        padding: {GAP_TIGHT}px {GAP}px;
        margin-right: {GAP_TIGHT}px;
        margin-bottom: {GAP_TIGHT}px;
        min-height: {CONTROL_HEIGHT - 16}px;
        font-size: 12px;
        font-weight: 500;
    }}
    QTabBar::tab:!selected {{
        background-color: {p.surface};
        color: {p.text_muted};
        border-color: {p.border};
    }}
    QTabBar::tab:!selected:hover {{
        background-color: {p.surface_raised};
        color: {p.text};
        border-color: {p.border_strong};
    }}
    QTabBar::tab:focus {{ border: 1px solid {p.accent}; }}
    QTabBar::tab:selected {{
        background-color: {p.accent};
        color: {p.accent_text};
        border-color: {p.accent};
        font-weight: 600;
    }}
    QTabBar::tab:selected:hover {{
        background-color: {p.text};
        border-color: {p.text};
        color: {p.accent_text};
    }}
    QTabBar::tab:disabled {{
        background-color: {p.surface};
        color: {p.text_faint};
        border-color: {p.border};
    }}

    QLabel#analysis {{ color: {p.text_muted}; font-size: 12px; }}
    QLabel#status {{ color: {p.text_muted}; font-size: 12px; }}
    QLabel#headline {{ color: {p.text}; font-size: 14px; font-weight: 600; }}
    QLabel#warning {{ color: {p.danger}; font-size: 12px; }}
    """


def apply(app, mode: Mode | None = None) -> Palette:
    """Apply the theme. `None` means: use the saved preference.

    An explicit mode is a choice the user just made, so it is written back.
    The saved one is only read — rewriting it on every launch would churn the
    settings file for nothing.
    """
    if mode is None:
        mode = load_mode()
    else:
        mode = Mode(mode)
        save_mode(mode)
    palette = resolve(app, mode)
    app.setStyleSheet(stylesheet(palette))
    return palette
