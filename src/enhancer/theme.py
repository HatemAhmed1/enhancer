"""Visual theme.

A restrained neutral palette in the shadcn/ui idiom: greys only, one subtle
accent, thin borders, generous but consistent spacing. No gradients, no glow,
no colour for its own sake — this is a tool, and the picture being worked on is
the only thing on screen that should draw the eye.

Colours are the shadcn "neutral" scale converted from HSL to hex.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def is_dark(app) -> bool:
    """Follow the system light or dark setting."""
    colour = app.palette().window().color()
    return colour.value() < 128


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
    QPushButton:hover {{ border-color: {p.text_faint}; }}
    QPushButton:pressed {{ background-color: {p.border}; }}
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
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {p.text_faint};
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
    QCheckBox::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
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
    QLabel#analysis {{ color: {p.text_muted}; font-size: 12px; }}
    QLabel#status {{ color: {p.text_muted}; font-size: 12px; }}
    """


def apply(app) -> Palette:
    """Apply the theme, following the system light or dark setting."""
    palette = DARK if is_dark(app) else LIGHT
    app.setStyleSheet(stylesheet(palette))
    return palette
