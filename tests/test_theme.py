"""The palette is restrained on purpose: greys, one accent, no colour for
its own sake. These tests keep it that way."""

import re

import pytest

from enhancer import theme


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_every_colour_is_a_hex_value(palette):
    for name in palette.__dataclass_fields__:
        value = getattr(palette, name)
        assert re.fullmatch(r"#[0-9a-f]{6}", value), f"{name} is not a hex colour"


def _is_grey(hex_colour: str) -> bool:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return max(r, g, b) - min(r, g, b) <= 6


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_the_interface_palette_is_neutral(palette):
    """No blues or purples. The picture is the only colour on screen."""
    neutral = ["background", "surface", "surface_raised", "border",
               "border_strong", "text", "text_muted", "text_faint",
               "accent", "accent_text", "selection"]
    for name in neutral:
        assert _is_grey(getattr(palette, name)), f"{name} is not neutral"


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_status_colours_are_the_only_coloured_ones(palette):
    """Danger and success may be coloured — they mean something."""
    assert not _is_grey(palette.danger)
    assert not _is_grey(palette.success)


def test_dark_is_dark_and_light_is_light():
    assert theme.DARK.background < "#333333"
    assert theme.LIGHT.background > "#cccccc"


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_text_contrasts_with_its_background(palette):
    def luma(c):
        r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    assert abs(luma(palette.text) - luma(palette.background)) > 150


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_stylesheet_builds_and_has_no_placeholders(palette):
    css = theme.stylesheet(palette)
    assert len(css) > 2000
    assert "{p." not in css, "an unsubstituted field was left in the stylesheet"
    assert "None" not in css


@pytest.mark.parametrize("palette", [theme.DARK, theme.LIGHT])
def test_stylesheet_has_no_gradients_or_glow(palette):
    """The look is flat and quiet by intent."""
    css = theme.stylesheet(palette).lower()
    for banned in ("gradient", "box-shadow", "drop-shadow", "glow"):
        assert banned not in css


def test_spacing_scale_is_ordered():
    assert theme.GAP_TIGHT < theme.GAP < theme.GAP_WIDE


def test_the_two_palettes_define_the_same_fields():
    assert theme.DARK.__dataclass_fields__.keys() == theme.LIGHT.__dataclass_fields__.keys()
