"""Model file names say nothing about what a model is for.

The dropdown shows the purpose instead, so choosing does not require knowing
what "Nomos2 realplksr dysample" means.
"""

import pytest

from enhancer.help_text import (
    MODEL_LABELS,
    display_name,
    model_category,
    model_rank,
    model_scale,
)


# --- scale, read from the file name -----------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("2xParimgCompact.pth", 2),
    ("4xNomos2_realplksr_dysample.pth", 4),
    ("4xPurePhoto-span.pth", 4),
    ("2xModernSpanimationV1.pth", 2),
])
def test_scale_when_the_number_comes_first(name, expected):
    assert model_scale(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("realesr-general-x4v3.pth", 4),
    ("RealESRGAN_x2plus.pth", 2),
    ("some_model_x8.pth", 8),
])
def test_scale_when_the_number_comes_second(name, expected):
    """Both orders are in the wild. Reading only one called a 4x model 2x."""
    assert model_scale(name) == expected


def test_scale_ignores_unrelated_digits():
    assert model_scale("4xNomos2_realplksr_dysample.pth") == 4


def test_scale_defaults_to_two_when_unstated():
    assert model_scale("mystery_weights.pth") == 2


# --- what a model is for ----------------------------------------------------


@pytest.mark.parametrize("stem", list(MODEL_LABELS))
def test_every_catalogue_model_has_a_purpose(stem):
    assert model_category(stem + ".pth") != "General"


@pytest.mark.parametrize("name,expected", [
    ("4x_SomeAnimeModel.pth", "Animation"),
    ("2x_cartoon_upscaler.pth", "Animation"),
    ("4x_photo_thing.pth", "Photos"),
    ("2x_face_restore.pth", "Faces"),
    ("4x_manga_text.pth", "Text & line art"),
])
def test_purpose_is_guessed_for_unknown_models(name, expected):
    assert model_category(name) == expected


def test_a_wholly_unknown_model_is_called_general():
    assert model_category("weights_final_v2.pth") == "General"


# --- the label shown in the dropdown ----------------------------------------


def test_label_leads_with_the_purpose():
    assert display_name("2xParimgCompact.pth").startswith("Video")


def test_label_states_the_enlargement():
    assert "2x" in display_name("2xParimgCompact.pth")
    assert "4x" in display_name("4xPurePhoto-span.pth")


def test_label_states_the_speed_for_known_models():
    assert "fastest" in display_name("2xParimgCompact.pth")
    assert "very slow" in display_name("RealESRGAN_x2plus.pth")


def test_label_says_what_the_source_becomes():
    """The number that matters is the output size, not the multiplier."""
    assert "4K" in display_name("2xParimgCompact.pth", source_width=1920)
    assert "8K" in display_name("4xPurePhoto-span.pth", source_width=1920)


def test_label_adapts_to_the_source():
    hd = display_name("2xParimgCompact.pth", source_width=1920)
    dvd = display_name("2xParimgCompact.pth", source_width=854)
    assert hd != dvd
    assert "4K" in hd


def test_label_omits_the_output_size_when_no_source_is_loaded():
    assert "→" not in display_name("2xParimgCompact.pth")


def test_an_unusual_output_size_is_given_in_pixels():
    """Naming 1708 wide "720p" would be wrong; 720p is 1280 wide."""
    assert "1708px wide" in display_name("2xParimgCompact.pth", source_width=854)


def test_unknown_models_still_get_a_useful_label():
    label = display_name("4x_SomeAnimeModel.pth", source_width=1920)
    assert "Animation" in label
    assert "4x" in label


# --- ordering ---------------------------------------------------------------


def test_the_usual_answer_sorts_first():
    assert model_rank("2xParimgCompact.pth") == 0


def test_specialist_models_sort_after_general_ones():
    assert model_rank("RealESRGAN_x2plus.pth") > model_rank("2xParimgCompact.pth")


def test_unknown_models_sort_last():
    assert model_rank("mystery.pth") > max(
        model_rank(stem + ".pth") for stem in MODEL_LABELS
    )


def test_ranks_are_distinct():
    ranks = [model_rank(stem + ".pth") for stem in MODEL_LABELS]
    assert len(ranks) == len(set(ranks))
