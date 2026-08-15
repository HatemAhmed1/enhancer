"""Every control the user can touch must be explained.

These tests are what stop a control being added without a description.
"""

import pytest

from enhancer.help_text import ARCH_NOTES, HELP, MODEL_NOTES, describe_model

# Must match the keys used by window.py.
REQUIRED_KEYS = [
    "source", "analysis", "model", "rescan", "no_restore",
    "degrain", "detail", "regrain", "deblock",
    "fps_mode", "fps_target", "fps_multiplier", "scene_threshold",
    "output", "segment_frames", "vram_budget", "cpu",
    "preview_button", "render_button", "cancel_button", "progress",
    "queue", "queue_start", "queue_stop", "queue_remove", "queue_clear",
    "forecast",
    "guide_button",
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_every_control_has_help(key):
    assert key in HELP, f"control {key!r} has no explanation"


# Controls where a decision has consequences, so a one-liner is not enough.
NEEDS_DETAIL = [
    "source", "analysis", "model", "no_restore",
    "degrain", "detail", "regrain", "deblock",
    "fps_mode", "fps_target", "fps_multiplier", "scene_threshold",
    "output", "segment_frames", "vram_budget", "cpu",
    "preview_button", "render_button", "progress",
    "queue", "queue_stop", "queue_remove", "forecast",
]

# Plain actions: complete in a line or two, and padding them would be worse.
SIMPLE = ["rescan", "guide_button", "cancel_button", "queue_start", "queue_clear"]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_help_says_something(key):
    assert len(HELP[key]) > 30, f"{key!r} explanation is too thin to be useful"


@pytest.mark.parametrize("key", NEEDS_DETAIL)
def test_consequential_controls_explain_the_trade_off(key):
    """A plain button can be one line. A control that changes the output cannot."""
    assert len(HELP[key]) > 90, f"{key!r} needs more than a label"


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_help_opens_with_a_short_first_line(key):
    """Lead with what it does; detail comes after."""
    first = HELP[key].split("\n")[0]
    assert len(first) <= 90, f"{key!r} opens with a wall of text"


def test_no_orphan_help_entries():
    """Text for a control that no longer exists is stale documentation."""
    assert set(HELP) == set(REQUIRED_KEYS)


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_help_avoids_unexplained_jargon(key):
    """These are for someone who does not know the internals."""
    banned = ["tensor", "fp16", "cuda", "spandrel", "ffmpeg", "hqdn3d",
              "bwdif", "fieldmatch", "kwarg", "dataclass", "callback"]
    lowered = HELP[key].lower()
    for word in banned:
        assert word not in lowered, f"{key!r} uses jargon: {word!r}"


@pytest.mark.parametrize("name", list(MODEL_NOTES))
def test_catalogue_models_say_what_they_suit(name):
    note = describe_model(name)
    assert "Best for:" in note or "best for" in note.lower()


def test_describe_model_tolerates_an_extension():
    assert describe_model("2xParimgCompact.pth") == describe_model("2xParimgCompact")


def test_describe_model_is_case_insensitive():
    assert describe_model("2XPARIMGCOMPACT.pth") == describe_model("2xParimgCompact")


def test_unknown_model_falls_back_to_architecture():
    note = describe_model("4x_SomeoneElses_span_model.pth")
    assert "SPAN" in note
    assert "Quadruples" in note


def test_completely_unknown_model_still_says_something_useful():
    note = describe_model("mystery_weights.pth")
    assert "added yourself" in note
    assert len(note) > 60


def test_every_arch_note_is_nonempty():
    assert all(len(v) > 20 for v in ARCH_NOTES.values())


def test_speed_guidance_is_present_for_the_slow_models():
    """The slow ones must warn, or someone starts a 24-hour render by mistake."""
    assert "slow" in describe_model("RealESRGAN_x2plus").lower()
    assert "slow" in describe_model("4xNomos2_realplksr_dysample").lower()


def test_degrain_help_points_at_the_waxy_skin_fix():
    assert "waxy" in HELP["degrain"].lower() or "airbrush" in HELP["degrain"].lower()
    assert "lower" in HELP["degrain"].lower()


def test_detail_help_explains_it_cannot_invent_texture():
    assert "cannot" in HELP["detail"].lower()


def test_every_key_is_classified():
    """A new control must be judged simple or consequential, not skipped."""
    assert set(NEEDS_DETAIL) | set(SIMPLE) == set(REQUIRED_KEYS)
    assert not (set(NEEDS_DETAIL) & set(SIMPLE))


@pytest.mark.parametrize("key", SIMPLE)
def test_simple_controls_stay_brief(key):
    """Padding a plain action out to a paragraph makes it worse, not better."""
    assert len(HELP[key]) < 200
