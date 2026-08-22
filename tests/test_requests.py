from pathlib import Path

import pytest

from enhancer.requests import RenderRequest


def _req(**kw):
    base = dict(
        model=Path("models/custom/m.pth"),
        source=Path("in.mp4"),
        output=Path("out.mkv"),
    )
    base.update(kw)
    return RenderRequest(**base)


def test_defaults_match_the_command_line_defaults():
    r = _req()
    assert r.degrain == 0.25
    assert r.detail_retention == 0.25
    assert r.regrain == 0.6
    assert r.deblock == 0.0
    assert r.target_fps is None


def test_job_dir_defaults_beside_the_output():
    assert _req(output=Path("/tmp/a.mkv")).job_dir == Path("/tmp/a.job")


def test_explicit_job_dir_is_respected():
    assert _req(job_dir=Path("/tmp/elsewhere")).job_dir == Path("/tmp/elsewhere")


def test_settings_dict_includes_every_pixel_affecting_choice():
    keys = set(_req().settings_dict(scale=2, tile=512, video_filter="hqdn3d=1:1:1:1"))
    for expected in (
        "model", "scale", "overlap", "deblock", "degrain",
        "detail_retention", "regrain", "target_fps", "video_filter",
    ):
        assert expected in keys


def test_tile_is_not_in_the_job_hash():
    """Tile comes from whatever graphics memory is free at the moment, and the
    out-of-memory runner already changes it mid-render. Hashing it made the
    same command refuse to resume itself with a browser open."""
    a = _req().settings_dict(scale=2, tile=512, video_filter="")
    b = _req().settings_dict(scale=2, tile=256, video_filter="")
    assert a == b
    assert "tile" not in a


def test_settings_dict_changes_when_a_slider_moves():
    a = _req().settings_dict(scale=2, tile=512, video_filter="")
    b = _req(regrain=0.9).settings_dict(scale=2, tile=512, video_filter="")
    assert a != b


def test_out_of_range_strength_is_rejected():
    with pytest.raises(ValueError):
        _req(degrain=1.5)


def test_negative_strength_is_rejected():
    with pytest.raises(ValueError):
        _req(detail_retention=-0.1)


def test_no_restore_zeroes_every_strength():
    r = _req(no_restore=True)
    assert (r.degrain, r.deblock, r.detail_retention, r.regrain) == (0.0, 0.0, 0.0, 0.0)


def test_no_restore_overrides_explicit_strengths():
    """The checkbox must win over whatever the sliders happen to say."""
    r = _req(no_restore=True, degrain=0.9, regrain=0.9)
    assert r.degrain == 0.0 and r.regrain == 0.0


def test_preview_range_is_none_by_default():
    assert _req().preview_frames is None


def test_preview_request_is_marked_as_such():
    assert _req(preview_frames=250).is_preview


def test_preview_output_is_distinct_from_the_real_output():
    r = _req(output=Path("/tmp/a.mkv"), preview_frames=250)
    assert r.output != Path("/tmp/a.mkv")
    assert "preview" in r.output.name


def test_preview_uses_a_separate_job_dir():
    """A preview must never be mistaken for progress on the real render."""
    r = _req(output=Path("/tmp/a.mkv"), preview_frames=250)
    assert r.job_dir != Path("/tmp/a.job")


def test_preview_keeps_the_original_extension():
    r = _req(output=Path("/tmp/a.mkv"), preview_frames=10)
    assert r.output.suffix == ".mkv"


def test_target_fps_below_source_is_rejected_at_validation():
    with pytest.raises(ValueError, match="below"):
        _req(target_fps=24.0).validate_against(src_fps=60.0)


def test_target_fps_equal_to_source_is_accepted():
    _req(target_fps=24.0).validate_against(src_fps=24.0)


def test_validation_passes_when_no_interpolation_requested():
    _req().validate_against(src_fps=60.0)


def test_absurd_target_fps_is_rejected():
    """--fps 100000 on a two-second clip was still grinding ten minutes later."""
    from enhancer.requests import MAX_OUTPUT_FPS

    with pytest.raises(ValueError, match="limit"):
        _req(target_fps=100000.0).validate_against(src_fps=25.0)
    _req(target_fps=MAX_OUTPUT_FPS).validate_against(src_fps=25.0)


def test_ordinary_high_frame_rates_are_still_allowed():
    for rate in (48.0, 50.0, 60.0, 120.0):
        _req(target_fps=rate).validate_against(src_fps=24.0)


def test_string_paths_are_coerced():
    r = RenderRequest(model="m.pth", source="in.mp4", output="out.mkv")
    assert isinstance(r.model, Path) and isinstance(r.output, Path)
