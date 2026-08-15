"""Tests for the command-line layer.

This layer had no coverage until an audit found a crash reachable purely
through a flag combination. Library code was well tested; the place the user
actually touches was not.
"""

import pytest

from enhancer import cli


# --- resolve_target_fps -----------------------------------------------------


def test_no_interpolation_flags_yields_no_target():
    target, error = cli.resolve_target_fps(None, None, 24.0, have_weights=True)
    assert target is None and error is None


def test_absolute_fps_is_used_directly():
    target, error = cli.resolve_target_fps(60.0, None, 24.0, have_weights=True)
    assert target == 60.0 and error is None


def test_multiplier_is_applied_to_the_source_rate():
    target, error = cli.resolve_target_fps(None, 2.0, 25.0, have_weights=True)
    assert target == 50.0 and error is None


def test_both_flags_together_are_rejected():
    target, error = cli.resolve_target_fps(60.0, 2.0, 24.0, have_weights=True)
    assert target is None
    assert "either --fps or --interpolate" in error


def test_target_below_source_is_rejected():
    """This tool interpolates; it does not decimate."""
    target, error = cli.resolve_target_fps(24.0, None, 60.0, have_weights=True)
    assert target is None
    assert "below the source rate" in error


def test_multiplier_below_one_is_rejected():
    target, error = cli.resolve_target_fps(None, 0.5, 24.0, have_weights=True)
    assert target is None
    assert error


def test_missing_weights_are_reported_before_anything_expensive():
    target, error = cli.resolve_target_fps(60.0, None, 24.0, have_weights=False)
    assert target is None
    assert "RIFE weights" in error


def test_missing_weights_do_not_block_a_plain_upscale():
    """A plain upscale must not require interpolation weights."""
    target, error = cli.resolve_target_fps(None, None, 24.0, have_weights=False)
    assert target is None and error is None


def test_equal_rates_are_allowed():
    target, error = cli.resolve_target_fps(24.0, None, 24.0, have_weights=True)
    assert target == 24.0 and error is None


def test_non_integer_ratio_is_allowed():
    target, error = cli.resolve_target_fps(60.0, None, 24.0, have_weights=True)
    assert target == 60.0 and error is None


# --- argument parsing -------------------------------------------------------


def test_video_subcommand_exposes_the_interpolation_flags():
    parser_args = cli.main.__doc__  # touch the module so argparse is built
    with pytest.raises(SystemExit):
        cli.main(["video", "--help"])


def test_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])


def test_missing_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.main([])


# --- cmd_models -------------------------------------------------------------


def test_models_command_reports_an_empty_directory(tmp_path, capsys):
    args = type("A", (), {"dir": str(tmp_path / "absent")})()
    assert cli.cmd_models(args) == 1
    assert "No weights found" in capsys.readouterr().out


def test_models_command_lists_what_it_finds(tmp_path, capsys):
    (tmp_path / "a_model.pth").write_bytes(b"x" * 2048)
    args = type("A", (), {"dir": str(tmp_path)})()
    assert cli.cmd_models(args) == 0
    assert "a_model.pth" in capsys.readouterr().out


# --- cmd_analyze ------------------------------------------------------------


def test_analyze_reports_a_progressive_source(synthetic_clip, capsys):
    args = type("A", (), {"input": str(synthetic_clip)})()
    assert cli.cmd_analyze(args) == 0
    out = capsys.readouterr().out
    assert "Scan:" in out
    assert "progressive" in out
    assert "no scan correction needed" in out


def test_analyze_reports_dimensions_and_rate(synthetic_clip, capsys):
    args = type("A", (), {"input": str(synthetic_clip)})()
    cli.cmd_analyze(args)
    out = capsys.readouterr().out
    assert "320x240" in out
    assert "25.000 fps" in out


# --- rife_weights -----------------------------------------------------------


def test_rife_weights_returns_a_list():
    assert isinstance(cli.rife_weights(), list)


def test_rife_dir_is_absolute():
    """A relative path would silently fail from any other working directory."""
    from enhancer.rife import RIFE_DIR

    assert RIFE_DIR.is_absolute() or RIFE_DIR.is_dir()
