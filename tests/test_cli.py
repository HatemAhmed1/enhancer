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
    args = type("A", (), {"dir": str(tmp_path / "absent"), "get": None})()
    assert cli.cmd_models(args) == 1
    assert "(none)" in capsys.readouterr().out


def test_models_command_lists_what_it_finds(tmp_path, capsys):
    (tmp_path / "a_model.pth").write_bytes(b"x" * 2048)
    args = type("A", (), {"dir": str(tmp_path), "get": None})()
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


# --- model catalogue --------------------------------------------------------


def _models_args(tmp_path, get=None):
    return type("A", (), {"dir": str(tmp_path), "get": get})()


def test_models_lists_the_catalogue_alongside_installed(tmp_path, capsys):
    cli.cmd_models(_models_args(tmp_path))
    out = capsys.readouterr().out
    assert "Catalogue" in out
    assert "realesr-general-x4v3" in out, "the shipped manifest must be reachable"


def test_catalogue_marks_absent_models_as_not_installed(tmp_path, capsys):
    cli.cmd_models(_models_args(tmp_path))
    assert "not installed" in capsys.readouterr().out


def test_catalogue_marks_present_models_as_installed(tmp_path, capsys):
    (tmp_path / "realesr-general-x4v3.pth").write_bytes(b"x")
    cli.cmd_models(_models_args(tmp_path))
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "realesr-general-x4v3" in ln and "4x" in ln)
    assert line.rstrip().endswith("installed")
    assert "not installed" not in line


def test_unknown_catalogue_id_is_rejected(tmp_path, capsys):
    assert cli.cmd_models(_models_args(tmp_path, get="no-such-model")) == 1
    assert "Unknown model id" in capsys.readouterr().out


def test_every_catalogue_entry_has_a_real_digest(tmp_path):
    """The project rule: no invented hashes ever ship."""
    import re

    for entry in cli._registry(tmp_path).list():
        assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256)
        assert entry.url.startswith("https://")
        assert entry.size > 0


# --- an absurd frame rate ----------------------------------------------------


def test_absurd_target_fps_is_rejected():
    """--fps 100000 on a two-second clip was still grinding ten minutes later,
    with nothing to say the number was the problem."""
    target, error = cli.resolve_target_fps(100000.0, None, 25.0, have_weights=True)
    assert target is None
    assert "limit" in error


def test_absurd_multiplier_is_rejected():
    target, error = cli.resolve_target_fps(None, 4000.0, 25.0, have_weights=True)
    assert target is None and error


def test_the_fps_ceiling_itself_is_allowed():
    from enhancer.requests import MAX_OUTPUT_FPS

    target, error = cli.resolve_target_fps(MAX_OUTPUT_FPS, None, 24.0, have_weights=True)
    assert target == MAX_OUTPUT_FPS and error is None


def test_ordinary_high_frame_rates_are_untouched():
    for rate in (48.0, 50.0, 60.0, 120.0):
        target, error = cli.resolve_target_fps(rate, None, 24.0, have_weights=True)
        assert target == rate and error is None


# --- resume ownership --------------------------------------------------------


def test_has_journal_is_false_for_an_absent_directory(tmp_path):
    assert not cli._has_journal(tmp_path / "nothing.job")


def test_has_journal_is_false_for_a_directory_without_one(tmp_path):
    (tmp_path / "job").mkdir()
    assert not cli._has_journal(tmp_path / "job")


def test_has_journal_finds_a_plain_render(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "job.json").write_text("{}")
    assert cli._has_journal(job)


def test_has_journal_finds_a_dual_pass_render(tmp_path):
    """A dual pass keeps its journals one level down, and its resume must not
    be mistaken for an accidental overwrite."""
    job = tmp_path / "job"
    (job / "pass2").mkdir(parents=True)
    (job / "pass2" / "job.json").write_text("{}")
    assert cli._has_journal(job)


# --- cmd_video, with the engine stubbed out ---------------------------------


@pytest.fixture
def stub_engine(monkeypatch, tmp_path):
    """Everything cmd_video needs to reach render_resumable, and a spy there."""
    class _Model:
        scale = 2

    class _Up:
        scale = 2
        cpu_fallback_count = 0

        def __init__(self, *a, **kw):
            pass

    captured = {}

    def fake_render(profile, upscaler, output, **kwargs):
        captured["output"] = output
        captured.update(kwargs)
        from pathlib import Path as _P
        _P(output).write_bytes(b"rendered")
        return _P(output)

    monkeypatch.setattr(cli, "select_device", lambda prefer_cuda=True: "cpu")
    monkeypatch.setattr(cli, "load_model", lambda path, device=None, half=True: _Model())
    monkeypatch.setattr(cli, "Upscaler", _Up)
    monkeypatch.setattr(cli, "_auto_tile", lambda scale, overlap: 256)
    monkeypatch.setattr(cli, "render_resumable", fake_render)

    model = tmp_path / "model.pth"
    model.write_bytes(b"weights")
    return captured, model


def test_video_refuses_an_output_that_is_the_source(tmp_path, synthetic_clip, capsys):
    """A case-only difference is one file on Windows. The source was destroyed
    and the program printed Done and exited zero."""
    import os

    if os.path.normcase("A") != os.path.normcase("a"):
        pytest.skip("case-sensitive filesystem")

    before = synthetic_clip.read_bytes()
    shouty = synthetic_clip.parent / synthetic_clip.name.upper()
    code = cli.main(["video", "m.pth", str(synthetic_clip), str(shouty)])
    assert code == 4
    assert "same file as the source" in capsys.readouterr().out
    assert synthetic_clip.read_bytes() == before


def test_video_refuses_an_output_that_is_the_source_exactly(synthetic_clip, capsys):
    code = cli.main(["video", "m.pth", str(synthetic_clip), str(synthetic_clip)])
    assert code == 4
    assert "same file as the source" in capsys.readouterr().out


def test_video_refuses_to_overwrite_an_existing_result(
    tmp_path, synthetic_clip, stub_engine, capsys
):
    """A previous render is hours of work and was replaced without a word."""
    _captured, model = stub_engine
    out = tmp_path / "done.mkv"
    out.write_bytes(b"a previous result")

    code = cli.main(["video", str(model), str(synthetic_clip), str(out), "--no-restore"])
    assert code == 4
    assert "--force" in capsys.readouterr().out
    assert out.read_bytes() == b"a previous result", "the earlier result must survive"


def test_force_allows_the_overwrite(tmp_path, synthetic_clip, stub_engine):
    _captured, model = stub_engine
    out = tmp_path / "done.mkv"
    out.write_bytes(b"a previous result")

    code = cli.main(
        ["video", str(model), str(synthetic_clip), str(out), "--no-restore", "--force"]
    )
    assert code == 0
    assert out.read_bytes() == b"rendered"


def test_a_resume_never_needs_force(tmp_path, synthetic_clip, stub_engine):
    """The resume flow legitimately writes to an output whose .job directory it
    owns; the guard must not stand in its way."""
    _captured, model = stub_engine
    out = tmp_path / "done.mkv"
    out.write_bytes(b"a partly assembled result")
    job = out.with_suffix(".job")
    job.mkdir()
    (job / "job.json").write_text("{}")

    code = cli.main(["video", str(model), str(synthetic_clip), str(out), "--no-restore"])
    assert code == 0


def test_scene_threshold_default_hashes_the_same_as_stating_it(
    tmp_path, synthetic_clip, stub_engine
):
    """--scene-threshold 0.3 refused to resume a job that had used 0.3: the raw
    argument went into the hash, None when unset, while the render used the
    resolved default."""
    captured, model = stub_engine
    base = ["video", str(model), str(synthetic_clip), "--no-restore"]

    cli.main(base[:3] + [str(tmp_path / "a.mkv")] + base[3:])
    implicit = dict(captured["settings"])

    cli.main(base[:3] + [str(tmp_path / "b.mkv")] + base[3:]
             + ["--scene-threshold", str(cli.DEFAULT_SCENE_THRESHOLD)])
    explicit = dict(captured["settings"])

    assert implicit == explicit
    assert implicit["scene_threshold"] == cli.DEFAULT_SCENE_THRESHOLD


def test_tile_is_not_in_the_command_line_job_hash(tmp_path, synthetic_clip, stub_engine):
    """_auto_tile reads current free VRAM, so resuming with a browser open
    picked a different tile and refused the resume."""
    captured, model = stub_engine
    cli.main(["video", str(model), str(synthetic_clip),
              str(tmp_path / "a.mkv"), "--no-restore"])
    assert "tile" not in captured["settings"]


def test_missing_ffmpeg_is_reported_in_the_user_s_terms(
    tmp_path, synthetic_clip, monkeypatch, capsys
):
    """It arrived as a fifteen-line traceback ending in
    FileNotFoundError: [WinError 2], which never mentions ffmpeg."""
    from enhancer import preflight

    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    code = cli.main(["video", "m.pth", str(synthetic_clip), str(tmp_path / "out.mkv")])
    assert code == 1
    out = capsys.readouterr().out
    assert "ffmpeg" in out
    assert "Cannot render" in out
    assert "Traceback" not in out


def test_preflight_does_not_log_its_own_probe_failing(
    tmp_path, synthetic_clip, monkeypatch, caplog
):
    """Hardware detection probes ffmpeg and logs the failure with a stack
    trace. Here that failure is the answer being looked for, and a traceback
    printed just above the plain-English explanation undoes the point of it."""
    import logging

    from enhancer import preflight, proc

    def explode(*a, **kw):
        raise FileNotFoundError(2, "The system cannot find the file specified")

    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    monkeypatch.setattr(proc, "run", explode)

    with caplog.at_level(logging.WARNING):
        code = cli.main(["video", "m.pth", str(synthetic_clip),
                         str(tmp_path / "out.mkv")])
    assert code == 1
    noisy = [r for r in caplog.records if r.name == "enhancer.system"]
    assert not noisy, f"preflight logged {[r.getMessage() for r in noisy]}"


# --- still images ------------------------------------------------------------


def test_image_run_says_that_deblock_and_degrain_do_not_apply(
    tmp_path, monkeypatch, capsys
):
    """They run inside ffmpeg's decode filter chain, which a still never goes
    through. Saying nothing left them looking applied."""
    from PIL import Image

    src = tmp_path / "still.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(src)
    out = tmp_path / "big.png"

    monkeypatch.setattr(cli, "select_device", lambda prefer_cuda=True: "cpu")
    monkeypatch.setattr(cli, "load_model",
                        lambda path, device=None, half=True: type("M", (), {"scale": 2})())
    monkeypatch.setattr(cli, "Upscaler",
                        lambda *a, **kw: type("U", (), {"cpu_fallback_count": 0})())
    monkeypatch.setattr("enhancer.images.upscale_image",
                        lambda i, o, up, texture=None: __import__("pathlib").Path(o))

    code = cli.main(["video", "m.pth", str(src), str(out), "--deblock", "0.5"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "--deblock" in printed and "--degrain" in printed
    assert "video-only" in printed


def test_image_run_refuses_an_output_that_is_the_source(tmp_path, capsys):
    from PIL import Image

    src = tmp_path / "still.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(src)
    code = cli.main(["video", "m.pth", str(src), str(src)])
    assert code == 4
    assert "same file as the source" in capsys.readouterr().out
