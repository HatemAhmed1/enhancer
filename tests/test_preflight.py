"""Prerequisites must be reported in terms a stranger can act on.

Before this existed, a machine without ffmpeg produced a FileNotFoundError from
inside a subprocess call, several frames deep, naming a program the user had
never heard of.
"""

import shutil
from pathlib import Path

import pytest

from enhancer import preflight


@pytest.fixture
def no_ffmpeg(monkeypatch):
    real = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda n, *a, **k: None if n in ("ffmpeg", "ffprobe") else real(n, *a, **k),
    )


def test_ffmpeg_is_found_on_this_machine():
    assert preflight.check_ffmpeg().ok


def test_a_missing_ffmpeg_blocks_and_says_how_to_get_it(no_ffmpeg):
    requirement = preflight.check_ffmpeg()

    assert not requirement.ok
    assert requirement.blocking, "a render cannot run without it"
    assert requirement.fix, "told the user it is missing but not what to do"


def test_the_fix_names_a_command_for_this_platform(no_ffmpeg):
    import platform

    fix = preflight.check_ffmpeg().fix
    expected = {"Windows": "winget", "Darwin": "brew"}.get(platform.system(), "apt")
    assert expected in fix


def test_missing_models_block_but_name_the_folder(tmp_path):
    requirement = preflight.check_models(tmp_path / "nothing")

    assert not requirement.ok
    assert requirement.blocking
    assert "nothing" in requirement.detail


def test_models_present_are_counted(tmp_path):
    (tmp_path / "SomeModel.pth").write_bytes(b"")
    requirement = preflight.check_models(tmp_path)

    assert requirement.ok
    assert "SomeModel" in requirement.detail


def test_no_graphics_card_is_a_warning_not_a_refusal(monkeypatch):
    """Telling someone with no card that they cannot proceed would be a lie."""
    import sys
    import types

    fake = types.SimpleNamespace(
        detect=lambda probe_encoders=True: types.SimpleNamespace(
            accelerator="cpu", gpu_name="", vram_bytes=0,
        )
    )
    monkeypatch.setitem(sys.modules, "enhancer.system", fake)

    requirement = preflight.check_accelerator()
    assert not requirement.ok
    assert not requirement.blocking, "a slow render is still a render"
    assert "slower" in requirement.detail.lower()


def test_detection_failing_never_blocks_startup(monkeypatch):
    import sys
    import types

    def explode(**_kwargs):
        raise RuntimeError("driver on fire")

    monkeypatch.setitem(sys.modules, "enhancer.system",
                        types.SimpleNamespace(detect=explode))
    assert preflight.check_accelerator().ok


def test_a_full_disk_is_reported_before_the_render_not_during(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil, "disk_usage",
        lambda _p: shutil._ntuple_diskusage(total=10 ** 12, used=10 ** 12, free=10 ** 8),
    )
    requirement = preflight.check_disk_space(tmp_path)

    assert not requirement.ok
    assert "0.1 GB free" in requirement.detail


def test_a_tight_disk_warns_without_refusing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil, "disk_usage",
        lambda _p: shutil._ntuple_diskusage(
            total=10 ** 12, used=0, free=8 * 1024 ** 3
        ),
    )
    requirement = preflight.check_disk_space(tmp_path)

    assert requirement.ok
    assert not requirement.essential
    assert "not a feature" in requirement.detail


def test_disk_space_of_a_path_that_does_not_exist_yet(tmp_path):
    """The output file is named before it is created."""
    requirement = preflight.check_disk_space(tmp_path / "deep" / "not" / "there.mkv")
    assert requirement.ok or not requirement.essential


def test_missing_interpolation_weights_only_cost_that_feature(monkeypatch):
    monkeypatch.setattr("enhancer.cli.rife_weights", lambda: [])
    requirement = preflight.check_interpolation()

    assert not requirement.ok
    assert not requirement.blocking
    assert "everything else works" in requirement.detail.lower()


def test_check_all_puts_the_essentials_first():
    names = [r.name for r in preflight.check_all("models/custom")]
    assert names[:2] == ["ffmpeg", "ffprobe"]


def test_nothing_installed_lists_every_blocker(no_ffmpeg, tmp_path):
    blocking = preflight.blocking(preflight.check_all(tmp_path / "none"))
    assert {r.name for r in blocking} == {"ffmpeg", "ffprobe", "Models"}


def test_the_summary_includes_the_fix_for_what_is_missing(no_ffmpeg, tmp_path):
    text = preflight.summary(preflight.check_all(tmp_path / "none"))
    assert "MISSING" in text
    assert "ffmpeg.org" in text or "brew" in text or "apt" in text


def test_no_check_ever_raises(tmp_path, monkeypatch):
    """A prerequisite check that crashes is worse than the missing tool."""
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(
        preflight.shutil, "disk_usage",
        lambda _p: (_ for _ in ()).throw(OSError("no such volume")),
    )
    for requirement in preflight.check_all(Path("/nonexistent"), Path("/nonexistent")):
        assert isinstance(requirement.name, str)
        assert isinstance(requirement.ok, bool)
