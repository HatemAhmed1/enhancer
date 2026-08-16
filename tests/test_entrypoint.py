"""The packaged application's entry point.

Mistakes here only show up in the built executable, where they are far more
awkward to diagnose than in a test.
"""

import subprocess
import sys

import pytest


def test_entry_point_is_importable():
    import enhancer.__main__ as entry

    assert callable(entry.main)


def test_entry_point_calls_freeze_support(monkeypatch):
    """Without this, a frozen build relaunches itself instead of starting a
    worker process, so the application opens over and over."""
    import enhancer.__main__ as entry

    called = []
    monkeypatch.setattr(entry.multiprocessing, "freeze_support",
                        lambda: called.append(True))
    monkeypatch.setattr(sys, "argv", ["enhancer", "models", "--dir", "nowhere"])
    entry.main()
    assert called, "freeze_support was not called"


def test_arguments_are_passed_through_to_the_command_line(monkeypatch, tmp_path, capsys):
    """The packaged build must still accept the subcommands."""
    import enhancer.__main__ as entry

    monkeypatch.setattr(sys, "argv", ["enhancer", "models", "--dir", str(tmp_path)])
    entry.main()
    assert "Catalogue" in capsys.readouterr().out


def test_no_arguments_opens_the_window(monkeypatch):
    """With no arguments the window opens, by way of the startup splash.

    The splash exists because loading the graphics libraries takes tens of
    seconds in a packaged build, and an application showing nothing for that
    long reads as broken.
    """
    import enhancer.__main__ as entry

    opened = []
    monkeypatch.setattr(sys, "argv", ["enhancer"])
    monkeypatch.setattr("enhancer.splash.run_with_splash",
                        lambda: opened.append(True) or 0)
    assert entry.main() == 0
    assert opened == [True]


def test_module_runs_as_a_script():
    """python -m enhancer must reach the same entry point."""
    result = subprocess.run(
        [sys.executable, "-m", "enhancer", "models", "--dir", "definitely-absent"],
        capture_output=True, text=True, timeout=120,
    )
    assert "Catalogue" in result.stdout or "(none)" in result.stdout


def test_entry_point_uses_absolute_imports():
    """A packaged build runs this file as a top-level script.

    Relative imports then fail with "attempted relative import with no known
    parent package" — but only in the built executable. Running it as
    `python -m enhancer` works fine, so nothing else here catches it.
    """
    import pathlib

    source = pathlib.Path("src/enhancer/__main__.py").read_text(encoding="utf-8")
    offenders = [line.strip() for line in source.splitlines()
                 if line.strip().startswith("from .")]
    assert not offenders, f"relative imports break the packaged build: {offenders}"


def test_build_excludes_only_whole_packages():
    """Excluding a submodule of a package that is kept breaks that package.

    torchvision.models.__init__ imports .detection by name; excluding it turned
    a working import into a circular-import failure at startup, visible only in
    the packaged application.
    """
    import ast
    import pathlib

    spec = pathlib.Path("enhancer.spec").read_text(encoding="utf-8")
    tree = ast.parse(spec)
    excludes: list[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "excludes" for t in node.targets)):
            excludes = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]

    assert excludes, "could not read the exclude list from the build definition"
    kept_roots = {"torch", "torchvision", "spandrel", "numpy", "yt_dlp"}
    for name in excludes:
        root = name.split(".")[0]
        assert root not in kept_roots, (
            f"{name!r} excludes part of {root!r}, which the application needs"
        )
