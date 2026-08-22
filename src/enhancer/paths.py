"""Where the application's own files live, however it was started.

Model folders were addressed as the relative path `models/custom`, which means
"relative to whatever directory the user happened to be in". Running the
packaged executable from anywhere but its own folder reported no models
installed, and `models --get` downloaded into the shell's working directory.
The README's promise that models are read from beside the executable was only
true by coincidence.

A frozen build and a source checkout put that folder in different places, so
the answer is worked out once, here.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODELS = "models"
CUSTOM = "custom"
RIFE = "rife"


def is_frozen() -> bool:
    """True inside a PyInstaller build."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """The folder the application was installed into.

    Frozen: beside the executable. PyInstaller's one-folder layout puts the
    code in `_internal`, so anchoring on `__file__` there would point at the
    wrong place — it has to be the executable.

    From source: the repository root, two levels above this file
    (`<root>/src/enhancer/paths.py`).
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def models_root() -> Path:
    """`models/` beside the application, unless one sits in the current folder.

    The local override is deliberate: it lets somebody keep a project's models
    with the project. It only wins when it actually exists, so it cannot
    silently shadow the installed set with an empty folder.
    """
    local = Path.cwd() / MODELS
    if local.is_dir():
        return local
    return app_dir() / MODELS


def custom_models_dir() -> Path:
    return models_root() / CUSTOM


def rife_models_dir() -> Path:
    return models_root() / RIFE


def ensure_models_dirs() -> Path:
    """Create the model folders if they are missing, and return the custom one.

    A fresh copy of the packaged build ships no `models` folder, so there was
    nowhere for a download to land and nothing for the user to drop a file
    into. Creating them is cheap and makes the folder self-explanatory.
    """
    custom = custom_models_dir()
    try:
        custom.mkdir(parents=True, exist_ok=True)
        rife_models_dir().mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only install directory is not a reason to refuse to start.
        pass
    return custom
