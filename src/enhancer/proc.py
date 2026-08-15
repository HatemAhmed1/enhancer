"""Subprocess helpers.

Windows opens a console window for every child process started from a program
that has no console of its own. The desktop app is launched with pythonw, so
without this a render flashes a black window on screen for each ffmpeg call —
several times per segment.
"""

from __future__ import annotations

import subprocess
import sys

# Present only on Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def hidden_flags() -> int:
    """Creation flags that suppress a child console window on Windows."""
    return _NO_WINDOW if sys.platform == "win32" else 0


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """`subprocess.run` that never flashes a console window."""
    kwargs.setdefault("creationflags", hidden_flags())
    return subprocess.run(cmd, **kwargs)


def popen(cmd: list[str], **kwargs) -> subprocess.Popen:
    """`subprocess.Popen` that never flashes a console window."""
    kwargs.setdefault("creationflags", hidden_flags())
    return subprocess.Popen(cmd, **kwargs)
