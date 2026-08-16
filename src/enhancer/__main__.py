"""Entry point for the packaged application.

Opens the window directly. The command line remains available through
`python -m enhancer.cli`, and the packaged build accepts the same subcommands
if any are passed.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # Required before anything else in a frozen build: without it, any child
    # process re-runs the whole program instead of the worker, which on
    # Windows means the application launches itself repeatedly.
    multiprocessing.freeze_support()

    # Absolute imports, not relative. A packaged build runs this file as a
    # top-level script with no parent package, so `from .cli import ...` fails
    # with "attempted relative import with no known parent package" — and only
    # in the built executable, never when run as `python -m enhancer`.
    if len(sys.argv) > 1:
        from enhancer.cli import main as cli_main

        return cli_main(sys.argv[1:])

    # Show something within a second. Loading the graphics libraries takes
    # tens of seconds in a packaged build, and an application that puts nothing
    # on screen for that long reads as broken rather than busy.
    from enhancer.splash import run_with_splash

    return run_with_splash()


if __name__ == "__main__":
    sys.exit(main())
