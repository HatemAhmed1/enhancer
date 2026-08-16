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
    from enhancer.cli import main as cli_main
    from enhancer.window import launch

    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])
    return launch([])


if __name__ == "__main__":
    sys.exit(main())
