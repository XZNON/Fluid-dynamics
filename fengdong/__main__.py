"""``fengdong`` — the console entry point (T205), and ``python -m fengdong``.

``DOCS/TASKS3.md`` § T205: *"a ``fengdong/`` package skeleton whose ``main``
prints a version and exits, so the entry point is real before the app
exists."* ``pyproject.toml`` binds the ``fengdong`` command to :func:`main`
here, and Rung I (``validate/install.py``) runs ``fengdong --version`` from a
fresh venv to prove the binding survives the wheel.

What this does: ``--version`` prints ``fengdong <version>`` and exits 0;
otherwise it opens the window (T207, ``DOCS/IDEA4.md`` § What Phase 2 is,
concretely) and returns when the window is closed. ``--version`` is answered
**first**, because it is what Rung I asks for and what a person types to check
an install worked.

Deliberately **no** ``flow``, ``lbm``, ``pygame`` or :mod:`fengdong.app` import
at module scope — the window is imported *inside* :func:`main`, after the
arguments are parsed. Constraint 17 permits the import; constraint 20 is why
it is deferred: ``fengdong --version`` should answer in milliseconds and should
answer even on a machine where numpy is broken, which is precisely the machine
on which the version is worth knowing (``tests/test_packaging.py`` and
``tests/test_widgets.py`` both assert it in a subprocess). The ``__name__``
guard is not decoration: ``tests/test_packaging.py`` imports this module to
check the entry point resolves, and without the guard collection would run
the command.
"""

from __future__ import annotations

import argparse
import sys

from fengdong import __version__

#: The program name, which is also the distribution name and the title bar
#: (**D-083**).
PROG: str = "fengdong"


def version_line() -> str:
    """The one line ``fengdong --version`` prints: ``"fengdong 0.2.0"``."""
    return f"{PROG} {__version__}"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``fengdong`` command.

    Args:
        argv: the command line without the program name; ``None`` reads
            :data:`sys.argv`, as ``argparse`` does.

    Returns:
        Process exit code — ``0`` on success. ``--version`` prints
        :func:`version_line` and returns 0 without importing the window;
        otherwise the window is opened and its exit code returned.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "FengDong - drop a picture on a window and watch the flow. "
            "With no arguments, opens the window."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the installed version and exit",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "warp"),
        default="numpy",
        help=(
            "which backend the plan's wall-clock estimate is for, and which "
            "will run the simulation (default numpy, the reference; warp "
            "needs the [gpu] extra) -- the same flag python -m flow takes"
        ),
    )
    args = parser.parse_args(argv)

    if args.version:
        print(version_line())
        sys.stdout.flush()
        return 0

    # Imported here and not above: this pulls in pygame, numpy and flow, none
    # of which --version needs (constraint 20; tests/test_packaging.py).
    from fengdong.app import App

    return App(backend=args.backend).run()


if __name__ == "__main__":
    raise SystemExit(main())
