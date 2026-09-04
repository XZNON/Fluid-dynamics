"""``fengdong`` — the console entry point (T205), and ``python -m fengdong``.

``DOCS/TASKS3.md`` § T205: *"a ``fengdong/`` package skeleton whose ``main``
prints a version and exits, so the entry point is real before the app
exists."* ``pyproject.toml`` binds the ``fengdong`` command to :func:`main`
here, and Rung I (``validate/install.py``) runs ``fengdong --version`` from a
fresh venv to prove the binding survives the wheel.

What this does today: prints ``fengdong <version>`` and exits 0. What it will
do from T207: open the window (``DOCS/IDEA4.md`` § What Phase 2 is,
concretely). The ``--version`` flag is kept when that happens, because it is
what Rung I asks for and what a person types to check an install worked.

Deliberately **no** ``flow`` or ``lbm`` import at module scope. Constraint 17
permits it, but ``fengdong --version`` should answer in milliseconds and
should answer even on a machine where numpy is broken — that is precisely the
machine on which the version is worth knowing. The ``__name__`` guard is not
decoration: ``tests/test_packaging.py`` imports this module to check the entry
point resolves, and without the guard collection would run the command.
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
        :func:`version_line` and returns 0; with no arguments it prints the
        same line and a note that the window lands with T207.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "FengDong - drop a picture on a window and watch the flow. "
            "The window is T207; today this command reports its version."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the installed version and exit",
    )
    args = parser.parse_args(argv)

    print(version_line())
    if not args.version:
        print(
            "the window is not built yet (DOCS/TASKS3.md T207); the product "
            "command today is `python -m flow`."
        )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
