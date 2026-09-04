"""``fengdong`` (风洞, *wind tunnel*) — the desktop application, and the distribution.

Implements ``DOCS/IDEA4.md`` § What Phase 2 is, concretely — the ``fengdong/``
box: *"a window, a drop target, a live view, a numbers panel"* — and § The five
things Phase 2 must get right (5): *"One command installs it, on a machine that
is not ours."* This package is what ``pip install fengdong`` installs and what
the ``fengdong`` command runs (**D-083**).

**T205 shipped the skeleton, T206 the widgets.** :func:`fengdong.__main__.main`
prints a version and exits, so the console entry point is real before the app
is; :mod:`fengdong.widgets` is the closed widget set (**D-083**), driven
headless and imported by nothing here yet — the window is T207, the live view
T208 and the drop T209 (``DOCS/PLAN3.md`` § Session map). Nothing here opens
a display.

Two rules govern everything that lands here, and both have tests:

* **Constraint 17** — ``fengdong/`` may import ``flow/``; ``flow/`` may
  **never** import ``fengdong/``. The app is a view, not a second brain: every
  solver parameter it displays comes from :func:`flow.autoconfig.plan`
  (``DOCS/IDEA4.md`` § The five things Phase 2 must get right (3)).
  ``tests/test_packaging.py`` asserts the direction by scanning the source, in
  the shape of the constraint-15 test, and Rung I (``validate/install.py``)
  asserts it again *inside the installed wheel*.
* **Constraint 13** — no lattice quantity in a widget. ``tau``, lattice ``U``,
  ``steps_per_frame``, cell counts and ``Cs`` are planned and printed by
  ``flow/``, never typed into ``fengdong/``.

The import name and the distribution name are the same, ``fengdong``, while
the product library underneath is ``flow`` — because ``flow`` is taken on PyPI
and ``fengdong`` is free (**D-083**). The version below is the single source:
``pyproject.toml`` reads it as a dynamic field.
"""

from __future__ import annotations

#: The distribution version, read by ``pyproject.toml`` (``dynamic = ["version"]``)
#: and printed by ``fengdong --version``. ``0.2.x`` is Phase 2; nothing is
#: published to PyPI in T205 (**Q-204**), so this is a build number rather than
#: a release number until the user decides otherwise.
__version__: str = "0.2.0"

__all__ = ["__version__"]
