"""``python -m flow`` (T109).

The entry point ``DOCS/TASKS2.md`` § T109 names. Everything is in
:mod:`flow.cli`; this file holds nothing of its own. Two details are
deliberate:

* ``from flow.cli import main`` reaches the **package's** copy, so the
  :class:`lbm.runner.Sink` subclasses ``lbm.record`` and ``lbm.render``
  registered are the ones every ``isinstance`` check in the process compares
  against — the double-import footnote at the foot of ``lbm/runner.py``, whose
  shape this follows.
* The ``__name__`` guard is not decoration. ``tests/test_flow_package.py``
  imports every module under ``flow.`` to scan its signatures for constraint
  13, and ``flow.__main__`` is one of them; without the guard, collecting the
  test suite would run the CLI.
"""

from __future__ import annotations

from flow.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
