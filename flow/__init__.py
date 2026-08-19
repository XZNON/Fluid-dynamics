"""``flow`` — the product layer above the solver.

**D-042**: ``flow/`` is a new top-level package beside ``lbm/``. ``flow/`` may
import ``lbm/``; ``lbm/`` may **never** import ``flow/`` (Phase 1 constraint
15), and ``tests/test_flow_package.py`` asserts it by scanning the source. That
one-directional import is what makes the Phase 3 XLB swap a substitution rather
than a rewrite.

Two rules govern everything that lands here, and both have tests:

* **Constraint 13** — no lattice quantity in any public signature. No ``tau``,
  no lattice ``U``, no ``steps_per_frame``, no cell counts. The inputs are a
  picture, a fluid, a speed, a size; everything else is derived and printed.
  :mod:`flow.quantity` is the outer face of that boundary and ``lbm/units.py``
  is the inner one.
* **Constraint 10** — ``flow/`` colours nothing. There is one ``render()`` and
  it lives in ``lbm/render.py``.

What exists so far (T104): physical quantities and the fluid library. The
chooser (``flow/autoconfig.py``, T105), the refusals (``flow/diagnose.py``,
T106), geometry preparation (``flow/prepare.py``, T107) and the
``Case``/``Result`` API (T108) land in the tasks named beside them —
``DOCS/TASKS2.md``.
"""

from __future__ import annotations

from flow.fluids import FLUIDS, Fluid, fluid, known_fluids
from flow.quantity import (
    DENSITY,
    DIMENSIONS,
    LENGTH,
    SI_UNIT,
    SPEED,
    TEMPERATURE,
    TIME,
    VISCOSITY,
    Quantity,
    parse,
    to_si,
    units_for,
)

__all__ = [
    "Quantity",
    "parse",
    "to_si",
    "units_for",
    "LENGTH",
    "SPEED",
    "TIME",
    "VISCOSITY",
    "TEMPERATURE",
    "DENSITY",
    "DIMENSIONS",
    "SI_UNIT",
    "Fluid",
    "FLUIDS",
    "fluid",
    "known_fluids",
]
