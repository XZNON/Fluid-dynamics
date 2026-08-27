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

What exists: physical quantities and the fluid library (T104); the chooser,
``flow/autoconfig.py`` (T105); the refusals and the live divergence probe,
``flow/diagnose.py`` (T106); geometry preparation, ``flow/prepare.py`` (T107);
and the front door over all four — ``flow/case.py``'s :class:`~flow.case.Case`
and ``flow/report.py``'s :class:`~flow.report.Result` (T108). The CLI over
:class:`~flow.case.Case` is T109 (``DOCS/TASKS2.md``).

The three lines the phase exists to make run (``DOCS/IDEA3.md`` § What Phase 1
is, concretely)::

    case = flow.Case.from_image("wing.png", fluid="air", speed="5 m/s", size="10 cm")
    case.explain()
    result = case.run(record="wake.mp4")
"""

from __future__ import annotations

from flow.case import Case
from flow.autoconfig import (
    QUALITY_LEVELS,
    Plan,
    Suggestion,
    Unrepresentable,
    plan,
)
from flow.diagnose import (
    EXAMPLE_MASK,
    REFUSAL_CLASSES,
    SUGGESTION_ORDER,
    Diverging,
    Monitor,
    apply_suggestion,
    classify,
    explain,
    suggest,
)
from flow.fluids import FLUIDS, Fluid, fluid, known_fluids
from flow.prepare import (
    MIN_BODY_CELLS,
    MIN_THICKNESS_CELLS,
    REPAIRS,
    THIN_BRANCH_DEPTH,
    VERDICTS,
    Fix,
    Prepared,
    apply_fix,
    measure,
    prepare,
    thin_branch_depth,
)
from flow.report import Result
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
    "Case",
    "Result",
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
    "Plan",
    "Suggestion",
    "Unrepresentable",
    "plan",
    "QUALITY_LEVELS",
    "Diverging",
    "Monitor",
    "REFUSAL_CLASSES",
    "SUGGESTION_ORDER",
    "EXAMPLE_MASK",
    "apply_suggestion",
    "classify",
    "explain",
    "suggest",
    "Prepared",
    "Fix",
    "prepare",
    "measure",
    "thin_branch_depth",
    "apply_fix",
    "REPAIRS",
    "VERDICTS",
    "MIN_BODY_CELLS",
    "MIN_THICKNESS_CELLS",
    "THIN_BRANCH_DEPTH",
]
