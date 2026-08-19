"""T105 — auto-configuration. One or more tests per acceptance criterion in
``DOCS/TASKS2.md`` § T105. Rung B (``validate/autoconfig.py``) is the physics
gate — running 5000-step cases against real fluids — and is not duplicated
here; these tests are the fast, no-simulation checks: the ``why`` contract,
guardrail enforcement, quality-level monotonicity, and the structure of
``Unrepresentable``.
"""

from __future__ import annotations

import numpy as np
import pytest

from flow.autoconfig import (
    QUALITY_CELLS,
    QUALITY_LEVELS,
    Plan,
    Suggestion,
    TAU_FLOOR,
    Unrepresentable,
    _plan_field_names,
    plan,
)
from flow.fluids import fluid
from lbm.geometry import circle
from lbm.units import BLUFF_BODY_SPEEDUP, U_LATTICE_MAX

DISC = circle(80, 80, 40.0, 40.0, 20.0)

#: A case comfortably representable at every quality level: Re ~ 80 at
#: quality="fast" (cells_per_length=30) keeps tau well above TAU_FLOOR
#: (needed: N >= Re * (TAU_FLOOR - 0.5) / (3 * U) ~ 21.3).
EASY = dict(fluid=fluid("water"), speed="0.004 m/s", size="0.02 m", mask=DISC)

#: Re ~ 2e6 — DOCS/STATE1.md D-038's own refusal case, air past a 1.5 m body
#: at 20 m/s. Comfortably beyond any resolution this project runs.
IMPOSSIBLE = dict(fluid=fluid("air"), speed="20 m/s", size="1.5 m", mask=DISC)


# ---------------------------------------------------------------------------
# "every field has a why entry"
# ---------------------------------------------------------------------------


def test_every_plan_field_has_a_why_entry():
    p = plan(**EASY, quality="balanced")
    for name in _plan_field_names():
        assert name in p.why, f"Plan.{name} has no why entry"
        assert isinstance(p.why[name], str) and p.why[name].strip(), (
            f"Plan.why[{name!r}] is empty"
        )


def test_why_has_no_entries_for_non_fields():
    """The flip side: why should not carry keys nobody asked to explain."""
    p = plan(**EASY, quality="balanced")
    assert set(p.why) == set(_plan_field_names())


# ---------------------------------------------------------------------------
# Guardrails, each citing its decision
# ---------------------------------------------------------------------------


def test_tau_guardrail_cites_its_decision():
    with pytest.raises(Unrepresentable) as excinfo:
        plan(**IMPOSSIBLE, quality="accurate")
    exc = excinfo.value
    assert exc.quantity == "tau"
    assert exc.value <= TAU_FLOOR
    assert "D-029" in exc.reason
    assert len(exc.suggestions) >= 1
    for s in exc.suggestions:
        assert isinstance(s, Suggestion)


def test_tau_guardrail_suggestions_actually_fix_the_case():
    """D-045: a suggestion is a testable claim. Feed it back through plan()."""
    with pytest.raises(Unrepresentable) as excinfo:
        plan(**IMPOSSIBLE, quality="accurate")
    for suggestion in excinfo.value.suggestions:
        case = dict(IMPOSSIBLE)
        if suggestion.change == "speed":
            case["speed"] = suggestion.value
        elif suggestion.change == "size":
            case["size"] = suggestion.value
        elif suggestion.change == "quality":
            fixed = plan(**IMPOSSIBLE, quality=suggestion.value)
            assert fixed.tau > TAU_FLOOR
            continue
        else:  # pragma: no cover - guards a future Suggestion.change value
            pytest.fail(f"unhandled suggestion.change {suggestion.change!r}")
        fixed = plan(**case, quality="accurate")
        assert fixed.tau > TAU_FLOOR


def test_u_lattice_guardrail_cites_constraint_3_and_d032():
    p = plan(**EASY, quality="balanced")
    peak = p.u_lattice * BLUFF_BODY_SPEEDUP
    assert peak < U_LATTICE_MAX
    assert "constraint 3" in p.why["u_lattice"]
    assert "D-032" in p.why["u_lattice"]


def test_blockage_and_downstream_guardrails_hold_by_construction():
    p = plan(**EASY, quality="balanced")
    ny, nx = p.domain
    blockage = p.cells_per_length / ny
    assert blockage < 0.10
    assert "constraint 12" in p.why["domain"]
    assert "D-019" in p.why["domain"]


def test_thickness_guardrail_refuses_a_disproportionately_thin_mask():
    # A vertical plate, 40 cells of cross-stream extent (its "D") but only one
    # cell wide throughout -- thickness scales with cells_per_length / 40,
    # which stays under the 3-cell floor at every quality level, unlike a
    # disc whose thickness scales with its own diameter.
    thin = np.zeros((200, 200), dtype=bool)
    thin[80:120, 100] = True
    with pytest.raises(Unrepresentable) as excinfo:
        plan(fluid=fluid("water"), speed="0.001 m/s", size="0.02 m", mask=thin, quality="fast")
    assert excinfo.value.quantity.startswith("min solid thickness")
    assert "D-017" in excinfo.value.reason


def test_empty_mask_is_unrepresentable():
    empty = np.zeros((20, 20), dtype=bool)
    with pytest.raises(Unrepresentable):
        plan(fluid=fluid("air"), speed="1 m/s", size="1 m", mask=empty, quality="fast")


# ---------------------------------------------------------------------------
# Quality levels: monotonic resolution and wall-clock estimate
# ---------------------------------------------------------------------------


def test_quality_levels_are_a_strict_refinement():
    plans = {q: plan(**EASY, quality=q) for q in QUALITY_LEVELS}
    cells = [plans[q].cells_per_length for q in QUALITY_LEVELS]
    assert cells == sorted(cells) and len(set(cells)) == len(cells), (
        "cells_per_length must strictly increase fast -> balanced -> accurate"
    )
    estimates = [plans[q].estimated_seconds("numpy") for q in QUALITY_LEVELS]
    assert estimates == sorted(estimates) and len(set(estimates)) == len(estimates), (
        "estimated_seconds must strictly increase fast -> balanced -> accurate"
    )
    steps = [plans[q].steps for q in QUALITY_LEVELS]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)


def test_quality_levels_share_u_lattice():
    """The one thing that does *not* change with quality (CLAUDE.md constraint 3)."""
    plans = {q: plan(**EASY, quality=q) for q in QUALITY_LEVELS}
    u_values = {plans[q].u_lattice for q in QUALITY_LEVELS}
    assert len(u_values) == 1


def test_unknown_quality_raises_value_error():
    with pytest.raises(ValueError):
        plan(**EASY, quality="ludicrous")


# ---------------------------------------------------------------------------
# estimated_seconds
# ---------------------------------------------------------------------------


def test_estimated_seconds_is_positive_and_backend_sensitive():
    p = plan(**EASY, quality="fast")
    numpy_est = p.estimated_seconds("numpy")
    warp_est = p.estimated_seconds("warp")
    assert numpy_est > 0.0
    assert warp_est > 0.0
    assert warp_est < numpy_est, "warp's measured rate is faster at every grid in the table"


def test_estimated_seconds_rejects_unknown_backend():
    p = plan(**EASY, quality="fast")
    with pytest.raises(ValueError):
        p.estimated_seconds("gpu-of-the-future")


# ---------------------------------------------------------------------------
# Unrepresentable's shape
# ---------------------------------------------------------------------------


def test_unrepresentable_carries_structured_fields_not_just_a_string():
    with pytest.raises(Unrepresentable) as excinfo:
        plan(**IMPOSSIBLE, quality="accurate")
    exc = excinfo.value
    assert isinstance(exc.reason, str) and exc.reason
    assert isinstance(exc.quantity, str) and exc.quantity
    assert isinstance(exc.value, float)
    assert isinstance(exc.limit, float)
    assert isinstance(exc.suggestions, list) and exc.suggestions
    for s in exc.suggestions:
        assert isinstance(s.change, str) and s.change
        assert isinstance(s.note, str) and s.note


# ---------------------------------------------------------------------------
# Reynolds reproduction (the analytic half; Rung B checks it through a run)
# ---------------------------------------------------------------------------


def test_re_matches_the_physical_inputs():
    p = plan(**EASY, quality="balanced")
    f = EASY["fluid"]
    from flow.quantity import parse, SPEED, LENGTH

    u_phys = parse(EASY["speed"], expect=SPEED).si
    l_phys = parse(EASY["size"], expect=LENGTH).si
    expected_re = u_phys * l_phys / f.nu.si
    assert abs(p.Re - expected_re) / expected_re < 1e-9
