"""T106 — ``flow/diagnose.py``: refusals in prose, fixes that run, divergence caught.

The rung (``validate/refusals.py``) is what proves a suggestion *works*; these
tests are what pin the things a rung cannot see — the wording contract of
:func:`flow.diagnose.explain`, the labelling constraint 16 asks for, and
:class:`flow.diagnose.Monitor`'s behaviour on cases small enough to run in a
unit test.

``DOCS/TASKS2.md`` § T106 · ``DOCS/IDEA3.md`` § The five things Phase 1 must
get right (2) · **D-045**.
"""

from __future__ import annotations

import numpy as np
import pytest

from flow.autoconfig import Suggestion, Unrepresentable, plan
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
from lbm.geometry import circle
from lbm.runner import Sim, SimConfig
from validate.refusals import CASES, BASE_REQUEST, _planargs, _synthetic

#: Words that must never appear in the first paragraph of a refusal. The
#: acceptance criterion names ``tau``, the lattice velocity and cell counts;
#: this list is a superset, because the criterion's point is that the *reader*
#: has never heard of any of them (``DOCS/TASKS2.md`` § T106 Notes).
LATTICE_WORDS = (
    "tau",
    "lattice",
    "cell",
    "cells",
    "timestep",
    "time step",
    "relaxation",
    "reynolds",
    "mach",
    "grid",
    "d-0",
    "constraint",
)


def _refusal(request: dict) -> Unrepresentable:
    """The refusal a request produces. Fails loudly if it plans instead."""
    try:
        plan(**_planargs(request))
    except Unrepresentable as exc:
        return exc
    raise AssertionError(f"{request} planned, but this test needs it to refuse")


def _first_paragraph(text: str) -> str:
    return text.split("\n\n", 1)[0]


# ---------------------------------------------------------------------------
# explain() — the wording contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.refusal_class + "-" + c.name[:20])
def test_first_paragraph_of_a_refusal_names_no_lattice_quantity(case):
    text = explain(_refusal(case.request))
    first = _first_paragraph(text).lower()
    leaks = [w for w in LATTICE_WORDS if w in first]
    assert not leaks, f"{case.name}: first paragraph names {leaks}\n{first}"


@pytest.mark.parametrize("refusal_class", ("speed_ceiling", "blockage"))
def test_the_unreachable_classes_are_explained_too(refusal_class):
    """A guardrail nothing can reach today still has to have prose ready."""
    text = explain(_synthetic(refusal_class))
    first = _first_paragraph(text).lower()
    assert not [w for w in LATTICE_WORDS if w in first], first
    assert "Details" in text


def test_the_numbers_are_available_in_a_second_section():
    exc = _refusal(CASES[0].request)
    text = explain(exc)
    head, _, details = text.partition("Details")
    assert details, "explain() must carry a details section"
    assert "tau" in details and "0.54" in details
    assert "tau" not in _first_paragraph(head)


def test_explain_restates_the_request_in_the_users_own_units():
    exc = _refusal(CASES[0].request)
    text = explain(
        exc, request={"fluid": "air", "speed": "20 m/s", "size": "1.5 m"}
    )
    first = _first_paragraph(text)
    assert first.startswith("Air at 20 m/s past a body 1.5 m across")


def test_explain_without_a_request_still_reads_as_a_sentence():
    text = explain(_refusal(CASES[0].request))
    assert _first_paragraph(text).startswith("This case is far more energetic")


# ---------------------------------------------------------------------------
# The D-038 case, pinned. A reword is a deliberate edit.
# ---------------------------------------------------------------------------

D038_GOLDEN = (
    "Air at 20 m/s past a body 1.5 m across is far more energetic than this "
    "simulator can represent. At that combination of speed, size and fluid the "
    "flow is turbulent, and this tool models smooth, orderly flow only -- it "
    "has no turbulence model. Running it anyway would produce a "
    "convincing-looking video of the wrong answer, so it refuses instead.\n"
    "\n"
    "What would work\n"
    "  1. Run it slower -- about 0.00121 m/s -- at the same size and in the "
    "same fluid. This is a different flow from the one you asked for -- not "
    "your case.\n"
    "  2. Run a smaller body -- about 9.1e-05 m across -- at the same speed "
    "and in the same fluid. This is a different flow from the one you asked "
    "for -- not your case.\n"
    "\n"
    "Details\n"
    "  refused because: tau would sit at or below the bluff-body stability "
    "floor of 0.54 -- the strictest of the project's three tau floors (D-029; "
    "see D-032 for the generic 0.51 floor and D-036 for Rung 3's 0.537)\n"
    "  tau = 0.500003 (limit 0.54)"
)


def test_the_d038_case_reads_exactly_this(capsys):
    """**D-038**: air, 20 m/s, a 1.5 m body -- Re 2e6, and the refusal is correct.

    Pinned in full so a reword is a deliberate edit rather than a drift. The
    same text is what ``validate/refusals.py`` § 2 prints.
    """
    exc = _refusal(CASES[0].request)
    text = explain(
        exc, request={"fluid": "air", "speed": "20 m/s", "size": "1.5 m"}
    )
    assert text == D038_GOLDEN


def test_the_d038_refusal_is_ascii():
    """A Windows console at its default codepage mojibakes an em dash (T104)."""
    D038_GOLDEN.encode("ascii")


# ---------------------------------------------------------------------------
# suggest() — one per class, physical, labelled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name[:24])
def test_every_refusal_class_gets_at_least_one_suggestion(case):
    found = suggest(**case.request)
    assert found, f"{case.name}: refused with no way forward (D-045)"
    for s in found:
        assert s.change in SUGGESTION_ORDER
        assert s.note.strip(), f"{s.change} carries no sentence"
        assert s.note.endswith("."), s.note


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name[:24])
def test_a_suggestion_that_changes_the_flow_says_so(case):
    """Constraint 16, one layer above a `Result`: never a silent substitution."""
    for s in suggest(**case.request):
        if s.change in ("speed", "size", "fluid"):
            assert "not your case" in s.note.lower(), s.note
        else:
            assert "not your case" not in s.note.lower(), s.note


def test_a_case_that_plans_gets_no_suggestions():
    """Nothing to fix must read as an empty list, not an invented alternative."""
    assert suggest(**BASE_REQUEST) == []


def test_the_case_preserving_fix_is_offered_first():
    """Only resolution keeps the flow the user asked for -- so it ranks first."""
    found = suggest(**CASES[1].request)  # the 3-cell plate
    assert found[0].change == "quality"


def test_a_more_viscous_fluid_is_offered_when_one_actually_works():
    """The fluid suggestion is checked before it is offered, never invented."""
    # Water at 0.5 m/s past a 0.2 m body is Re 1e5 -- refused. Honey (nu
    # 7.042e-3) is ~7000x water's viscosity and brings the same geometry back
    # into range.
    request = dict(
        fluid="water", speed="0.5 m/s", size="0.2 m", mask=EXAMPLE_MASK,
        quality="balanced",
    )
    found = suggest(**request)
    fluids = [s for s in found if s.change == "fluid"]
    assert fluids, [s.change for s in found]
    fixed = apply_suggestion(fluids[0], **request)
    plan(**fixed)  # the claim, executed


def test_no_fluid_is_invented_for_a_case_no_fluid_can_save():
    """Re 2e6 is beyond every entry in the library -- so none is offered."""
    found = suggest(**CASES[0].request)
    assert not [s for s in found if s.change == "fluid"]


# ---------------------------------------------------------------------------
# apply_suggestion() — the modified request
# ---------------------------------------------------------------------------


def test_apply_substitutes_exactly_one_field():
    request = dict(CASES[1].request)
    top = suggest(**request)[0]
    fixed = apply_suggestion(top, **request)
    assert fixed["quality"] == top.value
    assert fixed["speed"] == request["speed"]
    assert fixed["size"] == request["size"]
    assert fixed["mask"] is request["mask"]


def test_apply_swaps_in_the_worked_example_for_a_mask_refusal():
    request = dict(CASES[3].request)  # the empty picture
    top = suggest(**request)[0]
    assert top.change == "mask"
    fixed = apply_suggestion(top, **request)
    assert fixed["mask"] is EXAMPLE_MASK
    plan(**fixed)


def test_apply_refuses_a_change_it_cannot_make():
    bogus = Suggestion(change="gravity", value="off", note="turn it off.")
    with pytest.raises(ValueError, match="cannot apply"):
        apply_suggestion(bogus, **BASE_REQUEST)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name[:24])
def test_the_top_suggestion_makes_the_case_plan(case):
    """The cheap half of Rung D: it plans. The rung then runs it 2000 steps."""
    top = suggest(**case.request)[0]
    plan(**apply_suggestion(top, **case.request))


# ---------------------------------------------------------------------------
# classify() — and the guard has teeth
# ---------------------------------------------------------------------------


def test_every_refusal_class_is_registered():
    known = {classify(_synthetic(c)) for c in ("speed_ceiling", "blockage")}
    known |= {classify(_refusal(c.request)) for c in CASES}
    assert known == set(REFUSAL_CLASSES)


def test_an_unregistered_refusal_is_loud_rather_than_unexplained():
    """A guard that never fires is not a guard (T104's precedent)."""
    orphan = Unrepresentable(
        reason="something new", quantity="whatever", value=1.0, limit=2.0,
        suggestions=[],
    )
    with pytest.raises(ValueError, match="no explanation registered"):
        classify(orphan)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def _dying_sim(tau: float = 0.6, U: float = 0.25) -> Sim:
    """A small case driven far past the ceiling: dies in a few hundred steps."""
    ny, nx = 32, 96
    solid = circle(ny, nx, nx * 0.25, (ny - 1) / 2.0 + 0.5, 4.0)
    cfg = SimConfig(
        ny=ny, nx=nx, tau=tau, inlet_U=U, profile="uniform", use_inlet=True,
        use_outlet=True, convective_outlet=True, inlet_axis="x",
        check_geometry=False,
    )
    return Sim(cfg, solid)


def _healthy_sim() -> Sim:
    ny, nx = 32, 96
    solid = circle(ny, nx, nx * 0.25, (ny - 1) / 2.0 + 0.5, 4.0)
    cfg = SimConfig(
        ny=ny, nx=nx, tau=0.6, inlet_U=0.05, profile="uniform", use_inlet=True,
        use_outlet=True, convective_outlet=True, inlet_axis="x",
        check_geometry=False,
    )
    return Sim(cfg, solid)


def _run_until_caught(sim: Sim, monitor: Monitor, maxsteps: int) -> Diverging:
    for _ in range(maxsteps):
        sim.step()
        try:
            monitor(sim)
        except Diverging as exc:
            return exc
    raise AssertionError(f"Monitor never fired in {maxsteps} steps")


def test_monitor_catches_an_over_driven_case_and_names_the_cause():
    with pytest.warns(UserWarning):
        sim = _dying_sim()
    exc = _run_until_caught(sim, Monitor(every=25), 1000)
    assert exc.cause == "speed"
    assert exc.step > 0
    assert "faster" in exc.symptom
    assert exc.fix and exc.fix[0].isupper()
    assert set(exc.details) >= {"peak_speed", "mass_drift"}


def test_monitor_blames_the_relaxation_floor_when_that_is_the_real_cause():
    """``DOCS/IDEA2.md`` § Stability row 1: everything else is downstream of it."""
    with pytest.warns(UserWarning):
        sim = _dying_sim(tau=0.51, U=0.25)
    exc = _run_until_caught(sim, Monitor(every=25), 2000)
    assert exc.cause == "relaxation"


def test_monitor_says_nothing_about_a_healthy_run():
    """A tripwire that fires on a good run teaches the user to ignore it."""
    sim = _healthy_sim()
    monitor = Monitor(every=25)
    for _ in range(1000):
        sim.step()
        monitor(sim)
    assert monitor.samples == 40
    assert monitor.peak_speed < 0.1


def test_monitor_is_sampled_not_per_step():
    """Constraint 8: it runs on the physics thread, so it must stay cheap."""
    sim = _healthy_sim()
    monitor = Monitor(every=25)
    for _ in range(24):
        sim.step()
        monitor(sim)
    assert monitor.samples == 0
    sim.step()
    monitor(sim)
    assert monitor.samples == 1


def test_monitor_allocates_its_buffers_once():
    """``CLAUDE.md`` § conventions: never allocate inside the step loop."""
    sim = _healthy_sim()
    monitor = Monitor(every=1)
    sim.step()
    monitor(sim)
    first = monitor._sq
    for _ in range(10):
        sim.step()
        monitor(sim)
    assert monitor._sq is first


def test_monitor_is_a_per_step_probe():
    """**D-025**: ``run(..., per_step=...)`` takes it as-is."""
    from lbm.runner import run

    sim = _healthy_sim()
    monitor = Monitor(every=25)
    run(sim, steps=200, steps_per_frame=50, per_step=monitor, drop=False)
    assert monitor.samples == 8


def test_monitor_rejects_a_nonsense_cadence():
    with pytest.raises(ValueError):
        Monitor(every=0)
    with pytest.raises(ValueError):
        Monitor(over_ceiling_samples=0)


class _FakeSim:
    """The smallest thing :class:`Monitor` reads: ``u``, ``rho``, mask, config.

    Driving the probe by hand is the only way to pin the *sustained* rule: a
    real sim cannot be asked to cross the ceiling for exactly one sample.
    """

    class _Config:
        tau = 0.6

    def __init__(self, speeds: list[float]) -> None:
        self._speeds = speeds
        self.config = _FakeSim._Config()
        self.solid = np.zeros((4, 4), dtype=bool)
        self.step_count = 0

    def host_u(self) -> np.ndarray:
        u = np.zeros((2, 4, 4), dtype=np.float32)
        u[0, 2, 2] = self._speeds[min(self.step_count, len(self._speeds) - 1)]
        return u

    def host_rho(self) -> np.ndarray:
        return np.ones((4, 4), dtype=np.float32)

    def step(self) -> None:
        self.step_count += 1


def test_monitor_ignores_a_single_transient_sample_over_the_ceiling():
    """Measured, not stylistic: a healthy start-up can cross for one sample.

    ``DOCS/STATE2.md`` session 18 records the trace -- the D-029 case crosses
    0.1 briefly around step 155 and recovers, then dies at 1650. A probe that
    fires on the transient is a false-alarm generator, so the tripwire is
    *sustained*: ``over_ceiling_samples`` consecutive samples.
    """
    sim = _FakeSim([0.05, 0.15, 0.05, 0.05, 0.05, 0.05])
    monitor = Monitor(every=1, over_ceiling_samples=3)
    for _ in range(6):
        sim.step()
        monitor(sim)  # must not raise: one crossing, then recovery
    assert monitor.samples == 6


def test_monitor_fires_once_the_crossing_is_sustained():
    sim = _FakeSim([0.05, 0.15, 0.15, 0.15, 0.15])
    monitor = Monitor(every=1, over_ceiling_samples=3)
    with pytest.raises(Diverging) as excinfo:
        for _ in range(5):
            sim.step()
            monitor(sim)
    assert excinfo.value.cause == "speed"
    assert excinfo.value.step == 3  # the third consecutive crossing, not the first


def test_monitor_catches_mass_that_will_not_stay_put():
    """The third mode of ``DOCS/IDEA2.md`` § Stability, driven by hand."""

    class _Leaking(_FakeSim):
        def host_rho(self) -> np.ndarray:
            return np.full((4, 4), 1.0 + 0.02 * self.step_count, dtype=np.float32)

    sim = _Leaking([0.05] * 10)
    monitor = Monitor(every=1, mass_drift=0.01)
    with pytest.raises(Diverging) as excinfo:
        for _ in range(10):
            sim.step()
            monitor(sim)
    assert excinfo.value.cause == "mass"
    assert excinfo.value.details["mass_drift"] > 0.01


def test_diverging_message_carries_the_fix():
    exc = Diverging(
        cause="mass", symptom="the fluid is not staying put", fix="Do this.",
        step=42, details={"peak_speed": 0.2, "mass_drift": 0.5},
    )
    assert "caught at step 42" in str(exc)
    assert "Do this." in str(exc)
