"""Rung D — every refusal names a fix, and the fix actually runs.

``DOCS/IDEA3.md`` § Validation ladder: Rung D's known answer is *"apply the
tool's own suggestion; the resulting case must run"*. That is what makes
**D-045** a policy rather than a wording exercise — a suggestion is a testable
claim, and a claim this script cannot execute is a failing test.

Four sections, each printing its own ``[ok]`` / ``[FAIL]``:

1. **Refusals.** For every class in
   :data:`flow.diagnose.REFUSAL_CLASSES` that :func:`flow.autoconfig.plan` can
   actually be made to raise, take the tool's own **top** suggestion, feed it
   back through :func:`~flow.diagnose.apply_suggestion` and :func:`plan`, and
   run :data:`STEPS_PER_CASE` timesteps. No ``nan``, peak ``|u|`` under the
   constraint-3 ceiling.
2. **The D-038 case**, air at 20 m/s past a 1.5 m body, printed in full. Its
   user-facing text is pinned as a golden string in
   ``tests/test_diagnose.py``, so a reword is a deliberate edit.
3. **Monitor** against the three failure modes of ``DOCS/IDEA2.md``
   § Stability. Each mode is run twice: once bare, to find the step at which it
   produces ``nan``, and once with :class:`~flow.diagnose.Monitor`, to find the
   step at which it is *caught*. Both numbers are printed, with the fraction of
   the run's life the warning bought.
4. **Monitor's cost**, by alternating rounds with only one ``Sim`` resident
   (**D-035**), quoted with the CPU clock and the power state.

What this rung deliberately does **not** do: rasterise the user's own mask. A
plan's ``domain``/``cells_per_length`` is what a suggestion changed, and
turning an arbitrary picture into a grid is Rung C's subject (T107), so every
run here uses the same canonical disc ``validate/autoconfig.py`` holds fixed
for Rung B — stated rather than hidden.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flow.autoconfig import Plan, Suggestion, Unrepresentable, plan
from flow.diagnose import (
    REFUSAL_CLASSES,
    Diverging,
    Monitor,
    apply_suggestion,
    classify,
    explain,
    suggest,
)
from flow.fluids import fluid
from lbm.geometry import circle
from lbm.runner import Sim, SimConfig
from lbm.units import U_LATTICE_MAX
from validate.autoconfig import NOMINAL_MASK, build_solid

#: Timesteps every suggested case is run for. The T106 acceptance criterion's
#: own number.
STEPS_PER_CASE: int = 2000

#: Sampling cadence for the ``nan`` hunt and for :class:`Monitor` in section 3.
#: The two must match, or the two step numbers being compared are not measured
#: on the same clock.
SAMPLE_EVERY: int = 25


# ---------------------------------------------------------------------------
# Section 1 — the refusal classes
# ---------------------------------------------------------------------------


def _plate(width: int) -> np.ndarray:
    """A 40-cell-tall vertical plate ``width`` cells across, in a 200x200 field."""
    m = np.zeros((200, 200), dtype=bool)
    m[80:120, 100 : 100 + width] = True
    return m


EMPTY_MASK: np.ndarray = np.zeros((80, 80), dtype=bool)


@dataclass(frozen=True)
class RefusalCase:
    """One case that must refuse, and the class of refusal it must produce."""

    name: str
    refusal_class: str
    request: dict[str, Any]


#: Every refusal class reachable through :func:`plan`. ``speed_ceiling`` and
#: ``blockage`` are **not** here and that is not an omission: ``flow/autoconfig``
#: fixes the lattice velocity and the domain span by construction (**D-059**),
#: so no physical request reaches either guardrail. They are covered separately
#: below, and the fact that they are unreachable is printed rather than
#: silently skipped.
CASES: tuple[RefusalCase, ...] = (
    RefusalCase(
        name="relaxation - D-038: air, 20 m/s, 1.5 m body (Re 2e6)",
        refusal_class="relaxation",
        request=dict(
            fluid="air", speed="20 m/s", size="1.5 m", mask=NOMINAL_MASK,
            quality="balanced",
        ),
    ),
    RefusalCase(
        name="thickness - a 3-cell plate, fixable by resolution",
        refusal_class="thickness",
        request=dict(
            fluid="water", speed="0.001 m/s", size="0.02 m", mask=_plate(3),
            quality="fast",
        ),
    ),
    RefusalCase(
        name="thickness - a 1-cell plate, no resolution is enough",
        refusal_class="thickness",
        request=dict(
            fluid="water", speed="0.001 m/s", size="0.02 m", mask=_plate(1),
            quality="fast",
        ),
    ),
    RefusalCase(
        name="empty_mask - a picture with no body in it",
        refusal_class="empty_mask",
        request=dict(
            fluid="water", speed="0.005 m/s", size="0.02 m", mask=EMPTY_MASK,
            quality="fast",
        ),
    ),
)

#: A request that plans cleanly, used to prove the two unreachable classes'
#: suggestions are at least executable. Rung B's own accuracy case.
BASE_REQUEST: dict[str, Any] = dict(
    fluid="water", speed="0.005 m/s", size="0.02 m", mask=NOMINAL_MASK,
    quality="fast",
)


def run_plan(plan_: Plan, backend: str, steps: int = STEPS_PER_CASE) -> dict[str, float]:
    """Run ``steps`` timesteps of ``plan_`` on the canonical disc.

    Returns:
        ``{"peak_u": ..., "seconds": ...}``.

    Raises:
        AssertionError: on ``nan``, or a peak velocity at or above the
            constraint-3 ceiling.
    """
    solid = build_solid(plan_)
    ny, nx = plan_.domain
    cfg = SimConfig(
        ny=ny, nx=nx, tau=plan_.tau, inlet_U=plan_.u_lattice, profile="uniform",
        use_inlet=True, use_outlet=True, convective_outlet=True, inlet_axis="x",
        check_geometry=False, backend=backend,
    )
    sim = Sim(cfg, solid)
    start = time.perf_counter()
    for _ in range(steps):
        sim.step()
    f_host = sim.host_f()
    elapsed = time.perf_counter() - start
    assert np.isfinite(f_host).all(), f"nan after {steps} steps"
    u = sim.host_u()
    fluid_cells = ~sim.solid
    peak = float(np.sqrt(u[0][fluid_cells] ** 2 + u[1][fluid_cells] ** 2).max())
    assert peak < U_LATTICE_MAX, f"peak |u| {peak} at/above the ceiling"
    return {"peak_u": peak, "seconds": elapsed}


def _planargs(request: dict[str, Any]) -> dict[str, Any]:
    """``plan`` wants a :class:`~flow.fluids.Fluid`; a request may name one."""
    return {**request, "fluid": fluid(request["fluid"])}


def check_case(case: RefusalCase, backend: str) -> dict[str, Any]:
    """Refuse, suggest, apply the top suggestion, plan again, run.

    Raises:
        AssertionError: if the case does not refuse, refuses as the wrong
            class, offers no suggestion, or offers one that does not fix it.
    """
    try:
        plan(**_planargs(case.request))
    except Unrepresentable as exc:
        got = classify(exc)
        assert got == case.refusal_class, (
            f"{case.name}: refused as {got!r}, expected {case.refusal_class!r}"
        )
    else:  # pragma: no cover - a case that stops refusing is a real regression
        raise AssertionError(f"{case.name}: planned, but it must refuse")

    suggestions = suggest(**case.request)
    assert suggestions, f"{case.name}: refused with no way forward (D-045)"
    top = suggestions[0]
    fixed = apply_suggestion(top, **case.request)
    try:
        fixed_plan = plan(**fixed)
    except Unrepresentable as exc:  # pragma: no cover - the failing-test path
        raise AssertionError(
            f"{case.name}: the tool's own top suggestion ({top.change} -> "
            f"{top.value}) still refuses: {exc}"
        ) from None
    measured = run_plan(fixed_plan, backend)
    return {
        "case": case.name,
        "class": case.refusal_class,
        "suggestions": len(suggestions),
        "top": top,
        "plan": fixed_plan,
        **measured,
    }


def unreachable_classes() -> tuple[str, ...]:
    """Classes no physical request can reach, guarded by construction."""
    reached = {c.refusal_class for c in CASES}
    return tuple(c for c in REFUSAL_CLASSES if c not in reached)


def _synthetic(refusal_class: str) -> Unrepresentable:
    """The refusal ``plan`` would raise for a class nothing can reach.

    Built with the same fields and the same suggestion ``flow/autoconfig.py``
    constructs at that site, so section 1's ``[unreachable]`` rows still
    exercise :func:`explain` and :func:`apply_suggestion` end to end.
    """
    from flow.autoconfig import Suggestion as _S
    from flow.quantity import Quantity

    if refusal_class == "speed_ceiling":
        return Unrepresentable(
            reason=(
                "estimated peak lattice velocity (1.8x free stream, D-032's "
                "BLUFF_BODY_SPEEDUP) at or above the CLAUDE.md constraint 3 "
                "ceiling"
            ),
            quantity="peak |u_lattice|",
            value=0.12,
            limit=U_LATTICE_MAX,
            suggestions=[
                _S(
                    change="speed",
                    value=Quantity(0.0025, default_unit="m/s"),
                    note="halve the free stream to buy back Mach-number headroom",
                )
            ],
        )
    return Unrepresentable(
        reason=(
            "blockage ratio at or above CLAUDE.md constraint 12's 10% ceiling "
            "(D-019, D-026)"
        ),
        quantity="blockage",
        value=0.12,
        limit=0.10,
        suggestions=[
            _S(
                change="quality",
                value="fast",
                note=(
                    "a lower quality level does not change blockage (span "
                    "scales with resolution); widen the mask's own aspect "
                    "ratio instead"
                ),
            )
        ],
    )


def check_unreachable(refusal_class: str, backend: str) -> dict[str, Any]:
    """Explain a class nothing can reach, then run its suggestion on a base case."""
    exc = _synthetic(refusal_class)
    assert classify(exc) == refusal_class
    text = explain(exc)
    assert text.strip(), f"{refusal_class}: explain() produced nothing"
    top = exc.suggestions[0]
    fixed = apply_suggestion(top, **BASE_REQUEST)
    fixed_plan = plan(**fixed)
    measured = run_plan(fixed_plan, backend)
    return {"case": f"{refusal_class} (unreachable)", "top": top, **measured}


# ---------------------------------------------------------------------------
# Section 3 — Monitor against the three failure modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureMode:
    """A case built to die, and the cause :class:`Monitor` must name."""

    name: str
    cause: str
    maxsteps: int
    build: Any = field(repr=False)


def _disc_sim(ny: int, nx: int, tau: float, U: float, diameter: float, **kw: Any) -> Sim:
    solid = circle(ny, nx, nx * 0.25, (ny - 1) / 2.0 + 0.5, diameter / 2.0)
    base = dict(
        ny=ny, nx=nx, tau=tau, inlet_U=U, profile="uniform", use_inlet=True,
        use_outlet=True, convective_outlet=True, inlet_axis="x",
        check_geometry=False,
    )
    base.update(kw)
    return Sim(SimConfig(**base), solid)


def _mode_relaxation(backend: str) -> Sim:
    """``tau`` below the bluff-body floor — **D-029**'s own measured death."""
    return _disc_sim(64, 192, 0.533, 0.05, 16.0, backend=backend)


def _mode_speed(backend: str) -> Sim:
    """Driven far past the constraint-3 ceiling, with ``tau`` comfortable."""
    return _disc_sim(64, 192, 0.6, 0.25, 16.0, backend=backend)


def _mode_mass(backend: str) -> Sim:
    """Fed at the inlet, sealed downstream: the domain fills until it bursts."""
    ny, nx = 32, 64
    solid = circle(ny, nx, nx * 0.25, (ny - 1) / 2.0 + 0.5, 2.0)
    solid[:, -3:] = True
    cfg = SimConfig(
        ny=ny, nx=nx, tau=0.6, inlet_U=0.09, profile="uniform", use_inlet=True,
        use_outlet=False, inlet_axis="x", check_geometry=False, backend=backend,
    )
    return Sim(cfg, solid)


MODES: tuple[FailureMode, ...] = (
    FailureMode("tau below the stability floor (IDEA2 row 1)", "relaxation", 4000, _mode_relaxation),
    FailureMode("peak |u| past the 0.1 ceiling (IDEA2 hard rule)", "speed", 4000, _mode_speed),
    FailureMode("mass drift: sealed downstream end", "mass", 120_000, _mode_mass),
)


def nan_step(sim: Sim, maxsteps: int, every: int = SAMPLE_EVERY) -> int | None:
    """The first sampled step at which the state stops being finite."""
    for i in range(1, maxsteps + 1):
        sim.step()
        if i % every == 0:
            if not (
                np.isfinite(sim.host_f()).all() and np.isfinite(sim.host_rho()).all()
            ):
                return i
    return None


def caught_step(sim: Sim, maxsteps: int, monitor: Monitor) -> tuple[int | None, Diverging | None]:
    """Run with ``monitor`` as a ``per_step`` probe (**D-025**) until it fires."""
    for _ in range(maxsteps):
        sim.step()
        try:
            monitor(sim)
        except Diverging as exc:
            return exc.step, exc
    return None, None


def check_mode(mode: FailureMode, backend: str) -> dict[str, Any]:
    """Measure the bare death step and the caught step for one failure mode."""
    died = nan_step(mode.build(backend), mode.maxsteps)
    monitor = Monitor(every=SAMPLE_EVERY)
    caught, exc = caught_step(mode.build(backend), mode.maxsteps, monitor)
    return {
        "mode": mode.name,
        "expect_cause": mode.cause,
        "nan_step": died,
        "caught_step": caught,
        "exc": exc,
    }


def check_no_false_alarm(backend: str, steps: int = STEPS_PER_CASE) -> dict[str, Any]:
    """A healthy case must run its full length without :class:`Monitor` firing.

    A tripwire that fires on a good run is worse than no tripwire: it teaches
    the user to ignore it. Rung B's own accuracy case, at the resolution
    :func:`plan` picks for it.
    """
    healthy = plan(**_planargs(BASE_REQUEST))
    solid = build_solid(healthy)
    ny, nx = healthy.domain
    cfg = SimConfig(
        ny=ny, nx=nx, tau=healthy.tau, inlet_U=healthy.u_lattice,
        profile="uniform", use_inlet=True, use_outlet=True,
        convective_outlet=True, inlet_axis="x", check_geometry=False,
        backend=backend,
    )
    sim = Sim(cfg, solid)
    monitor = Monitor(every=SAMPLE_EVERY)
    for _ in range(steps):
        sim.step()
        monitor(sim)
    return {"peak_u": monitor.peak_speed, "drift": monitor.drift, "steps": steps}


# ---------------------------------------------------------------------------
# Section 4 — what Monitor costs
# ---------------------------------------------------------------------------


def monitor_cost(backend: str, steps: int = 600, rounds: int = 5) -> dict[str, float]:
    """Steps/s with and without the probe, by alternating rounds (**D-035**).

    Two consecutive runs of the *identical* path differ by more than 10% on
    this machine, so all-of-A-then-all-of-B awards the win to whichever ran
    during the quieter minute. Alternating rounds with one ``Sim`` resident and
    the best round per variant is the protocol Phase 0 settled on.
    """
    healthy = plan(**_planargs(BASE_REQUEST))
    solid = build_solid(healthy)
    ny, nx = healthy.domain
    cfg = SimConfig(
        ny=ny, nx=nx, tau=healthy.tau, inlet_U=healthy.u_lattice,
        profile="uniform", use_inlet=True, use_outlet=True,
        convective_outlet=True, inlet_axis="x", check_geometry=False,
        backend=backend,
    )

    def time_round(with_monitor: bool) -> float:
        sim = Sim(cfg, solid)
        monitor = Monitor(every=SAMPLE_EVERY) if with_monitor else None
        for _ in range(20):  # warm-up, outside the clock
            sim.step()
        start = time.perf_counter()
        for _ in range(steps):
            sim.step()
            if monitor is not None:
                monitor(sim)
        sim.host_f()
        return steps / (time.perf_counter() - start)

    bare = 0.0
    watched = 0.0
    for _ in range(rounds):
        bare = max(bare, time_round(False))
        watched = max(watched, time_round(True))
    return {
        "bare": bare,
        "watched": watched,
        "cost": (bare - watched) / bare,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

#: The cost ceiling from the T106 acceptance criterion.
COST_LIMIT: float = 0.02


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="numpy", choices=("numpy", "warp"))
    parser.add_argument(
        "--steps", type=int, default=STEPS_PER_CASE,
        help="timesteps each suggested case is run for",
    )
    args = parser.parse_args(argv)
    backend = args.backend
    failures: list[str] = []

    print("Rung D - every refusal names a fix, and the fix runs. backend =", backend)
    print()

    # --- 1. refusals ------------------------------------------------------
    print(f"1. refusal classes: refuse -> suggest -> apply -> plan -> {args.steps} steps")
    for case in CASES:
        try:
            r = check_case(case, backend)
            top: Suggestion = r["top"]
            print(
                f"   [ok]   {case.name}\n"
                f"          {r['suggestions']} suggestion(s); top: {top.change} -> "
                f"{_short(top.value)}\n"
                f"          replanned {r['plan'].domain[0]}x{r['plan'].domain[1]}, "
                f"ran {args.steps} steps clean, peak |u| {r['peak_u']:.4f} "
                f"({r['seconds']:.1f}s)"
            )
        except AssertionError as exc:
            failures.append(f"refusal {case.refusal_class}: {exc}")
            print(f"   [FAIL] {case.name}: {exc}")

    for cls in unreachable_classes():
        try:
            r = check_unreachable(cls, backend)
            print(
                f"   [ok]   {cls} - unreachable through plan() by construction "
                f"(D-059); explained, and its suggestion "
                f"({r['top'].change} -> {_short(r['top'].value)}) runs clean on "
                f"the base case, peak |u| {r['peak_u']:.4f}"
            )
        except AssertionError as exc:
            failures.append(f"refusal {cls}: {exc}")
            print(f"   [FAIL] {cls}: {exc}")
    print()

    # --- 2. the D-038 case, printed in full -------------------------------
    print("2. the D-038 case, as the user sees it")
    d038 = CASES[0].request
    try:
        plan(**_planargs(d038))
        failures.append("D-038 case planned instead of refusing")
        print("   [FAIL] it planned; D-038 says it must refuse")
    except Unrepresentable as exc:
        text = explain(
            exc,
            request={"fluid": "air", "speed": "20 m/s", "size": "1.5 m"},
        )
        for line in text.splitlines():
            print(f"   | {line}")
        first = text.split("\n\n", 1)[0].lower()
        leaks = [w for w in ("tau", "lattice", "cell", "timestep") if w in first]
        if leaks:
            failures.append(f"D-038 first paragraph leaks {leaks}")
            print(f"   [FAIL] first paragraph names {leaks}")
        else:
            print("   [ok]   first paragraph names no lattice quantity")
    print()

    # --- 3. Monitor -------------------------------------------------------
    print("3. Monitor: caught before nan, with the cause named")
    for mode in MODES:
        r = check_mode(mode, backend)
        died, caught, exc = r["nan_step"], r["caught_step"], r["exc"]
        if caught is None:
            failures.append(f"monitor {mode.cause}: never fired")
            print(f"   [FAIL] {mode.name}: Monitor never fired (nan at {died})")
            continue
        if exc.cause != mode.cause:
            failures.append(
                f"monitor {mode.name}: cause {exc.cause!r}, expected {mode.cause!r}"
            )
            print(f"   [FAIL] {mode.name}: cause {exc.cause!r}")
            continue
        if died is None:
            print(
                f"   [ok]   {mode.name}: caught at step {caught}; the bare run "
                f"never reached nan in {mode.maxsteps} steps -- it would have "
                "gone on producing a plausible, invalid answer"
            )
        elif caught < died:
            print(
                f"   [ok]   {mode.name}: caught at {caught}, nan at {died} "
                f"({(died - caught) / died:.1%} of the run's life earlier)"
            )
        else:
            failures.append(f"monitor {mode.name}: caught at {caught}, nan at {died}")
            print(f"   [FAIL] {mode.name}: caught at {caught}, not before nan at {died}")
        print(f"          cause {exc.cause!r}: {exc.symptom}")
        print(f"          fix: {exc.fix}")

    healthy = check_no_false_alarm(backend, args.steps)
    print(
        f"   [ok]   no false alarm: a healthy case ran {healthy['steps']} steps "
        f"untouched, peak |u| {healthy['peak_u']:.4f}, mass drift "
        f"{healthy['drift']:.2e}"
    )
    print()

    # --- 4. cost ----------------------------------------------------------
    print("4. Monitor's cost, alternating rounds, best round per variant (D-035)")
    state = _machine_state()
    cost = monitor_cost(backend)
    ok = cost["cost"] < COST_LIMIT
    if not ok:
        failures.append(f"monitor cost {cost['cost']:.2%} >= {COST_LIMIT:.0%}")
    print(
        f"   bare {cost['bare']:.1f} steps/s, watched {cost['watched']:.1f} "
        f"steps/s, cost {cost['cost']:.2%} (limit {COST_LIMIT:.0%}) "
        f"[{'ok' if ok else 'FAIL'}]"
    )
    if cost["cost"] <= 0.0:
        print(
            "   a cost at or below zero means the probe is cheaper than this "
            "machine's run-to-run spread: D-035 measured 12-21% between two "
            "runs of the *identical* path, so the honest reading is 'under "
            "the noise floor', not 'free'."
        )
    print(f"   conditions: {state}")
    print()

    if failures:
        print(f"Rung D: FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Rung D: PASS")
    return 0


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 48 else text[:45] + "..."


def _machine_state() -> str:
    """CPU clock and power state beside every timing (**D-035**)."""
    try:
        from bench import machine_state

        s = machine_state()
        return (
            f"{s.get('cpu')} at {s.get('cpu_mhz')} MHz of "
            f"{s.get('cpu_max_mhz')} MHz, on {s.get('power')}"
        )
    except Exception:  # noqa: BLE001 - conditions are context, not a gate
        return "unavailable"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
