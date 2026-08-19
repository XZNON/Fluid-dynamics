"""Rung B — ``flow/autoconfig.py`` chooses sane, stable, checkable parameters.

``DOCS/IDEA3.md`` § Validation ladder: Rung B's known answer is "the guardrails
(``tau`` floors, ``U < 0.1``, blockage, downstream fetch) and the analytic
``Re``, reproduced to 0.1%." Two things are checked, in order (**D-047**: the
harness before the code, and here after it, since T105 built both together):

1. **Accuracy** — :meth:`flow.autoconfig.Plan.estimated_seconds` predicted
   against a real, timed run of the committed cylinder case, within 25%.
2. **The sweep** — at least 24 cases (2 fluids x 2 speeds x 2 sizes x 3
   quality levels). Every case must (a) satisfy every guardrail on its
   *rasterised* geometry — checked independently via
   :func:`lbm.geometry.check_mask` rather than trusted from :class:`Plan`,
   (b) run 5000 steps with no ``nan`` and peak ``|u| < 0.1``, (c) reproduce its
   requested ``Re`` to 0.1% through :meth:`~lbm.units.LatticeUnits.reynolds`.

Every physical combination in the sweep is chosen to keep ``Re`` low enough
that even ``quality="fast"`` clears :data:`flow.autoconfig.TAU_FLOOR` — a case
that is *supposed* to refuse belongs to Rung D (T106), not here (**D-045**:
refusal is this project's correct answer to an unrepresentable case, and Rung
B's job is to prove the cases that *are* representable actually run clean).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import flow.autoconfig as ac
from flow.autoconfig import Plan, Unrepresentable, plan
from flow.fluids import fluid
from lbm.geometry import bounding_box, check_mask, circle
from lbm.runner import Sim, SimConfig
from lbm.units import BLUFF_BODY_SPEEDUP, U_LATTICE_MAX, LatticeUnits

#: A single representative disc, reused as the ``mask`` argument for every
#: case in this rung. Shape variety (hairlines, holes, awful anti-aliasing) is
#: Rung C's job (T107); Rung B holds geometry fixed and varies physics.
NOMINAL_MASK = circle(80, 80, 40.0, 40.0, 20.0)

STEPS_PER_CASE: int = 5000
RE_TOLERANCE: float = 0.001
ACCURACY_TOLERANCE: float = 0.25


def build_solid(plan_: Plan) -> np.ndarray:
    """Rasterise a disc of diameter ``plan_.cells_per_length`` into ``plan_.domain``.

    Mirrors ``validate/cylinder.py::cylinder_mask``: the digitised extent of a
    nominal-diameter disc is ``d + 1`` cells (**D-019**), and the leading edge
    sits :data:`flow.autoconfig.UPSTREAM_D` diameters from the inlet — the
    same placement :func:`flow.autoconfig.plan` assumed when it sized the
    domain, so the guardrails it computed analytically are the ones this
    function's output actually has.
    """
    ny, nx = plan_.domain
    n = plan_.cells_per_length
    cx = ac.UPSTREAM_D * n + n / 2.0
    cy = (ny - 1) / 2.0 + 0.5  # half-cell offset breaks mirror symmetry
    return circle(ny, nx, cx, cy, n / 2.0)


def run_case(
    *,
    fluid_name: str,
    speed_ms: float,
    size_m: float,
    quality: str,
    backend: str,
    steps: int = STEPS_PER_CASE,
) -> dict:
    """Plan, rasterise, run ``steps`` timesteps, and measure everything Rung B checks.

    Returns:
        A dict of measurements; raises ``AssertionError`` naming the first
        thing that failed, so a case failure reads as a stack trace pointing
        at the guardrail that broke rather than a bare PASS/FAIL line.
    """
    f = fluid(fluid_name)
    plan_ = plan(
        fluid=f,
        speed=f"{speed_ms} m/s",
        size=f"{size_m} m",
        mask=NOMINAL_MASK,
        quality=quality,
    )

    solid = build_solid(plan_)
    ny, nx = plan_.domain

    # (a) guardrails, on the actual rasterised geometry — independent of
    # Plan's own analytic estimate.
    check_mask(
        solid,
        "x",
        min_thickness_cells=3,
        min_downstream_lengths=8.0,
        max_blockage=0.10,
        strict=True,
        verbose=False,
    )
    peak_estimate = plan_.u_lattice * BLUFF_BODY_SPEEDUP
    assert peak_estimate < U_LATTICE_MAX, (
        f"{fluid_name}/{quality}: peak velocity estimate {peak_estimate} >= "
        f"ceiling {U_LATTICE_MAX}"
    )
    assert plan_.tau > ac.TAU_FLOOR, f"{fluid_name}/{quality}: tau {plan_.tau} at/below floor"

    # (b) run clean.
    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=plan_.tau,
        inlet_U=plan_.u_lattice,
        profile="uniform",
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        inlet_axis="x",
        check_geometry=False,  # already checked above, with the strict rules
        backend=backend,
    )
    sim = Sim(cfg, solid)
    for _ in range(steps):
        sim.step()

    f_host = sim.host_f()
    assert np.isfinite(f_host).all(), f"{fluid_name}/{quality}: nan after {steps} steps"
    u = sim.host_u()
    fluid_cells = ~sim.solid
    peak_u = float(np.sqrt(u[0][fluid_cells] ** 2 + u[1][fluid_cells] ** 2).max())
    assert peak_u < 0.1, f"{fluid_name}/{quality}: peak |u| {peak_u} >= 0.1 after {steps} steps"

    # (c) Re, reproduced through LatticeUnits.reynolds() — analytic, not
    # measured from the flow.
    nu_phys = f.nu.si
    requested_re = speed_ms * size_m / nu_phys
    units = LatticeUnits.from_physical(
        u_phys=speed_ms,
        l_phys=size_m,
        nu_phys=nu_phys,
        cells_per_length=plan_.cells_per_length,
        u_lattice=plan_.u_lattice,
    )
    reproduced_re = units.reynolds()
    re_error = abs(reproduced_re - requested_re) / requested_re
    assert re_error < RE_TOLERANCE, (
        f"{fluid_name}/{quality}: Re reproduction error {re_error:.4%} >= "
        f"{RE_TOLERANCE:.2%}"
    )

    return {
        "fluid": fluid_name,
        "quality": quality,
        "speed": speed_ms,
        "size": size_m,
        "Re": requested_re,
        "tau": plan_.tau,
        "domain": plan_.domain,
        "peak_u": peak_u,
        "re_error": re_error,
    }


def check_accuracy(backend: str) -> tuple[float, float, float]:
    """The committed cylinder case: predicted vs actual wall clock.

    Water, sized for Re 100 at ``quality="fast"`` (matches Rung 3's benchmark
    Reynolds number, at this module's own chosen resolution rather than Rung
    3's hand-tuned one — see the ``flow/autoconfig.py`` module docstring on
    why the two need not match cell for cell).

    Returns:
        ``(predicted_seconds, actual_seconds, relative_error)``.
    """
    f = fluid("water")
    plan_ = plan(
        fluid=f,
        speed="0.005 m/s",
        size="0.02 m",
        mask=NOMINAL_MASK,
        quality="fast",
    )
    solid = build_solid(plan_)
    ny, nx = plan_.domain
    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=plan_.tau,
        inlet_U=plan_.u_lattice,
        profile="uniform",
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        inlet_axis="x",
        check_geometry=False,
        backend=backend,
    )
    sim = Sim(cfg, solid)
    predicted = plan_.estimated_seconds(backend)

    start = time.perf_counter()
    for _ in range(plan_.steps):
        sim.step()
    _ = sim.host_f()  # force any lazy device work to complete before timing stops
    actual = time.perf_counter() - start

    error = abs(predicted - actual) / actual
    return predicted, actual, error


# --- sweep cases ---------------------------------------------------------

FLUIDS: tuple[str, ...] = ("water", "olive oil")
SPEEDS_MS: tuple[float, ...] = (0.002, 0.004)
SIZES_M: tuple[float, ...] = (0.01, 0.02)
QUALITIES: tuple[str, ...] = ("fast", "balanced", "accurate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="numpy", choices=("numpy", "warp"))
    args = parser.parse_args(argv)

    print("Rung B - flow.autoconfig, backend =", args.backend)
    print()
    print("1. accuracy: Plan.estimated_seconds vs a timed run")
    predicted, actual, error = check_accuracy(args.backend)
    accuracy_ok = error < ACCURACY_TOLERANCE
    print(
        f"   predicted {predicted:.2f}s, actual {actual:.2f}s, "
        f"error {error:.1%} (limit {ACCURACY_TOLERANCE:.0%}) "
        f"[{'ok' if accuracy_ok else 'FAIL'}]"
    )
    print()

    cases = [
        (f, s, sz, q)
        for f in FLUIDS
        for s in SPEEDS_MS
        for sz in SIZES_M
        for q in QUALITIES
    ]
    print(f"2. sweep: {len(cases)} cases x {STEPS_PER_CASE} steps")

    results: list[dict] = []
    failures: list[str] = []
    for fluid_name, speed_ms, size_m, quality in cases:
        try:
            result = run_case(
                fluid_name=fluid_name,
                speed_ms=speed_ms,
                size_m=size_m,
                quality=quality,
                backend=args.backend,
            )
            results.append(result)
            print(
                f"   {fluid_name:>10s} {quality:>8s}  Re={result['Re']:>8.3g}  "
                f"tau={result['tau']:.4f}  domain={result['domain']}  "
                f"peak|u|={result['peak_u']:.4f}  Re err={result['re_error']:.4%}  [ok]"
            )
        except (AssertionError, Unrepresentable, ValueError) as exc:
            failures.append(f"{fluid_name}/{quality}/{speed_ms}ms/{size_m}m: {exc}")
            print(f"   {fluid_name:>10s} {quality:>8s}  FAIL: {exc}")

    print()
    if results:
        worst_peak = max(results, key=lambda r: r["peak_u"])
        worst_re = max(results, key=lambda r: r["re_error"])
        print(f"   worst peak|u|: {worst_peak['peak_u']:.4f} ({worst_peak['fluid']}/{worst_peak['quality']})")
        print(f"   worst Re error: {worst_re['re_error']:.4%} ({worst_re['fluid']}/{worst_re['quality']})")

    sweep_ok = not failures and len(results) >= 24
    overall_ok = accuracy_ok and sweep_ok

    print()
    if failures:
        print(f"   {len(failures)} case(s) failed:")
        for line in failures:
            print("     -", line)
    print()
    print("PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
