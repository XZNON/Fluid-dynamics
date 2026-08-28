"""Rung E — the whole product path, end to end, against published physics, timed.

``DOCS/IDEA3.md`` § Validation ladder: *"Rung 3's cylinder — ``St`` 0.155-0.175,
``Cd`` 1.25-1.45 — reached through ``flow.Case`` from a PNG, inside a stated wall
clock"*, and § Goal: *"a correct, moving, believable answer from a picture of a
shape and three physical numbers — in under a minute, from a cold shell"*.

**Rung E is the phase.** It is the only rung that touches every box at once, and
it is deliberately the Phase 0 cylinder, so a regression in *judgement* shows up
as a regression in *physics*. The two bands are **imported** from
``validate/cylinder.py`` rather than copied, so this rung cannot drift from Rung
3's published numbers even by a typo.

What this script asserts, in order:

1. **Physics.** A committed PNG of a disc (``tests/data/shapes/disc.png``) plus
   a fluid, a speed and a size that give Re 100 — and **no lattice quantity
   anywhere in the invocation** (constraint 13) — reach ``Cd`` inside
   :data:`~validate.cylinder.CD_BAND` and ``St`` inside
   :data:`~validate.cylinder.ST_BAND`.
2. **The minute.** The wall clock from **process start** (``psutil``'s own
   ``create_time``, so the interpreter, the imports and the Warp context are all
   inside the number) to :meth:`flow.report.Result.summary` returning, against
   :data:`TIME_LIMIT_SECONDS`. It is a **gate** only on a backend in
   :data:`TIMED_BACKENDS`; on ``numpy`` the number is printed and the physics
   still has to pass, which is what proves the product path is independent of
   the port (the T110 contract's third criterion).

Every absolute number is printed with the CPU clock, the power state and the GPU
name beside it (**D-035**), via :func:`bench.machine_state`.

The domain this runs in is ``flow/autoconfig.py``'s, not Rung 3's hand-tuned
one, and the difference is the subject of **D-075** — the decision that answered
**Q-104**. The blockage and the fetches are printed here for that reason: they
are the reason the numbers land where they do.
"""

from __future__ import annotations

import argparse
import os
import time

import psutil

from bench import machine_state, print_machine_state
from flow.autoconfig import DOWNSTREAM_D, SPAN_D, UPSTREAM_D
from flow.case import Case
from flow.report import MIN_PERIODS, ST_PLAUSIBLE, TRANSIENT_FRACTION
from validate.cylinder import CD_BAND, ST_BAND

#: The committed picture. ``tests/data/shapes/disc.png`` is Rung C's own corpus
#: disc — already committed, already pinned pixel-for-pixel against its
#: generator by ``tests/test_prepare.py``, so this rung adds no new binary.
DISC_PNG: str = "tests/data/shapes/disc.png"

#: The three physical numbers. Water at 5 mm/s past a 2 cm body is
#: ``Re = 0.005 * 0.02 / 1.004e-6`` = **99.6** — Rung 3's Reynolds number, and
#: the case the project already uses (**D-074**). Nothing here is a lattice
#: quantity: a fluid by name, a speed, a size (constraint 13).
FLUID: str = "water"
SPEED: str = "5 mm/s"
SIZE: str = "2 cm"

#: Resolution level, in the product's own vocabulary. ``"fast"`` is 30 cells
#: across the body (``flow.autoconfig.QUALITY_CELLS``) and is the level the
#: whole product path has been measured at since session 20. Printed rather
#: than hidden: the summary names it, per constraint 16.
QUALITY: str = "fast"

#: Margin over the shortest run **D-070**'s gate 2 will accept. 5%: enough that
#: the rung is not sitting on an exact equality, small enough that it is two
#: seconds of the sixty. Every convective time above the gate is wall clock
#: spent against :data:`TIME_LIMIT_SECONDS`, so this is deliberately not
#: generous.
RUN_MARGIN: float = 1.05

#: Run length in convective times ``D / U`` — the same unit
#: ``validate/cylinder.py`` uses, and physical rather than lattice (it reaches
#: :meth:`flow.case.Case.run` as ``size / speed`` seconds).
#:
#: **Derived from D-070's gate 2, not chosen**, because the gate is what
#: decides whether there is a Strouhal number at all: the measurement window is
#: the last ``1 - TRANSIENT_FRACTION`` of the run and it has to hold
#: ``MIN_PERIODS`` of the *slowest plausible* shedding, which is
#: ``D / (U * ST_PLAUSIBLE[0])`` = 20 convective times each. That is 40
#: convective times of window and therefore exactly **80** of run — an equality
#: the rung should not be standing on, hence :data:`RUN_MARGIN`, and a number
#: that follows ``flow/report.py`` if those constants ever move rather than
#: silently going stale beside them. Session 21 measured the other side of this
#: gate: a 45000-step run reported ``St = None`` and was right to.
RUN_CONVECTIVE_TIMES: float = (
    RUN_MARGIN * MIN_PERIODS / (ST_PLAUSIBLE[0] * (1.0 - TRANSIENT_FRACTION))
)

#: The minute. ``idea.md`` § Definition of success and ``DOCS/PLAN2.md`` § M8.
TIME_LIMIT_SECONDS: float = 60.0

#: Backends the wall clock is a **gate** on. On any other backend it is printed
#: and the physics still gates — the T110 contract's third criterion.
TIMED_BACKENDS: tuple[str, ...] = ("warp",)


def run_rung(backend: str, *, explain: bool = False) -> dict[str, object]:
    """Drive the product path once and return everything the rung judges.

    Args:
        backend: registry name — ``"numpy"`` or ``"warp"``.
        explain: print :meth:`flow.case.Case.explain`'s plan before running.
            Off by default: the plan is not what this rung is timing, and
            printing it is not free on a 60-second budget.

    Returns:
        ``{"cd", "cd_std", "strouhal", "peak_u", "steps", "elapsed_run",
        "elapsed_total", "plan", "result"}``. ``elapsed_total`` is measured from
        **process start**, not from this call.
    """
    started = psutil.Process(os.getpid()).create_time()

    case = Case.from_image(
        DISC_PNG,
        fluid=FLUID,
        speed=SPEED,
        size=SIZE,
        quality=QUALITY,
        backend=backend,
    )
    plan = case.plan
    assert plan is not None, "the committed disc at Re 100 is representable"
    if explain:
        case.explain()

    # Physical seconds, not steps: `size / speed` is one convective time.
    seconds = RUN_CONVECTIVE_TIMES * _convective_time_seconds(case)
    # `quiet=True` and no sink is D-071's un-drawn run, reached from a script
    # the way `--no-live --quiet` reaches it from the CLI (**D-073**): nothing
    # is coloured, so the wall clock is the answer's and not the picture's.
    result = case.run(seconds=seconds, keep_frames=False, quiet=True)
    result.summary()
    elapsed_total = time.time() - started

    return {
        "cd": result.cd,
        "cd_std": result.cd_std,
        "strouhal": result.strouhal,
        "peak_u": result.peak_u,
        "steps": result.steps,
        "elapsed_run": result.elapsed,
        "elapsed_total": elapsed_total,
        "plan": plan,
        "result": result,
    }


def _convective_time_seconds(case: Case) -> float:
    """One convective time ``D / U`` in **physical** seconds.

    Taken from the case's own parsed inputs, so the run length is expressed in
    the user's units all the way down (constraint 13).
    """
    return case.size.si / case.speed.si


def main(argv: list[str] | None = None) -> int:
    """Run Rung E and print PASS/FAIL.

    Returns:
        Process exit code — ``0`` PASS, ``1`` FAIL.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="warp",
        choices=("numpy", "warp"),
        help="which backend to drive the product path on (default: warp, the "
             "backend M8's wall clock is quoted on)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="print Case.explain()'s plan before the run (off by default: the "
             "plan is not what this rung times)",
    )
    args = parser.parse_args(argv)

    print("Rung E - the minute: a picture and three physical numbers, end to end")
    print(
        f"  case:  {DISC_PNG}, {FLUID}, {SPEED}, {SIZE}, quality {QUALITY!r} "
        f"- no lattice quantity (constraint 13)"
    )
    print(
        f"  domain: {SPAN_D:g} D span, {UPSTREAM_D:g} D upstream, "
        f"{DOWNSTREAM_D:g} D downstream (flow/autoconfig.py, D-075)"
    )
    print()

    measured = run_rung(args.backend, explain=args.explain)

    # Collected *after* the run, deliberately. `machine_state` shells out to
    # WMI three times and to `nvidia-smi` once, which is seconds a user of the
    # product never spends — and the clock this rung reports starts at process
    # start, so anything done before the run is charged to the minute
    # (**D-035** asks for the conditions beside the number, not inside it).
    print()
    print_machine_state(machine_state(), args.backend)

    plan = measured["plan"]
    ny, nx = plan.domain  # type: ignore[union-attr]
    blockage = plan.cells_per_length / ny * 100.0  # type: ignore[union-attr]
    cd = float(measured["cd"])
    st = measured["strouhal"]
    elapsed = float(measured["elapsed_total"])
    timed = args.backend in TIMED_BACKENDS

    cd_ok = CD_BAND[0] <= cd <= CD_BAND[1]
    st_ok = st is not None and ST_BAND[0] <= float(st) <= ST_BAND[1]
    time_ok = elapsed < TIME_LIMIT_SECONDS

    print()
    print("1. physics - Rung 3's published bands, unwidened")
    print(
        f"   grid {ny}x{nx} = {ny * nx / 1000:.0f}k cells, blockage "
        f"{blockage:.2f}%, Re {plan.Re:.1f}, tau {plan.tau:.4f}, "  # type: ignore[union-attr]
        f"{measured['steps']} steps"
    )
    print(
        f"   Cd  {cd:.4f} +- {float(measured['cd_std']):.4f}   "
        f"band {CD_BAND[0]}..{CD_BAND[1]}   [{'ok' if cd_ok else 'FAIL'}]"
    )
    st_text = "None" if st is None else f"{float(st):.4f}"
    print(
        f"   St  {st_text}   band {ST_BAND[0]}..{ST_BAND[1]}   "
        f"[{'ok' if st_ok else 'FAIL'}]"
    )
    print(
        f"   peak |u| {float(measured['peak_u']):.5f} of the 0.1 ceiling "
        f"(constraint 3)"
    )

    print()
    print("2. the minute - process start to Result.summary()")
    print(
        f"   {elapsed:.1f} s wall clock ({float(measured['elapsed_run']):.1f} s "
        f"of it the run itself), limit {TIME_LIMIT_SECONDS:.0f} s   "
        f"[{'ok' if time_ok else 'FAIL'}"
        f"{'' if timed else ', not a gate on this backend'}]"
    )
    if not timed:
        print(
            f"   the wall clock gates on {'/'.join(TIMED_BACKENDS)} only; on "
            f"{args.backend!r} this rung proves the physics is independent of "
            "the port (T110 criterion 3)"
        )

    ok = cd_ok and st_ok and (time_ok or not timed)
    print()
    print(f"Rung E: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
