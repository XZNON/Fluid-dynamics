"""Steps/s benchmark for the LBM kernel, and the before/after table for T010.

``DOCS/IDEA2.md`` § Performance budget is the contract this measures::

    400x100    40k cells   ~500+ steps/s   interactive
    800x200   160k cells   ~150  steps/s   usable
    2000x500    1M cells   ~20   steps/s   record, don't watch

``DOCS/TASKS1.md`` § T010's pass floors are 400 / 120 / 15 steps/s on those
three grids, and it asks for the baseline to be captured **before** the first
optimisation lands — hence ``--save-baseline``, which writes
``DOCS/bench_baseline.json``, and the default mode, which re-measures and prints
the two columns side by side.

The benchmarked case is deliberately the *whole* step (``lbm.runner.Sim.step``),
not ``collide`` in isolation: it includes the Zou-He inlet, the convective
outlet, bounce-back on an immersed body and the two snapshot copies D-020
requires. Optimising a kernel that is not the one the rungs run is how a
performance pass produces a number nobody can reproduce.

Usage::

    myenv/Scripts/python.exe bench.py --save-baseline   # before optimising
    myenv/Scripts/python.exe bench.py                   # before/after table
    myenv/Scripts/python.exe bench.py --variants        # each win on its own
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from lbm.geometry import circle
from lbm.runner import Sim, SimConfig

#: The three grids of ``DOCS/IDEA2.md`` § Performance budget, as ``(ny, nx)``.
#: The table names them width x height, so 400x100 is ``ny=100, nx=400``.
GRIDS: tuple[tuple[int, int], ...] = ((100, 400), (200, 800), (500, 2000))

#: ``DOCS/TASKS1.md`` § T010 pass floors, steps/s, keyed by cell count.
FLOORS: dict[int, float] = {40_000: 400.0, 160_000: 120.0, 1_000_000: 15.0}

#: The budget's own expectations, for context in the printed table.
BUDGET: dict[int, float] = {40_000: 500.0, 160_000: 150.0, 1_000_000: 20.0}

BASELINE_PATH = Path("DOCS/bench_baseline.json")

#: Timed steps per grid. Chosen so every case runs for a couple of seconds at
#: the *baseline* speed; the same counts are reused after optimising so the two
#: columns measure the same work.
STEPS: dict[int, int] = {40_000: 400, 160_000: 150, 1_000_000: 30}
WARMUP: int = 10


def make_sim(ny: int, nx: int, **opts: Any) -> Sim:
    """A representative open-channel case: inlet, convective outlet, one body.

    The body is a disc of ``D = ny / 10`` a quarter of the way along, which is
    the shape Rung 3 runs. ``check_geometry`` is off because this is a timing
    harness, not a physics run — the sanity checks belong to the rungs, and a
    blockage warning on every bench invocation would train us to ignore them
    (``DOCS/STATE1.md`` D-018's reasoning, applied to the bench).

    Extra keyword arguments are passed to :class:`lbm.runner.SimConfig` only if
    that dataclass has the field, so this file works unchanged before and after
    the optimisation toggles exist.
    """
    d = max(6, ny // 10)
    solid = circle(ny, nx, nx / 4.0, ny / 2.0, d / 2.0)

    known = {f.name for f in dataclasses.fields(SimConfig)}
    extra = {k: v for k, v in opts.items() if k in known}

    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=0.6,
        inlet_U=0.05,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        check_geometry=False,
        **extra,
    )
    return Sim(cfg, solid)


def measure(
    ny: int,
    nx: int,
    steps: int,
    warmup: int = WARMUP,
    repeats: int = 3,
    **opts: Any,
) -> float:
    """Steps per second for one grid, warmed up first, **best of** ``repeats``.

    Best rather than mean: a slow repeat means something else on the machine got
    the core, which is noise about Windows and not about the kernel. The fastest
    run is the one least contaminated. Session 7 already saw this — Rung 3 took
    368.9 s alone and 633.3 s under contention for identical physics.

    Returns:
        Steps per second.
    """
    sim = make_sim(ny, nx, **opts)
    sim.run_steps(warmup)

    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        sim.run_steps(steps)
        elapsed = time.perf_counter() - t0
        best = max(best, steps / elapsed)

    if not np.isfinite(sim.f).all():
        raise RuntimeError(
            f"{nx}x{ny} produced non-finite f after {warmup + steps * repeats} "
            "steps — a benchmark that diverged is not measuring the kernel."
        )
    return best


def compare(
    variants: list[tuple[str, dict[str, Any]]], rounds: int = 5
) -> dict[str, dict[int, float]]:
    """Time several variants against each other, **interleaved**.

    Sequential A-then-B measurement is not trustworthy on this machine: two
    back-to-back runs of the *same* reference path measured 726.8 and 811.5
    steps/s at 400x100, a 12% spread, and 146.1 against 177.2 at 800x200. A
    thermal or scheduling drift of that size is larger than the effect being
    measured, and whichever variant ran second would win or lose by it.

    So every variant is timed once per round, in the same round, and the best
    round per variant is kept. Drift then hits all variants alike, and the best
    round is the one least contaminated by whatever else the OS was doing.

    Only **one** simulation is resident at a time. Holding both at once was
    tried and measured worse for *both* variants — 1M cells fell from ~21 to
    ~16 steps/s — which is the expected answer for a memory-bound kernel: two
    co-resident sims are ~500 MB of buffers competing for the same cache, and a
    cache effect measured under cache pressure is not the effect. Each round
    therefore builds a variant, times it, and drops it before the next.

    Returns:
        ``{variant name: {cells: steps_per_second}}``.
    """
    results: dict[str, dict[int, float]] = {name: {} for name, _ in variants}

    for ny, nx in GRIDS:
        cells = ny * nx
        steps = STEPS[cells]
        for _ in range(rounds):
            for name, opts in variants:
                sim = make_sim(ny, nx, **opts)
                sim.run_steps(WARMUP)
                t0 = time.perf_counter()
                sim.run_steps(steps)
                elapsed = time.perf_counter() - t0
                if not np.isfinite(sim.f).all():
                    raise RuntimeError(f"{name} diverged at {nx}x{ny}.")
                results[name][cells] = max(
                    results[name].get(cells, 0.0), steps / elapsed
                )
                del sim

    return results


def run_all(**opts: Any) -> dict[str, Any]:
    """Measure every grid in :data:`GRIDS`."""
    rows = []
    for ny, nx in GRIDS:
        cells = ny * nx
        sps = measure(ny, nx, STEPS[cells], **opts)
        rows.append({"ny": ny, "nx": nx, "cells": cells, "steps_per_s": sps})
        print(f"  {nx}x{ny:<4} {cells:>9,} cells   {sps:8.1f} steps/s")
    return {
        "rows": rows,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def print_table(
    baseline: dict[str, Any] | None,
    after: dict[str, Any],
    base_override: dict[int, float] | None = None,
) -> None:
    """The before/after table ``DOCS/TASKS1.md`` § T010 asks for.

    ``base_override`` is the reference path measured **in this process**, which
    is the fair comparison: the archived file was measured on a different day
    against a differently-loaded machine, and the drift between the two is
    larger than some of the wins.
    """
    base = {r["cells"]: r["steps_per_s"] for r in (baseline or {}).get("rows", [])}
    if base_override:
        base = base_override

    print()
    print(
        f"{'grid':>10}  {'cells':>9}  {'before':>9}  {'after':>9}  "
        f"{'speedup':>8}  {'floor':>6}  {'budget':>7}  result"
    )
    print("-" * 82)
    all_pass = True
    for row in after["rows"]:
        cells = row["cells"]
        b = base.get(cells)
        a = row["steps_per_s"]
        floor = FLOORS[cells]
        ok = a >= floor
        all_pass &= ok
        speedup = f"{a / b:.2f}x" if b else "—"
        print(
            f"{row['nx']}x{row['ny']:<5} {cells:>9,}  {_fmt(b):>9}  {_fmt(a):>9}  "
            f"{speedup:>8}  {floor:>6.0f}  {BUDGET[cells]:>7.0f}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print("-" * 82)
    print(f"budget: {'PASS' if all_pass else 'FAIL'}  (floors from DOCS/TASKS1.md § T010)")


def print_variants() -> None:
    """Each of the four cheap wins, measured on its own.

    ``DOCS/TASKS1.md`` § T010: "Applied, each measured separately". A win that
    does nothing can then be dropped rather than carried — § Notes says a win
    costing more than ~20 lines of clarity for under 10% speed is not a win.
    """
    variants: list[tuple[str, dict[str, Any]]] = [
        ("reference (T009 path)", {"fused": False}),
        ("fused collide/bb/stream", {"fused": True}),
    ]

    results = compare(variants)
    ref = results[variants[0][0]]

    for name, _ in variants:
        print(f"\n{name}")
        for ny, nx in GRIDS:
            print(f"  {nx}x{ny:<4} {results[name][ny * nx]:8.1f} steps/s")

    print()
    header = "  ".join(f"{nx}x{ny}" for ny, nx in GRIDS)
    print(f"{'variant':<28}  {header}   (x reference)")
    print("-" * 74)
    for name, _ in variants:
        cols = "  ".join(
            f"{results[name][ny * nx] / ref[ny * nx]:6.2f}x" for ny, nx in GRIDS
        )
        print(f"{name:<28}  {cols}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--save-baseline",
        action="store_true",
        help=f"measure and write {BASELINE_PATH} (run before optimising)",
    )
    p.add_argument(
        "--variants",
        action="store_true",
        help="measure each optimisation separately instead of the before/after table",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="baseline json to compare against",
    )
    args = p.parse_args(argv)

    if args.variants:
        print_variants()
        return 0

    print(f"numpy {np.__version__}, python {platform.python_version()}")
    print("measuring Sim.step (inlet + convective outlet + immersed disc)\n")

    if args.save_baseline:
        after = run_all()
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        after["kind"] = "baseline"
        args.baseline.write_text(json.dumps(after, indent=2), encoding="utf-8")
        print(f"\nbaseline written to {args.baseline}")
        print_table(None, after)
        return 0

    # "before" is the T009 path, still selectable as fused=False, measured here
    # in alternating rounds against the optimised one (see `compare`).
    results = compare([("before", {"fused": False}), ("after", {"fused": True})])
    rows = [
        {"ny": ny, "nx": nx, "cells": ny * nx, "steps_per_s": results["after"][ny * nx]}
        for ny, nx in GRIDS
    ]
    for row in rows:
        print(
            f"  {row['nx']}x{row['ny']:<4} {row['cells']:>9,} cells   "
            f"{results['before'][row['cells']]:8.1f} -> "
            f"{row['steps_per_s']:8.1f} steps/s"
        )

    print_table(None, {"rows": rows}, base_override=results["before"])

    if args.baseline.exists():
        archived = json.loads(args.baseline.read_text(encoding="utf-8"))
        note = "  ".join(
            f"{r['nx']}x{r['ny']}: {r['steps_per_s']:.1f}" for r in archived["rows"]
        )
        print(f"\narchived pre-change baseline (session 10, before any edit): {note}")
        print(
            "  measured cross-process on a differently-loaded machine; the "
            "'before' column above is the same code path measured just now."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
