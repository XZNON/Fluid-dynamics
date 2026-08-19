"""Steps/s benchmark for the LBM kernel, and the before/after table for T010.

``DOCS/IDEA2.md`` § Performance budget is the contract this measures::

    400x100    40k cells   ~500+ steps/s   interactive
    800x200   160k cells   ~150  steps/s   usable
    2000x500    1M cells   ~20   steps/s   record, don't watch

``old-Docs/TASKS1.md`` § T010's pass floors are 400 / 120 / 15 steps/s on those
three grids, and it asks for the baseline to be captured **before** the first
optimisation lands — hence ``--save-baseline``, which writes
``DOCS/bench_baseline.json``, and the default mode, which re-measures and prints
the two columns side by side.

The benchmarked case is deliberately the *whole* step (``lbm.runner.Sim.step``),
not ``collide`` in isolation: it includes the Zou-He inlet, the convective
outlet, bounce-back on an immersed body and the two snapshot copies D-020
requires. Optimising a kernel that is not the one the rungs run is how a
performance pass produces a number nobody can reproduce.

Phase 1 (T103) adds a second budget on top of that one. ``DOCS/IDEA3.md``
§ Performance budget asks the Warp backend for **>=2000 / >=250 / >=150 steps/s
at 40k / 1M / 2M cells**, and the reasoning behind those floors is bandwidth: one
step at 2M cells moves roughly ``9 x 4 bytes x 2M x 2`` = 144 MB, so at a
realistic 60% of the 3050's bandwidth there is about 800 steps/s of headroom and
the floors sit well under it on purpose. ``--backend warp`` measures that column
and prints the device's memory footprint beside it.

**Every absolute number here is quoted with its conditions** (**D-035**): the
CPU clock from ``Win32_Processor.CurrentClockSpeed``, the power state, and — for
a device backend — the GPU name and driver. Ratios survive throttling because
both columns throttle together; absolute numbers do not. :func:`machine_state`
collects them and every mode prints them.

Usage::

    myenv/Scripts/python.exe bench.py --save-baseline   # before optimising
    myenv/Scripts/python.exe bench.py                   # before/after table
    myenv/Scripts/python.exe bench.py --variants        # each win on its own
    myenv/Scripts/python.exe bench.py --backend warp    # the Phase 1 budget
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from lbm.backends import BackendUnavailableError, get_backend
from lbm.geometry import circle
from lbm.runner import Sim, SimConfig

#: The three grids of ``DOCS/IDEA2.md`` § Performance budget, as ``(ny, nx)``.
#: The table names them width x height, so 400x100 is ``ny=100, nx=400``.
GRIDS: tuple[tuple[int, int], ...] = ((100, 400), (200, 800), (500, 2000))

#: ``DOCS/IDEA3.md`` § Performance budget's three grids, as ``(ny, nx)``. The
#: middle one is dropped and 2000x1000 added: Phase 1's floors are quoted at
#: 40k / 1M / 2M cells, and 2M is deliberately near the 4 GB card's ceiling.
GPU_GRIDS: tuple[tuple[int, int], ...] = ((100, 400), (500, 2000), (1000, 2000))

#: ``old-Docs/TASKS1.md`` § T010 pass floors, steps/s, keyed by cell count.
FLOORS: dict[int, float] = {40_000: 400.0, 160_000: 120.0, 1_000_000: 15.0}

#: ``DOCS/IDEA3.md`` § Performance budget floors for a device backend, steps/s.
GPU_FLOORS: dict[int, float] = {40_000: 2000.0, 1_000_000: 250.0, 2_000_000: 150.0}

#: The budget's own expectations, for context in the printed table.
BUDGET: dict[int, float] = {40_000: 500.0, 160_000: 150.0, 1_000_000: 20.0}

#: ``DOCS/IDEA3.md`` § Performance budget's targets, for context.
GPU_BUDGET: dict[int, float] = {40_000: 5000.0, 1_000_000: 600.0, 2_000_000: 400.0}

BASELINE_PATH = Path("DOCS/bench_baseline.json")

#: Timed steps per grid. Chosen so every case runs for a couple of seconds at
#: the *baseline* speed; the same counts are reused after optimising so the two
#: columns measure the same work.
STEPS: dict[int, int] = {40_000: 400, 160_000: 150, 1_000_000: 30}

#: The same idea at GPU speed: a couple of seconds per case against the floors.
GPU_STEPS: dict[int, int] = {40_000: 4000, 1_000_000: 600, 2_000_000: 400}

WARMUP: int = 10


def machine_state() -> dict[str, Any]:
    """The conditions every absolute number in this file must be quoted with.

    **D-035**, generalised by ``DOCS/IDEA3.md`` § Performance budget: an absolute
    steps/s figure from this machine is only meaningful with the CPU clock beside
    it — the identical build measured 696.7 steps/s at the rated 3201 MHz and
    383.6 on battery at 1802 MHz — and, for a device backend, the GPU name and
    driver too.

    Returns:
        ``{"cpu_mhz", "cpu_max_mhz", "power", "cpu", "gpu", "driver"}``. A field
        this platform cannot answer is ``None``; nothing here raises, because a
        benchmark that will not run without WMI is a benchmark nobody runs.
    """
    state: dict[str, Any] = {
        "cpu": platform.processor(),
        "cpu_mhz": None,
        "cpu_max_mhz": None,
        "power": None,
        "gpu": None,
        "driver": None,
    }

    def _wmi(query: str, fields: str) -> str | None:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance {query}).{fields}"],
                capture_output=True, text=True, timeout=30,
            )
            return out.stdout.strip() or None
        except Exception:  # noqa: BLE001 - conditions are context, not a gate
            return None

    mhz = _wmi("Win32_Processor", "CurrentClockSpeed")
    max_mhz = _wmi("Win32_Processor", "MaxClockSpeed")
    battery = _wmi("Win32_Battery", "BatteryStatus")
    if mhz:
        state["cpu_mhz"] = int(mhz.split()[0])
    if max_mhz:
        state["cpu_max_mhz"] = int(max_mhz.split()[0])
    if battery:
        # BatteryStatus 2 is "AC power"; anything else is running on battery,
        # which is the state that cost 160k cells its floor in session 10.
        state["power"] = "mains" if battery.split()[0] == "2" else "battery"
    elif battery is None:
        state["power"] = "unknown"

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        line = out.stdout.strip().splitlines()[0]
        name, driver = (part.strip() for part in line.split(","))
        state["gpu"], state["driver"] = name, driver
    except Exception:  # noqa: BLE001 - no GPU is a fact, not a failure
        pass

    return state


def print_machine_state(state: dict[str, Any], backend: str) -> None:
    """Print the conditions **D-035** requires beside any absolute number.

    Args:
        state: the dict :func:`machine_state` returned.
        backend: the backend the numbers were measured on.
    """
    clock = (
        f"{state['cpu_mhz']} MHz of {state['cpu_max_mhz']} MHz"
        if state["cpu_mhz"]
        else "unknown"
    )
    print(f"backend {backend}   numpy {np.__version__}   "
          f"python {platform.python_version()}")
    print(f"  cpu:   {state['cpu']}")
    print(f"  clock: {clock}   power: {state['power']}   (D-035)")
    if state["gpu"]:
        print(f"  gpu:   {state['gpu']}   driver {state['driver']}")


def device_memory(backend: str) -> tuple[int, int]:
    """Free and total device memory in bytes, or ``(0, 0)``.

    ``DOCS/TASKS2.md`` § T103: *"The GPU memory footprint at 2M cells is printed
    and fits the 4 GB card with room for the display path."* Free memory before
    and after a :class:`lbm.runner.Sim` is built is how that footprint is
    measured, so this is called on both sides.

    Args:
        backend: registry name.

    Returns:
        ``(free, total)`` in bytes; ``(0, 0)`` on a backend with no such notion.
    """
    try:
        be = get_backend(backend)
    except BackendUnavailableError:
        return (0, 0)
    free = getattr(be, "free_memory", None)
    total = getattr(be, "total_memory", None)
    if free is None or total is None:
        return (0, 0)
    return (int(free()), int(total()))


def make_sim(ny: int, nx: int, **opts: Any) -> Sim:
    """A representative open-channel case: inlet, convective outlet, one body.

    The body is a disc of ``D = ny / 10`` a quarter of the way along, which is
    the shape Rung 3 runs. ``check_geometry`` is off because this is a timing
    harness, not a physics run — the sanity checks belong to the rungs, and a
    blockage warning on every bench invocation would train us to ignore them
    (``old-Docs/STATE1.md`` D-018's reasoning, applied to the bench).

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

    if not np.isfinite(sim.host_f()).all():
        raise RuntimeError(
            f"{nx}x{ny} produced non-finite f after {warmup + steps * repeats} "
            "steps — a benchmark that diverged is not measuring the kernel."
        )
    return best


def compare(
    variants: list[tuple[str, dict[str, Any]]],
    rounds: int = 5,
    grids: tuple[tuple[int, int], ...] = GRIDS,
    steps_by_cells: dict[int, int] | None = None,
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
    steps_by_cells = STEPS if steps_by_cells is None else steps_by_cells

    for ny, nx in grids:
        cells = ny * nx
        steps = steps_by_cells[cells]
        for _ in range(rounds):
            for name, opts in variants:
                sim = make_sim(ny, nx, **opts)
                sim.run_steps(WARMUP)
                t0 = time.perf_counter()
                sim.run_steps(steps)
                elapsed = time.perf_counter() - t0
                if not np.isfinite(sim.host_f()).all():
                    raise RuntimeError(f"{name} diverged at {nx}x{ny}.")
                results[name][cells] = max(
                    results[name].get(cells, 0.0), steps / elapsed
                )
                del sim

    return results


def compare_backends(
    backends: tuple[str, ...],
    grids: tuple[tuple[int, int], ...] = GPU_GRIDS,
    steps_for: dict[str, dict[int, int]] | None = None,
    rounds: int = 5,
) -> dict[str, dict[int, float]]:
    """Time several **backends** against each other, interleaved (**D-035**).

    The same protocol :func:`compare` uses for the fused/unfused variants, with
    one difference forced by the spread in speed: each backend gets its own step
    count per grid, because 400 steps is two seconds of Warp at 2M cells and
    nearly a minute of NumPy. The *rate* is what is compared, so the counts may
    differ; the **alternation** is what matters, and it is unchanged — every
    backend is timed once per round, in the same round, and the best round per
    backend is kept, so thermal and scheduling drift hits all of them alike.

    Only one :class:`lbm.runner.Sim` is resident at a time, for the reason
    **D-035** records: holding two variants' buffers at once dropped *both* from
    ~21 to ~16 steps/s at 1M cells, because a cache-locality effect cannot be
    measured under cache pressure.

    Args:
        backends: registry names, in the order to print them.
        grids: ``(ny, nx)`` pairs.
        steps_for: ``{backend: {cells: steps}}``. ``None`` uses
            :data:`GPU_STEPS` for every backend.
        rounds: rounds per grid.

    Returns:
        ``{backend: {cells: steps_per_second}}``.
    """
    results: dict[str, dict[int, float]] = {name: {} for name in backends}

    for ny, nx in grids:
        cells = ny * nx
        for _ in range(rounds):
            for name in backends:
                steps = (GPU_STEPS if steps_for is None else steps_for[name])[cells]
                sim = make_sim(ny, nx, backend=name)
                sim.run_steps(WARMUP)
                t0 = time.perf_counter()
                sim.run_steps(steps)
                elapsed = time.perf_counter() - t0
                if not np.isfinite(sim.host_f()).all():
                    raise RuntimeError(f"{name} diverged at {nx}x{ny}.")
                results[name][cells] = max(
                    results[name].get(cells, 0.0), steps / elapsed
                )
                del sim

    return results


def measure_footprint(backend: str, ny: int, nx: int) -> dict[str, Any]:
    """Device memory a :class:`lbm.runner.Sim` of this size occupies.

    ``DOCS/TASKS2.md`` § T103: *"The GPU memory footprint at 2M cells is printed
    and fits the 4 GB card with room for the display path."* Measured as the
    drop in the device's free memory across construction, rather than computed
    from the buffer list, so that anything the backend allocates behind the seam
    is counted too.

    Args:
        backend: registry name.
        ny: rows.
        nx: columns.

    Two numbers, because either alone lies. **Accounted** sums the capacity of
    every device array the :class:`lbm.runner.Sim` holds, which is exact and
    independent of the allocator. **Observed** is the drop in the device's free
    memory across construction, which is what actually leaves the card — and
    which reads as zero once Warp's memory pool has retained a previous
    allocation of the same size, so it is only meaningful in a fresh process and
    is printed as context rather than as the claim.

    Args:
        backend: registry name.
        ny: rows.
        nx: columns.

    Returns:
        ``{"cells", "accounted", "observed", "free_after", "total",
        "per_buffer", "arrays"}`` — bytes, except ``arrays`` (a count), with
        zeros where the backend has no device memory.
    """
    free_before, total = device_memory(backend)
    sim = make_sim(ny, nx, backend=backend)
    sim.run_steps(2)
    free_after, _ = device_memory(backend)

    accounted = 0
    arrays = 0
    seen: set[int] = set()
    for value in vars(sim).values():
        cap = getattr(value, "capacity", None)
        ptr = getattr(value, "ptr", None)
        if cap is None or ptr is None or ptr in seen:
            continue
        seen.add(ptr)
        accounted += int(cap)
        arrays += 1

    del sim
    return {
        "cells": ny * nx,
        "accounted": accounted,
        "observed": max(0, free_before - free_after),
        "free_after": free_after,
        "total": total,
        "per_buffer": 9 * 4 * ny * nx,
        "arrays": arrays,
    }


def print_gpu_table(
    backend: str,
    results: dict[str, dict[int, float]],
    reference: str | None,
    footprint: dict[str, Any],
) -> bool:
    """``DOCS/IDEA3.md`` § Performance budget, measured.

    Args:
        backend: the backend under test.
        results: what :func:`compare_backends` returned.
        reference: the backend to show a speedup against, or ``None``.
        footprint: what :func:`measure_footprint` returned for the largest grid.

    Returns:
        ``True`` when every floor is cleared.
    """
    print()
    print(
        f"{'grid':>11}  {'cells':>9}  {'numpy':>9}  {backend:>9}  "
        f"{'speedup':>8}  {'floor':>6}  {'target':>7}  result"
    )
    print("-" * 84)
    all_pass = True
    for ny, nx in GPU_GRIDS:
        cells = ny * nx
        a = results[backend][cells]
        b = results[reference][cells] if reference else None
        floor = GPU_FLOORS[cells]
        ok = a >= floor
        all_pass &= ok
        speedup = f"{a / b:.0f}x" if b else "—"
        print(
            f"{nx}x{ny:<6} {cells:>9,}  {_fmt(b):>9}  {a:>9.1f}  "
            f"{speedup:>8}  {floor:>6.0f}  {GPU_BUDGET[cells]:>7.0f}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    print("-" * 84)
    print(
        f"budget: {'PASS' if all_pass else 'FAIL'}  "
        f"(floors from DOCS/IDEA3.md § Performance budget)"
    )

    if footprint["total"]:
        mib = 1024.0 * 1024.0
        print()
        print(
            f"device memory at {footprint['cells']:,} cells: "
            f"{footprint['accounted'] / mib:.0f} MiB in "
            f"{footprint['arrays']} Sim-owned arrays "
            f"({footprint['per_buffer'] / mib:.0f} MiB per (9, ny, nx) buffer), "
            f"{footprint['observed'] / mib:.0f} MiB observed as a drop in free "
            f"memory"
        )
        print(
            f"  {footprint['free_after'] / mib:.0f} MiB still free of "
            f"{footprint['total'] / mib:.0f} MiB — room for the display path "
            f"(one (ny, nx) vorticity field and its RGB frame are "
            f"{4 * footprint['cells'] / mib:.0f} + "
            f"{3 * footprint['cells'] / mib:.0f} MiB, and both live on the host)"
        )
        print(
            "  the observed drop reads 0 once Warp's memory pool has retained an "
            "earlier\n  allocation of the same size; the accounted figure is the "
            "one to trust."
        )
    return all_pass


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
    """The before/after table ``old-Docs/TASKS1.md`` § T010 asks for.

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
    print(f"budget: {'PASS' if all_pass else 'FAIL'}  (floors from old-Docs/TASKS1.md § T010)")


def print_variants() -> None:
    """Each of the four cheap wins, measured on its own.

    ``old-Docs/TASKS1.md`` § T010: "Applied, each measured separately". A win that
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
    p.add_argument(
        "--backend",
        default=None,
        help="measure a device backend against DOCS/IDEA3.md § Performance "
        "budget (>=2000 / >=250 / >=150 steps/s at 40k / 1M / 2M cells), "
        "alternating rounds against numpy (D-035). Omit for the Phase 0 table.",
    )
    p.add_argument(
        "--rounds", type=int, default=5, help="rounds per grid (D-035)"
    )
    args = p.parse_args(argv)

    state = machine_state()

    if args.backend:
        try:
            get_backend(args.backend)
        except BackendUnavailableError as exc:
            print(f"SKIP - {exc}")
            return 2
        print_machine_state(state, args.backend)
        print("measuring Sim.step (inlet + convective outlet + immersed disc)")
        print(
            f"  alternating rounds, best round per backend, one Sim resident "
            f"(D-035); {args.rounds} rounds"
        )
        steps_for = {
            args.backend: GPU_STEPS,
            # NumPy gets fewer steps at the two big grids so a round stays a
            # couple of seconds rather than a minute; the *rate* is compared.
            "numpy": {40_000: 400, 1_000_000: 30, 2_000_000: 20},
        }
        # Measured first, and in this order deliberately: the observed drop in
        # free memory is only a measurement while the allocator's pool is still
        # empty, and every timed round below fills it.
        ny, nx = GPU_GRIDS[-1]
        footprint = measure_footprint(args.backend, ny, nx)
        results = compare_backends(
            (args.backend, "numpy"),
            steps_for=steps_for,
            rounds=args.rounds,
        )
        ok = print_gpu_table(args.backend, results, "numpy", footprint)
        return 0 if ok else 1

    if args.variants:
        print_variants()
        return 0

    print_machine_state(state, "numpy")
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
