"""Rung F — the Smagorinsky closure is switchable, and switching it off is free.

``DOCS/IDEA4.md`` § Validation ladder, Rung F::

    The closure is switchable and does not disturb what already works.
    Cs = 0 reproduces plain BGK **bitwise** on both backends (numpy.array_equal
    on f after 1000 steps); and Rung 3 with the closure **on** still prints
    Cd 1.25-1.45, St 0.155-0.175.

This is the gate for **T201** on numpy and, with T202's ``--backend``, for the
whole of **M9**. Run it from the repo root::

    myenv/Scripts/python.exe -m validate.les
    myenv/Scripts/python.exe -m validate.les --steps 200   # a faster smoke

Why the bitwise clause is first, and why it is written this way
---------------------------------------------------------------
``CLAUDE.md`` constraint 19, from **D-081**: *a closure you cannot switch off is
a closure you cannot validate against*, and nine green rungs are what Phase 2
puts at risk. ``DOCS/PLAN3.md`` § Why this order (3) makes it the first thing
T201 writes, before the model does anything.

"Bitwise what Phase 1 shipped" needs a Phase 1 to compare against, and once
``lbm/core.py`` has been edited there isn't one. So :func:`_phase1_collide` and
:func:`_phase1_collide_stream` below are **frozen verbatim transcriptions** of
those two functions as they stood at the end of Phase 1 (session 22, commit
before T201). They are the oracle, they are deliberately duplicated code, and
**they must never be edited** — an edit that makes this rung pass is an edit
that deletes the rung. If a future task genuinely changes the base arithmetic,
that is a constraint 1 violation and the answer is a decision in
``DOCS/STATE3.md``, not a change here.

The comparison runs the real :class:`lbm.runner.Sim` on **Rung 3's own case**,
swapping only the two collision kernels through the backend seam (**D-054**),
so everything else about the timestep — the D-020 order, the boundaries, the
``f_bb`` snapshot — is shared between the two runs by construction and cannot
be the thing that agrees.

What the three clauses check
----------------------------
1. **Bitwise degeneracy, both paths.** ``cs_smag = 0`` against the frozen Phase
   1 kernels after ``--steps`` steps, fused and unfused, plus fused against
   unfused — which **D-055** already requires and which the closure must not
   break.
2. **The closure engages, and only ever adds viscosity.** On the same case with
   ``Cs = 0.17``: ``tau_eff >= tau`` in every cell, ``nu_t >= 0`` in every cell,
   and ``max(nu_t) > 0`` so that clause 3 is not passing because the model is
   silently inert. The cylinder wake is the strong-shear case the T201 contract
   asks for. ``max(nu_t) / nu`` is printed because it is the quantity
   **D-082**'s fidelity bands are decided from (T204).
3. **Rung 3 survives the closure.** The full Rung 3 case with ``Cs = 0.17``,
   through :func:`validate.cylinder.run_cylinder` itself, still landing in the
   published bands ``Cd`` 1.25-1.45 and ``St`` 0.155-0.175.

Not here, deliberately
----------------------
``--backend`` is **T202**'s (``DOCS/TASKS3.md`` § T202): the Warp kernels do not
exist yet, and :class:`lbm.backends.warp_backend.WarpBackend` raises
``NotImplementedError`` for a non-zero ``cs_smag`` rather than pretending. The
whole-of-Rung-F claim is therefore *numpy only* until that task lands, and this
module says so in its output.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lbm.core import (
    CS_SMAG_LITERATURE,
    E,
    OPP,
    Q,
    _shift_blocks,
    equilibrium,
    macroscopic,
    smagorinsky_tau_eff,
)
from lbm.probe import eddy_viscosity
from lbm.runner import Sim
from validate.cylinder import (
    CD_BAND,
    KICK_FACTOR,
    ST_BAND,
    U,
    cylinder_mask,
    make_config,
    report as cylinder_report,
    run_cylinder,
    tau_for,
)

#: Steps of the bitwise comparison. ``DOCS/IDEA4.md`` § Validation ladder names
#: 1000, and it is a *floor* rather than a target: bitwise equality either holds
#: from step 1 or it does not hold at all, and 1000 steps of a shedding wake is
#: simply enough arithmetic that a divergence anywhere in the collision would
#: have shown up as something other than zero by now.
BITWISE_STEPS: int = 1000

#: The Smagorinsky constant clauses 2 and 3 run at — the literature value, from
#: :mod:`lbm.core`, not a second copy (constraint 4's "no physics constant
#: twice"). Phase 2 does not tune it (``DOCS/TASKS3.md`` § T201 Notes).
CS: float = CS_SMAG_LITERATURE


# --- the frozen Phase 1 oracle. DO NOT EDIT. ---------------------------------
#
# Verbatim transcriptions of lbm.core.collide and lbm.core.collide_stream as
# they stood at the end of Phase 1 (M8, session 22), before T201 gave them a
# `cs_smag` keyword. Docstrings trimmed to the reasoning that bears on the
# arithmetic; every operation, its order and its dtype are unchanged.
#
# `_shift_blocks` is imported rather than transcribed: it is pure slicing with
# no arithmetic in it, T201 did not touch it, and copying it here would make
# this file claim to freeze something it does not actually test.


def _phase1_collide(
    f: NDArray[np.float32], feq: NDArray[np.float32], tau: float
) -> None:
    """FROZEN Phase 1 ``lbm.core.collide``. Do not edit.

    ``f -= (f - feq) / tau``, written as three in-place operations so that no
    temporary is allocated, with ``omega = 1 / tau``.
    """
    if tau <= 0.5:
        raise ValueError(
            f"tau must be greater than 0.5 (got tau={tau!r}): "
            "collision with tau <= 0.5 gives non-positive viscosity and diverges."
        )
    one_minus_omega = np.float32(1.0 - 1.0 / tau)

    f -= feq
    f *= one_minus_omega
    f += feq


def _phase1_collide_stream(
    f: NDArray[np.float32],
    feq: NDArray[np.float32],
    tau: float,
    buf: NDArray[np.float32],
    *,
    f_pre: NDArray[np.float32] | None = None,
    solid: NDArray[np.bool_] | None = None,
    f_bb: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """FROZEN Phase 1 ``lbm.core.collide_stream``. Do not edit.

    Collide, bounce back and stream in one pass per direction (T010, **D-033**,
    **D-055**).
    """
    if tau <= 0.5:
        raise ValueError(
            f"tau must be greater than 0.5 (got {tau!r}): "
            "collision with tau <= 0.5 gives non-positive viscosity and diverges."
        )
    if solid is not None and f_pre is None:
        raise ValueError(
            "collide_stream needs f_pre (the pre-collision copy, D-011) to "
            "bounce back off solid: f_pre[OPP[i]] is the reflection."
        )

    one_minus_omega = np.float32(1.0 - 1.0 / tau)

    for i in range(Q):
        s = f[i] if f_bb is None else f_bb[i]

        np.subtract(f[i], feq[i], out=s)
        s *= one_minus_omega
        s += feq[i]

        if solid is not None:
            np.copyto(s, f_pre[OPP[i]], where=solid)

        ex = int(E[i, 0])
        ey = int(E[i, 1])
        dst = buf[i]
        for dy, sy in _shift_blocks(ey):
            for dx, sx in _shift_blocks(ex):
                dst[dy, dx] = s[sy, sx]

    np.copyto(f, buf)
    return f


class Phase1Backend:
    """The NumPy backend with its two collision kernels frozen at Phase 1.

    Everything except :meth:`collide` and :meth:`collide_stream` is forwarded to
    the real backend, so the two runs being compared share every buffer
    allocation, every boundary condition and the whole D-020 ordering. What
    differs between them is exactly the arithmetic under test, which is the
    point of the T101 seam (**D-054**).

    Attributes:
        name: ``"phase1"`` — never registered; this is a test double, reached by
            assigning to :attr:`lbm.runner.Sim.backend` after construction.
    """

    name: str = "phase1"

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def __getattr__(self, item: str) -> object:
        return getattr(self.inner, item)

    def collide(
        self,
        f: NDArray[np.float32],
        feq: NDArray[np.float32],
        tau: float,
        *,
        cs_smag: float = 0.0,
        smag_out: NDArray[np.float32] | None = None,
        smag_work: NDArray[np.float32] | None = None,
    ) -> None:
        """Phase 1's collision. A non-zero ``cs_smag`` is a bug in the caller."""
        if cs_smag != 0.0:
            raise AssertionError(
                "the frozen Phase 1 oracle has no closure; it is what "
                "cs_smag = 0 is compared against."
            )
        _phase1_collide(f, feq, tau)

    def collide_stream(
        self,
        f: NDArray[np.float32],
        feq: NDArray[np.float32],
        tau: float,
        buf: NDArray[np.float32],
        *,
        f_pre: NDArray[np.float32] | None = None,
        solid: NDArray[np.bool_] | None = None,
        f_bb: NDArray[np.float32] | None = None,
        cs_smag: float = 0.0,
        smag_out: NDArray[np.float32] | None = None,
        smag_work: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Phase 1's fused pass. A non-zero ``cs_smag`` is a bug in the caller."""
        if cs_smag != 0.0:
            raise AssertionError(
                "the frozen Phase 1 oracle has no closure; it is what "
                "cs_smag = 0 is compared against."
            )
        return _phase1_collide_stream(
            f, feq, tau, buf, f_pre=f_pre, solid=solid, f_bb=f_bb
        )


# --- the case ----------------------------------------------------------------


@dataclass
class Case:
    """Rung 3's case, built once and run several ways.

    Attributes:
        solid: the full mask, ``(ny, nx)`` ``bool``.
        cylinder: the disc alone, ``(ny, nx)`` ``bool``.
        d_cells: the **measured** diameter (D-019).
        tau: the relaxation time Re 100 implies.
        nu: ``(tau - 0.5) / 3``, lattice units.
    """

    solid: NDArray[np.bool_]
    cylinder: NDArray[np.bool_]
    d_cells: float
    tau: float
    nu: float


def build_case(backend: str = "numpy") -> tuple[Case, object]:
    """Rung 3's mask and ``tau``, and a :class:`lbm.runner.SimConfig` factory.

    Exactly :mod:`validate.cylinder`'s own setup — the mask, the measured
    diameter, ``tau`` from ``Re``, the startup kick — so that "bitwise on Rung
    3's case" means the case Rung 3 actually runs.

    Args:
        backend: the T101 registry name. Only ``"numpy"`` is meaningful until
            T202.

    Returns:
        ``(case, make)`` where ``make(fused, cs_smag)`` builds the config.
    """
    from validate.cylinder import RE, D_CELLS
    from lbm.geometry import bounding_box

    solid, cylinder, _, _ = cylinder_mask(D_CELLS)
    box = bounding_box(cylinder)
    assert box is not None
    d_cells = float(box[1] - box[0] + 1)
    nu, tau = tau_for(RE, U, d_cells)
    ny, nx = solid.shape

    def make(*, fused: bool, cs_smag: float) -> object:
        cfg = make_config(
            ny=ny,
            nx=nx,
            tau=tau,
            u=U,
            outlet_lam=None,
            verbose_mask=False,
            inlet_uy=KICK_FACTOR * U,
            backend=backend,
            cs_smag=cs_smag,
        )
        return cfg.replace(check_geometry=False, fused=fused)

    return Case(solid, cylinder, d_cells, tau, nu), make


def _run(cfg: object, solid: NDArray[np.bool_], steps: int, *, phase1: bool) -> Sim:
    """``steps`` timesteps of ``cfg``, optionally on the frozen Phase 1 kernels.

    The oracle is installed **after** construction, so both runs allocate their
    state through the same backend and start from a byte-identical ``f``.
    """
    sim = Sim(cfg, solid)  # type: ignore[arg-type]
    if phase1:
        sim.backend = Phase1Backend(sim.backend)  # type: ignore[assignment]
    sim.run_steps(steps)
    return sim


# --- clause 1: bitwise degeneracy --------------------------------------------


@dataclass
class BitwiseResult:
    """What clause 1 measured.

    Attributes:
        steps: timesteps each run advanced.
        fused_equal: closure-off fused ``f`` equals the frozen Phase 1 fused
            ``f`` under :func:`numpy.array_equal`.
        unfused_equal: the same on the unfused path.
        paths_equal: closure-off fused equals closure-off unfused (**D-055**).
        fused_ulps: worst absolute difference on the fused path — ``0.0`` when
            it passes, and a number worth printing when it does not.
        unfused_ulps: the same for the unfused path.
        seconds: wall clock for all four runs.
    """

    steps: int
    fused_equal: bool
    unfused_equal: bool
    paths_equal: bool
    fused_diff: float
    unfused_diff: float
    seconds: float


def check_bitwise(case: Case, make, steps: int) -> BitwiseResult:
    """Constraint 19: ``cs_smag = 0`` is Phase 1's collision, bit for bit.

    Four runs of Rung 3's case: ``{fused, unfused}`` x ``{today, frozen}``.
    """
    start = time.perf_counter()

    new_fused = _run(make(fused=True, cs_smag=0.0), case.solid, steps, phase1=False)
    old_fused = _run(make(fused=True, cs_smag=0.0), case.solid, steps, phase1=True)
    new_plain = _run(make(fused=False, cs_smag=0.0), case.solid, steps, phase1=False)
    old_plain = _run(make(fused=False, cs_smag=0.0), case.solid, steps, phase1=True)

    a, b = new_fused.host_f(), old_fused.host_f()
    c, d = new_plain.host_f(), old_plain.host_f()

    return BitwiseResult(
        steps=steps,
        fused_equal=bool(np.array_equal(a, b)),
        unfused_equal=bool(np.array_equal(c, d)),
        paths_equal=bool(np.array_equal(a, c)),
        fused_diff=float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()),
        unfused_diff=float(np.abs(c.astype(np.float64) - d.astype(np.float64)).max()),
        seconds=time.perf_counter() - start,
    )


# --- clause 2: the closure engages, and only ever adds viscosity -------------


@dataclass
class ClosureResult:
    """What clause 2 measured, on ``Cs = 0.17`` after ``steps`` steps.

    Attributes:
        steps: timesteps advanced.
        cs: the Smagorinsky constant used.
        tau: the base relaxation time.
        nu: ``(tau - 0.5) / 3``.
        tau_eff_min: smallest ``tau_eff`` over the domain.
        tau_eff_max: largest ``tau_eff`` over the domain.
        nu_t_max: largest eddy viscosity over the domain, lattice units.
        nu_t_mean: domain average eddy viscosity.
        nu_t_min: smallest eddy viscosity — must not be negative.
        nu_t_zero_at_zero: ``eddy_viscosity(..., cs_smag=0)`` is exactly zero
            everywhere on the same state.
        seconds: wall clock.
    """

    steps: int
    cs: float
    tau: float
    nu: float
    tau_eff_min: float
    tau_eff_max: float
    nu_t_min: float
    nu_t_max: float
    nu_t_mean: float
    nu_t_zero_at_zero: bool
    seconds: float


def check_closure(case: Case, make, steps: int, cs: float = CS) -> ClosureResult:
    """The closure fires, ``tau_eff >= tau`` everywhere, ``nu_t >= 0`` everywhere.

    A shedding cylinder wake is the strong-shear case the T201 contract asks
    for. ``max(nu_t) / nu`` is the quantity **D-082**'s fidelity bands are read
    off, so it is printed here even though nothing in T201 consumes it.
    """
    start = time.perf_counter()
    sim = _run(make(fused=True, cs_smag=cs), case.solid, steps, phase1=False)

    f = sim.host_f()
    rho, u = macroscopic(f.copy())
    feq = equilibrium(rho, u)

    tau_eff = smagorinsky_tau_eff(f, feq, case.tau, cs)
    nu_t = eddy_viscosity(f, feq, case.tau, cs)
    nu_t0 = eddy_viscosity(f, feq, case.tau, 0.0)

    return ClosureResult(
        steps=steps,
        cs=cs,
        tau=case.tau,
        nu=case.nu,
        tau_eff_min=float(tau_eff.min()),
        tau_eff_max=float(tau_eff.max()),
        nu_t_min=float(nu_t.min()),
        nu_t_max=float(nu_t.max()),
        nu_t_mean=float(nu_t.mean()),
        nu_t_zero_at_zero=bool((nu_t0 == 0.0).all()),
        seconds=time.perf_counter() - start,
    )


# --- reporting ---------------------------------------------------------------


def report(
    bit: BitwiseResult,
    clo: ClosureResult,
    cyl_passed: bool | None,
    backend: str,
) -> bool:
    """Print every check and return whether Rung F passed."""
    print()
    print("  measured")
    print(f"    bitwise clause   4 x {bit.steps} steps in {bit.seconds:.1f} s")
    print(f"    closure clause   {clo.steps} steps at Cs = {clo.cs} "
          f"in {clo.seconds:.1f} s")
    print(f"    tau              {clo.tau:.6f}   nu = (tau - 0.5)/3 = "
          f"{clo.nu:.6f}")
    print(f"    tau_eff          {clo.tau_eff_min:.6f} .. {clo.tau_eff_max:.6f}")
    print(f"    nu_t             {clo.nu_t_min:.3e} .. {clo.nu_t_max:.3e}   "
          f"mean {clo.nu_t_mean:.3e}")
    print(f"    max(nu_t) / nu   {clo.nu_t_max / clo.nu:.4f}   "
          f"(the quantity D-082's fidelity bands are read off; T204 consumes it)")

    checks: list[tuple[str, bool, str]] = [
        (
            "cs_smag=0 fused is bitwise Phase 1 (constraint 19)",
            bit.fused_equal,
            f"array_equal after {bit.steps} steps, "
            f"worst |diff| {bit.fused_diff:.3e}",
        ),
        (
            "cs_smag=0 unfused is bitwise Phase 1 (constraint 19)",
            bit.unfused_equal,
            f"array_equal after {bit.steps} steps, "
            f"worst |diff| {bit.unfused_diff:.3e}",
        ),
        (
            "fused and unfused agree bitwise (D-055)",
            bit.paths_equal,
            f"array_equal after {bit.steps} steps",
        ),
        (
            "tau_eff >= tau in every cell (the closure only adds)",
            clo.tau_eff_min >= clo.tau,
            f"min tau_eff {clo.tau_eff_min:.6f} vs tau {clo.tau:.6f}",
        ),
        (
            "nu_t >= 0 in every cell (constraint 2 through tau)",
            clo.nu_t_min >= 0.0,
            f"min {clo.nu_t_min:.3e}",
        ),
        (
            "the closure is not inert: max(nu_t) > 0",
            clo.nu_t_max > 0.0,
            f"max {clo.nu_t_max:.3e} = {clo.nu_t_max / clo.nu:.2%} of nu",
        ),
        (
            "nu_t is exactly 0 when cs_smag = 0",
            clo.nu_t_zero_at_zero,
            "on the same state, every cell",
        ),
    ]

    if cyl_passed is not None:
        cd_lo, cd_hi = CD_BAND
        st_lo, st_hi = ST_BAND
        checks.append(
            (
                f"Rung 3 with Cs = {clo.cs}: Cd {cd_lo}-{cd_hi}, "
                f"St {st_lo}-{st_hi}",
                cyl_passed,
                "validate.cylinder's own report, above",
            )
        )

    width = max(len(name) for name, _, _ in checks)
    print()
    for name, ok, detail in checks:
        print(f"    [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    passed = all(ok for _, ok, _ in checks)
    print()
    if backend == "numpy":
        print("  scope: numpy only. The Warp half of Rung F is T202, which is")
        print("         also what answers Q-201 (DOCS/STATE3.md).")
    print("PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run Rung F and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Rung F — the Smagorinsky closure is switchable "
        "(Cs = 0 is bitwise BGK; Rung 3 survives Cs = 0.17)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=BITWISE_STEPS,
        help=f"steps for the bitwise and closure clauses (default {BITWISE_STEPS})",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=CS,
        help=f"Smagorinsky constant for clauses 2 and 3 (default {CS}, the "
        f"literature value; Phase 2 does not tune it)",
    )
    parser.add_argument(
        "--skip-cylinder",
        action="store_true",
        help="skip clause 3, the full Rung 3 case with the closure on. It is "
        "the expensive clause (~45k steps) and this makes the cheap ones "
        "runnable in a few seconds. A skipped clause is reported, never "
        "silently dropped.",
    )
    args = parser.parse_args(argv)

    # Nothing here draws, and this makes that true even for a stray SDL init.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    backend = "numpy"

    print("Rung F — the Smagorinsky closure, switchable "
          "(DOCS/IDEA4.md § Validation ladder)")
    print(f"  backend {backend}   Cs = {args.cs} (literature; "
          f"lbm.core.CS_SMAG_LITERATURE)   filter width 1 lattice unit")
    print("  oracle: the frozen Phase 1 collide / collide_stream in this file")
    print()

    case, make = build_case(backend)
    ny, nx = case.solid.shape
    print(f"  Rung 3's case: grid {ny} x {nx}   D = {case.d_cells:.0f} cells   "
          f"tau {case.tau:.6f}   nu {case.nu:.6f}")
    print(f"  running 4 x {args.steps} steps for the bitwise clause ...",
          flush=True)
    bit = check_bitwise(case, make, args.steps)

    print(f"  running {args.steps} steps at Cs = {args.cs} for the closure "
          f"clause ...", flush=True)
    clo = check_closure(case, make, args.steps, args.cs)

    cyl_passed: bool | None = None
    if not args.skip_cylinder:
        print()
        print("  clause 3 — the full Rung 3 case with the closure on:")
        print()
        res = run_cylinder(
            headless=True,
            bench_steps=0,
            verbose_mask=False,
            backend=backend,
            cs_smag=args.cs,
        )
        cyl_passed = cylinder_report(res)
        print()
        print("  (back in Rung F)")
    else:
        print()
        print("  clause 3 SKIPPED by --skip-cylinder: Rung 3 with the closure "
              "on was not run.")

    passed = report(bit, clo, cyl_passed, backend)
    if args.skip_cylinder:
        print("  NOTE: clause 3 was skipped, so this is not a full Rung F pass.")
        return 0 if passed else 1
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
