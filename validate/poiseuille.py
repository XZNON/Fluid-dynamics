"""Rung 1 — Poiseuille flow in a force-driven channel.

``DOCS/IDEA2.md`` § "Validation ladder", Rung 1::

    Empty channel, no-slip top and bottom, body force.
    Exact answer: u(y) = (G / 2nu) * y * (H - y)
    Pass condition: L2 error under 1%, and halving (tau - 0.5) doubles
    centreline velocity. This catches every sign error in collide.

This is the gate for **M1**. Run it from the repo root::

    myenv/Scripts/python.exe -m validate.poiseuille

Geometry here is two solid rows built inline with NumPy. That is deliberate —
``lbm/geometry.py`` and its ``channel_walls`` helper are T004, and Rung 1 must
not wait on them.

Wall-offset convention: half-way bounce-back, so the walls are the planes
``y = 0.5`` and ``y = ny - 1.5``, the channel height is ``H = ny - 2``, and the
analytic profile is evaluated at ``y_ = y - 0.5``. See
:func:`lbm.boundary.bounce_back` and ``old-Docs/STATE1.md`` D-009.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lbm.backends import get_backend
from lbm.core import Q, W, nu_from_tau

# --- case definition ---------------------------------------------------------

NY: int = 22  # 2 solid rows + 20 fluid rows -> H = 20
NX: int = 16  # periodic in x; the solution is x-invariant, so this is plenty
TAU: float = 0.6

#: Body force in x, lattice units. Chosen so the base case peaks near 0.04 and
#: the halved-viscosity case near 0.08 — both comfortably under the 0.1 lattice
#: velocity ceiling (``CLAUDE.md`` constraint 3), which the run asserts.
GX: float = 2.6667e-5

MAX_STEPS: int = 60_000
CHECK_EVERY: int = 100

#: Convergence threshold on ``max|du| / peak|u|`` measured over ``CHECK_EVERY``
#: steps. It cannot be tightened much below this: in ``float32``, ``u`` is a
#: difference of ``f`` values of order 0.1 divided by rho, so its round-off floor
#: is about ``eps * |f| / |u| ~ 1.2e-7 * 0.4 / 0.04 = 1.2e-6`` per step-to-step
#: comparison. Measured floor is 1.7e-6; asking for 1e-9 just runs to the cap.
RESIDUAL_TOL: float = 5e-6


@dataclass
class Result:
    """What one channel run measured."""

    tau: float
    nu: float
    steps: int
    residual: float
    ux_profile: NDArray[np.float64]  # (ny - 2,), fluid rows only
    ux_analytic: NDArray[np.float64]  # (ny - 2,)
    peak_u: float
    mass_drift: float
    l2_error: float
    centreline: float
    saw_nan: bool


def channel_mask(ny: int, nx: int) -> NDArray[np.bool_]:
    """Solid top and bottom rows, shape ``(ny, nx)`` (``CLAUDE.md`` c12)."""
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


def analytic_profile(
    ny: int, gx: float, nu: float, wall_offset: float = 0.5
) -> NDArray[np.float64]:
    """``u(y) = (G / 2nu) * y * (H - y)`` on the fluid rows.

    ``DOCS/IDEA2.md`` § Validation ladder, Rung 1. With half-way bounce-back the
    wall planes are half a cell outside the last fluid node, so ``H = ny - 2``
    and fluid row ``y`` sits at ``y - 0.5``.

    Args:
        ny: rows including the two solid ones.
        gx: body force per unit volume.
        nu: kinematic viscosity, from :func:`lbm.core.nu_from_tau`.
        wall_offset: distance from the last fluid node to the wall plane. 0.5 is
            the half-way convention this code actually implements (D-009); 0.0
            (wall on the last fluid node, ``H = ny - 3``) and 1.0 (wall on the
            solid node, ``H = ny - 1``) are the two rival conventions, offered
            here so :func:`main` can print the comparison rather than assert it
            from memory.

    Returns:
        ``ux`` on the ``ny - 2`` fluid rows, ``float64``.
    """
    y = np.arange(1, ny - 1, dtype=np.float64) - (1.0 - wall_offset)
    h = float(ny - 2) - 2.0 * (0.5 - wall_offset)
    return (gx / (2.0 * nu)) * y * (h - y)


def run_channel(
    ny: int = NY,
    nx: int = NX,
    tau: float = TAU,
    gx: float = GX,
    max_steps: int = MAX_STEPS,
    residual_tol: float = RESIDUAL_TOL,
    mass_drift_steps: int = 5000,
    backend: str = "numpy",
) -> Result:
    """Drive an empty channel with a constant body force to steady state.

    Every buffer is allocated once, before the loop, and passed into the solver
    functions (``CLAUDE.md`` § conventions). The timestep order is the one
    documented in :mod:`lbm.boundary`: copy, macroscopic, force-corrected
    velocity, equilibrium, collide, force source, bounce-back, stream.

    Args:
        ny: rows, including the two solid ones.
        nx: columns; periodic.
        tau: BGK relaxation time. ``nu`` follows from it and nothing else
            (``CLAUDE.md`` constraint 2).
        gx: body force per unit volume in ``+x``, lattice units.
        max_steps: hard cap; the run normally converges well before it.
        residual_tol: convergence threshold on ``max|du| / peak|u|`` measured
            over ``CHECK_EVERY`` steps.
        mass_drift_steps: total mass is sampled at step 0 and at this step, and
            the relative difference is reported.
        backend: which :class:`lbm.backends.Backend` runs the kernels (T101's
            seam, T103's flag). ``"numpy"`` is the oracle (**D-043**). This is
            the only rung that exercises **both halves of the Guo body force**,
            because it is the only case that switches the scheme on at all
            (**D-010**, **D-033**).

    Returns:
        A :class:`Result` with the measured profile, errors and diagnostics.
    """
    nu = nu_from_tau(tau)
    solid = channel_mask(ny, nx)
    fluid_rows = slice(1, ny - 1)
    g = (gx, 0.0)
    be = get_backend(backend)

    # --- preallocation; nothing below the loop header allocates ---
    # Through the backend since T103, so the state lives wherever the kernels
    # do. On ``"numpy"`` these are the same ``np.empty`` arrays as in Phase 0.
    f = be.empty((Q, ny, nx))
    f_pre = be.empty((Q, ny, nx))
    feq = be.empty((Q, ny, nx))
    buf = be.empty((Q, ny, nx))
    rho = be.empty((ny, nx))
    u = be.empty((2, ny, nx))
    work = be.empty((3, ny, nx))
    solid_dev = be.upload(solid)
    ux_prev = np.zeros((ny, nx), dtype=np.float32)

    # Rest state: f = w_i * rho with rho = 1, u = 0.
    rest = np.empty((Q, ny, nx), dtype=np.float32)
    rest[:] = W[:, None, None]
    be.upload(rest, dst=f)

    mass0 = float(np.sum(rest, dtype=np.float64))
    mass_at_drift_step = mass0
    peak_u = 0.0
    saw_nan = False
    residual = np.inf
    steps = 0

    for step in range(1, max_steps + 1):
        be.copy(f_pre, f)

        be.macroscopic(f, rho, u)
        be.force_velocity_shift(rho, u, g, work)
        be.equilibrium(rho, u, feq, work)
        be.collide(f, feq, tau)
        be.apply_body_force(f, rho, u, tau, g, work)
        be.bounce_back(f, f_pre, solid_dev)
        be.stream(f, buf)

        steps = step

        if step == mass_drift_steps:
            mass_at_drift_step = float(np.sum(be.download(f), dtype=np.float64))

        if step % CHECK_EVERY == 0:
            # The only host transfer in the loop, and it is on the *check*
            # cadence rather than the step cadence (constraint 8): 22x16 cells,
            # once every 100 steps.
            f_host = be.download(f)
            u_host = be.download(u)
            if not np.isfinite(f_host).all():
                saw_nan = True
                break
            ux = u_host[0]
            peak = float(np.max(np.abs(u_host[:, fluid_rows, :])))
            peak_u = max(peak_u, peak)
            if peak > 0.0:
                residual = float(np.max(np.abs(ux - ux_prev))) / peak
            np.copyto(ux_prev, ux)
            if residual < residual_tol and step > mass_drift_steps:
                break

    # Final macroscopic state, force-corrected the same way the loop does it.
    be.macroscopic(f, rho, u)
    be.force_velocity_shift(rho, u, g, work)
    u_host = be.download(u)
    peak_u = max(peak_u, float(np.max(np.abs(u_host[:, fluid_rows, :]))))

    ux_profile = np.asarray(
        u_host[0, fluid_rows, :].mean(axis=1), dtype=np.float64
    )
    ux_analytic = analytic_profile(ny, gx, nu)
    l2 = float(
        np.linalg.norm(ux_profile - ux_analytic) / np.linalg.norm(ux_analytic)
    )

    mass_now = mass_at_drift_step if steps >= mass_drift_steps else float(
        np.sum(be.download(f), dtype=np.float64)
    )
    mass_drift = abs(mass_now - mass0) / abs(mass0)

    return Result(
        tau=tau,
        nu=nu,
        steps=steps,
        residual=residual,
        ux_profile=ux_profile,
        ux_analytic=ux_analytic,
        peak_u=peak_u,
        mass_drift=mass_drift,
        l2_error=l2,
        centreline=float(ux_profile.max()),
        saw_nan=saw_nan,
    )


def nan_check(
    steps: int = 20_000, tau: float = 0.6, backend: str = "numpy"
) -> bool:
    """True if ``f`` is still finite after ``steps`` steps at ``tau``.

    A separate short run rather than a flag on the main one, because the main
    run stops at convergence and would otherwise never reach 20000 steps.
    """
    res = run_channel(
        tau=tau, max_steps=steps, residual_tol=0.0, backend=backend
    )
    return not res.saw_nan and np.isfinite(res.ux_profile).all()


def main(argv: list[str] | None = None) -> int:
    """Run every Rung 1 check and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Rung 1 — Poiseuille flow")
    parser.add_argument("--ny", type=int, default=NY)
    parser.add_argument("--nx", type=int, default=NX)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--gx", type=float, default=GX)
    parser.add_argument(
        "--backend",
        default="numpy",
        help="compute backend (T101 seam): 'numpy' (the oracle, D-043) or "
        "'warp'. The published band is the same band either way.",
    )
    args = parser.parse_args(argv)

    tau = args.tau
    tau_half = 0.5 + (tau - 0.5) / 2.0  # halves (tau - 0.5), hence halves nu

    print("Rung 1 — Poiseuille flow (DOCS/IDEA2.md § Validation ladder)")
    print(f"  grid {args.ny} x {args.nx}   H = {args.ny - 2}   "
          f"gx = {args.gx:.4e}   backend = {args.backend}")
    print("  walls: half-way bounce-back, planes at y = 0.5 and y = ny - 1.5")
    print()

    base = run_channel(
        ny=args.ny, nx=args.nx, tau=tau, gx=args.gx, backend=args.backend
    )
    half = run_channel(
        ny=args.ny, nx=args.nx, tau=tau_half, gx=args.gx, backend=args.backend
    )

    print(
        f"  base   tau = {base.tau:.4f}  nu = {base.nu:.6f}  "
        f"steps = {base.steps}  residual = {base.residual:.2e}"
    )
    print(
        f"  halved tau = {half.tau:.4f}  nu = {half.nu:.6f}  "
        f"steps = {half.steps}  residual = {half.residual:.2e}"
    )
    print()

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "L2 relative error vs u(y) = (G/2nu) y (H-y) < 1%",
            base.l2_error < 0.01,
            f"{base.l2_error * 100:.4f}%",
        )
    )

    ratio = half.centreline / base.centreline
    checks.append(
        (
            "halving (tau - 0.5) doubles centreline velocity, within 2%",
            abs(ratio - 2.0) / 2.0 < 0.02,
            f"ratio = {ratio:.5f} "
            f"({base.centreline:.6f} -> {half.centreline:.6f})",
        )
    )

    checks.append(
        (
            "mass drift over 5000 steps < 1e-4 relative",
            base.mass_drift < 1e-4,
            f"{base.mass_drift:.3e}",
        )
    )

    no_nan = nan_check(steps=20_000, tau=0.6, backend=args.backend)
    checks.append(("no nan after 20000 steps at tau = 0.6", no_nan, "finite"))

    peak = max(base.peak_u, half.peak_u)
    checks.append(
        (
            "peak lattice velocity < 0.1 (CLAUDE.md constraint 3)",
            peak < 0.1,
            f"peak |u| = {peak:.5f} "
            f"(base {base.peak_u:.5f}, halved {half.peak_u:.5f})",
        )
    )

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    print()
    print("  wall-offset convention (closes Q-001; the code implements 0.5):")
    for offset, label in ((0.0, "H = ny-3, wall on last fluid node"),
                          (0.5, "H = ny-2, wall halfway  <-- D-009"),
                          (1.0, "H = ny-1, wall on solid node")):
        ana = analytic_profile(args.ny, args.gx, base.nu, wall_offset=offset)
        err = float(np.linalg.norm(base.ux_profile - ana) / np.linalg.norm(ana))
        print(f"    offset {offset:.1f}  {label:<36}  L2 = {err * 100:7.3f}%")

    print()
    print("  y      u_measured     u_analytic     abs error")
    y = np.arange(1, args.ny - 1) - 0.5
    for yy, um, ua in zip(y, base.ux_profile, base.ux_analytic):
        print(f"  {yy:5.1f}  {um: .8f}  {ua: .8f}  {abs(um - ua): .2e}")

    passed = all(ok for _, ok, _ in checks)
    print()
    print("PASS" if passed else "FAIL")
    print(f"  L2 relative error: {base.l2_error * 100:.4f}%")
    print(f"  peak lattice velocity: {peak:.5f}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
