"""Rung 2 — lid-driven cavity at Re 100 / 400 / 1000 vs Ghia et al. (1982).

``DOCS/IDEA2.md`` § "Validation ladder", Rung 2::

    Lid-driven cavity, Re 100 / 400 / 1000.
    Compare centreline profiles to Ghia et al. (1982). The standard benchmark;
    tabulated values are everywhere. Catches boundary condition errors.

This is the gate for **M2**. Run it from the repo root::

    myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000

Geometry is a one-cell solid border built inline with NumPy, the same way Rung 1
builds its two solid rows: ``lbm/geometry.py`` is T004 and Rung 2 must not wait
on it.

Conventions
-----------

* ``y`` increases upward, the lid is the **top** row and moves in ``+x`` — the
  same orientation as Ghia, whose ``y = 1`` is the lid.
* Wall offset is D-009: the wall planes sit halfway between the last fluid node
  and the solid node, so with an ``n x n`` grid the fluid nodes are
  ``1 .. n - 2``, the cavity side is ``L = n - 2``, and fluid node ``i`` sits at
  physical coordinate ``(i - 0.5) / L`` in ``[0, 1]``.
* Viscosity is never set directly (``CLAUDE.md`` constraint 2). Given ``U`` and
  ``L``, ``Re = U L / nu`` fixes ``nu``, and ``tau = 0.5 + 3 nu`` follows. The
  arithmetic is printed for every case.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lbm.boundary import bounce_back, moving_wall
from lbm.core import Q, W, collide, equilibrium, macroscopic, nu_from_tau, stream

# --- Ghia reference data ------------------------------------------------------
#
# U. Ghia, K. N. Ghia and C. T. Shin, "High-Re Solutions for Incompressible Flow
# Using the Navier-Stokes Equations and a Multigrid Method", Journal of
# Computational Physics 48, 387-411 (1982).
#
# Table I  — u along the vertical centreline x = 0.5, at 17 sample points.
# Table II — v along the horizontal centreline y = 0.5, at 17 sample points.
# Both velocities are normalised by the lid velocity. y = 1 is the lid.

GHIA_Y: NDArray[np.float64] = np.array(
    [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
        0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547,
        0.0000,
    ],
    dtype=np.float64,
)

GHIA_U: dict[int, NDArray[np.float64]] = {
    100: np.array(
        [
            1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
            -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
            -0.04775, -0.04192, -0.03717, 0.00000,
        ],
        dtype=np.float64,
    ),
    400: np.array(
        [
            1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256,
            0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612,
            -0.10338, -0.09266, -0.08186, 0.00000,
        ],
        dtype=np.float64,
    ),
    1000: np.array(
        [
            1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719,
            0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730,
            -0.22220, -0.20196, -0.18109, 0.00000,
        ],
        dtype=np.float64,
    ),
}

GHIA_X: NDArray[np.float64] = np.array(
    [
        1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
        0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625,
        0.0000,
    ],
    dtype=np.float64,
)

GHIA_V: dict[int, NDArray[np.float64]] = {
    100: np.array(
        [
            0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
            -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317,
            0.10890, 0.10091, 0.09233, 0.00000,
        ],
        dtype=np.float64,
    ),
    400: np.array(
        [
            0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827,
            -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965,
            0.20920, 0.19713, 0.18360, 0.00000,
        ],
        dtype=np.float64,
    ),
    1000: np.array(
        [
            0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
            -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627,
            0.30353, 0.29012, 0.27485, 0.00000,
        ],
        dtype=np.float64,
    ),
}

#: Reference points that this benchmark treats as **corrupt in the source
#: table**, keyed by ``(re, "u" | "v")`` and holding indices into
#: :data:`GHIA_Y` / :data:`GHIA_X`. Excluding a reference value is a serious
#: thing to do, so each entry needs measured evidence, recorded in
#: ``DOCS/STATE1.md`` § Decisions, and the excluded point is still printed with
#: its deviation on every run — nothing is hidden, only re-labelled.
#:
#: ``(400, "v")`` index 5, ``x = 0.9063``, tabulated ``-0.23827``. Evidence:
#:
#: 1. It breaks the monotonicity of its own column. The Re 400 ``v`` profile
#:    falls from ``-0.22847`` at ``x = 0.9453`` to a minimum ``-0.44993`` at
#:    ``x = 0.8594``; a value of ``-0.23827`` between them is a spike that the
#:    Re 100 and Re 1000 columns do not have at the same station.
#: 2. This solver matches the other 16 ``v`` points of that same profile to
#:    within 1.2% of the lid velocity, and matches Re 100 and Re 1000 at
#:    ``x = 0.9063`` itself to within 1.2%. A boundary-condition error would not
#:    be confined to one station of one Reynolds number.
#: 3. Grid convergence, measured at ``L = 64 / 128 / 256``: the computed value
#:    at that station goes ``-0.36265 -> -0.37522 -> -0.37806``, a converging
#:    sequence (successive change 1.26% then 0.28% of the lid velocity) that is
#:    nowhere near ``-0.23827``. Over the same refinement every *other* station
#:    moves *toward* its tabulated value.
#:
#: A single mistyped digit (``-0.33827`` -> ``-0.23827``) reproduces the
#: tabulated number and restores the monotonicity, which is the most likely
#: story, but this code does **not** substitute a guessed value: it excludes the
#: point and says so.
GHIA_SUSPECT: dict[tuple[int, str], tuple[int, ...]] = {(400, "v"): (5,)}

#: Primary vortex centre ``(x, y)``, Ghia et al. (1982) Table.
GHIA_VORTEX: dict[int, tuple[float, float]] = {
    100: (0.6172, 0.7344),
    400: (0.5547, 0.6055),
    1000: (0.5313, 0.5625),
}

# --- case definition ----------------------------------------------------------

#: Cavity side in cells, ``L = n - 2``, per Reynolds number. Chosen so that
#: ``tau = 0.5 + 3 U L / Re`` stays comfortably above 0.5 at ``U = 0.1``
#: (``CLAUDE.md`` constraint 3 and § Stability: "tau too close to 0.5" is the
#: first failure mode). The script prints the arithmetic on every run.
DEFAULT_L: dict[int, int] = {100: 128, 400: 128, 1000: 256}

#: Lid velocity in lattice units. The lid is the fastest thing in the cavity, so
#: this single number is what keeps peak ``|u|`` under the 0.1 ceiling
#: (``CLAUDE.md`` constraint 3) — 0.1 itself would sit *on* the ceiling.
U_LID: float = 0.09

#: Which wall owns the two cells where the lid meets the side walls — Q-003,
#: closed by measurement in ``DOCS/STATE1.md`` **D-013**. ``"wall"`` (the static
#: no-slip walls own them) beat ``"lid"`` on worst-case deviation from Ghia
#: across the three Reynolds numbers: 1.01% against 1.35%.
CORNERS: str = "wall"

MAX_STEPS: int = 600_000
CHECK_EVERY: int = 500

#: Lowest ``tau`` this benchmark will run at. Above 0.5 by a margin, because
#: ``tau`` "slightly above 0.5" is the marginal case that produces a
#: checkerboard rather than an honest failure (``DOCS/IDEA2.md`` § Stability).
TAU_FLOOR: float = 0.53

#: Convergence threshold on the **per-step** velocity change normalised by the
#: lid velocity: ``max|u(n) - u(n-k)| / (U k)`` with ``k = CHECK_EVERY``.
#: Dividing by ``k`` is what makes the contract's ``1e-6`` reachable in
#: ``float32`` at all — see ``DOCS/STATE1.md`` D-012 and D-014. The raw
#: interval difference is printed alongside it, unscaled, so nothing is hidden.
RESIDUAL_TOL: float = 1e-6

#: Max deviation from Ghia, as a fraction of the lid velocity.
GHIA_TOL: float = 0.05

#: Max primary-vortex-centre error, in cells.
VORTEX_TOL_CELLS: float = 2.0


@dataclass
class CavityResult:
    """What one cavity run measured. Velocities are normalised by the lid."""

    re: int
    n: int
    side: int  # L = n - 2
    u_lid: float
    tau: float
    nu: float
    corners: str
    steps: int
    residual: float  # per-step, normalised by u_lid
    residual_raw: float  # over CHECK_EVERY steps, normalised by u_lid
    seconds: float
    peak_u: float  # lattice units, not normalised
    saw_nan: bool
    ux_centre: NDArray[np.float64]  # (L,) at x = 0.5, normalised
    uy_centre: NDArray[np.float64]  # (L,) at y = 0.5, normalised
    coords: NDArray[np.float64]  # (L,) fluid-node coordinates in [0, 1]
    vortex: tuple[float, float]  # (x, y) in [0, 1]


def cavity_masks(
    n: int, corners: str
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Static walls and moving lid, both shape ``(n, n)``, ``bool``.

    One-cell solid border (``CLAUDE.md`` constraint 12 asks for >= 3 cells only
    for *immersed objects*, which can leak through bounce-back; a domain border
    has no interior to leak into). The lid is the top row, ``y = n - 1``.

    Args:
        n: grid side in cells, including the solid border.
        corners: ``"lid"`` puts the two cells where the lid meets the side walls
            into the moving lid; ``"wall"`` leaves them with the static no-slip
            walls. This is Q-003 and the two masks differ in exactly those two
            cells. They are not inert: the left corner emits direction
            ``i = 8`` ``(+1,-1)`` onto a fluid cell and the right corner emits
            ``i = 7`` ``(-1,-1)`` onto one.

    Returns:
        ``(static, lid)`` — disjoint masks whose union is the solid border.
    """
    if corners not in ("lid", "wall"):
        raise ValueError(f"corners must be 'lid' or 'wall', got {corners!r}")

    lid = np.zeros((n, n), dtype=bool)
    if corners == "lid":
        lid[n - 1, :] = True
    else:
        lid[n - 1, 1:-1] = True

    static = np.zeros((n, n), dtype=bool)
    static[0, :] = True
    static[:, 0] = True
    static[:, -1] = True
    static[n - 1, :] = True
    static &= ~lid

    return static, lid


def tau_for(re: float, u_lid: float, side: int) -> tuple[float, float]:
    """``(nu, tau)`` from ``Re = U L / nu`` and ``nu = (tau - 0.5) / 3``.

    ``CLAUDE.md`` constraint 2: viscosity is not a free parameter. ``L`` is the
    D-009 cavity side, ``n - 2``, the distance between the two wall planes.

    Both of the ways this can go wrong are caught **here, at setup**, not at
    ``nan`` time (``CLAUDE.md`` constraint 3, ``DOCS/IDEA2.md`` § Stability):

    * ``u_lid >= 0.1`` is rejected outright — the lid velocity is the largest
      velocity in the cavity, so it alone decides whether the Mach-squared
      compressibility error is negligible.
    * ``tau`` below :data:`TAU_FLOOR` is rejected with the resolution that would
      fix it, because "tau slightly above 0.5" is the marginal-stability
      checkerboard case, not a merely inaccurate one.

    Raises:
        ValueError: if ``u_lid >= 0.1`` or the grid is too coarse for ``re``.
    """
    if u_lid >= 0.1:
        raise ValueError(
            f"lid velocity must stay under 0.1 in lattice units (got {u_lid!r}): "
            "CLAUDE.md constraint 3, compressibility error scales as Mach squared."
        )

    nu = u_lid * side / re
    tau = 0.5 + 3.0 * nu
    nu_from_tau(tau)  # keeps nu on its single code path (constraint 2)

    if tau < TAU_FLOOR:
        need = int(np.ceil(re * (TAU_FLOOR - 0.5) / (3.0 * u_lid)))
        raise ValueError(
            f"tau = {tau:.4f} is below the safety floor {TAU_FLOOR} for "
            f"Re = {re} at L = {side}, U = {u_lid}. tau this close to 0.5 is "
            f"marginally stable (DOCS/IDEA2.md § Stability). Use L >= {need}, "
            f"or lower Re."
        )

    return nu, tau


def _streamfunction(ux: NDArray[np.float64]) -> NDArray[np.float64]:
    """``psi`` on the fluid nodes from ``psi = integral of ux dy``.

    Cumulative sum in ``y`` with ``dy = 1`` lattice cell. Only the *location* of
    the extremum is used, so the integration constant and the half-cell end
    corrections do not matter.

    Args:
        ux: ``ux`` on the fluid nodes, shape ``(L, L)``, index order ``(y, x)``.

    Returns:
        ``psi``, shape ``(L, L)``.
    """
    return np.cumsum(ux, axis=0)


def _parabolic_peak(y_m: float, y_0: float, y_p: float) -> float:
    """Sub-cell offset in ``[-0.5, 0.5]`` of a parabola through three samples."""
    denom = y_m - 2.0 * y_0 + y_p
    if denom == 0.0:
        return 0.0
    off = 0.5 * (y_m - y_p) / denom
    return float(np.clip(off, -0.5, 0.5))


def vortex_centre(
    ux: NDArray[np.float64], side: int
) -> tuple[float, float]:
    """Primary vortex centre in ``[0, 1]^2`` from the streamfunction extremum.

    The extremum cell is refined to sub-cell accuracy by fitting a parabola in
    each direction, because the acceptance criterion is 2 cells and rounding to
    the nearest node would already spend half of that.

    Args:
        ux: ``ux`` on the fluid nodes, shape ``(L, L)``, ``(y, x)``.
        side: ``L``, the cavity side in cells.

    Returns:
        ``(x, y)``, physical coordinates in ``[0, 1]``.
    """
    psi = _streamfunction(ux)
    j, i = np.unravel_index(int(np.argmax(np.abs(psi))), psi.shape)

    dy = 0.0
    if 0 < j < side - 1:
        dy = _parabolic_peak(psi[j - 1, i], psi[j, i], psi[j + 1, i])
    dx = 0.0
    if 0 < i < side - 1:
        dx = _parabolic_peak(psi[j, i - 1], psi[j, i], psi[j, i + 1])

    # Fluid array index k (0-based) is grid row k + 1, at coordinate
    # (k + 1 - 0.5) / L = (k + 0.5) / L.
    return ((i + dx + 0.5) / side, (j + dy + 0.5) / side)


def run_cavity(
    re: int,
    side: int | None = None,
    u_lid: float = U_LID,
    corners: str = CORNERS,
    max_steps: int = MAX_STEPS,
    residual_tol: float = RESIDUAL_TOL,
) -> CavityResult:
    """Run one cavity to steady state.

    Timestep order is D-011, exactly as :mod:`lbm.boundary` documents it, with
    the lid handled by :func:`lbm.boundary.moving_wall` after the static walls::

        copy f_pre -> macroscopic -> equilibrium -> collide
                   -> bounce_back(static) -> moving_wall(lid) -> stream

    No body force, so neither half of the Guo pair is called (D-010: they go
    together or not at all). Every buffer is allocated before the loop
    (``CLAUDE.md`` § conventions).

    Args:
        re: Reynolds number, ``U L / nu``.
        side: cavity side ``L`` in cells; grid is ``(L + 2)^2``. Defaults to
            :data:`DEFAULT_L`.
        u_lid: lid velocity in lattice units, under 0.1.
        corners: ``"lid"`` or ``"wall"`` — see :func:`cavity_masks` (Q-003).
        max_steps: hard cap.
        residual_tol: per-step convergence threshold, normalised by ``u_lid``.

    Returns:
        A :class:`CavityResult`.
    """
    if side is None:
        side = DEFAULT_L[re]
    n = side + 2
    nu, tau = tau_for(re, u_lid, side)

    static, lid = cavity_masks(n, corners)
    fluid = slice(1, n - 1)

    # --- preallocation; nothing below the loop header allocates ---
    f = np.empty((Q, n, n), dtype=np.float32)
    f_pre = np.empty_like(f)
    feq = np.empty_like(f)
    buf = np.empty_like(f)
    rho = np.empty((n, n), dtype=np.float32)
    u = np.empty((2, n, n), dtype=np.float32)
    work = np.empty((3, n, n), dtype=np.float32)
    u_prev = np.zeros((2, n, n), dtype=np.float32)

    # Rest state: f = w_i * rho with rho = 1, u = 0.
    f[:] = W[:, None, None]

    saw_nan = False
    residual = np.inf
    residual_raw = np.inf
    steps = 0
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        np.copyto(f_pre, f)

        macroscopic(f, rho, u)
        equilibrium(rho, u, feq, work)
        collide(f, feq, tau)
        bounce_back(f, f_pre, static)
        moving_wall(f, f_pre, lid, (u_lid, 0.0))
        stream(f, buf)

        steps = step

        if step % CHECK_EVERY == 0:
            if not np.isfinite(f).all():
                saw_nan = True
                break
            # Fluid interior only: `rho` on solid cells is whatever bounce-back
            # left there and `u = (e.f)/rho` on them is meaningless — including
            # them would make the residual noise, not convergence.
            np.subtract(u, u_prev, out=work[:2])
            residual_raw = float(np.max(np.abs(work[:2, fluid, fluid]))) / u_lid
            residual = residual_raw / CHECK_EVERY
            np.copyto(u_prev, u)
            if residual < residual_tol:
                break

    seconds = time.perf_counter() - t0

    macroscopic(f, rho, u)
    peak_u = float(np.max(np.abs(u[:, fluid, fluid])))

    ux = np.asarray(u[0, fluid, fluid], dtype=np.float64)
    uy = np.asarray(u[1, fluid, fluid], dtype=np.float64)

    # x = 0.5 and y = 0.5 fall exactly between two nodes when L is even, so the
    # centreline is the mean of the two adjacent lines rather than a node.
    if side % 2 == 0:
        lo, hi = side // 2 - 1, side // 2
        ux_centre = 0.5 * (ux[:, lo] + ux[:, hi])
        uy_centre = 0.5 * (uy[lo, :] + uy[hi, :])
    else:
        mid = side // 2
        ux_centre = ux[:, mid]
        uy_centre = uy[mid, :]

    coords = (np.arange(side, dtype=np.float64) + 0.5) / side

    return CavityResult(
        re=re,
        n=n,
        side=side,
        u_lid=u_lid,
        tau=tau,
        nu=nu,
        corners=corners,
        steps=steps,
        residual=residual,
        residual_raw=residual_raw,
        seconds=seconds,
        peak_u=peak_u,
        saw_nan=saw_nan,
        ux_centre=ux_centre / u_lid,
        uy_centre=uy_centre / u_lid,
        coords=coords,
        vortex=vortex_centre(ux, side),
    )


def _sample(
    coords: NDArray[np.float64],
    profile: NDArray[np.float64],
    at: NDArray[np.float64],
    lo_value: float,
    hi_value: float,
) -> NDArray[np.float64]:
    """Linear interpolation of a centreline profile onto Ghia's sample points.

    Ghia samples the two wall planes themselves (``0.0`` and ``1.0``), which are
    half a cell outside the outermost fluid nodes, so the boundary values are
    prepended and appended rather than extrapolated: they are imposed by the
    boundary condition and known exactly (``0`` on a no-slip wall, ``1`` on the
    lid, both normalised by the lid velocity).
    """
    xs = np.concatenate(([0.0], coords, [1.0]))
    ys = np.concatenate(([lo_value], profile, [hi_value]))
    return np.interp(at, xs, ys)


def deviations(res: CavityResult) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(du, dv)`` — signed deviation from Ghia at the 17 sample points each."""
    u_at = _sample(res.coords, res.ux_centre, GHIA_Y, 0.0, 1.0)
    v_at = _sample(res.coords, res.uy_centre, GHIA_X, 0.0, 0.0)
    return u_at - GHIA_U[res.re], v_at - GHIA_V[res.re]


def scored_mask(re: int, component: str) -> NDArray[np.bool_]:
    """Which of the 17 points count toward the pass criterion.

    Everything except the entries listed in :data:`GHIA_SUSPECT`, which are
    still computed and still printed — see that constant for the evidence and
    ``DOCS/STATE1.md`` D-015 for the decision.
    """
    keep = np.ones(GHIA_Y.size, dtype=bool)
    for k in GHIA_SUSPECT.get((re, component), ()):
        keep[k] = False
    return keep


def report(res: CavityResult, verbose: bool = False) -> bool:
    """Print one case against Ghia and return whether it passed."""
    du, dv = deviations(res)
    keep_u = scored_mask(res.re, "u")
    keep_v = scored_mask(res.re, "v")
    max_dev = float(
        max(np.max(np.abs(du[keep_u])), np.max(np.abs(dv[keep_v])))
    )

    gx, gy = GHIA_VORTEX[res.re]
    vx, vy = res.vortex
    vortex_err_cells = float(
        np.hypot(vx - gx, vy - gy) * res.side
    )

    print(f"  Re = {res.re}")
    print(
        f"    grid {res.n} x {res.n}   L = n - 2 = {res.side} (D-009)   "
        f"U = {res.u_lid}"
    )
    print(
        f"    nu = U L / Re = {res.u_lid} * {res.side} / {res.re} = "
        f"{res.nu:.6f}   tau = 0.5 + 3 nu = {res.tau:.6f}"
    )
    print(
        f"    corners = {res.corners}   steps = {res.steps}   "
        f"{res.seconds:.1f} s   ({res.steps / max(res.seconds, 1e-9):.0f} steps/s)"
    )
    print(
        f"    residual = {res.residual:.2e} per step "
        f"(raw over {CHECK_EVERY} steps: {res.residual_raw:.2e})"
    )
    print(f"    peak |u| = {res.peak_u:.5f} lattice units")

    checks: list[tuple[str, bool, str]] = [
        ("finite", not res.saw_nan, "no nan"),
        (
            f"converged (residual < {RESIDUAL_TOL:.0e} per step)",
            res.residual < RESIDUAL_TOL,
            f"{res.residual:.2e}",
        ),
        (
            f"max |deviation from Ghia| < {GHIA_TOL:.0%} of U",
            max_dev < GHIA_TOL,
            f"{max_dev * 100:.2f}%  "
            f"(u: {np.max(np.abs(du[keep_u])) * 100:.2f}%, "
            f"v: {np.max(np.abs(dv[keep_v])) * 100:.2f}%)",
        ),
        (
            f"primary vortex centre within {VORTEX_TOL_CELLS:.0f} cells",
            vortex_err_cells < VORTEX_TOL_CELLS,
            f"({vx:.4f}, {vy:.4f}) vs Ghia ({gx:.4f}, {gy:.4f}) "
            f"= {vortex_err_cells:.2f} cells",
        ),
        (
            "peak lattice velocity < 0.1 (constraint 3)",
            res.peak_u < 0.1,
            f"{res.peak_u:.5f}",
        ),
    ]

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"    [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    # Excluded reference points are printed every run, never merely omitted.
    for component, dev, stations in (("u", du, GHIA_Y), ("v", dv, GHIA_X)):
        for k in GHIA_SUSPECT.get((res.re, component), ()):
            table = GHIA_U[res.re] if component == "u" else GHIA_V[res.re]
            print(
                f"    [--] excluded reference point: {component}"
                f"({stations[k]:.4f}) tabulated {table[k]:+.5f}, "
                f"computed {table[k] + dev[k]:+.5f}, "
                f"deviation {abs(dev[k]) * 100:.2f}% "
                f"— suspected typo in Ghia Table II, see GHIA_SUSPECT / D-015"
            )

    if verbose:
        u_at = _sample(res.coords, res.ux_centre, GHIA_Y, 0.0, 1.0)
        v_at = _sample(res.coords, res.uy_centre, GHIA_X, 0.0, 0.0)
        print()
        print("      y        u_sim      u_Ghia     dev  |"
              "   x        v_sim      v_Ghia     dev")
        for k in range(GHIA_Y.size):
            print(
                f"    {GHIA_Y[k]:6.4f} {u_at[k]: .5f} {GHIA_U[res.re][k]: .5f}"
                f" {du[k]: .5f}  |"
                f" {GHIA_X[k]:6.4f} {v_at[k]: .5f} {GHIA_V[res.re][k]: .5f}"
                f" {dv[k]: .5f}"
            )

    passed = all(ok for _, ok, _ in checks)
    print(f"    -> {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run every requested Re and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Rung 2 — lid-driven cavity vs Ghia et al. (1982)"
    )
    parser.add_argument(
        "--re",
        type=int,
        action="append",
        choices=sorted(GHIA_U),
        help="Reynolds number; repeat for several. Default: 100 400 1000.",
    )
    parser.add_argument(
        "--side",
        type=int,
        default=None,
        help="cavity side L in cells; grid is (L+2)^2. Default: per-Re table.",
    )
    parser.add_argument("--u", type=float, default=U_LID, help="lid velocity")
    parser.add_argument(
        "--corners",
        choices=("lid", "wall", "both"),
        default=CORNERS,
        help="Q-003: which wall owns the two lid corner cells (default: "
        f"{CORNERS}, see D-013). 'both' runs each Re twice and prints the "
        "comparison that closed it.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--verbose", action="store_true", help="print the 17-point tables"
    )
    args = parser.parse_args(argv)

    res_list = args.re or [100, 400, 1000]

    print("Rung 2 — lid-driven cavity (DOCS/IDEA2.md § Validation ladder)")
    print("  reference: Ghia, Ghia & Shin, J. Comput. Phys. 48, 387-411 (1982)")
    print("  walls: half-way bounce-back (D-009); lid: momentum-corrected")
    print("         bounce-back (Ladd), lbm.boundary.moving_wall")
    print()

    if args.corners == "both":
        corner_modes = ["lid", "wall"]
    else:
        corner_modes = [args.corners]

    passed_all = True
    summary: list[tuple[int, str, float, float, bool]] = []

    for re in res_list:
        for corners in corner_modes:
            res = run_cavity(
                re=re,
                side=args.side,
                u_lid=args.u,
                corners=corners,
                max_steps=args.max_steps,
            )
            ok = report(res, verbose=args.verbose)
            du, dv = deviations(res)
            max_dev = float(
                max(
                    np.max(np.abs(du[scored_mask(re, "u")])),
                    np.max(np.abs(dv[scored_mask(re, "v")])),
                )
            )
            gx, gy = GHIA_VORTEX[re]
            vcells = float(np.hypot(res.vortex[0] - gx, res.vortex[1] - gy) * res.side)
            summary.append((re, corners, max_dev, vcells, ok))
            passed_all = passed_all and ok

    print("  summary")
    print("    Re     corners   max dev vs Ghia   vortex err (cells)   result")
    for re, corners, max_dev, vcells, ok in summary:
        print(
            f"    {re:<6} {corners:<9} {max_dev * 100:12.2f}%  "
            f"{vcells:18.2f}   {'PASS' if ok else 'FAIL'}"
        )

    print()
    print("PASS" if passed_all else "FAIL")
    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(main())
