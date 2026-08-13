"""Measurement: vorticity, momentum-exchange forces, Strouhal number, residual.

Implements ``DOCS/IDEA2.md`` § "What to actually draw" (vorticity) and the
measurements Rung 3 is scored on — § "Validation ladder", Rung 3: ``St ~ 0.164``
and ``Cd ~ 1.34``.

**Why this module exists.** Without it, "the wake looks right" is the whole
verification story, and ``old-Docs/PLAN1.md`` § Risks names a plausible-looking
wrong simulation as the main failure mode of the project. These four functions
turn the picture into numbers.

The nine lattice constants come from :mod:`lbm.core` and are never redefined
(``CLAUDE.md`` constraint 4). Everything here is lattice units; physical units
are ``lbm/units.py``'s job (T009) and never reach this module.

All four take optional preallocated outputs (``old-Docs/STATE1.md`` D-006) — T007
calls :func:`vorticity` and :func:`forces` on every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lbm.core import E, E_F32, OPP, Q

__all__ = [
    "BoundaryLinks",
    "boundary_links",
    "forces",
    "residual",
    "strouhal",
    "vorticity",
]


# --- vorticity --------------------------------------------------------------


def vorticity(
    u: NDArray[np.float32],
    *,
    solid: NDArray[np.bool_] | None = None,
    out: NDArray[np.float32] | None = None,
    work: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Vorticity of a 2D velocity field.

    ``DOCS/IDEA2.md`` § What to actually draw::

        omega = d(uy)/dx - d(ux)/dy

    Central differences in the interior, one-sided (first-order) on the four
    edges. Solid cells are set to ``nan``: the velocity there is
    ``(e . f) / rho`` with a density bounce-back left behind, which is
    meaningless (``old-Docs/STATE1.md`` D-014), and ``nan`` makes a renderer skip
    the cell instead of painting a bright artefact on the obstacle.

    This is computed **here, not in** ``render.py`` (``CLAUDE.md``
    constraint 9). ``render.py`` colours arrays; it does no physics. The
    colormap wants a diverging map with **fixed** symmetric limits, which is
    also its problem, not this function's.

    Written out with slices rather than :func:`numpy.gradient`, which allocates
    two full fields per call and would break "never allocate inside the step
    loop" when T007 draws every frame.

    Axis convention: axis 0 of the field is ``y`` and axis 1 is ``x``
    (``CLAUDE.md`` constraint 4), and ``u[0]`` is ``ux``, ``u[1]`` is ``uy``
    (D-005). Row index increases with ``y``, so a flow turning from ``+x``
    toward ``+y`` as ``x`` grows has positive vorticity.

    Args:
        u: velocity, shape ``(2, ny, nx)``, ``float32``.
        solid: optional solid mask, shape ``(ny, nx)``, ``bool``. Those cells
            come back ``nan``.
        out: optional preallocated output, shape ``(ny, nx)``, ``float32``.
        work: optional scratch, shape ``(ny, nx)``, ``float32``.

    Returns:
        ``out``, shape ``(ny, nx)``, ``float32``.
    """
    ny, nx = u.shape[1], u.shape[2]

    if out is None:
        out = np.empty((ny, nx), dtype=np.float32)
    if work is None:
        work = np.empty((ny, nx), dtype=np.float32)

    ux, uy = u[0], u[1]

    _d_dx(uy, out)
    _d_dy(ux, work)
    out -= work

    if solid is not None:
        np.copyto(out, np.float32(np.nan), where=solid)

    return out


def _d_dx(field: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """``d(field)/dx`` (axis 1): central in the interior, one-sided at the edges."""
    nx = field.shape[1]
    if nx < 2:
        out[:] = np.float32(0.0)
        return

    np.subtract(field[:, 2:], field[:, :-2], out=out[:, 1:-1])
    out[:, 1:-1] *= np.float32(0.5)
    np.subtract(field[:, 1], field[:, 0], out=out[:, 0])
    np.subtract(field[:, -1], field[:, -2], out=out[:, -1])


def _d_dy(field: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    """``d(field)/dy`` (axis 0): central in the interior, one-sided at the edges."""
    ny = field.shape[0]
    if ny < 2:
        out[:] = np.float32(0.0)
        return

    np.subtract(field[2:], field[:-2], out=out[1:-1])
    out[1:-1] *= np.float32(0.5)
    np.subtract(field[1], field[0], out=out[0])
    np.subtract(field[-1], field[-2], out=out[-1])


# --- momentum-exchange forces -----------------------------------------------


@dataclass(frozen=True)
class BoundaryLinks:
    """Precomputed bounce-back links between fluid cells and solid cells.

    Built **once** from the mask by :func:`boundary_links` and reused every step
    (``old-Docs/TASKS1.md`` § T005 Notes: "write it so the link list is precomputed
    once from the mask, not rebuilt per step"). This is correctness and clarity
    rather than an optimisation, so it does not run into ``CLAUDE.md``
    constraint 6 — collide and stream are untouched.

    Attributes:
        dirs: directions ``i`` (a subset of ``1..8``) that have at least one
            link. Direction 0 never does; it does not move.
        idx: for each ``i`` in ``dirs``, the flat indices into a ``(ny, nx)``
            field of the **fluid** cells whose neighbour at ``x + e_i`` is
            solid.
        shape: the ``(ny, nx)`` the indices refer to.
        count: total number of links.
    """

    dirs: tuple[int, ...]
    idx: tuple[NDArray[np.int64], ...]
    shape: tuple[int, int]
    count: int


def boundary_links(solid: NDArray[np.bool_]) -> BoundaryLinks:
    """Enumerate the fluid-to-solid links of a mask.

    A link is a pair (fluid cell ``x_f``, direction ``i``) with ``x_f + e_i``
    solid — exactly the pairs along which :func:`lbm.boundary.bounce_back`
    reverses a population, and therefore exactly the pairs that exchange
    momentum with the body (``CLAUDE.md`` constraint 1: bounce-back only, so
    momentum exchange is computed over bounce-back links and nothing else).

    The neighbour lookup wraps, matching :func:`lbm.core.stream`, which is
    periodic on both axes.

    Args:
        solid: solid mask, shape ``(ny, nx)``, ``bool`` — ``True`` is wall.

    Returns:
        A :class:`BoundaryLinks`.
    """
    solid = np.asarray(solid, dtype=bool)
    ny, nx = solid.shape
    fluid = ~solid

    dirs: list[int] = []
    idx: list[NDArray[np.int64]] = []
    count = 0

    for i in range(1, Q):
        ex, ey = int(E[i, 0]), int(E[i, 1])
        # neighbour[y, x] == solid[y + ey, x + ex], periodic like stream()
        neighbour = np.roll(solid, (-ey, -ex), axis=(0, 1))
        link = fluid & neighbour
        if not link.any():
            continue
        where = np.flatnonzero(link.reshape(-1))
        dirs.append(i)
        idx.append(where)
        count += int(where.size)

    return BoundaryLinks(
        dirs=tuple(dirs), idx=tuple(idx), shape=(ny, nx), count=count
    )


def forces(
    f_pre: NDArray[np.float32],
    f_post: NDArray[np.float32],
    solid: NDArray[np.bool_] | BoundaryLinks,
    *,
    U: float,
    D: float,
    rho0: float = 1.0,
) -> tuple[float, float]:
    """Drag and lift coefficients by momentum exchange.

    ``DOCS/IDEA2.md`` § Validation ladder, Rung 3 (``Cd ~ 1.34``). The
    momentum-exchange method (Ladd 1994; Mei, Yu, Shyy & Luo 2002) sums, over
    every bounce-back link, the momentum the fluid hands to the body::

        F = sum_links e_i ( f_pre[i](x_f) + f_post[opp(i)](x_f) )

    with ``x_f`` the fluid cell and ``x_f + e_i`` solid. The first term is the
    population leaving ``x_f`` toward the wall; the second is the reflected one
    arriving back. Momentum delivered to the body along that link is
    ``(in) - (out) = e_i f_out - e_opp(i) f_ret = e_i (f_out + f_ret)`` — the
    sum, not the difference, because the return trip carries momentum in the
    opposite direction. Getting that sign backwards is the classic way to
    produce a beautifully converged, exactly wrong ``Cd``.

    Which two snapshots (``old-Docs/STATE1.md`` **D-020**)
    -------------------------------------------------
    ``f_pre`` is the **pre-stream** state — after :func:`lbm.core.collide`,
    after :func:`lbm.boundary.apply_body_force`, after
    :func:`lbm.boundary.bounce_back`, *before* :func:`lbm.core.stream`.
    ``f_post`` is the state **after** ``stream``.

    This is deliberately **not** the ``f_pre`` of
    :func:`lbm.boundary.bounce_back`, which is the *pre-collision* copy
    (D-011). The runner keeps a second ``(9, ny, nx)`` buffer for this one::

        bounce_back(f, f_pre, solid)
        np.copyto(f_bb, f)          # <- forces' f_pre
        stream(f, buf)
        Cd, Cl = forces(f_bb, f, links, U=U, D=D)

    Because bounce-back is exactly ``f[j](x_s) = f_pre_collision[opp(j)](x_s)``,
    the reflected term can equivalently be read off the pre-stream array on the
    solid side: ``f_post[opp(i)](x_f) == f_pre[opp(i)](x_f + e_i)``. A unit test
    asserts that identity, which is what pins the timing above.

    Coefficients
    ------------
    ::

        Cd = 2 Fx / (rho0 U^2 D)
        Cl = 2 Fy / (rho0 U^2 D)

    ``D`` must come from :func:`lbm.geometry.bounding_box` — the cross-stream
    extent of the object's bounding box (``old-Docs/STATE1.md`` **D-019**). It is
    the same ``D`` ``check_mask`` uses for the blockage and downstream rules;
    inventing a second definition of characteristic length is how a 10% error
    in ``Cd`` gets blamed on the solver.

    The force is per unit depth and per timestep, and ``dt = 1`` in lattice
    units, so the sum above *is* the force.

    Args:
        f_pre: pre-stream distribution, shape ``(9, ny, nx)``, ``float32``.
        f_post: post-stream distribution, same shape and dtype.
        solid: either the solid mask, shape ``(ny, nx)``, ``bool``, or a
            :class:`BoundaryLinks` built from it. Pass the links from the
            runner — building them per step is what the contract forbids.
        U: reference velocity, lattice units. Must be nonzero.
        D: characteristic length in cells (D-019). Must be nonzero.
        rho0: reference density, 1.0 in lattice units.

    Returns:
        ``(Cd, Cl)`` as Python floats. A mask with no fluid-to-solid link gives
        ``(0.0, 0.0)`` exactly.

    Raises:
        ValueError: if ``U`` or ``D`` is zero, or the shapes disagree.
    """
    if U == 0.0 or D == 0.0:
        raise ValueError(
            f"forces needs a nonzero reference velocity and length "
            f"(got U={U!r}, D={D!r}): Cd = 2 Fx / (rho0 U^2 D)."
        )
    if f_pre.shape != f_post.shape:
        raise ValueError(
            f"f_pre and f_post must have the same shape "
            f"(got {f_pre.shape} and {f_post.shape})."
        )

    links = solid if isinstance(solid, BoundaryLinks) else boundary_links(solid)
    if links.shape != f_pre.shape[1:]:
        raise ValueError(
            f"link list is for a {links.shape} grid but f is "
            f"{f_pre.shape[1:]}."
        )

    pre_flat = f_pre.reshape(Q, -1)
    post_flat = f_post.reshape(Q, -1)

    fx = 0.0
    fy = 0.0
    for i, where in zip(links.dirs, links.idx):
        # float64 accumulation: the per-link populations are ~1e-2 and there
        # can be thousands of them, so a float32 sum would lose digits the
        # Rung 3 tolerance cares about. The state stays float32; only this
        # reduction is widened.
        total = float(
            pre_flat[i, where].sum(dtype=np.float64)
            + post_flat[OPP[i], where].sum(dtype=np.float64)
        )
        fx += float(E_F32[i, 0]) * total
        fy += float(E_F32[i, 1]) * total

    scale = 2.0 / (rho0 * U * U * D)
    return fx * scale, fy * scale


# --- Strouhal number --------------------------------------------------------


def strouhal(
    cl_series: NDArray[np.float64],
    dt: float,
    D: float,
    U: float,
    *,
    transient: float = 0.3,
) -> float:
    """Strouhal number from the dominant frequency of a lift history.

    ``DOCS/IDEA2.md`` § Validation ladder, Rung 3: ``St ~ 0.164`` for a cylinder
    at Re 100. ::

        St = f D / U

    with ``f`` the shedding frequency in inverse lattice time.

    Method: drop the first ``transient`` fraction of the series (a wake takes
    hundreds of convective times to lock in, and the ramp-up dominates the
    spectrum if it is left in), subtract the mean so the DC bin cannot win,
    apply a Hann window, take the ``rfft``, and pick the largest non-DC
    magnitude. The peak bin is then refined by fitting a parabola to the log
    magnitudes of the peak and its two neighbours — a Hann-windowed peak is very
    nearly Gaussian in the bin index, so its logarithm is very nearly a
    parabola, and the fit recovers the true frequency to a small fraction of a
    bin. Bin spacing alone is ``1/(N dt)``, which for a realistic Rung 3 series
    is coarser than the 1% the acceptance criterion asks for; the interpolation
    is what makes the criterion reachable without running ten times longer.

    Args:
        cl_series: lift-coefficient history, one sample per ``dt`` of lattice
            time, shape ``(n,)``. Anything float; internally widened to
            ``float64`` — this is an offline reduction over a recorded series,
            not a per-step path.
        dt: lattice time between consecutive samples. If the runner records
            ``Cl`` once every ``k`` steps, ``dt = k``.
        D: characteristic length in cells (``old-Docs/STATE1.md`` D-019).
        U: reference velocity, lattice units.
        transient: leading fraction of the series discarded, default 0.3.

    Returns:
        ``St``. ``0.0`` if the retained series has no resolvable oscillation
        (a flat or monotone signal), which is the honest answer for "the
        cylinder is not shedding" rather than a spurious peak.

    Raises:
        ValueError: if ``transient`` is outside ``[0, 1)``, if ``dt``, ``D`` or
            ``U`` is zero, or if fewer than 8 samples survive the transient cut
            — too few to place a peak and its two neighbours.
    """
    if not 0.0 <= transient < 1.0:
        raise ValueError(f"transient must be in [0, 1) (got {transient!r}).")
    if dt == 0.0 or D == 0.0 or U == 0.0:
        raise ValueError(
            f"strouhal needs nonzero dt, D and U (got dt={dt!r}, D={D!r}, U={U!r})."
        )

    series = np.asarray(cl_series, dtype=np.float64).ravel()
    tail = series[int(round(transient * series.size)) :]
    if tail.size < 8:
        raise ValueError(
            f"only {tail.size} samples survive the {transient:.0%} transient cut "
            f"(series length {series.size}) — too few to identify a frequency. "
            "Record a longer Cl history."
        )

    tail = tail - tail.mean()
    window = np.hanning(tail.size)
    spectrum = np.abs(np.fft.rfft(tail * window))

    if spectrum.size < 2:
        return 0.0
    k0 = int(np.argmax(spectrum[1:])) + 1  # skip DC

    peak = spectrum[k0]
    if peak <= 0.0 or not np.isfinite(peak):
        return 0.0

    # Parabolic refinement on log magnitude; skipped at the spectrum's ends,
    # where there is no neighbour on one side.
    delta = 0.0
    if 0 < k0 < spectrum.size - 1:
        a, b, c = (
            np.log(max(spectrum[k0 - 1], 1e-300)),
            np.log(peak),
            np.log(max(spectrum[k0 + 1], 1e-300)),
        )
        denom = a - 2.0 * b + c
        if denom != 0.0:
            delta = 0.5 * (a - c) / denom
            if not np.isfinite(delta) or abs(delta) > 1.0:
                delta = 0.0

    freq = (k0 + delta) / (tail.size * dt)
    return float(freq * D / U)


# --- residual ---------------------------------------------------------------


def residual(
    u_now: NDArray[np.float32],
    u_prev: NDArray[np.float32],
    U: float,
    *,
    solid: NDArray[np.bool_] | None = None,
    work: NDArray[np.float32] | None = None,
) -> float:
    """Steady-state residual: the largest velocity change, scaled by ``U``.

    ::

        residual = max |u_now - u_prev| / U

    the maximum taken over both components and all **fluid** cells.

    Fluid cells only (``old-Docs/STATE1.md`` **D-014**): ``rho`` on a solid cell is
    whatever bounce-back left there, so ``u = (e . f) / rho`` on it is
    meaningless. Rung 2's residual read ``8.4e+01`` and the script "failed to
    converge" for exactly that reason before the mask was applied. Pass
    ``solid`` whenever the domain has one.

    **There is a floor** (``old-Docs/STATE1.md`` **D-012**). ``u`` is a
    near-cancelling sum of ``f ~ 0.4`` divided by ``rho``, so in ``float32`` its
    round-off is about ``eps |f| / |u| ~ 1.2e-6``; the measured per-step floor is
    ``1.7e-6``. A per-step tolerance below that is unreachable and simply burns
    the step cap. To ask for less, compare fields ``k`` steps apart and divide by
    ``k``, which measures the same rate of change with the floor pushed down by
    ``k`` — that is what ``validate/cavity.py`` does with ``k = 500`` to make a
    ``1e-6`` criterion meaningful.

    Args:
        u_now: current velocity, shape ``(2, ny, nx)``, ``float32``.
        u_prev: earlier velocity, same shape and dtype.
        U: reference velocity used to non-dimensionalise, lattice units.
        solid: optional solid mask, shape ``(ny, nx)``, ``bool``. Solid cells
            are excluded.
        work: optional scratch, shape ``(2, ny, nx)``, ``float32``.

    Returns:
        The residual, a Python float.

    Raises:
        ValueError: if ``U`` is zero or the shapes disagree.
    """
    if U == 0.0:
        raise ValueError("residual needs a nonzero reference velocity U.")
    if u_now.shape != u_prev.shape:
        raise ValueError(
            f"u_now and u_prev must have the same shape "
            f"(got {u_now.shape} and {u_prev.shape})."
        )

    if work is None:
        work = np.empty_like(u_now)

    np.subtract(u_now, u_prev, out=work)
    np.abs(work, out=work)

    if solid is not None:
        zero = np.float32(0.0)
        np.copyto(work[0], zero, where=solid)
        np.copyto(work[1], zero, where=solid)

    return float(work.max()) / U
