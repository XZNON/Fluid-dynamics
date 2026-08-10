"""Solid walls and body forcing.

Implements ``DOCS/IDEA2.md`` § "The method, in the order the code runs it",
step 5 (bounce-back), plus the Guo body-force scheme that Rung 1 needs to drive
a channel. Inlet and outlet boundaries are T005 and land here later.

The nine lattice constants are imported from :mod:`lbm.core` and never
redefined (``CLAUDE.md`` constraint 4). Everything here is lattice units.

Ordering within a timestep
--------------------------

Bounce-back is applied to the **post-collision** state using a copy of the
**pre-collision** state, and it happens *before* streaming::

    np.copyto(f_pre, f)                    # pre-collision copy
    rho, u = macroscopic(f, rho, u)
    force_velocity_shift(rho, u, g)        # Guo: u += F / (2 rho)
    equilibrium(rho, u, feq, work)
    collide(f, feq, tau)
    apply_body_force(f, rho, u, tau, g, work)
    bounce_back(f, f_pre, solid)           # overwrite solid cells
    stream(f, buf)

That order is what makes the reflection work: ``f_pre[OPP[i]]`` at a solid cell
is the population that streamed *into* the solid last step travelling in
direction ``OPP[i]``. Writing it into slot ``i`` sends it straight back out
along ``E[i]`` on the next stream.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lbm.core import E_F32, OPP, Q, W

__all__ = [
    "bounce_back",
    "moving_wall",
    "force_velocity_shift",
    "apply_body_force",
]


def bounce_back(
    f: NDArray[np.float32],
    f_pre: NDArray[np.float32],
    solid: NDArray[np.bool_],
) -> None:
    """Half-way bounce-back on solid cells, in place.

    ``DOCS/IDEA2.md`` § The method, step 5::

        on solid cells, f[i] = f_pre_stream[opp[i]]

    ``f_pre`` must be the copy taken **before** collision of this timestep (see
    the module docstring for the required call order). The populations it holds
    at a solid cell are the ones that arrived there from the neighbouring fluid
    during the previous stream; reversing them and letting the next stream carry
    them out is the reflection.

    Wall-offset convention (``DOCS/STATE1.md`` § Decisions, **D-009**, closing
    Q-001)
    ----------------------------------------------------------------------
    The no-slip plane sits **halfway between the last fluid node and the first
    solid node**, not on a node. For a channel whose solid rows are ``y = 0``
    and ``y = ny - 1``, the fluid nodes are ``y = 1 .. ny - 2``, the walls are
    the planes ``y = 0.5`` and ``y = ny - 1.5``, and therefore::

        H  = ny - 2                     # channel height between the walls
        y_ = y - 0.5                    # wall-relative coordinate of fluid row y

    so the analytic Poiseuille profile is evaluated at ``y_``, giving ``u = 0``
    at ``y_ = 0`` and ``y_ = H``, and a maximum at the midpoint of the fluid
    rows. Measured by ``validate/poiseuille.py``, which prints all three rival
    conventions every run: halfway 0.365%, wall-on-last-fluid-node
    (``H = ny - 3``) 14.763%, wall-on-solid-node (``H = ny - 1``) 12.746%. Rung
    2's cavity ``L`` uses the same convention.

    The residual 0.365% is a **uniform** velocity deficit of about ``1.1e-4``,
    not a shape error, and it is a known property of BGK bounce-back rather than
    a bug: the effective wall sits at ``0.5 - delta`` with ``delta`` depending on
    ``tau``, and coincides with the exact halfway plane only when
    ``(tau - 0.5)^2 = 3/16``. Removing it needs TRT or MRT, which Phase 0
    deliberately excludes (``CLAUDE.md`` constraint 1).

    Args:
        f: post-collision distribution, shape ``(9, ny, nx)``, ``float32``,
            modified in place on solid cells only.
        f_pre: pre-collision copy of ``f``, same shape and dtype.
        solid: solid mask, shape ``(ny, nx)``, ``bool`` — ``True`` is wall
            (``CLAUDE.md`` constraint 12).
    """
    for i in range(Q):
        np.copyto(f[i], f_pre[OPP[i]], where=solid)


def moving_wall(
    f: NDArray[np.float32],
    f_pre: NDArray[np.float32],
    wall: NDArray[np.bool_],
    u_wall: tuple[float, float],
    rho_w: float = 1.0,
) -> None:
    """Momentum-corrected (Ladd) bounce-back on a moving wall, in place.

    ``DOCS/IDEA2.md`` § "Validation ladder", Rung 2 — the lid of the lid-driven
    cavity. This is **momentum-corrected bounce-back**, not Zou–He::

        f[i] = f_pre[OPP[i]] + 2 w_i rho_w (e_i . u_wall) / cs2
             = f_pre[OPP[i]] + 6 w_i rho_w (e_i . u_wall)

    With ``u_wall = 0`` it degenerates exactly to :func:`bounce_back`, which is
    the consistency check the unit tests make. The correction term is the
    momentum the wall hands to the reflected population: for a lid at the top row
    moving in ``+x``, the directions that re-enter the fluid are those with
    ``ey = -1``, and the term adds to ``i = 8`` ``(+1,-1)`` while subtracting from
    ``i = 7`` ``(-1,-1)``, so the reflected populations carry net ``+x`` momentum
    into the fluid. That is the drag the lid exerts.

    Call order and the ``f_pre`` it consumes are the same as
    :func:`bounce_back` — see the module docstring (``DOCS/STATE1.md`` D-011).
    Apply it **after** :func:`bounce_back` if the masks overlap; this function
    writes the complete reflection for its cells, so the last writer wins.

    Wall offset is unchanged from D-009: the no-slip / moving-wall plane sits
    **halfway** between the last fluid node and the solid node. For a cavity of
    ``n x n`` cells with a one-cell solid border, the fluid nodes are
    ``1 .. n - 2`` and the characteristic length is ``L = n - 2``.

    Corner cells (``DOCS/STATE1.md`` § Decisions, **D-013**, closes Q-003)
    ---------------------------------------------------------------------
    The two cells where the lid meets the side walls are ambiguous: they are
    solid, they touch both walls, and the diagonal population they emit
    (``i = 8`` from the left corner, ``i = 7`` from the right) lands on a fluid
    cell, so the choice is not cosmetic.

    **They belong to the static side walls, not to the lid.** Measured, not
    argued: ``validate/cavity.py --corners both`` runs every Reynolds number
    both ways and prints the comparison. Max deviation from Ghia, as a fraction
    of the lid velocity::

        Re      corners=lid    corners=wall
        100        0.51%          0.75%
        400        1.21%          0.42%
        1000       1.35%          1.01%

    Worst case across the three: **1.01% for wall against 1.35% for lid**, so
    ``wall`` wins on the number the acceptance criterion actually measures. It
    is also the physically defensible reading — the corner is where the moving
    and stationary walls meet, and the velocity there is genuinely singular;
    giving the cell the lid velocity injects the full lid momentum right at the
    singularity, which is precisely where the truncation error is worst.

    This function does not decide for the caller: it applies the wall velocity
    to whatever mask it is given. The decision lives in ``validate/cavity.py``'s
    ``CORNERS`` and in ``DOCS/STATE1.md`` D-013.

    Args:
        f: post-collision distribution, shape ``(9, ny, nx)``, ``float32``,
            modified in place on ``wall`` cells only.
        f_pre: pre-collision copy of ``f``, same shape and dtype.
        wall: mask of the moving solid cells, shape ``(ny, nx)``, ``bool``.
        u_wall: wall velocity ``(ux, uy)``, lattice units. Must stay well under
            0.1 (``CLAUDE.md`` constraint 3); the correction is linear in it and
            the equilibrium it feeds is a Mach-squared truncation.
        rho_w: wall density used in the correction. 1.0 is the standard
            incompressible choice and the density in a cavity stays within a
            fraction of a percent of it; the error is ``O(Ma^2)`` on a term that
            is already ``O(Ma)``.
    """
    uwx, uwy = float(u_wall[0]), float(u_wall[1])

    for i in range(Q):
        np.copyto(f[i], f_pre[OPP[i]], where=wall)

        eu = float(E_F32[i, 0]) * uwx + float(E_F32[i, 1]) * uwy
        if eu != 0.0:
            # 2 / cs2 == 6
            c = np.float32(6.0 * float(W[i]) * rho_w * eu)
            np.add(f[i], c, out=f[i], where=wall)


def force_velocity_shift(
    rho: NDArray[np.float32],
    u: NDArray[np.float32],
    g: tuple[float, float],
    work: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Guo's half-force correction to the velocity, in place.

    With a body force present, the velocity that enters the equilibrium — and
    the velocity that is the physically correct one to report — is

    .. math:: u = \\frac{1}{\\rho}\\left(\\sum_i e_i f_i + \\frac{F}{2}\\right)

    (Guo, Zheng & Shi 2002). :func:`lbm.core.macroscopic` returns the bare first
    moment, so this adds the missing ``F / (2 rho)``. Skipping it is a
    first-order error in the force and shows up directly in the Rung 1 L2
    number, so it is not optional.

    Args:
        rho: density, shape ``(ny, nx)``, ``float32``.
        u: velocity, shape ``(2, ny, nx)``, ``float32``, ``(ux, uy)``
            (``DOCS/STATE1.md`` D-005). Modified in place.
        g: uniform body force per unit volume, ``(gx, gy)``, lattice units.
        work: optional preallocated scratch, shape ``(>=2, ny, nx)``,
            ``float32``. Supply it to make the call allocation-free (D-006).

    Returns:
        ``u``, corrected — the same object that was passed in.
    """
    if work is None:
        work = np.empty((2,) + rho.shape, dtype=np.float32)

    inv_rho, tmp = work[0], work[1]
    np.reciprocal(rho, out=inv_rho)

    for c in (0, 1):
        half_g = 0.5 * g[c]
        if half_g != 0.0:
            np.multiply(inv_rho, np.float32(half_g), out=tmp)
            u[c] += tmp

    return u


def apply_body_force(
    f: NDArray[np.float32],
    rho: NDArray[np.float32],
    u: NDArray[np.float32],
    tau: float,
    g: tuple[float, float],
    work: NDArray[np.float32] | None = None,
) -> None:
    """Guo forcing source term, added to the post-collision distribution.

    Guo, Zheng & Shi (2002), the second-order-consistent body force for BGK::

        S_i = (1 - 1/(2 tau)) w_i [ (e_i - u)/cs2 + (e_i.u) e_i / cs2^2 ] . F

    which with ``cs2 = 1/3`` is::

        S_i = (1 - 1/(2 tau)) w_i [ 3 (e_i.F - u.F) + 9 (e_i.u)(e_i.F) ]

    Call it **after** :func:`lbm.core.collide` and with the ``u`` that
    :func:`force_velocity_shift` has already corrected — the two halves of the
    scheme are a pair and using one without the other reintroduces the
    first-order error the scheme exists to remove.

    ``sum_i S_i == 0`` exactly (the two ``u.F`` contributions cancel under
    ``sum_i w_i e_i = 0`` and ``sum_i w_i e_ia e_ib = cs2 delta_ab``), so mass is
    conserved to round-off; Rung 1 asserts that drift directly.

    The force is applied on every cell, solid included; :func:`bounce_back` runs
    afterwards and overwrites solid cells wholesale, so the wasted work is
    harmless. Masking it is a T010 optimisation (``CLAUDE.md`` constraint 6).

    Args:
        f: post-collision distribution, shape ``(9, ny, nx)``, ``float32``,
            modified in place.
        rho: density, shape ``(ny, nx)``, ``float32``. Unused by the formula
            above — it is in the signature because Guo's scheme is stated per
            unit volume and callers that switch to a per-unit-mass force need
            it; keeping the signature stable now avoids reshaping the API in
            T005.
        u: force-corrected velocity, shape ``(2, ny, nx)``, ``float32``.
        tau: BGK relaxation time, greater than 0.5.
        g: uniform body force per unit volume, ``(gx, gy)``, lattice units.
        work: optional preallocated scratch, shape ``(3, ny, nx)``, ``float32``
            — the same buffer :func:`lbm.core.equilibrium` takes, and safe to
            share with it since the two calls never overlap.
    """
    del rho  # see the docstring; kept for API stability into T005

    ny, nx = u.shape[1], u.shape[2]
    if work is None:
        work = np.empty((3, ny, nx), dtype=np.float32)

    ux, uy = u[0], u[1]
    ug, eu, tmp = work[0], work[1], work[2]

    gx, gy = float(g[0]), float(g[1])
    pref = 1.0 - 0.5 / tau

    # ug = 3 (u . F): the same on every direction, so hoisted out of the loop
    # and premultiplied by 3 since that is its only use.
    np.multiply(ux, np.float32(gx), out=ug)
    np.multiply(uy, np.float32(gy), out=tmp)
    ug += tmp
    ug *= np.float32(3.0)

    for i in range(Q):
        ex, ey = float(E_F32[i, 0]), float(E_F32[i, 1])
        eg = ex * gx + ey * gy  # e_i . F, a scalar

        # eu = e_i . u
        np.multiply(ux, np.float32(ex), out=eu)
        np.multiply(uy, np.float32(ey), out=tmp)
        eu += tmp

        # tmp = 3 (e_i.F - u.F) + 9 (e_i.u)(e_i.F)
        np.multiply(eu, np.float32(9.0 * eg), out=tmp)
        tmp += np.float32(3.0 * eg)
        tmp -= ug

        tmp *= np.float32(pref * float(W[i]))
        f[i] += tmp
