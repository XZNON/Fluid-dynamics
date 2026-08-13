"""Solid walls, open boundaries and body forcing.

Implements ``DOCS/IDEA2.md`` § "The method, in the order the code runs it",
step 5 (bounce-back) and step 6 (inlet velocity, outlet zero-gradient), plus the
Guo body-force scheme that Rung 1 needs to drive a channel.

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
    np.copyto(f_bb, f)                     # pre-stream copy, for probe.forces
    stream(f, buf)
    outlet_zero_gradient(f, prev=f_out)    # step 6, open boundaries
    inlet_velocity(f, U, u_in=u_in, work=inlet_work)

That order is what makes the reflection work: ``f_pre[OPP[i]]`` at a solid cell
is the population that streamed *into* the solid last step travelling in
direction ``OPP[i]``. Writing it into slot ``i`` sends it straight back out
along ``E[i]`` on the next stream.

The open boundaries come **after** :func:`lbm.core.stream` and not before
(``old-Docs/STATE1.md`` D-020). ``stream`` is periodic in ``x``, so after it the
inlet column holds wrap-around garbage in exactly its ``ex = +1`` populations
and the outlet column holds it in the ``ex = -1`` ones — precisely the unknowns
these two functions overwrite. ``f_bb`` is the extra ``(9, ny, nx)`` buffer
:func:`lbm.probe.forces` needs; it is the **pre-stream** state, distinct from
``f_pre``, which is pre-collision.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from lbm.core import CS2, E_F32, OPP, Q, W

__all__ = [
    "bounce_back",
    "moving_wall",
    "inlet_profile",
    "inlet_velocity",
    "outlet_zero_gradient",
    "force_velocity_shift",
    "apply_body_force",
]

#: Lattice velocity ceiling (``CLAUDE.md`` constraint 3). Compressibility error
#: scales as Mach squared, so anything at or above this is out of the model's
#: range. Warned about here; ``lbm/units.py`` (T009) is where it raises.
U_MAX: float = 0.1


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

    Wall-offset convention (``old-Docs/STATE1.md`` § Decisions, **D-009**, closing
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
    :func:`bounce_back` — see the module docstring (``old-Docs/STATE1.md`` D-011).
    Apply it **after** :func:`bounce_back` if the masks overlap; this function
    writes the complete reflection for its cells, so the last writer wins.

    Wall offset is unchanged from D-009: the no-slip / moving-wall plane sits
    **halfway** between the last fluid node and the solid node. For a cavity of
    ``n x n`` cells with a one-cell solid border, the fluid nodes are
    ``1 .. n - 2`` and the characteristic length is ``L = n - 2``.

    Corner cells (``old-Docs/STATE1.md`` § Decisions, **D-013**, closes Q-003)
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
    ``CORNERS`` and in ``old-Docs/STATE1.md`` D-013.

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


def inlet_profile(
    ny: int,
    U: float,
    profile: str = "uniform",
    *,
    solid: NDArray[np.bool_] | None = None,
    col: int = 0,
    uy: float = 0.0,
) -> NDArray[np.float32]:
    """The prescribed inlet velocity profile, built once at setup.

    ``DOCS/IDEA2.md`` § The method, step 6 — the velocity that
    :func:`inlet_velocity` imposes. Two profiles, selectable:

    ``"uniform"``
        ``ux = U`` on every fluid row of the inlet column.
    ``"parabolic"``
        ``ux = 4 U y_ (H - y_) / H^2`` — the Poiseuille shape, **peak** ``U``
        (not mean; the mean is ``2U/3``).

    The parabola uses the half-way wall convention (``old-Docs/STATE1.md`` D-009):
    with fluid rows ``y0 .. y1`` in the inlet column, the walls are the planes
    ``y0 - 0.5`` and ``y1 + 0.5``, so ``H = y1 - y0 + 1`` and
    ``y_ = y - (y0 - 0.5)``. That puts ``ux = 0`` exactly on the wall planes
    rather than on the last fluid row, which is the convention Rung 1 measured
    at 0.365% against 12.7%/14.8% for the two rivals.

    Solid rows get zero velocity — :func:`inlet_velocity` skips them anyway.

    Args:
        ny: number of rows in the domain.
        U: peak (parabolic) or uniform (uniform) streamwise lattice velocity.
        profile: ``"uniform"`` or ``"parabolic"``.
        solid: optional solid mask, shape ``(ny, nx)``, ``bool``. Its column
            ``col`` decides which rows are fluid. ``None`` means every row is.
        col: index of the inlet column, used only to read ``solid``.
        uy: cross-stream inlet velocity, uniform over the fluid rows. Normally
            0; nonzero is how a run is deliberately perturbed to trip the
            cylinder wake out of its unstable-symmetric state.

    Returns:
        ``u_in``, shape ``(2, ny)``, ``float32``, ``(ux, uy)``
        (``old-Docs/STATE1.md`` D-005).

    Raises:
        ValueError: on an unknown ``profile``, or if the inlet column has no
            fluid rows.

    Warns:
        UserWarning: if the resulting ``max|u| >= 0.1`` (``CLAUDE.md``
            constraint 3). A warning here and not an error: the ceiling is a
            modelling limit, and ``lbm/units.py`` (T009) is the layer that
            refuses outright.
    """
    if profile not in ("uniform", "parabolic"):
        raise ValueError(
            f"profile must be 'uniform' or 'parabolic' (got {profile!r})."
        )

    fluid_rows = (
        np.ones(ny, dtype=bool) if solid is None else ~np.asarray(solid)[:, col]
    )
    rows = np.flatnonzero(fluid_rows)
    if rows.size == 0:
        raise ValueError(
            f"inlet column {col} has no fluid rows — the whole column is solid."
        )

    u_in = np.zeros((2, ny), dtype=np.float32)

    if profile == "uniform":
        u_in[0, rows] = np.float32(U)
    else:
        y0, y1 = int(rows[0]), int(rows[-1])
        H = float(y1 - y0 + 1)
        y_ = rows.astype(np.float32) - np.float32(y0 - 0.5)
        u_in[0, rows] = np.float32(4.0 * U) * y_ * (np.float32(H) - y_) / np.float32(H * H)

    u_in[1, rows] = np.float32(uy)

    peak = float(np.max(np.hypot(u_in[0], u_in[1])))
    if peak >= U_MAX:
        warnings.warn(
            f"inlet peak lattice velocity {peak:.4f} >= {U_MAX} "
            "(CLAUDE.md constraint 3): compressibility error scales as Mach "
            "squared and this profile is outside the model's range. Lower U, or "
            "raise the resolution and lower U in proportion.",
            UserWarning,
            stacklevel=2,
        )

    return u_in


def inlet_velocity(
    f: NDArray[np.float32],
    U: float = 0.0,
    *,
    profile: str = "uniform",
    solid: NDArray[np.bool_] | None = None,
    col: int = 0,
    uy: float = 0.0,
    u_in: NDArray[np.float32] | None = None,
    work: NDArray[np.float32] | None = None,
    fluid: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Zou–He velocity inlet on a left-facing column, in place.

    ``DOCS/IDEA2.md`` § The method, step 6. **This is Zou–He**, not
    bounce-back: an inlet has to give the column a density consistent with the
    velocity it prescribes, and bounce-back cannot — it reflects whatever
    arrives. ``DOCS/IDEA2.md`` § Stability lists "sim fine but wake is wrong"
    against a bad inlet.

    Call it **after** :func:`lbm.core.stream` (module docstring,
    ``old-Docs/STATE1.md`` D-020). At that moment the three ``ex = +1`` populations
    in the inlet column are unknown — nothing upstream streamed into them, and
    the periodic wrap has filled them with the outlet's values. Those three are
    what this overwrites. With this package's direction order
    (``0`` rest, ``1 +x``, ``2 +y``, ``3 -x``, ``4 -y``, ``5 ++``, ``6 -+``,
    ``7 --``, ``8 +-``) the unknowns are ``i = 1, 5, 8`` and Zou–He gives::

        rho = (f0 + f2 + f4 + 2 (f3 + f6 + f7)) / (1 - ux)
        f1  = f3 + (2/3) rho ux
        f5  = f7 - (1/2)(f2 - f4) + (1/6) rho ux + (1/2) rho uy
        f8  = f6 + (1/2)(f2 - f4) + (1/6) rho ux - (1/2) rho uy

    The first line is mass conservation solved for the unknown density; the
    other three are chosen so that the zeroth and both first moments of the
    completed column reproduce ``rho``, ``rho ux`` and ``rho uy`` exactly. The
    unit tests assert all three moments rather than the formulas, so a
    transcription error cannot pass.

    Solid rows in the inlet column are left untouched — :func:`bounce_back` owns
    them.

    Allocation (``CLAUDE.md`` § conventions): pass ``u_in``, ``work`` **and**
    ``fluid`` and the call allocates nothing. Building the profile every step
    would allocate ``O(ny)``, so the runner (T006) keeps the returned ``u_in``
    and hands it back (``old-Docs/STATE1.md`` D-006). ``fluid`` closes the same hole
    for the row mask: without it this function evaluates ``~solid[:, col]``, an
    ``O(ny)`` boolean, on **every** step — transient, freed immediately, and
    therefore invisible to a heap-growth test, but an allocation inside the step
    loop all the same. Session 6 recorded it as the last one; T010's
    preallocation audit is what closed it.

    Args:
        f: distribution, shape ``(9, ny, nx)``, ``float32``. Modified in place
            in column ``col`` only, and only in directions 1, 5 and 8.
        U: inlet velocity passed to :func:`inlet_profile`. Ignored when ``u_in``
            is supplied.
        profile: ``"uniform"`` or ``"parabolic"``. Ignored when ``u_in`` is
            supplied.
        solid: optional solid mask, shape ``(ny, nx)``, ``bool``. Rows that are
            solid in column ``col`` are skipped.
        col: index of the inlet column. Any column may be used, but it is
            always treated as **left-facing** — the unknowns are the ``+x``
            directions.
        uy: cross-stream inlet velocity. Ignored when ``u_in`` is supplied.
        u_in: prescribed profile, shape ``(2, ny)``, ``float32``. Built by
            :func:`inlet_profile` when ``None``.
        work: optional scratch, shape ``(>=5, ny)``, ``float32``.
        fluid: precomputed ``~solid[:, col]``, shape ``(ny,)``, ``bool``. Supply
            it to make the call allocation-free; ``None`` derives it from
            ``solid`` each call.

    Returns:
        ``u_in`` — the profile that was imposed, for the caller to cache.
    """
    ny = f.shape[1]

    if u_in is None:
        u_in = inlet_profile(ny, U, profile, solid=solid, col=col, uy=uy)

    if work is None:
        work = np.empty((5, ny), dtype=np.float32)
    rho, rho_ux, dn, half_ruy, tmp = work[0], work[1], work[2], work[3], work[4]

    fc = f[:, :, col]  # (9, ny) view into f
    ux, uy_arr = u_in[0], u_in[1]

    if fluid is None and solid is not None:
        fluid = ~solid[:, col]

    # rho = (f0 + f2 + f4 + 2 (f3 + f6 + f7)) / (1 - ux)
    np.add(fc[3], fc[6], out=rho)
    rho += fc[7]
    rho *= np.float32(2.0)
    rho += fc[0]
    rho += fc[2]
    rho += fc[4]
    np.subtract(np.float32(1.0), ux, out=tmp)
    rho /= tmp

    np.multiply(rho, ux, out=rho_ux)

    # The (1/2)(f2 - f4) transverse correction, shared by f5 and f8. Read
    # before anything is written; directions 2 and 4 are never overwritten here.
    np.subtract(fc[2], fc[4], out=dn)
    dn *= np.float32(0.5)

    np.multiply(rho, uy_arr, out=half_ruy)
    half_ruy *= np.float32(0.5)

    # f1 = f3 + (2/3) rho ux
    np.multiply(rho_ux, np.float32(2.0 / 3.0), out=tmp)
    tmp += fc[3]
    _write_col(fc, 1, tmp, fluid)

    # f5 = f7 - (1/2)(f2 - f4) + (1/6) rho ux + (1/2) rho uy
    np.multiply(rho_ux, np.float32(1.0 / 6.0), out=tmp)
    tmp -= dn
    tmp += half_ruy
    tmp += fc[7]
    _write_col(fc, 5, tmp, fluid)

    # f8 = f6 + (1/2)(f2 - f4) + (1/6) rho ux - (1/2) rho uy
    np.multiply(rho_ux, np.float32(1.0 / 6.0), out=tmp)
    tmp += dn
    tmp -= half_ruy
    tmp += fc[6]
    _write_col(fc, 8, tmp, fluid)

    return u_in


def _write_col(
    fc: NDArray[np.float32],
    i: int,
    value: NDArray[np.float32],
    fluid: NDArray[np.bool_] | None,
) -> None:
    """Write ``value`` into direction ``i`` of a column view, fluid rows only."""
    if fluid is None:
        np.copyto(fc[i], value)
    else:
        np.copyto(fc[i], value, where=fluid)


def outlet_zero_gradient(
    f: NDArray[np.float32],
    *,
    col: int = -1,
    src: int = -2,
    prev: NDArray[np.float32] | None = None,
    lam: float | None = None,
) -> None:
    """Zero-gradient outflow, in place — plain copy or convective.

    ``DOCS/IDEA2.md`` § The method, step 6, and § Stability, the row
    "reflections from the right edge / outlet BC reflecting".

    **Plain copy** (``prev=None``, the default). The outlet column is given the
    whole distribution of its inboard neighbour, all nine directions::

        f[:, :, -1] = f[:, :, -2]

    **Convective** (pass ``prev``). The outlet column is advanced by the 1D
    advection equation ``df/dt + lam df/dx = 0``, discretised implicitly::

        f[:, :, -1] = (prev + lam f[:, :, -2]) / (1 + lam)

    where ``prev`` is this column's value at the previous step. The plain copy
    is exactly the ``lam -> inf`` limit of that expression, which is why both
    live in one function.

    Which to use, measured (``old-Docs/STATE1.md`` **D-021**)
    ----------------------------------------------------
    A wave leaving through a copied column reflects far more than the name
    "zero-gradient" suggests. ``tests/test_probe.py`` fires a smooth Gaussian
    pressure pulse (``sigma = 10`` cells) at the boundary and measures what
    comes back:

    ======================================  ==========
    outlet                                  reflected
    ======================================  ==========
    plain copy                              **35%**
    convective, ``lam = 0.4``               4.7%
    convective, ``lam = cs = 0.577``        **0.6%**
    convective, ``lam = 1.0``               7.5%
    ======================================  ==========

    The minimum is sharp and it sits exactly at the lattice speed of sound,
    which is what the theory says: a disturbance travelling out at speed ``c``
    is absorbed perfectly by an advection boundary tuned to ``c``, and a
    pressure pulse travels at ``cs``. Hence ``lam`` defaults to ``sqrt(CS2)``
    when ``prev`` is supplied, and the runner supplies it. The acceptance
    criterion — under 5% — is met by the convective form, not by the copy.

    ``lam = U`` instead of ``cs`` is the other defensible tuning: vorticity in a
    wake is *advected* at roughly the free-stream speed rather than radiated at
    ``cs``. It is exposed rather than chosen here because Rung 3 is the run that
    can measure which one leaves the wake cleaner.

    Call it **after** :func:`lbm.core.stream` and **before**
    :func:`inlet_velocity` (module docstring, ``old-Docs/STATE1.md`` D-020).
    Streaming is periodic in ``x``, so without this the outlet column's
    ``ex = -1`` populations are whatever wrapped around from the inlet.

    Constraint 12's "object at least 8 diameters from the outlet" is the other
    half of this; :func:`lbm.geometry.check_mask` enforces it. This boundary
    makes the outlet quiet, not free.

    Args:
        f: distribution, shape ``(9, ny, nx)``, ``float32``, modified in place
            in column ``col`` only.
        col: outlet column, default the last.
        src: column copied from, default the second-to-last. Passing
            ``col=0, src=1`` makes the *left* edge absorbing instead, which is
            how the reflection test stops its left-going pulse from wrapping
            back in.
        prev: the outlet column at the previous step, shape ``(9, ny)``,
            ``float32``. Owned by the caller (the runner allocates it once) and
            **updated in place** by this call, so the next step gets it for
            free. ``None`` selects the plain copy.
        lam: advection speed of the convective form, lattice units. Defaults to
            ``sqrt(CS2)``. Ignored when ``prev`` is ``None``.
    """
    out = f[:, :, col]

    if prev is None:
        np.copyto(out, f[:, :, src])
        return

    if lam is None:
        lam = float(np.sqrt(CS2))

    np.multiply(f[:, :, src], np.float32(lam), out=out)
    out += prev
    out *= np.float32(1.0 / (1.0 + lam))
    np.copyto(prev, out)


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
            (``old-Docs/STATE1.md`` D-005). Modified in place.
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
