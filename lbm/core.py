"""D2Q9 lattice constants, macroscopic reduction and equilibrium distribution.

Implements ``DOCS/IDEA2.md`` § "The method, in the order the code runs it",
steps 1 through 4: macroscopic, equilibrium, collide, stream.

**This module is the single source of truth for the nine lattice constants**
(``CLAUDE.md`` constraint 4). No other module may redefine ``E``, ``W``,
``OPP`` or ``CS2`` — import them from here.

Conventions used throughout ``lbm/``:

* State is ``f`` of shape ``(9, ny, nx)``, index order ``(direction, y, x)``,
  dtype ``float32`` (constraint 4).
* Velocity is ``u`` of shape ``(2, ny, nx)``, index order ``(component, y, x)``
  with component 0 = ``ux`` and component 1 = ``uy``, matching the ``(ex, ey)``
  column order of ``E``.
* Lattice velocity magnitude must stay under 0.1 (constraint 3); the
  compressibility error of this model scales as Mach squared, and the
  equilibrium below is the second-order truncation that assumes it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- D2Q9 constants ---------------------------------------------------------
# Order is load-bearing: it is the order in DOCS/IDEA2.md § The method, and
# OPP indexes into it. Do not reorder.

#: Discrete velocities, shape ``(9, 2)``, columns ``(ex, ey)``.
E: NDArray[np.int32] = np.array(
    [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)],
    dtype=np.int32,
)

#: Lattice weights, shape ``(9,)``. Sum to 1.
W: NDArray[np.float32] = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=np.float32,
)

#: Index of the reversed direction, shape ``(9,)``. ``E[OPP[i]] == -E[i]``.
OPP: NDArray[np.int32] = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

#: Lattice speed of sound squared.
CS2: float = 1.0 / 3.0

#: ``E`` as ``float32``, for arithmetic. ``E`` itself stays integer so that
#: geometry and streaming can use it as an offset without casting.
E_F32: NDArray[np.float32] = E.astype(np.float32)

Q: int = 9


def nu_from_tau(tau: float) -> float:
    """Kinematic viscosity in lattice units from the relaxation time.

    ``DOCS/IDEA2.md`` § The method: ``nu = cs2 * (tau - 0.5) = (tau - 0.5) / 3``.

    Viscosity is not a free parameter (``CLAUDE.md`` constraint 2) — this is the
    only path to it, and there is deliberately no ``nu`` setter anywhere in the
    package. ``tau -> 0.5`` means ``nu -> 0`` means the simulation blows up, so
    ``tau <= 0.5`` is an error rather than a warning.

    Args:
        tau: BGK relaxation time, lattice units. Must be greater than 0.5.

    Returns:
        Kinematic viscosity in lattice units.

    Raises:
        ValueError: if ``tau <= 0.5``.
    """
    if tau <= 0.5:
        raise ValueError(
            f"tau must be greater than 0.5 (got tau={tau!r}): "
            "nu = (tau - 0.5) / 3, so tau <= 0.5 gives non-positive viscosity "
            "and the simulation diverges."
        )
    return (tau - 0.5) / 3.0


def macroscopic(
    f: NDArray[np.float32],
    rho: NDArray[np.float32] | None = None,
    u: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Density and velocity from the distribution function.

    ``DOCS/IDEA2.md`` § The method, step 1::

        rho = f.sum(0)
        u   = (e . f) / rho

    Args:
        f: distribution function, shape ``(9, ny, nx)``, ``float32``.
        rho: optional preallocated output, shape ``(ny, nx)``, ``float32``.
        u: optional preallocated output, shape ``(2, ny, nx)``, ``float32``,
            C-contiguous. Passing both outputs makes this call allocation-free,
            which is how the runner (T006) will use it.

    Returns:
        ``(rho, u)`` — shapes ``(ny, nx)`` and ``(2, ny, nx)``, both ``float32``.
        Component 0 of ``u`` is ``ux``, component 1 is ``uy``.
    """
    _, ny, nx = f.shape

    rho = np.sum(f, axis=0, dtype=np.float32, out=rho)

    if u is None:
        u = np.empty((2, ny, nx), dtype=np.float32)
    # (2, 9) @ (9, ny*nx) -> (2, ny*nx). Views, not copies: f and u are
    # C-contiguous, so the reshapes are free.
    np.matmul(E_F32.T, f.reshape(Q, -1), out=u.reshape(2, -1))
    u /= rho  # broadcasts (ny, nx) over both components

    return rho, u


def equilibrium(
    rho: NDArray[np.float32],
    u: NDArray[np.float32],
    feq: NDArray[np.float32] | None = None,
    work: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Second-order equilibrium distribution.

    ``DOCS/IDEA2.md`` § The method, step 2::

        feq_i = w_i * rho * (1 + 3(e_i.u) + 4.5(e_i.u)^2 - 1.5 u^2)

    This is a Mach-squared truncation of the Maxwellian, so it is only accurate
    while ``|u| < 0.1`` in lattice units (``CLAUDE.md`` constraint 3). The unit
    tests probe that range and no wider for exactly this reason.

    Args:
        rho: density, shape ``(ny, nx)``, ``float32``.
        u: velocity, shape ``(2, ny, nx)``, ``float32``, ``(ux, uy)``.
        feq: optional preallocated output, shape ``(9, ny, nx)``, ``float32``.
        work: optional preallocated scratch, shape ``(3, ny, nx)``, ``float32``.
            Supply both to make this call allocation-free (T006 does).

    Returns:
        ``feq``, shape ``(9, ny, nx)``, ``float32``.
    """
    ny, nx = rho.shape

    if feq is None:
        feq = np.empty((Q, ny, nx), dtype=np.float32)
    if work is None:
        work = np.empty((3, ny, nx), dtype=np.float32)

    ux, uy = u[0], u[1]
    usq, eu, tmp = work[0], work[1], work[2]

    # usq = ux^2 + uy^2, then premultiplied by 1.5 since that is its only use.
    np.multiply(ux, ux, out=usq)
    np.multiply(uy, uy, out=tmp)
    usq += tmp
    usq *= 1.5

    for i in range(Q):
        ex, ey = E_F32[i, 0], E_F32[i, 1]

        # eu = e_i . u
        np.multiply(ux, ex, out=eu)
        np.multiply(uy, ey, out=tmp)
        eu += tmp

        # feq_i = w_i * rho * (1 + 3 eu + 4.5 eu^2 - 1.5 u^2)
        out = feq[i]
        np.multiply(eu, eu, out=out)
        out *= 4.5
        np.multiply(eu, 3.0, out=tmp)
        out += tmp
        out += 1.0
        out -= usq
        out *= rho
        out *= W[i]

    return feq



# --- the Smagorinsky closure (T201) -----------------------------------------
#
# CLAUDE.md constraint 1 in its Phase 2 form (DOCS/STATE3.md **D-081**): exactly
# one turbulence closure, named, additive and switchable. Everything below is
# inert unless a caller passes a non-zero ``cs_smag``, and constraint 19 makes
# that a bitwise claim rather than a reassuring word.

#: The Smagorinsky constant at its literature value. Phase 2 does **not** tune
#: it (``DOCS/TASKS3.md`` § T201 Notes): a different value is a decision with a
#: measurement, recorded in ``DOCS/STATE3.md`` § Decisions, not an edit here.
#: It is not a default — the closure's default is **off**, ``cs_smag = 0.0``.
CS_SMAG_LITERATURE: float = 0.17

#: ``18 * sqrt(2)`` — the coefficient of the ``|Q|`` term under the root in
#: :func:`smagorinsky_tau_eff`, for ``cs2 = 1/3`` and a filter width of one
#: lattice unit. Derived in that function's docstring; defined once here so no
#: backend re-derives it (constraint 4's "no physics constant twice"), and kept
#: in ``float64`` so a backend rounds it to ``float32`` once, host-side, in
#: NumPy's own expression order (**D-057**).
SMAG_Q_COEFF: float = 18.0 * np.sqrt(2.0)


def smagorinsky_tau_eff(
    f: NDArray[np.float32],
    feq: NDArray[np.float32],
    tau: float,
    cs_smag: float,
    out: NDArray[np.float32] | None = None,
    work: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Per-cell effective relaxation time from the second moment of ``f - feq``.

    ``DOCS/IDEA4.md`` § The five things Phase 2 must get right, (1) and (2).
    The model is Smagorinsky as adapted to LBM by **Hou, Sterling, Chen and
    Doolen, "A Lattice Boltzmann Subgrid Model for High Reynolds Number Flows",
    Fields Inst. Comm. 6, 151-166 (1996)**; the same form is set out in Krueger
    et al., *The Lattice Boltzmann Method* (2017) section 12.3.

    Every normalisation choice, stated rather than inherited
    -------------------------------------------------------
    * **The filter width is one lattice unit**, ``Delta = 1``, so
      ``(Cs Delta)^2`` is ``Cs^2`` and the grid *is* the filter. There is no
      second length scale in this model and none is exposed.
    * **The strain norm is** ``|S| = sqrt(2 S_ab S_ab)``, and the eddy viscosity
      is ``nu_t = (Cs Delta)^2 |S|``. This is the convention that makes ``|S|``
      reduce to ``|du/dy|`` in a simple shear.
    * **The non-equilibrium momentum flux is** ``Q_ab = sum_i e_ia e_ib
      (f_i - feq_i)`` with **no** factor of two folded in, and
      ``|Q| = sqrt(Q_ab Q_ab)`` — in 2D, ``sqrt(Qxx^2 + 2 Qxy^2 + Qyy^2)``.

    The algebra, in full, because a test pins it
    -------------------------------------------
    Chapman-Enskog gives ``Q_ab = -2 rho cs2 tau_eff S_ab``, so::

        |S| = sqrt(2) |Q| / (2 rho cs2 tau_eff)

    and ``nu_t = cs2 (tau_eff - tau)`` (**constraint 2**: the closure moves the
    *relaxation time*, and viscosity is read off it, never assigned). Equating
    the two and clearing ``tau_eff`` gives a quadratic::

        tau_eff^2 - tau tau_eff - sqrt(2) Cs^2 |Q| / (2 rho cs2^2) = 0

    whose positive root, with ``cs2 = 1/3`` so ``1 / (2 cs2^2) = 4.5``, is::

        tau_eff = 0.5 (tau + sqrt(tau^2 + 18 sqrt(2) Cs^2 |Q| / rho))

    ``18 sqrt(2)`` is :data:`SMAG_Q_COEFF`. XLB's own 2D closure
    (``Autodesk/XLB:xlb/operator/collision/smagorinsky_les_bgk.py``, read in
    session 23 as a **cross-check and not as a source**) carries ``36`` in that
    position, which is this coefficient times ``sqrt(2)`` — the difference is
    exactly the strain-norm convention named above, and it is why that choice is
    written down here instead of assumed.

    Why ``tau_eff >= tau`` always
    -----------------------------
    ``|Q| >= 0`` and ``rho > 0``, so the radicand is never below ``tau^2``, and
    IEEE ``sqrt(x * x) == |x|`` exactly under round-to-nearest. The scalars are
    therefore built from ``tau32 = float32(tau)`` and ``tau32 * tau32`` rather
    than from a ``float64`` square rounded afterwards: that is what makes the
    ``cs_smag -> 0`` limit land exactly on ``tau`` instead of one ulp below it.
    The closure adds viscosity and never removes it, and a test asserts it on a
    strongly sheared case.

    Args:
        f: distribution function, shape ``(9, ny, nx)``, ``float32``. Read only.
        feq: equilibrium distribution, same shape and dtype. Read only.
        tau: base BGK relaxation time, greater than 0.5 (constraint 2).
        cs_smag: the Smagorinsky constant ``Cs``. **0.0 switches the closure
            off**, and this function then returns ``tau`` in every cell,
            exactly (constraint 19).
        out: optional preallocated output, shape ``(ny, nx)``, ``float32``.
        work: optional preallocated scratch, shape ``(4, ny, nx)``, ``float32``.
            Supply both to make this call allocation-free, which is how
            :class:`lbm.runner.Sim` uses it.

    Returns:
        ``tau_eff``, shape ``(ny, nx)``, ``float32``, elementwise ``>= tau``.

    Raises:
        ValueError: if ``tau <= 0.5`` or ``cs_smag < 0``.
    """
    if tau <= 0.5:
        raise ValueError(
            f"tau must be greater than 0.5 (got tau={tau!r}): "
            "nu = (tau - 0.5) / 3, so tau <= 0.5 gives non-positive viscosity."
        )
    if cs_smag < 0.0:
        raise ValueError(
            f"cs_smag must be non-negative (got {cs_smag!r}): the closure adds "
            "eddy viscosity and never removes it (CLAUDE.md constraint 2)."
        )

    _, ny, nx = f.shape
    if out is None:
        out = np.empty((ny, nx), dtype=np.float32)
    tau32 = np.float32(tau)

    if cs_smag == 0.0:
        # Constraint 19's limit, taken exactly rather than approached: no
        # arithmetic at all, so nu_t = cs2 (tau_eff - tau) is exactly zero and
        # the collision callers take their Phase 1 branch anyway.
        out.fill(tau32)
        return out

    if work is None:
        work = np.empty((4, ny, nx), dtype=np.float32)
    qxx, qxy, qyy, tmp = work[0], work[1], work[2], work[3]

    qxx.fill(0.0)
    qxy.fill(0.0)
    qyy.fill(0.0)

    # Q_ab = sum_i e_ia e_ib (f_i - feq_i). On D2Q9 every ``e`` component is in
    # {-1, 0, +1}, so each coefficient is exactly 0 or +-1 and the branches
    # below *are* the multiplication, performed exactly and without a second
    # temporary.
    for i in range(Q):
        ex = float(E_F32[i, 0])
        ey = float(E_F32[i, 1])
        if ex == 0.0 and ey == 0.0:
            continue  # the rest population carries no momentum flux

        np.subtract(f[i], feq[i], out=tmp)  # f_i^neq

        if ex != 0.0:
            qxx += tmp
        if ey != 0.0:
            qyy += tmp
        if ex * ey > 0.0:
            qxy += tmp
        elif ex * ey < 0.0:
            qxy -= tmp

    # |Q| = sqrt(Qxx^2 + 2 Qxy^2 + Qyy^2), in place, reusing the components.
    np.multiply(qxx, qxx, out=qxx)
    np.multiply(qxy, qxy, out=qxy)
    qxy *= np.float32(2.0)
    np.multiply(qyy, qyy, out=qyy)
    qxx += qxy
    qxx += qyy
    np.sqrt(qxx, out=qxx)

    # rho = sum_i f_i, into the component slot that is now free.
    np.sum(f, axis=0, dtype=np.float32, out=qyy)

    # tau_eff = 0.5 (tau + sqrt(tau^2 + 18 sqrt(2) Cs^2 |Q| / rho)).
    np.divide(qxx, qyy, out=out)
    out *= np.float32(SMAG_Q_COEFF * cs_smag * cs_smag)
    out += np.float32(tau32 * tau32)
    np.sqrt(out, out=out)
    out += tau32
    out *= np.float32(0.5)
    return out


def smagorinsky_omega(
    f: NDArray[np.float32],
    feq: NDArray[np.float32],
    tau: float,
    cs_smag: float,
    out: NDArray[np.float32] | None = None,
    work: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Per-cell **inverse** effective relaxation time, ``1 / tau_eff``.

    ``DOCS/IDEA4.md`` § The five things Phase 2 must get right, (2). The whole
    of the model is :func:`smagorinsky_tau_eff`; this is the reciprocal
    :func:`collide` and :func:`collide_stream` actually multiply by, and it is
    the T201 contract's named entry point (``DOCS/TASKS3.md`` § T201).

    Constraint 2 is why the two are separate functions rather than one:
    ``tau_eff`` is the quantity the model computes and the quantity viscosity is
    read off (:func:`lbm.probe.eddy_viscosity`), so the reciprocal is taken
    once, here, and never inverted back. ``1 / (1 / tau)`` is not ``tau`` in
    ``float32``, and an ``nu_t`` derived through that round trip would sit a few
    ulps from zero with the closure off instead of exactly zero.

    Args:
        f: distribution function, shape ``(9, ny, nx)``, ``float32``.
        feq: equilibrium distribution, same shape and dtype.
        tau: base BGK relaxation time, greater than 0.5.
        cs_smag: the Smagorinsky constant. ``0.0`` returns ``1 / tau`` in every
            cell.
        out: optional preallocated output, shape ``(ny, nx)``, ``float32``.
        work: optional preallocated scratch, shape ``(4, ny, nx)``, ``float32``.

    Returns:
        ``omega_eff = 1 / tau_eff``, shape ``(ny, nx)``, ``float32``.

    Raises:
        ValueError: if ``tau <= 0.5`` or ``cs_smag < 0``.
    """
    out = smagorinsky_tau_eff(f, feq, tau, cs_smag, out=out, work=work)
    np.divide(np.float32(1.0), out, out=out)
    return out


def collide(
    f: NDArray[np.float32],
    feq: NDArray[np.float32],
    tau: float,
    *,
    cs_smag: float = 0.0,
    smag_out: NDArray[np.float32] | None = None,
    smag_work: NDArray[np.float32] | None = None,
) -> None:
    """BGK collision, in place.

    ``DOCS/IDEA2.md`` § The method, step 3::

        f -= (f - feq) / tau

    Written as three in-place array operations rather than the literal
    expression, so that no temporary is allocated (``CLAUDE.md`` § conventions,
    "never allocate inside the step loop")::

        f -= feq        ->  f - feq
        f *= 1 - omega  ->  (f - feq)(1 - omega)
        f += feq        ->  feq + (f - feq)(1 - omega) == f - omega (f - feq)

    with ``omega = 1 / tau``. This is algebraically identical to the spec form,
    and the unit test asserts it against the literal expression.

    ``tau`` is the only handle on viscosity — ``nu = (tau - 0.5) / 3``
    (``CLAUDE.md`` constraint 2). Collide and stream are deliberately **not**
    fused; fusion is T010 and is gated on Rung 3 (constraint 6).

    The Smagorinsky closure (T201, ``DOCS/IDEA4.md`` § (1) and (2))
    --------------------------------------------------------------
    With ``cs_smag != 0`` the scalar ``omega = 1 / tau`` becomes the ``(ny, nx)``
    field :func:`smagorinsky_omega` returns, broadcast over the nine directions.
    Nothing else about the collision changes: it is the same three in-place
    operations in the same order, and the closure enters only through what they
    are multiplied by. That is what "additive and switchable" means in
    **D-081**.

    ``cs_smag == 0.0`` takes an explicit branch back to the two lines above
    rather than multiplying a zero-valued term into the arithmetic. That is
    **constraint 19**, and it is deliberate to the ulp: with the closure on, the
    per-cell factor is built by ``float32`` array operations, and a scalar
    ``float32(1 - 1/tau)`` is not required to equal ``1 - float32(1/tau)``
    computed elementwise. The branch makes ``Cs = 0`` *bitwise* what Phase 1
    shipped instead of within-a-tolerance of it — and it is the same branch
    **Q-201** tells T202 to carry onto the GPU, where **D-053**'s fused
    multiply-add contraction would otherwise make an algebraically-zero term
    change the result.

    Args:
        f: distribution function, shape ``(9, ny, nx)``, ``float32``. Modified
            in place; its buffer identity never changes.
        feq: equilibrium distribution, same shape and dtype.
        tau: BGK relaxation time, greater than 0.5.
        cs_smag: Smagorinsky constant. ``0.0`` — the default — is plain BGK,
            bitwise (constraint 19).
        smag_out: optional preallocated ``(ny, nx)`` ``float32`` for the closure's
            per-cell factor. **Used as scratch**: on return it holds
            ``1 - omega_eff``, not ``omega_eff``. Ignored when ``cs_smag`` is 0.
        smag_work: optional preallocated ``(4, ny, nx)`` ``float32`` scratch for
            :func:`smagorinsky_tau_eff`. Ignored when ``cs_smag`` is 0.

    Raises:
        ValueError: if ``tau <= 0.5`` (via :func:`nu_from_tau`'s condition), or
            if ``cs_smag < 0``.
    """
    if tau <= 0.5:
        raise ValueError(
            f"tau must be greater than 0.5 (got tau={tau!r}): "
            "collision with tau <= 0.5 gives non-positive viscosity and diverges."
        )

    if cs_smag == 0.0:
        # Phase 1's collision, verbatim, and the closure below returns
        # before reaching it rather than sharing any arithmetic with it
        # (constraint 19).
        one_minus_omega = np.float32(1.0 - 1.0 / tau)

        f -= feq
        f *= one_minus_omega
        f += feq
        return

    # omega_eff is computed from the **pre-collision** f, before anything below
    # touches it, and is then turned into ``1 - omega_eff`` in its own buffer so
    # that the three operations stay the three operations.
    scale = smagorinsky_omega(f, feq, tau, cs_smag, out=smag_out, work=smag_work)
    np.subtract(np.float32(1.0), scale, out=scale)

    f -= feq
    f *= scale
    f += feq


def collide_stream(
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
    """Collide, bounce back and stream in **one pass per direction**.

    ``DOCS/IDEA2.md`` § Performance budget, third cheap win: "fuse collide+stream
    into one pass over ``f``". T010 is the task that spends constraint 6, which
    lifted when Rung 3 went green.

    The fusion crosses :func:`lbm.boundary.bounce_back`, which sits *between*
    collide and stream in the D-020 order — that is the point. Unfused, the step
    walks the whole ``(9, ny, nx)`` array seven times (three collide operations,
    the masked reflection, the ``f_bb`` snapshot, the shift into ``buf``, the
    copy back). Fused, direction ``i`` is loaded once and stays in cache for all
    of it::

        for i in 0..8:
            s   = feq[i] + (f[i] - feq[i]) (1 - 1/tau)   # collide
            s   = f_pre[OPP[i]]  where solid             # bounce back
            buf[i] = shift(s, E[i])                      # stream

    with ``s`` being ``f_bb[i]`` when the caller wants the pre-stream snapshot
    :func:`lbm.probe.forces` consumes (``old-Docs/STATE1.md`` **D-020**), and ``f[i]``
    itself when it does not. ``buf`` is copied back into ``f`` at the end so that
    ``f`` keeps its buffer identity, which T006's allocation test asserts.

    **The arithmetic is unchanged, element by element and in the same order**, so
    the result is *bitwise* equal to the unfused sequence
    ``collide -> bounce_back -> copyto(f_bb) -> stream``. That is what keeps
    constraint 11's bit-identical restart true across the fusion, and
    ``tests/test_perf.py`` asserts the equality on a small grid rather than
    trusting this paragraph.

    The Guo body force (D-010) is **not** folded in: it is applied between
    collision and bounce-back, and the only case that uses it is Rung 1, a 22x16
    channel where speed is irrelevant. :class:`lbm.runner.Sim` keeps the unfused
    sequence when a force is present, so Rung 1's arithmetic is untouched.

    Args:
        f: distribution, shape ``(9, ny, nx)``, ``float32``, modified in place.
        feq: equilibrium distribution, same shape and dtype.
        tau: BGK relaxation time, greater than 0.5 (constraint 2).
        buf: preallocated scratch, same shape and dtype. Holds the streamed
            state; not otherwise meaningful after the call.
        f_pre: pre-**collision** copy of ``f`` (D-011), same shape and dtype.
            Required when ``solid`` is given.
        solid: solid mask, shape ``(ny, nx)``, ``bool``. ``None`` skips the
            reflection entirely.
        f_bb: preallocated ``(9, ny, nx)`` ``float32`` to receive the
            **pre-stream** state (D-020). ``None`` stages in ``f`` instead, which
            is fine for a caller that never measures forces.
        cs_smag: Smagorinsky constant (T201). ``0.0`` — the default — is plain
            BGK, **bitwise** (constraint 19); see :func:`collide` for why that
            is a branch and not a zero-valued term. The field is computed once,
            before the direction loop, from the pre-collision ``f``: the loop
            writes into ``f_bb`` and reads ``f``, so there is exactly one moment
            at which the whole pre-collision state exists and this is it.
        smag_out: optional preallocated ``(ny, nx)`` ``float32``, used as
            scratch; on return it holds ``1 - omega_eff``.
        smag_work: optional preallocated ``(4, ny, nx)`` ``float32`` scratch.

    Returns:
        ``f``, collided, reflected and streamed — the same object passed in.

    Raises:
        ValueError: if ``tau <= 0.5``, ``cs_smag < 0``, or ``solid`` is given
            without ``f_pre``.
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

    # A ``float32`` scalar with the closure off — Phase 1's own expression,
    # multiplied in exactly as Phase 1 multiplied it — and an ``(ny, nx)``
    # ``float32`` field with it on, which broadcasts over one direction's plane
    # the same way. Constraint 19 lives in this branch (see :func:`collide`).
    scale: NDArray[np.float32] | np.float32
    if cs_smag == 0.0:
        scale = np.float32(1.0 - 1.0 / tau)
    else:
        scale = smagorinsky_omega(
            f, feq, tau, cs_smag, out=smag_out, work=smag_work
        )
        np.subtract(np.float32(1.0), scale, out=scale)

    for i in range(Q):
        # Where the post-collision, post-reflection state for direction i goes.
        # It is the pre-stream snapshot probe.forces consumes (D-020).
        s = f[i] if f_bb is None else f_bb[i]

        # collide: s = feq[i] + (f[i] - feq[i]) (1 - omega), the same three
        # operations in the same order as `collide`, on one direction.
        np.subtract(f[i], feq[i], out=s)
        s *= scale
        s += feq[i]

        # bounce back: solid cells emit what arrived at them, reversed (D-011).
        if solid is not None:
            np.copyto(s, f_pre[OPP[i]], where=solid)

        # stream: shift along E[i] into buf, periodic on both axes.
        ex = int(E[i, 0])
        ey = int(E[i, 1])
        dst = buf[i]
        for dy, sy in _shift_blocks(ey):
            for dx, sx in _shift_blocks(ex):
                dst[dy, dx] = s[sy, sx]

    np.copyto(f, buf)
    return f


def _shift_blocks(shift: int) -> list[tuple[slice, slice]]:
    """Destination/source slice pairs for a periodic shift of +-1 along one axis.

    ``dst[d] = src[s]`` for each returned ``(d, s)`` reproduces
    ``np.roll(src, shift, axis)`` without allocating the temporary that
    ``np.roll`` itself would. Only shifts in ``{-1, 0, +1}`` occur on D2Q9, but
    the arithmetic below is valid for any ``|shift| < n``.
    """
    if shift == 0:
        return [(slice(None), slice(None))]
    if shift > 0:
        # bulk moves up by `shift`; the top `shift` rows wrap to the bottom
        return [
            (slice(shift, None), slice(None, -shift)),
            (slice(None, shift), slice(-shift, None)),
        ]
    return [
        (slice(None, shift), slice(-shift, None)),
        (slice(shift, None), slice(None, -shift)),
    ]


def stream(f: NDArray[np.float32], buf: NDArray[np.float32]) -> NDArray[np.float32]:
    """Advection, in place, periodic in both directions.

    ``DOCS/IDEA2.md`` § The method, step 4::

        f[i] = np.roll(np.roll(f[i], ey_i, axis=0), ex_i, axis=1)

    **Sign convention.** A positive ``roll`` shift moves array *contents* toward
    higher indices, so the population in direction ``i`` at cell ``(y, x)`` ends
    up at ``(y + ey_i, x + ex_i)`` — it moves one cell **along** ``E[i]``, which
    is what advection means. Axis 0 is ``y`` and takes ``ey``; axis 1 is ``x``
    and takes ``ex`` (constraint 4's ``(direction, y, x)`` order). The unit test
    puts a single-cell spike in each direction and asserts it lands on
    ``cell + E[i]``.

    Both axes wrap. A channel is therefore periodic in ``x`` for free, and
    wrap-around in ``y`` is harmless as long as the top and bottom rows are
    solid — :func:`lbm.boundary.bounce_back` overwrites those cells anyway.

    The shift is done with block copies into ``buf`` and copied back, rather
    than with ``np.roll``, because ``np.roll`` allocates. ``f`` keeps its buffer
    identity across the call, which T006's restart test relies on. Collide and
    stream stay separate passes (constraint 6).

    Args:
        f: distribution function, shape ``(9, ny, nx)``, ``float32``. Modified
            in place.
        buf: preallocated scratch of the same shape and dtype. Its contents
            after the call are the streamed state and are not otherwise
            meaningful.

    Returns:
        ``f``, streamed — the same object that was passed in.
    """
    for i in range(Q):
        ex = int(E[i, 0])
        ey = int(E[i, 1])
        src = f[i]
        dst = buf[i]
        for dy, sy in _shift_blocks(ey):
            for dx, sx in _shift_blocks(ex):
                dst[dy, dx] = src[sy, sx]

    np.copyto(f, buf)
    return f
