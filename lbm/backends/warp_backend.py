"""The Warp backend: the whole timestep on the GPU, arithmetic unchanged.

Implements ``DOCS/IDEA3.md`` § What Phase 1 is, concretely — the ``lbm/`` box,
"numpy or warp backend, same API, same physics" — and § Performance budget,
whose floors T103 measures against.

Why ``equilibrium`` was written first (T102): at 1M cells it is **39.9 ms of a
~75 ms step** on NumPy (``old-Docs/STATE1.md`` § Performance baseline), over half
the budget.

What this module is not
-----------------------
It is **not a rewrite of the physics**. Every kernel below is a transcription of
its :mod:`lbm.core` / :mod:`lbm.boundary` counterpart, term for term and *in the
same order* (``CLAUDE.md`` constraint 1 in its Phase 1 form). Where a kernel
would read "better" written differently — the obvious one is ``equilibrium``,
where a GPU thread has registers and does not need :mod:`lbm.core`'s ``work``
scratch — the arithmetic is still emitted in core's order and the difference is
noted in a comment. NumPy is the oracle (**D-043**); a GPU that disagrees with it
is a broken backend, never a new answer.

Where a boundary's arithmetic depends on a **scalar** that NumPy computes in
``float64`` and then rounds once to ``float32`` — the Ladd wall correction
``6 w_i rho_w (e_i . u_wall)``, Guo's ``(1 - 1/(2 tau)) w_i`` prefactor, the
convective outlet's ``1 / (1 + lam)`` — that scalar is computed **on the host**,
in exactly NumPy's expression order, and uploaded. Recomputing it inside the
kernel in ``float32`` would be a second rounding and a difference constraint 1
does not permit us to introduce for tidiness.

The nine constants
------------------
``E``, ``E_F32``, ``W``, ``OPP`` and ``CS2`` are **imported from**
:mod:`lbm.core` and uploaded to the device **once, at construction**
(``CLAUDE.md`` constraint 4 / "no physics constant twice"). No literal lattice
constant appears in a kernel below. The numeric literals that do appear are the
ones the corresponding NumPy function itself writes as literals — ``1.5``,
``3.0``, ``4.5`` in :func:`_equilibrium_kernel`, ``2/3`` and ``1/6`` in
:func:`_inlet_kernel` — and rewriting them through ``CS2`` would change the
emitted arithmetic, which constraint 1 forbids.

Where the state lives (T103, superseding **D-052**)
---------------------------------------------------
**The state lives on the device.** T102 took *host* arrays at this boundary and
copied in and out per kernel call (**D-052**), which is why it quoted no speed
number. T103 widens the seam the way whole-step parity forces it
(:mod:`lbm.backends`, "What T103 added"): :meth:`WarpBackend.empty` and
:meth:`WarpBackend.zeros` allocate **device** arrays, :class:`lbm.runner.Sim`
owns those, and every kernel below takes and returns them. A timestep therefore
moves **no bytes across the bus**. The only transfers left are
:meth:`WarpBackend.download` / :meth:`WarpBackend.to_host`, which
:class:`lbm.runner.Sim` calls at *frame* and *probe* cadence — constraint 8, the
live path must never block the physics — and which are the only calls that
synchronise the device.

Boolean masks are held as ``uint8`` on the device (Warp has no packed ``bool``
array and a per-cell ``int32`` mask would be four times the traffic in the
reflection). :meth:`upload` converts on the way in and :meth:`download` converts
back, so a caller only ever sees ``bool`` — the one dtype mapping in this file,
and the reason ``uint8`` device arrays are always masks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

import warp as wp

from lbm.core import CS2, E, E_F32, OPP, Q, SMAG_Q_COEFF, W

__all__ = ["WarpBackend"]


# Zou-He's two rational coefficients, as ``float32`` constants rather than
# in-kernel literals, so that what the kernel multiplies by is visibly the same
# number ``lbm.boundary.inlet_velocity`` writes as ``np.float32(2.0 / 3.0)``.
_TWO_THIRDS = wp.constant(wp.float32(2.0 / 3.0))
_ONE_SIXTH = wp.constant(wp.float32(1.0 / 6.0))


# --- kernels ----------------------------------------------------------------
# One transcription each of lbm.core.macroscopic / equilibrium / collide /
# stream / collide_stream and lbm.boundary.bounce_back / moving_wall /
# inlet_velocity / outlet_zero_gradient / force_velocity_shift /
# apply_body_force. Read them beside the originals; the operation order is the
# same.


@wp.kernel
def _macroscopic_kernel(
    f: wp.array3d(dtype=wp.float32),
    e: wp.array2d(dtype=wp.float32),
    rho: wp.array2d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
) -> None:
    """``rho = f.sum(0)``; ``u = (e . f) / rho``. One thread per cell.

    :func:`lbm.core.macroscopic` does the sum with ``np.sum(..., axis=0)`` and
    the dot with a ``(2, 9) @ (9, ny*nx)`` matmul, then divides ``u`` by ``rho``
    in place. Both reductions run over ``i`` in index order there and here.

    Args:
        f: distribution, ``(9, ny, nx)`` ``float32``, device.
        e: ``E_F32`` from :mod:`lbm.core`, ``(9, 2)`` ``float32``, device.
        rho: output density, ``(ny, nx)`` ``float32``, device.
        u: output velocity, ``(2, ny, nx)`` ``float32``, device.
    """
    y, x = wp.tid()

    r = float(0.0)
    for i in range(9):
        r += f[i, y, x]

    ux = float(0.0)
    uy = float(0.0)
    for i in range(9):
        ux += e[i, 0] * f[i, y, x]
        uy += e[i, 1] * f[i, y, x]

    rho[y, x] = r
    u[0, y, x] = ux / r
    u[1, y, x] = uy / r


@wp.kernel
def _equilibrium_kernel(
    rho: wp.array2d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    e: wp.array2d(dtype=wp.float32),
    w: wp.array(dtype=wp.float32),
    feq: wp.array3d(dtype=wp.float32),
) -> None:
    """``feq_i = w_i rho (1 + 3(e_i.u) + 4.5(e_i.u)^2 - 1.5 u^2)``.

    ``DOCS/IDEA2.md`` § The method, step 2, and :func:`lbm.core.equilibrium`
    operation for operation. ``usq`` is hoisted out of the direction loop and
    premultiplied by 1.5 (**D-008**) — kept here even though a GPU thread would
    not care, because the *arithmetic* is what constraint 1 pins, not the
    motivation for it. Core's ``work`` scratch has no analogue: its three
    ``(ny, nx)`` temporaries exist so NumPy can avoid allocating, and a thread
    holds them in registers instead.

    Args:
        rho: density, ``(ny, nx)`` ``float32``, device.
        u: velocity, ``(2, ny, nx)`` ``float32``, device.
        e: ``E_F32`` from :mod:`lbm.core`, ``(9, 2)`` ``float32``, device.
        w: ``W`` from :mod:`lbm.core`, ``(9,)`` ``float32``, device.
        feq: output equilibrium, ``(9, ny, nx)`` ``float32``, device.
    """
    y, x = wp.tid()

    ux = u[0, y, x]
    uy = u[1, y, x]

    # usq = 1.5 * (ux^2 + uy^2), hoisted out of the direction loop (D-008)
    usq = ux * ux
    tmp = uy * uy
    usq += tmp
    usq *= 1.5

    r = rho[y, x]

    for i in range(9):
        # eu = e_i . u
        eu = ux * e[i, 0]
        tmp = uy * e[i, 1]
        eu += tmp

        # feq_i = w_i * rho * (1 + 3 eu + 4.5 eu^2 - 1.5 u^2), in core's order:
        # out = eu*eu; out *= 4.5; tmp = 3*eu; out += tmp; out += 1;
        # out -= usq; out *= rho; out *= w_i
        out = eu * eu
        out *= 4.5
        tmp = eu * 3.0
        out += tmp
        out += 1.0
        out -= usq
        out *= r
        out *= w[i]

        feq[i, y, x] = out


@wp.kernel
def _collide_kernel(
    f: wp.array3d(dtype=wp.float32),
    feq: wp.array3d(dtype=wp.float32),
    one_minus_omega: wp.float32,
) -> None:
    """BGK collision, in place: ``f = feq + (f - feq)(1 - omega)``.

    The same three operations in the same order as :func:`lbm.core.collide`
    (``f -= feq``; ``f *= 1 - omega``; ``f += feq``), which is algebraically
    ``f -= (f - feq)/tau``. ``one_minus_omega`` is computed **on the host**, in
    ``float32``, exactly as core does — the kernel never re-derives it from
    ``tau`` and never touches ``nu`` (``CLAUDE.md`` constraint 2).

    Args:
        f: distribution, ``(9, ny, nx)`` ``float32``, device, modified in place.
        feq: equilibrium, ``(9, ny, nx)`` ``float32``, device.
        one_minus_omega: ``1 - 1/tau``, computed host-side in ``float32``.
    """
    i, y, x = wp.tid()

    v = f[i, y, x]
    v -= feq[i, y, x]
    v *= one_minus_omega
    v += feq[i, y, x]
    f[i, y, x] = v


@wp.kernel
def _smag_scale_kernel(
    f: wp.array3d(dtype=wp.float32),
    feq: wp.array3d(dtype=wp.float32),
    e: wp.array2d(dtype=wp.float32),
    smag_coeff: wp.float32,
    tau32: wp.float32,
    tau_sq: wp.float32,
    scale: wp.array2d(dtype=wp.float32),
) -> None:
    """``1 - 1/tau_eff`` per cell — the Smagorinsky closure (T202).

    A term-for-term transcription of :func:`lbm.core.smagorinsky_tau_eff`
    followed by :func:`lbm.core.smagorinsky_omega`'s single reciprocal and the
    ``1 - omega`` the collision multiplies by, all in one thread per cell.
    ``DOCS/IDEA4.md`` § The five things Phase 2 must get right, (1) and (2);
    the normalisation and the derivation are **D-085**, written out once in
    core's docstring and deliberately not repeated here.

    Where NumPy walks the nine directions with whole-array temporaries
    (``qxx``, ``qxy``, ``qyy``, ``tmp``), a thread has registers and needs no
    scratch — the difference the module docstring allows, since the operation
    *order* is unchanged: ``Q_ab`` accumulates over ``i`` in index order, then
    ``|Q| = sqrt(Qxx^2 + 2 Qxy^2 + Qyy^2)``, then ``|Q| / rho``, then the
    coefficient, then ``tau^2``, then the root. ``rho`` is summed over ``i`` in
    index order exactly as :func:`_macroscopic_kernel` sums it.

    The three ``float64``-then-rounded scalars — ``18 sqrt(2) Cs^2``, ``tau``
    and ``tau^2`` — are computed **on the host** in NumPy's own expression order
    by :meth:`WarpBackend._smag_scalars` and passed in (**D-057**). Nothing here
    re-derives a constant, and :data:`lbm.core.SMAG_Q_COEFF` never appears in a
    kernel (constraint 4).

    Args:
        f: pre-collision distribution, ``(9, ny, nx)`` ``float32``, device.
        feq: equilibrium, ``(9, ny, nx)`` ``float32``, device.
        e: ``E_F32`` from :mod:`lbm.core`, ``(9, 2)`` ``float32``, device.
        smag_coeff: ``float32(SMAG_Q_COEFF * cs_smag * cs_smag)``, host-side.
        tau32: ``float32(tau)``, host-side.
        tau_sq: ``float32(tau32 * tau32)``, host-side.
        scale: output, ``(ny, nx)`` ``float32``, device. Receives
            ``1 - omega_eff``, which is what NumPy's ``smag_out`` also holds on
            return from :func:`lbm.core.collide`.
    """
    y, x = wp.tid()

    # Q_ab = sum_i e_ia e_ib (f_i - feq_i), and rho = sum_i f_i. Every ``e``
    # component on D2Q9 is in {-1, 0, +1}, so each coefficient is exactly 0 or
    # +-1 and the branches below *are* the multiplication, as they are in core.
    qxx = float(0.0)
    qxy = float(0.0)
    qyy = float(0.0)
    rho = float(0.0)

    for i in range(9):
        fi = f[i, y, x]
        rho += fi

        ex = e[i, 0]
        ey = e[i, 1]
        if ex != 0.0 or ey != 0.0:  # the rest population carries no flux
            neq = fi - feq[i, y, x]
            if ex != 0.0:
                qxx += neq
            if ey != 0.0:
                qyy += neq
            if ex * ey > 0.0:
                qxy += neq
            elif ex * ey < 0.0:
                qxy -= neq

    # |Q| = sqrt(Qxx^2 + 2 Qxy^2 + Qyy^2), in core's order.
    qxx = qxx * qxx
    qxy = qxy * qxy
    qxy = qxy * 2.0
    qyy = qyy * qyy
    qxx = qxx + qxy
    qxx = qxx + qyy
    qmag = wp.sqrt(qxx)

    # tau_eff = 0.5 (tau + sqrt(tau^2 + 18 sqrt(2) Cs^2 |Q| / rho)).
    t = qmag / rho
    t = t * smag_coeff
    t = t + tau_sq
    t = wp.sqrt(t)
    t = t + tau32
    t = t * 0.5

    # omega_eff = 1 / tau_eff, reciprocated once and never inverted back
    # (**D-085**), then the 1 - omega the collision actually multiplies by.
    scale[y, x] = 1.0 - 1.0 / t


@wp.kernel
def _collide_smag_kernel(
    f: wp.array3d(dtype=wp.float32),
    feq: wp.array3d(dtype=wp.float32),
    scale: wp.array2d(dtype=wp.float32),
) -> None:
    """:func:`_collide_kernel` with a per-cell ``1 - omega`` (T202).

    The same three operations in the same order as :func:`lbm.core.collide`
    with the closure on: the only difference from :func:`_collide_kernel` is
    that the factor is read from an ``(ny, nx)`` field instead of a scalar,
    which is exactly what NumPy's broadcast does. Kept as a **separate kernel**
    rather than a flag on the first one so that ``cs_smag = 0`` launches the
    Phase 1 kernel with not one instruction changed — constraint 19, and the
    answer this task records for **Q-201**.

    Args:
        f: distribution, ``(9, ny, nx)`` ``float32``, device, modified in place.
        feq: equilibrium, ``(9, ny, nx)`` ``float32``, device.
        scale: ``1 - omega_eff``, ``(ny, nx)`` ``float32``, device, from
            :func:`_smag_scale_kernel`.
    """
    i, y, x = wp.tid()

    v = f[i, y, x]
    v -= feq[i, y, x]
    v *= scale[y, x]
    v += feq[i, y, x]
    f[i, y, x] = v


@wp.kernel
def _stream_kernel(
    src: wp.array3d(dtype=wp.float32),
    e: wp.array2d(dtype=wp.int32),
    dst: wp.array3d(dtype=wp.float32),
) -> None:
    """Advection, periodic on both axes: ``dst[i, y, x] = src[i, y-ey, x-ex]``.

    :func:`lbm.core.stream` **scatters** (``dst[y+ey, x+ex] = src[y, x]``, as
    block copies); a kernel **gathers** the identical assignment read backwards,
    because one thread per destination cell has no write conflicts. Advection
    still moves each population one cell **along** ``E[i]``; the spike test in
    ``tests/test_warp_backend.py`` asserts that on the GPU, direction by
    direction, exactly as Phase 0's does on the host.

    Streaming only moves values — no arithmetic — so this is the one kernel that
    is bit-identical to NumPy rather than merely within tolerance.

    Args:
        src: source distribution, ``(9, ny, nx)`` ``float32``, device.
        e: ``E`` from :mod:`lbm.core`, ``(9, 2)`` ``int32``, device.
        dst: destination, ``(9, ny, nx)`` ``float32``, device. Must not alias
            ``src``.
    """
    i, y, x = wp.tid()

    ny = dst.shape[1]
    nx = dst.shape[2]

    # |e| <= 1 on D2Q9, so a single wrap-around correction is enough.
    sy = y - e[i, 1]
    if sy < 0:
        sy += ny
    elif sy >= ny:
        sy -= ny

    sx = x - e[i, 0]
    if sx < 0:
        sx += nx
    elif sx >= nx:
        sx -= nx

    dst[i, y, x] = src[i, sy, sx]


@wp.kernel
def _bounce_back_kernel(
    f: wp.array3d(dtype=wp.float32),
    f_pre: wp.array3d(dtype=wp.float32),
    solid: wp.array2d(dtype=wp.uint8),
    opp: wp.array(dtype=wp.int32),
) -> None:
    """Half-way bounce-back on solid cells, in place: ``f[i] = f_pre[OPP[i]]``.

    :func:`lbm.boundary.bounce_back`, term for term. ``f_pre`` is the copy taken
    **before** collision of this timestep (**D-011**); the populations it holds
    at a solid cell are the ones that arrived there from the neighbouring fluid
    during the previous stream, and reversing them is the reflection.

    One thread per **cell**, looping the nine directions, rather than one per
    ``(i, y, x)``: the mask is then read once per cell instead of nine times,
    which at 2M cells is the difference between 2 MB and 18 MB of mask traffic.

    Args:
        f: post-collision distribution, ``(9, ny, nx)`` ``float32``, device,
            modified in place on solid cells only.
        f_pre: pre-collision copy, ``(9, ny, nx)`` ``float32``, device.
        solid: solid mask, ``(ny, nx)`` ``uint8``, nonzero is wall.
        opp: ``OPP`` from :mod:`lbm.core`, ``(9,)`` ``int32``, device.
    """
    y, x = wp.tid()

    if solid[y, x] == wp.uint8(0):
        return

    for i in range(9):
        f[i, y, x] = f_pre[opp[i], y, x]


@wp.kernel
def _moving_wall_kernel(
    f: wp.array3d(dtype=wp.float32),
    f_pre: wp.array3d(dtype=wp.float32),
    wall: wp.array2d(dtype=wp.uint8),
    opp: wp.array(dtype=wp.int32),
    c: wp.array(dtype=wp.float32),
) -> None:
    """Momentum-corrected (Ladd) bounce-back on a moving wall, in place.

    :func:`lbm.boundary.moving_wall`::

        f[i] = f_pre[OPP[i]] + 6 w_i rho_w (e_i . u_wall)

    The nine correction scalars ``c[i]`` are computed **on the host** by
    :meth:`WarpBackend.moving_wall`, in NumPy's own expression order
    (``np.float32(6.0 * float(W[i]) * rho_w * eu)`` — one ``float64``
    multiplication chain, rounded once), because recomputing them per thread in
    ``float32`` would round three more times. NumPy skips the addition entirely
    where ``e_i . u_wall == 0``, so the kernel skips it where ``c[i] == 0``.

    Args:
        f: post-collision distribution, ``(9, ny, nx)`` ``float32``, device,
            modified in place on ``wall`` cells only.
        f_pre: pre-collision copy, ``(9, ny, nx)`` ``float32``, device.
        wall: moving-wall mask, ``(ny, nx)`` ``uint8``, nonzero is wall.
        opp: ``OPP`` from :mod:`lbm.core`, ``(9,)`` ``int32``, device.
        c: the nine correction scalars, ``(9,)`` ``float32``, device.
    """
    y, x = wp.tid()

    if wall[y, x] == wp.uint8(0):
        return

    for i in range(9):
        v = f_pre[opp[i], y, x]
        if c[i] != 0.0:
            v += c[i]
        f[i, y, x] = v


@wp.kernel
def _inlet_kernel(
    f: wp.array3d(dtype=wp.float32),
    u_in: wp.array2d(dtype=wp.float32),
    fluid: wp.array(dtype=wp.uint8),
    col: wp.int32,
    masked: wp.int32,
) -> None:
    """Zou-He velocity inlet on a left-facing column, in place. One thread a row.

    :func:`lbm.boundary.inlet_velocity`, operation for operation. With this
    package's direction order the unknowns after streaming are ``i = 1, 5, 8``::

        rho = (f0 + f2 + f4 + 2 (f3 + f6 + f7)) / (1 - ux)
        f1  = f3 + (2/3) rho ux
        f5  = f7 - (1/2)(f2 - f4) + (1/6) rho ux + (1/2) rho uy
        f8  = f6 + (1/2)(f2 - f4) + (1/6) rho ux - (1/2) rho uy

    Directions 2 and 4 are read before anything is written and are never
    overwritten here, so the transverse correction is the same value NumPy's
    ``dn`` holds. Solid rows are skipped — :func:`lbm.boundary.bounce_back` owns
    them.

    Args:
        f: distribution, ``(9, ny, nx)`` ``float32``, device, modified in place
            in column ``col`` and directions 1, 5 and 8 only.
        u_in: prescribed profile, ``(2, ny)`` ``float32``, device.
        fluid: row mask, ``(ny,)`` ``uint8``, nonzero is fluid.
        col: the inlet column, already resolved to a non-negative index.
        masked: nonzero to honour ``fluid``; zero writes every row.
    """
    y = wp.tid()

    if masked != 0:
        if fluid[y] == wp.uint8(0):
            return

    f0 = f[0, y, col]
    f2 = f[2, y, col]
    f3 = f[3, y, col]
    f4 = f[4, y, col]
    f6 = f[6, y, col]
    f7 = f[7, y, col]

    ux = u_in[0, y]
    uy = u_in[1, y]

    # rho = (f0 + f2 + f4 + 2 (f3 + f6 + f7)) / (1 - ux)
    rho = f3 + f6
    rho += f7
    rho *= 2.0
    rho += f0
    rho += f2
    rho += f4
    tmp = 1.0 - ux
    rho /= tmp

    rho_ux = rho * ux

    # The (1/2)(f2 - f4) transverse correction, shared by f5 and f8.
    dn = f2 - f4
    dn *= 0.5

    half_ruy = rho * uy
    half_ruy *= 0.5

    # f1 = f3 + (2/3) rho ux
    t = rho_ux * _TWO_THIRDS
    t += f3
    f[1, y, col] = t

    # f5 = f7 - (1/2)(f2 - f4) + (1/6) rho ux + (1/2) rho uy
    t = rho_ux * _ONE_SIXTH
    t -= dn
    t += half_ruy
    t += f7
    f[5, y, col] = t

    # f8 = f6 + (1/2)(f2 - f4) + (1/6) rho ux - (1/2) rho uy
    t = rho_ux * _ONE_SIXTH
    t += dn
    t -= half_ruy
    t += f6
    f[8, y, col] = t


@wp.kernel
def _outlet_copy_kernel(
    f: wp.array3d(dtype=wp.float32),
    col: wp.int32,
    src: wp.int32,
) -> None:
    """Plain zero-gradient outflow: ``f[:, :, col] = f[:, :, src]``.

    :func:`lbm.boundary.outlet_zero_gradient` with ``prev=None``. All nine
    directions of a ``(9, ny, nx)`` ``float32`` distribution; **D-021** measured
    this form reflecting 35% of an outgoing pressure pulse, which is why it is
    the documented default and not the one the runner uses.

    Args:
        f: distribution, ``(9, ny, nx)`` ``float32``, device, modified in place
            in column ``col`` only.
        col: outlet column, already resolved to a non-negative index.
        src: source column, already resolved.
    """
    i, y = wp.tid()
    f[i, y, col] = f[i, y, src]


@wp.kernel
def _outlet_convective_kernel(
    f: wp.array3d(dtype=wp.float32),
    prev: wp.array2d(dtype=wp.float32),
    col: wp.int32,
    src: wp.int32,
    lam: wp.float32,
    inv_one_plus_lam: wp.float32,
) -> None:
    """Convective outflow: ``f[:, :, col] = (prev + lam f[:, :, src])/(1 + lam)``.

    :func:`lbm.boundary.outlet_zero_gradient` with ``prev`` supplied, in NumPy's
    order (``out = f_src * lam``; ``out += prev``; ``out *= 1/(1+lam)``;
    ``prev = out``). ``lam`` defaults to ``sqrt(CS2)``, which **D-021** measured
    at 0.6% reflection against the plain copy's 35%. ``1/(1 + lam)`` is computed
    host-side, in ``float64`` then rounded once, exactly as NumPy does.

    Args:
        f: distribution, ``(9, ny, nx)`` ``float32``, device, modified in place
            in column ``col`` only.
        prev: the outlet column at the previous step, ``(9, ny)`` ``float32``,
            device, updated in place.
        col: outlet column, resolved to a non-negative index.
        src: source column, resolved.
        lam: advection speed as ``float32``.
        inv_one_plus_lam: ``1 / (1 + lam)`` as ``float32``.
    """
    i, y = wp.tid()

    v = f[i, y, src] * lam
    v += prev[i, y]
    v *= inv_one_plus_lam
    f[i, y, col] = v
    prev[i, y] = v


@wp.kernel
def _force_shift_kernel(
    rho: wp.array2d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    half_gx: wp.float32,
    half_gy: wp.float32,
    do_x: wp.int32,
    do_y: wp.int32,
) -> None:
    """Guo's half-force correction to the velocity, in place: ``u += F/(2 rho)``.

    :func:`lbm.boundary.force_velocity_shift`. NumPy computes
    ``inv_rho = np.reciprocal(rho)`` once and then, per component and only when
    ``0.5 * g[c] != 0``, ``u[c] += inv_rho * np.float32(0.5 * g[c])``. The
    ``do_x`` / ``do_y`` flags are that skip: a component with no force must not
    pick up a rounding from a multiply-add by zero.

    Args:
        rho: density, ``(ny, nx)`` ``float32``, device.
        u: velocity, ``(2, ny, nx)`` ``float32``, device, modified in place.
        half_gx: ``np.float32(0.5 * gx)``, computed host-side.
        half_gy: ``np.float32(0.5 * gy)``, computed host-side.
        do_x: nonzero when ``0.5 * gx != 0``.
        do_y: nonzero when ``0.5 * gy != 0``.
    """
    y, x = wp.tid()

    inv_rho = 1.0 / rho[y, x]

    if do_x != 0:
        u[0, y, x] = u[0, y, x] + inv_rho * half_gx
    if do_y != 0:
        u[1, y, x] = u[1, y, x] + inv_rho * half_gy


@wp.kernel
def _body_force_kernel(
    f: wp.array3d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    e: wp.array2d(dtype=wp.float32),
    gx32: wp.float32,
    gy32: wp.float32,
    c9: wp.array(dtype=wp.float32),
    c3: wp.array(dtype=wp.float32),
    cw: wp.array(dtype=wp.float32),
) -> None:
    """Guo forcing source term, added to the post-collision distribution.

    :func:`lbm.boundary.apply_body_force`::

        S_i = (1 - 1/(2 tau)) w_i [ 3 (e_i.F - u.F) + 9 (e_i.u)(e_i.F) ]

    ``ug = 3 (u . F)`` is hoisted out of the direction loop and premultiplied by
    3, as NumPy does. The three per-direction scalars — ``9 (e_i.F)``,
    ``3 (e_i.F)`` and ``(1 - 1/(2 tau)) w_i`` — are computed **on the host** by
    :meth:`WarpBackend.apply_body_force` in NumPy's expression order and passed
    in as ``(9,)`` arrays, so no extra rounding enters.

    Args:
        f: post-collision distribution, ``(9, ny, nx)`` ``float32``, device,
            modified in place.
        u: force-corrected velocity, ``(2, ny, nx)`` ``float32``, device.
        e: ``E_F32`` from :mod:`lbm.core`, ``(9, 2)`` ``float32``, device.
        gx32: ``np.float32(gx)``.
        gy32: ``np.float32(gy)``.
        c9: ``np.float32(9 * e_i.F)`` per direction, ``(9,)`` ``float32``.
        c3: ``np.float32(3 * e_i.F)`` per direction, ``(9,)`` ``float32``.
        cw: ``np.float32((1 - 1/(2 tau)) * W[i])`` per direction, ``(9,)``.
    """
    y, x = wp.tid()

    ux = u[0, y, x]
    uy = u[1, y, x]

    # ug = 3 (u . F), hoisted out of the direction loop
    ug = ux * gx32
    tmp = uy * gy32
    ug += tmp
    ug *= 3.0

    for i in range(9):
        # eu = e_i . u
        eu = ux * e[i, 0]
        tmp = uy * e[i, 1]
        eu += tmp

        # tmp = 3 (e_i.F - u.F) + 9 (e_i.u)(e_i.F), then scaled by pref * w_i
        tmp = eu * c9[i]
        tmp += c3[i]
        tmp -= ug
        tmp *= cw[i]

        f[i, y, x] = f[i, y, x] + tmp


@wp.kernel
def _collide_bb_kernel(
    f: wp.array3d(dtype=wp.float32),
    feq: wp.array3d(dtype=wp.float32),
    f_pre: wp.array3d(dtype=wp.float32),
    solid: wp.array2d(dtype=wp.uint8),
    opp: wp.array(dtype=wp.int32),
    one_minus_omega: wp.float32,
    has_solid: wp.int32,
    s: wp.array3d(dtype=wp.float32),
) -> None:
    """Collide then bounce back, into ``s`` — the first half of the fused pass.

    The collide half is :func:`lbm.core.collide`'s three operations in its order
    and the reflection is :func:`lbm.boundary.bounce_back`'s assignment, which is
    exactly what :func:`lbm.core.collide_stream` does per direction (**D-033**).
    ``s`` is the **pre-stream** snapshot :func:`lbm.probe.forces` consumes
    (**D-020**) when the caller supplies ``f_bb``, and ``f`` itself when it does
    not.

    One thread per cell, looping the nine directions, so the mask is read once
    per cell.

    Args:
        f: pre-collision distribution, ``(9, ny, nx)`` ``float32``, device.
        feq: equilibrium, ``(9, ny, nx)`` ``float32``, device.
        f_pre: pre-collision copy for the reflection (**D-011**), device. Equal
            to ``f`` when the caller knows ``f`` is not written before it is
            read — see :meth:`WarpBackend.collide_stream`.
        solid: solid mask, ``(ny, nx)`` ``uint8``, nonzero is wall.
        opp: ``OPP`` from :mod:`lbm.core`, ``(9,)`` ``int32``, device.
        one_minus_omega: ``1 - 1/tau``, computed host-side in ``float32``.
        has_solid: zero skips the reflection entirely.
        s: output, ``(9, ny, nx)`` ``float32``, device. May be ``f``.
    """
    y, x = wp.tid()

    wall = int(0)
    if has_solid != 0:
        if solid[y, x] != wp.uint8(0):
            wall = 1

    for i in range(9):
        if wall != 0:
            # bounce back: solid cells emit what arrived at them, reversed.
            s[i, y, x] = f_pre[opp[i], y, x]
        else:
            # collide: s = feq[i] + (f[i] - feq[i]) (1 - omega)
            v = f[i, y, x]
            v -= feq[i, y, x]
            v *= one_minus_omega
            v += feq[i, y, x]
            s[i, y, x] = v


@wp.kernel
def _collide_bb_smag_kernel(
    f: wp.array3d(dtype=wp.float32),
    feq: wp.array3d(dtype=wp.float32),
    f_pre: wp.array3d(dtype=wp.float32),
    solid: wp.array2d(dtype=wp.uint8),
    opp: wp.array(dtype=wp.int32),
    e: wp.array2d(dtype=wp.float32),
    smag_coeff: wp.float32,
    tau32: wp.float32,
    tau_sq: wp.float32,
    has_solid: wp.int32,
    s: wp.array3d(dtype=wp.float32),
    scale: wp.array2d(dtype=wp.float32),
) -> None:
    """:func:`_collide_bb_kernel` with the closure, factor and all (T202).

    Two loops over the nine directions in one thread: the first is
    :func:`_smag_scale_kernel`'s reduction, the second is
    :func:`_collide_bb_kernel` unchanged but for reading ``1 - omega_eff`` from
    a register instead of a kernel argument. The reflection is untouched —
    bounce-back is an assignment and no closure reaches it (constraint 1's
    "bounce-back walls", unchanged).

    **Why the reduction is folded in here and not on the unfused path.** The
    step is memory-bound, and a separate scale kernel is a second full pass over
    ``f`` and ``feq`` — 18 planes of read traffic added to a kernel that already
    reads 18 and writes 9. Folded in, the second loop re-reads what the first
    loop just pulled into cache for the same cells. Measured on an RTX 3050 at
    2M cells (``bench.py --backend warp --les``): the closure cost **27.7%** of
    the BGK step rate as two kernels and **9.8%** as one. :meth:`WarpBackend.collide`
    keeps the two-kernel form because *its* threads are one per
    ``(direction, cell)``, so folding there would make every thread redo the
    whole nine-direction reduction.

    The arithmetic is unchanged either way — same operations, same order, same
    ``float32`` — and ``scale`` is still written so the buffer holds what NumPy's
    ``smag_out`` holds on return.

    Args:
        f: pre-collision distribution, ``(9, ny, nx)`` ``float32``, device.
        feq: equilibrium, ``(9, ny, nx)`` ``float32``, device.
        f_pre: pre-collision copy for the reflection (**D-011**), device.
        solid: solid mask, ``(ny, nx)`` ``uint8``, nonzero is wall.
        opp: ``OPP`` from :mod:`lbm.core`, ``(9,)`` ``int32``, device.
        e: ``E_F32`` from :mod:`lbm.core`, ``(9, 2)`` ``float32``, device.
        smag_coeff: ``float32(SMAG_Q_COEFF * cs_smag * cs_smag)``, host-side.
        tau32: ``float32(tau)``, host-side.
        tau_sq: ``float32(tau32 * tau32)``, host-side.
        has_solid: zero skips the reflection entirely.
        s: output, ``(9, ny, nx)`` ``float32``, device. May be ``f``.
        scale: output, ``(ny, nx)`` ``float32``, device. Receives
            ``1 - omega_eff``, as NumPy's ``smag_out`` does.
    """
    y, x = wp.tid()

    # --- the closure, exactly :func:`_smag_scale_kernel`'s arithmetic --------
    qxx = float(0.0)
    qxy = float(0.0)
    qyy = float(0.0)
    rho = float(0.0)

    for i in range(9):
        fi = f[i, y, x]
        rho += fi

        ex = e[i, 0]
        ey = e[i, 1]
        if ex != 0.0 or ey != 0.0:
            neq = fi - feq[i, y, x]
            if ex != 0.0:
                qxx += neq
            if ey != 0.0:
                qyy += neq
            if ex * ey > 0.0:
                qxy += neq
            elif ex * ey < 0.0:
                qxy -= neq

    qxx = qxx * qxx
    qxy = qxy * qxy
    qxy = qxy * 2.0
    qyy = qyy * qyy
    qxx = qxx + qxy
    qxx = qxx + qyy
    qmag = wp.sqrt(qxx)

    t = qmag / rho
    t = t * smag_coeff
    t = t + tau_sq
    t = wp.sqrt(t)
    t = t + tau32
    t = t * 0.5

    one_minus_omega = 1.0 - 1.0 / t
    scale[y, x] = one_minus_omega

    # --- the collision, exactly :func:`_collide_bb_kernel`'s ----------------
    wall = int(0)
    if has_solid != 0:
        if solid[y, x] != wp.uint8(0):
            wall = 1

    for i in range(9):
        if wall != 0:
            # bounce back: solid cells emit what arrived at them, reversed.
            s[i, y, x] = f_pre[opp[i], y, x]
        else:
            # collide: s = feq[i] + (f[i] - feq[i]) (1 - omega_eff)
            v = f[i, y, x]
            v -= feq[i, y, x]
            v *= one_minus_omega
            v += feq[i, y, x]
            s[i, y, x] = v


# --- the backend ------------------------------------------------------------

#: Host dtype -> Warp dtype. ``bool`` becomes ``uint8`` because Warp has no
#: packed boolean array; that mapping is the reason a ``uint8`` device array in
#: this module is always a mask (see the module docstring).
_WP_DTYPE: dict[Any, Any] = {
    np.dtype(np.float32): wp.float32,
    np.dtype(np.bool_): wp.uint8,
    np.dtype(np.uint8): wp.uint8,
    np.dtype(np.int32): wp.int32,
}


class WarpBackend:
    """:class:`lbm.backends.Backend` on a Warp device, state included.

    Implements ``DOCS/TASKS2.md`` § T102 (the four kernels) and § T103 (the
    boundaries, the fused pass, and the device-resident state that makes the
    performance budget reachable).

    Attributes:
        name: ``"warp"`` — the registry key.
        device: the Warp device the kernels run on and the state lives on.
    """

    name: str = "warp"

    def __init__(self, device: str | None = None, shape: Any = None) -> None:
        """Initialise Warp, upload the constants, compile the kernels.

        Args:
            device: a Warp device string, e.g. ``"cuda:0"`` or ``"cpu"``.
                ``None`` takes :func:`warp.get_preferred_device`, which is the
                first CUDA device when there is one and the CPU otherwise — so
                the parity rung runs on a machine without a GPU, at CPU speed,
                rather than not running at all.
            shape: accepted and ignored. T102 used it to preallocate device
                buffers keyed by grid shape (**D-052**); T103 hands allocation
                to the caller through :meth:`empty` / :meth:`zeros`, so there is
                nothing to size in advance. Kept so a T102-era call still works.

        Raises:
            RuntimeError: if Warp cannot initialise or has no usable device.
        """
        del shape  # see the docstring: allocation moved to the caller in T103
        wp.init()
        self.device = (
            wp.get_device(device) if device is not None else wp.get_preferred_device()
        )

        # The nine constants, uploaded once (constraint 4). They come from
        # lbm.core and are never redefined here; tests/test_backends.py scans
        # this module's AST to assert exactly that.
        self._e_i32 = wp.array(
            np.ascontiguousarray(E), dtype=wp.int32, device=self.device
        )
        self._e_f32 = wp.array(
            np.ascontiguousarray(E_F32), dtype=wp.float32, device=self.device
        )
        self._w = wp.array(
            np.ascontiguousarray(W), dtype=wp.float32, device=self.device
        )
        self._opp = wp.array(
            np.ascontiguousarray(OPP), dtype=wp.int32, device=self.device
        )
        self._cs2 = wp.array(
            np.array([CS2], dtype=np.float32), dtype=wp.float32, device=self.device
        )

        # Per-direction scalar coefficients the Ladd wall and the Guo source
        # term need. Nine floats each, allocated once here so that neither
        # boundary allocates inside the step loop (``CLAUDE.md`` conventions).
        self._coef_wall = wp.zeros(Q, dtype=wp.float32, device=self.device)
        self._coef_9 = wp.zeros(Q, dtype=wp.float32, device=self.device)
        self._coef_3 = wp.zeros(Q, dtype=wp.float32, device=self.device)
        self._coef_w = wp.zeros(Q, dtype=wp.float32, device=self.device)

        # Stand-in for an absent mask: Warp kernels take no ``None``, so an
        # unmasked call passes this 1x1 array and a zero flag.
        self._no_mask2d = wp.zeros((1, 1), dtype=wp.uint8, device=self.device)
        self._no_mask1d = wp.zeros(1, dtype=wp.uint8, device=self.device)

        # Compile now rather than inside the first timed loop.
        wp.load_module(module=__name__, device=self.device)

    # -- allocation and transfer ------------------------------------------

    def empty(self, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
        """See :meth:`lbm.backends.Backend.empty` — a **device** array.

        Warp has no uninitialised allocation, so this is :meth:`zeros`. The
        distinction only matters for a buffer read before it is written, which
        would be a bug on either backend.

        Args:
            shape: e.g. ``(9, ny, nx)``.
            dtype: ``float32`` or ``bool`` (stored as ``uint8``).

        Returns:
            A device :class:`warp.array`.
        """
        return self.zeros(shape, dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
        """See :meth:`lbm.backends.Backend.zeros` — a **device** array.

        Args:
            shape: e.g. ``(2, ny, nx)``.
            dtype: ``float32`` or ``bool`` (stored as ``uint8``).

        Returns:
            A zero-filled device :class:`warp.array`.

        Raises:
            ValueError: on a dtype this project does not use on a device.
        """
        wp_dtype = self._wp_dtype(dtype)
        dims = (int(shape),) if isinstance(shape, int) else tuple(int(n) for n in shape)
        return wp.zeros(dims, dtype=wp_dtype, device=self.device)

    def copy(self, dst: Any, src: Any) -> None:
        """See :meth:`lbm.backends.Backend.copy` — a device-to-device copy.

        Args:
            dst: destination device array.
            src: source device array of the same shape and dtype.
        """
        wp.copy(dst, src)

    def upload(self, host: NDArray[Any], dst: Any = None) -> Any:
        """See :meth:`lbm.backends.Backend.upload` — host to device.

        ``bool`` host arrays are converted to ``uint8`` on the way in; nothing
        else is converted, so a ``float32`` array crosses the bus unchanged and
        the round trip through :meth:`download` is bit-exact.

        Args:
            host: a NumPy array. Made C-contiguous if it is not.
            dst: an existing device array of the same shape, or ``None``.

        Returns:
            The device array holding ``host``'s data.
        """
        arr = np.ascontiguousarray(host)
        if arr.dtype == np.bool_:
            arr = arr.view(np.uint8)
        wp_dtype = self._wp_dtype(arr.dtype)
        if dst is None:
            return wp.array(arr, dtype=wp_dtype, device=self.device)
        wp.copy(dst, wp.array(arr, dtype=wp_dtype, copy=False, device="cpu"))
        return dst

    def download(self, src: Any, out: NDArray[Any] | None = None) -> NDArray[Any]:
        """See :meth:`lbm.backends.Backend.download` — device to host.

        **This is the only call that synchronises the device**, which is why
        :class:`lbm.runner.Sim` reaches it at frame and probe cadence and never
        inside :meth:`lbm.runner.Sim.step` (constraint 8).

        A ``uint8`` device array is a boolean mask (see the module docstring) and
        comes back as ``bool`` unless ``out`` says otherwise.

        Args:
            src: a device array.
            out: an existing host array of the same shape, or ``None``.

        Returns:
            Host NumPy holding ``src``'s data.
        """
        if out is None:
            host = src.numpy()
            if host.dtype == np.uint8:
                return host.view(np.bool_)
            return host
        view = out.view(np.uint8) if out.dtype == np.bool_ else out
        wp.copy(
            wp.array(
                np.ascontiguousarray(view),
                dtype=src.dtype,
                copy=False,
                device="cpu",
            ),
            src,
        )
        wp.synchronize_device(self.device)
        return out

    @staticmethod
    def _wp_dtype(dtype: Any) -> Any:
        """The Warp dtype for a host dtype.

        Args:
            dtype: a NumPy dtype or something :func:`numpy.dtype` accepts.

        Returns:
            The corresponding Warp scalar type.

        Raises:
            ValueError: on a dtype this project never puts on a device.
        """
        key = np.dtype(dtype)
        try:
            return _WP_DTYPE[key]
        except KeyError:
            raise ValueError(
                f"the Warp backend holds float32 and boolean arrays "
                f"(CLAUDE.md constraint 4); got {key}."
            ) from None

    def free_memory(self) -> int:
        """Bytes of device memory still free, or 0 where Warp cannot say.

        Used by ``bench.py`` to print the footprint at 2M cells against the
        card's 4 GB (``DOCS/TASKS2.md`` § T103), and by the tests that assert a
        step loop allocates nothing.

        Returns:
            Free device bytes; ``0`` on a CPU device, which has no such number.
        """
        return int(getattr(self.device, "free_memory", 0) or 0)

    def total_memory(self) -> int:
        """Bytes of device memory in total, or 0 where Warp cannot say.

        Returns:
            Total device bytes; ``0`` on a CPU device.
        """
        return int(getattr(self.device, "total_memory", 0) or 0)

    # -- kernels ----------------------------------------------------------

    def macroscopic(self, f: Any, rho: Any = None, u: Any = None) -> tuple[Any, Any]:
        """See :meth:`lbm.backends.Backend.macroscopic`.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array.
            rho: optional ``(ny, nx)`` ``float32`` device output.
            u: optional ``(2, ny, nx)`` ``float32`` device output.

        Returns:
            ``(rho, u)`` — ``(ny, nx)`` and ``(2, ny, nx)`` device arrays.
        """
        _, ny, nx = f.shape
        if rho is None:
            rho = self.zeros((ny, nx))
        if u is None:
            u = self.zeros((2, ny, nx))

        wp.launch(
            _macroscopic_kernel,
            dim=(ny, nx),
            inputs=[f, self._e_f32, rho, u],
            device=self.device,
        )
        return rho, u

    def equilibrium(
        self, rho: Any, u: Any, feq: Any = None, work: Any = None
    ) -> Any:
        """See :meth:`lbm.backends.Backend.equilibrium`.

        Args:
            rho: ``(ny, nx)`` ``float32`` device array.
            u: ``(2, ny, nx)`` ``float32`` device array.
            feq: optional ``(9, ny, nx)`` ``float32`` device output.
            work: ignored — a thread keeps core's three ``(ny, nx)``
                temporaries in registers. Accepted so that the signature
                matches the protocol term for term.

        Returns:
            ``feq``, ``(9, ny, nx)`` ``float32`` device array.
        """
        del work  # see the docstring: registers, not scratch arrays
        ny, nx = rho.shape
        if feq is None:
            feq = self.zeros((Q, ny, nx))

        wp.launch(
            _equilibrium_kernel,
            dim=(ny, nx),
            inputs=[rho, u, self._e_f32, self._w, feq],
            device=self.device,
        )
        return feq

    def collide(
        self,
        f: Any,
        feq: Any,
        tau: float,
        *,
        cs_smag: float = 0.0,
        smag_out: Any = None,
        smag_work: Any = None,
    ) -> None:
        """See :meth:`lbm.backends.Backend.collide`.

        The closure's **signature** landed here in T201 and its **kernels** in
        T202. ``cs_smag = 0.0`` — the default, and what every rung but Rung F
        runs — launches :func:`_collide_kernel`, the Phase 1 kernel with not one
        instruction changed, so it is bitwise its own previous self by
        construction rather than by tolerance (constraint 19; the answer this
        task records for **Q-201** is two compiled kernels, not one guarded
        branch).

        With the closure on, one extra launch precedes the collision:
        :func:`_smag_scale_kernel` reduces the second moment of ``f - feq`` into
        the ``(ny, nx)`` factor :func:`_collide_smag_kernel` then multiplies by.
        It reads the **pre-collision** ``f``, which is why it is a separate
        launch and not folded in — exactly as :func:`lbm.core.collide` computes
        the field before touching ``f``.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in place.
            feq: ``(9, ny, nx)`` ``float32`` device array.
            tau: relaxation time, greater than 0.5.
            cs_smag: Smagorinsky constant. ``0.0`` is plain BGK, bitwise.
            smag_out: optional preallocated ``(ny, nx)`` ``float32`` **device**
                array for the closure's per-cell factor. **Used as scratch**: on
                return it holds ``1 - omega_eff``, matching what NumPy leaves
                there. Ignored when ``cs_smag`` is 0; allocated here when the
                closure is on and the caller supplied none, which
                :class:`lbm.runner.Sim` never does — it preallocates.
            smag_work: accepted and unused. NumPy needs ``(4, ny, nx)`` of
                whole-array scratch for the reduction; a GPU thread has
                registers, so there is nothing to stage.

        Raises:
            ValueError: if ``tau <= 0.5`` — the check and the message are
                :func:`lbm.core.collide`'s (``CLAUDE.md`` constraint 2) — or if
                ``cs_smag < 0``.
        """
        del smag_work  # see the docstring: registers, not scratch
        _, ny, nx = f.shape

        if cs_smag == 0.0:
            # Phase 1's kernel, reached by a branch rather than by a term that
            # multiplies to zero. **D-053** records that the device contracts
            # ``x * a + b`` into one rounding where NumPy does two, so an
            # algebraically-zero closure term is not automatically bitwise
            # inert; a separate launch of the unmodified kernel is.
            one_minus_omega = self._one_minus_omega(tau)
            wp.launch(
                _collide_kernel,
                dim=(Q, ny, nx),
                inputs=[f, feq, one_minus_omega],
                device=self.device,
            )
            return

        smag_coeff, tau32, tau_sq = self._smag_scalars(tau, cs_smag)
        scale = self.empty((ny, nx)) if smag_out is None else smag_out
        wp.launch(
            _smag_scale_kernel,
            dim=(ny, nx),
            inputs=[f, feq, self._e_f32, smag_coeff, tau32, tau_sq, scale],
            device=self.device,
        )
        wp.launch(
            _collide_smag_kernel,
            dim=(Q, ny, nx),
            inputs=[f, feq, scale],
            device=self.device,
        )

    def stream(self, f: Any, buf: Any) -> Any:
        """See :meth:`lbm.backends.Backend.stream`.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in place.
            buf: ``(9, ny, nx)`` ``float32`` device scratch. Left holding the
                streamed state, as :func:`lbm.core.stream` leaves it.

        Returns:
            ``f`` — the same object passed in, so its identity survives the
            call (T006's restart test depends on that).
        """
        _, ny, nx = f.shape
        wp.launch(
            _stream_kernel,
            dim=(Q, ny, nx),
            inputs=[f, self._e_i32, buf],
            device=self.device,
        )
        wp.copy(f, buf)
        return f

    def collide_stream(
        self,
        f: Any,
        feq: Any,
        tau: float,
        buf: Any,
        *,
        f_pre: Any = None,
        solid: Any = None,
        f_bb: Any = None,
        cs_smag: float = 0.0,
        smag_out: Any = None,
        smag_work: Any = None,
    ) -> Any:
        """See :meth:`lbm.backends.Backend.collide_stream` (**D-033**).

        Two launches, not one: the collide-and-reflect pass writes the pre-stream
        snapshot, and the stream pass gathers from it. That is the same
        arithmetic in the same order as :func:`lbm.core.collide_stream`, so
        fused and unfused agree **bitwise on this backend** — which is what
        ``tests/test_warp_backend.py`` asserts and what keeps constraint 11's
        bit-identical restart true across the switch. The "one pass per
        direction" of **D-033** was a cache argument for a CPU; on a device the
        equivalent saving is that ``f_bb`` is streamed straight into ``f`` and
        ``buf`` is never touched, which removes a full ``(9, ny, nx)``
        device-to-device copy per step.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in place.
            feq: ``(9, ny, nx)`` ``float32`` device array.
            tau: relaxation time, greater than 0.5.
            buf: ``(9, ny, nx)`` ``float32`` device scratch. Used only when
                ``f_bb`` is ``None``, because then the snapshot stages in ``f``
                and streaming needs somewhere else to land.
            f_pre: pre-collision copy (**D-011**), device. Required when
                ``solid`` is given.
            solid: ``(ny, nx)`` mask (``bool`` on the host, ``uint8`` on the
                device), or ``None`` to skip the reflection.
            f_bb: ``(9, ny, nx)`` ``float32`` device array receiving the
                pre-stream snapshot (**D-020**). ``None`` stages in ``f``.

        Returns:
            ``f`` — the same object passed in.

        Raises:
            ValueError: if ``tau <= 0.5``, or ``solid`` is given without
                ``f_pre``.
        """
        if solid is not None and f_pre is None:
            raise ValueError(
                "collide_stream needs f_pre (the pre-collision copy, D-011) to "
                "bounce back off solid: f_pre[OPP[i]] is the reflection."
            )
        del smag_work  # registers, not scratch — see :meth:`collide`

        _, ny, nx = f.shape
        s = f if f_bb is None else f_bb
        has_solid = 0 if solid is None else 1

        if cs_smag == 0.0:
            # Constraint 19 on the fused path: Phase 1's kernel, launched
            # unchanged. See :meth:`collide` for why this is a branch and not a
            # zero-valued term.
            one_minus_omega = self._one_minus_omega(tau)
            wp.launch(
                _collide_bb_kernel,
                dim=(ny, nx),
                inputs=[
                    f,
                    feq,
                    f if f_pre is None else f_pre,
                    self._no_mask2d if solid is None else solid,
                    self._opp,
                    one_minus_omega,
                    has_solid,
                    s,
                ],
                device=self.device,
            )
        else:
            # One launch, and it has to be: the factor is computed from the
            # **pre-collision** ``f``, and this kernel writes ``s``, which may
            # *be* ``f``. Each thread therefore reduces its own cell's nine
            # directions before it writes any of them — the per-thread form of
            # "there is exactly one moment at which the whole pre-collision
            # state exists", which is what :func:`lbm.core.collide_stream`
            # relies on when it computes the field before the direction loop.
            smag_coeff, tau32, tau_sq = self._smag_scalars(tau, cs_smag)
            scale = self.empty((ny, nx)) if smag_out is None else smag_out
            wp.launch(
                _collide_bb_smag_kernel,
                dim=(ny, nx),
                inputs=[
                    f,
                    feq,
                    f if f_pre is None else f_pre,
                    self._no_mask2d if solid is None else solid,
                    self._opp,
                    self._e_f32,
                    smag_coeff,
                    tau32,
                    tau_sq,
                    has_solid,
                    s,
                    scale,
                ],
                device=self.device,
            )

        if s is f:
            # The snapshot staged in ``f``; the gather cannot read and write the
            # same array, so it lands in ``buf`` and is copied back.
            wp.launch(
                _stream_kernel,
                dim=(Q, ny, nx),
                inputs=[f, self._e_i32, buf],
                device=self.device,
            )
            wp.copy(f, buf)
        else:
            wp.launch(
                _stream_kernel,
                dim=(Q, ny, nx),
                inputs=[s, self._e_i32, f],
                device=self.device,
            )
        return f

    # -- boundaries -------------------------------------------------------

    def bounce_back(self, f: Any, f_pre: Any, solid: Any) -> None:
        """See :meth:`lbm.backends.Backend.bounce_back`.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in place on
                solid cells.
            f_pre: pre-collision copy (**D-011**), device.
            solid: ``(ny, nx)`` device mask.
        """
        _, ny, nx = f.shape
        wp.launch(
            _bounce_back_kernel,
            dim=(ny, nx),
            inputs=[f, f_pre, solid, self._opp],
            device=self.device,
        )

    def moving_wall(
        self,
        f: Any,
        f_pre: Any,
        wall: Any,
        u_wall: tuple[float, float],
        rho_w: float = 1.0,
    ) -> None:
        """See :meth:`lbm.backends.Backend.moving_wall`.

        The nine ``6 w_i rho_w (e_i . u_wall)`` scalars are built here, in
        :func:`lbm.boundary.moving_wall`'s own ``float64``-then-round-once
        order, and uploaded into the ``(9,)`` device array allocated at
        construction — so this call allocates nothing on the device.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in place on
                ``wall`` cells.
            f_pre: pre-collision copy (**D-011**), device.
            wall: ``(ny, nx)`` device mask of the moving cells.
            u_wall: ``(ux, uy)`` lattice velocity of the wall.
            rho_w: wall density used in the correction.
        """
        uwx, uwy = float(u_wall[0]), float(u_wall[1])
        coef = np.zeros(Q, dtype=np.float32)
        for i in range(Q):
            eu = float(E_F32[i, 0]) * uwx + float(E_F32[i, 1]) * uwy
            if eu != 0.0:
                # 2 / cs2 == 6; the same expression lbm.boundary.moving_wall
                # evaluates, rounded to float32 exactly once.
                coef[i] = np.float32(6.0 * float(W[i]) * rho_w * eu)
        self.upload(coef, dst=self._coef_wall)

        _, ny, nx = f.shape
        wp.launch(
            _moving_wall_kernel,
            dim=(ny, nx),
            inputs=[f, f_pre, wall, self._opp, self._coef_wall],
            device=self.device,
        )

    def inlet_velocity(
        self,
        f: Any,
        *,
        col: int = 0,
        u_in: Any,
        work: Any = None,
        fluid: Any = None,
    ) -> None:
        """See :meth:`lbm.backends.Backend.inlet_velocity`.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in column
                ``col``.
            col: inlet column; a negative index is resolved here.
            u_in: ``(2, ny)`` ``float32`` device array.
            work: ignored — a thread holds NumPy's five ``(ny,)`` temporaries in
                registers. Accepted so the signature matches the protocol.
            fluid: ``(ny,)`` device mask of fluid rows, or ``None``.
        """
        del work  # registers, not scratch arrays
        _, ny, nx = f.shape
        wp.launch(
            _inlet_kernel,
            dim=ny,
            inputs=[
                f,
                u_in,
                self._no_mask1d if fluid is None else fluid,
                int(col % nx),
                0 if fluid is None else 1,
            ],
            device=self.device,
        )

    def outlet_zero_gradient(
        self,
        f: Any,
        *,
        col: int = -1,
        src: int = -2,
        prev: Any = None,
        lam: float | None = None,
    ) -> None:
        """See :meth:`lbm.backends.Backend.outlet_zero_gradient` (**D-021**).

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in column
                ``col``.
            col: outlet column; a negative index is resolved here.
            src: source column; a negative index is resolved here.
            prev: ``(9, ny)`` ``float32`` device array holding the previous
                outlet column, updated in place. ``None`` selects the copy.
            lam: advection speed. ``None`` means ``sqrt(CS2)``.
        """
        _, ny, nx = f.shape
        col_i = int(col % nx)
        src_i = int(src % nx)

        if prev is None:
            wp.launch(
                _outlet_copy_kernel,
                dim=(Q, ny),
                inputs=[f, col_i, src_i],
                device=self.device,
            )
            return

        if lam is None:
            lam = float(np.sqrt(CS2))
        wp.launch(
            _outlet_convective_kernel,
            dim=(Q, ny),
            inputs=[
                f,
                prev,
                col_i,
                src_i,
                float(np.float32(lam)),
                float(np.float32(1.0 / (1.0 + lam))),
            ],
            device=self.device,
        )

    # -- the Guo body force, both halves ----------------------------------

    def force_velocity_shift(
        self, rho: Any, u: Any, g: tuple[float, float], work: Any = None
    ) -> Any:
        """See :meth:`lbm.backends.Backend.force_velocity_shift`.

        Args:
            rho: ``(ny, nx)`` ``float32`` device array.
            u: ``(2, ny, nx)`` ``float32`` device array, modified in place.
            g: ``(gx, gy)`` lattice body force per unit volume.
            work: ignored — the reciprocal lives in a register.

        Returns:
            ``u`` — the same object passed in.
        """
        del work  # registers, not scratch arrays
        half_gx = 0.5 * float(g[0])
        half_gy = 0.5 * float(g[1])
        ny, nx = rho.shape
        wp.launch(
            _force_shift_kernel,
            dim=(ny, nx),
            inputs=[
                rho,
                u,
                float(np.float32(half_gx)),
                float(np.float32(half_gy)),
                0 if half_gx == 0.0 else 1,
                0 if half_gy == 0.0 else 1,
            ],
            device=self.device,
        )
        return u

    def apply_body_force(
        self,
        f: Any,
        rho: Any,
        u: Any,
        tau: float,
        g: tuple[float, float],
        work: Any = None,
    ) -> None:
        """See :meth:`lbm.backends.Backend.apply_body_force`.

        The three per-direction scalars are built here in
        :func:`lbm.boundary.apply_body_force`'s expression order and uploaded
        into the ``(9,)`` device arrays allocated at construction, so this call
        allocates nothing on the device.

        Args:
            f: ``(9, ny, nx)`` ``float32`` device array, modified in place.
            rho: ``(ny, nx)`` ``float32`` device array. Unused by the formula;
                in the signature for API stability, as in Phase 0.
            u: ``(2, ny, nx)`` ``float32`` force-corrected velocity, device.
            tau: relaxation time, greater than 0.5.
            g: ``(gx, gy)`` lattice body force per unit volume.
            work: ignored — a thread keeps the temporaries in registers.
        """
        del rho, work  # see the docstring
        gx, gy = float(g[0]), float(g[1])
        pref = 1.0 - 0.5 / tau

        c9 = np.zeros(Q, dtype=np.float32)
        c3 = np.zeros(Q, dtype=np.float32)
        cw = np.zeros(Q, dtype=np.float32)
        for i in range(Q):
            eg = float(E_F32[i, 0]) * gx + float(E_F32[i, 1]) * gy
            c9[i] = np.float32(9.0 * eg)
            c3[i] = np.float32(3.0 * eg)
            cw[i] = np.float32(pref * float(W[i]))
        self.upload(c9, dst=self._coef_9)
        self.upload(c3, dst=self._coef_3)
        self.upload(cw, dst=self._coef_w)

        _, ny, nx = f.shape
        wp.launch(
            _body_force_kernel,
            dim=(ny, nx),
            inputs=[
                f,
                u,
                self._e_f32,
                float(np.float32(gx)),
                float(np.float32(gy)),
                self._coef_9,
                self._coef_3,
                self._coef_w,
            ],
            device=self.device,
        )

    # -- the portability contract -----------------------------------------

    def to_host(self, f: Any) -> NDArray[np.float32]:
        """Backend array -> host ``(9, ny, nx)`` ``float32`` (constraint 4).

        The portability contract, and **the only path a checkpoint takes**
        (**D-050**): a Warp checkpoint is readable on NumPy because ``f`` goes
        out through here and nowhere else. Accepts a host array too, so a caller
        that already has one is not punished for it.

        Args:
            f: a device ``(9, ny, nx)`` ``float32`` Warp array, or the
                equivalent host array.

        Returns:
            ``(9, ny, nx)`` ``float32`` in host memory. A fresh array, not a
            view of device memory.

        Raises:
            ValueError: if the shape or dtype is not the host contract.
        """
        host = f.numpy() if isinstance(f, wp.array) else f
        self._check_host(host)
        return host

    def from_host(self, arr: NDArray[np.float32]) -> Any:
        """Host ``(9, ny, nx)`` ``float32`` -> a new device array.

        The inverse of :meth:`to_host` and bit-exact with it: no arithmetic
        happens on either side, so ``to_host(from_host(a))`` equals ``a`` under
        :func:`numpy.array_equal`.

        Args:
            arr: ``(9, ny, nx)`` ``float32`` in host memory.

        Returns:
            A device :class:`warp.array` holding the same bits.

        Raises:
            ValueError: if the shape or dtype is not the host contract.
        """
        self._check_host(arr)
        return self.upload(arr)

    @staticmethod
    def _one_minus_omega(tau: float) -> float:
        """``1 - 1/tau`` in ``float32``, with core's ``tau`` check and message.

        Computed on the host so that the kernel never re-derives it and never
        touches ``nu`` (``CLAUDE.md`` constraint 2).

        Args:
            tau: BGK relaxation time.

        Returns:
            ``float(np.float32(1 - 1/tau))``.

        Raises:
            ValueError: if ``tau <= 0.5``.
        """
        if tau <= 0.5:
            raise ValueError(
                f"tau must be greater than 0.5 (got tau={tau!r}): "
                "collision with tau <= 0.5 gives non-positive viscosity and diverges."
            )
        return float(np.float32(1.0 - 1.0 / tau))

    @staticmethod
    def _smag_scalars(tau: float, cs_smag: float) -> tuple[float, float, float]:
        """The closure's three host-side scalars, in NumPy's expression order.

        **D-057**, applied to T202: a ``float64`` quantity that NumPy rounds to
        ``float32`` **once** is computed on the host and uploaded, never
        re-derived per thread. :func:`lbm.core.smagorinsky_tau_eff` folds
        exactly these three, and the expressions below are transcribed from it
        character for character —
        ``np.float32(SMAG_Q_COEFF * cs_smag * cs_smag)``, ``np.float32(tau)``
        and ``np.float32(tau32 * tau32)`` — because a different association
        order would round differently. :data:`lbm.core.SMAG_Q_COEFF` is kept in
        ``float64`` for this reason and is imported, never restated
        (constraint 4).

        ``tau32 * tau32`` in particular is why ``tau_eff -> tau`` exactly in the
        ``cs_smag -> 0`` limit: IEEE ``sqrt(x * x) == |x|``, which a ``float64``
        ``tau**2`` rounded afterwards would not guarantee (**D-085**).

        Args:
            tau: BGK relaxation time, greater than 0.5.
            cs_smag: the Smagorinsky constant, non-negative.

        Returns:
            ``(smag_coeff, tau32, tau_sq)`` as Python floats holding
            ``float32`` values, ready to pass to :func:`_smag_scale_kernel`.

        Raises:
            ValueError: if ``tau <= 0.5`` or ``cs_smag < 0`` — the checks and
                the messages are :func:`lbm.core.smagorinsky_tau_eff`'s.
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
        tau32 = np.float32(tau)
        return (
            float(np.float32(SMAG_Q_COEFF * cs_smag * cs_smag)),
            float(tau32),
            float(np.float32(tau32 * tau32)),
        )

    @staticmethod
    def _check_host(arr: NDArray[np.float32]) -> None:
        """Reject anything that is not ``(9, ny, nx)`` ``float32``.

        Args:
            arr: the candidate host array.

        Raises:
            ValueError: on a wrong rank, a wrong leading dimension, or a dtype
                that is not ``float32``.
        """
        if arr.ndim != 3 or arr.shape[0] != Q:
            raise ValueError(
                f"host distributions are (9, ny, nx) (CLAUDE.md constraint 4, "
                f"DOCS/STATE2.md D-046); got shape {arr.shape}."
            )
        if arr.dtype != np.float32:
            raise ValueError(
                f"host distributions are float32 (CLAUDE.md constraint 4); "
                f"got {arr.dtype}."
            )

    def __repr__(self) -> str:
        return f"WarpBackend(name={self.name!r}, device={str(self.device)!r})"
