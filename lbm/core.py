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


def collide(f: NDArray[np.float32], feq: NDArray[np.float32], tau: float) -> None:
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

    Args:
        f: distribution function, shape ``(9, ny, nx)``, ``float32``. Modified
            in place; its buffer identity never changes.
        feq: equilibrium distribution, same shape and dtype.
        tau: BGK relaxation time, greater than 0.5.

    Raises:
        ValueError: if ``tau <= 0.5`` (via :func:`nu_from_tau`'s condition).
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


def collide_stream(
    f: NDArray[np.float32],
    feq: NDArray[np.float32],
    tau: float,
    buf: NDArray[np.float32],
    *,
    f_pre: NDArray[np.float32] | None = None,
    solid: NDArray[np.bool_] | None = None,
    f_bb: NDArray[np.float32] | None = None,
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
    :func:`lbm.probe.forces` consumes (``DOCS/STATE1.md`` **D-020**), and ``f[i]``
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

    Returns:
        ``f``, collided, reflected and streamed — the same object passed in.

    Raises:
        ValueError: if ``tau <= 0.5``, or ``solid`` is given without ``f_pre``.
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
        # Where the post-collision, post-reflection state for direction i goes.
        # It is the pre-stream snapshot probe.forces consumes (D-020).
        s = f[i] if f_bb is None else f_bb[i]

        # collide: s = feq[i] + (f[i] - feq[i]) (1 - omega), the same three
        # operations in the same order as `collide`, on one direction.
        np.subtract(f[i], feq[i], out=s)
        s *= one_minus_omega
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
