"""The backend seam: one protocol, one implementation per compute target.

Implements ``DOCS/IDEA3.md`` § What Phase 1 is, concretely — the ``lbm/`` box in
the pipeline diagram, which "gains a backend seam and a Warp implementation and
**no new physics**".

Why a seam at all (``DOCS/PLAN2.md`` § Why this order): T101 introduces it with
only the NumPy backend behind it, so the session that introduces Warp introduces
*only* Warp. A seam invented during a port is a seam shaped by one
implementation.

What a backend owns and what it does not
----------------------------------------
A backend owns **the kernels and its own state layout** (``DOCS/STATE2.md``
**D-046**, constraint 4 in its Phase 1 form). It does not own the timestep: the
D-011/D-020 order lives in :meth:`lbm.runner.Sim.step` and is identical on every
backend, because the order is physics and the kernels are arithmetic.

The two host methods are the portability contract. Whatever a backend keeps
internally, :meth:`Backend.to_host` must hand back ``(9, ny, nx)`` ``float32``
in ``(direction, y, x)`` order, and :meth:`Backend.from_host` must accept the
same. That is what makes a checkpoint written on one backend loadable on another
(**D-050**) and what makes cross-backend agreement *measurable* rather than
merely asserted — the hook Rung A (``validate/parity.py``, T102) hangs off, and
the shape of the answer to **Q-103**.

The nine D2Q9 constants (``E``, ``W``, ``OPP``, ``CS2``) come from
:mod:`lbm.core` in **every** backend — uploaded to a device, never redefined
(constraint 4, "no physics constant twice").

What T103 added, and why the seam had to widen
---------------------------------------------
T101 covered kernels and the two host transfers and **nothing else** (**D-051**):
buffer allocation, the open boundaries, the Guo body force and the probes stayed
outside, because two implementations is the number that reveals the right seam
and one plus a guess is not. T102 then paid for that with **D-052** — a Warp
backend that took *host* arrays and copied in and out per call, and was
therefore slower than NumPy.

T103 widens it, in the shape whole-step parity forced:

* **Allocation and transfer** — :meth:`Backend.empty`, :meth:`Backend.zeros`,
  :meth:`Backend.copy`, :meth:`Backend.upload`, :meth:`Backend.download`. Every
  array :class:`lbm.runner.Sim` owns is allocated *by the backend*, so on a
  device backend the state lives on the device and a timestep moves no bytes
  across the bus at all. That is the one change without which nothing in
  ``DOCS/IDEA3.md`` § Performance budget is reachable.
* **The boundaries** — :meth:`Backend.moving_wall`,
  :meth:`Backend.inlet_velocity`, :meth:`Backend.outlet_zero_gradient`, beside
  T101's :meth:`Backend.bounce_back`. All four of Phase 0's, so the whole
  timestep runs where the state is.
* **Both halves of the Guo body force** — :meth:`Backend.force_velocity_shift`
  and :meth:`Backend.apply_body_force`. They go together or not at all
  (**D-010**), and Rung 1 is the case that needs them.

**Backend arrays are opaque handles.** Everything above takes and returns
whatever the backend allocated; only :meth:`Backend.download` /
:meth:`Backend.to_host` produce host NumPy. On :class:`~lbm.backends.numpy_backend.NumpyBackend`
the handle *is* an :class:`numpy.ndarray` and every transfer is the identity, so
the reference path is untouched by the widening — which is what keeps the four
Phase 0 rungs printing session-11's digits.

Still not in this protocol: the probes (:mod:`lbm.probe`), which read host
arrays at *frame* cadence and never at step cadence (constraint 8), and any
force argument on ``collide_stream`` — the Guo source term goes between
collision and bounce-back, so a forced run takes the unfused path (**D-033**).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Backend",
    "BackendUnavailableError",
    "available_backends",
    "get_backend",
]


@runtime_checkable
class Backend(Protocol):
    """The kernels :class:`lbm.runner.Sim` reaches every timestep.

    Every method has the signature of its :mod:`lbm.core` / :mod:`lbm.boundary`
    counterpart, term for term, so that the NumPy backend is a delegation and
    not a translation. Array shapes below are the host shapes; a device backend
    may hold the same data in any layout it likes as long as
    :meth:`to_host` / :meth:`from_host` convert.

    Attributes:
        name: the registry key this backend answers to, e.g. ``"numpy"``.
    """

    name: str

    def macroscopic(
        self,
        f: NDArray[np.float32],
        rho: NDArray[np.float32] | None = None,
        u: NDArray[np.float32] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Density and velocity from ``f``.

        Args:
            f: distribution, ``(9, ny, nx)`` ``float32``.
            rho: optional preallocated output, ``(ny, nx)`` ``float32``.
            u: optional preallocated output, ``(2, ny, nx)`` ``float32``,
                component 0 ``ux``, component 1 ``uy``.

        Returns:
            ``(rho, u)`` — ``(ny, nx)`` and ``(2, ny, nx)``, both ``float32``.
        """
        ...

    def equilibrium(
        self,
        rho: NDArray[np.float32],
        u: NDArray[np.float32],
        feq: NDArray[np.float32] | None = None,
        work: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Second-order equilibrium distribution.

        Args:
            rho: density, ``(ny, nx)`` ``float32``.
            u: velocity, ``(2, ny, nx)`` ``float32``.
            feq: optional preallocated output, ``(9, ny, nx)`` ``float32``.
            work: optional preallocated scratch, ``(3, ny, nx)`` ``float32``.

        Returns:
            ``feq``, ``(9, ny, nx)`` ``float32``.
        """
        ...

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
        """BGK collision, in place, optionally with the Smagorinsky closure.

        Args:
            f: distribution, ``(9, ny, nx)`` ``float32``, modified in place.
            feq: equilibrium, ``(9, ny, nx)`` ``float32``.
            tau: relaxation time, greater than 0.5 (constraint 2).
            cs_smag: Smagorinsky constant (T201, **D-081**). ``0.0`` — the
                default — is plain BGK and must be **bitwise** what the backend
                produced before the closure existed (constraint 19); a backend
                takes an explicit branch for it rather than multiplying in a
                zero-valued term. On the GPU that branch is not optional:
                **D-053** documents that ``x * a + b`` contracts into one
                rounding there where NumPy does two, so an algebraically-zero
                term is not automatically bitwise inert (**Q-201**).
            smag_out: optional preallocated ``(ny, nx)`` ``float32`` for the
                closure's per-cell factor, used as scratch. Ignored when
                ``cs_smag`` is 0.
            smag_work: optional preallocated ``(4, ny, nx)`` ``float32`` scratch.
                Ignored when ``cs_smag`` is 0, and a backend whose threads have
                registers may ignore it entirely — the Warp one does (T202).
        """
        ...

    def bounce_back(
        self,
        f: NDArray[np.float32],
        f_pre: NDArray[np.float32],
        solid: NDArray[np.bool_],
    ) -> None:
        """Half-way bounce-back on solid cells, in place.

        Args:
            f: post-collision distribution, ``(9, ny, nx)`` ``float32``,
                modified in place on solid cells only.
            f_pre: **pre-collision** copy of ``f`` (**D-011**), same shape and
                dtype.
            solid: solid mask, ``(ny, nx)`` ``bool``, ``True`` is wall.
        """
        ...

    def stream(
        self, f: NDArray[np.float32], buf: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Advection, in place, periodic in both directions.

        Args:
            f: distribution, ``(9, ny, nx)`` ``float32``, modified in place.
            buf: preallocated scratch, same shape and dtype.

        Returns:
            ``f``, streamed — the same object passed in.
        """
        ...

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
        """Collide, bounce back, snapshot and stream in one pass per direction.

        The fusion crosses ``bounce_back`` because the D-020 order puts the
        reflection *between* collide and stream (**D-033**). Both the fused and
        the unfused route through this protocol stay selectable, and on a given
        backend they must agree bitwise.

        Args:
            f: distribution, ``(9, ny, nx)`` ``float32``, modified in place.
            feq: equilibrium, same shape and dtype.
            tau: relaxation time, greater than 0.5.
            buf: preallocated scratch, same shape and dtype.
            f_pre: pre-**collision** copy of ``f`` (**D-011**), same shape and
                dtype. Required when ``solid`` is given.
            solid: solid mask, ``(ny, nx)`` ``bool``. ``None`` skips the
                reflection.
            f_bb: preallocated ``(9, ny, nx)`` ``float32`` receiving the
                **pre-stream** state :func:`lbm.probe.forces` consumes
                (**D-020**). ``None`` stages in ``f``.
            cs_smag: Smagorinsky constant (T201, **D-081**). ``0.0`` — the
                default — is plain BGK and must be **bitwise** what the backend
                produced before the closure existed (constraint 19); a backend
                takes an explicit branch for it rather than multiplying in a
                zero-valued term. On the GPU that branch is not optional:
                **D-053** documents that ``x * a + b`` contracts into one
                rounding there where NumPy does two, so an algebraically-zero
                term is not automatically bitwise inert (**Q-201**).
            smag_out: optional preallocated ``(ny, nx)`` ``float32`` for the
                closure's per-cell factor, used as scratch. Ignored when
                ``cs_smag`` is 0.
            smag_work: optional preallocated ``(4, ny, nx)`` ``float32`` scratch.
                Ignored when ``cs_smag`` is 0, and a backend whose threads have
                registers may ignore it entirely — the Warp one does (T202).

        Returns:
            ``f``, collided, reflected and streamed — the object passed in.
        """
        ...

    # -- allocation and transfer (T103) -----------------------------------

    def empty(self, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
        """An uninitialised backend array of ``shape``.

        The allocation half of the seam (**D-051**, widened by T103). Every
        buffer :class:`lbm.runner.Sim` owns comes from here, which is what lets
        a device backend keep the whole state — ``f`` ``(9, ny, nx)``, ``feq``,
        ``f_bb``, ``buf``, ``rho`` ``(ny, nx)``, ``u`` ``(2, ny, nx)`` — in
        device memory and move nothing across the bus during a timestep.

        Args:
            shape: the array shape, e.g. ``(9, ny, nx)``.
            dtype: NumPy dtype. ``float32`` (constraint 4) or ``bool`` for a
                mask; nothing else is used by this project.

        Returns:
            A backend-owned array. Opaque: pass it back to the kernels, or take
            it through :meth:`download` to read it on the host.
        """
        ...

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
        """A zero-filled backend array of ``shape``.

        Same contract as :meth:`empty` — ``(9, ny, nx)`` and friends — with the
        contents defined.

        Args:
            shape: the array shape.
            dtype: NumPy dtype.

        Returns:
            A backend-owned array of zeros.
        """
        ...

    def copy(self, dst: Any, src: Any) -> None:
        """``dst[:] = src``, both backend arrays of the same shape and dtype.

        The seam's :func:`numpy.copyto`. Used for the pre-collision copy
        (**D-011**) and the pre-stream ``f_bb`` snapshot (**D-020**), both
        ``(9, ny, nx)`` ``float32``.

        Args:
            dst: destination backend array.
            src: source backend array, same shape and dtype.
        """
        ...

    def upload(self, host: NDArray[Any], dst: Any = None) -> Any:
        """Host NumPy -> backend array, of **any** shape.

        The general transfer. :meth:`from_host` is this plus constraint 4's
        ``(9, ny, nx)`` ``float32`` check, and is the only one a checkpoint
        uses. This one also carries masks ``(ny, nx)`` ``bool``, the inlet
        profile ``(2, ny)`` and the outlet's previous column ``(9, ny)``.

        Args:
            host: a C-contiguous NumPy array.
            dst: an existing backend array of the same shape to write into.
                ``None`` allocates a new one.

        Returns:
            The backend array holding ``host``'s data — ``dst`` when given.
        """
        ...

    def download(self, src: Any, out: NDArray[Any] | None = None) -> NDArray[Any]:
        """Backend array -> host NumPy, of **any** shape.

        The general inverse of :meth:`upload`; :meth:`to_host` is this plus
        constraint 4's ``(9, ny, nx)`` ``float32`` check. Called at *frame* and
        *probe* cadence and never at step cadence (constraint 8) — the sim must
        never block on the display, and on a device backend this is the one call
        that synchronises.

        Args:
            src: a backend array.
            out: an existing host array of the same shape to read into.
                ``None`` allocates one.

        Returns:
            Host NumPy holding ``src``'s data. On a host backend this may be
            ``src`` itself, so treat it as read-only.
        """
        ...

    # -- boundaries (T103) -------------------------------------------------

    def moving_wall(
        self,
        f: Any,
        f_pre: Any,
        wall: Any,
        u_wall: tuple[float, float],
        rho_w: float = 1.0,
    ) -> None:
        """Momentum-corrected (Ladd) bounce-back on a moving wall, in place.

        ``f[i] = f_pre[OPP[i]] + 6 w_i rho_w (e_i . u_wall)`` on ``wall`` cells.
        Rung 2's lid. With ``u_wall = (0, 0)`` it degenerates exactly to
        :meth:`bounce_back`.

        Args:
            f: post-collision distribution, ``(9, ny, nx)`` ``float32``,
                modified in place on ``wall`` cells only.
            f_pre: pre-**collision** copy (**D-011**), same shape and dtype.
            wall: mask of the moving solid cells, ``(ny, nx)`` ``bool``.
            u_wall: wall velocity ``(ux, uy)``, lattice units, under 0.1.
            rho_w: wall density used in the correction.
        """
        ...

    def inlet_velocity(
        self,
        f: Any,
        *,
        col: int = 0,
        u_in: Any,
        work: Any = None,
        fluid: Any = None,
    ) -> None:
        """Zou-He velocity inlet on a left-facing column, in place.

        Runs **after** :meth:`stream` (**D-020**): streaming is periodic in
        ``x``, so the inlet column's three ``ex = +1`` populations
        (``i = 1, 5, 8``) hold wrap-around garbage and are exactly the unknowns
        this overwrites.

        Args:
            f: distribution, ``(9, ny, nx)`` ``float32``, modified in place in
                column ``col`` and directions 1, 5 and 8 only.
            col: index of the inlet column.
            u_in: the prescribed profile, ``(2, ny)`` ``float32``, built once by
                :func:`lbm.boundary.inlet_profile`.
            work: optional scratch, ``(>=5, ny)`` ``float32``.
            fluid: precomputed ``~solid[:, col]``, ``(ny,)`` ``bool``. ``None``
                writes every row.
        """
        ...

    def outlet_zero_gradient(
        self,
        f: Any,
        *,
        col: int = -1,
        src: int = -2,
        prev: Any = None,
        lam: float | None = None,
    ) -> None:
        """Zero-gradient outflow, in place - plain copy or convective (**D-021**).

        ``prev=None`` gives ``f[:, :, col] = f[:, :, src]`` over all nine
        directions of a ``(9, ny, nx)`` ``float32`` distribution; otherwise
        ``f[:, :, col] = (prev + lam f[:, :, src]) / (1 + lam)``, with ``lam``
        defaulting to ``sqrt(CS2)`` — 0.6% reflection against the copy's 35%.

        Args:
            f: distribution, ``(9, ny, nx)`` ``float32``, modified in place in
                column ``col`` only.
            col: outlet column.
            src: column read from.
            prev: the outlet column at the previous step, ``(9, ny)``
                ``float32``, **updated in place**. ``None`` selects the copy.
            lam: advection speed, lattice units. ``None`` means ``sqrt(CS2)``.
        """
        ...

    # -- the Guo body force, both halves (T103) ----------------------------

    def force_velocity_shift(
        self, rho: Any, u: Any, g: tuple[float, float], work: Any = None
    ) -> Any:
        """Guo's half-force correction, ``u += F / (2 rho)``, in place.

        The first half of the pair (**D-010**); the two go together or not at
        all. It runs *before* :meth:`equilibrium`, on the ``u`` that
        :meth:`macroscopic` just produced from ``f`` ``(9, ny, nx)``.

        Args:
            rho: density, ``(ny, nx)`` ``float32``.
            u: velocity, ``(2, ny, nx)`` ``float32``, modified in place.
            g: body force per unit volume ``(gx, gy)``, lattice units.
            work: optional scratch, ``(>=2, ny, nx)`` ``float32``.

        Returns:
            ``u`` — the same object passed in.
        """
        ...

    def apply_body_force(
        self,
        f: Any,
        rho: Any,
        u: Any,
        tau: float,
        g: tuple[float, float],
        work: Any = None,
    ) -> None:
        """Guo's source term, added to the post-collision distribution.

        The second half of the pair: runs after :meth:`collide` and before
        :meth:`bounce_back`, which is why a forced run takes the unfused path
        (**D-033**).

        Args:
            f: post-collision distribution, ``(9, ny, nx)`` ``float32``,
                modified in place.
            rho: density, ``(ny, nx)`` ``float32``. Unused by the formula; in
                the signature for API stability.
            u: force-corrected velocity, ``(2, ny, nx)`` ``float32``.
            tau: relaxation time, greater than 0.5.
            g: body force per unit volume ``(gx, gy)``, lattice units.
            work: optional scratch, ``(3, ny, nx)`` ``float32``.
        """
        ...

    def to_host(self, f: NDArray[np.float32]) -> NDArray[np.float32]:
        """Backend array -> host ``(9, ny, nx)`` ``float32``.

        The portability contract (constraint 4, Phase 1 form): whatever the
        backend holds internally, what comes back here is ``(direction, y, x)``
        ``float32``. Checkpoints and cross-backend comparison both go through
        this method and nothing else.

        Args:
            f: a backend-owned distribution array.

        Returns:
            ``(9, ny, nx)`` ``float32``, host memory.
        """
        ...

    def from_host(self, arr: NDArray[np.float32]) -> NDArray[np.float32]:
        """Host ``(9, ny, nx)`` ``float32`` -> backend array.

        The inverse of :meth:`to_host`, and bit-exact with it: for any host
        array ``a``, ``to_host(from_host(a))`` equals ``a`` under
        :func:`numpy.array_equal`.

        Args:
            arr: ``(9, ny, nx)`` ``float32`` in host memory.

        Returns:
            The same data in the backend's own layout.
        """
        ...


# Imported last: ``registry`` imports the concrete backends, which import
# ``lbm.core``. Re-exporting here keeps ``from lbm.backends import get_backend``
# the one import a caller needs.
from lbm.backends.registry import (  # noqa: E402
    BackendUnavailableError,
    available_backends,
    get_backend,
)
