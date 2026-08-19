"""The reference backend: pure NumPy, delegating to :mod:`lbm.core`.

Implements ``DOCS/IDEA3.md`` § What Phase 1 is, concretely — the ``lbm/`` box,
"numpy or warp backend, same API, same physics".

This backend is **the oracle, not the legacy path** (``DOCS/STATE2.md``
**D-043**): every Warp claim from T102 onward is checked against it, and a GPU
that disagrees with it is a broken backend rather than a new answer. It is also
the fallback ``DOCS/PLAN2.md`` § Risks reaches for if the port has to be demoted
— which is only a config change because of this file.

Every method here is a **delegation**, not a reimplementation. The functions in
:mod:`lbm.core` and :func:`lbm.boundary.bounce_back` are unchanged by T101, and
they are called with the same arguments in the same order, so the arithmetic —
and therefore the float ordering constraint 11 depends on — is untouched by the
seam. The four Phase 0 rungs printing their session-11 digits is the assertion
that this is true.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lbm.boundary import apply_body_force as _apply_body_force
from lbm.boundary import bounce_back as _bounce_back
from lbm.boundary import force_velocity_shift as _force_velocity_shift
from lbm.boundary import inlet_velocity as _inlet_velocity
from lbm.boundary import moving_wall as _moving_wall
from lbm.boundary import outlet_zero_gradient as _outlet_zero_gradient
from lbm.core import Q
from lbm.core import collide as _collide
from lbm.core import collide_stream as _collide_stream
from lbm.core import equilibrium as _equilibrium
from lbm.core import macroscopic as _macroscopic
from lbm.core import stream as _stream

__all__ = ["NumpyBackend"]


class NumpyBackend:
    """:class:`lbm.backends.Backend` over the Phase 0 NumPy kernels.

    Stateless: it holds no buffers and no device context, so one instance can
    serve any number of :class:`lbm.runner.Sim` objects of any size. A device
    backend will not have that property, which is why the caller obtains one
    through :func:`lbm.backends.get_backend` rather than by constructing it.

    Attributes:
        name: ``"numpy"`` — the registry key.
    """

    name: str = "numpy"

    # -- kernels ----------------------------------------------------------

    def macroscopic(
        self,
        f: NDArray[np.float32],
        rho: NDArray[np.float32] | None = None,
        u: NDArray[np.float32] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """See :meth:`lbm.backends.Backend.macroscopic`.

        Args:
            f: ``(9, ny, nx)`` ``float32``.
            rho: optional ``(ny, nx)`` ``float32`` output.
            u: optional ``(2, ny, nx)`` ``float32`` output.

        Returns:
            ``(rho, u)`` — ``(ny, nx)`` and ``(2, ny, nx)``, ``float32``.
        """
        return _macroscopic(f, rho, u)

    def equilibrium(
        self,
        rho: NDArray[np.float32],
        u: NDArray[np.float32],
        feq: NDArray[np.float32] | None = None,
        work: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """See :meth:`lbm.backends.Backend.equilibrium`.

        Args:
            rho: ``(ny, nx)`` ``float32``.
            u: ``(2, ny, nx)`` ``float32``.
            feq: optional ``(9, ny, nx)`` ``float32`` output.
            work: optional ``(3, ny, nx)`` ``float32`` scratch.

        Returns:
            ``feq``, ``(9, ny, nx)`` ``float32``.
        """
        return _equilibrium(rho, u, feq, work)

    def collide(
        self, f: NDArray[np.float32], feq: NDArray[np.float32], tau: float
    ) -> None:
        """See :meth:`lbm.backends.Backend.collide`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place.
            feq: ``(9, ny, nx)`` ``float32``.
            tau: relaxation time, greater than 0.5.
        """
        _collide(f, feq, tau)

    def bounce_back(
        self,
        f: NDArray[np.float32],
        f_pre: NDArray[np.float32],
        solid: NDArray[np.bool_],
    ) -> None:
        """See :meth:`lbm.backends.Backend.bounce_back`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place on solid cells.
            f_pre: pre-collision copy (**D-011**), ``(9, ny, nx)`` ``float32``.
            solid: ``(ny, nx)`` ``bool``.
        """
        _bounce_back(f, f_pre, solid)

    def stream(
        self, f: NDArray[np.float32], buf: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """See :meth:`lbm.backends.Backend.stream`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place.
            buf: ``(9, ny, nx)`` ``float32`` scratch.

        Returns:
            ``f`` — the same object passed in.
        """
        return _stream(f, buf)

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
    ) -> NDArray[np.float32]:
        """See :meth:`lbm.backends.Backend.collide_stream` (**D-033**).

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place.
            feq: ``(9, ny, nx)`` ``float32``.
            tau: relaxation time, greater than 0.5.
            buf: ``(9, ny, nx)`` ``float32`` scratch.
            f_pre: pre-collision copy (**D-011**), ``(9, ny, nx)`` ``float32``.
            solid: ``(ny, nx)`` ``bool``, or ``None`` to skip the reflection.
            f_bb: ``(9, ny, nx)`` ``float32`` pre-stream snapshot (**D-020**).

        Returns:
            ``f`` — the same object passed in.
        """
        return _collide_stream(
            f, feq, tau, buf, f_pre=f_pre, solid=solid, f_bb=f_bb
        )

    # -- allocation and transfer (T103) -----------------------------------
    #
    # The NumPy backend's "device" is the host, so every method below is either
    # an allocation or the identity. That is the property that keeps the
    # reference path bit-for-bit unchanged by the seam widening (**D-043**):
    # nothing is copied that Phase 0 did not copy, and nothing is converted.

    def empty(self, shape: tuple[int, ...], dtype: Any = np.float32) -> NDArray[Any]:
        """See :meth:`lbm.backends.Backend.empty`.

        Args:
            shape: e.g. ``(9, ny, nx)``.
            dtype: NumPy dtype.

        Returns:
            ``np.empty(shape, dtype)`` — the same call :class:`lbm.runner.Sim`
            made directly before T103.
        """
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float32) -> NDArray[Any]:
        """See :meth:`lbm.backends.Backend.zeros`.

        Args:
            shape: e.g. ``(2, ny, nx)``.
            dtype: NumPy dtype.

        Returns:
            ``np.zeros(shape, dtype)``.
        """
        return np.zeros(shape, dtype=dtype)

    def copy(self, dst: NDArray[Any], src: NDArray[Any]) -> None:
        """See :meth:`lbm.backends.Backend.copy` — :func:`numpy.copyto`.

        Args:
            dst: destination array.
            src: source array of the same shape and dtype.
        """
        np.copyto(dst, src)

    def upload(self, host: NDArray[Any], dst: Any = None) -> NDArray[Any]:
        """See :meth:`lbm.backends.Backend.upload`.

        Args:
            host: a NumPy array.
            dst: an existing array to write into, or ``None`` for a new one.

        Returns:
            ``dst`` when given, otherwise a fresh copy of ``host``. A copy and
            not the array itself, so that a backend array is always distinct
            from the caller's — the property a device backend has for free and
            the one Rung A's harness relies on when it hands the same host input
            to both backends.
        """
        if dst is None:
            return np.array(host, copy=True)
        np.copyto(dst, host)
        return dst

    def download(self, src: NDArray[Any], out: NDArray[Any] | None = None) -> NDArray[Any]:
        """See :meth:`lbm.backends.Backend.download` — the identity here.

        Args:
            src: a host array.
            out: an existing host array to read into, or ``None``.

        Returns:
            ``src`` itself when ``out`` is ``None`` — no bits move, which is
            what makes :meth:`lbm.runner.Sim.host_u` free on this backend.
            Treat it as read-only.
        """
        if out is None:
            return src
        np.copyto(out, src)
        return out

    # -- boundaries (T103) -------------------------------------------------

    def moving_wall(
        self,
        f: NDArray[np.float32],
        f_pre: NDArray[np.float32],
        wall: NDArray[np.bool_],
        u_wall: tuple[float, float],
        rho_w: float = 1.0,
    ) -> None:
        """See :meth:`lbm.backends.Backend.moving_wall`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place on ``wall``.
            f_pre: pre-collision copy, ``(9, ny, nx)`` ``float32``.
            wall: ``(ny, nx)`` ``bool``.
            u_wall: ``(ux, uy)`` lattice velocity of the wall.
            rho_w: wall density in the correction term.
        """
        _moving_wall(f, f_pre, wall, u_wall, rho_w)

    def inlet_velocity(
        self,
        f: NDArray[np.float32],
        *,
        col: int = 0,
        u_in: NDArray[np.float32],
        work: NDArray[np.float32] | None = None,
        fluid: NDArray[np.bool_] | None = None,
    ) -> None:
        """See :meth:`lbm.backends.Backend.inlet_velocity`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in column ``col``.
            col: inlet column.
            u_in: ``(2, ny)`` ``float32`` prescribed profile.
            work: ``(>=5, ny)`` ``float32`` scratch.
            fluid: ``(ny,)`` ``bool`` row mask.
        """
        _inlet_velocity(f, col=col, u_in=u_in, work=work, fluid=fluid)

    def outlet_zero_gradient(
        self,
        f: NDArray[np.float32],
        *,
        col: int = -1,
        src: int = -2,
        prev: NDArray[np.float32] | None = None,
        lam: float | None = None,
    ) -> None:
        """See :meth:`lbm.backends.Backend.outlet_zero_gradient` (**D-021**).

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in column ``col``.
            col: outlet column.
            src: column read from.
            prev: ``(9, ny)`` ``float32`` previous outlet column, updated in
                place. ``None`` selects the plain copy.
            lam: advection speed; ``None`` means ``sqrt(CS2)``.
        """
        _outlet_zero_gradient(f, col=col, src=src, prev=prev, lam=lam)

    # -- the Guo body force, both halves (T103) ----------------------------

    def force_velocity_shift(
        self,
        rho: NDArray[np.float32],
        u: NDArray[np.float32],
        g: tuple[float, float],
        work: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """See :meth:`lbm.backends.Backend.force_velocity_shift`.

        Args:
            rho: ``(ny, nx)`` ``float32``.
            u: ``(2, ny, nx)`` ``float32``, modified in place.
            g: ``(gx, gy)`` lattice body force.
            work: ``(>=2, ny, nx)`` ``float32`` scratch.

        Returns:
            ``u`` — the same object passed in.
        """
        return _force_velocity_shift(rho, u, g, work)

    def apply_body_force(
        self,
        f: NDArray[np.float32],
        rho: NDArray[np.float32],
        u: NDArray[np.float32],
        tau: float,
        g: tuple[float, float],
        work: NDArray[np.float32] | None = None,
    ) -> None:
        """See :meth:`lbm.backends.Backend.apply_body_force`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place.
            rho: ``(ny, nx)`` ``float32``, unused by the formula.
            u: ``(2, ny, nx)`` ``float32`` force-corrected velocity.
            tau: relaxation time.
            g: ``(gx, gy)`` lattice body force.
            work: ``(3, ny, nx)`` ``float32`` scratch.
        """
        _apply_body_force(f, rho, u, tau, g, work)

    # -- the portability contract -----------------------------------------

    def to_host(self, f: NDArray[np.float32]) -> NDArray[np.float32]:
        """Identity, with the shape and dtype contract checked.

        The NumPy backend's device *is* the host, so this returns the array
        itself rather than a copy — no bits move and none can change, which is
        what keeps constraint 11's bit-identical restart true through the seam.
        A device backend copies here; the caller must treat the result as
        read-only either way.

        Args:
            f: ``(9, ny, nx)`` ``float32``.

        Returns:
            The same array, ``(9, ny, nx)`` ``float32``.

        Raises:
            ValueError: if the shape or dtype is not the host contract.
        """
        self._check_host(f)
        return f

    def from_host(self, arr: NDArray[np.float32]) -> NDArray[np.float32]:
        """Identity, with the shape and dtype contract checked.

        Args:
            arr: ``(9, ny, nx)`` ``float32``.

        Returns:
            The same array.

        Raises:
            ValueError: if the shape or dtype is not the host contract.
        """
        self._check_host(arr)
        return arr

    @staticmethod
    def _check_host(arr: NDArray[np.float32]) -> None:
        """Reject anything that is not ``(9, ny, nx)`` ``float32``.

        Constraint 4 in its Phase 1 form (**D-046**) makes this shape the one
        thing every backend agrees on. Checking it here rather than trusting it
        is what makes a layout mistake in a future backend fail at the seam
        instead of three rungs later.

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
        return f"NumpyBackend(name={self.name!r})"
