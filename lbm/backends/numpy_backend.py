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

import numpy as np
from numpy.typing import NDArray

from lbm.boundary import bounce_back as _bounce_back
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
