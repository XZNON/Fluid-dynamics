"""The Warp backend: the four hot kernels on the GPU, arithmetic unchanged.

Implements ``DOCS/IDEA3.md`` § What Phase 1 is, concretely — the ``lbm/`` box,
"numpy or warp backend, same API, same physics" — and § Performance budget,
whose floors T103 measures against. **T102 writes the kernels; T103 moves the
whole timestep and the boundaries onto the device.**

Why ``equilibrium`` is first: at 1M cells it is **39.9 ms of a ~75 ms step**
(``old-Docs/STATE1.md`` § Performance baseline), over half the budget.

What this module is not
-----------------------
It is **not a rewrite of the physics**. Every kernel below is a transcription of
its :mod:`lbm.core` counterpart, term for term and *in the same order*
(``CLAUDE.md`` constraint 1 in its Phase 1 form). Where a kernel would read
"better" written differently — the obvious one is ``equilibrium``, where a GPU
thread has registers and does not need :mod:`lbm.core`'s ``work`` scratch — the
arithmetic is still emitted in core's order and the difference is noted in a
comment. NumPy is the oracle (**D-043**); a GPU that disagrees with it is a
broken backend, never a new answer.

The nine constants
------------------
``E``, ``E_F32``, ``W``, ``OPP`` and ``CS2`` are **imported from**
:mod:`lbm.core` and uploaded to the device **once, at construction**
(``CLAUDE.md`` constraint 4 / "no physics constant twice"). No literal lattice
constant appears in a kernel below. The three numeric literals that do appear in
:func:`_equilibrium_kernel` — ``1.5``, ``3.0``, ``4.5`` — are the ones
:func:`lbm.core.equilibrium` itself writes as literals; they are ``1/(2 cs2)``,
``1/cs2`` and ``1/(2 cs2^2)``, and rewriting them in terms of the uploaded
``CS2`` would change the emitted arithmetic, which constraint 1 forbids.
``CS2`` is uploaded anyway, because T103's boundaries take it.

Where the state lives (and where it does not, yet)
--------------------------------------------------
**D-051**: the ``Backend`` protocol covers kernels and the two host transfers
and nothing else — :class:`lbm.runner.Sim` still owns its ``(9, ny, nx)``
*host* buffers. So this backend takes host arrays at its boundary, keeps its own
**preallocated** device buffers, and copies in and out per call. Those copies
are the reason T102 quotes no speed number: the seam widening that lets ``Sim``
hold device state is T103's, along with the fused ``collide_stream`` and the
performance table (constraint 6's replacement — *no backend optimisation before
its parity rung passes*).

Device buffers are allocated **once per grid shape** and reused: construct with
``WarpBackend(shape=(ny, nx))`` to allocate them up front, or let the first call
for a shape allocate them. Either way no call after the first allocates device
memory, which ``tests/test_warp_backend.py`` asserts over 1000 steps' worth of
launches.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

import warp as wp

from lbm.core import CS2, E, E_F32, OPP, Q, W

__all__ = ["WarpBackend"]


# --- kernels ----------------------------------------------------------------
# One transcription each of lbm.core.macroscopic / equilibrium / collide /
# stream. Read them beside the originals; the operation order is the same.


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
        dst: destination, ``(9, ny, nx)`` ``float32``, device.
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


# --- device buffers ---------------------------------------------------------


class _GridBuffers:
    """The device arrays one grid shape needs, allocated once.

    ``CLAUDE.md`` § Coding conventions: "preallocate, never allocate inside the
    step loop". The host-side equivalent lives on :class:`lbm.runner.Sim`; this
    is its device half until T103 merges the two (**D-051**).

    Attributes:
        f: ``(9, ny, nx)`` ``float32`` device array.
        feq: ``(9, ny, nx)`` ``float32`` device array.
        buf: ``(9, ny, nx)`` ``float32`` device array — stream's destination.
        rho: ``(ny, nx)`` ``float32`` device array.
        u: ``(2, ny, nx)`` ``float32`` device array.
    """

    __slots__ = ("f", "feq", "buf", "rho", "u")

    def __init__(self, ny: int, nx: int, device: Any) -> None:
        """Allocate every device buffer for a ``(ny, nx)`` grid.

        Args:
            ny: rows.
            nx: columns.
            device: the Warp device to allocate on.
        """
        self.f = wp.zeros((Q, ny, nx), dtype=wp.float32, device=device)
        self.feq = wp.zeros((Q, ny, nx), dtype=wp.float32, device=device)
        self.buf = wp.zeros((Q, ny, nx), dtype=wp.float32, device=device)
        self.rho = wp.zeros((ny, nx), dtype=wp.float32, device=device)
        self.u = wp.zeros((2, ny, nx), dtype=wp.float32, device=device)


class WarpBackend:
    """:class:`lbm.backends.Backend` on a Warp device.

    Implements ``DOCS/TASKS2.md`` § T102. The four kernels above plus the two
    host transfers; the boundaries and the fused ``collide_stream`` are T103's
    and raise :class:`NotImplementedError` here.

    Attributes:
        name: ``"warp"`` — the registry key.
        device: the Warp device the kernels run on.
    """

    name: str = "warp"

    def __init__(
        self,
        device: str | None = None,
        shape: tuple[int, int] | None = None,
    ) -> None:
        """Initialise Warp, upload the constants, optionally preallocate.

        Args:
            device: a Warp device string, e.g. ``"cuda:0"`` or ``"cpu"``.
                ``None`` takes :func:`warp.get_preferred_device`, which is the
                first CUDA device when there is one and the CPU otherwise — so
                the parity rung runs on a machine without a GPU, at CPU speed,
                rather than not running at all.
            shape: ``(ny, nx)`` to allocate device buffers for immediately.
                ``None`` allocates them on the first call for a shape, once.

        Raises:
            RuntimeError: if Warp cannot initialise or has no usable device.
        """
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

        self._buffers: dict[tuple[int, int], _GridBuffers] = {}

        # Compile now rather than inside the first timed loop.
        wp.load_module(module=__name__, device=self.device)

        if shape is not None:
            self._grid(int(shape[0]), int(shape[1]))

    # -- buffers and transfers --------------------------------------------

    def _grid(self, ny: int, nx: int) -> _GridBuffers:
        """The device buffers for a ``(ny, nx)`` grid, allocating once.

        Args:
            ny: rows.
            nx: columns.

        Returns:
            The cached :class:`_GridBuffers` for that shape.
        """
        key = (ny, nx)
        bufs = self._buffers.get(key)
        if bufs is None:
            bufs = _GridBuffers(ny, nx, self.device)
            self._buffers[key] = bufs
        return bufs

    @staticmethod
    def _wrap(host: NDArray[np.float32]) -> Any:
        """A zero-copy Warp view of a host ``float32`` array.

        :func:`warp.copy` between this view and a device array is the transfer;
        wrapping allocates no host and no device memory, which is what keeps
        "no allocation per call" true while ``Sim`` still owns the host buffers
        (**D-051**).

        Args:
            host: a C-contiguous ``float32`` NumPy array.

        Returns:
            A ``device="cpu"`` :class:`warp.array` aliasing ``host``.

        Raises:
            ValueError: if ``host`` is not C-contiguous ``float32``.
        """
        if host.dtype != np.float32:
            raise ValueError(
                f"the Warp backend transfers float32 only (CLAUDE.md "
                f"constraint 4); got {host.dtype}."
            )
        if not host.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "the Warp backend needs C-contiguous host arrays; got a "
                "non-contiguous view. Pass the buffer itself, not a slice."
            )
        return wp.array(host, dtype=wp.float32, copy=False, device="cpu")

    def _upload(self, host: NDArray[np.float32], dev: Any) -> None:
        """Host -> device, into an already-allocated device array.

        Args:
            host: C-contiguous ``float32`` source.
            dev: device array of the same shape.
        """
        wp.copy(dev, self._wrap(host))

    def _download(self, dev: Any, host: NDArray[np.float32]) -> None:
        """Device -> host, into an already-allocated host array.

        Args:
            dev: device array source.
            host: C-contiguous ``float32`` destination of the same shape.
        """
        wp.copy(self._wrap(host), dev)
        wp.synchronize_device(self.device)

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
        _, ny, nx = f.shape
        g = self._grid(ny, nx)

        if rho is None:
            rho = np.empty((ny, nx), dtype=np.float32)
        if u is None:
            u = np.empty((2, ny, nx), dtype=np.float32)

        self._upload(f, g.f)
        wp.launch(
            _macroscopic_kernel,
            dim=(ny, nx),
            inputs=[g.f, self._e_f32, g.rho, g.u],
            device=self.device,
        )
        self._download(g.rho, rho)
        self._download(g.u, u)
        return rho, u

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
            work: ignored — a thread keeps core's three ``(ny, nx)``
                temporaries in registers. Accepted so that the signature
                matches the protocol term for term.

        Returns:
            ``feq``, ``(9, ny, nx)`` ``float32``.
        """
        del work  # see the docstring: registers, not scratch arrays
        ny, nx = rho.shape
        g = self._grid(ny, nx)

        if feq is None:
            feq = np.empty((Q, ny, nx), dtype=np.float32)

        self._upload(rho, g.rho)
        self._upload(u, g.u)
        wp.launch(
            _equilibrium_kernel,
            dim=(ny, nx),
            inputs=[g.rho, g.u, self._e_f32, self._w, g.feq],
            device=self.device,
        )
        self._download(g.feq, feq)
        return feq

    def collide(
        self, f: NDArray[np.float32], feq: NDArray[np.float32], tau: float
    ) -> None:
        """See :meth:`lbm.backends.Backend.collide`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place.
            feq: ``(9, ny, nx)`` ``float32``.
            tau: relaxation time, greater than 0.5.

        Raises:
            ValueError: if ``tau <= 0.5`` — the check and the message are
                :func:`lbm.core.collide`'s (``CLAUDE.md`` constraint 2).
        """
        if tau <= 0.5:
            raise ValueError(
                f"tau must be greater than 0.5 (got tau={tau!r}): "
                "collision with tau <= 0.5 gives non-positive viscosity and diverges."
            )
        one_minus_omega = np.float32(1.0 - 1.0 / tau)

        _, ny, nx = f.shape
        g = self._grid(ny, nx)

        self._upload(f, g.f)
        self._upload(feq, g.feq)
        wp.launch(
            _collide_kernel,
            dim=(Q, ny, nx),
            inputs=[g.f, g.feq, float(one_minus_omega)],
            device=self.device,
        )
        self._download(g.f, f)

    def stream(
        self, f: NDArray[np.float32], buf: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """See :meth:`lbm.backends.Backend.stream`.

        Args:
            f: ``(9, ny, nx)`` ``float32``, modified in place.
            buf: ``(9, ny, nx)`` ``float32`` scratch. Left holding the streamed
                state, as :func:`lbm.core.stream` leaves it.

        Returns:
            ``f`` — the same object passed in, so its buffer identity survives
            the call (T006's restart test depends on that).
        """
        _, ny, nx = f.shape
        g = self._grid(ny, nx)

        self._upload(f, g.f)
        wp.launch(
            _stream_kernel,
            dim=(Q, ny, nx),
            inputs=[g.f, self._e_i32, g.buf],
            device=self.device,
        )
        self._download(g.buf, f)
        np.copyto(buf, f)
        return f

    # -- T103's half of the backend ---------------------------------------

    def bounce_back(
        self,
        f: NDArray[np.float32],
        f_pre: NDArray[np.float32],
        solid: NDArray[np.bool_],
    ) -> None:
        """Not on the GPU yet — see ``DOCS/TASKS2.md`` T103.

        Args:
            f: ``(9, ny, nx)`` ``float32``.
            f_pre: pre-collision copy (**D-011**), same shape and dtype.
            solid: ``(ny, nx)`` ``bool``.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError("see DOCS/TASKS2.md T103")

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
        """Not on the GPU yet — see ``DOCS/TASKS2.md`` T103.

        Args:
            f: ``(9, ny, nx)`` ``float32``.
            feq: ``(9, ny, nx)`` ``float32``.
            tau: relaxation time, greater than 0.5.
            buf: ``(9, ny, nx)`` ``float32`` scratch.
            f_pre: pre-collision copy (**D-011**).
            solid: ``(ny, nx)`` ``bool``, or ``None``.
            f_bb: ``(9, ny, nx)`` ``float32`` pre-stream snapshot (**D-020**).

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError("see DOCS/TASKS2.md T103")

    # -- the portability contract -----------------------------------------

    def to_host(self, f: Any) -> NDArray[np.float32]:
        """Backend array -> host ``(9, ny, nx)`` ``float32`` (constraint 4).

        Accepts either a device :class:`warp.array` or the host array ``Sim``
        currently owns (**D-051**), because in T102 the state still lives on the
        host and the device arrays are this backend's own buffers. When T103
        moves the state onto the device, only the first branch is reached, and
        this method stays the only path a checkpoint takes (**D-050**).

        Args:
            f: a device ``(9, ny, nx)`` ``float32`` Warp array, or the
                equivalent host array.

        Returns:
            ``(9, ny, nx)`` ``float32`` in host memory.

        Raises:
            ValueError: if the shape or dtype is not the host contract.
        """
        host = f.numpy() if isinstance(f, wp.array) else f
        self._check_host(host)
        return host

    def from_host(self, arr: NDArray[np.float32]) -> NDArray[np.float32]:
        """Host ``(9, ny, nx)`` ``float32`` -> backend array.

        The inverse of :meth:`to_host` and bit-exact with it: no arithmetic
        happens on either side, so ``to_host(from_host(a))`` equals ``a`` under
        :func:`numpy.array_equal`.

        Args:
            arr: ``(9, ny, nx)`` ``float32`` in host memory.

        Returns:
            The same data in the backend's layout — which, in T102, is the host
            layout (**D-051**).

        Raises:
            ValueError: if the shape or dtype is not the host contract.
        """
        self._check_host(arr)
        return arr

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
