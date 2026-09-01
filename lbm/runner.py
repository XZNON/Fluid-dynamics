"""The continuous loop: one timestep, many timesteps per frame, and restart.

Implements ``DOCS/IDEA2.md`` § "Continuous simulation — the part that matters
most". The simulation is a *stream*, not a batch job:

* :class:`Sim` owns ``f``, ``solid``, ``step_count`` and **every** buffer the
  step needs, so :meth:`Sim.step` allocates nothing (``CLAUDE.md`` conventions,
  ``old-Docs/STATE1.md`` D-006).
* :func:`steps_per_frame` is **computed** from the target playback speed
  (constraint 7). A hardcoded 20 is exactly what this module exists to avoid.
* :class:`RingBuffer` sits between the physics and the sink. When it fills it
  drops the **oldest display frame** and never a simulation step (constraint 8).
* :func:`save_checkpoint` / :func:`load_checkpoint` pickle ``f``, ``solid``,
  ``step_count`` and the config — the entire state — and a resumed run is
  bit-identical (constraint 11), which ``tests/test_runner.py`` asserts rather
  than claims.

Since T101 (``DOCS/IDEA3.md`` § What Phase 1 is, concretely) this module reaches
**every kernel through** :attr:`Sim.backend` and imports none of them directly —
``lbm.core`` is imported here for its constants (``Q``, ``W``) and nothing else,
which ``tests/test_backends.py`` asserts at import level. ``config.backend``
picks the implementation by name; ``"numpy"`` is the default and the reference
oracle (**D-043**).

Everything here is lattice units (``lbm/units.py``, T009, converts at the
boundary). Rendering is not here: :class:`Sink` is abstract and only
:class:`NullSink` is implemented — the live sink is T007 and the recording sinks
are T011 (constraint 10).

The timestep order
------------------

``old-Docs/STATE1.md`` **D-020**, also carried by ``lbm/boundary.py``'s module
docstring::

    copyto(f_pre, f)                       # pre-collision copy, for bounce_back
    macroscopic(f, rho, u)
    force_velocity_shift(rho, u, g, work)  # Guo, first half
    equilibrium(rho, u, feq, work)
    collide(f, feq, tau)
    apply_body_force(f, rho, u, tau, g, work)   # Guo, second half
    bounce_back(f, f_pre, solid)
    copyto(f_bb, f)                        # pre-stream copy, for probe.forces
    stream(f, buf)
    outlet_zero_gradient(f, prev=out_prev) # after stream: it is periodic in x
    inlet_velocity(f, u_in=u_in, work=inlet_work)

``f_pre`` (pre-collision) and ``f_bb`` (pre-stream) are two buffers with two
meanings and are deliberately not merged.

**Every line above is reached through** :attr:`Sim.backend` **and never imported
here** (T101 for the kernels, T103 for the boundaries and the body force): the
*order* is physics and is the same on every backend, while the arithmetic inside
each call is the backend's. :mod:`lbm.runner` imports no kernel and no boundary,
and ``tests/test_backends.py`` asserts it.

On the fused path (**D-033**) the first line is **absent**, and that is a
removal rather than a shortcut: :func:`lbm.core.collide_stream` stages every
direction in ``f_bb`` and does not write ``f`` until the stream lands, so ``f``
is still the pre-collision state when the reflection reads it. Passing ``f``
where D-011's copy would go is bitwise identical — asserted, not argued, in
``tests/test_backends.py`` and ``tests/test_warp_backend.py`` — and it removes a
whole ``(9, ny, nx)`` copy per step, which at 2M cells is 144 MB of bandwidth the
GPU budget cannot spare. It is only valid **because** ``f_bb`` is supplied: with
``f_bb=None`` the pass stages in ``f`` itself and the alias would read values it
had already overwritten.

Why the checkpoint is still only three things
---------------------------------------------

The convective outlet (D-021) carries the previous outlet column, ``out_prev``,
across steps, which looks like a fourth piece of state. It is not: nothing after
:func:`lbm.boundary.outlet_zero_gradient` writes the outlet column — the inlet
is a different column — so at the end of every step ``out_prev`` is byte-identical
to ``f[:, :, outlet_col]`` and :func:`load_checkpoint` rebuilds it from ``f``.
The restart test exercises a run with the convective outlet on for that reason.
"""

from __future__ import annotations

import abc
import pickle
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from lbm.backends import Backend, get_backend
from lbm.boundary import U_MAX, inlet_profile
from lbm.core import Q, W
from lbm.geometry import bounding_box, check_mask
from lbm.probe import BoundaryLinks, boundary_links, forces, residual, vorticity

__all__ = [
    "SimConfig",
    "Sim",
    "RingBuffer",
    "RunStats",
    "Sink",
    "NullSink",
    "run",
    "steps_per_frame",
    "save_checkpoint",
    "load_checkpoint",
    "demo_domain",
    "main",
]

#: Bumped if the pickle layout ever changes. :func:`load_checkpoint` refuses a
#: format it does not know rather than unpacking it wrongly.
CHECKPOINT_FORMAT: int = 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Everything about a run except the mask and the state itself.

    Plain scalars only — no arrays — so it pickles cheaply and a checkpoint can
    carry it verbatim (``old-Docs/TASKS1.md`` § T006: the checkpoint is ``f``,
    ``solid``, ``step_count`` and *the config*). The mask travels beside it,
    not inside it.

    All lattice units (``DOCS/IDEA2.md`` § Module layout: physical units are
    converted by ``lbm/units.py`` at the boundary and never reach the solver).

    Attributes:
        ny: rows.
        nx: columns.
        tau: BGK relaxation time. ``nu = (tau - 0.5) / 3``; ``tau <= 0.5`` is
            rejected (constraint 2).
        inlet_U: inlet velocity passed to :func:`lbm.boundary.inlet_profile`.
        profile: ``"uniform"`` or ``"parabolic"``.
        inlet_uy: cross-stream inlet velocity.
        inlet_col: column the Zou–He inlet writes.
        outlet_col: column the outflow condition writes.
        outlet_src: column the outflow condition reads.
        outlet_lam: advection speed of the convective outlet. ``None`` means
            ``sqrt(CS2)``, which D-021 measured at 0.6% reflection against 35%
            for the bare copy.
        use_inlet: apply the Zou–He inlet each step.
        use_outlet: apply the outflow condition each step.
        convective_outlet: use the convective form (owns ``out_prev``) rather
            than the plain copy. Ignored when ``use_outlet`` is false.
        g: body force ``(gx, gy)``, Guo scheme (D-010). ``(0, 0)`` skips both
            halves of the scheme entirely.
        rho0: initial and reference density.
        D: characteristic length in cells for the force coefficients. ``None``
            derives it from the mask bounding box (D-019).
        inlet_axis: ``"x"`` or ``"y"``, passed to
            :func:`lbm.geometry.check_mask`.
        check_geometry: run the mask sanity checks at setup (constraint 12).
        verbose_mask: let ``check_mask`` print its report.
        checkpoint_every: auto-checkpoint interval in steps. ``0`` is **off**,
            which is the default.
        checkpoint_path: where auto-checkpoints are written.
        fused: use :func:`lbm.core.collide_stream`, which walks each direction
            once instead of walking the whole array for collide, bounce-back,
            the ``f_bb`` snapshot and the shift separately (T010). Bitwise
            identical to the unfused sequence — ``tests/test_perf.py`` asserts
            it — so this is a speed switch and never a physics one. ``False``
            selects the T009 path, which is what the equality test compares
            against. Ignored when a body force is present: the Guo source term
            (D-010) is applied between collision and bounce-back, and Rung 1,
            the only case that uses it, is 22x16 cells.
        backend: which :class:`lbm.backends.Backend` runs the kernels, by
            registry name (T101, ``DOCS/IDEA3.md`` § What Phase 1 is,
            concretely). ``"numpy"`` is the default and the reference oracle
            (**D-043**); ``"warp"`` arrives in T102. A name nothing answers to
            raises :class:`ValueError` at :class:`Sim` construction, listing
            what is available. It is a plain string so the config still pickles
            cheaply and a checkpoint carries it verbatim (**D-050**).
        cs_smag: the Smagorinsky constant ``Cs`` of the T201 closure
            (``DOCS/IDEA4.md`` § The five things Phase 2 must get right, (1)
            and (2)). **``0.0`` is the default and means the closure is off**,
            which is constraint 19: with it off the collision is bitwise what
            Phase 1 shipped, and the nine rungs below this phase are therefore
            untouched by the closure's existence. The literature value is
            :data:`lbm.core.CS_SMAG_LITERATURE` and Phase 2 does not tune it.
            A plain float, so the config still pickles cheaply and a checkpoint
            carries it verbatim; it adds no *state* — ``tau_eff`` is derived
            every step and ``f``, ``solid`` and ``step_count`` remain the whole
            checkpoint (constraint 11, **D-022**, **D-050**).
    """

    ny: int
    nx: int
    tau: float
    inlet_U: float = 0.0
    profile: str = "uniform"
    inlet_uy: float = 0.0
    inlet_col: int = 0
    outlet_col: int = -1
    outlet_src: int = -2
    outlet_lam: float | None = None
    use_inlet: bool = False
    use_outlet: bool = False
    convective_outlet: bool = True
    g: tuple[float, float] = (0.0, 0.0)
    rho0: float = 1.0
    D: float | None = None
    inlet_axis: str = "x"
    check_geometry: bool = True
    verbose_mask: bool = False
    checkpoint_every: int = 0
    checkpoint_path: str | None = None
    fused: bool = True
    backend: str = "numpy"
    cs_smag: float = 0.0

    def replace(self, **changes: Any) -> "SimConfig":
        """A copy with fields overridden (``dataclasses.replace``)."""
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# steps_per_frame
# ---------------------------------------------------------------------------


def steps_per_frame(dt: float, fps: float = 60.0, speed: float = 1.0) -> int:
    """How many timesteps one rendered frame is worth.

    ``DOCS/IDEA2.md`` § Decouple simulation from rendering: "``steps_per_frame``
    is the knob that makes the video look like real time regardless of grid
    size. Compute it from the target physical playback speed — do not hardcode
    20." ``CLAUDE.md`` constraint 7 says the same thing and this function is
    what satisfies it.

    The arithmetic, in full::

        dt                 seconds of physical time advanced by one timestep
        1 / fps            seconds of wall-clock time one displayed frame lasts
        speed              playback rate: 1.0 real time, 10.0 ten times faster

        physical time to show in one frame = speed / fps          [s]
        steps to advance that much time    = (speed / fps) / dt   [steps]

        steps_per_frame = round(speed / (fps * dt)),  at least 1

    Worked example. A cylinder at ``D = 20`` cells with ``U = 0.05`` lattice and
    a physical free stream of 1 m/s: one lattice step is
    ``dt = (U_lattice / U_phys) * dx`` seconds, say ``dt = 5e-4 s``. At 60 fps
    and real-time playback, ``1 / (60 * 5e-4) = 33`` steps per frame. Refine the
    grid by 2 and ``dt`` halves, so the same playback speed asks for 66 steps —
    which is the whole point of computing it rather than fixing it.

    ``dt`` is a plain scalar handed in by the caller (``lbm/units.py``, T009
    computes it from the grid spacing and the velocity scale). Grid size enters
    through ``dt``, which is where it enters physically; no physical units cross
    into this package.

    Args:
        dt: seconds of physical time per lattice timestep. Must be positive.
        fps: target display rate, frames per second. Must be positive.
        speed: playback rate. ``1.0`` is real time; ``0.5`` is slow motion.

    Returns:
        Timesteps per rendered frame, at least 1.

    Raises:
        ValueError: if any argument is not positive.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive (got {dt!r}): it is seconds per timestep.")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive (got {fps!r}).")
    if speed <= 0.0:
        raise ValueError(f"speed must be positive (got {speed!r}).")

    return max(1, int(round(speed / (fps * dt))))


# ---------------------------------------------------------------------------
# Ring buffer and sinks
# ---------------------------------------------------------------------------


class RingBuffer:
    """A bounded frame queue that drops the oldest frame when it is full.

    ``DOCS/IDEA2.md`` § Never block the sim on the display: "If the display is
    slow, the physics should not stutter. Put a small ring buffer between them.
    If it fills, drop display frames, never simulation steps."

    That is ``CLAUDE.md`` constraint 8, and it is the reason :meth:`push` never
    blocks and never returns a failure the producer has to handle: a full buffer
    costs a *display* frame and is recorded in :attr:`dropped`.

    Thread-safe: :func:`run` drains it from a consumer thread while the sim
    thread pushes.

    Attributes:
        maxlen: capacity in frames.
        dropped: frames discarded because the buffer was full.
        pushed: frames accepted (including ones later dropped).
    """

    def __init__(self, maxlen: int = 4) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be at least 1 (got {maxlen!r}).")
        self._q: deque[Any] = deque()
        self.maxlen: int = int(maxlen)
        self.dropped: int = 0
        self.pushed: int = 0
        self._lock = threading.Lock()

    def push(self, item: Any) -> bool:
        """Add a frame, evicting the oldest if the buffer is full.

        Returns:
            ``True`` if nothing was evicted, ``False`` if a frame was dropped.
        """
        with self._lock:
            self.pushed += 1
            evicted = False
            if len(self._q) >= self.maxlen:
                self._q.popleft()
                self.dropped += 1
                evicted = True
            self._q.append(item)
            return not evicted

    def pop(self) -> Any | None:
        """Remove and return the oldest frame, or ``None`` if empty."""
        with self._lock:
            if not self._q:
                return None
            return self._q.popleft()

    def clear(self) -> None:
        """Discard every buffered frame without counting them as dropped."""
        with self._lock:
            self._q.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    def __repr__(self) -> str:
        return (
            f"RingBuffer(maxlen={self.maxlen}, len={len(self)}, "
            f"pushed={self.pushed}, dropped={self.dropped})"
        )


class Sink(abc.ABC):
    """Where rendered frames go.

    ``DOCS/IDEA2.md`` § Three output sinks, same frame source — live, record,
    headless — all fed by one ``render()``. ``CLAUDE.md`` constraint 10: do not
    write three renderers.

    Only the abstract base and :class:`NullSink` exist in T006. The live pygame
    sink is T007 and the MP4/GIF sinks are T011; the ring buffer is built first,
    deliberately, because a fake slow sink proves frame-dropping far more
    cleanly than a real window does (``old-Docs/TASKS1.md`` § T006 Notes).
    """

    @abc.abstractmethod
    def push(self, frame: Any) -> None:
        """Consume one frame. May be slow; the sim never waits on it."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release whatever the sink holds. Called once at the end of a run."""

    def __enter__(self) -> "Sink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullSink(Sink):
    """A sink that counts frames and does nothing else.

    The headless case of ``DOCS/IDEA2.md`` § Three output sinks, and what the
    runner's own tests measure against.

    Attributes:
        count: frames pushed.
        closed: whether :meth:`close` has been called.
    """

    def __init__(self) -> None:
        self.count: int = 0
        self.closed: bool = False

    def push(self, frame: Any) -> None:
        self.count += 1

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------


class Sim:
    """One simulation: the state, every buffer, and one timestep.

    ``DOCS/IDEA2.md`` § The method, in the order the code runs it — this class
    calls those functions and owns nothing else. The physics lives in
    :mod:`lbm.core`, :mod:`lbm.boundary` and :mod:`lbm.probe`; the constants
    live in :mod:`lbm.core` and are never redefined (constraint 4).

    Buffers are allocated once here and reused forever, which is what makes
    :meth:`step` allocation-free (``CLAUDE.md`` conventions; D-006 gave every
    hot function an optional preallocated output for exactly this caller).

    Where the state lives (T103)
    ----------------------------
    Every state and scratch array below is allocated **by the backend**, so on
    ``"numpy"`` it is an :class:`numpy.ndarray` exactly as in Phase 0 and on
    ``"warp"`` it is a device array and a timestep moves nothing across the bus.
    Code outside this class that wants to *read* the state calls
    :meth:`host_f`, :meth:`host_u`, :meth:`host_rho` or :meth:`host_f_bb`, which
    are free on a host backend and a synchronising download on a device one —
    and are therefore called on the frame or probe cadence, never per step
    (constraint 8). :meth:`load_f` is the write half.

    Attributes:
        config: the :class:`SimConfig` this was built from.
        f: distribution function, ``(9, ny, nx)`` ``float32``, backend array.
        solid: solid mask, shape ``(ny, nx)``, ``bool`` — always on the host,
            because the geometry checks and the force integral read it.
        step_count: timesteps executed since the run began.
        rho: density, ``(ny, nx)`` ``float32`` backend array, refreshed each
            step.
        u: velocity, ``(2, ny, nx)`` ``float32`` backend array, refreshed each
            step. With a body force this is the Guo-corrected velocity (D-010).
        f_pre: pre-**collision** copy (D-011), ``(9, ny, nx)``. Written only on
            the unfused path; the fused pass needs no copy — see :meth:`step`.
        f_bb: pre-**stream** snapshot (D-020), ``(9, ny, nx)``, what
            :func:`lbm.probe.forces` consumes.
        u_in: the inlet profile, ``(2, ny)`` ``float32``, on the **host**. Edit
            it in place and call :meth:`refresh_inlet_profile`.
        links: bounce-back links, built once from the mask (D-020).
        D: characteristic length in cells (D-019).
    """

    def __init__(
        self,
        config: SimConfig,
        solid: NDArray[np.bool_] | None = None,
        *,
        f: NDArray[np.float32] | None = None,
        step_count: int = 0,
    ) -> None:
        """Allocate the state and every buffer.

        Args:
            config: the run configuration.
            solid: solid mask, shape ``(ny, nx)``, ``bool``. ``None`` is an
                empty domain.
            f: an existing distribution to adopt, shape ``(9, ny, nx)``,
                ``float32``. Copied, not aliased. ``None`` initialises to the
                equilibrium of ``rho0`` and the inlet profile.
            step_count: steps already executed — set by
                :func:`load_checkpoint`.

        Raises:
            ValueError: on ``tau <= 0.5`` (constraint 2), a mask/state whose
                shape disagrees with the config, or a ``config.backend`` no
                registered backend answers to (T101).
        """
        cfg = config
        if cfg.tau <= 0.5:
            raise ValueError(
                f"tau must exceed 0.5 (got {cfg.tau!r}): nu = (tau - 0.5) / 3, "
                f"so tau -> 0.5 means nu -> 0 and the simulation blows up "
                f"(CLAUDE.md constraint 2)."
            )
        if cfg.ny < 3 or cfg.nx < 3:
            raise ValueError(f"grid must be at least 3x3 (got {cfg.ny}x{cfg.nx}).")

        self.config = cfg
        ny, nx = cfg.ny, cfg.nx

        # The seam (T101). Every kernel this class calls goes through here and
        # nowhere else, so swapping the compute target is a config change
        # (``DOCS/PLAN2.md`` § Risks, the pressure valve behind D-043). Resolved
        # before anything is allocated: a bad name should cost nothing.
        self.backend: Backend = get_backend(cfg.backend)

        if solid is None:
            solid = np.zeros((ny, nx), dtype=bool)
        else:
            solid = np.ascontiguousarray(solid, dtype=bool)
            if solid.shape != (ny, nx):
                raise ValueError(
                    f"solid must be {(ny, nx)} to match the config "
                    f"(got {solid.shape})."
                )
        self.solid: NDArray[np.bool_] = solid

        if cfg.check_geometry and solid.any():
            check_mask(solid, cfg.inlet_axis, verbose=cfg.verbose_mask)

        # --- state -------------------------------------------------------
        # Allocated **by the backend** (T103): on ``"numpy"`` these are plain
        # ``np.empty`` arrays and nothing about Phase 0 changes, while on a
        # device backend they are device arrays and a timestep moves no bytes
        # across the bus. See :mod:`lbm.backends`, "What T103 added".
        be = self.backend
        self.step_count: int = int(step_count)
        self.f: Any = be.empty((Q, ny, nx))

        # --- buffers, allocated once (D-006, D-020) ----------------------
        self.f_pre: Any = be.empty((Q, ny, nx))  # pre-collision (D-011)
        self.f_bb: Any = be.empty((Q, ny, nx))  # pre-stream (D-020)
        self.buf: Any = be.empty((Q, ny, nx))  # stream scratch
        self.rho: Any = be.empty((ny, nx))
        self.u: Any = be.empty((2, ny, nx))
        self.feq: Any = be.empty((Q, ny, nx))
        self.work: Any = be.empty((3, ny, nx))
        self.out_prev: Any = be.empty((Q, ny))
        self.inlet_work: Any = be.empty((5, ny))

        # Probe buffers stay on the **host**: vorticity, the force integral and
        # the residual all read host arrays at frame or probe cadence and never
        # at step cadence (constraint 8 — never block the sim on the display).
        self.u_prev: NDArray[np.float32] = np.zeros((2, ny, nx), dtype=np.float32)
        self.omega: NDArray[np.float32] = np.empty((ny, nx), dtype=np.float32)
        self.vort_work: NDArray[np.float32] = np.empty((ny, nx), dtype=np.float32)
        self.res_work: NDArray[np.float32] = np.empty((2, ny, nx), dtype=np.float32)

        # True when the backend's arrays *are* host arrays, in which case every
        # host accessor below is free. Asked of the allocation rather than of
        # the backend's name, so a future host backend inherits it.
        self._host_state: bool = isinstance(self.f, np.ndarray)
        self._mirrors: dict[str, NDArray[np.float32]] = {}

        # Built once from the mask, reused every step (D-020, T005 contract).
        # Host: the force integral is a probe, not a kernel.
        self.links: BoundaryLinks = boundary_links(solid)

        # The mask the kernels index, on the backend's side of the seam. On
        # ``"numpy"`` this is a copy of ``self.solid`` and costs ``ny * nx``
        # bytes; on a device it is the ``uint8`` mask the reflection reads.
        self._solid_dev: Any = be.upload(self.solid)

        # The inlet profile is built once and handed back every step; building
        # it per step would allocate O(ny) inside the loop.
        self.u_in: NDArray[np.float32] = inlet_profile(
            ny,
            cfg.inlet_U,
            cfg.profile,
            solid=solid,
            col=cfg.inlet_col,
            uy=cfg.inlet_uy,
        )
        self._u_in_dev: Any = be.upload(self.u_in)

        self.D: float = float(cfg.D) if cfg.D is not None else self._derive_D()

        self._g: tuple[float, float] = (float(cfg.g[0]), float(cfg.g[1]))
        self._forced: bool = self._g != (0.0, 0.0)

        # The inlet's fluid-row mask, built once. Without it
        # lbm.boundary.inlet_velocity evaluates `~solid[:, col]` every step —
        # the last allocation inside the step loop (session 6's note, closed by
        # T010's preallocation audit).
        self._inlet_fluid: NDArray[np.bool_] = np.ascontiguousarray(
            ~self.solid[:, cfg.inlet_col]
        )
        self._inlet_fluid_dev: Any = be.upload(self._inlet_fluid)

        # Fusion is a speed switch, never a physics one (T010). It is skipped
        # when a body force is present: Guo's source term goes between collision
        # and bounce-back, and the only forced case is Rung 1's 22x16 channel.
        self._fused: bool = bool(cfg.fused) and not self._forced

        # Solid cells cost nothing to skip only if there are any at all; with an
        # empty domain the reflection is a no-op and the mask need not be read.
        self._has_solid: bool = bool(self.solid.any())

        # The Smagorinsky closure (T201). Its two buffers are allocated **only
        # when it is on**, so a run with ``cs_smag = 0`` — every rung below
        # Phase 2, and every default — allocates exactly the arrays Phase 1
        # allocated and no others. ``tests/test_runner.py`` counts them.
        self._cs_smag: float = float(cfg.cs_smag)
        if self._cs_smag < 0.0:
            raise ValueError(
                f"cs_smag must be non-negative (got {cfg.cs_smag!r}): the "
                f"closure adds eddy viscosity and never removes it "
                f"(CLAUDE.md constraint 2)."
            )
        self.smag_out: Any = None
        self.smag_work: Any = None
        if self._cs_smag != 0.0:
            self.smag_out = be.empty((ny, nx))
            self.smag_work = be.empty((4, ny, nx))

        if f is None:
            self._init_equilibrium()
        else:
            f = np.asarray(f)
            if f.shape != (Q, ny, nx):
                raise ValueError(
                    f"f must be {(Q, ny, nx)} to match the config (got {f.shape})."
                )
            # Through the seam: an adopted state is always the host layout
            # ``(9, ny, nx)`` float32 (constraint 4 in its D-046 form), and the
            # backend is what turns it into whatever it runs on. No arithmetic
            # happens on either side, so the bits are the checkpoint's bits and
            # constraint 11 holds.
            be.upload(np.ascontiguousarray(f, dtype=np.float32), dst=self.f)

        # The convective outlet's previous column. At the end of every step it
        # equals f[:, :, outlet_col] exactly (nothing later writes that column),
        # so seeding it from f here makes a fresh run and a resumed run agree
        # bit-for-bit — see the module docstring.
        be.upload(
            np.ascontiguousarray(be.download(self.f)[:, :, cfg.outlet_col]),
            dst=self.out_prev,
        )

        self._warn_if_too_fast()

    # -- setup helpers ----------------------------------------------------

    def _derive_D(self) -> float:
        """Characteristic length from the mask bounding box (D-019).

        The cross-stream extent of the immersed object's bounding box — the
        same ``D`` :func:`lbm.geometry.check_mask` uses for the blockage and
        downstream rules. Inventing a second definition is how a 10% error in
        ``Cd`` gets blamed on the solver.
        """
        from lbm.geometry import strip_solid_border

        inner = strip_solid_border(self.solid)
        box = bounding_box(inner)
        if box is None:
            return 1.0
        y0, y1, x0, x1 = box
        if self.config.inlet_axis == "x":
            return float(y1 - y0 + 1)
        return float(x1 - x0 + 1)

    def _init_equilibrium(self) -> None:
        """Seed ``f`` with the equilibrium of ``rho0`` and the inlet profile.

        A uniform-density field at the inlet velocity everywhere: the cheapest
        start that does not put a pressure shock in the domain on step 1.

        Built on the **host** and uploaded once, because a device backend has no
        ``fill`` and this runs at setup rather than in the step loop.
        """
        cfg = self.config
        ny, nx = cfg.ny, cfg.nx

        rho = np.full((ny, nx), np.float32(cfg.rho0), dtype=np.float32)
        u = np.zeros((2, ny, nx), dtype=np.float32)
        if cfg.use_inlet:
            # Broadcast the inlet column's profile across every column.
            u[0] = self.u_in[0][:, None]
            u[1] = self.u_in[1][:, None]

        self.backend.upload(rho, dst=self.rho)
        self.backend.upload(u, dst=self.u)
        self.backend.equilibrium(self.rho, self.u, self.f, self.work)
        np.copyto(self.u_prev, u)

    def _warn_if_too_fast(self) -> None:
        """Warn at setup, not at ``nan`` time (``CLAUDE.md`` constraint 3).

        Compressibility error scales as Mach squared, so any config path that
        can produce ``|u| >= 0.1`` has to say so before the run, not after.
        """
        peak = float(np.abs(self.u_in).max()) if self.u_in.size else 0.0
        if peak >= U_MAX:
            warnings.warn(
                f"inlet peak lattice velocity {peak:.4f} >= {U_MAX} "
                f"(CLAUDE.md constraint 3): compressibility error scales as "
                f"Mach squared. Lower inlet_U or refine the grid.",
                stacklevel=3,
            )

    # -- the timestep -----------------------------------------------------

    def step(self) -> None:
        """One full timestep, in place, allocating nothing.

        The order is ``old-Docs/STATE1.md`` **D-020** — see the module docstring for
        the annotated listing and ``lbm/boundary.py`` for the reasoning behind
        each position. Nothing here allocates: every array it touches was
        created in :meth:`__init__`, which is what
        ``tests/test_runner.py::test_step_allocates_nothing`` asserts with
        ``tracemalloc`` and the buffer identity of ``f``.

        Every kernel is reached through :attr:`Sim.backend` and never imported
        directly (T101): the order below is physics and is the same on every
        backend, while the arithmetic inside each call is the backend's. On
        ``"numpy"`` each call is a delegation to the :mod:`lbm.core` function of
        the same name, which is why the seam cannot move a bit.

        Collision, the reflection, the ``f_bb`` snapshot and the shift are one
        fused pass per direction (``backend.collide_stream``, T010) unless
        ``config.fused`` is off or a body force is present. The fused and
        unfused paths are **bitwise** equal — ``tests/test_perf.py`` asserts it
        — so the switch cannot change a rung's answer or break constraint 11's
        bit-identical restart.
        """
        cfg = self.config
        f = self.f

        backend = self.backend

        if self._fused:
            # No pre-collision copy on this path, and that is not a shortcut.
            # The fused pass stages every direction in ``f_bb`` (D-020) and does
            # not write ``f`` until the stream lands, so ``f`` *is* still the
            # pre-collision state when the reflection reads it — passing ``f``
            # where D-011's copy would go is bitwise identical and removes a
            # whole ``(9, ny, nx)`` copy per step. It is only valid because
            # ``f_bb`` is supplied: with ``f_bb=None`` the pass stages in ``f``
            # itself and the alias would read values it had already overwritten.
            backend.macroscopic(f, self.rho, self.u)
            backend.equilibrium(self.rho, self.u, self.feq, self.work)
            backend.collide_stream(
                f,
                self.feq,
                cfg.tau,
                self.buf,
                f_pre=f if self._has_solid else None,
                solid=self._solid_dev if self._has_solid else None,
                f_bb=self.f_bb,
                # Off unless the config turned it on, and off is a branch inside
                # the kernel rather than a zero-valued term (constraint 19).
                cs_smag=self._cs_smag,
                smag_out=self.smag_out,
                smag_work=self.smag_work,
            )
        else:
            backend.copy(self.f_pre, f)  # pre-collision copy (D-011)
            backend.macroscopic(f, self.rho, self.u)

            if self._forced:
                backend.force_velocity_shift(self.rho, self.u, self._g, self.work)

            backend.equilibrium(self.rho, self.u, self.feq, self.work)
            backend.collide(
                f,
                self.feq,
                cfg.tau,
                cs_smag=self._cs_smag,
                smag_out=self.smag_out,
                smag_work=self.smag_work,
            )

            if self._forced:
                backend.apply_body_force(
                    f, self.rho, self.u, cfg.tau, self._g, self.work
                )

            backend.bounce_back(f, self.f_pre, self._solid_dev)
            backend.copy(self.f_bb, f)  # pre-stream copy, probe.forces (D-020)
            backend.stream(f, self.buf)

        if cfg.use_outlet:
            backend.outlet_zero_gradient(
                f,
                col=cfg.outlet_col,
                src=cfg.outlet_src,
                prev=self.out_prev if cfg.convective_outlet else None,
                lam=cfg.outlet_lam,
            )
        if cfg.use_inlet:
            backend.inlet_velocity(
                f,
                col=cfg.inlet_col,
                u_in=self._u_in_dev,
                work=self.inlet_work,
                fluid=self._inlet_fluid_dev,
            )

        self.step_count += 1

    def run_steps(self, n: int) -> None:
        """Advance ``n`` timesteps."""
        for _ in range(n):
            self.step()

    # -- host views of backend state (T103) -------------------------------

    def _host(self, key: str, dev: Any, shape: tuple[int, ...]) -> NDArray[np.float32]:
        """A host copy of one backend array, into a mirror allocated once.

        On a host backend this is the array itself and costs nothing. On a device
        backend it is a synchronising download into a preallocated mirror, so a
        caller that reads the state every frame allocates nothing per frame
        (``CLAUDE.md`` § conventions) and the transfer happens on the **frame**
        cadence rather than the step cadence (constraint 8).

        Args:
            key: mirror name, unique per array.
            dev: the backend array to read.
            shape: its shape, for the mirror's first allocation.

        Returns:
            Host ``float32`` of ``shape``. Treat it as read-only: on a host
            backend it *is* the simulation's buffer.
        """
        if self._host_state:
            return dev
        mirror = self._mirrors.get(key)
        if mirror is None:
            mirror = np.empty(shape, dtype=np.float32)
            self._mirrors[key] = mirror
        self.backend.download(dev, mirror)
        return mirror

    def host_f(self) -> NDArray[np.float32]:
        """The distribution on the host, ``(9, ny, nx)`` ``float32``."""
        return self._host("f", self.f, (Q, self.config.ny, self.config.nx))

    def host_f_bb(self) -> NDArray[np.float32]:
        """The pre-stream snapshot on the host, ``(9, ny, nx)`` (**D-020**)."""
        return self._host("f_bb", self.f_bb, (Q, self.config.ny, self.config.nx))

    def host_rho(self) -> NDArray[np.float32]:
        """Density on the host, ``(ny, nx)`` ``float32``."""
        return self._host("rho", self.rho, (self.config.ny, self.config.nx))

    def host_u(self) -> NDArray[np.float32]:
        """Velocity on the host, ``(2, ny, nx)`` ``float32``, ``(ux, uy)``."""
        return self._host("u", self.u, (2, self.config.ny, self.config.nx))

    def load_f(self, f: NDArray[np.float32]) -> None:
        """Overwrite the distribution from a host ``(9, ny, nx)`` ``float32``.

        The write half of :meth:`host_f`, for a caller that has to seed or repair
        the state at setup — ``validate/polygons.py`` fills the solid interior
        with ``w_i rho0`` so the first bounce-back has something sane to reflect.
        The convective outlet's previous column is reseeded from the new ``f``,
        exactly as :func:`load_checkpoint` does, so the run stays consistent.

        Args:
            f: ``(9, ny, nx)`` ``float32`` in host memory.

        Raises:
            ValueError: if the shape does not match the config.
        """
        cfg = self.config
        arr = np.ascontiguousarray(f, dtype=np.float32)
        if arr.shape != (Q, cfg.ny, cfg.nx):
            raise ValueError(
                f"f must be {(Q, cfg.ny, cfg.nx)} to match the config "
                f"(got {arr.shape})."
            )
        self.backend.upload(arr, dst=self.f)
        self.backend.upload(
            np.ascontiguousarray(arr[:, :, cfg.outlet_col]), dst=self.out_prev
        )

    def refresh_inlet_profile(self) -> None:
        """Re-upload :attr:`u_in` after a caller has edited it in place.

        :attr:`u_in` ``(2, ny)`` ``float32`` is the host copy; the inlet kernel
        reads the backend's. On ``"numpy"`` they are the same bytes and this is a
        no-op in effect, but a device backend needs telling — which is why the
        demo's startup kick (``DEMO_KICK_FACTOR``) calls it when it zeroes the
        cross-stream component.
        """
        self.backend.upload(self.u_in, dst=self._u_in_dev)

    # -- diagnostics (buffer owners, not new physics) ---------------------

    def vorticity(self) -> NDArray[np.float32]:
        """Vorticity of the current velocity field, into the owned buffer.

        ``DOCS/IDEA2.md`` § What to actually draw — vorticity, not speed
        (constraint 9). The field is ``self.u``, which :meth:`step` refreshed;
        call :meth:`step` at least once first on a resumed sim.

        Returns:
            ``(ny, nx)`` ``float32``, **the runner's buffer** — a caller that
            keeps the frame must copy it.
        """
        return vorticity(
            self.host_u(), solid=self.solid, out=self.omega, work=self.vort_work
        )

    def forces(self) -> tuple[float, float]:
        """``(Cd, Cl)`` by momentum exchange over the precomputed links.

        Uses the two snapshots D-020 fixed: ``f_bb`` (pre-stream) and ``f``
        (post-stream), both from the most recent :meth:`step`.
        """
        return forces(
            self.host_f_bb(),
            self.host_f(),
            self.links,
            U=self.config.inlet_U,
            D=self.D,
            rho0=self.config.rho0,
        )

    def residual(self, U: float | None = None) -> float:
        """Velocity change since the last :meth:`mark_residual`, scaled by ``U``.

        Fluid cells only (D-014). ``float32`` puts a floor near ``1.7e-6`` on a
        per-step residual (D-012); to ask for less, mark, run ``k`` steps, and
        divide the answer by ``k``.
        """
        ref = self.config.inlet_U if U is None else U
        if ref == 0.0:
            raise ValueError(
                "residual needs a nonzero reference velocity: pass U= when the "
                "config has no inlet velocity (a body-force channel has none)."
            )
        return residual(
            self.host_u(), self.u_prev, ref, solid=self.solid, work=self.res_work
        )

    def mark_residual(self) -> None:
        """Snapshot the current velocity as the residual's reference."""
        np.copyto(self.u_prev, self.host_u())

    # -- checkpointing ----------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> Path:
        """Pickle the whole state. See :func:`save_checkpoint`."""
        return save_checkpoint(self, path)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"Sim({cfg.ny}x{cfg.nx}, tau={cfg.tau}, step={self.step_count}, "
            f"solid={int(self.solid.sum())} cells)"
        )


# ---------------------------------------------------------------------------
# Checkpoint / restart
# ---------------------------------------------------------------------------


def save_checkpoint(sim: Sim, path: str | Path) -> Path:
    """Pickle exactly ``f``, ``solid``, ``step_count`` and the config.

    ``DOCS/IDEA2.md`` § Restartability: "Long runs die. ``f``, ``mask``, and the
    step count are the entire state — pickle them every N steps. Resume must
    produce a bit-identical continuation." That is ``CLAUDE.md`` constraint 11,
    and it is a **tested** claim
    (``tests/test_runner.py::test_restart_is_bit_identical``).

    Nothing else goes in — no buffers, no derived fields. Everything else is
    either scratch that :meth:`Sim.step` overwrites before reading, or
    recoverable from ``f``: the convective outlet's ``out_prev`` is
    byte-identical to ``f[:, :, outlet_col]`` at the end of every step, so
    :func:`load_checkpoint` rebuilds it (see the module docstring).

    Args:
        sim: the simulation to save.
        path: destination file. Parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        # Written in the **host** layout, ``(9, ny, nx)`` float32, whatever the
        # backend holds internally (constraint 4 in its D-046 form). That is
        # what makes a checkpoint portable across backends (**D-050**); on
        # NumPy ``to_host`` is the identity, so nothing moves and nothing
        # rounds.
        "f": sim.backend.to_host(sim.f),
        "solid": sim.solid,
        "step_count": sim.step_count,
        "config": sim.config,
    }
    with open(path, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_checkpoint(path: str | Path, backend: str | None = None) -> Sim:
    """Rebuild a :class:`Sim` from a checkpoint, ready to continue.

    The continuation is bit-identical to the run that was saved: ``f`` is
    restored byte-for-byte, the buffers it reads before writing are all
    rewritten by :meth:`Sim.step` before use, and ``out_prev`` is reconstructed
    from ``f`` (module docstring). ``float32`` throughout and no RNG in the step
    path mean there is nothing left to diverge.

    Which backend it resumes on (**D-050**)
    ---------------------------------------
    The checkpoint's contents are unchanged by T101 — still exactly ``f``,
    ``solid``, ``step_count``, the config and a ``format`` integer (**D-022**).
    The backend name rides *inside* the config, because it is configuration and
    not state, and ``f`` is stored in the portable host layout, so a checkpoint
    written on one backend is loadable on another. ``backend=`` overrides the
    saved name for exactly that case; omitting it resumes where the run left
    off. A checkpoint written before T101 has no ``backend`` field at all and
    picks up the dataclass default, ``"numpy"``, which is the backend it ran on.

    Args:
        path: a file written by :func:`save_checkpoint`.
        backend: resume on this backend instead of the saved one. ``None``
            keeps the config's. Bit-identical continuation is a **within**-
            backend guarantee (constraint 11 in its D-046 form); resuming on a
            different backend is a deliberate act with a tolerance, not a
            promise of identical bits.

    Returns:
        A :class:`Sim` at the saved ``step_count``.

    Raises:
        ValueError: if the pickle is not a checkpoint of a known format, or if
            ``backend`` names one that is not registered.
    """
    with open(Path(path), "rb") as fh:
        state = pickle.load(fh)

    if not isinstance(state, dict) or "format" not in state:
        raise ValueError(f"{path} is not an lbm checkpoint.")
    if state["format"] != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint format {state['format']} is not "
            f"{CHECKPOINT_FORMAT}; this build cannot read it."
        )

    cfg: SimConfig = state["config"]
    if backend is not None:
        cfg = cfg.replace(backend=backend)
    # The mask checks already ran when the original Sim was built; re-running
    # them on load would print or warn a second time for no new information.
    return Sim(
        cfg.replace(check_geometry=False),
        state["solid"],
        f=state["f"],
        step_count=state["step_count"],
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    """What one :func:`run` did.

    Attributes:
        steps: timesteps executed by this call.
        frames: frames produced.
        delivered: frames the sink actually received.
        dropped: frames the ring buffer discarded (display frames only).
        elapsed: wall-clock seconds.
        checkpoints: auto-checkpoints written.
    """

    steps: int = 0
    frames: int = 0
    delivered: int = 0
    dropped: int = 0
    elapsed: float = 0.0
    checkpoints: int = 0

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.elapsed if self.elapsed > 0.0 else float("nan")


def _default_field(sim: Sim) -> NDArray[np.float32]:
    """The frame source: a copy of the vorticity field (constraint 9).

    A copy, because the sink consumes frames asynchronously and the sim's buffer
    is about to be overwritten. One allocation per *frame*, not per step, so the
    "never allocate inside the step loop" rule is intact.
    """
    return sim.vorticity().copy()


def run(
    sim: Sim,
    sink: Sink | None = None,
    *,
    frames: int | None = None,
    steps: int | None = None,
    steps_per_frame: int = 1,
    field: Callable[[Sim], Any] | None = None,
    buffer_size: int = 4,
    drop: bool = True,
    stop: Callable[[Sim], bool] | None = None,
    buffer: RingBuffer | None = None,
    per_step: Callable[[Sim], None] | None = None,
) -> RunStats:
    """Run the simulation, feeding frames to a sink through a ring buffer.

    ``DOCS/IDEA2.md`` § Decouple simulation from rendering::

        while running:
            for _ in range(steps_per_frame):
                step()
            frame = render(field())
            sink.push(frame)

    with the ring buffer of § Never block the sim on the display between the
    last two lines. ``steps_per_frame`` is a **computed** number — see
    :func:`steps_per_frame` (constraint 7); passing a literal 20 here is what
    that function exists to prevent.

    Two modes, which are the two halves of ``DOCS/IDEA2.md`` § Three output
    sinks:

    ``drop=True`` (**live**)
        A consumer thread drains the buffer into the sink. The sim thread only
        pushes, so a slow sink can never stall the physics; the buffer fills and
        the **oldest display frames** are dropped (constraint 8). Only the
        display is threaded — the physics is a single loop and untouched
        (constraint 6).
    ``drop=False`` (**record**, T011)
        Drained inline. The sink sees every frame in order and the sim waits for
        it, which is what a fixed-framerate video writer needs.

    Args:
        sim: the simulation to advance.
        sink: destination for frames. ``None`` means :class:`NullSink`.
        frames: number of frames to produce. ``None`` runs until ``steps`` or
            ``stop``.
        steps: total timesteps to run. Rounded up to a whole frame.
        steps_per_frame: timesteps between frames, from :func:`steps_per_frame`.
        field: ``sim -> frame``. Defaults to a copy of the vorticity field.
        buffer_size: ring buffer capacity in frames.
        drop: see above.
        stop: optional predicate; the run ends when it returns ``True``.
        buffer: an existing :class:`RingBuffer` to use, so a caller can inspect
            its counters afterwards.
        per_step: optional probe called with ``sim`` after **every** timestep,
            on the physics thread. A time series that has to be sampled at the
            step rate — the ``Cl`` history Rung 3 hands to
            :func:`lbm.probe.strouhal` — cannot be sampled at the *frame* rate
            through ``field``: one frame is many timesteps and the shedding
            period is only a couple of thousand steps, so frame-rate sampling
            aliases. This hook exists so that measurement does not need a
            second copy of the loop (``old-Docs/STATE1.md`` **D-025**). It must not
            render or touch a sink — that is what the ring buffer is for.

    Returns:
        A :class:`RunStats`.

    Raises:
        ValueError: if neither ``frames`` nor ``steps`` nor ``stop`` bounds the
            run, or ``steps_per_frame < 1``.
    """
    if steps_per_frame < 1:
        raise ValueError(f"steps_per_frame must be at least 1 (got {steps_per_frame!r}).")
    if frames is None and steps is None and stop is None:
        raise ValueError(
            "run needs a bound: pass frames=, steps= or stop=. An unbounded "
            "loop belongs to the caller, not here."
        )

    if frames is None and steps is not None:
        frames = -(-steps // steps_per_frame)  # ceil

    own_sink = sink is None
    sink = NullSink() if sink is None else sink
    field = _default_field if field is None else field
    ring = RingBuffer(buffer_size) if buffer is None else buffer

    stats = RunStats()
    start = time.perf_counter()

    done = threading.Event()
    consumer: threading.Thread | None = None

    def _drain_forever() -> None:
        while True:
            item = ring.pop()
            if item is None:
                if done.is_set():
                    return
                time.sleep(0.0005)
                continue
            sink.push(item)
            stats.delivered += 1

    if drop:
        consumer = threading.Thread(target=_drain_forever, name="lbm-sink", daemon=True)
        consumer.start()

    every = sim.config.checkpoint_every
    ckpt_path = sim.config.checkpoint_path

    try:
        n = 0
        while frames is None or n < frames:
            for _ in range(steps_per_frame):
                sim.step()
                stats.steps += 1
                if per_step is not None:
                    per_step(sim)
                if every > 0 and ckpt_path and sim.step_count % every == 0:
                    save_checkpoint(sim, ckpt_path)
                    stats.checkpoints += 1

            ring.push(field(sim))
            stats.frames += 1
            n += 1

            if not drop:
                item = ring.pop()
                while item is not None:
                    sink.push(item)
                    stats.delivered += 1
                    item = ring.pop()

            if stop is not None and stop(sim):
                break
    finally:
        done.set()
        if consumer is not None:
            consumer.join()
        # Anything still queued in record mode goes out; in live mode the
        # consumer has already drained what survived.
        item = ring.pop()
        while item is not None:
            sink.push(item)
            stats.delivered += 1
            item = ring.pop()
        stats.dropped = ring.dropped
        stats.elapsed = time.perf_counter() - start
        if own_sink:
            sink.close()

    return stats


# ---------------------------------------------------------------------------
# The command line — `python -m lbm.runner`
# ---------------------------------------------------------------------------
#
# T011, ``old-Docs/TASKS1.md``: "one command that takes a PNG plus physical numbers
# and produces an MP4. M4 — the first thing another person can use."
#
# Everything below is the *boundary* of the package in the sense of
# ``CLAUDE.md`` § Coding conventions: metres, seconds and m/s appear here and in
# ``lbm/units.py`` and nowhere else. The conversion is
# :meth:`lbm.units.LatticeUnits.from_physical` and the number that reaches the
# solver is always lattice. Imports of ``lbm.units``, ``lbm.geometry``,
# ``lbm.render`` and ``lbm.record`` are inside the functions, so ``import
# lbm.runner`` stays as cheap and as headless as it was before T011.

#: Kinematic viscosity in m^2/s for the fluids ``--fluid`` names. Ordinary
#: reference values at 20 C; anything else is ``--nu`` or ``--re``.
FLUIDS: dict[str, float] = {
    "air": 1.5e-5,
    "water": 1.0e-6,
    "honey": 2.0e-3,
}

#: Domain around the body, in measured body diameters. The blockage that
#: results is ``1 / span_d`` — 8.3% at 12, inside constraint 12's 10% — and the
#: downstream fetch is over the 8 D the same constraint asks for. Rung 3 uses a
#: 24 D span because it is comparing a *number* against an unconfined reference
#: (D-026); a demo is comparing a picture against the eye, and 12 D halves the
#: cell count for a wake that looks the same.
DEMO_SPAN_D: float = 12.0
DEMO_UPSTREAM_D: float = 6.0
DEMO_DOWNSTREAM_D: float = 10.0

#: Startup kick, as in ``validate/cylinder.py``: a cross-stream inlet velocity
#: of ``KICK_FACTOR * U`` for the first ``KICK_TC`` convective times, then zero.
#: A symmetric body on a symmetric grid stays symmetric far longer than physics
#: would, and a five-second clip has no time to spare — Rung 3 measures shedding
#: established by ~70 D/U *with* a 10% kick. The demo kicks harder because it is
#: making a picture, not a measurement, and the kick is off long before the end.
DEMO_KICK_FACTOR: float = 0.20
DEMO_KICK_TC: float = 5.0

#: Colour limits as a multiple of ``U / D``, the natural vorticity scale of the
#: wake (constraint 9: fixed and symmetric, never per-frame).
DEMO_VMAX_FACTOR: float = 4.0

#: One line pointing at the Phase 1 CLI, printed by :func:`main` and repeated in
#: ``--help``. **D-072** decided ``Q-101``: this entry point is *kept, working
#: and tested*, rather than reduced to a pointer, so the M4 gate in
#: ``old-Docs/STATE1.md`` § Snapshot stays literally reproducible and the
#: solver-level knobs stay reachable. Delegating was never available: ``flow/``
#: may import ``lbm/`` and ``lbm/`` may **never** import ``flow/`` (constraint
#: 15), so this is a string and not a call.
PHASE1_CLI_POINTER: str = (
    "note: `python -m flow` is the Phase 1 command and is the one to reach for "
    "-- a picture,\n  a fluid, a speed and a size, with every lattice number "
    "derived and printed. This one is\n  kept for the solver-level knobs it "
    "has and `flow` deliberately does not: --re / --nu,\n  --resolution in "
    "cells, --span-d / --upstream-d / --downstream-d, --u-lattice,\n  "
    "--tau-floor and --checkpoint (D-072)."
)


def _body_mask(
    path: str | Path, cells: int, *, invert: bool = False, verbose: bool = True
) -> tuple[NDArray[np.bool_], int, int]:
    """Load a geometry file and crop it to the body's bounding box.

    The mask a demo needs is the *body*, not a picture-shaped domain:
    :func:`lbm.geometry.from_png` fills the grid it is given, so asking it for
    the whole domain would stretch the body across the inlet and the outlet.
    The body is loaded at its own scale here and placed into a domain by
    :func:`demo_domain`.

    ``check=False`` on this call and **not** because the check is unwanted —
    constraint 12 is checked on the assembled domain instead, where blockage and
    downstream distance mean something and where :class:`Sim` runs
    :func:`lbm.geometry.check_mask` itself. On a grid cropped to the body the
    blockage is 100% by construction and the report would be noise; the
    thickness rule, the one that catches a hairline in a downscaled PNG (D-031),
    is unaffected by where the body sits and still fires on the domain.

    Args:
        path: ``.png`` (or any Pillow format) or ``.svg``.
        cells: target cross-stream size in cells — the resolution the physical
            units are derived at. The *measured* extent that comes back may be
            a cell or two less, and that measured number is what everything
            downstream uses (D-019).
        invert: swap the solid/fluid sense of the image. Applied by the loader,
            before the crop: inverting a mask that has already been cropped to
            its own bounding box would produce an empty body.
        verbose: print what was loaded.

    Returns:
        ``(body, bh, bw)`` — the cropped mask and its shape.
    """
    from lbm.geometry import bounding_box, from_png, from_svg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"geometry file not found: {path}")

    aspect = 1.0
    if path.suffix.lower() != ".svg":
        from PIL import Image

        with Image.open(path) as im:
            w_img, h_img = im.size
        aspect = w_img / h_img

    def load(box_rows: int) -> NDArray[np.bool_] | None:
        """Rasterise into a ``box_rows``-tall box and crop to the body."""
        rows = max(3, int(box_rows))
        cols = max(3, int(round(rows * aspect)))
        if path.suffix.lower() == ".svg":
            mask = from_svg(path, (rows, cols), check=False, verbose=False)
            if invert:
                mask = ~mask
        else:
            mask = from_png(
                path, (rows, cols), invert=invert, check=False, verbose=False
            )
        box = bounding_box(mask)
        if box is None:
            return None
        y0, y1, x0, x1 = box
        return np.ascontiguousarray(mask[y0 : y1 + 1, x0 : x1 + 1])

    # `cells` is the size of the **body**, not of the picture, because it is
    # what `LatticeUnits.from_physical` is handed as `cells_per_length` and
    # therefore what `tau = 0.5 + 3 U N / Re` is computed from (D-019). A PNG
    # with a wide margin would otherwise quietly run at a fraction of the
    # requested resolution — and `tau` with it, straight into D-029's measured
    # blow-up band. So: rasterise, measure the body, rescale the box by the
    # shortfall, repeat. Two passes are enough for any margin; the loop stops
    # early when the body is already big enough.
    box_rows = int(cells)
    body = load(box_rows)
    for _ in range(3):
        if body is None:
            break
        if body.shape[0] >= cells:
            break
        box_rows = int(np.ceil(box_rows * cells / body.shape[0]))
        body = load(box_rows)

    if body is None:
        raise ValueError(
            f"{path} produced an empty mask at {cells} cells across: nothing "
            f"was solid after thresholding. Try --invert, or a larger "
            f"--resolution."
        )
    if verbose:
        print(
            f"from {path.name}: body {body.shape[0]} x {body.shape[1]} cells "
            f"({int(body.sum())} solid) rasterised in a {box_rows}-row box"
        )
    return body, body.shape[0], body.shape[1]


def demo_domain(
    body: NDArray[np.bool_],
    *,
    span_d: float = DEMO_SPAN_D,
    upstream_d: float = DEMO_UPSTREAM_D,
    downstream_d: float = DEMO_DOWNSTREAM_D,
    offset: int = 1,
) -> tuple[NDArray[np.bool_], int]:
    """Place a body in an open channel sized in its own diameters.

    ``CLAUDE.md`` constraint 12: object at least 8 diameters from the outlet,
    blockage under ~10%. Both follow from the two arguments — the blockage is
    ``1 / span_d`` exactly, because ``D`` is the body's cross-stream extent and
    the lateral boundaries are periodic (D-026), so there are no wall rows to
    take out of the denominator.

    Args:
        body: the cropped body mask, ``(bh, bw)``.
        span_d: cross-stream fluid span, in diameters.
        upstream_d: inlet-to-leading-edge distance, in diameters.
        downstream_d: trailing-edge-to-outlet distance, in diameters.
        offset: cross-stream displacement of the body from the centre line, in
            cells. One cell breaks the grid's mirror symmetry, which is half of
            why shedding starts; the startup kick is the other half.

    Returns:
        ``(solid, d_measured)`` — the domain mask ``(ny, nx)`` and the body's
        cross-stream extent in cells (the ``D`` of D-019).
    """
    bh, bw = body.shape
    d = float(bh)
    ny = int(round(span_d * d))
    nx = int(round((upstream_d + downstream_d) * d)) + bw
    if ny <= bh + 2 or nx <= bw + 2:
        raise ValueError(
            f"domain {ny}x{nx} is not larger than the body {bh}x{bw}: raise "
            f"--span-d / --upstream-d / --downstream-d."
        )

    solid = np.zeros((ny, nx), dtype=bool)
    y0 = (ny - bh) // 2 + offset
    x0 = int(round(upstream_d * d))
    solid[y0 : y0 + bh, x0 : x0 + bw] = body
    return solid, bh


def _build_parser() -> Any:
    """The ``python -m lbm.runner`` argument parser."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m lbm.runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run a 2D lattice-Boltzmann flow past a shape and watch it, record "
            "it, or both.\n\n"
            "The geometry is a picture, the flow is described in physical "
            "units, and the lattice numbers (dx, dt, tau, U) are derived by "
            "lbm.units.LatticeUnits.from_physical — which refuses a case it "
            "cannot represent instead of running it badly."
        ),
        epilog=(
            "examples:\n"
            "  python -m lbm.runner --geometry tests/data/test_body.png "
            "--re 100 --velocity 20 --length 1.5 --seconds 5 --out wake.mp4\n"
            "  python -m lbm.runner --demo cylinder --re 100 --velocity 20 "
            "--length 1.5 --seconds 3 --live\n"
            "  python -m lbm.runner --demo cylinder --re 100 --velocity 20 "
            "--length 1.5 --seconds 2 --out clip.gif --live\n"
            "\n"
            "note: --fluid air at 20 m/s past a 1.5 m body is Re 2e6, which "
            "BGK with bounce-back\n"
            "and no turbulence model cannot resolve; the run is refused with "
            "the resolution it\n"
            "would need. --re describes a case this solver can actually "
            "represent.\n"
            "\n" + PHASE1_CLI_POINTER + "\n"
        ),
    )

    g = p.add_argument_group("geometry")
    g.add_argument("--geometry", metavar="PATH", help="PNG or SVG of the body")
    g.add_argument(
        "--demo",
        choices=("cylinder",),
        help="a built-in body instead of --geometry",
    )
    g.add_argument(
        "--resolution",
        type=int,
        default=30,
        metavar="N",
        help="cells across the body (default 30). Raises tau: "
        "tau = 0.5 + 3 U N / Re",
    )
    g.add_argument("--invert", action="store_true", help="swap solid and fluid")
    g.add_argument("--span-d", type=float, default=DEMO_SPAN_D, metavar="D")
    g.add_argument("--upstream-d", type=float, default=DEMO_UPSTREAM_D, metavar="D")
    g.add_argument("--downstream-d", type=float, default=DEMO_DOWNSTREAM_D, metavar="D")

    ph = p.add_argument_group("physics (metres, seconds, m/s)")
    fluid = ph.add_mutually_exclusive_group()
    fluid.add_argument("--fluid", choices=sorted(FLUIDS), help="named fluid")
    fluid.add_argument("--nu", type=float, metavar="M2/S", help="kinematic viscosity")
    fluid.add_argument("--re", type=float, metavar="RE", help="Reynolds number")
    ph.add_argument("--velocity", type=float, required=True, metavar="M/S")
    ph.add_argument("--length", type=float, required=True, metavar="M",
                    help="cross-stream size of the body")
    ph.add_argument("--seconds", type=float, required=True, metavar="S",
                    help="physical time to simulate")
    ph.add_argument("--u-lattice", type=float, default=None, metavar="U",
                    help="lattice velocity representing --velocity "
                         "(default 0.05, ceiling 0.1 — constraint 3)")

    o = p.add_argument_group("output")
    o.add_argument("--out", metavar="PATH", help=".mp4 / .gif — implies --record")
    o.add_argument("--frames-dir", metavar="DIR",
                   help="directory of numbered PNGs — implies --headless")
    o.add_argument("--live", action="store_true", help="pygame window")
    o.add_argument("--record", action="store_true", help="write --out")
    o.add_argument("--headless", action="store_true", help="write --frames-dir")
    o.add_argument("--fps", type=float, default=60.0)
    o.add_argument("--speed", type=float, default=1.0,
                   help="playback rate: 1.0 real time, 10.0 ten times faster")
    o.add_argument("--scale", type=int, default=1, help="live window magnification")
    o.add_argument("--vmax", type=float, default=None,
                   help="colour limit; default 4 U / D (symmetric, fixed)")
    o.add_argument("--quality", type=float, default=8.0, help="MP4 quality 0-10")

    r = p.add_argument_group("run")
    r.add_argument("--checkpoint", metavar="PATH")
    r.add_argument("--checkpoint-every", type=int, default=0, metavar="N")
    r.add_argument("--tau-floor", type=float, default=None,
                   help="override lbm.units' 0.51 floor (it is already the "
                        "loosest floor in the project — D-032)")
    r.add_argument("--quiet", action="store_true")
    return p


def _resolve_sinks(args: Any) -> tuple[Sink, list[Sink], bool]:
    """Build the sinks the flags ask for.

    ``--live``, ``--record`` and ``--headless`` are composable
    (``old-Docs/TASKS1.md`` § T011), which is what :class:`lbm.record.TeeSink` is
    for. The *mode* is not composable and must not be: **D-024** allows exactly
    two, and any sink that writes a **file** picks ``drop=False`` — a video with
    a missing frame and a PNG series with a gap in the numbering are both wrong
    output rather than slow output, and the sim is allowed to wait for a writer.
    ``drop=True`` is therefore reached only by a live-only run, which is exactly
    the case constraint 8 describes.

    Returns:
        ``(sink, members, drop)``.
    """
    from lbm.record import HeadlessSink, RecordSink, TeeSink
    from lbm.render import LiveSink

    want_record = args.record or bool(args.out)
    want_headless = args.headless or bool(args.frames_dir)
    want_live = args.live or not (want_record or want_headless)

    if want_record and not args.out:
        raise ValueError("--record needs --out PATH (.mp4 or .gif)")
    if want_headless and not args.frames_dir:
        args.frames_dir = "frames"

    members: list[Sink] = []
    if want_live:
        members.append(LiveSink(scale=args.scale, title="lbm — vorticity"))
    if want_record:
        members.append(RecordSink(args.out, fps=args.fps, quality=args.quality))
    if want_headless:
        members.append(HeadlessSink(args.frames_dir))

    sink: Sink = members[0] if len(members) == 1 else TeeSink(*members)
    return sink, members, not (want_record or want_headless)


def main(argv: list[str] | None = None) -> int:
    """``python -m lbm.runner`` — picture in, physical numbers in, video out.

    The whole of ``old-Docs/PLAN1.md`` § Milestone gates for **M4**: "An arbitrary
    PNG becomes a mask, runs in physical units, and records an MP4 — end to end,
    one command."

    What it does, in order: load the geometry and crop it to the body; place it
    in a domain sized in its own diameters (constraint 12); convert the physical
    case with :meth:`lbm.units.LatticeUnits.from_physical`, which **raises**
    rather than run an unrepresentable case (constraint 3 / 2, D-032); compute
    ``steps_per_frame`` from ``dt`` (constraint 7, D-023); render vorticity with
    fixed symmetric limits (constraint 9, D-028); and feed the one ``render()``
    output to whichever sinks the flags asked for (constraint 10).

    Returns:
        Process exit status: ``0`` on success, ``2`` on a refused configuration
        or a missing tool — with the message, never a traceback.
    """
    from lbm.geometry import circle
    from lbm.record import HeadlessSink, RecordSink, frame_count
    from lbm.render import LiveSink, render
    from lbm.units import LatticeUnits, TAU_FLOOR, U_LATTICE_DEFAULT

    parser = _build_parser()
    args = parser.parse_args(argv)
    say = (lambda *a, **k: None) if args.quiet else print

    if not args.geometry and not args.demo:
        parser.error("give --geometry PATH or --demo cylinder")
    if args.fluid is None and args.nu is None and args.re is None:
        parser.error("describe the fluid: --fluid air | --nu 1.5e-5 | --re 100")

    # D-072 / Q-101: this entry point still works, and still says where the
    # product command is. Printed before anything can refuse, so a user who
    # meets the refusal has already been told about `python -m flow`.
    say(f"\n  {PHASE1_CLI_POINTER}")

    # --- geometry ---------------------------------------------------------
    try:
        if args.demo == "cylinder":
            n = max(6, int(args.resolution))
            body = circle(n + 1, n + 1, n / 2.0, n / 2.0, n / 2.0)
            rows = body.any(axis=1)
            cols = body.any(axis=0)
            body = np.ascontiguousarray(body[rows][:, cols])
        else:
            body, _bh, _bw = _body_mask(
                args.geometry,
                int(args.resolution),
                invert=args.invert,
                verbose=not args.quiet,
            )
        solid, d_measured = demo_domain(
            body,
            span_d=args.span_d,
            upstream_d=args.upstream_d,
            downstream_d=args.downstream_d,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"geometry: {exc}", file=sys.stderr)
        return 2
    ny, nx = solid.shape

    # --- physical -> lattice ---------------------------------------------
    # The one place metres and seconds exist. lbm/units.py raises on a case
    # that would violate constraint 3 or sit on the tau floor, and names the
    # resolution that fixes it (D-032); the CLI prints that message and stops.
    # Softening it here would be the project's stated main failure mode with a
    # command-line switch attached.
    try:
        units = LatticeUnits.from_physical(
            u_phys=args.velocity,
            l_phys=args.length,
            nu_phys=(FLUIDS[args.fluid] if args.fluid else args.nu),
            re=args.re,
            cells_per_length=float(d_measured),
            u_lattice=(U_LATTICE_DEFAULT if args.u_lattice is None else args.u_lattice),
            tau_floor=(TAU_FLOOR if args.tau_floor is None else args.tau_floor),
        )
    except ValueError as exc:
        print(f"\nthis case cannot be simulated as described:\n  {exc}",
              file=sys.stderr)
        print(
            "\n  Nothing here is a tolerance that can be loosened: nu = "
            "(tau - 0.5) / 3 (CLAUDE.md constraint 2) and the lattice velocity "
            "ceiling is compressibility error (constraint 3).\n"
            "  Raise --resolution to the number above, or describe a "
            "resolvable case with --re.",
            file=sys.stderr,
        )
        return 2

    if args.seconds <= 0.0:
        parser.error("--seconds must be positive")

    u = units.U
    spf = steps_per_frame(units.dt, args.fps, args.speed)
    total_steps = max(1, int(round(args.seconds / units.dt)))
    vmax = args.vmax if args.vmax is not None else DEMO_VMAX_FACTOR * u / d_measured
    t_conv = d_measured / u
    kick_steps = int(round(DEMO_KICK_TC * t_conv))

    say(f"\nlbm — {args.demo or args.geometry}")
    say(units.summary())
    say(f"  domain:   {ny} x {nx} = {ny * nx / 1e3:.0f}k cells   "
        f"D = {d_measured:.0f} cells measured (D-019)   "
        f"blockage {d_measured / ny * 100:.1f}%   sides periodic (D-026)")
    say(f"  time:     {args.seconds:g} s = {total_steps} steps "
        f"= {total_steps / t_conv:.0f} convective times D/U")
    say(f"  frames:   steps_per_frame = round({args.speed:g} / "
        f"({args.fps:g} * dt)) = {spf} (constraint 7, D-023)   "
        f"-> {-(-total_steps // spf)} frames at {args.fps:g} fps")
    say(f"  colour:   vorticity, +-{vmax:.5f} fixed and symmetric "
        f"(constraint 9, D-028)")
    say("  geometry checks (constraint 12):")

    # --- the sim ----------------------------------------------------------
    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=units.tau,
        inlet_U=u,
        profile="uniform",
        inlet_uy=DEMO_KICK_FACTOR * u,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        inlet_axis="x",
        check_geometry=True,
        verbose_mask=not args.quiet,
        checkpoint_every=int(args.checkpoint_every),
        checkpoint_path=args.checkpoint,
    )
    sim = Sim(cfg, solid)

    # D-030: Sim seeds the whole domain, solid included, with the equilibrium of
    # the inlet profile, so at step 0 there is fluid moving at U *inside* the
    # body and bounce-back reverses it every step rather than clearing it. The
    # rest state is a fixed point of both bounce-back and streaming.
    rest = np.float32(cfg.rho0) * W.astype(np.float32)
    seed = sim.host_f().copy()
    for i in range(Q):
        seed[i][sim.solid] = rest[i]
    sim.load_f(seed)

    try:
        sink, members, drop = _resolve_sinks(args)
    except (ValueError, RuntimeError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    kinds = ", ".join(type(s).__name__ for s in members)
    say(f"  sinks:    {kinds}   mode: "
        f"{'drop=True (live only, display frames may be dropped — constraint 8)' if drop else 'drop=False (a file is being written, every frame in order — D-024)'}")
    say(f"\n  running {total_steps} steps ...", flush=True)

    def probe(s: Sim) -> None:
        """Switch the startup kick off in place (see DEMO_KICK_FACTOR)."""
        if s.step_count == kick_steps:
            s.u_in[1].fill(0.0)
            s.refresh_inlet_profile()

    live = next((s for s in members if isinstance(s, LiveSink)), None)

    stats = run(
        sim,
        sink,
        steps=total_steps,
        steps_per_frame=spf,
        field=lambda s: render(s.vorticity(), vmax),
        drop=drop,
        per_step=probe,
        stop=(lambda _s: bool(getattr(live, "quit_requested", False)))
        if live is not None
        else None,
    )
    sink.close()

    say(f"  done: {stats.steps} steps in {stats.elapsed:.1f} s "
        f"({stats.steps_per_second:.1f} steps/s), {stats.frames} frames, "
        f"{stats.delivered} delivered, {stats.dropped} dropped")

    peak = float(np.abs(sim.host_u()[:, ~sim.solid]).max())
    say(f"  peak |u| {peak:.5f} (limit 0.1, constraint 3)"
        f"{'  ** OVER THE LIMIT **' if peak >= 0.1 else ''}")
    if not np.isfinite(sim.host_f()).all():
        print("  the simulation produced nan — the case was unstable.",
              file=sys.stderr)
        return 1

    for s in members:
        if isinstance(s, RecordSink):
            written = frame_count(s.path)
            size = s.path.stat().st_size
            say(f"  wrote {s.path} — {written} frames at {s.fps:g} fps "
                f"({size / 1e6:.2f} MB); sink pushed {s.frames}")
            if written != s.frames:
                print(f"  frame count mismatch: pushed {s.frames}, file has "
                      f"{written}", file=sys.stderr)
                return 1
        elif isinstance(s, HeadlessSink):
            say(f"  wrote {s.frames} PNGs to {s.directory}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    # ``python -m lbm.runner`` imports the package first (which imports this
    # module) and then executes this file a second time under the name
    # ``__main__``, so there are two copies of every class defined above. The
    # package's copy is the one ``lbm.record`` and ``lbm.render`` subclass
    # ``Sink`` from, so the CLI runs from that copy and every isinstance check
    # in the process compares like with like. runpy's RuntimeWarning about the
    # double import is cosmetic and is the price of the entry point the
    # contract names.
    from lbm.runner import main as _package_main

    raise SystemExit(_package_main())
