"""Rung F — the Smagorinsky closure is switchable, and switching it off is free.

``DOCS/IDEA4.md`` § Validation ladder, Rung F::

    The closure is switchable and does not disturb what already works.
    Cs = 0 reproduces plain BGK **bitwise** on both backends (numpy.array_equal
    on f after 1000 steps); and Rung 3 with the closure **on** still prints
    Cd 1.25-1.45, St 0.155-0.175.

This is the gate for **T201** on numpy and, with T202's ``--backend``, on warp
too — the pair of them being half of **M9** (Rung G is the other half). Run it
from the repo root::

    myenv/Scripts/python.exe -m validate.les
    myenv/Scripts/python.exe -m validate.les --backend warp
    myenv/Scripts/python.exe -m validate.les --steps 200   # a faster smoke

Why the bitwise clause is first, and why it is written this way
---------------------------------------------------------------
``CLAUDE.md`` constraint 19, from **D-081**: *a closure you cannot switch off is
a closure you cannot validate against*, and nine green rungs are what Phase 2
puts at risk. ``DOCS/PLAN3.md`` § Why this order (3) makes it the first thing
T201 writes, before the model does anything.

"Bitwise what Phase 1 shipped" needs a Phase 1 to compare against, and once
``lbm/core.py`` has been edited there isn't one. So :func:`_phase1_collide` and
:func:`_phase1_collide_stream` below are **frozen verbatim transcriptions** of
those two functions as they stood at the end of Phase 1 (session 22, commit
before T201). They are the oracle, they are deliberately duplicated code, and
**they must never be edited** — an edit that makes this rung pass is an edit
that deletes the rung. If a future task genuinely changes the base arithmetic,
that is a constraint 1 violation and the answer is a decision in
``DOCS/STATE3.md``, not a change here.

T202 adds the **second** oracle the same rule governs: the Warp backend's own
Phase 1 kernels, frozen here as :func:`_phase1_collide_kernel` and
:func:`_phase1_collide_bb_kernel` (**D-087**, extended). One copy, in this file,
never edited. A GPU oracle is needed for the same reason the host one is —
comparing the backend against itself proves nothing — and it is a *transcription*
rather than an import so that a later edit to ``lbm/backends/warp_backend.py``
cannot silently move both sides of the comparison at once.

The comparison runs the real :class:`lbm.runner.Sim` on **Rung 3's own case**,
swapping only the two collision kernels through the backend seam (**D-054**),
so everything else about the timestep — the D-020 order, the boundaries, the
``f_bb`` snapshot — is shared between the two runs by construction and cannot
be the thing that agrees.

What the three clauses check
----------------------------
1. **Bitwise degeneracy, both paths.** ``cs_smag = 0`` against the frozen Phase
   1 kernels after ``--steps`` steps, fused and unfused, plus fused against
   unfused — which **D-055** already requires and which the closure must not
   break.
2. **The closure engages, and only ever adds viscosity.** On the same case with
   ``Cs = 0.17``: ``tau_eff >= tau`` in every cell, ``nu_t >= 0`` in every cell,
   and ``max(nu_t) > 0`` so that clause 3 is not passing because the model is
   silently inert. The cylinder wake is the strong-shear case the T201 contract
   asks for. ``max(nu_t) / nu`` is printed because it is the quantity
   **D-082**'s fidelity bands are decided from (T204).
3. **Rung 3 survives the closure.** The full Rung 3 case with ``Cs = 0.17``,
   through :func:`validate.cylinder.run_cylinder` itself, still landing in the
   published bands ``Cd`` 1.25-1.45 and ``St`` 0.155-0.175.

Clause 4, and why it lives here rather than in Rung A
-----------------------------------------------------
``--backend warp`` adds a fourth clause: **cross-backend agreement with the
closure on**, held to the two numbers Rung A already publishes — per-kernel
worst under ``1e-6`` in ``f`` units (**D-053**) and whole-step
``max|Delta u| / U`` under ``1e-4`` at 1000 steps (**D-056**). It runs
:func:`validate.parity.whole_step` itself, with ``cs_smag`` threaded through
:func:`validate.parity.step_case`, so the case and the bars are Rung A's own and
not a second copy of them. Rung A keeps the closure **off**, which is what
constraint 19 says it should measure; the closure-on version of the same
question belongs to the rung that owns the closure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lbm.core import (
    CS_SMAG_LITERATURE,
    E,
    OPP,
    Q,
    _shift_blocks,
    equilibrium,
    macroscopic,
    smagorinsky_tau_eff,
)
from lbm.backends import BackendUnavailableError, get_backend
from lbm.probe import eddy_viscosity
from lbm.runner import Sim
from validate.cylinder import (
    CD_BAND,
    KICK_FACTOR,
    ST_BAND,
    U,
    cylinder_mask,
    make_config,
    report as cylinder_report,
    run_cylinder,
    tau_for,
)

#: Steps of the bitwise comparison. ``DOCS/IDEA4.md`` § Validation ladder names
#: 1000, and it is a *floor* rather than a target: bitwise equality either holds
#: from step 1 or it does not hold at all, and 1000 steps of a shedding wake is
#: simply enough arithmetic that a divergence anywhere in the collision would
#: have shown up as something other than zero by now.
BITWISE_STEPS: int = 1000

#: The Smagorinsky constant clauses 2 and 3 run at — the literature value, from
#: :mod:`lbm.core`, not a second copy (constraint 4's "no physics constant
#: twice"). Phase 2 does not tune it (``DOCS/TASKS3.md`` § T201 Notes).
CS: float = CS_SMAG_LITERATURE


# --- the frozen Phase 1 oracle. DO NOT EDIT. ---------------------------------
#
# Verbatim transcriptions of lbm.core.collide and lbm.core.collide_stream as
# they stood at the end of Phase 1 (M8, session 22), before T201 gave them a
# `cs_smag` keyword. Docstrings trimmed to the reasoning that bears on the
# arithmetic; every operation, its order and its dtype are unchanged.
#
# `_shift_blocks` is imported rather than transcribed: it is pure slicing with
# no arithmetic in it, T201 did not touch it, and copying it here would make
# this file claim to freeze something it does not actually test.


def _phase1_collide(
    f: NDArray[np.float32], feq: NDArray[np.float32], tau: float
) -> None:
    """FROZEN Phase 1 ``lbm.core.collide``. Do not edit.

    ``f -= (f - feq) / tau``, written as three in-place operations so that no
    temporary is allocated, with ``omega = 1 / tau``.
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


def _phase1_collide_stream(
    f: NDArray[np.float32],
    feq: NDArray[np.float32],
    tau: float,
    buf: NDArray[np.float32],
    *,
    f_pre: NDArray[np.float32] | None = None,
    solid: NDArray[np.bool_] | None = None,
    f_bb: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """FROZEN Phase 1 ``lbm.core.collide_stream``. Do not edit.

    Collide, bounce back and stream in one pass per direction (T010, **D-033**,
    **D-055**).
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
        s = f[i] if f_bb is None else f_bb[i]

        np.subtract(f[i], feq[i], out=s)
        s *= one_minus_omega
        s += feq[i]

        if solid is not None:
            np.copyto(s, f_pre[OPP[i]], where=solid)

        ex = int(E[i, 0])
        ey = int(E[i, 1])
        dst = buf[i]
        for dy, sy in _shift_blocks(ey):
            for dx, sx in _shift_blocks(ex):
                dst[dy, dx] = s[sy, sx]

    np.copyto(f, buf)
    return f


class Phase1Backend:
    """The NumPy backend with its two collision kernels frozen at Phase 1.

    Everything except :meth:`collide` and :meth:`collide_stream` is forwarded to
    the real backend, so the two runs being compared share every buffer
    allocation, every boundary condition and the whole D-020 ordering. What
    differs between them is exactly the arithmetic under test, which is the
    point of the T101 seam (**D-054**).

    Attributes:
        name: ``"phase1"`` — never registered; this is a test double, reached by
            assigning to :attr:`lbm.runner.Sim.backend` after construction.
    """

    name: str = "phase1"

    def __init__(self, inner: object) -> None:
        self.inner = inner

    def __getattr__(self, item: str) -> object:
        return getattr(self.inner, item)

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
        """Phase 1's collision. A non-zero ``cs_smag`` is a bug in the caller."""
        if cs_smag != 0.0:
            raise AssertionError(
                "the frozen Phase 1 oracle has no closure; it is what "
                "cs_smag = 0 is compared against."
            )
        _phase1_collide(f, feq, tau)

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
        """Phase 1's fused pass. A non-zero ``cs_smag`` is a bug in the caller."""
        if cs_smag != 0.0:
            raise AssertionError(
                "the frozen Phase 1 oracle has no closure; it is what "
                "cs_smag = 0 is compared against."
            )
        return _phase1_collide_stream(
            f, feq, tau, buf, f_pre=f_pre, solid=solid, f_bb=f_bb
        )


# --- the frozen Phase 1 **warp** oracle (T202). DO NOT EDIT. -----------------
#
# Verbatim transcriptions of lbm.backends.warp_backend._collide_kernel and
# ._collide_bb_kernel as they stood at the end of Phase 1 (M8, session 22),
# before T202 gave the backend a closure. Same rule as the NumPy oracle above:
# one copy, in this file, never edited.
#
# `_stream_kernel` is imported rather than transcribed, for the reason
# `_shift_blocks` is on the host side: streaming moves values and does no
# arithmetic, T202 did not touch it, and copying it here would make this file
# claim to freeze something it does not test.
#
# Guarded because warp-lang is an optional dependency: `python -m validate.les`
# on a machine without it must still run the numpy half.

try:  # pragma: no cover - exercised only where warp-lang is installed
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None  # type: ignore[assignment]

if wp is not None:
    # Streaming has no arithmetic in it (see the comment above), so the
    # live kernel is the frozen one.
    from lbm.backends.warp_backend import _stream_kernel as _warp_stream_kernel

    @wp.kernel
    def _phase1_collide_kernel(
        f: wp.array3d(dtype=wp.float32),
        feq: wp.array3d(dtype=wp.float32),
        one_minus_omega: wp.float32,
    ) -> None:
        """FROZEN Phase 1 warp ``_collide_kernel``. Do not edit."""
        i, y, x = wp.tid()

        v = f[i, y, x]
        v -= feq[i, y, x]
        v *= one_minus_omega
        v += feq[i, y, x]
        f[i, y, x] = v

    @wp.kernel
    def _phase1_collide_bb_kernel(
        f: wp.array3d(dtype=wp.float32),
        feq: wp.array3d(dtype=wp.float32),
        f_pre: wp.array3d(dtype=wp.float32),
        solid: wp.array2d(dtype=wp.uint8),
        opp: wp.array(dtype=wp.int32),
        one_minus_omega: wp.float32,
        has_solid: wp.int32,
        s: wp.array3d(dtype=wp.float32),
    ) -> None:
        """FROZEN Phase 1 warp ``_collide_bb_kernel``. Do not edit."""
        y, x = wp.tid()

        wall = int(0)
        if has_solid != 0:
            if solid[y, x] != wp.uint8(0):
                wall = 1

        for i in range(9):
            if wall != 0:
                s[i, y, x] = f_pre[opp[i], y, x]
            else:
                v = f[i, y, x]
                v -= feq[i, y, x]
                v *= one_minus_omega
                v += feq[i, y, x]
                s[i, y, x] = v


class Phase1WarpBackend:
    """The Warp backend with its two collision kernels frozen at Phase 1.

    :class:`Phase1Backend`'s argument, on the GPU: everything but
    :meth:`collide` and :meth:`collide_stream` forwards to the real backend, so
    both runs share every device allocation, every boundary launch and the whole
    D-020 ordering, and what differs is exactly the arithmetic under test.

    The two methods below transcribe the **method bodies** of
    :meth:`lbm.backends.warp_backend.WarpBackend.collide` and
    ``.collide_stream`` as Phase 1 shipped them — the launch shapes and argument
    order included, because those are part of what "the same kernel" means — and
    launch the frozen kernels above.

    Attributes:
        name: ``"phase1-warp"`` — never registered; a test double, reached by
            assigning to :attr:`lbm.runner.Sim.backend` after construction.
    """

    name: str = "phase1-warp"

    def __init__(self, inner: object) -> None:
        self.inner = inner
        # Compile now rather than inside the comparison, exactly as the real
        # backend's constructor does.
        wp.load_module(module=__name__, device=inner.device)  # type: ignore[attr-defined]

    def __getattr__(self, item: str) -> object:
        return getattr(self.inner, item)

    def collide(
        self,
        f: object,
        feq: object,
        tau: float,
        *,
        cs_smag: float = 0.0,
        smag_out: object = None,
        smag_work: object = None,
    ) -> None:
        """Phase 1's collision. A non-zero ``cs_smag`` is a bug in the caller."""
        if cs_smag != 0.0:
            raise AssertionError(
                "the frozen Phase 1 oracle has no closure; it is what "
                "cs_smag = 0 is compared against."
            )
        one_minus_omega = self.inner._one_minus_omega(tau)  # type: ignore[attr-defined]
        _, ny, nx = f.shape  # type: ignore[attr-defined]
        wp.launch(
            _phase1_collide_kernel,
            dim=(Q, ny, nx),
            inputs=[f, feq, one_minus_omega],
            device=self.inner.device,  # type: ignore[attr-defined]
        )

    def collide_stream(
        self,
        f: object,
        feq: object,
        tau: float,
        buf: object,
        *,
        f_pre: object = None,
        solid: object = None,
        f_bb: object = None,
        cs_smag: float = 0.0,
        smag_out: object = None,
        smag_work: object = None,
    ) -> object:
        """Phase 1's fused pass. A non-zero ``cs_smag`` is a bug in the caller."""
        if cs_smag != 0.0:
            raise AssertionError(
                "the frozen Phase 1 oracle has no closure; it is what "
                "cs_smag = 0 is compared against."
            )
        if solid is not None and f_pre is None:
            raise ValueError(
                "collide_stream needs f_pre (the pre-collision copy, D-011) to "
                "bounce back off solid: f_pre[OPP[i]] is the reflection."
            )
        inner = self.inner
        one_minus_omega = inner._one_minus_omega(tau)  # type: ignore[attr-defined]

        _, ny, nx = f.shape  # type: ignore[attr-defined]
        s = f if f_bb is None else f_bb
        has_solid = 0 if solid is None else 1

        wp.launch(
            _phase1_collide_bb_kernel,
            dim=(ny, nx),
            inputs=[
                f,
                feq,
                f if f_pre is None else f_pre,
                inner._no_mask2d if solid is None else solid,  # type: ignore[attr-defined]
                inner._opp,  # type: ignore[attr-defined]
                one_minus_omega,
                has_solid,
                s,
            ],
            device=inner.device,  # type: ignore[attr-defined]
        )

        if s is f:
            wp.launch(
                _warp_stream_kernel,
                dim=(Q, ny, nx),
                inputs=[f, inner._e_i32, buf],  # type: ignore[attr-defined]
                device=inner.device,  # type: ignore[attr-defined]
            )
            wp.copy(f, buf)
        else:
            wp.launch(
                _warp_stream_kernel,
                dim=(Q, ny, nx),
                inputs=[s, inner._e_i32, f],  # type: ignore[attr-defined]
                device=inner.device,  # type: ignore[attr-defined]
            )
        return f


#: One oracle per backend, keyed by the registry name the case runs on.
PHASE1_ORACLES: dict[str, object] = {"numpy": Phase1Backend}
if wp is not None:
    PHASE1_ORACLES["warp"] = Phase1WarpBackend


# --- the case ----------------------------------------------------------------


@dataclass
class Case:
    """Rung 3's case, built once and run several ways.

    Attributes:
        solid: the full mask, ``(ny, nx)`` ``bool``.
        cylinder: the disc alone, ``(ny, nx)`` ``bool``.
        d_cells: the **measured** diameter (D-019).
        tau: the relaxation time Re 100 implies.
        nu: ``(tau - 0.5) / 3``, lattice units.
    """

    solid: NDArray[np.bool_]
    cylinder: NDArray[np.bool_]
    d_cells: float
    tau: float
    nu: float


def build_case(backend: str = "numpy") -> tuple[Case, object]:
    """Rung 3's mask and ``tau``, and a :class:`lbm.runner.SimConfig` factory.

    Exactly :mod:`validate.cylinder`'s own setup — the mask, the measured
    diameter, ``tau`` from ``Re``, the startup kick — so that "bitwise on Rung
    3's case" means the case Rung 3 actually runs.

    Args:
        backend: the T101 registry name — ``"numpy"`` or, since T202,
            ``"warp"``.

    Returns:
        ``(case, make)`` where ``make(fused, cs_smag)`` builds the config.
    """
    from validate.cylinder import RE, D_CELLS
    from lbm.geometry import bounding_box

    solid, cylinder, _, _ = cylinder_mask(D_CELLS)
    box = bounding_box(cylinder)
    assert box is not None
    d_cells = float(box[1] - box[0] + 1)
    nu, tau = tau_for(RE, U, d_cells)
    ny, nx = solid.shape

    def make(*, fused: bool, cs_smag: float) -> object:
        cfg = make_config(
            ny=ny,
            nx=nx,
            tau=tau,
            u=U,
            outlet_lam=None,
            verbose_mask=False,
            inlet_uy=KICK_FACTOR * U,
            backend=backend,
            cs_smag=cs_smag,
        )
        return cfg.replace(check_geometry=False, fused=fused)

    return Case(solid, cylinder, d_cells, tau, nu), make


def _run(cfg: object, solid: NDArray[np.bool_], steps: int, *, phase1: bool) -> Sim:
    """``steps`` timesteps of ``cfg``, optionally on the frozen Phase 1 kernels.

    The oracle is installed **after** construction, so both runs allocate their
    state through the same backend and start from a byte-identical ``f``. Which
    oracle depends on the backend the case is running on — :data:`PHASE1_ORACLES`
    holds one per backend, and there is exactly one copy of each.
    """
    sim = Sim(cfg, solid)  # type: ignore[arg-type]
    if phase1:
        name = sim.backend.name
        oracle = PHASE1_ORACLES.get(name)
        if oracle is None:
            raise RuntimeError(
                f"no frozen Phase 1 oracle for backend {name!r}: constraint 19 "
                "is a claim about every backend, and it cannot be measured "
                "against a backend compared with itself."
            )
        sim.backend = oracle(sim.backend)  # type: ignore[assignment,operator]
    sim.run_steps(steps)
    return sim


# --- clause 1: bitwise degeneracy --------------------------------------------


@dataclass
class BitwiseResult:
    """What clause 1 measured.

    Attributes:
        steps: timesteps each run advanced.
        fused_equal: closure-off fused ``f`` equals the frozen Phase 1 fused
            ``f`` under :func:`numpy.array_equal`.
        unfused_equal: the same on the unfused path.
        paths_equal: closure-off fused equals closure-off unfused (**D-055**).
        fused_ulps: worst absolute difference on the fused path — ``0.0`` when
            it passes, and a number worth printing when it does not.
        unfused_ulps: the same for the unfused path.
        seconds: wall clock for all four runs.
    """

    steps: int
    fused_equal: bool
    unfused_equal: bool
    paths_equal: bool
    fused_diff: float
    unfused_diff: float
    seconds: float


def check_bitwise(case: Case, make, steps: int) -> BitwiseResult:
    """Constraint 19: ``cs_smag = 0`` is Phase 1's collision, bit for bit.

    Four runs of Rung 3's case: ``{fused, unfused}`` x ``{today, frozen}``.
    """
    start = time.perf_counter()

    new_fused = _run(make(fused=True, cs_smag=0.0), case.solid, steps, phase1=False)
    old_fused = _run(make(fused=True, cs_smag=0.0), case.solid, steps, phase1=True)
    new_plain = _run(make(fused=False, cs_smag=0.0), case.solid, steps, phase1=False)
    old_plain = _run(make(fused=False, cs_smag=0.0), case.solid, steps, phase1=True)

    a, b = new_fused.host_f(), old_fused.host_f()
    c, d = new_plain.host_f(), old_plain.host_f()

    return BitwiseResult(
        steps=steps,
        fused_equal=bool(np.array_equal(a, b)),
        unfused_equal=bool(np.array_equal(c, d)),
        paths_equal=bool(np.array_equal(a, c)),
        fused_diff=float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()),
        unfused_diff=float(np.abs(c.astype(np.float64) - d.astype(np.float64)).max()),
        seconds=time.perf_counter() - start,
    )


# --- clause 2: the closure engages, and only ever adds viscosity -------------


@dataclass
class ClosureResult:
    """What clause 2 measured, on ``Cs = 0.17`` after ``steps`` steps.

    Attributes:
        steps: timesteps advanced.
        cs: the Smagorinsky constant used.
        tau: the base relaxation time.
        nu: ``(tau - 0.5) / 3``.
        tau_eff_min: smallest ``tau_eff`` over the domain.
        tau_eff_max: largest ``tau_eff`` over the domain.
        nu_t_max: largest eddy viscosity over the domain, lattice units.
        nu_t_mean: domain average eddy viscosity.
        nu_t_min: smallest eddy viscosity — must not be negative.
        nu_t_zero_at_zero: ``eddy_viscosity(..., cs_smag=0)`` is exactly zero
            everywhere on the same state.
        seconds: wall clock.
    """

    steps: int
    cs: float
    tau: float
    nu: float
    tau_eff_min: float
    tau_eff_max: float
    nu_t_min: float
    nu_t_max: float
    nu_t_mean: float
    nu_t_zero_at_zero: bool
    seconds: float


def check_closure(case: Case, make, steps: int, cs: float = CS) -> ClosureResult:
    """The closure fires, ``tau_eff >= tau`` everywhere, ``nu_t >= 0`` everywhere.

    A shedding cylinder wake is the strong-shear case the T201 contract asks
    for. ``max(nu_t) / nu`` is the quantity **D-082**'s fidelity bands are read
    off, so it is printed here even though nothing in T201 consumes it.
    """
    start = time.perf_counter()
    sim = _run(make(fused=True, cs_smag=cs), case.solid, steps, phase1=False)

    f = sim.host_f()
    rho, u = macroscopic(f.copy())
    feq = equilibrium(rho, u)

    tau_eff = smagorinsky_tau_eff(f, feq, case.tau, cs)
    nu_t = eddy_viscosity(f, feq, case.tau, cs)
    nu_t0 = eddy_viscosity(f, feq, case.tau, 0.0)

    return ClosureResult(
        steps=steps,
        cs=cs,
        tau=case.tau,
        nu=case.nu,
        tau_eff_min=float(tau_eff.min()),
        tau_eff_max=float(tau_eff.max()),
        nu_t_min=float(nu_t.min()),
        nu_t_max=float(nu_t.max()),
        nu_t_mean=float(nu_t.mean()),
        nu_t_zero_at_zero=bool((nu_t0 == 0.0).all()),
        seconds=time.perf_counter() - start,
    )


# --- clause 4: cross-backend agreement with the closure ON -------------------


@dataclass
class CrossResult:
    """What clause 4 measured — only runs when ``--backend`` is not numpy.

    Attributes:
        backend: the backend compared against numpy.
        cs: the Smagorinsky constant both sides ran.
        kernel_worst: worst absolute difference over every grid and both
            collision kernels, ``f`` units. Held to **D-053**'s 1e-6.
        kernel_where: which kernel and grid produced ``kernel_worst``.
        step_du: ``max|Delta u| / U`` at the last rung of
            :data:`validate.parity.STEP_LADDER`. Held to **D-056**'s 1e-4.
        step_points: ``(steps, du, df)`` at every rung, so the growth rate is
            visible and not merely bounded.
        step_nu_t_ratio: ``max(nu_t) / nu`` on the whole-step case's own final
            state. Printed because the whole-step number alone cannot say
            whether the closure was *engaged* in the case that produced it.
            It is: ~9% of ``nu`` on Rung A's own case. The whole-step figure
            nevertheless lands on the same digits Rung A publishes with the
            closure **off**, because the disagreement is dominated by the FMA
            contractions **D-053** already documents and the closure perturbs
            both backends coherently. A reader comparing the two numbers is
            owed that sentence rather than left to assume the clause was inert.
        finite: both sides stayed finite the whole way.
        seconds: wall clock.
    """

    backend: str
    cs: float
    kernel_worst: float
    kernel_where: str
    step_du: float
    step_points: list[tuple[int, float, float]]
    step_nu_t_ratio: float
    finite: bool
    seconds: float


def check_cross_backend(backend: str, cs: float = CS) -> CrossResult:
    """The closure agrees across backends to the bars Rung A already publishes.

    ``DOCS/TASKS3.md`` § T202, second acceptance criterion: *per-kernel worst
    under 1e-6 in ``f`` units (**D-053**'s bar), whole step under 1e-4 in
    ``max|Delta u|/U`` at 1000 steps (**D-056**'s bar). The measured numbers are
    printed and recorded, not just compared.*

    Both halves run **Rung A's own harness** — :func:`validate.parity.random_state`
    and its grids for the kernels, :func:`validate.parity.whole_step` with
    ``cs_smag`` threaded through for the step — so the bars are measured on the
    case they were set on. ``feq`` is uploaded from NumPy to both sides, exactly
    as :func:`validate.parity.compare_kernels` does, because this measures the
    **collision** and not the equilibrium error fed through it.

    Args:
        backend: the backend to compare against numpy.
        cs: the Smagorinsky constant, non-zero — with the closure off this
            clause would be measuring Rung A over again.

    Returns:
        A :class:`CrossResult`.
    """
    from lbm.backends import get_backend
    from validate.parity import (
        GRIDS,
        STEP_LADDER,
        TAU,
        random_state,
        step_case,
        whole_step,
    )

    start = time.perf_counter()
    ref = get_backend("numpy")
    dut = get_backend(backend)

    worst = 0.0
    where = "-"
    for ny, nx in GRIDS:
        rho, u, f = random_state(ny, nx)
        feq_host = ref.download(ref.equilibrium(ref.upload(rho), ref.upload(u)))

        # collide, in place, from the same f and the same feq.
        a = ref.upload(f)
        b = dut.upload(f)
        ref.collide(a, ref.upload(feq_host), TAU, cs_smag=cs)
        dut.collide(b, dut.upload(feq_host), TAU, cs_smag=cs)
        d = float(np.max(np.abs(ref.download(a) - dut.download(b))))
        if d > worst:
            worst, where = d, f"collide at {nx}x{ny}"

        # collide_stream, the fused path, unmasked — the reflection is an
        # assignment and Rung A already compares it on its own.
        a = ref.upload(f)
        b = dut.upload(f)
        ref.collide_stream(
            a, ref.upload(feq_host), TAU, ref.empty((Q, ny, nx)), cs_smag=cs
        )
        dut.collide_stream(
            b, dut.upload(feq_host), TAU, dut.empty((Q, ny, nx)), cs_smag=cs
        )
        d = float(np.max(np.abs(ref.download(a) - dut.download(b))))
        if d > worst:
            worst, where = d, f"collide_stream at {nx}x{ny}"

    points = whole_step(backend, STEP_LADDER, cs_smag=cs)

    # How hard the closure was biting in the case that produced those numbers.
    cfg, solid = step_case("numpy", cs_smag=cs)
    sim = Sim(cfg, solid)
    sim.run_steps(STEP_LADDER[-1])
    fs = sim.host_f()
    rho_s, u_s = macroscopic(fs.copy())
    nu_t = eddy_viscosity(fs, equilibrium(rho_s, u_s), cfg.tau, cs)
    ratio = float(nu_t.max()) / ((cfg.tau - 0.5) / 3.0)

    return CrossResult(
        backend=backend,
        cs=cs,
        kernel_worst=worst,
        kernel_where=where,
        step_du=points[-1].du_over_u,
        step_points=[(p.steps, p.du_over_u, p.df) for p in points],
        step_nu_t_ratio=ratio,
        finite=all(p.finite for p in points),
        seconds=time.perf_counter() - start,
    )


# --- reporting ---------------------------------------------------------------


def report(
    bit: BitwiseResult,
    clo: ClosureResult,
    cyl_passed: bool | None,
    backend: str,
    cross: CrossResult | None = None,
) -> bool:
    """Print every check and return whether Rung F passed."""
    print()
    print("  measured")
    print(f"    bitwise clause   4 x {bit.steps} steps in {bit.seconds:.1f} s")
    print(f"    closure clause   {clo.steps} steps at Cs = {clo.cs} "
          f"in {clo.seconds:.1f} s")
    print(f"    tau              {clo.tau:.6f}   nu = (tau - 0.5)/3 = "
          f"{clo.nu:.6f}")
    print(f"    tau_eff          {clo.tau_eff_min:.6f} .. {clo.tau_eff_max:.6f}")
    print(f"    nu_t             {clo.nu_t_min:.3e} .. {clo.nu_t_max:.3e}   "
          f"mean {clo.nu_t_mean:.3e}")
    print(f"    max(nu_t) / nu   {clo.nu_t_max / clo.nu:.4f}   "
          f"(the quantity D-082's fidelity bands are read off; T204 consumes it)")
    if cross is not None:
        print(f"    cross-backend    numpy vs {cross.backend} at Cs = {cross.cs} "
              f"in {cross.seconds:.1f} s")
        print(f"      per-kernel     {cross.kernel_worst:.3e} f units "
              f"({cross.kernel_where}), bar 1e-6 (D-053)")
        growth = "   ".join(
            f"{n}: {du:.3e}" for n, du, _ in cross.step_points
        )
        print(f"      whole step     max|du|/U   {growth}   bar 1e-4 (D-056)")
        print(f"      on that case   max(nu_t)/nu {cross.step_nu_t_ratio:.3e} "
              f"-- the closure is engaged; the whole-step figure still")
        print("                     matches Rung A's closure-off digits "
              "because D-053's FMA contractions dominate it")

    checks: list[tuple[str, bool, str]] = [
        (
            "cs_smag=0 fused is bitwise Phase 1 (constraint 19)",
            bit.fused_equal,
            f"array_equal after {bit.steps} steps, "
            f"worst |diff| {bit.fused_diff:.3e}",
        ),
        (
            "cs_smag=0 unfused is bitwise Phase 1 (constraint 19)",
            bit.unfused_equal,
            f"array_equal after {bit.steps} steps, "
            f"worst |diff| {bit.unfused_diff:.3e}",
        ),
        (
            "fused and unfused agree bitwise (D-055)",
            bit.paths_equal,
            f"array_equal after {bit.steps} steps",
        ),
        (
            "tau_eff >= tau in every cell (the closure only adds)",
            clo.tau_eff_min >= clo.tau,
            f"min tau_eff {clo.tau_eff_min:.6f} vs tau {clo.tau:.6f}",
        ),
        (
            "nu_t >= 0 in every cell (constraint 2 through tau)",
            clo.nu_t_min >= 0.0,
            f"min {clo.nu_t_min:.3e}",
        ),
        (
            "the closure is not inert: max(nu_t) > 0",
            clo.nu_t_max > 0.0,
            f"max {clo.nu_t_max:.3e} = {clo.nu_t_max / clo.nu:.2%} of nu",
        ),
        (
            "nu_t is exactly 0 when cs_smag = 0",
            clo.nu_t_zero_at_zero,
            "on the same state, every cell",
        ),
    ]

    if cross is not None:
        from validate.parity import STEP_TOL, TOL

        checks.extend(
            [
                (
                    f"numpy vs {cross.backend} per-kernel with the closure on "
                    f"< {TOL:.0e} f units (D-053)",
                    cross.kernel_worst < TOL,
                    f"worst {cross.kernel_worst:.3e} ({cross.kernel_where})",
                ),
                (
                    f"numpy vs {cross.backend} whole step with the closure on "
                    f"< {STEP_TOL:.0e} max|du|/U at "
                    f"{cross.step_points[-1][0]} steps (D-056)",
                    cross.finite and cross.step_du < STEP_TOL,
                    f"{cross.step_du:.3e}"
                    + ("" if cross.finite else "  (NON-FINITE)"),
                ),
            ]
        )

    if cyl_passed is not None:
        cd_lo, cd_hi = CD_BAND
        st_lo, st_hi = ST_BAND
        checks.append(
            (
                f"Rung 3 with Cs = {clo.cs}: Cd {cd_lo}-{cd_hi}, "
                f"St {st_lo}-{st_hi}",
                cyl_passed,
                "validate.cylinder's own report, above",
            )
        )

    width = max(len(name) for name, _, _ in checks)
    print()
    for name, ok, detail in checks:
        print(f"    [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    passed = all(ok for _, ok, _ in checks)
    print()
    print(f"  scope: {backend}. Rung F is a claim about **every** backend "
          f"(constraint 19),")
    print("         so it is green only when it has been run on each of them.")
    print("PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run Rung F and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Rung F — the Smagorinsky closure is switchable "
        "(Cs = 0 is bitwise BGK; Rung 3 survives Cs = 0.17)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=BITWISE_STEPS,
        help=f"steps for the bitwise and closure clauses (default {BITWISE_STEPS})",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=CS,
        help=f"Smagorinsky constant for clauses 2 and 3 (default {CS}, the "
        f"literature value; Phase 2 does not tune it)",
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="the T101 backend to run every clause on (default numpy). "
        "Constraint 19 is a claim about every backend, so Rung F is green "
        "only once it has been run on each; --backend warp additionally "
        "measures cross-backend agreement with the closure on (T202).",
    )
    parser.add_argument(
        "--skip-cross",
        action="store_true",
        help="skip clause 4, the cross-backend comparison. Only meaningful "
        "with a non-numpy --backend, and a skipped clause is reported.",
    )
    parser.add_argument(
        "--skip-cylinder",
        action="store_true",
        help="skip clause 3, the full Rung 3 case with the closure on. It is "
        "the expensive clause (~45k steps) and this makes the cheap ones "
        "runnable in a few seconds. A skipped clause is reported, never "
        "silently dropped.",
    )
    args = parser.parse_args(argv)

    # Nothing here draws, and this makes that true even for a stray SDL init.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    backend = args.backend
    try:
        get_backend(backend)
    except BackendUnavailableError as exc:
        print(f"SKIP - {exc}")
        return 2

    print("Rung F — the Smagorinsky closure, switchable "
          "(DOCS/IDEA4.md § Validation ladder)")
    print(f"  backend {backend}   Cs = {args.cs} (literature; "
          f"lbm.core.CS_SMAG_LITERATURE)   filter width 1 lattice unit")
    print("  oracle: the frozen Phase 1 collide / collide_stream in this file")
    print()

    case, make = build_case(backend)
    ny, nx = case.solid.shape
    print(f"  Rung 3's case: grid {ny} x {nx}   D = {case.d_cells:.0f} cells   "
          f"tau {case.tau:.6f}   nu {case.nu:.6f}")
    print(f"  running 4 x {args.steps} steps for the bitwise clause ...",
          flush=True)
    bit = check_bitwise(case, make, args.steps)

    print(f"  running {args.steps} steps at Cs = {args.cs} for the closure "
          f"clause ...", flush=True)
    clo = check_closure(case, make, args.steps, args.cs)

    cyl_passed: bool | None = None
    if not args.skip_cylinder:
        print()
        print("  clause 3 — the full Rung 3 case with the closure on:")
        print()
        res = run_cylinder(
            headless=True,
            bench_steps=0,
            verbose_mask=False,
            backend=backend,
            cs_smag=args.cs,
        )
        cyl_passed = cylinder_report(res)
        print()
        print("  (back in Rung F)")
    else:
        print()
        print("  clause 3 SKIPPED by --skip-cylinder: Rung 3 with the closure "
              "on was not run.")

    cross: CrossResult | None = None
    if backend != "numpy" and not args.skip_cross:
        print()
        print(f"  clause 4 — numpy vs {backend} with the closure on, against "
              f"Rung A's own bars ...", flush=True)
        cross = check_cross_backend(backend, args.cs)
    elif backend != "numpy":
        print()
        print("  clause 4 SKIPPED by --skip-cross: cross-backend agreement "
              "with the closure on was not measured.")

    passed = report(bit, clo, cyl_passed, backend, cross)
    if args.skip_cylinder:
        print("  NOTE: clause 3 was skipped, so this is not a full Rung F pass.")
    if backend != "numpy" and args.skip_cross:
        print("  NOTE: clause 4 was skipped, so this is not a full Rung F pass.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
