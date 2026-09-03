"""Tests for the T201 Smagorinsky closure — ``DOCS/TASKS3.md`` § T201.

One test per acceptance criterion in that contract, plus the invariants
``CLAUDE.md`` constraints 1, 2, 4 and **19** put on it:

* the **exact algebra** is pinned twice — once against the published quadratic
  written out independently, and once against a velocity field constructed from
  a *chosen* strain rate, which is the only check that pins the filter width and
  the strain norm together rather than the shape of the formula;
* ``cs_smag = 0`` is **bitwise** the collision Phase 1 shipped, on the fused and
  the unfused path, and the two agree bitwise with each other (**D-055**);
* ``nu_t = cs2 (tau_eff - tau)`` is derived through ``tau``, is never negative,
  and is **exactly** zero with the closure off;
* the closure allocates nothing inside the step loop and nothing at all when it
  is off;
* nothing in ``lbm/``, ``flow/`` or ``validate/`` turns it on except the rung
  that tests it.

The frozen Phase 1 oracle lives in :mod:`validate.les` (Rung F) and is imported
here rather than copied, so there is exactly one transcription of Phase 1's
collision in the repository and the unit tests and the rung cannot drift apart.
"""

from __future__ import annotations

import subprocess
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from lbm.backends import Backend, available_backends, get_backend
from lbm.core import (
    CS2,
    CS_SMAG_LITERATURE,
    E_F32,
    Q,
    SMAG_Q_COEFF,
    W,
    collide,
    collide_stream,
    equilibrium,
    macroscopic,
    smagorinsky_omega,
    smagorinsky_tau_eff,
)
from lbm.geometry import channel_walls, circle
from lbm.probe import eddy_viscosity
from lbm.runner import Sim, SimConfig
from validate.les import Phase1Backend, _phase1_collide

NY, NX = 24, 40
TAU = 0.6
CS = CS_SMAG_LITERATURE

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def random_state(seed: int = 0) -> NDArray[np.float32]:
    """A plausible ``(9, ny, nx)`` ``float32`` distribution, near equilibrium.

    Built as ``feq`` of a smooth velocity field plus a small perturbation, so
    the non-equilibrium part is the size a real run's is rather than the size
    uniform noise would give.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:NY, 0:NX].astype(np.float32)
    rho = np.ones((NY, NX), dtype=np.float32)
    u = np.empty((2, NY, NX), dtype=np.float32)
    u[0] = 0.05 * np.sin(2.0 * np.pi * y / NY)
    u[1] = 0.02 * np.cos(2.0 * np.pi * x / NX)
    f = equilibrium(rho, u)
    f += rng.normal(0.0, 1e-3, f.shape).astype(np.float32)
    return np.ascontiguousarray(f, dtype=np.float32)


def feq_of(f: NDArray[np.float32]) -> NDArray[np.float32]:
    """The equilibrium of ``f``'s own macroscopic moments."""
    rho, u = macroscopic(f.copy())
    return equilibrium(rho, u)


def channel_with_cylinder() -> NDArray[np.bool_]:
    """Walls top and bottom plus a disc — a case with real shear in it."""
    return channel_walls(NY, NX) | circle(NY, NX, cx=16.0, cy=11.5, radius=4.0)


def flow_config(**over) -> SimConfig:
    """A driven channel: Zou-He inlet, convective outlet, an obstacle."""
    cfg = SimConfig(
        ny=NY,
        nx=NX,
        tau=TAU,
        inlet_U=0.05,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        check_geometry=False,
    )
    return cfg.replace(**over) if over else cfg


def q_tensor(
    f: NDArray[np.float32], feq: NDArray[np.float32]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(Qxx, Qxy, Qyy)`` in ``float64``, by an independent route.

    ``np.einsum`` over the whole nine directions at once, in double precision —
    deliberately *not* the accumulation :func:`lbm.core.smagorinsky_tau_eff`
    performs, so that agreement is evidence about the formula rather than about
    the loop.
    """
    neq = (f.astype(np.float64) - feq.astype(np.float64))
    e = E_F32.astype(np.float64)
    qxx = np.einsum("i,iyx->yx", e[:, 0] * e[:, 0], neq)
    qxy = np.einsum("i,iyx->yx", e[:, 0] * e[:, 1], neq)
    qyy = np.einsum("i,iyx->yx", e[:, 1] * e[:, 1], neq)
    return qxx, qxy, qyy


# ---------------------------------------------------------------------------
# The algebra, pinned
# ---------------------------------------------------------------------------


def test_the_docstring_names_its_source_and_its_filter_width():
    """T201: the docstring cites Hou et al. (1996) and states ``Delta = 1``.

    The normalisation of an LBM Smagorinsky model is a *choice*, and the T201
    contract is explicit that the choice has to be written down: XLB's own
    coefficient differs from ours by exactly the strain-norm convention. A
    docstring that does not say which convention it took is a docstring that
    cannot be checked against a paper.
    """
    doc = smagorinsky_tau_eff.__doc__ or ""
    assert "Hou" in doc and "1996" in doc
    assert "filter width is one lattice unit" in doc
    assert "Delta = 1" in doc
    assert "sqrt(2 S_ab S_ab)" in doc


def test_tau_eff_solves_the_published_quadratic():
    """The exact algebra, against the quadratic written out independently.

    ``tau_eff^2 - tau tau_eff - sqrt(2) Cs^2 |Q| / (2 rho cs2^2) = 0``, with
    ``|Q|`` from :func:`q_tensor`'s einsum rather than from the implementation's
    own accumulation. The residual is compared to ``tau^2``, which is the scale
    the terms actually have.
    """
    f = random_state()
    feq = feq_of(f)
    tau_eff = smagorinsky_tau_eff(f, feq, TAU, CS).astype(np.float64)

    qxx, qxy, qyy = q_tensor(f, feq)
    q_norm = np.sqrt(qxx * qxx + 2.0 * qxy * qxy + qyy * qyy)
    rho = f.astype(np.float64).sum(axis=0)

    const = np.sqrt(2.0) * CS * CS * q_norm / (2.0 * rho * CS2 * CS2)
    residual = tau_eff * tau_eff - TAU * tau_eff - const

    assert np.abs(residual).max() < 1e-6 * TAU * TAU


def test_the_coefficient_under_the_root_is_18_root_2():
    """``SMAG_Q_COEFF`` is the constant the docstring derives, not a fitted one.

    ``1 / (2 cs2^2) = 4.5`` with ``cs2 = 1/3``, so the ``4 C`` of the quadratic
    formula carries ``4 * 4.5 * sqrt(2) = 18 sqrt(2)``. Stated here as a number
    so that a future edit to either the constant or the derivation trips over
    the other one.
    """
    assert SMAG_Q_COEFF == pytest.approx(18.0 * np.sqrt(2.0), rel=0, abs=0)
    assert 4.0 * (1.0 / (2.0 * CS2 * CS2)) * np.sqrt(2.0) == pytest.approx(
        SMAG_Q_COEFF
    )
    # XLB's own 2D closure carries 36 in this position; the ratio is exactly the
    # strain-norm convention the docstring names (session 23's cross-check).
    assert 36.0 / SMAG_Q_COEFF == pytest.approx(np.sqrt(2.0))


def test_it_recovers_a_strain_rate_it_was_given():
    """The whole normalisation chain, against a **chosen** strain rate.

    This is the test that pins the filter width and the strain norm together,
    and it runs the model backwards. Pick a target eddy viscosity, hence a
    target ``tau_eff``; build the non-equilibrium part Chapman-Enskog says a
    flow with that strain and that relaxation time has::

        f_i^(1) = -(w_i rho tau_eff / cs2) (e_ia e_ib - cs2 delta_ab) S_ab

    for a pure shear ``S_xy = S_yx = a / 2``, whose norm under this module's
    convention is ``|S| = sqrt(2 S_ab S_ab) = |a|``; then ``nu_t = Cs^2 |a|``
    and ``tau_eff = tau + nu_t / cs2`` is the fixed point the solver must find.
    Nothing here is taken from the implementation except the answer.
    """
    a = 0.02  # the shear rate du/dy, lattice units
    nu_t = CS * CS * abs(a)  # Delta = 1, so (Cs Delta)^2 = Cs^2
    tau_eff_true = TAU + nu_t / CS2

    rho = np.ones((NY, NX), dtype=np.float64)
    s_xy = 0.5 * a
    s = np.array([[0.0, s_xy], [s_xy, 0.0]], dtype=np.float64)

    e = E_F32.astype(np.float64)
    f1 = np.empty((Q, NY, NX), dtype=np.float64)
    for i in range(Q):
        proj = 0.0
        for aa in range(2):
            for bb in range(2):
                delta = 1.0 if aa == bb else 0.0
                proj += (e[i, aa] * e[i, bb] - CS2 * delta) * s[aa, bb]
        f1[i] = -(float(W[i]) * rho * tau_eff_true / CS2) * proj

    feq = equilibrium(
        rho.astype(np.float32), np.zeros((2, NY, NX), dtype=np.float32)
    )
    f = (feq.astype(np.float64) + f1).astype(np.float32)

    tau_eff = smagorinsky_tau_eff(f, np.ascontiguousarray(feq), TAU, CS)
    assert float(tau_eff.min()) == pytest.approx(tau_eff_true, rel=2e-4)
    assert float(tau_eff.max()) == pytest.approx(tau_eff_true, rel=2e-4)

    nu_t_field = eddy_viscosity(f, np.ascontiguousarray(feq), TAU, CS)
    assert float(nu_t_field.mean()) == pytest.approx(nu_t, rel=2e-4)


def test_omega_is_exactly_the_reciprocal_of_tau_eff():
    """One model, two views: ``omega_eff`` is ``1 / tau_eff`` and nothing else."""
    f = random_state(1)
    feq = feq_of(f)
    tau_eff = smagorinsky_tau_eff(f, feq, TAU, CS)
    omega = smagorinsky_omega(f, feq, TAU, CS)
    assert np.array_equal(omega, np.float32(1.0) / tau_eff)


# ---------------------------------------------------------------------------
# Constraint 19 — off is off, bitwise
# ---------------------------------------------------------------------------


def test_tau_eff_is_exactly_tau_when_the_closure_is_off():
    """Constraint 19's limit, taken exactly rather than approached."""
    f = random_state(2)
    feq = feq_of(f)
    tau_eff = smagorinsky_tau_eff(f, feq, TAU, 0.0)
    assert np.array_equal(tau_eff, np.full((NY, NX), np.float32(TAU)))


def test_collide_with_cs_zero_is_bitwise_phase_1():
    """``collide(..., cs_smag=0)`` is the Phase 1 function, bit for bit."""
    f_new = random_state(3)
    f_old = f_new.copy()
    feq = feq_of(f_new)

    collide(f_new, feq, TAU)
    _phase1_collide(f_old, feq, TAU)
    assert np.array_equal(f_new, f_old)

    # ... and explicitly passing the default changes nothing.
    f_kw = random_state(3)
    collide(f_kw, feq, TAU, cs_smag=0.0)
    assert np.array_equal(f_kw, f_old)


@pytest.mark.parametrize("fused", [True, False])
def test_a_whole_run_with_the_closure_off_is_bitwise_phase_1(fused):
    """Rung F's first clause, in miniature: 200 steps, both paths.

    The rung runs 1000 steps of Rung 3's case; this runs 200 of a 24x40 channel
    so that ``pytest`` stays fast. Same oracle, same assertion.
    """
    mask = channel_with_cylinder()

    new = Sim(flow_config(fused=fused), mask)
    new.run_steps(200)

    old = Sim(flow_config(fused=fused), mask)
    old.backend = Phase1Backend(old.backend)
    old.run_steps(200)

    assert np.array_equal(new.host_f(), old.host_f())


def test_fused_and_unfused_agree_bitwise_with_the_closure_on():
    """**D-055** survives the closure: the fusion is still a speed switch."""
    mask = channel_with_cylinder()

    a = Sim(flow_config(fused=True, cs_smag=CS), mask)
    a.run_steps(200)
    b = Sim(flow_config(fused=False, cs_smag=CS), mask)
    b.run_steps(200)

    assert np.array_equal(a.host_f(), b.host_f())


def test_the_closure_actually_changes_the_answer():
    """A guard against the whole suite passing because the model is inert."""
    mask = channel_with_cylinder()
    off = Sim(flow_config(cs_smag=0.0), mask)
    off.run_steps(200)
    on = Sim(flow_config(cs_smag=CS), mask)
    on.run_steps(200)
    assert not np.array_equal(off.host_f(), on.host_f())


# ---------------------------------------------------------------------------
# Constraint 2 — viscosity comes from the relaxation time, and only adds
# ---------------------------------------------------------------------------


def test_nu_t_is_zero_exactly_when_the_closure_is_off():
    """``nu_t == 0``, not ``nu_t ~ 0``: T201's acceptance criterion."""
    f = random_state(4)
    feq = feq_of(f)
    nu_t = eddy_viscosity(f, feq, TAU, 0.0)
    assert nu_t.dtype == np.float32
    assert nu_t.shape == (NY, NX)
    assert np.array_equal(nu_t, np.zeros((NY, NX), dtype=np.float32))


def test_nu_t_is_never_negative_and_is_cs2_times_the_tau_increment():
    """``nu_t = cs2 (tau_eff - tau)``, derived through ``tau`` (constraint 2)."""
    f = random_state(5)
    feq = feq_of(f)
    tau_eff = smagorinsky_tau_eff(f, feq, TAU, CS)
    nu_t = eddy_viscosity(f, feq, TAU, CS)

    assert float(nu_t.min()) >= 0.0
    assert np.allclose(
        nu_t, np.float32(CS2) * (tau_eff - np.float32(TAU)), rtol=0, atol=0
    )


def test_tau_eff_never_dips_below_tau_on_a_strongly_sheared_case():
    """T201: ``tau_eff >= tau`` for every cell, always — on real shear.

    200 steps of a channel with a cylinder in it produces a separated wake and
    a boundary layer on each wall, which is where a sign error in the strain
    norm would show up as a cell relaxing *faster* than the fluid does.
    """
    sim = Sim(flow_config(cs_smag=CS), channel_with_cylinder())
    sim.run_steps(200)
    f = sim.host_f()
    feq = feq_of(f)

    tau_eff = smagorinsky_tau_eff(f, feq, TAU, CS)
    assert float(tau_eff.min()) >= TAU
    assert float(tau_eff.max()) > TAU  # the wake is not a uniform flow
    assert float(eddy_viscosity(f, feq, TAU, CS).min()) >= 0.0


@pytest.mark.parametrize("bad", [-1e-6, -0.17])
def test_a_negative_cs_is_refused(bad):
    """The closure adds viscosity and never removes it — refused at the door."""
    f = random_state(6)
    feq = feq_of(f)
    with pytest.raises(ValueError, match="non-negative"):
        smagorinsky_tau_eff(f, feq, TAU, bad)
    with pytest.raises(ValueError, match="non-negative"):
        Sim(flow_config(cs_smag=bad), channel_with_cylinder())


@pytest.mark.parametrize("tau", [0.5, 0.4])
def test_tau_at_or_below_a_half_is_still_refused(tau):
    """Constraint 2 is unchanged by the closure."""
    f = random_state(7)
    feq = feq_of(f)
    with pytest.raises(ValueError, match="0.5"):
        smagorinsky_tau_eff(f, feq, tau, CS)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def test_smagorinsky_with_buffers_allocates_nothing():
    """``out=`` and ``work=`` make the call allocation-free (conventions)."""
    f = random_state(8)
    feq = feq_of(f)
    out = np.empty((NY, NX), dtype=np.float32)
    work = np.empty((4, NY, NX), dtype=np.float32)

    smagorinsky_tau_eff(f, feq, TAU, CS, out=out, work=work)  # warm up

    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    for _ in range(50):
        smagorinsky_tau_eff(f, feq, TAU, CS, out=out, work=work)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grown = sum(
        st.size_diff for st in after.compare_to(base, "filename") if st.size_diff > 0
    )
    assert grown < 20_000, f"the closure grew the heap by {grown} bytes"


def test_sim_allocates_the_closure_buffers_only_when_the_closure_is_on():
    """T201: ``Sim`` preallocates ``out=`` **only** when the closure is on."""
    mask = channel_with_cylinder()

    off = Sim(flow_config(), mask)
    assert off.smag_out is None and off.smag_work is None

    on = Sim(flow_config(cs_smag=CS), mask)
    assert on.smag_out.shape == (NY, NX)
    assert on.smag_work.shape == (4, NY, NX)
    assert on.smag_out.dtype == np.float32 and on.smag_work.dtype == np.float32


@pytest.mark.parametrize("cs", [0.0, CS])
def test_the_step_loop_allocates_nothing_with_the_closure_off_or_on(cs):
    """Constraint's conventions: never allocate inside the step loop.

    Parameterised over both states because the interesting claim is not just
    that the closure-off path is unchanged — it is that turning the closure on
    does not start allocating per step either.
    """
    sim = Sim(flow_config(cs_smag=cs), channel_with_cylinder())
    sim.run_steps(50)

    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    sim.run_steps(500)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grown = sum(
        st.size_diff for st in after.compare_to(base, "filename") if st.size_diff > 0
    )
    assert grown < 20_000, f"step loop grew the heap by {grown} bytes"


def test_f_keeps_its_buffer_identity_with_the_closure_on():
    """Constraint 11's precondition: ``f`` is never rebound."""
    sim = Sim(flow_config(cs_smag=CS), channel_with_cylinder())
    before = sim.f.__array_interface__["data"]
    sim.run_steps(100)
    assert sim.f.__array_interface__["data"] == before


# ---------------------------------------------------------------------------
# The seam, and the closure defaulting off across the tree
# ---------------------------------------------------------------------------


def test_the_backend_protocol_carries_cs_smag_and_numpy_implements_it():
    """``Backend.collide`` / ``collide_stream`` gained the keyword (T201)."""
    import inspect

    from lbm.backends.numpy_backend import NumpyBackend

    for owner in (Backend, NumpyBackend):
        for name in ("collide", "collide_stream"):
            params = inspect.signature(getattr(owner, name)).parameters
            assert "cs_smag" in params, (owner, name)
            assert params["cs_smag"].default == 0.0
            assert params["cs_smag"].kind is inspect.Parameter.KEYWORD_ONLY
            assert "smag_out" in params and "smag_work" in params

    be = NumpyBackend()
    assert isinstance(be, Backend)


def test_the_numpy_backend_delegates_the_closure_to_core():
    """The backend is a delegation, not a second implementation (**D-043**)."""
    from lbm.backends.numpy_backend import NumpyBackend

    f_be = random_state(9)
    f_core = f_be.copy()
    feq = feq_of(f_be)

    NumpyBackend().collide(f_be, feq, TAU, cs_smag=CS)
    collide(f_core, feq, TAU, cs_smag=CS)
    assert np.array_equal(f_be, f_core)


# ---------------------------------------------------------------------------
# T202 — the closure on the Warp backend
# ---------------------------------------------------------------------------

warp_only = pytest.mark.skipif(
    "warp" not in available_backends(), reason="warp-lang is not installed"
)

#: Rung A's own per-kernel bar (**D-053**), in ``f`` units. The closure is held
#: to the same number, not a widened one — ``DOCS/TASKS3.md`` § T202.
PARITY_TOL = 1e-6


@warp_only
def test_warp_agrees_with_numpy_on_the_closure_to_rung_as_own_bar():
    """T202: the same arithmetic on the GPU, to **D-053**'s 1e-6 in ``f`` units.

    NumPy is the oracle (**D-043**); a GPU that disagrees with it is a broken
    backend and not a new answer. Both collision entry points are checked,
    because :meth:`WarpBackend.collide` and ``.collide_stream`` reach the
    closure by different launches.
    """
    be = get_backend("warp")
    f = random_state(21)
    feq = feq_of(f)

    ref = f.copy()
    collide(ref, feq, TAU, cs_smag=CS)
    dev = be.from_host(f.copy())
    be.collide(dev, be.from_host(feq), TAU, cs_smag=CS)
    assert np.abs(be.to_host(dev) - ref).max() < PARITY_TOL

    ref = f.copy()
    collide_stream(ref, feq, TAU, np.empty_like(f), cs_smag=CS)
    dev = be.from_host(f.copy())
    be.collide_stream(
        dev, be.from_host(feq), TAU, be.empty((Q, NY, NX)), cs_smag=CS
    )
    assert np.abs(be.to_host(dev) - ref).max() < PARITY_TOL


@warp_only
def test_warp_with_the_closure_off_is_bitwise_its_own_bgk_kernel():
    """Constraint 19 on the GPU, at the kernel level.

    **Q-201** asked whether a term that is algebraically zero stays bitwise
    inert on a device that contracts ``x * a + b`` into one rounding
    (**D-053**). The recorded answer is that the question is never asked: with
    ``cs_smag = 0`` the backend launches the *Phase 1 kernel*, unmodified, so
    equality is by construction. Rung F measures the same claim over 1000 steps
    of Rung 3's case against the frozen oracle in :mod:`validate.les`; this is
    its cheap unit-level form.
    """
    be = get_backend("warp")
    f = random_state(22)
    feq = feq_of(f)

    a = be.from_host(f.copy())
    b = be.from_host(f.copy())
    be.collide(a, be.from_host(feq), TAU)
    be.collide(b, be.from_host(feq), TAU, cs_smag=0.0)
    assert np.array_equal(be.to_host(a), be.to_host(b))

    a = be.from_host(f.copy())
    b = be.from_host(f.copy())
    be.collide_stream(a, be.from_host(feq), TAU, be.empty((Q, NY, NX)))
    be.collide_stream(
        b, be.from_host(feq), TAU, be.empty((Q, NY, NX)), cs_smag=0.0
    )
    assert np.array_equal(be.to_host(a), be.to_host(b))


@warp_only
def test_warp_folds_the_closures_scalars_host_side_in_numpys_order():
    """**D-057**: a ``float64``-then-rounded scalar is rounded once, on the host.

    The three the closure needs are ``18 sqrt(2) Cs^2``, ``tau`` and ``tau^2``,
    and :meth:`WarpBackend._smag_scalars` must produce exactly what
    :func:`lbm.core.smagorinsky_tau_eff` folds — not a value a kernel could have
    recomputed in ``float32``, which would be a second rounding.
    """
    be = get_backend("warp")
    coeff, tau32, tau_sq = be._smag_scalars(TAU, CS)

    assert coeff == float(np.float32(SMAG_Q_COEFF * CS * CS))
    assert tau32 == float(np.float32(TAU))
    assert tau_sq == float(np.float32(np.float32(TAU) * np.float32(TAU)))
    # tau^2 rounded once from float64 is a *different* number in general; the
    # point of the expression above is that it is not this one.
    assert tau_sq == float(np.float32(np.float32(TAU) ** 2))

    with pytest.raises(ValueError):
        be._smag_scalars(0.5, CS)
    with pytest.raises(ValueError):
        be._smag_scalars(TAU, -1e-9)


@warp_only
def test_warp_never_restates_a_lattice_constant_for_the_closure():
    """Constraint 4, extended to :data:`lbm.core.SMAG_Q_COEFF` by T202.

    The coefficient is imported and folded host-side; no kernel may carry it,
    and ``18``, ``sqrt(2)`` and ``25.45...`` must not appear as literals in the
    module. The check is over the source of ``lbm/backends/warp_backend.py``
    itself, in the shape ``tests/test_backends.py`` already uses for ``E``,
    ``W`` and ``OPP``.
    """
    src = (REPO / "lbm" / "backends" / "warp_backend.py").read_text(
        encoding="utf-8"
    )
    body = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "from lbm.core import" in body and "SMAG_Q_COEFF" in body
    for literal in ("25.45", "18.0 * np", "np.sqrt(2"):
        assert literal not in body, f"{literal!r} restated in the warp backend"


@warp_only
def test_the_closure_adds_nothing_to_the_checkpoint_on_warp(tmp_path):
    """Constraint 11 with the closure on: ``tau_eff`` is derived, not state.

    ``DOCS/TASKS3.md`` § T202 asks for this to be asserted rather than assumed.
    ``f``, ``solid`` and ``step_count`` are still the entire checkpoint
    (**D-022**, **D-050**), and a restart is still bit-identical *within* the
    backend — which is the whole claim, since a closure that carried hidden
    state would resume onto a different ``tau_eff`` and diverge on step one.
    """
    import pickle

    from lbm.runner import load_checkpoint, save_checkpoint

    solid = channel_walls(NY, NX) | circle(NY, NX, NX / 4.0, NY / 2.0, 3.0)
    cfg = SimConfig(
        ny=NY,
        nx=NX,
        tau=TAU,
        inlet_U=0.05,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        check_geometry=False,
        backend="warp",
        cs_smag=CS,
    )

    sim = Sim(cfg, solid)
    sim.run_steps(60)
    path = save_checkpoint(sim, tmp_path / "les.pkl")
    sim.run_steps(60)
    reference = sim.host_f()

    with open(path, "rb") as fh:
        state = pickle.load(fh)
    assert set(state) == {"f", "solid", "step_count", "config", "format"}

    resumed = load_checkpoint(path)
    assert resumed.config.cs_smag == CS
    resumed.run_steps(60)
    assert np.array_equal(resumed.host_f(), reference)


def test_the_closure_is_off_everywhere_it_is_not_being_tested():
    """T201: ``git grep cs_smag`` finds only definitions, defaults and Rung F.

    The machine-checked form of "the closure defaults off", done over the
    **syntax** rather than over the text so that a docstring cannot pass or fail
    it. Every ``cs_smag`` parameter declared anywhere in ``lbm/``, ``flow/`` or
    ``validate/`` must default to ``0.0``, and every call that passes
    ``cs_smag=`` must pass either that same literal or a variable threaded
    through from a caller — never a non-zero constant. ``validate/les.py`` is
    exempt because turning the closure on is that rung's entire job.

    ``flow/`` may mention it since **T204**, and the rule there is narrower and
    sharper: the product layer may *plan* the closure and *pass* what it planned,
    but **no function anywhere in ``flow/`` may take a ``cs_smag`` parameter**.
    That is constraint 13 read exactly — ``Cs`` is a planned, printed quantity
    like ``tau`` and never something a caller fills in — and it is what stops
    the band from quietly becoming a knob. ``flow.autoconfig.Plan.cs_smag`` is a
    field on a frozen output record, which is the same **D-060** exemption
    ``tau`` and ``dx`` already live under.
    """
    import ast

    bad_defaults: list[str] = []
    bad_calls: list[str] = []
    flow_params: list[str] = []

    # The two rungs whose entire job is turning the closure on. Everything else
    # is held to "declared with a 0.0 default, passed only as a variable".
    closure_rungs = {"validate/les.py", "validate/fidelity.py"}

    for tree_name in ("lbm", "flow", "validate"):
        for path in sorted((REPO / tree_name).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "cs_smag" not in text:
                continue
            rel = path.relative_to(REPO).as_posix()
            if tree_name == "flow":
                for node in ast.walk(ast.parse(text, filename=rel)):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    args = node.args
                    if "cs_smag" in [
                        a.arg for a in args.args + args.kwonlyargs + args.posonlyargs
                    ]:
                        flow_params.append(f"{rel}:{node.lineno}: {node.name}")
            if rel in closure_rungs:
                continue

            tree = ast.parse(text, filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = node.args
                    names = [a.arg for a in args.args + args.kwonlyargs]
                    if "cs_smag" not in names:
                        continue
                    defaults = dict(
                        zip(
                            [a.arg for a in args.args][-len(args.defaults) :],
                            args.defaults,
                        )
                    ) if args.defaults else {}
                    defaults.update(
                        {
                            a.arg: d
                            for a, d in zip(args.kwonlyargs, args.kw_defaults)
                            if d is not None
                        }
                    )
                    default = defaults.get("cs_smag")
                    if default is None:
                        # A **required** cs_smag is the closure's own three
                        # entry points (smagorinsky_tau_eff, smagorinsky_omega,
                        # eddy_viscosity): a caller of the model has to say what
                        # Cs it means, and there is no way to switch anything on
                        # by omission. Only a *default* can default the closure
                        # on, so only a default is checked.
                        continue
                    if not (
                        isinstance(default, ast.Constant) and default.value == 0.0
                    ):
                        bad_defaults.append(
                            f"{rel}:{node.lineno}: {node.name} defaults "
                            f"cs_smag to something other than 0.0"
                        )
                elif isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg != "cs_smag":
                            continue
                        value = kw.value
                        if isinstance(value, ast.Constant):
                            if value.value != 0.0:
                                bad_calls.append(
                                    f"{rel}:{kw.value.lineno}: "
                                    f"cs_smag={value.value!r}"
                                )
                        elif not isinstance(value, (ast.Name, ast.Attribute)):
                            bad_calls.append(
                                f"{rel}:{kw.value.lineno}: cs_smag=<expression>"
                            )

    assert not flow_params, (
        "CLAUDE.md constraint 13: a function in flow/ takes cs_smag as a "
        f"parameter, which makes it a knob rather than a planned quantity: "
        f"{flow_params}"
    )
    assert not bad_defaults, f"the closure does not default off: {bad_defaults}"
    assert not bad_calls, f"the closure is switched on outside Rung F: {bad_calls}"


def test_the_frozen_oracle_is_still_frozen():
    """Rung F's Phase 1 transcription matches the committed Phase 1 source.

    The oracle only means anything if it is Phase 1's arithmetic, so this test
    asks ``git`` for ``lbm/core.py`` as it stood at ``HEAD`` before this session
    and compares the three operations of ``collide``. It skips rather than fails
    where git is unavailable — the transcription is still reviewed by eye and by
    :func:`test_collide_with_cs_zero_is_bitwise_phase_1` either way.
    """
    try:
        blob = subprocess.run(
            ["git", "show", "HEAD:lbm/core.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git is not available")
    if blob.returncode != 0:  # pragma: no cover
        pytest.skip("lbm/core.py is not in HEAD")

    committed = blob.stdout
    if "cs_smag" in committed:  # pragma: no cover
        pytest.skip("HEAD already contains T201; the pre-closure source is older")

    for line in (
        "    one_minus_omega = np.float32(1.0 - 1.0 / tau)",
        "    f -= feq",
        "    f *= one_minus_omega",
        "    f += feq",
    ):
        assert line in committed, line

    frozen = Path(REPO / "validate" / "les.py").read_text(encoding="utf-8")
    assert "one_minus_omega = np.float32(1.0 - 1.0 / tau)" in frozen
