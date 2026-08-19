"""Tests for :mod:`lbm.backends.warp_backend` — T102's kernels, T103's timestep.

One test per acceptance criterion in ``DOCS/TASKS2.md`` § T102 and § T103, plus
the invariants the seam promised in T101 and this backend has to keep: the
constants uploaded once from :mod:`lbm.core`, per-kernel **and per-boundary**
agreement with the NumPy oracle inside ``1e-6`` in ``f`` units, the spike test
that checks ``stream`` **without** going through NumPy, fused and unfused
agreeing **bitwise** on this backend (**D-033**), a whole timestep that stays on
the device, a checkpoint that crosses backends (**D-050**), a restart that is
bit-identical *within* the backend (constraint 11 in its **D-046** form), and no
device allocation per step.

The rung is the other half of the evidence and it is not here: Rung A's tables of
measured differences come from ``myenv/Scripts/python.exe -m validate.parity``,
and these tests deliberately reuse that script's comparison functions rather than
re-deriving them, so a change to the rung cannot silently diverge from the unit
tests.

Everything skips cleanly where ``warp-lang`` is not installed: the contract is
that the Phase 0 rungs are unaffected, and a machine without the dependency must
still show them green rather than erroring.
"""

from __future__ import annotations

import numpy as np
import pytest

from lbm.backends import Backend, available_backends, get_backend
from lbm.core import CS2, E, E_F32, OPP, Q, W
from lbm.geometry import channel_walls, circle
from lbm.runner import Sim, SimConfig, load_checkpoint, save_checkpoint

pytestmark = pytest.mark.skipif(
    "warp" not in available_backends(),
    reason="warp-lang is not installed (myenv/Scripts/pip.exe install warp-lang)",
)

NY, NX = 24, 40

#: The task's tolerance, ``f`` units. It is not a knob — see
#: ``validate/parity.py``'s module docstring.
TOL = 1e-6


@pytest.fixture(scope="module")
def warp_backend():
    """One backend for the module: construction compiles kernels (~2 s)."""
    return get_backend("warp")


@pytest.fixture(scope="module")
def numpy_backend():
    """The oracle (**D-043**)."""
    return get_backend("numpy")


@pytest.fixture(scope="module")
def state():
    """``(rho, u, f)`` from the rung's own generator, so both agree on inputs."""
    from validate.parity import random_state

    return random_state(NY, NX)


def flow_case(backend: str, ny: int = NY, nx: int = NX):
    """A driven channel with an obstacle: inlet, convective outlet, bounce-back.

    Every boundary the timestep runs, in one small case.

    Args:
        backend: registry name for :attr:`lbm.runner.SimConfig.backend`.
        ny: rows.
        nx: columns.

    Returns:
        ``(config, solid)``.
    """
    solid = channel_walls(ny, nx) | circle(ny, nx, nx / 3.0, ny / 2.0, 3.0)
    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=0.6,
        inlet_U=0.05,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        check_geometry=False,
        backend=backend,
    )
    return cfg, solid


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def test_warp_backend_satisfies_the_protocol(warp_backend):
    assert isinstance(warp_backend, Backend)
    assert warp_backend.name == "warp"


def test_the_registry_reaches_it_by_name(warp_backend):
    assert "warp" in available_backends()
    assert type(warp_backend).__name__ == "WarpBackend"


# ---------------------------------------------------------------------------
# Criterion — the nine constants come from lbm.core, uploaded once
# ---------------------------------------------------------------------------


def test_the_constants_on_the_device_are_lbm_core_s(warp_backend):
    """Constraint 4: uploaded, never redefined.

    Reading them back off the device is the check that matters — an AST scan
    (``tests/test_backends.py``) proves nothing was *assigned*, this proves the
    values the kernels actually index are core's.
    """
    assert np.array_equal(warp_backend._e_i32.numpy(), E)
    assert np.array_equal(warp_backend._e_f32.numpy(), E_F32)
    assert np.array_equal(warp_backend._w.numpy(), W)
    assert np.array_equal(warp_backend._opp.numpy(), OPP)
    assert warp_backend._cs2.numpy()[0] == np.float32(CS2)


def test_the_constants_are_uploaded_once_not_per_call(warp_backend, state):
    """The device pointers must not move across kernel calls."""
    rho, u, f = state
    be = warp_backend
    names = ("_e_i32", "_e_f32", "_w", "_opp", "_cs2")
    before = [getattr(be, n).ptr for n in names]

    f_d, rho_d, u_d = be.upload(f), be.upload(rho), be.upload(u)
    buf = be.empty((Q, NY, NX))
    for _ in range(5):
        be.macroscopic(f_d, rho_d, u_d)
        feq = be.equilibrium(rho_d, u_d)
        be.collide(f_d, feq, 0.6)
        be.stream(f_d, buf)

    assert [getattr(be, n).ptr for n in names] == before


# ---------------------------------------------------------------------------
# Criterion — per-kernel and per-boundary agreement with NumPy, <= 1e-6
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [(16, 16), (NY, NX), (64, 96)])
def test_every_kernel_agrees_with_numpy_within_tolerance(grid):
    """Rung A's own comparison, at unit-test size (``DOCS/IDEA3.md`` Rung A)."""
    from validate.parity import compare_kernels

    ny, nx = grid
    for c in compare_kernels("warp", ny, nx):
        assert c.max_abs <= TOL, (
            f"{c.kernel}/{c.quantity} on {ny}x{nx}: max|d| = {c.max_abs:.3e} "
            f"> {TOL:.0e} — not explainable by float ordering "
            f"(DOCS/TASKS2.md T102)"
        )


@pytest.mark.parametrize("grid", [(24, 48), (64, 96)])
def test_every_boundary_agrees_with_numpy_within_tolerance(grid):
    """``DOCS/TASKS2.md`` § T103, first criterion: **each separately**."""
    from validate.parity import compare_boundaries

    ny, nx = grid
    seen = set()
    for c in compare_boundaries("warp", ny, nx):
        seen.add(c.kernel)
        assert c.max_abs <= TOL, (
            f"{c.kernel}/{c.quantity} on {ny}x{nx}: max|d| = {c.max_abs:.3e} "
            f"> {TOL:.0e} (DOCS/TASKS2.md T103)"
        )
    # Every boundary Phase 0 ships, named, so a boundary that stops being
    # compared fails here rather than passing by absence.
    assert {
        "bounce_back",
        "moving_wall",
        "inlet_velocity",
        "outlet(copy)",
        "outlet(conv)",
        "force_shift",
        "body_force",
    } <= seen


def test_the_reflections_are_bitwise_because_they_are_assignments():
    """``bounce_back`` does no arithmetic, so anything but bitwise is a bug.

    Recorded as its own test because **D-053** makes the same point for
    ``stream`` and ``macroscopic``: knowing *which* comparisons are bitwise is
    what turns a future difference there into a bug rather than float ordering.
    """
    from validate.parity import compare_boundaries

    bitwise = {
        c.kernel: c.bitwise
        for c in compare_boundaries("warp", NY, NX)
        if c.kernel in ("bounce_back", "outlet(copy)", "moving_wall(u=0)")
    }
    assert all(bitwise.values()), bitwise


def test_collide_stays_within_one_ulp_of_numpy(numpy_backend, warp_backend, state):
    """The only difference collide may show is a single rounding.

    ``f = feq + (f - feq)(1 - omega)`` ends in a multiply and an add, which the
    GPU compiler contracts into one fused multiply-add where NumPy rounds
    twice. Measured on the rung's grids that is **1.49e-08**, half an ulp at
    ``f ~ 0.2``, and it is bitwise equal on some data — so the test bounds the
    magnitude rather than asserting either outcome.
    """
    rho, u, f = state
    feq = numpy_backend.download(
        numpy_backend.equilibrium(numpy_backend.upload(rho), numpy_backend.upload(u))
    )

    a = numpy_backend.upload(f)
    b = warp_backend.upload(f)
    numpy_backend.collide(a, numpy_backend.upload(feq), 0.6)
    warp_backend.collide(b, warp_backend.upload(feq), 0.6)
    diff = np.max(np.abs(numpy_backend.download(a) - warp_backend.download(b)))
    assert float(diff) <= 3e-8


def test_stream_is_bitwise_equal_to_numpy(numpy_backend, warp_backend, state):
    """Streaming is a permutation — no arithmetic, so no room to drift."""
    _, _, f = state
    a = numpy_backend.upload(f)
    b = warp_backend.upload(f)
    numpy_backend.stream(a, numpy_backend.empty((Q, NY, NX)))
    warp_backend.stream(b, warp_backend.empty((Q, NY, NX)))
    assert np.array_equal(numpy_backend.download(a), warp_backend.download(b))


def test_equilibrium_ignores_the_work_scratch_and_still_matches(
    numpy_backend, warp_backend, state
):
    """``work`` is a NumPy allocation dodge; a thread uses registers."""
    rho, u, _ = state
    ref = numpy_backend.download(
        numpy_backend.equilibrium(numpy_backend.upload(rho), numpy_backend.upload(u))
    )

    rho_d, u_d = warp_backend.upload(rho), warp_backend.upload(u)
    without = warp_backend.download(warp_backend.equilibrium(rho_d, u_d)).copy()
    with_scratch = warp_backend.download(
        warp_backend.equilibrium(rho_d, u_d, work=warp_backend.empty((3, NY, NX)))
    )

    assert np.array_equal(without, with_scratch)
    assert float(np.max(np.abs(ref - without))) <= TOL


def test_preallocated_outputs_are_written_in_place(warp_backend, state):
    rho, u, f = state
    rho_d, u_d, f_d = (warp_backend.upload(a) for a in (rho, u, f))

    rho_out = warp_backend.empty((NY, NX))
    u_out = warp_backend.empty((2, NY, NX))
    got_rho, got_u = warp_backend.macroscopic(f_d, rho_out, u_out)
    assert got_rho is rho_out and got_u is u_out

    feq_out = warp_backend.empty((Q, NY, NX))
    assert warp_backend.equilibrium(rho_d, u_d, feq_out) is feq_out


def test_stream_keeps_f_s_buffer_identity_and_fills_buf(warp_backend, state):
    """``lbm.core.stream``'s contract, which T006's restart test depends on."""
    _, _, f = state
    work = warp_backend.upload(f)
    buf = warp_backend.empty((Q, NY, NX))

    returned = warp_backend.stream(work, buf)

    assert returned is work
    assert np.array_equal(warp_backend.download(buf), warp_backend.download(work))


# ---------------------------------------------------------------------------
# Criterion — the spike test, on the GPU, independent of parity
# ---------------------------------------------------------------------------


def test_a_spike_lands_one_cell_along_e_for_all_nine_directions():
    """Phase 0's ``stream`` check, re-run on the device.

    Parity alone would pass two backends that are wrong in the same way; this
    has a known answer of its own (``lbm.core.stream``'s sign convention).
    """
    from validate.parity import spike_directions

    assert spike_directions("warp") == [True] * Q


# ---------------------------------------------------------------------------
# Criterion — the fused path, bitwise against the unfused one (D-033)
# ---------------------------------------------------------------------------


def test_fused_and_unfused_agree_bitwise_on_this_backend(warp_backend, state):
    """**D-033**: the fusion is a speed switch, never a physics one.

    Bitwise *on a given backend* — not across them. The fused pass writes the
    same three operations in the same order into the ``f_bb`` snapshot and then
    streams it, which is exactly what the unfused sequence
    ``collide -> bounce_back -> copy -> stream`` does.
    """
    rho, u, f = state
    be = warp_backend
    solid = channel_walls(NY, NX) | circle(NY, NX, NX / 3.0, NY / 2.0, 3.0)
    solid_d = be.upload(solid)
    feq = be.equilibrium(be.upload(rho), be.upload(u))

    unfused = be.upload(f)
    f_pre = be.upload(f)
    f_bb_u = be.empty((Q, NY, NX))
    be.collide(unfused, feq, 0.6)
    be.bounce_back(unfused, f_pre, solid_d)
    be.copy(f_bb_u, unfused)
    be.stream(unfused, be.empty((Q, NY, NX)))

    fused = be.upload(f)
    f_bb_f = be.empty((Q, NY, NX))
    be.collide_stream(
        fused,
        feq,
        0.6,
        be.empty((Q, NY, NX)),
        f_pre=be.upload(f),
        solid=solid_d,
        f_bb=f_bb_f,
    )

    assert np.array_equal(be.download(unfused).copy(), be.download(fused))
    assert np.array_equal(be.download(f_bb_u).copy(), be.download(f_bb_f))


def test_the_fused_path_may_alias_f_as_f_pre_only_when_f_bb_is_given(
    warp_backend, state
):
    """The one aliasing :meth:`lbm.runner.Sim.step` relies on, checked.

    With ``f_bb`` supplied the fused pass never writes ``f`` before reading it,
    so passing ``f`` where **D-011**'s pre-collision copy would go is bitwise
    identical — and that is what lets the timestep skip a whole ``(9, ny, nx)``
    copy per step. This test is the argument made executable.
    """
    rho, u, f = state
    be = warp_backend
    solid = channel_walls(NY, NX) | circle(NY, NX, NX / 3.0, NY / 2.0, 3.0)
    solid_d = be.upload(solid)
    feq = be.equilibrium(be.upload(rho), be.upload(u))

    with_copy = be.upload(f)
    bb_a = be.empty((Q, NY, NX))
    be.collide_stream(
        with_copy, feq, 0.6, be.empty((Q, NY, NX)),
        f_pre=be.upload(f), solid=solid_d, f_bb=bb_a,
    )

    aliased = be.upload(f)
    bb_b = be.empty((Q, NY, NX))
    be.collide_stream(
        aliased, feq, 0.6, be.empty((Q, NY, NX)),
        f_pre=aliased, solid=solid_d, f_bb=bb_b,
    )

    assert np.array_equal(be.download(with_copy).copy(), be.download(aliased))


def test_collide_stream_refuses_solid_without_f_pre(warp_backend, state):
    _, _, f = state
    be = warp_backend
    solid = be.upload(np.zeros((NY, NX), dtype=bool))
    with pytest.raises(ValueError, match="f_pre"):
        be.collide_stream(
            be.upload(f), be.upload(f), 0.6, be.empty((Q, NY, NX)), solid=solid
        )


# ---------------------------------------------------------------------------
# Criterion — the whole timestep, on the device
# ---------------------------------------------------------------------------


def test_the_state_lives_on_the_device(warp_backend):
    """T103 supersedes **D-052**: ``Sim`` owns device arrays, not host ones."""
    import warp as wp

    cfg, solid = flow_case("warp")
    sim = Sim(cfg, solid)

    for name in ("f", "f_pre", "f_bb", "buf", "rho", "u", "feq", "work"):
        assert isinstance(getattr(sim, name), wp.array), name
    assert str(sim.f.device) == str(sim.backend.device)


def test_a_whole_step_matches_numpy_and_stays_bounded():
    """Rung A's whole-step half, at unit-test size (``DOCS/TASKS2.md`` T103)."""
    from validate.parity import STEP_TOL, whole_step

    points = whole_step("warp", ladder=(10, 100))
    for pt in points:
        assert pt.finite
        assert pt.du_over_u < STEP_TOL, (
            f"{pt.steps} steps: max|du|/U = {pt.du_over_u:.3e} >= {STEP_TOL:.0e}"
        )


def test_the_host_accessors_read_the_device_without_allocating(warp_backend):
    """:meth:`lbm.runner.Sim.host_u` and friends reuse one mirror per array."""
    cfg, solid = flow_case("warp")
    sim = Sim(cfg, solid)
    sim.run_steps(3)

    first = sim.host_u()
    sim.run_steps(3)
    second = sim.host_u()
    assert first is second  # the mirror, allocated once
    assert first.shape == (2, cfg.ny, cfg.nx) and first.dtype == np.float32


# ---------------------------------------------------------------------------
# Criterion — the checkpoint crosses backends; restart within one is bitwise
# ---------------------------------------------------------------------------


def test_a_checkpoint_written_on_warp_resumes_on_numpy(tmp_path):
    """**D-050** and constraint 4: ``f`` goes out through ``to_host``."""
    from validate.parity import STEP_TOL

    cfg, solid = flow_case("warp")
    sim = Sim(cfg, solid)
    sim.run_steps(50)

    path = save_checkpoint(sim, tmp_path / "warp.pkl")
    resumed = load_checkpoint(path, backend="numpy")

    assert resumed.config.backend == "numpy"
    assert resumed.step_count == sim.step_count
    assert np.array_equal(resumed.host_f(), sim.host_f())

    sim.run_steps(50)
    resumed.run_steps(50)
    du = float(np.max(np.abs(sim.host_u() - resumed.host_u()))) / cfg.inlet_U
    assert du < STEP_TOL, du


def test_restart_within_warp_is_bit_identical(tmp_path):
    """Constraint 11 in its **D-046** form: bitwise *within* a backend."""
    cfg, solid = flow_case("warp")
    sim = Sim(cfg, solid)
    sim.run_steps(40)

    path = save_checkpoint(sim, tmp_path / "same.pkl")
    resumed = load_checkpoint(path)
    assert resumed.config.backend == "warp"

    sim.run_steps(40)
    resumed.run_steps(40)
    assert np.array_equal(sim.host_f(), resumed.host_f())


# ---------------------------------------------------------------------------
# Criterion — no allocation per step
# ---------------------------------------------------------------------------


def test_a_thousand_timesteps_allocate_no_device_memory(warp_backend):
    """Device buffers are allocated once by ``Sim`` and then reused.

    Two assertions, because either alone is weak: the buffer **pointers** must
    not move (a leak that reallocates the same size would keep free memory flat
    only by luck), and free device memory must not fall (a cache that grows
    would keep the pointers stable). The free-memory bound is loose enough to
    survive another process touching the GPU mid-test and still tight enough to
    catch a per-step leak of anything this size: 1000 steps leaking one
    ``(9, 24, 40)`` buffer would be ~35 MB.
    """
    cfg, solid = flow_case("warp")
    sim = Sim(cfg, solid)
    sim.run_steps(10)  # warm up: compilation and any first-call allocation

    names = ("f", "f_pre", "f_bb", "buf", "rho", "u", "feq", "work", "out_prev")
    before = [getattr(sim, n).ptr for n in names]
    free_before = sim.backend.free_memory()

    sim.run_steps(1000)

    assert [getattr(sim, n).ptr for n in names] == before

    if free_before:
        leaked = free_before - sim.backend.free_memory()
        assert leaked < 16 * 1024 * 1024, (
            f"device free memory fell by {leaked / 1e6:.1f} MB over 1000 "
            f"timesteps — something allocates per step"
        )


# ---------------------------------------------------------------------------
# The portability contract, allocation and transfer, and the guards
# ---------------------------------------------------------------------------


def test_host_round_trip_is_bit_exact(warp_backend, state):
    _, _, f = state
    assert np.array_equal(warp_backend.to_host(warp_backend.from_host(f)), f)


def test_upload_download_round_trips_masks_as_bool(warp_backend):
    """Masks live as ``uint8`` on the device and come back ``bool``."""
    mask = circle(NY, NX, 20.0, 12.0, 4.0)
    dev = warp_backend.upload(mask)
    back = warp_backend.download(dev)
    assert back.dtype == np.bool_
    assert np.array_equal(back, mask)


def test_empty_and_zeros_return_device_arrays_of_the_asked_shape(warp_backend):
    import warp as wp

    a = warp_backend.zeros((Q, NY, NX))
    assert isinstance(a, wp.array) and a.shape == (Q, NY, NX)
    assert not warp_backend.download(a).any()
    b = warp_backend.empty((2, NY))
    assert b.shape == (2, NY)


def test_an_unsupported_dtype_is_refused(warp_backend):
    with pytest.raises(ValueError, match="float32"):
        warp_backend.empty((4, 4), dtype=np.float64)


def test_the_host_contract_is_checked_not_trusted(warp_backend):
    with pytest.raises(ValueError, match="9, ny, nx"):
        warp_backend.from_host(np.zeros((4, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        warp_backend.from_host(np.zeros((Q, 8, 8), dtype=np.float64))


def test_to_host_accepts_a_device_array(warp_backend, state):
    """Constraint 4: whatever the backend holds, ``to_host`` yields the layout.

    Since T103 the state *is* on the device, so this is the branch a checkpoint
    takes, and what it hands back must be a ``(9, ny, nx)`` ``float32`` host
    array (**D-050**).
    """
    import warp as wp

    _, _, f = state
    dev = wp.array(f, dtype=wp.float32, device=warp_backend.device)
    out = warp_backend.to_host(dev)

    assert out.shape == f.shape and out.dtype == np.float32
    assert np.array_equal(out, f)


def test_collide_rejects_tau_at_or_below_a_half(warp_backend, state):
    """Constraint 2, with :func:`lbm.core.collide`'s own message."""
    rho, u, f = state
    be = warp_backend
    feq = be.equilibrium(be.upload(rho), be.upload(u))
    with pytest.raises(ValueError, match="greater than 0.5"):
        be.collide(be.upload(f), feq, 0.5)
