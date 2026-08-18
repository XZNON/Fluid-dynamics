"""Tests for :mod:`lbm.backends.warp_backend` — T102, the Warp kernels.

One test per acceptance criterion in ``DOCS/TASKS2.md`` § T102, plus the
invariants the seam already promised in T101 and this backend has to keep: the
constants uploaded once from :mod:`lbm.core`, per-kernel agreement with the
NumPy oracle inside ``1e-6`` in ``f`` units, the spike test that checks
``stream`` **without** going through NumPy, no device allocation after the
first call, and a bit-exact host round trip.

The rung is the other half of the evidence and it is not here: Rung A's table of
measured differences comes from ``myenv/Scripts/python.exe -m validate.parity
--kernels``, and these tests deliberately reuse that script's comparison
functions rather than re-deriving them, so a change to the rung cannot silently
diverge from the unit tests.

Everything skips cleanly where ``warp-lang`` is not installed: T102's contract
is that the Phase 0 rungs are unaffected, and a machine without the dependency
must still show them green rather than erroring.
"""

from __future__ import annotations

import numpy as np
import pytest

from lbm.backends import Backend, available_backends, get_backend
from lbm.core import CS2, E, E_F32, OPP, Q, W

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
    before = [
        warp_backend._e_i32.ptr,
        warp_backend._e_f32.ptr,
        warp_backend._w.ptr,
        warp_backend._opp.ptr,
        warp_backend._cs2.ptr,
    ]

    for _ in range(5):
        warp_backend.macroscopic(f.copy())
        warp_backend.equilibrium(rho.copy(), u.copy())
        warp_backend.collide(f.copy(), warp_backend.equilibrium(rho, u), 0.6)
        warp_backend.stream(f.copy(), np.empty_like(f))

    after = [
        warp_backend._e_i32.ptr,
        warp_backend._e_f32.ptr,
        warp_backend._w.ptr,
        warp_backend._opp.ptr,
        warp_backend._cs2.ptr,
    ]
    assert before == after


# ---------------------------------------------------------------------------
# Criterion — per-kernel agreement with NumPy, <= 1e-6 in f units
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


def test_collide_stays_within_one_ulp_of_numpy(numpy_backend, warp_backend, state):
    """The only difference collide may show is a single rounding.

    ``f = feq + (f - feq)(1 - omega)`` ends in a multiply and an add, which the
    GPU compiler contracts into one fused multiply-add where NumPy rounds
    twice. Measured on the rung's grids that is **1.49e-08**, half an ulp at
    ``f ~ 0.2``, and it is bitwise equal on some data — so the test bounds the
    magnitude rather than asserting either outcome.
    """
    rho, u, f = state
    feq = numpy_backend.equilibrium(rho.copy(), u.copy())

    a, b = f.copy(), f.copy()
    numpy_backend.collide(a, feq, 0.6)
    warp_backend.collide(b, feq, 0.6)
    assert float(np.max(np.abs(a - b))) <= 3e-8


def test_stream_is_bitwise_equal_to_numpy(numpy_backend, warp_backend, state):
    """Streaming is a permutation — no arithmetic, so no room to drift."""
    _, _, f = state
    a, b = f.copy(), f.copy()
    numpy_backend.stream(a, np.empty_like(a))
    warp_backend.stream(b, np.empty_like(b))
    assert np.array_equal(a, b)


def test_equilibrium_ignores_the_work_scratch_and_still_matches(
    numpy_backend, warp_backend, state
):
    """``work`` is a NumPy allocation dodge; a thread uses registers."""
    rho, u, _ = state
    ref = numpy_backend.equilibrium(rho.copy(), u.copy())

    without = warp_backend.equilibrium(rho.copy(), u.copy())
    with_scratch = warp_backend.equilibrium(
        rho.copy(), u.copy(), work=np.zeros((3, NY, NX), dtype=np.float32)
    )

    assert np.array_equal(without, with_scratch)
    assert float(np.max(np.abs(ref - without))) <= TOL


def test_preallocated_outputs_are_written_in_place(warp_backend, state):
    rho, u, f = state

    rho_out = np.empty((NY, NX), dtype=np.float32)
    u_out = np.empty((2, NY, NX), dtype=np.float32)
    got_rho, got_u = warp_backend.macroscopic(f.copy(), rho_out, u_out)
    assert got_rho is rho_out and got_u is u_out

    feq_out = np.empty((Q, NY, NX), dtype=np.float32)
    assert warp_backend.equilibrium(rho, u, feq_out) is feq_out


def test_stream_keeps_f_s_buffer_identity_and_fills_buf(warp_backend, state):
    """``lbm.core.stream``'s contract, which T006's restart test depends on."""
    _, _, f = state
    work = f.copy()
    buf = np.empty_like(work)

    returned = warp_backend.stream(work, buf)

    assert returned is work
    assert np.array_equal(buf, work)


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
# Criterion — no allocation per call
# ---------------------------------------------------------------------------


def test_a_thousand_steps_worth_of_calls_allocate_no_device_memory(warp_backend, state):
    """Device buffers are allocated once per shape and then reused.

    Two assertions, because either alone is weak: the buffer **pointers** must
    not move (a leak that reallocates the same size would keep free memory
    flat only by luck), and free device memory must not fall (a cache that
    grows would keep the pointers stable). The free-memory bound is loose
    enough to survive another process touching the GPU mid-test and still tight
    enough to catch a per-call leak of anything this backend allocates: 1000
    iterations leaking one ``(9, 24, 40)`` buffer would be ~35 MB.
    """
    rho, u, f = state
    feq = warp_backend.equilibrium(rho.copy(), u.copy())
    buf = np.empty_like(f)

    # Warm up: the first call for a shape is the one that allocates.
    warp_backend.macroscopic(f.copy())
    warp_backend.collide(f.copy(), feq, 0.6)
    warp_backend.stream(f.copy(), buf)

    grid = warp_backend._grid(NY, NX)
    ptrs_before = (grid.f.ptr, grid.feq.ptr, grid.buf.ptr, grid.rho.ptr, grid.u.ptr)
    buffers_before = len(warp_backend._buffers)
    free_before = getattr(warp_backend.device, "free_memory", None)

    work = f.copy()
    for _ in range(1000):
        r, uu = warp_backend.macroscopic(work)
        warp_backend.equilibrium(r, uu, feq)
        warp_backend.collide(work, feq, 0.6)
        warp_backend.stream(work, buf)

    grid_after = warp_backend._grid(NY, NX)
    ptrs_after = (
        grid_after.f.ptr,
        grid_after.feq.ptr,
        grid_after.buf.ptr,
        grid_after.rho.ptr,
        grid_after.u.ptr,
    )
    assert ptrs_after == ptrs_before
    assert len(warp_backend._buffers) == buffers_before

    if free_before is not None:
        free_after = warp_backend.device.free_memory
        leaked = free_before - free_after
        assert leaked < 16 * 1024 * 1024, (
            f"device free memory fell by {leaked / 1e6:.1f} MB over 1000 steps' "
            f"worth of kernel calls — something allocates per call"
        )


# ---------------------------------------------------------------------------
# The portability contract, and the guards
# ---------------------------------------------------------------------------


def test_host_round_trip_is_bit_exact(warp_backend, state):
    _, _, f = state
    assert np.array_equal(warp_backend.to_host(warp_backend.from_host(f)), f)


def test_the_host_contract_is_checked_not_trusted(warp_backend):
    with pytest.raises(ValueError, match="9, ny, nx"):
        warp_backend.from_host(np.zeros((4, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        warp_backend.from_host(np.zeros((Q, 8, 8), dtype=np.float64))


def test_to_host_accepts_a_device_array(warp_backend, state):
    """Constraint 4: whatever the backend holds, ``to_host`` yields the layout.

    T103 moves the state onto the device; this is the branch that will carry it,
    and a checkpoint written then must still be a ``(9, ny, nx)`` ``float32``
    host array (**D-050**).
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
    feq = warp_backend.equilibrium(rho.copy(), u.copy())
    with pytest.raises(ValueError, match="greater than 0.5"):
        warp_backend.collide(f.copy(), feq, 0.5)


def test_a_non_contiguous_host_array_is_refused_rather_than_silently_copied(
    warp_backend, state
):
    _, _, f = state
    with pytest.raises(ValueError, match="C-contiguous"):
        warp_backend.stream(f[:, ::2], np.empty_like(f[:, ::2]))


def test_the_boundaries_and_the_fused_path_name_t103(warp_backend, state):
    """Stubs raise ``NotImplementedError`` naming their task (``CLAUDE.md``)."""
    _, _, f = state
    solid = np.zeros((NY, NX), dtype=bool)

    with pytest.raises(NotImplementedError, match="T103"):
        warp_backend.bounce_back(f.copy(), f.copy(), solid)
    with pytest.raises(NotImplementedError, match="T103"):
        warp_backend.collide_stream(f.copy(), f.copy(), 0.6, np.empty_like(f))
