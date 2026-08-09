"""Unit tests for the timestep: collide, stream, bounce-back, body force (T002).

Covers ``DOCS/IDEA2.md`` § "The method, in the order the code runs it", steps
3-5, plus the Guo forcing pair. The integration test is ``validate/poiseuille.py``
(Rung 1) — these tests pin the pieces so a Rung 1 failure has somewhere to point.
"""

from __future__ import annotations

import numpy as np
import pytest

from lbm.boundary import apply_body_force, bounce_back, force_velocity_shift
from lbm.core import (
    E,
    OPP,
    Q,
    W,
    collide,
    equilibrium,
    macroscopic,
    stream,
)

NY, NX = 7, 9


def random_f(seed: int = 0) -> np.ndarray:
    """A positive, plausible ``(9, ny, nx)`` distribution."""
    rng = np.random.default_rng(seed)
    f = W[:, None, None] * rng.uniform(0.9, 1.1, size=(Q, NY, NX))
    return np.ascontiguousarray(f, dtype=np.float32)


# --- collide -----------------------------------------------------------------


def test_collide_matches_the_literal_expression() -> None:
    """The three in-place ops equal ``f - (f - feq)/tau`` (IDEA2 step 3)."""
    f = random_f(1)
    feq = random_f(2)
    tau = 0.6

    expected = f - (f - feq) / np.float32(tau)
    collide(f, feq, tau)

    assert np.allclose(f, expected, rtol=1e-6, atol=1e-8)


def test_collide_is_in_place_and_allocation_free() -> None:
    """``f`` keeps its buffer, and the call allocates nothing (conventions)."""
    import tracemalloc

    f = random_f(3)
    feq = random_f(4)
    ptr_before = f.__array_interface__["data"][0]

    collide(f, feq, 0.6)  # warm up any one-off numpy internals

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        collide(f, feq, 0.6)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    assert f.__array_interface__["data"][0] == ptr_before
    assert after - before < f.nbytes


def test_collide_leaves_equilibrium_untouched() -> None:
    """A distribution already at equilibrium is a fixed point of collision."""
    rho = np.ones((NY, NX), dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[0] = 0.05
    feq = equilibrium(rho, u)
    f = feq.copy()

    collide(f, feq, 0.7)

    assert np.allclose(f, feq, rtol=0, atol=1e-9)


def test_collide_conserves_mass_and_momentum() -> None:
    """BGK collision changes neither moment (feq shares them by construction)."""
    f = random_f(5)
    rho, u = macroscopic(f)
    feq = equilibrium(rho, u)
    mass_before = f.sum(axis=0).copy()
    mom_before = np.einsum("ic,iyx->cyx", E.astype(np.float32), f)

    collide(f, feq, 0.6)

    assert np.allclose(f.sum(axis=0), mass_before, rtol=1e-6, atol=1e-8)
    # Momentum is a near-cancelling sum of f values of order 0.45, so its
    # float32 round-off floor is ~1e-7 absolute regardless of how small the
    # momentum itself is. Anything tighter tests the dtype, not the physics.
    assert np.allclose(
        np.einsum("ic,iyx->cyx", E.astype(np.float32), f),
        mom_before,
        rtol=1e-4,
        atol=1e-6,
    )


def test_collide_rejects_tau_at_or_below_half() -> None:
    f = random_f(6)
    feq = random_f(7)
    with pytest.raises(ValueError, match="tau"):
        collide(f, feq, 0.5)


# --- stream ------------------------------------------------------------------


@pytest.mark.parametrize("i", range(Q))
def test_stream_moves_a_spike_one_cell_along_e(i: int) -> None:
    """A single-cell spike in direction ``i`` lands on ``cell + E[i]``.

    This is the sign-convention test the T002 contract names: axis 0 shifts by
    ``ey``, axis 1 by ``ex``, contents move *along* ``E[i]``.
    """
    f = np.zeros((Q, NY, NX), dtype=np.float32)
    buf = np.empty_like(f)
    y0, x0 = 3, 4
    f[i, y0, x0] = 1.0

    stream(f, buf)

    ex, ey = int(E[i, 0]), int(E[i, 1])
    assert f[i, (y0 + ey) % NY, (x0 + ex) % NX] == pytest.approx(1.0)
    assert f.sum() == pytest.approx(1.0)


def test_stream_equals_np_roll_for_every_direction() -> None:
    """The block-copy shift reproduces the spec's ``np.roll`` form exactly."""
    f = random_f(8)
    buf = np.empty_like(f)
    expected = np.stack(
        [
            np.roll(np.roll(f[i], int(E[i, 1]), axis=0), int(E[i, 0]), axis=1)
            for i in range(Q)
        ]
    )

    stream(f, buf)

    assert np.array_equal(f, expected)


def test_stream_keeps_the_buffer_identity_and_conserves_mass() -> None:
    f = random_f(9)
    buf = np.empty_like(f)
    ptr_before = f.__array_interface__["data"][0]
    mass_before = float(f.sum(dtype=np.float64))

    out = stream(f, buf)

    assert out is f
    assert f.__array_interface__["data"][0] == ptr_before
    assert float(f.sum(dtype=np.float64)) == pytest.approx(mass_before, rel=1e-6)


def test_stream_wraps_periodically_on_both_axes() -> None:
    """Corner direction 5 = (1, 1) carries an edge cell to the opposite corner."""
    f = np.zeros((Q, NY, NX), dtype=np.float32)
    buf = np.empty_like(f)
    f[5, NY - 1, NX - 1] = 2.0

    stream(f, buf)

    assert f[5, 0, 0] == pytest.approx(2.0)


def test_nine_streams_are_the_identity_on_a_closed_loop() -> None:
    """Streaming direction ``i`` then ``OPP[i]`` returns the original field."""
    f = random_f(10)
    original = f.copy()
    buf = np.empty_like(f)

    # roll everything one way, then undo it by swapping into the opposite slots
    stream(f, buf)
    f[:] = f[OPP]
    stream(f, buf)
    f[:] = f[OPP]

    assert np.allclose(f, original, rtol=0, atol=1e-9)


# --- bounce-back -------------------------------------------------------------


def test_bounce_back_reverses_populations_on_solid_cells() -> None:
    """``f[i] = f_pre[OPP[i]]`` on solid, untouched on fluid (IDEA2 step 5)."""
    f = random_f(11)
    f_pre = random_f(12)
    fluid_before = f.copy()
    solid = np.zeros((NY, NX), dtype=bool)
    solid[0, :] = True
    solid[2, 3] = True

    bounce_back(f, f_pre, solid)

    for i in range(Q):
        assert np.array_equal(f[i][solid], f_pre[OPP[i]][solid])
        assert np.array_equal(f[i][~solid], fluid_before[i][~solid])


def test_bounce_back_gives_zero_velocity_after_a_full_reflection() -> None:
    """A wall row that reflects a symmetric arrival carries no net momentum."""
    f = np.zeros((Q, NY, NX), dtype=np.float32)
    f_pre = random_f(13)
    solid = np.zeros((NY, NX), dtype=bool)
    solid[0, :] = True

    bounce_back(f, f_pre, solid)

    # momentum on the wall row is exactly minus the pre-stream momentum there
    ef = np.einsum("ic,iyx->cyx", E.astype(np.float32), f)
    ef_pre = np.einsum("ic,iyx->cyx", E.astype(np.float32), f_pre)
    assert np.allclose(ef[:, 0, :], -ef_pre[:, 0, :], rtol=1e-6, atol=1e-8)


def test_bounce_back_is_a_no_op_with_an_empty_mask() -> None:
    f = random_f(14)
    before = f.copy()
    bounce_back(f, random_f(15), np.zeros((NY, NX), dtype=bool))
    assert np.array_equal(f, before)


# --- body force (Guo) --------------------------------------------------------


def test_force_velocity_shift_adds_half_the_force_over_rho() -> None:
    rho = np.full((NY, NX), 1.2, dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    g = (1e-4, -3e-5)

    force_velocity_shift(rho, u, g)

    assert np.allclose(u[0], 0.5 * g[0] / 1.2, rtol=1e-5, atol=1e-12)
    assert np.allclose(u[1], 0.5 * g[1] / 1.2, rtol=1e-5, atol=1e-12)


def test_body_force_source_sums_to_zero_over_directions() -> None:
    """``sum_i S_i == 0`` — the forcing adds momentum but never mass."""
    f = np.zeros((Q, NY, NX), dtype=np.float32)
    rho = np.ones((NY, NX), dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[0] = 0.03
    u[1] = -0.01

    apply_body_force(f, rho, u, 0.6, (2e-4, 5e-5))

    assert np.allclose(f.sum(axis=0), 0.0, rtol=0, atol=1e-9)


def test_body_force_adds_the_expected_momentum() -> None:
    """First moment of the source is ``(1 - 1/(2 tau)) F`` (Guo et al. 2002)."""
    f = np.zeros((Q, NY, NX), dtype=np.float32)
    rho = np.ones((NY, NX), dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[0] = 0.04
    tau = 0.8
    g = (3e-4, -1e-4)

    apply_body_force(f, rho, u, tau, g)

    moment = np.einsum("ic,iyx->cyx", E.astype(np.float32), f)
    expected = (1.0 - 0.5 / tau)
    assert np.allclose(moment[0], expected * g[0], rtol=1e-4, atol=1e-10)
    assert np.allclose(moment[1], expected * g[1], rtol=1e-4, atol=1e-10)


def test_body_force_matches_the_direct_guo_formula() -> None:
    """The hoisted, in-place form equals the textbook expression term by term."""
    rng = np.random.default_rng(42)
    f = np.zeros((Q, NY, NX), dtype=np.float32)
    rho = np.ones((NY, NX), dtype=np.float32)
    u = rng.uniform(-0.05, 0.05, size=(2, NY, NX)).astype(np.float32)
    tau = 0.65
    g = np.array([1e-4, -2e-4], dtype=np.float64)

    apply_body_force(f, rho, u, tau, (float(g[0]), float(g[1])))

    ef = E.astype(np.float64)
    u64 = u.astype(np.float64)
    expected = np.empty((Q, NY, NX), dtype=np.float64)
    for i in range(Q):
        eu = ef[i, 0] * u64[0] + ef[i, 1] * u64[1]
        bracket = (
            (ef[i, 0] - u64[0]) / (1 / 3) + eu * ef[i, 0] / (1 / 3) ** 2
        ) * g[0] + (
            (ef[i, 1] - u64[1]) / (1 / 3) + eu * ef[i, 1] / (1 / 3) ** 2
        ) * g[
            1
        ]
        expected[i] = (1.0 - 0.5 / tau) * float(W[i]) * bracket

    assert np.allclose(f, expected, rtol=1e-4, atol=1e-12)


def test_zero_force_is_a_no_op() -> None:
    f = random_f(16)
    before = f.copy()
    rho, u = macroscopic(f)
    apply_body_force(f, rho, u, 0.6, (0.0, 0.0))
    assert np.allclose(f, before, rtol=0, atol=1e-9)
