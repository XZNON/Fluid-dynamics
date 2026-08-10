"""Unit tests for the moving wall and the cavity harness (T003).

Pins :func:`lbm.boundary.moving_wall` and the pieces of ``validate/cavity.py``
that are arithmetic rather than physics, so that a Rung 2 failure has somewhere
to point. The integration test is ``validate/cavity.py`` itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from lbm.boundary import bounce_back, moving_wall
from lbm.core import E, OPP, Q, W
from validate.cavity import (
    GHIA_SUSPECT,
    GHIA_U,
    GHIA_V,
    GHIA_X,
    GHIA_Y,
    cavity_masks,
    scored_mask,
    tau_for,
    vortex_centre,
)

NY, NX = 7, 9


def random_f(seed: int = 0) -> np.ndarray:
    """A positive, plausible ``(9, ny, nx)`` distribution."""
    rng = np.random.default_rng(seed)
    f = W[:, None, None] * rng.uniform(0.9, 1.1, size=(Q, NY, NX))
    return np.ascontiguousarray(f, dtype=np.float32)


def wall_mask() -> np.ndarray:
    """Top row solid."""
    solid = np.zeros((NY, NX), dtype=bool)
    solid[-1, :] = True
    return solid


# --- moving_wall --------------------------------------------------------------


def test_zero_wall_velocity_is_plain_bounce_back() -> None:
    """``u_wall = 0`` must reproduce :func:`bounce_back` exactly."""
    f_pre = random_f(1)
    solid = wall_mask()

    a = random_f(2)
    b = a.copy()

    bounce_back(a, f_pre, solid)
    moving_wall(b, f_pre, solid, (0.0, 0.0))

    np.testing.assert_array_equal(a, b)


def test_correction_is_the_ladd_term() -> None:
    """``f[i] = f_pre[OPP[i]] + 6 w_i rho_w (e_i . u_wall)`` on wall cells."""
    f_pre = random_f(3)
    f = random_f(4)
    solid = wall_mask()
    u_wall = (0.07, -0.02)
    rho_w = 0.98

    expected = f.copy()
    for i in range(Q):
        eu = float(E[i, 0]) * u_wall[0] + float(E[i, 1]) * u_wall[1]
        expected[i][solid] = (
            f_pre[OPP[i]][solid] + np.float32(6.0 * float(W[i]) * rho_w * eu)
        )

    moving_wall(f, f_pre, solid, u_wall, rho_w=rho_w)
    np.testing.assert_allclose(f, expected, rtol=0, atol=1e-7)


def test_fluid_cells_are_untouched() -> None:
    f_pre = random_f(5)
    f = random_f(6)
    before = f.copy()
    solid = wall_mask()

    moving_wall(f, f_pre, solid, (0.1, 0.0))

    np.testing.assert_array_equal(f[:, :-1, :], before[:, :-1, :])


def test_correction_conserves_mass_on_the_wall() -> None:
    """``sum_i 6 w_i (e_i . u_w) == 0``, so the lid adds momentum, not mass."""
    u_wall = (0.1, 0.03)
    total = sum(
        6.0 * float(W[i]) * (float(E[i, 0]) * u_wall[0] + float(E[i, 1]) * u_wall[1])
        for i in range(Q)
    )
    assert abs(total) < 1e-12


def test_lid_injects_positive_x_momentum() -> None:
    """A lid moving in ``+x`` must send more ``+x`` than ``-x`` into the fluid.

    The populations that re-enter the fluid from a top-row lid are those with
    ``ey = -1``: ``i = 4, 7, 8``. Their net ``x`` momentum must be positive.
    """
    f_pre = np.broadcast_to(W[:, None, None], (Q, NY, NX)).astype(np.float32).copy()
    f = f_pre.copy()
    solid = wall_mask()

    moving_wall(f, f_pre, solid, (0.1, 0.0))

    downward = [i for i in range(Q) if int(E[i, 1]) == -1]
    net_x = sum(float(E[i, 0]) * float(f[i, -1, 0]) for i in downward)
    assert net_x > 0.0


def test_moving_wall_does_not_allocate() -> None:
    """No temporary per call — ``CLAUDE.md`` § conventions."""
    import tracemalloc

    f_pre = random_f(7)
    f = random_f(8)
    solid = wall_mask()

    moving_wall(f, f_pre, solid, (0.1, 0.0))  # warm up

    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for _ in range(20):
        moving_wall(f, f_pre, solid, (0.1, 0.0))
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grown = sum(s.size_diff for s in snap1.compare_to(snap0, "filename"))
    assert grown < f.nbytes


# --- cavity harness -----------------------------------------------------------


def test_masks_are_disjoint_and_cover_the_border() -> None:
    n = 8
    for corners in ("lid", "wall"):
        static, lid = cavity_masks(n, corners)
        assert not np.any(static & lid)

        border = np.zeros((n, n), dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        np.testing.assert_array_equal(static | lid, border)


def test_corner_modes_differ_in_exactly_two_cells() -> None:
    """Q-003 is a two-cell question — assert that it really is."""
    n = 8
    _, lid_a = cavity_masks(n, "lid")
    _, lid_b = cavity_masks(n, "wall")
    assert int(np.sum(lid_a ^ lid_b)) == 2
    assert lid_a[n - 1, 0] and lid_a[n - 1, n - 1]
    assert not lid_b[n - 1, 0] and not lid_b[n - 1, n - 1]


def test_bad_corner_mode_raises() -> None:
    with pytest.raises(ValueError, match="corners"):
        cavity_masks(8, "middle")


def test_tau_for_inverts_the_viscosity_relation() -> None:
    """``Re = U L / nu`` and ``nu = (tau - 0.5)/3``, constraint 2."""
    nu, tau = tau_for(re=1000, u_lid=0.09, side=256)
    assert nu == pytest.approx(0.09 * 256 / 1000)
    assert (tau - 0.5) / 3.0 == pytest.approx(nu)


def test_tau_for_rejects_a_grid_that_would_need_tau_near_half() -> None:
    """A too-coarse grid at high Re must fail at setup, not at nan time."""
    with pytest.raises(ValueError, match="tau"):
        tau_for(re=1000, u_lid=0.09, side=32)


def test_tau_for_rejects_a_lid_at_the_mach_ceiling() -> None:
    """Constraint 3 is checked at setup, and 0.1 is not "under 0.1"."""
    with pytest.raises(ValueError, match="0.1"):
        tau_for(re=100, u_lid=0.1, side=128)


def test_vortex_centre_finds_a_planted_extremum() -> None:
    """A ``ux`` field whose streamfunction peaks at a known point."""
    side = 64
    k = (np.arange(side) + 0.5) / side
    x0, y0 = 0.6, 0.7
    xx, yy = np.meshgrid(k, k)
    # psi = -exp(-((x-x0)^2 + (y-y0)^2)/s^2); ux = d psi / dy, and the
    # streamfunction reconstruction integrates it straight back.
    s2 = 0.02
    psi = -np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / s2)
    ux = np.gradient(psi, k, axis=0)

    vx, vy = vortex_centre(ux, side)
    assert abs(vx - x0) * side < 2.0
    assert abs(vy - y0) * side < 2.0


def test_only_one_reference_point_is_excluded() -> None:
    """The exclusion list is one point of one profile and stays that way.

    If a later change needs to exclude more, that is a decision to argue in
    ``DOCS/STATE1.md``, not a quiet edit — this test is the tripwire.
    """
    assert GHIA_SUSPECT == {(400, "v"): (5,)}
    assert GHIA_X[5] == pytest.approx(0.9063)

    for re in (100, 400, 1000):
        assert scored_mask(re, "u").all()
    assert scored_mask(100, "v").all()
    assert scored_mask(1000, "v").all()
    assert int(np.sum(~scored_mask(400, "v"))) == 1


def test_the_excluded_point_has_the_wrong_curvature_for_its_column() -> None:
    """The stated reason for the exclusion, asserted rather than left in prose.

    Approaching the trough near the right wall the ``v`` profile is concave: the
    tabulated value at ``x = 0.9063`` sits **below** the straight chord joining
    its two neighbours at ``x = 0.9453`` and ``x = 0.8594``. Re 100 and Re 1000
    both do. The Re 400 entry sits *above* its chord — the profile would have to
    bend the other way at exactly one station of exactly one Reynolds number.
    """
    lo, mid, hi = 4, 5, 6
    t = (GHIA_X[lo] - GHIA_X[mid]) / (GHIA_X[lo] - GHIA_X[hi])

    def above_chord(v: np.ndarray) -> float:
        chord = v[lo] + t * (v[hi] - v[lo])
        return float(v[mid] - chord)

    assert above_chord(GHIA_V[100]) < 0.0
    assert above_chord(GHIA_V[1000]) < 0.0
    assert above_chord(GHIA_V[400]) > 0.05


def test_ghia_tables_are_the_standard_17_points() -> None:
    assert GHIA_Y.size == 17 and GHIA_X.size == 17
    for re in (100, 400, 1000):
        assert GHIA_U[re].size == 17
        assert GHIA_V[re].size == 17
    # y = 1 is the lid, y = 0 the floor; u there is the boundary value itself.
    assert GHIA_Y[0] == 1.0 and GHIA_Y[-1] == 0.0
    for re in (100, 400, 1000):
        assert GHIA_U[re][0] == 1.0
        assert GHIA_U[re][-1] == 0.0
        assert GHIA_V[re][0] == 0.0
        assert GHIA_V[re][-1] == 0.0
