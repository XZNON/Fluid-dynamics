"""Unit tests for ``lbm.core`` — T001 acceptance criteria (``old-Docs/TASKS1.md``).

The random fields are drawn with ``rho in [0.9, 1.1]`` and ``|u| < 0.1`` because
that is the range the D2Q9 equilibrium is valid over (``CLAUDE.md`` constraint
3, compressibility error scales as Mach squared). Testing wider would be
testing a regime the solver is not allowed to run in.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from lbm.core import CS2, E, OPP, Q, W, equilibrium, macroscopic, nu_from_tau

NY, NX = 7, 11
TOL = 1e-5


def random_state(
    seed: int = 0, ny: int = NY, nx: int = NX, umax: float = 0.09
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Random ``rho`` in ``[0.9, 1.1]`` and ``u`` with ``|u| <= umax < 0.1``."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.9, 1.1, size=(ny, nx)).astype(np.float32)

    # Draw a direction and a magnitude so the |u| bound is exact, not per-component.
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(ny, nx))
    mag = rng.uniform(0.0, umax, size=(ny, nx))
    u = np.stack([mag * np.cos(theta), mag * np.sin(theta)]).astype(np.float32)

    assert np.max(np.hypot(u[0], u[1])) < 0.1
    return rho, u


# --- constants --------------------------------------------------------------


def test_constants_match_spec_in_order() -> None:
    """E, W, OPP, CS2 verbatim from DOCS/IDEA2.md § The method, same index order."""
    assert E.tolist() == [
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]
    ]
    assert np.allclose(
        W, [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
    )
    assert OPP.tolist() == [0, 3, 4, 1, 2, 7, 8, 5, 6]
    assert CS2 == pytest.approx(1 / 3)
    assert E.shape == (Q, 2) and W.shape == (Q,) and OPP.shape == (Q,)


def test_weights_sum_to_one() -> None:
    assert W.sum() == pytest.approx(1.0, abs=np.finfo(np.float32).eps * Q)


def test_opposite_directions_are_negatives() -> None:
    for i in range(Q):
        assert (E[OPP[i]] == -E[i]).all(), f"E[OPP[{i}]] != -E[{i}]"


def test_lattice_speed_of_sound_from_weights() -> None:
    """sum_i w_i * ex_i^2 == cs2 — the identity that makes the equilibrium work."""
    ex = E[:, 0].astype(np.float64)
    ey = E[:, 1].astype(np.float64)
    assert (W * ex * ex).sum() == pytest.approx(CS2, abs=1e-6)
    assert (W * ey * ey).sum() == pytest.approx(CS2, abs=1e-6)


# --- nu_from_tau ------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.51, 0.6, 1.0, 2.5])
def test_nu_from_tau_formula(tau: float) -> None:
    assert nu_from_tau(tau) == pytest.approx((tau - 0.5) / 3)
    assert nu_from_tau(tau) == pytest.approx(CS2 * (tau - 0.5))


@pytest.mark.parametrize("tau", [0.5, 0.4, 0.0, -1.0])
def test_nu_from_tau_rejects_tau_at_or_below_half(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        nu_from_tau(tau)


# --- equilibrium moments ----------------------------------------------------


def test_equilibrium_zeroth_moment_is_rho() -> None:
    rho, u = random_state(seed=1)
    feq = equilibrium(rho, u)
    assert np.max(np.abs(feq.sum(axis=0) - rho)) < TOL


def test_equilibrium_first_moment_is_rho_u() -> None:
    ny, nx = NY, NX
    rho, u = random_state(seed=2, ny=ny, nx=nx)
    feq = equilibrium(rho, u)
    mom = (E.T.astype(np.float32) @ feq.reshape(Q, -1)).reshape(2, ny, nx)
    assert np.max(np.abs(mom - rho * u)) < TOL


def test_equilibrium_at_rest_is_weights_times_rho() -> None:
    rho = np.full((NY, NX), 1.0, dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    feq = equilibrium(rho, u)
    for i in range(Q):
        assert np.allclose(feq[i], W[i], atol=TOL)


def test_equilibrium_second_moment_is_pressure_plus_momentum_flux() -> None:
    """sum_i feq_i ex_i ey_i == rho*(cs2*delta_ab + u_a u_b), to O(u^3)."""
    rho, u = random_state(seed=3)
    feq = equilibrium(rho, u)
    ex = E[:, 0].astype(np.float32)[:, None, None]
    ey = E[:, 1].astype(np.float32)[:, None, None]

    pxx = (feq * ex * ex).sum(axis=0)
    pxy = (feq * ex * ey).sum(axis=0)
    assert np.max(np.abs(pxx - rho * (CS2 + u[0] * u[0]))) < TOL
    assert np.max(np.abs(pxy - rho * u[0] * u[1])) < TOL


# --- round trip -------------------------------------------------------------


def test_macroscopic_equilibrium_round_trip() -> None:
    rho, u = random_state(seed=4)
    rho2, u2 = macroscopic(equilibrium(rho, u))
    assert np.max(np.abs(rho2 - rho)) < TOL
    assert np.max(np.abs(u2 - u)) < TOL


def test_macroscopic_on_uniform_rest_state() -> None:
    rho = np.full((NY, NX), 1.3, dtype=np.float32)
    f = (W[:, None, None] * rho).astype(np.float32)
    rho2, u2 = macroscopic(f)
    assert np.allclose(rho2, 1.3, atol=TOL)
    assert np.max(np.abs(u2)) < TOL


# --- dtype and shape contract (constraint 4) --------------------------------


def test_returned_arrays_are_float32_with_the_agreed_shapes() -> None:
    rho, u = random_state(seed=5)
    feq = equilibrium(rho, u)
    assert feq.dtype == np.float32 and feq.shape == (Q, NY, NX)

    rho2, u2 = macroscopic(feq)
    assert rho2.dtype == np.float32 and rho2.shape == (NY, NX)
    assert u2.dtype == np.float32 and u2.shape == (2, NY, NX)
    assert W.dtype == np.float32
    assert np.issubdtype(E.dtype, np.integer)
    assert np.issubdtype(OPP.dtype, np.integer)


def test_output_buffers_are_written_in_place_not_reallocated() -> None:
    """Preallocation path: the runner (T006) owns the buffers, not these functions."""
    rho, u = random_state(seed=6)
    feq = np.empty((Q, NY, NX), dtype=np.float32)
    work = np.empty((3, NY, NX), dtype=np.float32)
    out = equilibrium(rho, u, feq=feq, work=work)
    assert out is feq

    rho_out = np.empty((NY, NX), dtype=np.float32)
    u_out = np.empty((2, NY, NX), dtype=np.float32)
    rho2, u2 = macroscopic(feq, rho=rho_out, u=u_out)
    assert rho2 is rho_out and u2 is u_out
    assert np.max(np.abs(rho2 - rho)) < TOL
    assert np.max(np.abs(u2 - u)) < TOL


def test_direction_and_axis_order_convention() -> None:
    """u[0] is ux and moves along axis 2 (x); u[1] is uy and moves along axis 1 (y)."""
    rho = np.ones((NY, NX), dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[0] = 0.05  # pure +x flow
    feq = equilibrium(rho, u)
    # Direction 1 is +x, direction 3 is -x: more mass moves downwind.
    assert (feq[1] > feq[3]).all()
    # +y (2) and -y (4) stay balanced.
    assert np.allclose(feq[2], feq[4], atol=TOL)
