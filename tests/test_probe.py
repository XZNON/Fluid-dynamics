"""Unit tests for the open boundaries and the probes (T005).

Covers ``DOCS/IDEA2.md`` § "The method" step 6 (inlet velocity, outlet
zero-gradient) and § "What to actually draw" (vorticity), plus the three
measurements Rung 3 is scored on: ``Cd``, ``St`` and the steady-state residual.

T005 has no rung of its own — these tests *are* its gate (``DOCS/TASKS1.md``
§ T005). Rung 3 (``validate/cylinder.py``, T007) is what finally audits
:func:`lbm.probe.forces` against ``Cd ~ 1.34``.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np
import pytest

from lbm.boundary import (
    apply_body_force,
    bounce_back,
    force_velocity_shift,
    inlet_profile,
    inlet_velocity,
    outlet_zero_gradient,
)
from lbm.core import (
    CS2,
    E,
    OPP,
    Q,
    W,
    collide,
    equilibrium,
    macroscopic,
    stream,
)
from lbm.geometry import bounding_box, channel_walls, circle, rectangle
from lbm.probe import boundary_links, forces, residual, strouhal, vorticity

NY, NX = 11, 17


def random_f(seed: int = 0, ny: int = NY, nx: int = NX) -> np.ndarray:
    """A positive, plausible ``(9, ny, nx)`` distribution."""
    rng = np.random.default_rng(seed)
    f = W[:, None, None] * rng.uniform(0.9, 1.1, size=(Q, ny, nx))
    return np.ascontiguousarray(f, dtype=np.float32)


# --- inlet: profile ----------------------------------------------------------


def test_uniform_profile_is_flat_on_the_fluid_rows() -> None:
    """``profile='uniform'`` gives every fluid row the same ``ux``."""
    solid = channel_walls(NY, NX)
    u_in = inlet_profile(NY, 0.05, "uniform", solid=solid)

    assert u_in.shape == (2, NY)
    assert u_in.dtype == np.float32
    assert np.allclose(u_in[0, 1:-1], 0.05)
    assert u_in[0, 0] == 0.0 and u_in[0, -1] == 0.0  # solid wall rows
    assert np.all(u_in[1] == 0.0)


def test_parabolic_profile_peaks_at_U_and_vanishes_on_the_D009_wall_planes() -> None:
    """Parabola uses the halfway wall convention (D-009), peak ``U``."""
    solid = channel_walls(NY, NX)
    U = 0.04
    u_in = inlet_profile(NY, U, "parabolic", solid=solid)

    rows = np.arange(NY)
    y0, y1 = 1, NY - 2
    H = y1 - y0 + 1
    y_ = rows[y0 : y1 + 1] - (y0 - 0.5)
    expected = 4.0 * U * y_ * (H - y_) / (H * H)

    assert np.allclose(u_in[0, y0 : y1 + 1], expected, atol=1e-7)
    # zero at the wall planes y0 - 0.5 and y1 + 0.5, i.e. extrapolating the
    # parabola half a cell beyond the first and last fluid rows
    assert u_in[0, y0] > 0.0 and u_in[0, y1] > 0.0
    assert np.isclose(u_in[0].max(), U, rtol=2e-2)


def test_profile_without_a_mask_treats_every_row_as_fluid() -> None:
    u_in = inlet_profile(NY, 0.03, "uniform")
    assert np.allclose(u_in[0], 0.03)


def test_profile_carries_the_cross_stream_component() -> None:
    solid = channel_walls(NY, NX)
    u_in = inlet_profile(NY, 0.03, "uniform", solid=solid, uy=0.001)
    assert np.allclose(u_in[1, 1:-1], 0.001)
    assert u_in[1, 0] == 0.0


def test_profile_warns_above_the_mach_ceiling() -> None:
    """Constraint 3: ``|u| >= 0.1`` warns at setup, not at nan time."""
    with pytest.warns(UserWarning, match="constraint 3"):
        inlet_profile(NY, 0.12, "uniform")


def test_profile_does_not_warn_below_the_ceiling() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inlet_profile(NY, 0.09, "parabolic")


def test_profile_rejects_an_unknown_name_and_a_solid_column() -> None:
    with pytest.raises(ValueError, match="uniform"):
        inlet_profile(NY, 0.05, "plug")
    solid = np.ones((NY, NX), dtype=bool)
    with pytest.raises(ValueError, match="no fluid rows"):
        inlet_profile(NY, 0.05, "uniform", solid=solid)


# --- inlet: Zou-He -----------------------------------------------------------


def test_zou_he_reproduces_the_prescribed_velocity() -> None:
    """The completed column has exactly the ``ux``, ``uy`` that were asked for."""
    f = random_f(11)
    u_in = inlet_profile(NY, 0.06, "parabolic")

    inlet_velocity(f, u_in=u_in)

    rho, u = macroscopic(f)
    assert np.allclose(u[0, :, 0], u_in[0], atol=1e-6)
    assert np.allclose(u[1, :, 0], u_in[1], atol=1e-6)


def test_zou_he_reproduces_a_cross_stream_velocity_too() -> None:
    f = random_f(12)
    u_in = inlet_profile(NY, 0.05, "uniform", uy=0.01)

    inlet_velocity(f, u_in=u_in)

    _, u = macroscopic(f)
    assert np.allclose(u[0, :, 0], 0.05, atol=1e-6)
    assert np.allclose(u[1, :, 0], 0.01, atol=1e-6)


def test_zou_he_density_is_self_consistent() -> None:
    """The rho it solves for is the rho the completed column actually has."""
    f = random_f(13)
    u_in = inlet_profile(NY, 0.07, "uniform")
    fc = f[:, :, 0]
    ux = u_in[0]

    expected_rho = (fc[0] + fc[2] + fc[4] + 2.0 * (fc[3] + fc[6] + fc[7])) / (1.0 - ux)

    inlet_velocity(f, u_in=u_in)

    assert np.allclose(f[:, :, 0].sum(axis=0), expected_rho, rtol=1e-6)


def test_zou_he_touches_only_the_unknown_directions_of_its_column() -> None:
    """Only ``i = 1, 5, 8`` in the inlet column change; nothing else does."""
    f = random_f(14)
    before = f.copy()

    inlet_velocity(f, 0.05)

    assert np.array_equal(f[:, :, 1:], before[:, :, 1:])
    for i in (0, 2, 3, 4, 6, 7):
        assert np.array_equal(f[i, :, 0], before[i, :, 0])
    for i in (1, 5, 8):
        assert not np.array_equal(f[i, :, 0], before[i, :, 0])


def test_zou_he_skips_solid_rows() -> None:
    """Wall cells in the inlet column belong to bounce-back, not to the inlet."""
    solid = channel_walls(NY, NX)
    f = random_f(15)
    before = f.copy()

    inlet_velocity(f, 0.05, solid=solid)

    assert np.array_equal(f[:, 0, 0], before[:, 0, 0])
    assert np.array_equal(f[:, -1, 0], before[:, -1, 0])
    assert not np.array_equal(f[1, 1:-1, 0], before[1, 1:-1, 0])


def test_zou_he_leaves_a_matching_equilibrium_field_alone() -> None:
    """A field already at the inlet's own equilibrium is a fixed point."""
    rho = np.ones((NY, NX), dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[0] = 0.05
    f = equilibrium(rho, u)
    before = f.copy()

    inlet_velocity(f, 0.05)

    assert np.allclose(f, before, atol=1e-7)


def test_zou_he_is_allocation_free_with_cached_buffers() -> None:
    """D-006: pass ``u_in`` and ``work`` and the per-step call allocates nothing."""
    f = random_f(16)
    u_in = inlet_profile(NY, 0.05, "uniform")
    work = np.empty((5, NY), dtype=np.float32)

    inlet_velocity(f, u_in=u_in, work=work)  # warm up numpy internals

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        inlet_velocity(f, u_in=u_in, work=work)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    assert after - before < f.nbytes


# --- outlet ------------------------------------------------------------------


def test_outlet_copies_the_second_to_last_column() -> None:
    f = random_f(21)

    outlet_zero_gradient(f)

    assert np.array_equal(f[:, :, -1], f[:, :, -2])


def test_outlet_can_absorb_at_the_left_edge_too() -> None:
    f = random_f(22)
    outlet_zero_gradient(f, col=0, src=1)
    assert np.array_equal(f[:, :, 0], f[:, :, 1])


NX_PULSE = 400
CS = float(np.sqrt(CS2))


def _pulse_reflection(lam: float | None, nx: int = NX_PULSE, tau: float = 0.6) -> float:
    """Fire a smooth pressure pulse at the outlet; return reflected/incident.

    A one-row, y-periodic domain with no walls: the only physics is a sound
    wave, which splits into a left- and a right-going half travelling at
    ``cs = 0.577`` cells/step. Both edges get the same boundary so the
    left-going half cannot wrap around (``stream`` is periodic) and pollute the
    measurement on the right.

    Timing, all from ``cs``: the right-going half reaches the outlet at about
    ``t = (nx/2)/cs ~ 346``, so its amplitude is sampled just inside the outlet
    over ``t = 250..400``. Anything reflected then travels back inward, and is
    sampled over ``t = 450..650`` in the right half of the domain — a window
    that ends before the *left* edge's reflection can arrive at ``t ~ 690``.
    """
    ny = 1
    rho = np.ones((ny, nx), dtype=np.float32)
    x = np.arange(nx, dtype=np.float32)
    sigma = 10.0
    rho += np.float32(0.01) * np.exp(-((x - nx * 0.5) ** 2) / (2.0 * sigma**2))[None, :]

    f = equilibrium(rho, np.zeros((2, ny, nx), dtype=np.float32))
    feq = np.empty_like(f)
    buf = np.empty_like(f)
    work = np.empty((3, ny, nx), dtype=np.float32)

    prev_r = f[:, :, -1].copy() if lam is not None else None
    prev_l = f[:, :, 0].copy() if lam is not None else None

    incident = 0.0
    reflected = 0.0
    for step in range(1, 701):
        rho_n, u_n = macroscopic(f)
        equilibrium(rho_n, u_n, feq, work)
        collide(f, feq, tau)
        stream(f, buf)
        outlet_zero_gradient(f, prev=prev_r, lam=lam)
        outlet_zero_gradient(f, col=0, src=1, prev=prev_l, lam=lam)

        drho = np.abs(f.sum(axis=0)[0] - 1.0)
        if 250 <= step <= 400:
            incident = max(incident, float(drho[nx - 3]))
        if 450 <= step <= 650:
            reflected = max(reflected, float(drho[nx // 2 : nx - 5].max()))

    return reflected / incident


def test_outlet_reflects_less_than_five_percent_of_a_pressure_pulse() -> None:
    """Acceptance criterion: a pulse crossing the outlet reflects < 5%.

    Met by the convective form at ``lam = cs`` (``DOCS/STATE1.md`` D-021).
    """
    ratio = _pulse_reflection(lam=CS)

    assert ratio < 0.05, f"outlet reflected {100 * ratio:.2f}% of the pulse"


def test_the_plain_copy_outlet_is_the_reflecting_one() -> None:
    """D-021, measured: the bare column copy reflects far more than 5%.

    This is why :func:`outlet_zero_gradient` takes ``prev`` and ``lam`` at all.
    Pinning the bad number keeps the docstring's table honest and stops anyone
    from "simplifying" the convective form away.
    """
    copy_ratio = _pulse_reflection(lam=None)
    convective_ratio = _pulse_reflection(lam=CS)

    assert copy_ratio > 0.2
    assert convective_ratio < copy_ratio / 10.0


def test_convective_outlet_is_worst_tuned_away_from_the_sound_speed() -> None:
    """The minimum sits at ``lam = cs``; mistuning by 2x costs an order of magnitude."""
    at_cs = _pulse_reflection(lam=CS)
    too_fast = _pulse_reflection(lam=2.0 * CS)

    assert at_cs < too_fast


def test_convective_outlet_defaults_lambda_to_the_sound_speed() -> None:
    f = random_f(23)
    prev_a, prev_b = f[:, :, -1].copy(), f[:, :, -1].copy()

    outlet_zero_gradient(f.copy(), prev=prev_a)
    outlet_zero_gradient(f.copy(), prev=prev_b, lam=CS)

    assert np.allclose(prev_a, prev_b, atol=0.0)


def test_convective_outlet_updates_prev_in_place() -> None:
    f = random_f(24)
    prev = f[:, :, -1].copy()
    ptr = prev.__array_interface__["data"][0]

    outlet_zero_gradient(f, prev=prev)

    assert prev.__array_interface__["data"][0] == ptr
    assert np.array_equal(prev, f[:, :, -1])


# --- vorticity ---------------------------------------------------------------


def test_vorticity_of_a_linear_shear_is_constant() -> None:
    """``ux = a y`` gives ``omega = -a`` everywhere, edges included."""
    a = 0.001
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[0] = a * np.arange(NY, dtype=np.float32)[:, None]

    w = vorticity(u)

    assert w.shape == (NY, NX)
    assert w.dtype == np.float32
    assert np.allclose(w, -a, atol=1e-7)


def test_vorticity_of_solid_body_rotation_is_twice_the_rate() -> None:
    """``u = omega x r`` gives a uniform ``2 omega`` (one-sided edges included)."""
    rate = 0.0005
    y, x = np.meshgrid(
        np.arange(NY, dtype=np.float32), np.arange(NX, dtype=np.float32), indexing="ij"
    )
    u = np.empty((2, NY, NX), dtype=np.float32)
    u[0] = -rate * (y - NY / 2)
    u[1] = rate * (x - NX / 2)

    w = vorticity(u)

    assert np.allclose(w, 2.0 * rate, atol=1e-7)


def test_vorticity_sign_convention() -> None:
    """``omega = d(uy)/dx - d(ux)/dy``, with axis 0 = y and axis 1 = x."""
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[1] = 0.002 * np.arange(NX, dtype=np.float32)[None, :]  # uy grows with x

    w = vorticity(u)

    assert np.allclose(w, 0.002, atol=1e-7)


def test_vorticity_edges_are_one_sided_and_interior_is_central() -> None:
    """Matches ``np.gradient``, which is central inside and one-sided at edges."""
    rng = np.random.default_rng(5)
    u = rng.normal(0.0, 0.01, size=(2, NY, NX)).astype(np.float32)

    w = vorticity(u)
    expected = np.gradient(u[1], axis=1) - np.gradient(u[0], axis=0)

    assert np.allclose(w, expected, atol=1e-6)


def test_vorticity_is_nan_on_solid_cells_and_nowhere_else() -> None:
    solid = circle(NY, NX, cx=8.0, cy=5.0, radius=2.0)
    rng = np.random.default_rng(6)
    u = rng.normal(0.0, 0.01, size=(2, NY, NX)).astype(np.float32)

    w = vorticity(u, solid=solid)

    assert np.array_equal(np.isnan(w), solid)


def test_vorticity_is_allocation_free_with_buffers() -> None:
    u = np.zeros((2, NY, NX), dtype=np.float32)
    u[0] = 0.001 * np.arange(NY, dtype=np.float32)[:, None]
    out = np.empty((NY, NX), dtype=np.float32)
    work = np.empty((NY, NX), dtype=np.float32)

    vorticity(u, out=out, work=work)  # warm up

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        vorticity(u, out=out, work=work)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    assert after - before < out.nbytes


# --- boundary links ----------------------------------------------------------


def test_link_list_of_an_isolated_block_counts_every_face_and_corner() -> None:
    """A 3x3 solid block in open fluid has 8 links per surrounding fluid cell layer.

    Counting by hand: each of the 8 fluid cells orthogonally adjacent to a face
    contributes, plus diagonals. The invariant asserted here is the one that
    matters: every link points from fluid to solid, and no link is duplicated.
    """
    solid = rectangle(NY, NX, x0=7, y0=4, x1=9, y1=6)
    links = boundary_links(solid)

    flat_solid = solid.reshape(-1)
    seen: set[tuple[int, int]] = set()
    for i, where in zip(links.dirs, links.idx):
        for flat in where:
            y, x = divmod(int(flat), NX)
            assert not solid[y, x]  # source is fluid
            ny_, nx_ = (y + int(E[i, 1])) % NY, (x + int(E[i, 0])) % NX
            assert solid[ny_, nx_]  # target is solid
            assert (i, int(flat)) not in seen
            seen.add((i, int(flat)))

    assert links.count == len(seen)
    assert links.count > 0
    assert flat_solid.sum() == 9


def test_link_list_is_empty_without_an_obstacle() -> None:
    links = boundary_links(np.zeros((NY, NX), dtype=bool))
    assert links.count == 0
    assert links.dirs == ()


# --- forces ------------------------------------------------------------------


def _uniform_flow(ny: int, nx: int, U: float) -> np.ndarray:
    rho = np.ones((ny, nx), dtype=np.float32)
    u = np.zeros((2, ny, nx), dtype=np.float32)
    u[0] = U
    return equilibrium(rho, u)


def test_forces_are_zero_without_an_obstacle() -> None:
    """Acceptance criterion: uniform flow, no obstacle, ``|Cd| < 1e-6``."""
    U = 0.05
    solid = np.zeros((NY, NX), dtype=bool)
    f = _uniform_flow(NY, NX, U)
    buf = np.empty_like(f)

    f_pre = f.copy()
    stream(f, buf)

    cd, cl = forces(f_pre, f, solid, U=U, D=10.0)

    assert abs(cd) < 1e-6
    assert abs(cl) < 1e-6


def test_forces_vanish_on_a_body_in_quiescent_fluid() -> None:
    """Still fluid at uniform density pushes a block equally from all sides."""
    solid = rectangle(NY, NX, x0=7, y0=4, x1=9, y1=6)
    rho = np.ones((NY, NX), dtype=np.float32)
    u = np.zeros((2, NY, NX), dtype=np.float32)
    f = equilibrium(rho, u)
    f_pre_collision = f.copy()
    buf = np.empty_like(f)

    bounce_back(f, f_pre_collision, solid)
    f_pre = f.copy()
    stream(f, buf)

    cd, cl = forces(f_pre, f, solid, U=0.05, D=3.0)

    assert abs(cd) < 1e-6
    assert abs(cl) < 1e-6


def test_wall_drag_balances_the_body_force_in_poiseuille_flow() -> None:
    """Momentum balance: at steady state the wall drag equals ``gx`` times the fluid area.

    The one test in this file that checks the **magnitude** of
    :func:`lbm.probe.forces`, not just its sign. In a steady body-force-driven
    channel every bit of momentum the force injects has to leave through the
    walls, so ``sum Fx`` over the bounce-back links must equal ``gx * A_fluid``
    exactly, with no empirical constant to hide an error behind. Rung 3's
    ``Cd ~ 1.34`` is the other audit (T007); this one is available now and is
    what makes the ``forces`` implementation more than plausible.

    Setup is Rung 1's (``validate/poiseuille.py``): 22x16, ``tau = 0.6``.
    Passing ``U = D = 1`` makes ``Cd`` simply ``2 Fx``.
    """
    ny, nx = 22, 16
    tau, gx = 0.6, 2.6667e-5
    solid = channel_walls(ny, nx)

    f = equilibrium(
        np.ones((ny, nx), dtype=np.float32), np.zeros((2, ny, nx), dtype=np.float32)
    )
    feq = np.empty_like(f)
    buf = np.empty_like(f)
    work = np.empty((3, ny, nx), dtype=np.float32)
    f_pre_collision = np.empty_like(f)
    f_pre = np.empty_like(f)
    links = boundary_links(solid)

    for _ in range(15000):
        np.copyto(f_pre_collision, f)
        rho, u = macroscopic(f)
        force_velocity_shift(rho, u, (gx, 0.0))
        equilibrium(rho, u, feq, work)
        collide(f, feq, tau)
        apply_body_force(f, rho, u, tau, (gx, 0.0), work)
        bounce_back(f, f_pre_collision, solid)
        np.copyto(f_pre, f)
        stream(f, buf)

    cd, cl = forces(f_pre, f, links, U=1.0, D=1.0)
    fx, fy = cd / 2.0, cl / 2.0
    injected = gx * int((~solid).sum())

    assert np.isclose(fx, injected, rtol=5e-3), f"wall drag {fx} vs injected {injected}"
    assert abs(fy) < 1e-5 * injected / gx


def _channel_forces(
    solid: np.ndarray,
    *,
    U: float = 0.05,
    D: float,
    tau: float = 0.65,
    steps: int = 1200,
    average_over: int = 200,
) -> tuple[float, float]:
    """Run an open channel around ``solid`` and return the late-time mean (Cd, Cl).

    The full T005 timestep, in the order ``DOCS/STATE1.md`` D-011 and D-020 fix:
    pre-collision copy, collide, bounce-back, **pre-stream copy**, stream,
    convective outlet, Zou–He inlet. The last ``average_over`` steps are
    averaged, because the impulsive start rings acoustically for several hundred
    steps — a single instantaneous ``Cd`` this early is a sound wave, not drag.

    This is a *sign and symmetry* harness, not a benchmark: the grid is small and
    the blockage is well over the 10% ``check_mask`` allows. The number itself is
    Rung 3's business (T007).
    """
    ny, nx = solid.shape
    rho = np.ones((ny, nx), dtype=np.float32)
    u = np.zeros((2, ny, nx), dtype=np.float32)
    u[0] = U
    u[0][solid] = 0.0
    f = equilibrium(rho, u)

    feq = np.empty_like(f)
    buf = np.empty_like(f)
    work = np.empty((3, ny, nx), dtype=np.float32)
    f_pre_collision = np.empty_like(f)
    f_pre = np.empty_like(f)
    prev_out = f[:, :, -1].copy()
    u_in = inlet_profile(ny, U, "uniform", solid=solid)
    inlet_work = np.empty((5, ny), dtype=np.float32)
    links = boundary_links(solid)

    cds: list[float] = []
    cls: list[float] = []
    for step in range(steps):
        np.copyto(f_pre_collision, f)
        rho_n, u_n = macroscopic(f)
        equilibrium(rho_n, u_n, feq, work)
        collide(f, feq, tau)
        bounce_back(f, f_pre_collision, solid)
        np.copyto(f_pre, f)
        stream(f, buf)
        outlet_zero_gradient(f, prev=prev_out)
        inlet_velocity(f, solid=solid, u_in=u_in, work=inlet_work)

        if step >= steps - average_over:
            cd, cl = forces(f_pre, f, links, U=U, D=D)
            cds.append(cd)
            cls.append(cl)

    return float(np.mean(cds)), float(np.mean(cls))


def _cylinder_channel(cy: float) -> tuple[np.ndarray, float]:
    """A channel with a cylinder at cross-stream position ``cy``. Returns (mask, D)."""
    ny, nx = 41, 140
    solid = channel_walls(ny, nx) | circle(ny, nx, cx=35.0, cy=cy, radius=4.5)
    y0, y1, _, _ = bounding_box(circle(ny, nx, cx=35.0, cy=cy, radius=4.5))
    return solid, float(y1 - y0 + 1)


def test_forces_on_a_body_in_a_channel_point_downstream() -> None:
    """A cylinder in a ``+x`` channel feels ``+x`` drag and, centred, no lift."""
    solid, D = _cylinder_channel(cy=20.0)

    cd, cl = _channel_forces(solid, D=D)

    assert cd > 0.0, f"a body in a +x flow must feel +x drag (got Cd={cd})"
    assert abs(cl) < 0.02 * abs(cd), f"symmetric case leaked lift: Cd={cd}, Cl={cl}"


def test_lift_flips_sign_when_the_geometry_is_mirrored() -> None:
    """An off-centre cylinder feels lift, and mirroring the channel reverses it.

    Pins the ``y`` half of the momentum-exchange sum against the ``x`` half: a
    sign error in ``ey`` would survive the centred test above but not this one.
    """
    solid, D = _cylinder_channel(cy=14.0)
    mirrored = solid[::-1].copy()

    _, cl = _channel_forces(solid, D=D)
    _, cl_mirror = _channel_forces(mirrored, D=D)

    assert abs(cl) > 1e-3, f"an off-centre body should feel lift (got Cl={cl})"
    assert np.isclose(cl, -cl_mirror, rtol=1e-2), f"{cl} vs {cl_mirror}"


def test_the_two_snapshot_form_equals_the_solid_side_form() -> None:
    """Pins D-020's timing: ``f_post[opp(i)](x_f) == f_pre[opp(i)](x_f + e_i)``.

    Both readings of the returning population must agree, or the pre-stream copy
    is being taken at the wrong point in the step.
    """
    solid = circle(NY, NX, cx=8.0, cy=5.0, radius=2.0)
    f = random_f(31)
    f_pre_collision = f.copy()
    buf = np.empty_like(f)

    bounce_back(f, f_pre_collision, solid)
    f_pre = f.copy()
    stream(f, buf)

    links = boundary_links(solid)
    for i, where in zip(links.dirs, links.idx):
        ex, ey = int(E[i, 0]), int(E[i, 1])
        for flat in where:
            y, x = divmod(int(flat), NX)
            ys, xs = (y + ey) % NY, (x + ex) % NX
            assert f[OPP[i], y, x] == f_pre[OPP[i], ys, xs]


def test_forces_accept_a_prebuilt_link_list() -> None:
    solid = circle(NY, NX, cx=8.0, cy=5.0, radius=2.0)
    f = random_f(32)
    f_pre_collision = f.copy()
    buf = np.empty_like(f)
    bounce_back(f, f_pre_collision, solid)
    f_pre = f.copy()
    stream(f, buf)

    from_mask = forces(f_pre, f, solid, U=0.05, D=5.0)
    from_links = forces(f_pre, f, boundary_links(solid), U=0.05, D=5.0)

    assert from_mask == from_links


def test_forces_use_the_D019_characteristic_length() -> None:
    """``D`` is the cross-stream bbox extent (D-019); halving it doubles ``Cd``."""
    solid = circle(NY, NX, cx=8.0, cy=5.0, radius=2.0)
    y0, y1, _, _ = bounding_box(solid)
    D = y1 - y0 + 1
    f = random_f(33)
    f_pre_collision = f.copy()
    buf = np.empty_like(f)
    bounce_back(f, f_pre_collision, solid)
    f_pre = f.copy()
    stream(f, buf)

    cd_full, _ = forces(f_pre, f, solid, U=0.05, D=D)
    cd_half, _ = forces(f_pre, f, solid, U=0.05, D=D / 2.0)

    assert np.isclose(cd_half, 2.0 * cd_full, rtol=1e-9)


def test_forces_reject_degenerate_scales() -> None:
    solid = np.zeros((NY, NX), dtype=bool)
    f = random_f(34)
    with pytest.raises(ValueError, match="nonzero"):
        forces(f, f, solid, U=0.0, D=5.0)
    with pytest.raises(ValueError, match="nonzero"):
        forces(f, f, solid, U=0.05, D=0.0)


# --- Strouhal ----------------------------------------------------------------


def test_strouhal_recovers_a_synthetic_sine_within_one_percent() -> None:
    """Acceptance criterion: known frequency in, ``St`` out, within 1%."""
    f_true = 0.0043  # cycles per lattice time unit
    D, U = 20.0, 0.05
    dt = 5.0
    n = 4000
    t = np.arange(n) * dt
    cl = 0.4 * np.sin(2.0 * np.pi * f_true * t)

    st = strouhal(cl, dt, D, U)
    expected = f_true * D / U

    assert np.isclose(st, expected, rtol=0.01), f"got {st}, expected {expected}"


def test_strouhal_ignores_the_leading_transient() -> None:
    """A ramp occupying the first 30% must not move the answer."""
    f_true = 0.0043
    D, U, dt, n = 20.0, 0.05, 5.0, 4000
    t = np.arange(n) * dt
    clean = 0.4 * np.sin(2.0 * np.pi * f_true * t)

    dirty = clean.copy()
    cut = int(0.3 * n)
    dirty[:cut] *= np.linspace(0.0, 1.0, cut)  # growth transient
    dirty[:cut] += np.linspace(2.0, 0.0, cut)  # plus a big decaying drift

    assert np.isclose(strouhal(dirty, dt, D, U), strouhal(clean, dt, D, U), rtol=1e-9)


def test_strouhal_beats_the_raw_bin_spacing() -> None:
    """The parabolic refinement is what makes 1% reachable, so assert it works.

    The frequency is placed deliberately between two FFT bins; the nearest-bin
    answer is then wrong by more than 1% and the interpolated one is not.
    """
    D, U, dt = 20.0, 0.05, 1.0
    n = 1000
    tail_n = n - int(0.3 * n)
    f_true = (17.5) / (tail_n * dt)  # exactly half a bin off
    t = np.arange(n) * dt
    cl = np.sin(2.0 * np.pi * f_true * t)

    st = strouhal(cl, dt, D, U)
    nearest_bin_st = (17.0 / (tail_n * dt)) * D / U

    assert np.isclose(st, f_true * D / U, rtol=0.01)
    assert not np.isclose(nearest_bin_st, f_true * D / U, rtol=0.01)


def test_strouhal_scales_with_D_and_U() -> None:
    f_true = 0.004
    t = np.arange(3000) * 4.0
    cl = np.sin(2.0 * np.pi * f_true * t)

    base = strouhal(cl, 4.0, 20.0, 0.05)

    assert np.isclose(strouhal(cl, 4.0, 40.0, 0.05), 2.0 * base, rtol=1e-9)
    assert np.isclose(strouhal(cl, 4.0, 20.0, 0.10), base / 2.0, rtol=1e-9)


def test_strouhal_returns_zero_for_a_flat_series() -> None:
    assert strouhal(np.zeros(200), 1.0, 20.0, 0.05) == 0.0


def test_strouhal_rejects_a_series_too_short_to_measure() -> None:
    with pytest.raises(ValueError, match="too few"):
        strouhal(np.sin(np.arange(9)), 1.0, 20.0, 0.05)
    with pytest.raises(ValueError, match="transient"):
        strouhal(np.sin(np.arange(500)), 1.0, 20.0, 0.05, transient=1.0)


# --- residual ----------------------------------------------------------------


def test_residual_is_the_scaled_max_difference() -> None:
    u_prev = np.zeros((2, NY, NX), dtype=np.float32)
    u_now = u_prev.copy()
    u_now[1, 3, 4] = 0.002
    u_now[0, 1, 1] = -0.001

    assert np.isclose(residual(u_now, u_prev, 0.05), 0.002 / 0.05, rtol=1e-6)


def test_residual_of_an_unchanged_field_is_zero() -> None:
    rng = np.random.default_rng(41)
    u = rng.normal(0.0, 0.01, size=(2, NY, NX)).astype(np.float32)
    assert residual(u, u.copy(), 0.05) == 0.0


def test_residual_excludes_solid_cells() -> None:
    """D-014: nonsense velocities on solid cells must not reach the answer.

    Rung 2's residual read ``8.4e+01`` until the mask was applied. This is that
    regression, pinned.
    """
    solid = circle(NY, NX, cx=8.0, cy=5.0, radius=2.0)
    u_prev = np.zeros((2, NY, NX), dtype=np.float32)
    u_now = u_prev.copy()
    u_now[0][solid] = 84.0  # bounce-back leftovers
    u_now[1, 0, 0] = 0.001  # the only real change, on a fluid cell

    assert np.isclose(residual(u_now, u_prev, 0.05, solid=solid), 0.001 / 0.05, rtol=1e-6)
    assert residual(u_now, u_prev, 0.05) > 1e3  # unmasked, it is dominated by junk


def test_residual_is_allocation_free_with_a_buffer() -> None:
    rng = np.random.default_rng(42)
    u_now = rng.normal(0.0, 0.01, size=(2, NY, NX)).astype(np.float32)
    u_prev = rng.normal(0.0, 0.01, size=(2, NY, NX)).astype(np.float32)
    work = np.empty_like(u_now)

    residual(u_now, u_prev, 0.05, work=work)  # warm up

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        residual(u_now, u_prev, 0.05, work=work)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    assert after - before < work.nbytes


def test_residual_rejects_a_zero_reference_velocity() -> None:
    u = np.zeros((2, NY, NX), dtype=np.float32)
    with pytest.raises(ValueError, match="nonzero"):
        residual(u, u, 0.0)
