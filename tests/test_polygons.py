"""Tests for the Rung 4 setup — T008, :mod:`validate.polygons`.

The rung itself is an integration test that takes minutes and prints PASS/FAIL.
What ``pytest`` pins is the *setup* and the one claim T008 makes about the
solver that a printed ``Cd`` does not cover: that the sharp corners do not leak,
i.e. **no fluid velocity inside the solid**.

The setup failures worth pinning are the ones session 7 already paid for once —
a domain that trips ``check_mask`` (``old-Docs/PLAN1.md`` § Risks), ``tau`` derived
from anything other than ``Re`` (constraint 2), and a force integral that
includes the channel walls (``Cd = 6.65`` where the body's own is 1.57).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from lbm.core import W, macroscopic
from lbm.geometry import bounding_box, check_mask, min_thickness, strip_solid_border
from lbm.probe import boundary_links, forces
from lbm.runner import Sim
from validate.cylinder import make_config, tau_for
from validate.polygons import (
    CD_BAND,
    CD_REF,
    POLY_VERTS,
    RE,
    SIDE_CELLS,
    TAU_FLOOR,
    U,
    Case,
    body_mask,
    cases,
    convex_body,
    interior_solid,
    seed_solid_at_rest,
    square_body,
    tau_for_rung4,
)


# --- the two masks ------------------------------------------------------------


@pytest.mark.parametrize("name", ["square", "polygon"])
def test_the_default_domain_passes_every_mask_check(name: str) -> None:
    """No warning, not a suppressed warning (``CLAUDE.md`` constraint 12).

    Sharp corners are exactly where the thickness rule matters, so this check
    has to *pass* rather than be silenced — ``old-Docs/TASKS1.md`` § T008.
    """
    solid, _body, _cx, _cy = body_mask(cases()[name])
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        messages = check_mask(solid, "x", verbose=False)

    assert messages == []
    assert not log, f"unexpected warnings: {[str(w.message) for w in log]}"


@pytest.mark.parametrize("name", ["square", "polygon"])
def test_the_domain_meets_the_three_numeric_rules(name: str) -> None:
    solid, body, _cx, _cy = body_mask(cases()[name])
    ny, nx = solid.shape
    box = bounding_box(strip_solid_border(solid))
    assert box is not None
    y0, y1, _x0, x1 = box
    d = y1 - y0 + 1

    assert d / ny < 0.10  # blockage on the fluid span (D-019); sides periodic
    assert (nx - 1 - x1) / d >= 8.0  # wake room before the outlet
    assert min_thickness(body) >= 3
    assert body.sum() > 0


def test_the_square_is_actually_a_square() -> None:
    """``regular_polygon(..., 4, rotate=pi/4)`` and nothing else (T004).

    Two things are worth asserting rather than assuming: the bounding box is
    square, and the box is *full* — a rotated diamond would also give a square
    box while filling only half of it.
    """
    _solid, body, _cx, _cy = body_mask(cases()["square"])
    box = bounding_box(body)
    assert box is not None
    y0, y1, x0, x1 = box
    assert (y1 - y0) == (x1 - x0)
    assert body[y0 : y1 + 1, x0 : x1 + 1].all()
    assert (y1 - y0 + 1) == body.sum() ** 0.5


def test_the_second_case_is_a_convex_polygon_and_is_not_the_square() -> None:
    """``old-Docs/TASKS1.md`` § T008 asks for "an arbitrary convex polygon"."""
    verts = np.asarray(POLY_VERTS, dtype=np.float64)
    assert verts.shape[0] >= 5
    edges = np.roll(verts, -1, axis=0) - verts
    nxt = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
    assert np.all(cross > 0.0), f"not convex: {cross}"

    # Not symmetric about either axis — nothing in the solver may rely on it.
    assert not np.allclose(verts[:, 1], -np.sort(-verts[:, 1]))
    _solid, body, _cx, _cy = body_mask(cases()["polygon"])
    _solid_s, square, _cx_s, _cy_s = body_mask(cases()["square"])
    assert not np.array_equal(body, square)
    assert body.sum() < square.sum()  # it is not a filled box


def test_the_two_body_factories_are_the_geometry_primitives() -> None:
    """No new geometry code in T008 — T004's primitives, called with arguments."""
    from lbm.geometry import polygon, regular_polygon

    import math

    ny, nx, cx, cy, half = 60, 60, 30.5, 29.5, 10.0
    assert np.array_equal(
        square_body(ny, nx, cx, cy, half),
        regular_polygon(ny, nx, 4, cx, cy, half * math.sqrt(2.0), rotate=math.pi / 4.0),
    )
    assert np.array_equal(
        convex_body(ny, nx, cx, cy, half),
        polygon(ny, nx, [(cx + half * vx, cy + half * vy) for vx, vy in POLY_VERTS]),
    )


def test_the_body_is_offset_half_a_cell_from_the_centreline() -> None:
    solid, _body, _cx, cy = body_mask(cases()["square"])
    assert cy != (solid.shape[0] - 1) / 2.0


# --- the case setup -----------------------------------------------------------


def test_tau_comes_from_re_and_nothing_else() -> None:
    """Constraint 2 — the physics is reused from Rung 3, not re-derived."""
    nu, tau = tau_for_rung4(RE, U, SIDE_CELLS)
    assert nu == pytest.approx(U * SIDE_CELLS / RE)
    assert tau == pytest.approx(0.5 + 3.0 * nu)
    assert (nu, tau) == tau_for(RE, U, SIDE_CELLS)  # same arithmetic, extra floor


def test_the_default_case_clears_both_ceilings_it_is_squeezed_between() -> None:
    """The two constraints that pull opposite ways here (see :data:`U`)."""
    _nu, tau = tau_for_rung4(RE, U, SIDE_CELLS)
    assert tau > TAU_FLOOR  # stability, measured
    assert U < 0.1  # constraint 3
    assert 1.70 * U < 0.1  # and so is the measured peak, with margin


def test_a_marginal_tau_is_refused_at_setup_not_at_nan_time() -> None:
    """The failure this rung actually hit: ``tau = 0.5346`` ran 26727 steps and
    reported ``Cd = nan``. D-016's 0.53 floor lets that through; this one does
    not, and it names the ``D`` that fixes it.

    Since T010 closed Q-004, ``tau = 0.5346`` is refused one level down as well
    — :data:`validate.cylinder.TAU_FLOOR` is 0.537 — so this asserts only that
    it is refused, not which of the two floors got there first.
    """
    assert TAU_FLOOR > 0.53
    with pytest.raises(ValueError, match="0.5346"):
        tau_for_rung4(RE, 0.055, 21)  # tau = 0.5346, measured to blow up
    with pytest.raises(ValueError, match="constraint 3"):
        tau_for_rung4(RE, 0.12, 27)


def test_rung_4_keeps_its_own_stricter_floor_above_rung_3s() -> None:
    """The band between the two floors is Rung 4's alone.

    A square accelerates the flow further round itself than a disc does, so it
    has less margin: ``tau = 0.5378`` is Rung 3's own measured-stable operating
    point and is refused here. If this ever stops raising, Rung 4 has silently
    inherited Rung 3's tolerance.
    """
    from validate.cylinder import TAU_FLOOR as CYL_FLOOR

    assert CYL_FLOOR < TAU_FLOOR
    _nu, tau = tau_for(RE, 0.06, 21)  # 0.5378: fine for Rung 3
    assert CYL_FLOOR < tau <= TAU_FLOOR
    with pytest.raises(ValueError, match="stability floor"):
        tau_for_rung4(RE, 0.06, 21)


def test_the_acceptance_band_is_the_contract_one() -> None:
    """``old-Docs/TASKS1.md`` § T008: Cd within 1.4-1.6, ref ~1.5. Not adjustable."""
    assert CD_BAND == (1.4, 1.6)
    assert CD_BAND[0] < CD_REF < CD_BAND[1]


def test_only_the_square_asserts_a_reference_value() -> None:
    """The polygon case reports; it does not judge (``old-Docs/TASKS1.md`` § T008)."""
    table = cases()
    assert table["square"].cd_band == CD_BAND
    assert table["polygon"].cd_band is None
    assert table["polygon"].cd_ref is None
    assert table["square"].require_shedding
    assert not table["polygon"].require_shedding


# --- the solver claim: sharp corners do not leak ------------------------------


def _small_case(name: str) -> Case:
    """The same bodies on a domain small enough for a unit test."""
    return cases(side_cells=8)[name]


def _run_small(name: str, *, seed: bool, steps: int = 300, wall: int = 0) -> Sim:
    case = _small_case(name)
    solid, _body, _cx, _cy = body_mask(
        case, upstream_d=3.0, downstream_d=8.5, span_d=10.5, wall=wall
    )
    cfg = make_config(
        ny=solid.shape[0], nx=solid.shape[1], tau=0.6, u=0.05,
        outlet_lam=None, verbose_mask=False,
    )
    sim = Sim(cfg.replace(check_geometry=False), solid)
    if seed:
        seed_solid_at_rest(sim)
    for _ in range(steps):
        sim.step()
    macroscopic(sim.f, sim.rho, sim.u)
    return sim


@pytest.mark.parametrize("name", ["square", "polygon"])
def test_no_fluid_velocity_inside_the_solid(name: str) -> None:
    """The T008 corner criterion: ``|u| < 1e-6`` on cells inside the body.

    Bounce-back reverses each population back along the direction it arrived
    from, so a population that entered the surface layer from the fluid leaves
    the way it came and never reaches the second layer. If a staircased corner
    leaked — if some direction at a convex corner had no partner to bounce back
    into — the interior is where it would show up.

    The surface layer itself is excluded on purpose: see
    :func:`validate.polygons.interior_solid`. It holds the reflection in flight
    and is *supposed* to be non-zero.
    """
    sim = _run_small(name, seed=True)
    inner = interior_solid(sim.solid)
    assert inner.sum() > 0
    speed = np.sqrt(sim.u[0][inner] ** 2 + sim.u[1][inner] ** 2)
    assert float(speed.max()) < 1e-6, f"leak inside the body: {speed.max():.3e}"


@pytest.mark.parametrize("name", ["square", "polygon"])
def test_the_body_interior_holds_exactly_the_rest_state(name: str) -> None:
    """Stronger than the velocity check, and it is the reason it holds.

    The interior is a closed subsystem: every population it exchanges with the
    surface layer is one that came from the interior in the first place. The
    rest state ``w_i rho0`` is a fixed point of both bounce-back (``W`` is
    symmetric under ``OPP``) and streaming (uniform), so an interior seeded at
    rest is still *bit-identically* at rest hundreds of steps later.
    """
    sim = _run_small(name, seed=True)
    inner = interior_solid(sim.solid)
    for i in range(9):
        assert np.array_equal(
            sim.f[i][inner], np.full(int(inner.sum()), W[i], dtype=np.float32)
        )


def test_without_the_rest_seed_the_interior_never_clears_itself() -> None:
    """Why :func:`validate.polygons.seed_solid_at_rest` exists — measured.

    ``Sim`` seeds the whole domain, solid included, with the equilibrium of the
    inlet profile. Bounce-back does not clear that: it *reverses* it, so the
    interior oscillates at ``+-U`` forever. The criterion above would then be
    measuring the initial condition rather than the boundary condition, which is
    the failure mode ``DOCS/IDEA2.md`` § Validation ladder exists to catch.
    """
    sim = _run_small("square", seed=False)
    inner = interior_solid(sim.solid)
    speed = np.sqrt(sim.u[0][inner] ** 2 + sim.u[1][inner] ** 2)
    assert float(speed.max()) > 1e-3  # order U = 0.05, not order zero


def test_the_rest_seed_touches_no_fluid_cell() -> None:
    """It is a change to the initial condition inside the wall, and nothing else."""
    case = _small_case("square")
    solid, _body, _cx, _cy = body_mask(
        case, upstream_d=3.0, downstream_d=8.5, span_d=10.5
    )
    cfg = make_config(
        ny=solid.shape[0], nx=solid.shape[1], tau=0.6, u=0.05,
        outlet_lam=None, verbose_mask=False,
    )
    sim = Sim(cfg.replace(check_geometry=False), solid)
    before = sim.f.copy()
    seed_solid_at_rest(sim)
    fluid = ~sim.solid
    assert np.array_equal(sim.f[:, fluid], before[:, fluid])
    assert not np.array_equal(sim.f[:, sim.solid], before[:, sim.solid])


def test_the_force_integral_must_exclude_the_channel_walls() -> None:
    """Session 7's trap, re-pinned for Rung 4's mask builder.

    ``Sim.links`` comes from the whole mask, so with walls in it ``Sim.forces()``
    reports the channel's friction alongside the body's drag. Rung 4 runs with
    periodic sides so the two coincide today; this builds a **walled** mask
    explicitly, because the correctness of ``Cd`` must not depend on that.
    """
    case = _small_case("square")
    solid, body, _cx, _cy = body_mask(
        case, upstream_d=3.0, downstream_d=8.5, span_d=10.5, wall=1
    )
    cfg = make_config(
        ny=solid.shape[0], nx=solid.shape[1], tau=0.6, u=0.05,
        outlet_lam=None, verbose_mask=False,
    )
    sim = Sim(cfg.replace(check_geometry=False), solid)
    seed_solid_at_rest(sim)
    for _ in range(60):
        sim.step()

    cd_all, _ = sim.forces()
    cd_body, _ = forces(sim.f_bb, sim.f, boundary_links(body), U=0.05, D=sim.D)

    assert cd_all != pytest.approx(cd_body, rel=0.05)
    assert abs(cd_body) < abs(cd_all)


def test_interior_solid_excludes_the_surface_layer_and_nothing_else() -> None:
    solid = np.zeros((11, 11), dtype=bool)
    solid[3:8, 3:8] = True  # 5x5 block
    inner = interior_solid(solid)
    assert inner.sum() == 9  # the 3x3 core
    assert inner[4:7, 4:7].all()
    assert not inner[3].any()
