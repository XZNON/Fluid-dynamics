"""Tests for the Rung 3 setup — T007, :mod:`validate.cylinder`.

The rung itself is an integration test that takes minutes and prints PASS/FAIL;
what is worth pinning in ``pytest`` is the *setup*, because every way this rung
goes wrong quietly is a setup error:

* a domain that trips ``check_mask`` (``old-Docs/PLAN1.md`` § Risks: "cylinder shows
  no shedding" is almost always blockage or space),
* ``tau`` derived from anything other than ``Re`` (constraint 2),
* and — the one that actually happened while building this — a force integral
  that includes the channel walls, which reads ``Cd = 6.65`` where the body's
  own is 1.57.
"""

from __future__ import annotations

import numpy as np
import pytest

from lbm.geometry import bounding_box, check_mask, strip_solid_border
from lbm.probe import boundary_links, forces
from lbm.runner import Sim
from validate.cylinder import (
    CD_BAND,
    CD_REF,
    RE,
    ST_BAND,
    ST_REF,
    U,
    cylinder_mask,
    make_config,
    tau_for,
)


def test_the_default_domain_passes_every_mask_check() -> None:
    """No warning, not a suppressed warning (``CLAUDE.md`` constraint 12)."""
    import warnings

    solid, _cyl, _cx, _cy = cylinder_mask()
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        messages = check_mask(solid, "x", verbose=False)

    assert messages == []
    assert not log, f"unexpected warnings: {[str(w.message) for w in log]}"


def test_the_domain_meets_the_three_numeric_rules() -> None:
    solid, cylinder, _cx, _cy = cylinder_mask()
    ny, nx = solid.shape
    box = bounding_box(strip_solid_border(solid))
    assert box is not None
    y0, y1, x0, x1 = box
    d = y1 - y0 + 1

    assert d / (ny - 2) < 0.10  # blockage, fluid span denominator (D-019)
    assert (nx - 1 - x1) / d >= 8.0  # wake room before the outlet
    assert cylinder.sum() > 0 and not cylinder[0].any()  # walls are not the body


def test_the_cylinder_is_offset_half_a_cell_from_the_centreline() -> None:
    """The perturbation that lets shedding start (``old-Docs/TASKS1.md`` § T007)."""
    solid, cylinder, _cx, cy = cylinder_mask()
    ny = solid.shape[0]
    assert cy != (ny - 1) / 2.0
    rows = np.flatnonzero(cylinder.any(axis=1))
    centre = 0.5 * (rows[0] + rows[-1])
    assert abs(centre - (ny - 1) / 2.0) > 0.0


def test_tau_comes_from_re_and_nothing_else() -> None:
    nu, tau = tau_for(100.0, 0.06, 24)
    assert nu == pytest.approx(0.06 * 24 / 100.0)
    assert tau == pytest.approx(0.5 + 3.0 * nu)


def test_a_marginal_or_too_fast_case_is_refused_at_setup_not_at_nan_time() -> None:
    with pytest.raises(ValueError, match="0.53"):
        tau_for(100.0, 0.06, 6)  # tau = 0.5108
    with pytest.raises(ValueError, match="constraint 3"):
        tau_for(100.0, 0.12, 24)


def test_the_default_case_clears_both_floors() -> None:
    _nu, tau = tau_for(RE, U, 24)
    assert tau > 0.53
    assert U < 0.1


def test_the_floor_refuses_the_tau_that_was_measured_to_produce_nan() -> None:
    """Q-004, closed in T010.

    D-029 measured a square at ``tau = 0.5346`` blowing up by step 3200 and a
    disc at 0.5330 by step 1500, after a 26728-step run reported ``Cd = nan``.
    The inherited floor of 0.53 admits both. This one does not.
    """
    from validate.cylinder import TAU_FLOOR

    assert TAU_FLOOR > 0.5346
    with pytest.raises(ValueError, match="0.537"):
        tau_for(RE, 0.055, 21)  # tau = 0.5346, measured to blow up


def test_the_floor_still_admits_rung_3s_own_published_case() -> None:
    """The reason the floor is 0.537 and not Rung 4's 0.54.

    Rung 3 runs at ``tau = 0.5378`` — measured stable over its full 45500 steps
    — and a floor of 0.54 would make the benchmark refuse the run that produced
    its own reference numbers.
    """
    from validate.cylinder import TAU_FLOOR

    _nu, tau = tau_for(RE, U, 21)
    assert tau == pytest.approx(0.5378, abs=1e-4)
    assert tau > TAU_FLOOR
    assert TAU_FLOOR < 0.54


def test_the_acceptance_bands_are_the_contract_ones() -> None:
    """The windows come from ``old-Docs/TASKS1.md`` § T007 and are not adjustable."""
    assert ST_BAND == (0.155, 0.175)
    assert CD_BAND == (1.25, 1.45)
    assert ST_BAND[0] < ST_REF < ST_BAND[1]
    assert CD_BAND[0] < CD_REF < CD_BAND[1]


def test_the_force_integral_must_exclude_the_channel_walls() -> None:
    """The bug this rung would otherwise report as physics.

    ``Sim.links`` is built from the whole mask — cylinder *and* walls — so
    ``Sim.forces()`` measures the channel's friction alongside the body's drag.
    On a small stand-in for the Rung 3 domain the two answers differ by a
    factor of several, and the wall-inclusive one is the plausible-looking
    wrong number ``DOCS/IDEA2.md`` § Validation ladder warns about.

    Rung 3 runs with periodic sides (``WALL = 0``) so the two link lists now
    coincide; this builds a **walled** mask explicitly, because what is being
    pinned is that the body's own force is what gets reported whatever the rest
    of the mask contains.
    """
    solid, cylinder, _cx, _cy = cylinder_mask(6, upstream_d=3.0, downstream_d=8.5,
                                              span_d=10.5, wall=1)
    cfg = make_config(
        ny=solid.shape[0], nx=solid.shape[1], tau=0.6, u=0.05,
        outlet_lam=None, verbose_mask=False,
    )
    sim = Sim(cfg.replace(check_geometry=False), solid)
    for _ in range(60):
        sim.step()

    cd_all, _ = sim.forces()
    cd_body, _ = forces(
        sim.f_bb, sim.f, boundary_links(cylinder), U=0.05, D=sim.D
    )

    assert cd_all != pytest.approx(cd_body, rel=0.05)
    assert abs(cd_body) < abs(cd_all)
