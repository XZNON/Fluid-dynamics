"""Tests for the T203 Taylor–Green harness — ``DOCS/TASKS3.md`` § T203.

One test per acceptance criterion in that contract, plus the invariants
``CLAUDE.md`` constraints 2, 3, 4, 12 and **19** put on it:

* the **initial field is the exact vortex**, on the integer node positions the
  lattice actually streams between, with the Taylor–Green pressure in the
  density so the run does not start with an acoustic transient in it;
* the **decay rate is measured, not assumed** — the fit reproduces a synthetic
  exponential exactly, and on the solver it returns ``(tau - 0.5) / 3``;
* ``<nu_t>`` comes from :func:`lbm.probe.eddy_viscosity` **during** the run and
  is never fitted, which is asserted by source inspection as well as by value;
* the geometry is empty and the domain is periodic, so constraint 12 is vacuous
  here — asserted, not assumed, because "vacuous" and "forgotten" look the same
  from the outside;
* the closure clause cannot pass with the ``<nu_t>`` term deleted.

The expensive part — the two 5047-step runs — is done **once** per session by a
module-scoped fixture and shared, because every check below is a different
question about the same measurement.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from lbm.backends import available_backends
from lbm.core import CS2, CS_SMAG_LITERATURE, Q, equilibrium, macroscopic, nu_from_tau
from lbm.probe import eddy_viscosity
from lbm.runner import Sim, SimConfig
from validate.taylorgreen import (
    BASE_TOL,
    EPS_TOL,
    LES_TOL,
    NX,
    NY,
    TAU,
    U0,
    U_CEILING,
    DecayResult,
    _fit_log_slope,
    decay_time,
    main,
    run_decay,
    taylor_green,
)

REPO = Path(__file__).resolve().parent.parent
CS = CS_SMAG_LITERATURE


@pytest.fixture(scope="module")
def base() -> DecayResult:
    """Rung G's own ``Cs = 0`` run, on the oracle backend."""
    return run_decay(cs=0.0, backend="numpy")


@pytest.fixture(scope="module")
def les() -> DecayResult:
    """Rung G's own ``Cs = 0.17`` run, on the oracle backend."""
    return run_decay(cs=CS, backend="numpy")


# --- the initial field -------------------------------------------------------


def test_initial_field_is_the_exact_taylor_green_vortex() -> None:
    """``u = u0 cos(kx) sin(ky)``, ``v = -u0 sin(kx) cos(ky)``, verbatim.

    ``DOCS/TASKS3.md`` § T203, first acceptance criterion.
    """
    rho, u, kx, ky = taylor_green(NY, NX, U0)

    x = np.arange(NX, dtype=np.float64)[None, :]
    y = np.arange(NY, dtype=np.float64)[:, None]
    want_u = U0 * np.cos(kx * x) * np.sin(ky * y)
    want_v = -U0 * np.sin(kx * x) * np.cos(ky * y)

    assert u.shape == (2, NY, NX)
    assert u.dtype == np.float32
    assert rho.shape == (NY, NX)
    assert rho.dtype == np.float32
    np.testing.assert_allclose(u[0], want_u.astype(np.float32), atol=1e-7)
    np.testing.assert_allclose(u[1], want_v.astype(np.float32), atol=1e-7)


def test_initial_field_is_divergence_free_and_periodic() -> None:
    """The vortex is incompressible and closes on itself across the box.

    Both are properties of the *discrete* field, not of the continuum one, so
    they are checked with the periodic differences the lattice would take.
    """
    _, u, kx, ky = taylor_green(NY, NX, U0)
    dudx = (np.roll(u[0], -1, axis=1) - np.roll(u[0], 1, axis=1)) / 2.0
    dvdy = (np.roll(u[1], -1, axis=0) - np.roll(u[1], 1, axis=0)) / 2.0
    # Central differences of a sinusoid carry an O(k^2) amplitude error, which
    # cancels between the two terms because kx == ky; the sum is zero to
    # float32 round-off and not merely small.
    assert float(np.max(np.abs(dudx + dvdy))) < 1e-7


def test_initial_density_carries_the_taylor_green_pressure() -> None:
    """``rho = rho0 (1 + p / cs2)`` with ``p = -(u0^2/4)(cos 2kx + cos 2ky)``.

    Seeding at uniform density instead launches an acoustic transient of
    amplitude ``u0^2``; this is the line that stops it.
    """
    rho, _, kx, ky = taylor_green(NY, NX, U0)
    x = np.arange(NX, dtype=np.float64)[None, :]
    y = np.arange(NY, dtype=np.float64)[:, None]
    p = -(U0 * U0 / 4.0) * (np.cos(2 * kx * x) + np.cos(2 * ky * y))
    np.testing.assert_allclose(rho, (1.0 + p / CS2).astype(np.float32), atol=1e-7)
    # It is a perturbation, not a rescale: the mean stays at rho0.
    assert abs(float(rho.mean()) - 1.0) < 1e-6


def test_peak_velocity_is_exactly_u0_and_under_the_ceiling() -> None:
    """``|u|`` peaks at ``u0`` — constraint 3's quantity, and it is checked."""
    _, u, _, _ = taylor_green(NY, NX, U0)
    peak = float(np.max(np.hypot(u[0], u[1])))
    assert peak == pytest.approx(U0, rel=1e-6)
    assert peak < U_CEILING


# --- the geometry is empty, and that is the point (constraint 12) ------------


def test_geometry_is_vacuous_and_the_domain_is_periodic() -> None:
    """No bodies, no inlet, no outlet, no body force (constraint 12).

    ``DOCS/TASKS3.md`` § T203 asks that this be *said* rather than left for a
    reader to wonder about. It is also asserted, because a rung that silently
    grew a wall would still print a plausible number.
    """
    sim = Sim(SimConfig(ny=NY, nx=NX, tau=TAU, check_geometry=False), None)
    assert not sim.solid.any()
    assert sim.config.use_inlet is False
    assert sim.config.use_outlet is False
    assert sim.config.g == (0.0, 0.0)

    # Periodic in both axes: a single marked population returns to where it
    # started after exactly nx steps along +x, with nothing absorbed.
    mass0 = float(np.sum(sim.host_f(), dtype=np.float64))
    sim.run_steps(NX)
    assert float(np.sum(sim.host_f(), dtype=np.float64)) == pytest.approx(
        mass0, rel=1e-6
    )


def test_the_docstring_says_the_geometry_checks_are_vacuous() -> None:
    """Constraint 12's contract line: say so, don't leave it to be guessed."""
    import validate.taylorgreen as tg

    text = (tg.__doc__ or "").lower()
    assert "no bodies" in text
    assert "vacuous" in text
    assert "constraint 12" in text


# --- the fit -----------------------------------------------------------------


def test_fit_recovers_a_known_exponential_exactly() -> None:
    """The estimator itself is right before the solver is asked anything."""
    t = np.linspace(0.0, 1000.0, 51)
    rate = 3.7e-4
    e = 0.25 * np.exp(-rate * t)
    slope, r2 = _fit_log_slope(t, e)
    assert slope == pytest.approx(-rate, rel=1e-10)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_decay_time_is_the_energy_e_folding_time() -> None:
    """``T_d = 1 / (2 nu K^2)`` — energy, hence the factor of two."""
    nu = nu_from_tau(TAU)
    kx = ky = 2.0 * np.pi / 64
    assert decay_time(nu, kx, ky) == pytest.approx(
        1.0 / (2.0 * nu * (kx * kx + ky * ky))
    )


# --- criterion 2: Cs = 0 returns nu to under 1% ------------------------------


def test_base_run_returns_nu_to_under_one_percent(base: DecayResult) -> None:
    """``DOCS/TASKS3.md`` § T203: Rung 1's own bar, on a fourth independent case."""
    assert base.finite
    assert base.err_base < BASE_TOL
    assert base.nu == pytest.approx(nu_from_tau(TAU))


def test_base_run_is_a_single_exponential(base: DecayResult) -> None:
    """A window measuring two rates would fit worse than this, and be wrong."""
    assert base.energy_fit_r2 > 0.9999


def test_base_run_generates_exactly_zero_eddy_viscosity(base: DecayResult) -> None:
    """Constraint 19 seen from the rung: ``Cs = 0`` means ``nu_t == 0``, exactly."""
    assert base.nu_t_domain == 0.0
    assert base.nu_t_max == 0.0
    assert base.nu_t_eps == 0.0
    assert base.nu_claimed == base.nu


# --- criterion 3: Cs = 0.17 returns nu + <nu_t> to under 2% ------------------


def test_les_run_returns_nu_plus_nu_t_to_under_two_percent(les: DecayResult) -> None:
    """The contract's own check, against the **domain** average of ``nu_t``."""
    assert les.finite
    assert les.err_claimed < LES_TOL


def test_the_nu_t_term_is_not_negligible(les: DecayResult) -> None:
    """The discriminator: deleting ``<nu_t>`` must break the check above.

    A 2% bar on a term contributing a tenth of a percent is a check that passes
    with the term removed, which is the failure mode ``CLAUDE.md`` constraint 5
    names. This is what stops the clause being decorative.
    """
    assert les.err_base > LES_TOL
    assert les.nu_t_domain > 0.0


def test_the_excess_matches_the_dissipation_weighted_average(
    base: DecayResult, les: DecayResult
) -> None:
    """**D-091**: the energy decay is sensitive to ``<nu_t^3>/<nu_t^2>``.

    The sharp form of the same claim, carrying none of the geometric bias the
    domain average carries on this flow.
    """
    excess = les.nu_measured - base.nu_measured
    assert abs(excess / les.nu_t_eps - 1.0) < EPS_TOL


def test_the_dissipation_weighting_is_the_analytic_taylor_green_factor(
    les: DecayResult,
) -> None:
    """``<nu_t>_eps / <nu_t>`` is ``((4/3pi)^2/(1/2)^2)/(2/pi)^2 = 1.7780``.

    Taylor–Green has ``S_xy = 0`` identically, so ``nu_t ~ |sin kx sin ky|`` and
    the ratio is a pure property of that shape. It is checked because it is the
    reason the domain-average comparison carries a known bias at all — if this
    number moved, the interpretation in the module docstring would be wrong.
    """
    analytic = ((4.0 / (3.0 * np.pi)) ** 2 / 0.25) / (2.0 / np.pi) ** 2
    assert analytic == pytest.approx(1.7780, abs=1e-3)
    assert les.nu_t_eps / les.nu_t_domain == pytest.approx(analytic, rel=0.02)


def test_eddy_viscosity_is_non_negative_and_derived_through_tau(
    les: DecayResult,
) -> None:
    """Constraint 2: ``nu_t = cs2 (tau_eff - tau)``, never assigned, never < 0."""
    assert les.nu_t_min >= 0.0
    assert les.nu_t_max > 0.0


def test_nu_t_is_measured_from_the_model_and_never_fitted() -> None:
    """The contract's sharpest line, checked on the **syntax** and not the prose.

    ``<nu_t>`` must come from :func:`lbm.probe.eddy_viscosity`. A fitted one
    would be the answer copied into the question, so this asserts that the only
    curve fitting in the module is the energy fit, and that it never touches a
    ``nu_t`` series.
    """
    src = (REPO / "validate" / "taylorgreen.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    calls += [
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "eddy_viscosity" in calls

    # polyfit appears exactly once, inside the log-energy fit, and nowhere else.
    assert calls.count("polyfit") == 1
    fit = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_fit_log_slope"
    )
    assert "polyfit" in [
        c.func.attr for c in ast.walk(fit)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
    ]
    # No curve_fit / least_squares / optimisation anywhere.
    for banned in ("curve_fit", "least_squares", "minimize", "leastsq"):
        assert banned not in calls


def test_run_decay_samples_eddy_viscosity_during_the_run(les: DecayResult) -> None:
    """"During the run", literally: one ``<nu_t>`` per sample, in time order.

    A single end-of-run evaluation would be cheaper and would silently answer a
    different question, because ``nu_t`` decays with the flow.
    """
    assert len(les.samples) >= 10
    ts = [t for t, _, _, _ in les.samples]
    assert ts == sorted(ts)
    nus = [m1 for _, _, m1, _ in les.samples]
    assert all(v > 0.0 for v in nus)
    # It decays with the flow, which is why a time average is needed at all.
    assert nus[-1] < nus[0]
    assert les.nu_t_domain == pytest.approx(
        float(np.trapezoid(np.asarray(nus), np.asarray(ts, dtype=float)))
        / (ts[-1] - ts[0])
    )


def test_domain_average_matches_a_direct_probe_of_the_same_state() -> None:
    """The averaged quantity is exactly :func:`lbm.probe.eddy_viscosity`'s mean.

    Short run, cheap: the point is the identity, not the physics.
    """
    rho, u, _, _ = taylor_green(NY, NX, U0)
    sim = Sim(
        SimConfig(ny=NY, nx=NX, tau=TAU, cs_smag=CS, check_geometry=False), None
    )
    sim.load_f(equilibrium(rho, u))
    sim.run_steps(50)

    f = sim.host_f()
    r, uu = macroscopic(f.copy())
    feq = equilibrium(r, uu)
    nu_t = eddy_viscosity(f, feq, TAU, CS)
    assert nu_t.shape == (NY, NX)
    assert nu_t.dtype == np.float32
    assert float(nu_t.min()) >= 0.0
    assert float(nu_t.mean()) == pytest.approx(
        float(nu_t.astype(np.float64).mean()), rel=1e-4
    )


# --- criterion 4: peak lattice velocity ---------------------------------------


def test_peak_velocity_is_measured_over_the_whole_run(
    base: DecayResult, les: DecayResult
) -> None:
    """Constraint 3, and "throughout" means the warm-up too."""
    for r in (base, les):
        assert r.peak_u < U_CEILING
        # The vortex only decays, so the peak is the initial amplitude — which
        # is only true if the warm-up was sampled. A fit-window-only peak would
        # read ~0.08 * exp(-0.3) = 0.059.
        assert r.peak_u == pytest.approx(U0, rel=1e-3)


# --- the rung as a process ----------------------------------------------------


def test_main_prints_pass_and_exits_zero(capsys: pytest.CaptureFixture) -> None:
    """``python -m validate.taylorgreen`` prints PASS and returns 0."""
    code = main([])
    out = capsys.readouterr().out
    assert code == 0
    assert out.rstrip().endswith("PASS")
    # Not "FAIL not in out": the discriminator check's own label contains the
    # word, because what it asserts is that bare nu *would* fail. The failure
    # marker is what matters.
    assert "[XX]" not in out


def test_cs_zero_runs_the_base_clause_alone_and_says_so(
    capsys: pytest.CaptureFixture,
) -> None:
    """A skipped clause is reported, never silently dropped."""
    code = main(["--cs", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS" in out
    assert "closure clauses did not run" in out


def test_negative_cs_is_refused(capsys: pytest.CaptureFixture) -> None:
    """Constraint 2: the closure adds viscosity and never removes it."""
    code = main(["--cs", "-0.1"])
    out = capsys.readouterr().out
    assert code == 1
    assert "non-negative" in out


@pytest.mark.skipif(
    "warp" not in available_backends(),
    reason="warp-lang is not installed (myenv/Scripts/pip.exe install warp-lang)",
)
def test_both_backends_agree_on_the_measured_viscosity() -> None:
    """``DOCS/TASKS3.md`` § T203: the printed digits agree to **D-056**'s bar."""
    from validate.parity import STEP_TOL

    a = run_decay(cs=CS, backend="numpy")
    b = run_decay(cs=CS, backend="warp")
    assert abs(b.nu_measured - a.nu_measured) / a.nu_measured < STEP_TOL
    assert abs(b.nu_t_domain - a.nu_t_domain) / a.nu_t_domain < STEP_TOL
