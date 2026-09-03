"""Rung G — the Taylor–Green vortex: the closure adds the viscosity it claims.

``DOCS/IDEA4.md`` § Validation ladder, Rung G::

    validate/taylorgreen.py — The closure does not pollute a resolved laminar
    flow. 2D Taylor-Green: u = u0 cos(kx) sin(ky) exp(-2 nu k^2 t), exact. At
    Cs = 0 the measured decay rate returns nu to under 1% (Rung 1's own bar);
    at Cs = 0.17 it returns nu + <nu_t> to under 2%, with <nu_t> computed from
    the model rather than fitted.

This is the gate for **T203**, and with Rung F it is **M9**. Run it from the
repo root::

    myenv/Scripts/python.exe -m validate.taylorgreen
    myenv/Scripts/python.exe -m validate.taylorgreen --backend warp
    myenv/Scripts/python.exe -m validate.taylorgreen --cs 0    # the base clause alone

Why Taylor–Green and not decaying turbulence
--------------------------------------------
``DOCS/TASKS3.md`` § T203 Notes is explicit: Taylor–Green is chosen **because it
has an exact solution**, and this project's ladder is built on known answers
rather than on statistical scalings. An enstrophy-cascade check is a better test
of a turbulence *model* and a worse test of *this* claim, which is that the
closure adds a known, small, computable amount of viscosity to a flow that is
fully resolved. No such check is written here.

Geometry (constraint 12) is vacuous here, and that is deliberate
----------------------------------------------------------------
There are **no bodies**. ``solid`` is all-``False``, streaming is periodic on
both axes (:func:`lbm.core._shift_blocks`), and there is no inlet, no outlet and
no body force — a configuration no other rung runs. So constraint 12's three
checks (≥3 cells thick, ≥8 D from the outlet, <10% blockage) are not skipped by
oversight and are not silently disabled: they have nothing to measure, and
:data:`SimConfig.check_geometry` is set ``False`` for that reason and no other.
The one geometric fact this rung *does* assert is periodicity, which it gets for
free from the empty mask and checks by measuring an exact periodic solution.

The measurement, in full
------------------------
The exact 2D Taylor–Green vortex on a doubly periodic box, one wavelength in
each direction, ``kx = ky = 2 pi / L``::

    u(x, y, t) =  u0 cos(kx x) sin(ky y) exp(-nu K^2 t)
    v(x, y, t) = -u0 sin(kx x) cos(ky y) exp(-nu K^2 t)      K^2 = kx^2 + ky^2

so the kinetic energy ``E = <(u^2 + v^2)/2>`` decays as ``exp(-2 nu K^2 t)`` and

    nu_measured = -slope(ln E vs t) / (2 K^2)

is a *rate* measurement, not a profile measurement — which is why it reads the
base viscosity roughly an order of magnitude tighter than Rung 1's 1% bar.
``lbm/units.py`` is not involved: everything here is lattice units, because
nothing physical enters (``CLAUDE.md`` § conventions).

Both the initial density and the fit window are chosen, not defaulted
--------------------------------------------------------------------
The state is seeded from the **equilibrium of the exact field**, including the
Taylor–Green pressure ``p = -(u0^2 / 4)(cos 2kx x + cos 2ky y)`` as
``rho = rho0 (1 + p / cs2)``. Seeding at uniform density instead launches an
acoustic transient of amplitude ``u0^2`` that the fit then has to outlive.

The fit window is expressed in units of the analytic e-folding time
``T_d = 1 / (2 nu K^2)``: :data:`WARMUP_TD` of warm-up discarded, then
:data:`WINDOW_TD` fitted. That matters for the closure clause specifically.
``nu_t`` is proportional to the strain rate and therefore decays with the flow,
so a window long enough to lose most of the amplitude would be comparing a
least-squares slope against a time average of a quantity that changed eightfold
across it. One e-folding of energy is a 39% loss of amplitude, and the
comparison is then well conditioned.

What "the closure adds what it claims" actually means here (**D-091**)
----------------------------------------------------------------------
Taylor–Green has ``S_xy = 0`` identically, so ``|S| = 2 u0 k |sin kx x sin ky y|``
and ``nu_t = Cs^2 |S|`` is **not uniform** — it is largest where the strain is.
The energy decay is set by the dissipation, which weights ``nu_t`` by
``S_ab S_ab``, and on this flow that weighted mean is

    <nu_t>_eps / <nu_t>_domain  =  (<|s|^3> / <s^2>) / <|s|>
                                =  ((4 / 3 pi)^2 / (1/2)^2) / (2 / pi)^2
                                =  1.7780

times the plain domain average. The contract (``DOCS/TASKS3.md`` § T203) asks
for the **domain average** against a 2% bar, and that is check 2 below,
unchanged. Check 4 is the sharp form of the same claim and carries no such
geometric bias: ``<nu_t>_eps = <nu_t^3> / <nu_t^2>``, a re-weighting of the very
same :func:`lbm.probe.eddy_viscosity` field with no analytic input and no
fitting, predicts the measured excess to well under a percent. Both are computed
**from the model during the run**. Neither is fitted to the decay curve, and a
fitted ``nu_t`` would prove nothing — it would be the answer copied into the
question.

Why check 3 exists
------------------
A 2% bar on a term that contributes 0.1% of ``nu`` is a check that passes with
the term deleted, which is exactly the failure mode constraint 5 names. So the
case is sized (see :data:`NX` and its neighbours) so that ``<nu_t> / nu`` is
large enough for check 2 to have teeth, and check 3 asserts it: replacing
``nu + <nu_t>`` with bare ``nu`` must **fail** check 2's own bar. A resolved
Taylor–Green makes ``nu_t`` small on purpose — that is Q-202's answer — so this
is the one thing about the case that had to be designed rather than defaulted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from lbm.backends import BackendUnavailableError, get_backend
from lbm.core import CS2, CS_SMAG_LITERATURE, Q, equilibrium, macroscopic, nu_from_tau
from lbm.probe import eddy_viscosity
from lbm.runner import Sim, SimConfig

# --- the case ----------------------------------------------------------------
# Sized so that all three of the following are true at once, which is a narrower
# window than it looks: the base clause needs a resolved flow, the closure clause
# needs nu_t to be a measurable fraction of nu, and constraint 3 caps u0. See the
# module docstring, "Why check 3 exists".

#: Rows and columns. One full wavelength across the box in each direction, so
#: 64 cells per wavelength — amply resolved for a sinusoid, which is what makes
#: "the closure does not pollute a **resolved** flow" a fair test.
NY: int = 64
NX: int = 64

#: Relaxation time. ``nu = (tau - 0.5) / 3 = 1 / 150`` (constraint 2 — viscosity
#: is never set directly). Low enough that the closure is a measurable fraction
#: of it, high enough to be nowhere near the ``tau -> 0.5`` stability edge.
TAU: float = 0.52

#: Peak lattice velocity of the initial field, and — because the vortex only
#: decays — the peak over the whole run. 20% below the 0.1 ceiling
#: (``CLAUDE.md`` constraint 3), and the same order as Rung 1's halved case,
#: which peaks near 0.08. The run **measures** this rather than assuming it.
U0: float = 0.08

#: Initial and reference density.
RHO0: float = 1.0

#: The Smagorinsky constant the closure clauses run at — the literature value,
#: imported from :mod:`lbm.core` and not restated (constraint 4's "no physics
#: constant twice"). Phase 2 does not tune it (``DOCS/TASKS3.md`` § T201 Notes).
CS: float = CS_SMAG_LITERATURE

#: Warm-up discarded before the fit, in units of the analytic e-folding time
#: ``T_d = 1 / (2 nu K^2)``. It buys two things: the acoustic transient of the
#: initial condition (a sound wave crosses the box in ``L / sqrt(cs2)`` ~ 111
#: steps, and ``0.3 T_d`` here is ~1500) and the relaxation of ``f`` onto its own
#: non-equilibrium part, which the equilibrium seed does not supply.
WARMUP_TD: float = 0.3

#: Length of the fit window, in the same units. One e-folding of **energy**,
#: i.e. the velocity amplitude falls to ``exp(-0.5) = 0.61`` across it.
WINDOW_TD: float = 1.0

#: Samples across the fit window. Each one is a host read of ``f`` (constraint 8:
#: probe cadence, never step cadence).
SAMPLES: int = 40

#: Samples taken during the warm-up. These contribute nothing to the fit; they
#: exist so that "peak |u| < 0.1 **throughout**" is measured over the whole run
#: rather than over the fitted part of it.
WARMUP_SAMPLES: int = 10

#: Rung 1's own bar, and the bar ``DOCS/TASKS3.md`` § T203 sets for the base
#: clause: the measured viscosity returns ``(tau - 0.5) / 3`` to under 1%.
BASE_TOL: float = 0.01

#: The closure clause's bar: the measured viscosity returns ``nu + <nu_t>`` to
#: under 2%, with ``<nu_t>`` the **domain** average of
#: :func:`lbm.probe.eddy_viscosity`. Also the bar check 3 asserts bare ``nu``
#: fails, which is what stops check 2 passing vacuously.
LES_TOL: float = 0.02

#: The dissipation-weighted check's bar. ``<nu_t>_eps = <nu_t^3> / <nu_t^2>`` is
#: the average the energy decay is actually sensitive to (module docstring,
#: **D-091**), so this comparison carries no geometric bias and the bar can be
#: tight. Measured at 0.3%; 5% is the allowance for the fit, not for the model.
EPS_TOL: float = 0.05

#: ``CLAUDE.md`` constraint 3. Imported nowhere because
#: :data:`lbm.runner.U_MAX` is the runner's own setup-time warning and this is
#: the rung's own measured assertion; they are the same number by definition and
#: a test would be checking Python, not physics.
U_CEILING: float = 0.1


def taylor_green(
    ny: int, nx: int, u0: float, rho0: float = RHO0
) -> tuple[NDArray[np.float32], NDArray[np.float32], float, float]:
    """The exact 2D Taylor–Green vortex, one wavelength across the box.

    ``DOCS/IDEA4.md`` § Validation ladder, Rung G. The field at ``t = 0``::

        u =  u0 cos(kx x) sin(ky y)
        v = -u0 sin(kx x) cos(ky y)
        p = -(u0^2 / 4) (cos 2kx x + cos 2ky y)

    with ``kx = 2 pi / nx`` and ``ky = 2 pi / ny``, evaluated at the integer node
    positions ``x = 0 .. nx-1``, ``y = 0 .. ny-1``. Integer nodes rather than
    cell centres because the lattice streams by whole cells: ``cos(kx x)`` is
    then exactly periodic under the shift the streaming step performs, and no
    part of the measured decay is a phase error.

    The pressure is included because a uniform-density seed launches an acoustic
    transient of amplitude ``u0^2`` that the fit would otherwise have to outlive;
    it enters as ``rho = rho0 (1 + p / cs2)``, the isothermal equation of state
    this model already implies.

    Args:
        ny: rows.
        nx: columns.
        u0: velocity amplitude, lattice units. This **is** the peak ``|u|``:
            ``|u|^2 / u0^2 = a + b - 2ab`` with ``a = cos^2 kx x`` and
            ``b = cos^2 ky y``, whose maximum over the unit square is 1.
        rho0: reference density.

    Returns:
        ``(rho, u, kx, ky)`` — ``rho`` shape ``(ny, nx)`` ``float32``, ``u``
        shape ``(2, ny, nx)`` ``float32`` with component 0 ``ux`` and component
        1 ``uy`` (constraint 4's index order), and the two wavenumbers as
        ``float``.
    """
    kx = 2.0 * np.pi / nx
    ky = 2.0 * np.pi / ny

    x = np.arange(nx, dtype=np.float64)[None, :]
    y = np.arange(ny, dtype=np.float64)[:, None]

    u = np.empty((2, ny, nx), dtype=np.float32)
    u[0] = (u0 * np.cos(kx * x) * np.sin(ky * y)).astype(np.float32)
    u[1] = (-u0 * np.sin(kx * x) * np.cos(ky * y)).astype(np.float32)

    p = -(u0 * u0 / 4.0) * (np.cos(2.0 * kx * x) + np.cos(2.0 * ky * y))
    rho = (rho0 * (1.0 + p / CS2)).astype(np.float32)

    return rho, np.broadcast_to(u, (2, ny, nx)).copy(), kx, ky


def decay_time(nu: float, kx: float, ky: float) -> float:
    """The analytic e-folding time of the **kinetic energy**, in timesteps.

    ``E ~ exp(-2 nu K^2 t)`` with ``K^2 = kx^2 + ky^2``, so ``T_d = 1 / (2 nu K^2)``.
    The velocity amplitude e-folds at twice this, which is the exact solution
    ``DOCS/IDEA4.md`` quotes as ``exp(-2 nu k^2 t)`` for ``kx = ky = k``.
    """
    return 1.0 / (2.0 * nu * (kx * kx + ky * ky))


@dataclass
class DecayResult:
    """What one Taylor–Green run measured.

    Attributes:
        backend: the T101 backend the run used.
        cs: the Smagorinsky constant it ran at. ``0.0`` is Phase 1's collision,
            bitwise (constraint 19, **D-086** / **D-088**).
        tau: the base relaxation time.
        nu: ``(tau - 0.5) / 3``, lattice units (constraint 2).
        kx: streamwise wavenumber.
        ky: cross-stream wavenumber.
        warmup: steps discarded before the fit.
        window: steps spanned by the fit.
        nu_measured: viscosity from the fitted energy decay rate,
            ``-slope / (2 K^2)``.
        nu_t_domain: time-averaged **domain** mean of
            :func:`lbm.probe.eddy_viscosity` over the fit window. Zero when
            ``cs`` is zero, exactly.
        nu_t_eps: time-averaged dissipation-weighted mean,
            ``<nu_t^3> / <nu_t^2>`` — the average the energy decay is sensitive
            to (**D-091**).
        nu_t_max: largest ``nu_t`` seen anywhere, any sample.
        nu_t_min: smallest ``nu_t`` seen anywhere, any sample. Never negative.
        peak_u: largest ``|u|`` over **every** sample, warm-up included
            (constraint 3).
        amplitude_ratio: velocity amplitude at the end of the window over its
            value at the start — how much of the vortex the fit actually spent.
        energy_fit_r2: coefficient of determination of the ``ln E`` fit. A pure
            exponential gives 1; a number below ~0.999 means the window is not
            measuring a single decay rate and the fit is not trustworthy.
        finite: ``f`` stayed finite at every sample.
        seconds: wall clock for the run.
        samples: ``(t, E, <nu_t>, <nu_t>_eps)`` per fit sample, for the plot a
            reader can make and for anyone bisecting a failure.
    """

    backend: str
    cs: float
    tau: float
    nu: float
    kx: float
    ky: float
    warmup: int
    window: int
    nu_measured: float
    nu_t_domain: float
    nu_t_eps: float
    nu_t_max: float
    nu_t_min: float
    peak_u: float
    amplitude_ratio: float
    energy_fit_r2: float
    finite: bool
    seconds: float
    samples: list[tuple[int, float, float, float]] = field(default_factory=list)

    @property
    def nu_claimed(self) -> float:
        """``nu + <nu_t>`` — what the model says the run's viscosity was."""
        return self.nu + self.nu_t_domain

    @property
    def err_claimed(self) -> float:
        """Relative error of the measurement against ``nu + <nu_t>``."""
        return abs(self.nu_measured - self.nu_claimed) / self.nu_claimed

    @property
    def err_base(self) -> float:
        """Relative error of the measurement against the **base** ``nu`` alone.

        With the closure off this is the base clause. With it on this is check
        3: the number that must **exceed** :data:`LES_TOL`, because if it did
        not, check 2 would pass with the ``<nu_t>`` term deleted.
        """
        return abs(self.nu_measured - self.nu) / self.nu


def _fit_log_slope(t: NDArray[np.float64], e: NDArray[np.float64]) -> tuple[float, float]:
    """Least-squares slope of ``ln e`` against ``t``, and the fit's ``R^2``.

    Returns:
        ``(slope, r2)``. ``r2`` is reported rather than asserted away: a decay
        that is not a single exponential shows up here first.
    """
    y = np.log(e)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (slope * t + intercept)
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return float(slope), r2


def run_decay(
    ny: int = NY,
    nx: int = NX,
    tau: float = TAU,
    u0: float = U0,
    cs: float = 0.0,
    backend: str = "numpy",
    warmup_td: float = WARMUP_TD,
    window_td: float = WINDOW_TD,
    samples: int = SAMPLES,
    warmup_samples: int = WARMUP_SAMPLES,
    f0: NDArray[np.float32] | None = None,
) -> DecayResult:
    """Decay one Taylor–Green vortex and measure the viscosity it decayed at.

    ``DOCS/IDEA4.md`` § Validation ladder, Rung G. The run is a plain
    :class:`lbm.runner.Sim` with no inlet, no outlet, no body force and an empty
    mask, so :meth:`lbm.runner.Sim.step` is collide → stream on a doubly
    periodic domain and nothing else. Every buffer belongs to ``Sim`` and is
    allocated once by it; this function allocates nothing inside the stepping
    loop except the four ``float`` diagnostics it appends per **sample**, which
    is probe cadence and not step cadence (constraint 8).

    ``<nu_t>`` is computed **from the model, during the run** — at each sample,
    from that sample's own ``f`` and its equilibrium, through
    :func:`lbm.probe.eddy_viscosity`, which derives it via ``tau_eff``
    (constraint 2). It is never fitted to the decay curve; the whole point of
    the rung is that the two are independent measurements of the same thing.

    Args:
        ny: rows.
        nx: columns.
        tau: base relaxation time, greater than 0.5.
        u0: initial velocity amplitude, lattice units, under 0.1 (constraint 3).
        cs: Smagorinsky constant. ``0.0`` runs Phase 1's collision bitwise
            (constraint 19).
        backend: the T101 registry name — ``"numpy"`` (the oracle, **D-043**) or
            ``"warp"``.
        warmup_td: warm-up discarded, in units of :func:`decay_time`.
        window_td: fit window length, same units.
        samples: samples across the fit window.
        warmup_samples: samples across the warm-up, for the peak-velocity check
            only.
        f0: an initial ``(9, ny, nx)`` ``float32`` distribution to adopt instead
            of seeding from the exact field. Used by the cross-backend clause so
            both backends start bit-identical, exactly as
            :func:`validate.parity.whole_step` does.

    Returns:
        A :class:`DecayResult`.
    """
    rho_h, u_h, kx, ky = taylor_green(ny, nx, u0)
    nu = nu_from_tau(tau)
    k2 = kx * kx + ky * ky
    t_d = decay_time(nu, kx, ky)

    warmup = int(round(warmup_td * t_d))
    every = max(1, int(round(window_td * t_d)) // samples)
    window = every * samples

    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=tau,
        rho0=RHO0,
        backend=backend,
        cs_smag=cs,
        # No bodies: the constraint 12 checks have nothing to measure here. See
        # the module docstring — this is the reason, not an exemption.
        check_geometry=False,
    )
    sim = Sim(cfg, None)
    sim.load_f(equilibrium(rho_h, u_h) if f0 is None else f0)

    start = time.perf_counter()
    peak_u = 0.0
    nu_t_max = 0.0
    nu_t_min = np.inf
    finite = True

    def sample() -> tuple[float, float, float, float]:
        """One host read: energy, the two ``nu_t`` averages, and peak ``|u|``.

        Returns ``(E, <nu_t>, <nu_t>_eps, peak|u|)``. ``float64`` for the
        reductions only — the state is and stays ``float32`` (constraint 4);
        summing 4096 ``float32`` energies in ``float32`` would lose bits the fit
        then has to see through.
        """
        f = sim.host_f()
        rho, u = macroscopic(f.copy())
        feq = equilibrium(rho, u)

        ux = u[0].astype(np.float64)
        uy = u[1].astype(np.float64)
        energy = 0.5 * float(np.mean(ux * ux + uy * uy))
        peak = float(np.max(np.sqrt(ux * ux + uy * uy)))

        nu_t = eddy_viscosity(f, feq, tau, cs).astype(np.float64)
        m1 = float(nu_t.mean())
        m2 = float((nu_t * nu_t).mean())
        m3 = float((nu_t * nu_t * nu_t).mean())
        # <nu_t>_eps = <nu_t^3> / <nu_t^2>: the dissipation-weighted mean, since
        # the weight S_ab S_ab is |S|^2 / 2 and |S| = nu_t / Cs^2, so the weight
        # is proportional to nu_t^2 with every constant cancelling in the ratio.
        eps = m3 / m2 if m2 > 0.0 else 0.0
        return energy, m1, eps, peak

    # --- warm-up: no fit contribution, but constraint 3 is measured here too --
    warm_every = max(1, warmup // max(1, warmup_samples))
    done = 0
    while done < warmup:
        _, _, _, pk = sample()
        peak_u = max(peak_u, pk)
        n = min(warm_every, warmup - done)
        sim.run_steps(n)
        done += n

    # --- the fit window ------------------------------------------------------
    ts: list[int] = []
    es: list[float] = []
    m1s: list[float] = []
    epss: list[float] = []

    for s in range(samples + 1):
        if s:
            sim.run_steps(every)
        energy, m1, eps, pk = sample()
        if not np.isfinite(sim.host_f()).all() or not np.isfinite(energy):
            finite = False
            break
        ts.append(warmup + s * every)
        es.append(energy)
        m1s.append(m1)
        epss.append(eps)
        peak_u = max(peak_u, pk)
        nu_t_max = max(nu_t_max, m1)
        nu_t_min = min(nu_t_min, m1)

    seconds = time.perf_counter() - start

    if not finite or len(ts) < 3:
        return DecayResult(
            backend=backend, cs=cs, tau=tau, nu=nu, kx=kx, ky=ky,
            warmup=warmup, window=window, nu_measured=float("nan"),
            nu_t_domain=0.0, nu_t_eps=0.0, nu_t_max=0.0, nu_t_min=0.0,
            peak_u=peak_u, amplitude_ratio=float("nan"), energy_fit_r2=0.0,
            finite=False, seconds=seconds,
        )

    t = np.asarray(ts, dtype=np.float64)
    e = np.asarray(es, dtype=np.float64)
    slope, r2 = _fit_log_slope(t, e)
    nu_measured = -slope / (2.0 * k2)

    span = float(t[-1] - t[0])
    nu_t_domain = float(np.trapezoid(np.asarray(m1s), t) / span)
    nu_t_eps = float(np.trapezoid(np.asarray(epss), t) / span)

    return DecayResult(
        backend=backend,
        cs=cs,
        tau=tau,
        nu=nu,
        kx=kx,
        ky=ky,
        warmup=warmup,
        window=window,
        nu_measured=nu_measured,
        nu_t_domain=nu_t_domain,
        nu_t_eps=nu_t_eps,
        nu_t_max=nu_t_max,
        nu_t_min=0.0 if nu_t_min is np.inf else float(nu_t_min),
        peak_u=peak_u,
        amplitude_ratio=float(np.sqrt(e[-1] / e[0])),
        energy_fit_r2=r2,
        finite=True,
        seconds=seconds,
        samples=list(zip(ts, es, m1s, epss)),
    )


# --- the cross-backend clause ------------------------------------------------


@dataclass
class CrossResult:
    """What the cross-backend clause measured. Only runs off ``numpy``.

    Attributes:
        backend: the backend compared against numpy.
        cs: the Smagorinsky constant both sides ran.
        step_du: ``max|Delta u| / u0`` at the end of the run. Held to
            **D-056**'s 1e-4, unwidened.
        step_df: worst ``|Delta f|`` at the same point, ``f`` units.
        nu_ref: viscosity numpy measured.
        nu_dut: viscosity the other backend measured.
        nu_rel: ``|nu_dut - nu_ref| / nu_ref``.
        steps: timesteps each side advanced.
        finite: both sides stayed finite.
        seconds: wall clock.
    """

    backend: str
    cs: float
    step_du: float
    step_df: float
    nu_ref: float
    nu_dut: float
    nu_rel: float
    steps: int
    finite: bool
    seconds: float


def check_cross_backend(
    backend: str, cs: float, ny: int = NY, nx: int = NX, tau: float = TAU,
    u0: float = U0,
) -> CrossResult:
    """Both backends run Rung G's own case and agree to **D-056**'s bar.

    ``DOCS/TASKS3.md`` § T203: *"Both backends pass, and the printed digits agree
    to the D-056 whole-step tolerance."* Same shape as
    :func:`validate.les.check_cross_backend` and for the same reason
    (**D-090**): a backend compared with itself proves nothing, and a second copy
    of the case would be a case whose agreement with the real rung nobody checks
    — so this runs *this module's* :func:`run_decay` on each backend rather than
    a copy of it.

    "Identical state" is made literal exactly as :func:`validate.parity.whole_step`
    does it: the NumPy run's initial ``f`` is handed to the other backend, so
    step 0 is bit-identical and the initial condition is not confused with the
    thing being measured.

    Args:
        backend: the backend to check against numpy.
        cs: Smagorinsky constant — the closure is **on** for this clause when
            the rung is run with it on, because the closure-off case is what
            Rung A already measures (**D-090**).
        ny: rows.
        nx: columns.
        tau: relaxation time.
        u0: velocity amplitude.

    Returns:
        A :class:`CrossResult`.
    """
    start = time.perf_counter()
    rho_h, u_h, _, _ = taylor_green(ny, nx, u0)
    f0 = equilibrium(rho_h, u_h)

    ref = run_decay(ny, nx, tau, u0, cs, "numpy", f0=f0)
    dut = run_decay(ny, nx, tau, u0, cs, backend, f0=f0)

    # Re-run both to the same step count and compare the state itself, rather
    # than only the fitted number: a fit can agree while the fields do not.
    steps = ref.warmup + ref.window
    cfg = SimConfig(ny=ny, nx=nx, tau=tau, rho0=RHO0, cs_smag=cs,
                    check_geometry=False)
    sim_ref = Sim(cfg.replace(backend="numpy"), None)
    sim_ref.load_f(f0)
    sim_dut = Sim(cfg.replace(backend=backend), None)
    sim_dut.load_f(f0)
    sim_ref.run_steps(steps)
    sim_dut.run_steps(steps)

    u_ref = sim_ref.host_u()
    u_dut = sim_dut.host_u()
    f_ref = sim_ref.host_f()
    f_dut = sim_dut.host_f()
    finite = bool(np.isfinite(f_ref).all() and np.isfinite(f_dut).all())

    du = float(np.max(np.abs(u_ref - u_dut))) / u0 if finite else float("inf")
    df = float(np.max(np.abs(f_ref - f_dut))) if finite else float("inf")
    nu_rel = abs(dut.nu_measured - ref.nu_measured) / ref.nu_measured

    return CrossResult(
        backend=backend,
        cs=cs,
        step_du=du,
        step_df=df,
        nu_ref=ref.nu_measured,
        nu_dut=dut.nu_measured,
        nu_rel=nu_rel,
        steps=steps,
        finite=finite,
        seconds=time.perf_counter() - start,
    )


# --- reporting ---------------------------------------------------------------


def _print_run(label: str, r: DecayResult) -> None:
    """One run's measured quantities, in the order a reader needs them."""
    print(f"    {label}")
    print(f"      window         warm-up {r.warmup} steps, then {r.window} "
          f"fitted in {len(r.samples)} samples ({r.seconds:.1f} s)")
    print(f"      amplitude      {r.amplitude_ratio:.4f} of its value at the "
          f"start of the window   ln E fit R^2 = {r.energy_fit_r2:.6f}")
    print(f"      nu             base {r.nu:.8f}   measured {r.nu_measured:.8f}")
    if r.cs:
        print(f"      <nu_t>         domain {r.nu_t_domain:.6e}   "
              f"dissipation-weighted {r.nu_t_eps:.6e}   "
              f"ratio {r.nu_t_eps / r.nu_t_domain:.4f}")
        print(f"      nu + <nu_t>    {r.nu_claimed:.8f}   "
              f"<nu_t>/nu = {r.nu_t_domain / r.nu:.4%}")
    print(f"      peak |u|       {r.peak_u:.5f}  (ceiling {U_CEILING}, "
          f"constraint 3)")


def report(
    base: DecayResult,
    les: DecayResult | None,
    cross: CrossResult | None,
    backend: str,
) -> bool:
    """Print every check and return whether Rung G passed."""
    print()
    print("  measured")
    _print_run(f"Cs = 0      (Phase 1's collision, bitwise -- D-086 / D-088)", base)
    if les is not None:
        print()
        _print_run(f"Cs = {les.cs}   (the closure engaged)", les)

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            f"Cs = 0: measured nu returns (tau-0.5)/3 to < {BASE_TOL:.0%}",
            base.finite and base.err_base < BASE_TOL,
            f"{base.err_base:.4%}   ({base.nu_measured:.8f} vs {base.nu:.8f})",
        )
    )
    checks.append(
        (
            "Cs = 0: <nu_t> is exactly zero (constraint 19)",
            base.nu_t_domain == 0.0 and base.nu_t_max == 0.0,
            "every sample, every cell",
        )
    )
    checks.append(
        (
            f"Cs = 0: peak |u| < {U_CEILING} throughout (constraint 3)",
            base.peak_u < U_CEILING,
            f"peak |u| = {base.peak_u:.5f} over {base.warmup + base.window} steps",
        )
    )

    if les is not None:
        checks.append(
            (
                f"Cs = {les.cs}: measured nu returns nu + <nu_t> to "
                f"< {LES_TOL:.0%}",
                les.finite and les.err_claimed < LES_TOL,
                f"{les.err_claimed:.4%}   ({les.nu_measured:.8f} vs "
                f"{les.nu_claimed:.8f})",
            )
        )
        checks.append(
            (
                f"  ... and bare nu would FAIL that bar (the term is not "
                f"negligible)",
                les.err_base > LES_TOL,
                f"{les.err_base:.4%} against the same {LES_TOL:.0%} bar -- "
                f"deleting <nu_t> breaks the check above",
            )
        )
        excess = les.nu_measured - base.nu_measured
        ratio = excess / les.nu_t_eps if les.nu_t_eps else float("inf")
        checks.append(
            (
                f"the excess equals the dissipation-weighted <nu_t> to "
                f"< {EPS_TOL:.0%} (D-091)",
                abs(ratio - 1.0) < EPS_TOL,
                f"excess {excess:.6e} / <nu_t>_eps {les.nu_t_eps:.6e} = "
                f"{ratio:.4f}",
            )
        )
        checks.append(
            (
                "nu_t >= 0 at every sample (constraint 2, through tau)",
                les.nu_t_min >= 0.0,
                f"min domain mean {les.nu_t_min:.3e}",
            )
        )
        checks.append(
            (
                f"Cs = {les.cs}: peak |u| < {U_CEILING} throughout "
                f"(constraint 3)",
                les.peak_u < U_CEILING,
                f"peak |u| = {les.peak_u:.5f} over "
                f"{les.warmup + les.window} steps",
            )
        )

    if cross is not None:
        from validate.parity import STEP_TOL

        checks.append(
            (
                f"numpy vs {cross.backend}: max|du|/u0 < {STEP_TOL:.0e} at "
                f"{cross.steps} steps (D-056)",
                cross.finite and cross.step_du < STEP_TOL,
                f"{cross.step_du:.3e}   (worst |df| {cross.step_df:.3e})"
                + ("" if cross.finite else "  (NON-FINITE)"),
            )
        )
        checks.append(
            (
                f"numpy vs {cross.backend}: the measured nu agrees to "
                f"< {STEP_TOL:.0e} relative",
                cross.nu_rel < STEP_TOL,
                f"{cross.nu_rel:.3e}   ({cross.nu_ref:.8f} vs "
                f"{cross.nu_dut:.8f})",
            )
        )

    width = max(len(name) for name, _, _ in checks)
    print()
    for name, ok, detail in checks:
        print(f"    [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    if les is not None:
        print()
        print(f"  Q-202, answered: on a resolved 2D Taylor-Green at Cs = "
              f"{les.cs}, <nu_t>/nu = {les.nu_t_domain / les.nu:.4%}")
        print(f"    for calibration, session 25 measured max(nu_t)/nu = 0.1910 "
              f"on Rung 3's shedding wake")
        print(f"    and 9.011e-02 on Rung A's smooth channel. The model does "
              f"not fire hard on a smooth flow.")

    passed = all(ok for _, ok, _ in checks)
    print()
    print(f"  scope: {backend}. Rung G is a claim about **every** backend, so "
          f"it is green")
    print("         only when it has been run on each of them.")
    print("PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run Rung G and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Rung G — Taylor–Green: the closure adds the viscosity it "
        "claims to add, against an exact analytic solution"
    )
    parser.add_argument("--ny", type=int, default=NY)
    parser.add_argument("--nx", type=int, default=NX)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument(
        "--u0",
        type=float,
        default=U0,
        help=f"initial velocity amplitude, lattice units (default {U0}); it is "
        f"also the peak |u| of the whole run, and the run measures that rather "
        f"than assuming it (constraint 3)",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=CS,
        help=f"Smagorinsky constant for the closure clauses (default {CS}, the "
        f"literature value; Phase 2 does not tune it). --cs 0 runs the base "
        f"clause alone and says so — it does not silently drop the rest.",
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="the T101 backend to run every clause on (default numpy, the "
        "oracle, D-043). --backend warp additionally measures cross-backend "
        "agreement against D-056's bar.",
    )
    parser.add_argument(
        "--skip-cross",
        action="store_true",
        help="skip the cross-backend clause. Only meaningful with a non-numpy "
        "--backend, and a skipped clause is reported, never silently dropped.",
    )
    args = parser.parse_args(argv)

    # Nothing here draws, and this makes that true even for a stray SDL init.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    backend = args.backend
    try:
        get_backend(backend)
    except BackendUnavailableError as exc:
        print(f"SKIP - {exc}")
        return 2

    if args.cs < 0.0:
        print("FAIL - --cs must be non-negative: the closure adds eddy "
              "viscosity and never removes it (CLAUDE.md constraint 2).")
        return 1

    nu = nu_from_tau(args.tau)
    rho_h, u_h, kx, ky = taylor_green(args.ny, args.nx, args.u0)
    t_d = decay_time(nu, kx, ky)

    print("Rung G — Taylor–Green decay (DOCS/IDEA4.md § Validation ladder)")
    print(f"  backend {backend}   Cs = {args.cs}   filter width 1 lattice unit")
    print(f"  grid {args.ny} x {args.nx}, doubly periodic, **no bodies** "
          f"(constraint 12 is vacuous here — see the module docstring)")
    print(f"  tau {args.tau:.6f}   nu = (tau - 0.5)/3 = {nu:.8f}   "
          f"u0 = {args.u0}   k = {kx:.6f}")
    print(f"  exact: u = u0 cos(kx) sin(ky) exp(-nu K^2 t), "
          f"E ~ exp(-2 nu K^2 t), K^2 = {kx * kx + ky * ky:.6e}")
    print(f"  analytic e-folding time of E: {t_d:.1f} steps   "
          f"(warm-up {WARMUP_TD} T_d, fit {WINDOW_TD} T_d)")
    print()

    print(f"  running the Cs = 0 clause ...", flush=True)
    base = run_decay(
        args.ny, args.nx, args.tau, args.u0, 0.0, backend
    )

    les: DecayResult | None = None
    if args.cs > 0.0:
        print(f"  running the Cs = {args.cs} clause ...", flush=True)
        les = run_decay(
            args.ny, args.nx, args.tau, args.u0, args.cs, backend
        )

    cross: CrossResult | None = None
    if backend != "numpy" and not args.skip_cross:
        print(f"  running the cross-backend clause (numpy vs {backend}) ...",
              flush=True)
        cross = check_cross_backend(
            backend, args.cs, args.ny, args.nx, args.tau, args.u0
        )

    passed = report(base, les, cross, backend)

    if les is None:
        print()
        print("  NOTE: --cs 0 ran the base clause alone. Rung G's closure "
              "clauses did not run,")
        print("        so this invocation cannot make Rung G green. Run "
              f"without --cs, or with --cs {CS}.")
    if backend != "numpy" and cross is None:
        print()
        print("  NOTE: the cross-backend clause was skipped (--skip-cross).")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
