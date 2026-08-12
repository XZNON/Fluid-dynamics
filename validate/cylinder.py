"""Rung 3 — circular cylinder at Re 100: Strouhal number and drag coefficient.

``DOCS/IDEA2.md`` § "Validation ladder", Rung 3::

    Cylinder, Re 100.
    Expect Strouhal ~0.164, Cd ~1.34. Both are well documented. This is the
    first run that looks impressive, and the first that measures something real.

This is the gate for **M3**. Run it from the repo root::

    myenv/Scripts/python.exe -m validate.cylinder              # live window
    myenv/Scripts/python.exe -m validate.cylinder --headless   # no display
    myenv/Scripts/python.exe -m validate.cylinder --headless --physical
                                                  # same case, set up from
                                                  # metres and seconds (T009)

What the script actually does
-----------------------------

1. Builds the mask from :mod:`lbm.geometry` primitives — a disc plus one-cell
   no-slip channel walls — and runs :func:`lbm.geometry.check_mask` **before**
   the run. ``DOCS/PLAN1.md`` § Risks assigns "cylinder shows no shedding" to
   insufficient space or excessive blockage, and this is the check that catches
   it in a second rather than in an hour.
2. Derives ``tau`` from ``Re`` (``CLAUDE.md`` constraint 2): ``nu = U D / Re``,
   ``tau = 0.5 + 3 nu``. Nothing sets ``nu`` directly.
3. Measures steps/s with the window shut and with the window open, and prints
   the difference. Constraint 8 in its measurable form.
4. Runs the wake past its transient, sampling ``Cd`` and ``Cl`` **every
   timestep** through :func:`lbm.runner.run`'s ``per_step`` probe (D-025) —
   frame-rate sampling would alias a shedding period only a couple of thousand
   steps long.
5. Prints ``St``, ``Cd``, the lift amplitude, peak lattice velocity, and
   PASS/FAIL.

Why these numbers
-----------------

* ``U = 0.06`` and not more. Constraint 3 caps the *peak* lattice velocity at
  0.1, and the flow accelerates to roughly ``1.5 U`` around the cylinder, so an
  inlet of 0.06 puts the peak near 0.09 — the same reasoning that set Rung 2's
  ``U = 0.09`` from its lid (**D-016**).
* ``D = 24`` cells. ``tau = 0.5 + 3 U D / Re`` rises with ``D``, and ``D = 24``
  at ``U = 0.06`` gives ``tau = 0.5432`` — clear of Rung 2's ``TAU_FLOOR``
  of 0.53 — while keeping the domain inside a few minutes of wall clock.
* Domain ``254 x 552``: fluid span ``10.5 D`` so blockage is 9.5% (under the
  10% of constraint 12), ``8 D`` upstream and ``14 D`` downstream (over the
  ``8 D`` of constraint 12).
* The cylinder centre is offset **half a cell** from the channel centreline.
  ``DOCS/TASKS1.md`` § T007 Notes: "a perfectly symmetric setup on a symmetric
  grid can stay symmetric far longer than physics would". Half a cell is the
  smallest perturbation that breaks the grid's mirror symmetry, and it is
  applied from step 0 rather than after a wasted run.

Conventions
-----------

* ``y`` increases upward; row 0 and row ``ny-1`` are the no-slip walls, wall
  planes halfway between the last fluid node and the solid node (**D-009**).
* Characteristic length ``D`` is the cross-stream extent of the mask bounding
  box (**D-019**) — the same ``D`` ``check_mask`` uses. It is derived from the
  mask by :class:`lbm.runner.Sim`, not passed in, so the digitised disc's actual
  height is what ``Cd`` is divided by.
* The outlet is convective at ``lam = sqrt(cs2)`` (**D-021**); ``--outlet-lam``
  re-runs with the other defensible tuning.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field as dc_field

import numpy as np
from numpy.typing import NDArray

from lbm.geometry import bounding_box, channel_walls, check_mask, circle
from lbm.probe import boundary_links, forces, strouhal
from lbm.render import render
from lbm.runner import NullSink, RingBuffer, Sim, SimConfig, Sink, run, steps_per_frame
from lbm.units import LatticeUnits

# --- reference values ---------------------------------------------------------
#
# Cylinder in cross flow at Re 100. The canonical numbers, quoted by
# DOCS/IDEA2.md § Validation ladder and reproduced in every LBM paper that
# validates a bluff body:
#
#   St = f D / U  ~ 0.164     (Williamson, J. Fluid Mech. 206, 579-627 (1989))
#   Cd            ~ 1.34      (Braza, Chassaing & Ha Minh, JFM 165 (1986);
#                              Park, Kwon & Choi, KSME Int. J. 12 (1998))
#
# The acceptance windows are DOCS/TASKS1.md § T007, not invented here.

ST_REF: float = 0.164
CD_REF: float = 1.34
ST_BAND: tuple[float, float] = (0.155, 0.175)
CD_BAND: tuple[float, float] = (1.25, 1.45)

#: Shedding is *confirmed*, never assumed: the peak-to-peak lift after the
#: transient must exceed this fraction of the mean drag. A steady, symmetric
#: wake produces Cl ~ 1e-5 and would otherwise report a meaningless "dominant
#: frequency" from numerical noise.
CL_AMPLITUDE_MIN: float = 0.01

# --- case setup ---------------------------------------------------------------

RE: float = 100.0
U: float = 0.06
D_CELLS: int = 20
UPSTREAM_D: float = 8.0
DOWNSTREAM_D: float = 12.0
#: Fluid span in diameters. Constraint 12's rule is "under ~10% blockage" and
#: that is a *floor* on the domain, not a target: measured on this case with
#: periodic sides, 15 D of span (6.35% blockage) gives ``Cd = 1.4635``, just
#: over the top of the acceptance band, because confinement accelerates the flow
#: past the body. 24 D puts the blockage at 4.2%, well inside the rule and far
#: enough from the walls that the number being compared to an *unconfined*
#: reference is one.
SPAN_D: float = 24.0

#: Thickness of the no-slip rows at top and bottom. **Zero** — the lateral
#: boundaries are periodic, which is what :func:`lbm.core.stream` already does
#: and therefore costs no code at all.
#:
#: Measured, not assumed. With one-cell no-slip walls the free stream grows a
#: boundary layer over the 8 D upstream fetch of thickness
#: ``~5 sqrt(nu x / U) = 5 sqrt(0.0144 * 200 / 0.06)`` = **34 cells per wall**,
#: so a nominal 9.5% blockage presents the cylinder with an effective one near
#: 13%, and the measured drag climbed past the top of the acceptance band:
#: ``Cd`` 1.49 -> 1.58 -> 1.64 over 5k / 10k / 15k steps on a 264 x 524 walled
#: domain. Periodic sides have no boundary layer to grow, so the free stream the
#: cylinder sees is the free stream that was asked for.
WALL: int = 0
OFFSET_CELLS: float = 0.5

#: Startup kick: a cross-stream inlet velocity of ``KICK_FACTOR * U`` for the
#: first ``KICK_TC`` convective times, then zero. The half-cell offset alone
#: breaks the symmetry but the instability then has to grow from a perturbation
#: of order one cell, which costs hundreds of convective times of wall clock
#: for no physics — the wake it grows into is the same one. The kick is
#: switched **off** long before the measurement window opens, and the check that
#: it left nothing behind is the mean ``Cl``, which is printed and is ~0.
KICK_TC: float = 3.0
KICK_FACTOR: float = 0.10

#: Convective times (``D / U`` steps each) to discard, then to measure over.
#: 70 convective times is past the onset of shedding at Re 100 given the kick
#: above; 60 more is ~10 shedding periods, enough for the FFT to resolve one
#: peak and for the mean drag to settle.
TRANSIENT_TC: float = 70.0
MEASURE_TC: float = 60.0

#: Display. ``dt`` is what makes ``steps_per_frame`` a *computed* number
#: (constraint 7, D-023): pick a physical scale for the case — a 10 cm cylinder
#: in a 1 m/s stream — and the seconds-per-timestep follow. A validate script
#: may hold physical units; ``lbm/`` may not.
#:
#: ``--physical`` (T009) runs the **same case** through
#: :meth:`lbm.units.LatticeUnits.from_physical` instead: the physical numbers
#: below plus ``nu = U_phys D_phys / Re`` go in, and ``tau``, the lattice ``U``
#: and ``dt`` come out. It is off by default so Rung 3's published numbers keep
#: coming from the code that produced them.
D_PHYS_M: float = 0.10
U_PHYS_MS: float = 1.0
FPS: float = 60.0
PLAYBACK_SPEED: float = 1.0

#: Fixed, symmetric colour limits (constraint 9). Derived from the case, not
#: from the data: ``U / D`` is the natural vorticity scale of the wake and the
#: shed cores run a few times that.
VMAX_FACTOR: float = 4.0

#: Width of the Gaussian applied to ``Cl`` before the frequency estimate, in
#: convective times ``D / U``. See :func:`lowpass` for the measurement that put
#: it here. The shedding period is about ``6 D/U``, so half a convective time
#: costs the wake peak ~10% of its amplitude and costs the acoustic peak
#: everything.
LOWPASS_SIGMA_TC: float = 0.5

PEAK_EVERY: int = 200


@dataclass
class CylinderResult:
    """Everything one cylinder run measured."""

    ny: int
    nx: int
    d_cells: float
    blockage: float
    tau: float
    nu: float
    u_inlet: float
    steps: int
    transient_steps: int
    seconds: float
    steps_per_second: float
    dt_seconds: float
    spf: int
    st: float
    cd_mean: float
    cd_amp: float
    cl_amp: float
    cl_mean: float
    peak_u: float
    saw_nan: bool
    frames: int
    dropped: int
    headless_sps: float = float("nan")
    live_sps: float = float("nan")
    cd_series: NDArray[np.float64] = dc_field(default_factory=lambda: np.empty(0))
    cl_series: NDArray[np.float64] = dc_field(default_factory=lambda: np.empty(0))


# --- setup --------------------------------------------------------------------


def cylinder_mask(
    d_cells: int = D_CELLS,
    *,
    upstream_d: float = UPSTREAM_D,
    downstream_d: float = DOWNSTREAM_D,
    span_d: float = SPAN_D,
    wall: int = WALL,
    offset: float = OFFSET_CELLS,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], float, float]:
    """Disc in a walled channel, sized so ``check_mask`` has nothing to say.

    ``DOCS/IDEA2.md`` § Geometry from a mask, and ``CLAUDE.md`` constraint 12:
    at least 3 cells thick (a ``D >= 6`` disc is), at least 8 D of wake before
    the outlet, blockage under 10% of the **fluid** span (D-019). Session 4
    measured the failure this avoids: a ``D = 21`` cylinder in a 121-row channel
    is 17.6% blockage and warns.

    Args:
        d_cells: cylinder diameter in cells.
        upstream_d: inlet-to-leading-edge distance, in diameters.
        downstream_d: trailing-edge-to-outlet distance, in diameters.
        span_d: fluid span (wall to wall), in diameters.
        wall: thickness of the no-slip rows at top and bottom. ``0`` leaves the
            lateral boundaries periodic — see :data:`WALL`.
        offset: cross-stream offset of the centre, in cells. Half a cell breaks
            the grid's mirror symmetry so shedding starts on its own.

    Returns:
        ``(solid, cylinder, cx, cy)`` — the full mask ``(ny, nx)`` ``bool``, the
        **cylinder alone** (the walls excluded), and the centre. The two masks
        are separate because the force integral must see the body and not the
        channel: with the walls in the link list the measured ``Cd`` is the
        cylinder's *plus* the wall friction of the whole channel, which on this
        domain is 6.65 against the body's 1.57 — a beautifully converged, wholly
        wrong answer of exactly the kind ``DOCS/IDEA2.md`` § Validation ladder
        exists to catch.

    A digitised disc of nominal diameter ``d`` centred on a cell centre spans
    ``d + 1`` cells, and that measured extent is the ``D`` of D-019 — so the
    domain is sized from ``d + 1``, which keeps the blockage the check actually
    computes under 10% rather than the nominal one.
    """
    d_eff = d_cells + 1  # digitised cross-stream extent; see above
    span = int(round(span_d * d_eff))
    ny = span + 2 * wall
    nx = int(round((upstream_d + downstream_d) * d_eff)) + d_cells
    cx = upstream_d * d_eff + d_cells / 2.0
    cy = (ny - 1) / 2.0 + offset

    cylinder = circle(ny, nx, cx, cy, d_cells / 2.0)
    solid = cylinder | channel_walls(ny, nx, wall) if wall > 0 else cylinder.copy()
    return solid, cylinder, cx, cy


def tau_for(re: float, u: float, d_cells: float) -> tuple[float, float]:
    """``(nu, tau)`` from the Reynolds number (``CLAUDE.md`` constraint 2).

    ``Re = U D / nu`` fixes the viscosity; ``nu = (tau - 0.5) / 3`` inverts to
    ``tau = 0.5 + 3 nu``. There is no other path to ``nu`` in this project, and
    ``tau`` near 0.5 is the classic way to get a plausible-looking checkerboard
    (``DOCS/IDEA2.md`` § Stability), so a marginal case is refused here rather
    than reported as a physics result.

    Raises:
        ValueError: if ``tau`` would sit at or below 0.53 (Rung 2's floor,
            D-016), or if ``u`` is at or above the constraint-3 ceiling.
    """
    if u >= 0.1:
        raise ValueError(
            f"inlet U = {u} is at or above the lattice ceiling of 0.1 "
            f"(CLAUDE.md constraint 3), and flow around a cylinder accelerates "
            f"to about 1.5 U on top of that."
        )
    nu = u * d_cells / re
    tau = 0.5 + 3.0 * nu
    if tau <= 0.53:
        need = re * (0.53 - 0.5) / (3.0 * u)
        raise ValueError(
            f"tau = {tau:.4f} is at or below the 0.53 floor (D-016) for "
            f"Re = {re}, U = {u}, D = {d_cells}. Use D >= {need:.0f} cells or "
            f"raise U (subject to the 0.1 ceiling)."
        )
    return nu, tau


def make_config(
    *,
    ny: int,
    nx: int,
    tau: float,
    u: float,
    outlet_lam: float | None,
    verbose_mask: bool,
    inlet_uy: float = 0.0,
) -> SimConfig:
    """The :class:`lbm.runner.SimConfig` for an open-channel cylinder run."""
    return SimConfig(
        ny=ny,
        nx=nx,
        tau=tau,
        inlet_U=u,
        profile="uniform",
        inlet_uy=inlet_uy,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        outlet_lam=outlet_lam,
        inlet_axis="x",
        check_geometry=True,
        verbose_mask=verbose_mask,
    )


# --- the run ------------------------------------------------------------------


def _peak_fluid_speed(sim: Sim) -> float:
    """Largest ``|u|`` over the **fluid** cells.

    ``u`` on a solid cell is ``(e.f)/rho`` with a bounce-back ``rho`` and means
    nothing (D-014); including it is what made Rung 2's first residual read
    ``8.4e+01``.
    """
    fluid = ~sim.solid
    ux = sim.u[0][fluid]
    uy = sim.u[1][fluid]
    return float(np.sqrt(ux * ux + uy * uy).max())


def lowpass(series: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Gaussian smoothing of a force history, before the frequency estimate.

    Why this is here and not a tuning knob
    --------------------------------------

    A momentum-exchange force history carries the wake **and** the domain's
    acoustics. The impulsive start radiates a pressure pulse; the convective
    outlet absorbs 0.6% of it (D-021) but the Zou-He velocity inlet reflects
    essentially all of it, so a standing wave rings for the whole run. Measured
    on the walled 264 x 524 domain, the ``Cl`` spectrum had two comparable
    peaks: the wake at period 2500 steps (power 1347) and an acoustic one at
    period 305 steps (power 1378) — and the *acoustic* one was marginally
    taller, so an unfiltered FFT reported ``St = 1.49``. The oscillation is
    real, it is simply not vortex shedding: its period barely moved when ``U``
    changed from 0.06 to 0.055, which no convected structure does.

    The cutoff is set by the **case**, not by the answer: ``sigma`` is a
    fraction of the convective time ``D / U``, and a Gaussian of that width
    attenuates the shedding peak (period ~6 D/U) by ~10% while attenuating
    anything at 8x the shedding frequency by four orders of magnitude. Nothing
    here is tuned to make a number land in a band — the amplitude reported for
    the shedding check is measured on the **raw** series, and only the frequency
    estimate sees the filtered one.

    Args:
        series: the history, shape ``(n,)``.
        sigma: kernel standard deviation in samples.

    Returns:
        The smoothed series, shape ``(n - 2 * ceil(3 sigma),)`` — ``valid``
        convolution, so no edge artefact enters the spectrum.
    """
    half = int(np.ceil(3.0 * sigma))
    if sigma <= 0.0 or series.size <= 2 * half + 8:
        return np.asarray(series, dtype=np.float64)
    t = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.asarray(series, dtype=np.float64), kernel, mode="valid")


def bench_steps_per_second(
    cfg: SimConfig,
    solid: NDArray[np.bool_],
    sink: Sink,
    *,
    steps: int,
    spf: int,
    vmax: float,
) -> float:
    """Steps per second for a fresh run of ``steps`` timesteps into ``sink``.

    The two legs of the constraint-8 measurement differ **only** in the sink:
    the same config, the same mask, the same number of steps, and the same
    ``render`` call per frame. What that isolates is the cost of the window,
    which is the number ``DOCS/TASKS1.md`` § T007 asks to stay under 10%.
    """
    sim = Sim(cfg.replace(check_geometry=False), solid)
    stats = run(
        sim,
        sink,
        steps=steps,
        steps_per_frame=spf,
        field=lambda s: render(s.vorticity(), vmax),
        drop=True,
    )
    return stats.steps_per_second


def run_cylinder(
    *,
    re: float = RE,
    u: float = U,
    d_cells: int = D_CELLS,
    transient_tc: float = TRANSIENT_TC,
    measure_tc: float = MEASURE_TC,
    outlet_lam: float | None = None,
    headless: bool = True,
    scale: int = 1,
    vmax: float | None = None,
    bench_steps: int = 4000,
    verbose_mask: bool = True,
    physical: bool = False,
) -> CylinderResult:
    """Set up, benchmark the window, run the wake, and measure.

    Returns:
        A :class:`CylinderResult`. Printing and pass/fail are :func:`report`'s
        job — this function measures and does not judge.
    """
    solid, cylinder, cx, cy = cylinder_mask(d_cells)
    ny, nx = solid.shape

    # Re is defined on the length the force coefficients are divided by, so the
    # viscosity comes from the **measured** extent of the digitised disc
    # (D-019), not from the nominal diameter. A 20-cell circle occupies 21 rows;
    # using 20 here would run the case at Re 105 and blame the 5% on the solver.
    box = bounding_box(cylinder)
    assert box is not None
    d_measured = float(box[1] - box[0] + 1)

    units: LatticeUnits | None = None
    if physical:
        # T009's acceptance criterion: the same case, described in metres and
        # seconds. Re fixes the fluid — nu = U_phys D_phys / Re — and the
        # resolution is the *measured* extent of the digitised disc (D-019), so
        # this is the identical case and not merely a similar one. tau then
        # comes out of lbm.units and nothing here computes a viscosity.
        units = LatticeUnits.from_physical(
            u_phys=U_PHYS_MS,
            l_phys=D_PHYS_M,
            nu_phys=U_PHYS_MS * D_PHYS_M / re,
            cells_per_length=d_measured,
            u_lattice=u,
        )
        nu, tau = units.nu, units.tau
    else:
        nu, tau = tau_for(re, u, d_measured)

    print("Rung 3 — circular cylinder at Re 100 "
          "(DOCS/IDEA2.md § Validation ladder)")
    print(f"  reference: St ~ {ST_REF} (Williamson 1989), "
          f"Cd ~ {CD_REF} (Braza et al. 1986)")
    print(f"  grid {ny} x {nx}   D = {d_measured:.0f} cells measured "
          f"(nominal {d_cells})   centre ({cx:.1f}, {cy:.1f})   "
          f"offset {OFFSET_CELLS} cell   "
          f"sides {'periodic' if WALL == 0 else f'{WALL}-cell no-slip'}")
    if units is None:
        print(f"  nu = U D / Re = {u} * {d_measured:.0f} / {re:.0f} = {nu:.6f}   "
              f"tau = 0.5 + 3 nu = {tau:.6f}")
    else:
        print("  lattice numbers derived by lbm.units.LatticeUnits.from_physical:")
        print(units.summary())
    print(f"  inlet: Zou-He uniform U = {u}   "
          f"outlet: convective, lam = "
          f"{'sqrt(cs2)' if outlet_lam is None else f'{outlet_lam}'} (D-021)")
    print("  geometry checks (constraint 12):")

    cfg = make_config(
        ny=ny, nx=nx, tau=tau, u=u, outlet_lam=outlet_lam,
        verbose_mask=verbose_mask, inlet_uy=KICK_FACTOR * u,
    )

    # check_mask runs inside Sim.__init__; building the sim here is what
    # performs the pre-run check, and a warning would surface now.
    sim = Sim(cfg, solid)

    # The force integral must see the cylinder alone. sim.links is built from
    # the *whole* mask, so with WALL > 0 it also carries every wall cell and
    # Sim.forces() then reports the channel's friction alongside the body's
    # drag — measured at Cd = 6.65 against the body's 1.57 on the walled domain
    # this rung started from. WALL is 0 now and the two link lists coincide;
    # the separate list stays because the correctness of the number must not
    # depend on that.
    body_links = boundary_links(cylinder)
    assert sim.D == d_measured, (sim.D, d_measured)

    blockage = d_measured / (ny - 2 * WALL)

    dt_seconds = (
        units.dt if units is not None else (u / U_PHYS_MS) * (D_PHYS_M / d_measured)
    )
    spf = steps_per_frame(dt_seconds, FPS, PLAYBACK_SPEED)
    if vmax is None:
        vmax = VMAX_FACTOR * u / d_measured

    t_conv = d_measured / u
    transient_steps = int(round(transient_tc * t_conv))
    measure_steps = int(round(measure_tc * t_conv))
    total_steps = transient_steps + measure_steps
    kick_steps = int(round(KICK_TC * t_conv))

    print(f"  D from the mask bounding box = {d_measured:.0f} cells (D-019)   "
          f"blockage {blockage * 100:.2f}% of the fluid span")
    print(f"  dt = {dt_seconds:.3e} s/step (D = {D_PHYS_M} m, "
          f"U = {U_PHYS_MS} m/s)   steps_per_frame = "
          f"round({PLAYBACK_SPEED} / ({FPS:.0f} * dt)) = {spf} (constraint 7)")
    print(f"  colour limits +-{vmax:.5f} = +-{VMAX_FACTOR} U / D, fixed "
          f"(constraint 9)")
    print(f"  convective time D/U = {t_conv:.0f} steps   "
          f"transient {transient_steps} steps ({transient_tc:.0f} D/U)   "
          f"measure {measure_steps} steps ({measure_tc:.0f} D/U)")
    print(f"  startup kick: inlet uy = {KICK_FACTOR} U for the first "
          f"{kick_steps} steps ({KICK_TC:.0f} D/U), then zero")
    print()

    # --- constraint 8: does opening the window cost the physics anything? ---
    live_sink: Sink | None = None
    headless_sps = live_sps = float("nan")
    if not headless and bench_steps > 0:
        print(f"  window cost (constraint 8), {bench_steps} steps each leg:")
        headless_sps = bench_steps_per_second(
            cfg, solid, NullSink(), steps=bench_steps, spf=spf, vmax=vmax
        )
        from lbm.render import LiveSink

        live_sink = LiveSink(scale=scale, title="Rung 3 — cylinder Re 100")
        live_sps = bench_steps_per_second(
            cfg, solid, live_sink, steps=bench_steps, spf=spf, vmax=vmax
        )
        delta = (live_sps - headless_sps) / headless_sps * 100.0
        print(f"    headless {headless_sps:8.1f} steps/s   "
              f"window {live_sps:8.1f} steps/s   change {delta:+.2f}%")
        print()
    elif not headless:
        from lbm.render import LiveSink

        live_sink = LiveSink(scale=scale, title="Rung 3 — cylinder Re 100")

    # --- the wake ---------------------------------------------------------
    cd_series = np.empty(total_steps, dtype=np.float64)
    cl_series = np.empty(total_steps, dtype=np.float64)
    peak_u = 0.0
    n = 0

    def probe(s: Sim) -> None:
        """Sample the force coefficients every timestep (D-025).

        Also switches the startup kick off, in place: ``u_in`` is the cached
        inlet profile the Zou-He boundary reads every step, so zeroing its
        cross-stream row is the whole of "stop kicking" and costs no
        reallocation.
        """
        nonlocal n, peak_u
        if n < total_steps:
            cd_series[n], cl_series[n] = forces(
                s.f_bb, s.f, body_links, U=u, D=d_measured, rho0=s.config.rho0
            )
        n += 1
        if n == kick_steps:
            s.u_in[1].fill(0.0)
        if n % PEAK_EVERY == 0:
            peak_u = max(peak_u, _peak_fluid_speed(s))

    ring = RingBuffer(4)
    sink: Sink = NullSink() if live_sink is None else live_sink

    def stop(_s: Sim) -> bool:
        return bool(getattr(live_sink, "quit_requested", False))

    print(f"  running {total_steps} steps ...", flush=True)
    start = time.perf_counter()
    stats = run(
        sim,
        sink,
        steps=total_steps,
        steps_per_frame=spf,
        field=lambda s: render(s.vorticity(), vmax),
        drop=True,
        buffer=ring,
        per_step=probe,
        stop=stop,
    )
    elapsed = time.perf_counter() - start
    if live_sink is not None:
        live_sink.close()

    ran = min(n, total_steps)
    cd_series = cd_series[:ran]
    cl_series = cl_series[:ran]
    peak_u = max(peak_u, _peak_fluid_speed(sim))
    saw_nan = not bool(np.isfinite(sim.f).all()) or not bool(
        np.isfinite(cd_series).all() and np.isfinite(cl_series).all()
    )

    # --- measure ----------------------------------------------------------
    window = slice(min(transient_steps, ran), ran)
    cd_w = cd_series[window]
    cl_w = cl_series[window]

    if cd_w.size < 2 or saw_nan:
        st = float("nan")
        cd_mean = float(np.mean(cd_w)) if cd_w.size else float("nan")
        cd_amp = cl_amp = cl_mean = float("nan")
    else:
        # The transient has already been cut, so strouhal keeps everything it
        # is handed; what it sees is the low-passed series (see lowpass) so the
        # domain's acoustics cannot outvote the wake.
        st = strouhal(
            lowpass(cl_w, LOWPASS_SIGMA_TC * t_conv),
            1.0,
            d_measured,
            u,
            transient=0.0,
        )
        cd_mean = float(np.mean(cd_w))
        cd_amp = float((cd_w.max() - cd_w.min()) / 2.0)
        cl_amp = float((cl_w.max() - cl_w.min()) / 2.0)  # raw, unfiltered
        cl_mean = float(np.mean(cl_w))

    return CylinderResult(
        ny=ny,
        nx=nx,
        d_cells=d_measured,
        blockage=blockage,
        tau=tau,
        nu=nu,
        u_inlet=u,
        steps=ran,
        transient_steps=transient_steps,
        seconds=elapsed,
        steps_per_second=stats.steps_per_second,
        dt_seconds=dt_seconds,
        spf=spf,
        st=st,
        cd_mean=cd_mean,
        cd_amp=cd_amp,
        cl_amp=cl_amp,
        cl_mean=cl_mean,
        peak_u=peak_u,
        saw_nan=saw_nan,
        frames=stats.frames,
        dropped=stats.dropped,
        headless_sps=headless_sps,
        live_sps=live_sps,
        cd_series=cd_series,
        cl_series=cl_series,
    )


# --- reporting ----------------------------------------------------------------


def report(res: CylinderResult) -> bool:
    """Print the measured numbers and every check. Returns whether it passed."""
    st_lo, st_hi = ST_BAND
    cd_lo, cd_hi = CD_BAND
    period = res.d_cells / (res.st * res.u_inlet) if res.st > 0 else float("nan")

    print()
    print("  measured")
    print(f"    steps            {res.steps} "
          f"({res.steps - res.transient_steps} after transient)")
    print(f"    wall clock       {res.seconds:.1f} s "
          f"({res.steps_per_second:.0f} steps/s)")
    print(f"    frames           {res.frames} produced, {res.dropped} dropped "
          f"by the ring buffer (display frames only, constraint 8)")
    print(f"    St               {res.st:.4f}   (ref {ST_REF}, "
          f"shedding period {period:.0f} steps)")
    print(f"    Cd               {res.cd_mean:.4f} +- {res.cd_amp:.4f}   "
          f"(ref {CD_REF})")
    print(f"    Cl               {res.cl_mean:+.4f} mean, "
          f"{res.cl_amp:.4f} amplitude "
          f"({res.cl_amp / abs(res.cd_mean) * 100:.1f}% of Cd)")
    print(f"    peak |u|         {res.peak_u:.5f} lattice units")

    checks: list[tuple[str, bool, str]] = [
        ("finite", not res.saw_nan, "no nan in f or in the force series"),
        (
            f"St in {st_lo}-{st_hi} (ref {ST_REF})",
            st_lo <= res.st <= st_hi,
            f"{res.st:.4f}  ({(res.st - ST_REF) / ST_REF * 100:+.1f}% vs ref)",
        ),
        (
            f"Cd in {cd_lo}-{cd_hi} (ref {CD_REF})",
            cd_lo <= res.cd_mean <= cd_hi,
            f"{res.cd_mean:.4f}  "
            f"({(res.cd_mean - CD_REF) / CD_REF * 100:+.1f}% vs ref)",
        ),
        (
            f"shedding present (Cl amplitude > {CL_AMPLITUDE_MIN:.0%} of Cd)",
            res.cl_amp > CL_AMPLITUDE_MIN * abs(res.cd_mean),
            f"{res.cl_amp:.4f} vs {CL_AMPLITUDE_MIN * abs(res.cd_mean):.4f}",
        ),
        (
            "blockage < 10% of the fluid span (constraint 12)",
            res.blockage < 0.10,
            f"{res.blockage * 100:.2f}%",
        ),
        (
            "peak lattice velocity < 0.1 (constraint 3)",
            res.peak_u < 0.1,
            f"{res.peak_u:.5f}",
        ),
    ]

    if np.isfinite(res.headless_sps):
        delta = (res.live_sps - res.headless_sps) / res.headless_sps * 100.0
        checks.append(
            (
                "window costs < 10% of steps/s (constraint 8)",
                abs(delta) < 10.0,
                f"{res.headless_sps:.0f} -> {res.live_sps:.0f} steps/s "
                f"({delta:+.2f}%)",
            )
        )

    width = max(len(name) for name, _, _ in checks)
    print()
    for name, ok, detail in checks:
        print(f"    [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    passed = all(ok for _, ok, _ in checks)
    print()
    print("PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run Rung 3 and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Rung 3 — circular cylinder at Re 100 (St ~ 0.164, Cd ~ 1.34)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run with no window (no pygame, no display needed)",
    )
    parser.add_argument("--re", type=float, default=RE)
    parser.add_argument("--u", type=float, default=U, help="inlet lattice velocity")
    parser.add_argument(
        "--diameter", type=int, default=D_CELLS, help="cylinder diameter in cells"
    )
    parser.add_argument(
        "--transient",
        type=float,
        default=TRANSIENT_TC,
        help="convective times (D/U) discarded before measuring",
    )
    parser.add_argument(
        "--measure",
        type=float,
        default=MEASURE_TC,
        help="convective times (D/U) measured over",
    )
    parser.add_argument(
        "--outlet-lam",
        type=float,
        default=None,
        help="convective outlet advection speed; default sqrt(cs2) (D-021). "
        "Pass the inlet U to measure the other defensible tuning.",
    )
    parser.add_argument("--scale", type=int, default=1, help="window magnification")
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=f"fixed symmetric colour limit; default {VMAX_FACTOR} U / D",
    )
    parser.add_argument(
        "--physical",
        action="store_true",
        help="derive tau, the lattice U and dt through lbm.units.LatticeUnits "
        "from a physical description of the same case (T009). Off by default: "
        "Rung 3's published numbers come from tau_for.",
    )
    parser.add_argument(
        "--bench-steps",
        type=int,
        default=4000,
        help="steps per leg of the window-cost measurement; 0 skips it",
    )
    args = parser.parse_args(argv)

    if args.headless:
        # Belt and braces: nothing here imports pygame in the headless path,
        # and this makes that true even for a stray SDL init.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    res = run_cylinder(
        re=args.re,
        u=args.u,
        d_cells=args.diameter,
        transient_tc=args.transient,
        measure_tc=args.measure,
        outlet_lam=args.outlet_lam,
        headless=args.headless,
        scale=args.scale,
        vmax=args.vmax,
        bench_steps=args.bench_steps,
        physical=args.physical,
    )
    return 0 if report(res) else 1


if __name__ == "__main__":
    sys.exit(main())
