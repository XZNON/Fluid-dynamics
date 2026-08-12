"""Rung 4 — square cylinder at Re 100, plus one arbitrary convex polygon.

``DOCS/IDEA2.md`` § "Validation ladder", Rung 4::

    Square cylinder, Re 100.
    Cd ~1.5. Confirms bluff bodies and sharp corners work.

This is the **last rung**. Run it from the repo root::

    myenv/Scripts/python.exe -m validate.polygons              # live window
    myenv/Scripts/python.exe -m validate.polygons --headless   # no display

What this is, structurally
--------------------------

Rung 3's domain with a different mask. Everything that was measured into place
in session 7 is reused rather than re-derived — :func:`validate.cylinder.tau_for`
(the ``tau > 0.53`` and ``U < 0.1`` refusals, D-016), :func:`validate.cylinder.lowpass`
(the case-scaled Gaussian in front of the frequency estimate, **D-027**),
:func:`validate.cylinder.make_config`, the periodic sides and 24 D fluid span
(**D-026**), the half-cell offset, the startup kick, and the **body-only** force
link list that session 7 learned about the hard way (``Sim.links`` is built from
the whole mask, which read ``Cd = 6.65`` against a body's 1.57).

What is new here is only the geometry: :func:`lbm.geometry.regular_polygon` with
``nsides=4, rotate=pi/4`` for the square, and :func:`lbm.geometry.polygon` with
an explicit convex vertex list for the second case.

The corners are staircased and that is the expected answer
----------------------------------------------------------

``CLAUDE.md`` constraint 1 is bounce-back only — no interpolated or curved
boundaries. A square on a Cartesian lattice is the one shape a staircase
represents *exactly* along its faces, but the two separation corners are still
resolved by whichever cell centres happen to fall inside the polygon, and that
biases ``Cd`` slightly high. The acceptance window is ±0.1 around 1.5 for
exactly that reason (``DOCS/TASKS1.md`` § T008 Notes), and a high ``Cd`` is a
reason to suspect the **domain** (blockage, upstream fetch) and the **link
list**, in that order — not a reason to reach for a better boundary condition.

Why the solid interior starts at rest
-------------------------------------

See :func:`seed_solid_at_rest`. ``Sim`` seeds the whole domain, solid included,
with the equilibrium of the inlet profile, which puts *moving fluid inside the
body* at step 0. Bounce-back gives those populations no physical meaning, but it
does not remove them either, and the T008 acceptance criterion is precisely
"no fluid velocity inside the solid".

Conventions
-----------

* ``y`` increases upward; the lateral boundaries are **periodic** (**D-026**),
  which is what :func:`lbm.core.stream` already does.
* Characteristic length ``D`` is the cross-stream extent of the body's bounding
  box (**D-019**) — the side length for a square — measured from the mask, not
  passed in, and it is what ``Cd`` is divided by and what ``Re`` is defined on.
* The outlet is convective at ``lam = sqrt(cs2)`` (**D-021**).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from lbm.core import Q, W
from lbm.geometry import (
    bounding_box,
    channel_walls,
    min_thickness,
    polygon,
    regular_polygon,
)
from lbm.probe import boundary_links, forces, strouhal
from lbm.render import render
from lbm.runner import NullSink, RingBuffer, Sim, Sink, run, steps_per_frame
from validate.cylinder import (
    D_PHYS_M,
    FPS,
    KICK_FACTOR,
    KICK_TC,
    LOWPASS_SIGMA_TC,
    OFFSET_CELLS,
    PEAK_EVERY,
    PLAYBACK_SPEED,
    SPAN_D,
    U_PHYS_MS,
    VMAX_FACTOR,
    WALL,
    _peak_fluid_speed,
    lowpass,
    make_config,
    tau_for,
)

# --- reference values ---------------------------------------------------------
#
# Square cylinder in cross flow at Re 100, faces normal and parallel to the
# stream:
#
#   Cd ~ 1.5      (DOCS/IDEA2.md § Validation ladder Rung 4; the canonical
#                  range in the literature is 1.4-1.5 — Sohankar, Norberg &
#                  Davidson, Int. J. Numer. Meth. Fluids 26 (1998); Sharma &
#                  Eswaran, Numer. Heat Transfer A 45 (2004))
#   St ~ 0.145    (same sources) — printed, not asserted: DOCS/TASKS1.md § T008
#                  puts a band on Cd only.
#
# The acceptance window is DOCS/TASKS1.md § T008, not invented here.

CD_REF: float = 1.5
CD_BAND: tuple[float, float] = (1.4, 1.6)
ST_REF: float = 0.145

#: Shedding is confirmed, never assumed — same rule as Rung 3.
CL_AMPLITUDE_MIN: float = 0.01

# --- case setup ---------------------------------------------------------------

RE: float = 100.0

#: Inlet lattice velocity and body size. **Both lower/larger than Rung 3's
#: ``U = 0.06, D = 21``, and both measured rather than guessed**, because this
#: case is squeezed between two constraints that pull opposite ways.
#:
#: *Constraint 3, from above.* A square blocks more than a disc of the same
#: cross-stream extent, so the flow accelerates further around it: at
#: ``U = 0.06`` this domain measured a peak of **0.10211**, over the ceiling,
#: where the disc at the same inlet peaked at 0.09685. The ratio is
#: ``peak = 1.79 U``, measured over a **full** run — a 20 D/U look-ahead reads
#: 1.70 and is optimistic, which cost this rung one 62679-step run that got
#: ``Cd = 1.5323`` right and then failed on ``peak = 0.10031``. At ``U = 0.053``
#: the peak measures 0.0952, and the ratio would have to reach 1.89 to fail.
#:
#: *Stability, from below.* ``tau = 0.5 + 3 U D / Re`` falls when ``U`` does, and
#: D-016's ``TAU_FLOOR`` of 0.53 turns out **not** to be a safe floor for a bluff
#: body in a free stream. Measured on a small stand-in domain, 60000 steps each:
#:
#: =====================  ======  ============================
#: case                   tau     outcome
#: =====================  ======  ============================
#: square, U 0.055, D 21  0.5346  blew up at step 3200
#: **disc**, U 0.055, D 20  0.5330  blew up at step **1500**
#: square, U 0.060, D 21  0.5378  survived, peak 0.1177
#: square, U 0.055, D 31  0.5512  survived, peak 0.1114
#: =====================  ======  ============================
#:
#: The *disc* dies sooner than the square at the same ``tau``, so this is a
#: relaxation-time limit and not a staircased-corner one — constraint 1 is not
#: implicated and there is nothing here a better boundary condition would fix.
#: See :data:`TAU_FLOOR`.
#:
#: The only way to hold both at once is to buy ``tau`` with ``D``, which does not
#: move the peak at all: ``D = 30`` at ``U = 0.053`` gives ``tau = 0.5477``, well
#: clear of the measured stability threshold, with the peak at 0.0952.
U: float = 0.053
SIDE_CELLS: int = 30

#: Stability floor on ``tau`` for this rung, above D-016's 0.53. Refused at
#: setup, naming the fix, rather than reported as a physics result an hour
#: later — ``DOCS/IDEA2.md`` § Stability lists a marginal ``tau`` as the classic
#: way to get a plausible-looking checkerboard, and here it does not even
#: checkerboard, it produces ``nan`` and a ``Cd`` of ``nan``. The measured
#: threshold is between 0.5346 (blows up) and 0.5378 (60000 steps clean); 0.54
#: is the next round number above it. Rung 3 is unaffected: it runs at 0.5378,
#: measured stable for its full 45500 steps and again in the table above.
TAU_FLOOR: float = 0.54
UPSTREAM_D: float = 8.0
#: Rung 3 used 12 D. Nine buys back some of the wall clock that ``D = 27`` costs
#: and is still over constraint 12's 8 D; the wake leaves through a convective
#: outlet (D-021) that reflects 0.6% of what reaches it.
DOWNSTREAM_D: float = 9.0

#: Convective times to discard, then to measure over. Rung 3's numbers for the
#: square (its separation points are pinned to the corners, so if anything it
#: sheds sooner than a disc); the reference-free polygon case only has to run
#: clean and report finite numbers, so it gets a shorter window.
TRANSIENT_TC: float = 70.0
MEASURE_TC: float = 60.0
POLY_TRANSIENT_TC: float = 40.0
POLY_MEASURE_TC: float = 30.0

#: An arbitrary convex polygon, in units of the body half-extent, centred on the
#: origin with ``x`` streamwise. Not a regular shape and not symmetric about
#: either axis — the point of the second case is that nothing about the solver
#: is special-cased to symmetry. Convexity is asserted by
#: ``tests/test_polygons.py``; the cross-stream extent is exactly ``2 h`` so the
#: domain sizing means the same thing as it does for the square.
POLY_VERTS: tuple[tuple[float, float], ...] = (
    (-1.00, -0.35),
    (-0.35, -1.00),
    (0.60, -0.80),
    (1.00, 0.15),
    (0.20, 1.00),
    (-0.75, 0.70),
)


BodyFn = Callable[[int, int, float, float, float], NDArray[np.bool_]]


def tau_for_rung4(re: float, u: float, d_cells: float) -> tuple[float, float]:
    """``(nu, tau)`` from ``Re``, refused below :data:`TAU_FLOOR`.

    :func:`validate.cylinder.tau_for` does the physics (``CLAUDE.md``
    constraint 2: ``nu = U D / Re``, ``tau = 0.5 + 3 nu``, and no other path to
    ``nu``) and enforces D-016's floor of 0.53 and constraint 3's ceiling on
    ``U``. This adds the **measured** floor for a bluff body in a free stream on
    top of it — see :data:`U` for the table, and :data:`TAU_FLOOR` for why 0.53
    is not enough.

    Raises:
        ValueError: if ``tau`` would sit at or below :data:`TAU_FLOOR`, naming
            the ``D`` that would fix it.
    """
    nu, tau = tau_for(re, u, d_cells)
    if tau <= TAU_FLOOR:
        need = re * (TAU_FLOOR - 0.5) / (3.0 * u)
        raise ValueError(
            f"tau = {tau:.4f} is at or below Rung 4's measured stability floor "
            f"of {TAU_FLOOR} for Re = {re}, U = {u}, D = {d_cells:.0f}. Measured: "
            f"tau = 0.5346 blows up by step 3200 and tau = 0.5378 survives 60000 "
            f"steps (a disc at tau = 0.5330 blows up by step 1500, so this is a "
            f"relaxation-time limit, not the corners). Use D >= {need:.0f} cells; "
            f"raising U instead runs into the 0.1 ceiling, since the peak here "
            f"measures 1.70 U."
        )
    return nu, tau


def square_body(ny: int, nx: int, cx: float, cy: float, half: float) -> NDArray[np.bool_]:
    """Axis-aligned square of side ``2 half``, from :func:`regular_polygon`.

    ``nsides=4, rotate=pi/4`` puts the four vertices at 45, 135, 225 and 315
    degrees, so the faces are axis-aligned and the circumradius is
    ``half * sqrt(2)`` — the wrapper's own docstring names this as Rung 4's
    shape.
    """
    return regular_polygon(ny, nx, 4, cx, cy, half * math.sqrt(2.0), rotate=math.pi / 4.0)


def convex_body(ny: int, nx: int, cx: float, cy: float, half: float) -> NDArray[np.bool_]:
    """The arbitrary convex polygon of :data:`POLY_VERTS`, scaled to ``half``."""
    verts = [(cx + half * vx, cy + half * vy) for vx, vy in POLY_VERTS]
    return polygon(ny, nx, verts)


@dataclass(frozen=True)
class Case:
    """One body to run: its mask, its window, and what is asserted about it."""

    name: str
    title: str
    body: BodyFn
    d_nominal: int
    transient_tc: float
    measure_tc: float
    #: ``None`` means "no reference value asserted" — the second case of
    #: ``DOCS/TASKS1.md`` § T008 only has to run clean and report finite forces.
    cd_band: tuple[float, float] | None
    cd_ref: float | None
    st_ref: float | None
    #: Whether shedding must be present. A square at Re 100 sheds; an arbitrary
    #: polygon is not promised to, so it is not required to.
    require_shedding: bool


def cases(
    *,
    side_cells: int = SIDE_CELLS,
    transient_tc: float = TRANSIENT_TC,
    measure_tc: float = MEASURE_TC,
) -> dict[str, Case]:
    """The two Rung 4 cases, keyed by ``--case`` name."""
    return {
        "square": Case(
            name="square",
            title="square cylinder at Re 100",
            body=square_body,
            d_nominal=side_cells,
            transient_tc=transient_tc,
            measure_tc=measure_tc,
            cd_band=CD_BAND,
            cd_ref=CD_REF,
            st_ref=ST_REF,
            require_shedding=True,
        ),
        "polygon": Case(
            name="polygon",
            title="arbitrary convex polygon at Re 100 (no reference asserted)",
            body=convex_body,
            d_nominal=side_cells,
            transient_tc=POLY_TRANSIENT_TC,
            measure_tc=POLY_MEASURE_TC,
            cd_band=None,
            cd_ref=None,
            st_ref=None,
            require_shedding=False,
        ),
    }


@dataclass
class PolygonResult:
    """Everything one Rung 4 case measured."""

    case: str
    title: str
    ny: int
    nx: int
    d_cells: float
    blockage: float
    downstream_d: float
    thickness: int
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
    cd_band: tuple[float, float] | None = None
    cd_ref: float | None = None
    st_ref: float | None = None
    require_shedding: bool = False
    cd_series: NDArray[np.float64] = dc_field(default_factory=lambda: np.empty(0))
    cl_series: NDArray[np.float64] = dc_field(default_factory=lambda: np.empty(0))


# --- setup --------------------------------------------------------------------


def body_mask(
    case: Case,
    *,
    upstream_d: float = UPSTREAM_D,
    downstream_d: float = DOWNSTREAM_D,
    span_d: float = SPAN_D,
    wall: int = WALL,
    offset: float = OFFSET_CELLS,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], float, float]:
    """Body in an open channel, sized so ``check_mask`` has nothing to say.

    Identical sizing to :func:`validate.cylinder.cylinder_mask` — that is the
    point, Rung 4 is Rung 3's domain with a different mask — so the same three
    rules of ``CLAUDE.md`` constraint 12 are satisfied the same way: at least 3
    cells thick (a ``D >= 6`` convex blob is), at least 8 D of wake before the
    outlet, blockage under 10% of the **fluid** span (D-019), and the lateral
    boundaries periodic so no wall boundary layer inflates the blockage the body
    actually feels (**D-026**).

    Args:
        case: which body to build.
        upstream_d: inlet-to-leading-edge distance, in diameters.
        downstream_d: trailing-edge-to-outlet distance, in diameters.
        span_d: fluid span, in diameters.
        wall: thickness of the no-slip rows; ``0`` leaves the sides periodic.
        offset: cross-stream offset of the centre, in cells.

    Returns:
        ``(solid, body, cx, cy)`` — the full mask ``(ny, nx)`` ``bool``, the
        **body alone** for the force integral, and the centre. The two are
        returned separately for the reason session 7 recorded: ``Sim.links``
        covers the whole mask, so integrating over it reports the channel's
        friction alongside the body's drag.
    """
    d_eff = case.d_nominal + 1  # conservative digitised extent, as in Rung 3
    span = int(round(span_d * d_eff))
    ny = span + 2 * wall
    nx = int(round((upstream_d + downstream_d) * d_eff)) + case.d_nominal
    cx = upstream_d * d_eff + case.d_nominal / 2.0
    cy = (ny - 1) / 2.0 + offset

    body = case.body(ny, nx, cx, cy, case.d_nominal / 2.0)
    solid = body | channel_walls(ny, nx, wall) if wall > 0 else body.copy()
    return solid, body, cx, cy


def seed_solid_at_rest(sim: Sim) -> None:
    """Overwrite ``f`` on solid cells with the rest equilibrium ``w_i rho0``.

    Setup, not a step-loop operation — called once, before the first timestep.

    Why
    ---
    :meth:`lbm.runner.Sim._init_equilibrium` seeds the **whole** domain with the
    equilibrium of the inlet profile, solid cells included, so at step 0 there is
    fluid moving at ``U`` inside the body. Bounce-back never gives those
    populations a physical meaning, but it does not clear them either: on a solid
    cell it writes ``f[i] = f_pre[opp[i]]``, which reverses the momentum every
    step rather than removing it, and ``stream`` then walks that junk one cell
    per step towards the surface. The rest state is the fixed point of both
    operations — ``w_i rho0`` is symmetric under ``opp`` and uniform, so
    reversing it and streaming it both leave it alone.

    This is what makes ``DOCS/TASKS1.md`` § T008's "no fluid velocity inside the
    solid" a statement about the *solver* rather than about the initial
    condition. It changes nothing on the fluid side of the surface at
    steady state; it removes an artefact from the startup transient.

    Args:
        sim: the simulation to seed, modified in place.
    """
    rho0 = np.float32(sim.config.rho0)
    for i in range(Q):
        sim.f[i][sim.solid] = W[i] * rho0


def interior_solid(solid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Solid cells with no fluid among their 8 neighbours — "inside the solid".

    The surface layer is excluded deliberately, and it is not a loophole: a
    solid cell adjacent to fluid is *where bounce-back happens*. ``stream``
    carries the fluid's populations into it and the next ``bounce_back``
    reverses them back out, so its ``(e.f)/rho`` is the reflection in flight and
    is meant to be non-zero. The cells that must stay dead are the ones behind
    that layer — if the corners leaked, that is where the leak would show.

    Args:
        solid: ``(ny, nx)`` bool mask.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on solid cells fully surrounded by
        solid. Domain edges count as fluid (the conservative direction).
    """
    inner = solid.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(solid)
            ys = slice(max(dy, 0), solid.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), solid.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), solid.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), solid.shape[1] + min(-dx, 0))
            shifted[yd, xd] = solid[ys, xs]
            inner &= shifted
    return inner


# --- the run ------------------------------------------------------------------


def run_case(
    case: Case,
    *,
    re: float = RE,
    u: float = U,
    outlet_lam: float | None = None,
    headless: bool = True,
    scale: int = 1,
    vmax: float | None = None,
    verbose_mask: bool = True,
) -> PolygonResult:
    """Set up, run the wake, and measure. Printing and pass/fail are :func:`report`."""
    solid, body, cx, cy = body_mask(case)
    ny, nx = solid.shape

    # Re is defined on the length the coefficients are divided by, so nu comes
    # from the **measured** extent of the digitised body (D-019), never the
    # nominal one — a 21-cell square that rasterises to 22 rows would otherwise
    # be run at Re 105 and the 5% blamed on the solver.
    box = bounding_box(body)
    assert box is not None
    y0, y1, x0, x1 = box
    d_measured = float(y1 - y0 + 1)
    nu, tau = tau_for_rung4(re, u, d_measured)

    print(f"Rung 4 — {case.title} (DOCS/IDEA2.md § Validation ladder)")
    if case.cd_ref is not None:
        print(f"  reference: Cd ~ {case.cd_ref} (band {case.cd_band[0]}-{case.cd_band[1]}), "
              f"St ~ {case.st_ref} (printed, not asserted)")
    else:
        print("  reference: none asserted — this case must run clean and report "
              "finite Cd / Cl (DOCS/TASKS1.md § T008)")
    print(f"  grid {ny} x {nx}   D = {d_measured:.0f} cells measured "
          f"(nominal {case.d_nominal})   streamwise extent {x1 - x0 + 1} cells   "
          f"centre ({cx:.1f}, {cy:.1f})   offset {OFFSET_CELLS} cell   "
          f"sides {'periodic' if WALL == 0 else f'{WALL}-cell no-slip'}")
    print(f"  nu = U D / Re = {u} * {d_measured:.0f} / {re:.0f} = {nu:.6f}   "
          f"tau = 0.5 + 3 nu = {tau:.6f}")
    print(f"  inlet: Zou-He uniform U = {u}   outlet: convective, lam = "
          f"{'sqrt(cs2)' if outlet_lam is None else f'{outlet_lam}'} (D-021)")
    print("  geometry checks (constraint 12):")

    cfg = make_config(
        ny=ny, nx=nx, tau=tau, u=u, outlet_lam=outlet_lam,
        verbose_mask=verbose_mask, inlet_uy=KICK_FACTOR * u,
    )

    # check_mask runs inside Sim.__init__ — building the sim here is the
    # pre-run check, and a warning surfaces now rather than after the run.
    sim = Sim(cfg, solid)
    seed_solid_at_rest(sim)

    # The force integral sees the body alone (session 7's trap). WALL is 0 so
    # the two link lists coincide today; the separate list stays because the
    # correctness of Cd must not depend on that.
    body_links = boundary_links(body)
    assert sim.D == d_measured, (sim.D, d_measured)

    blockage = d_measured / (ny - 2 * WALL)
    downstream_d = (nx - 1 - x1) / d_measured
    thickness = min_thickness(body)

    dt_seconds = (u / U_PHYS_MS) * (D_PHYS_M / d_measured)
    spf = steps_per_frame(dt_seconds, FPS, PLAYBACK_SPEED)
    if vmax is None:
        vmax = VMAX_FACTOR * u / d_measured

    t_conv = d_measured / u
    transient_steps = int(round(case.transient_tc * t_conv))
    measure_steps = int(round(case.measure_tc * t_conv))
    total_steps = transient_steps + measure_steps
    kick_steps = int(round(KICK_TC * t_conv))

    print(f"  D from the mask bounding box = {d_measured:.0f} cells (D-019)   "
          f"blockage {blockage * 100:.2f}% of the fluid span   "
          f"{downstream_d:.2f} D downstream   solid {thickness} cells thick")
    print(f"  dt = {dt_seconds:.3e} s/step (D = {D_PHYS_M} m, U = {U_PHYS_MS} m/s)   "
          f"steps_per_frame = round({PLAYBACK_SPEED} / ({FPS:.0f} * dt)) = {spf} "
          f"(constraint 7)")
    print(f"  colour limits +-{vmax:.5f} = +-{VMAX_FACTOR} U / D, fixed (constraint 9)")
    print(f"  convective time D/U = {t_conv:.0f} steps   "
          f"transient {transient_steps} steps ({case.transient_tc:.0f} D/U)   "
          f"measure {measure_steps} steps ({case.measure_tc:.0f} D/U)")
    print(f"  startup kick: inlet uy = {KICK_FACTOR} U for the first {kick_steps} "
          f"steps ({KICK_TC:.0f} D/U), then zero")
    print()

    live_sink: Sink | None = None
    if not headless:
        from lbm.render import LiveSink

        live_sink = LiveSink(scale=scale, title=f"Rung 4 — {case.title}")

    cd_series = np.empty(total_steps, dtype=np.float64)
    cl_series = np.empty(total_steps, dtype=np.float64)
    peak_u = 0.0
    n = 0

    def probe(s: Sim) -> None:
        """Sample the force coefficients every timestep (D-025).

        Frame-rate sampling would alias: the shedding period here is a couple of
        thousand steps and one frame is a few dozen. Also switches the startup
        kick off in place — ``u_in`` is the cached inlet profile the Zou-He
        boundary reads every step.
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

    window = slice(min(transient_steps, ran), ran)
    cd_w = cd_series[window]
    cl_w = cl_series[window]

    if cd_w.size < 2 or saw_nan:
        st = float("nan")
        cd_mean = float(np.mean(cd_w)) if cd_w.size else float("nan")
        cd_amp = cl_amp = cl_mean = float("nan")
    else:
        # The transient is already cut, so strouhal keeps what it is handed;
        # what it sees is the low-passed series (D-027) so the domain's
        # acoustics cannot outvote the wake. The amplitude below is raw.
        st = strouhal(
            lowpass(cl_w, LOWPASS_SIGMA_TC * t_conv), 1.0, d_measured, u, transient=0.0
        )
        cd_mean = float(np.mean(cd_w))
        cd_amp = float((cd_w.max() - cd_w.min()) / 2.0)
        cl_amp = float((cl_w.max() - cl_w.min()) / 2.0)
        cl_mean = float(np.mean(cl_w))

    return PolygonResult(
        case=case.name,
        title=case.title,
        ny=ny,
        nx=nx,
        d_cells=d_measured,
        blockage=blockage,
        downstream_d=downstream_d,
        thickness=thickness,
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
        cd_band=case.cd_band,
        cd_ref=case.cd_ref,
        st_ref=case.st_ref,
        require_shedding=case.require_shedding,
        cd_series=cd_series,
        cl_series=cl_series,
    )


# --- reporting ----------------------------------------------------------------


def report(res: PolygonResult) -> bool:
    """Print the measured numbers and every check. Returns whether it passed."""
    period = res.d_cells / (res.st * res.u_inlet) if res.st > 0 else float("nan")

    print()
    print(f"  measured — {res.case}")
    print(f"    steps            {res.steps} "
          f"({res.steps - res.transient_steps} after transient)")
    print(f"    wall clock       {res.seconds:.1f} s "
          f"({res.steps_per_second:.0f} steps/s)")
    print(f"    frames           {res.frames} produced, {res.dropped} dropped by the "
          f"ring buffer (display frames only, constraint 8)")
    ref_st = "" if res.st_ref is None else f" (ref {res.st_ref})"
    print(f"    St               {res.st:.4f}{ref_st}   "
          f"shedding period {period:.0f} steps")
    ref_cd = "" if res.cd_ref is None else f"   (ref {res.cd_ref})"
    print(f"    Cd               {res.cd_mean:.4f} +- {res.cd_amp:.4f}{ref_cd}")
    pct = res.cl_amp / abs(res.cd_mean) * 100.0 if res.cd_mean else float("nan")
    print(f"    Cl               {res.cl_mean:+.4f} mean, {res.cl_amp:.4f} amplitude "
          f"({pct:.1f}% of Cd)")
    print(f"    peak |u|         {res.peak_u:.5f} lattice units")

    checks: list[tuple[str, bool, str]] = [
        ("finite", not res.saw_nan, "no nan in f or in the force series"),
        (
            "Cd and Cl finite",
            bool(np.isfinite(res.cd_mean) and np.isfinite(res.cl_amp)),
            f"Cd {res.cd_mean:.4f}, Cl amplitude {res.cl_amp:.4f}",
        ),
    ]
    if res.cd_band is not None:
        lo, hi = res.cd_band
        assert res.cd_ref is not None
        checks.append(
            (
                f"Cd in {lo}-{hi} (ref {res.cd_ref})",
                lo <= res.cd_mean <= hi,
                f"{res.cd_mean:.4f}  "
                f"({(res.cd_mean - res.cd_ref) / res.cd_ref * 100:+.1f}% vs ref)",
            )
        )
    if res.require_shedding:
        checks.append(
            (
                f"shedding present (Cl amplitude > {CL_AMPLITUDE_MIN:.0%} of Cd)",
                res.cl_amp > CL_AMPLITUDE_MIN * abs(res.cd_mean),
                f"{res.cl_amp:.4f} vs {CL_AMPLITUDE_MIN * abs(res.cd_mean):.4f}",
            )
        )
    checks += [
        (
            "blockage < 10% of the fluid span (constraint 12)",
            res.blockage < 0.10,
            f"{res.blockage * 100:.2f}%",
        ),
        (
            "at least 8 D downstream (constraint 12)",
            res.downstream_d >= 8.0,
            f"{res.downstream_d:.2f} D",
        ),
        (
            "solid at least 3 cells thick (constraint 12)",
            res.thickness >= 3,
            f"{res.thickness} cells",
        ),
        (
            "peak lattice velocity < 0.1 (constraint 3)",
            res.peak_u < 0.1,
            f"{res.peak_u:.5f}",
        ),
    ]

    width = max(len(name) for name, _, _ in checks)
    print()
    for name, ok, detail in checks:
        print(f"    [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    passed = all(ok for _, ok, _ in checks)
    print()
    print(f"{res.case}: {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv: list[str] | None = None) -> int:
    """Run Rung 4 and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Rung 4 — square cylinder at Re 100 (Cd ~ 1.5), plus one "
        "arbitrary convex polygon"
    )
    parser.add_argument(
        "--headless", action="store_true", help="run with no window (no pygame needed)"
    )
    parser.add_argument(
        "--case",
        choices=("square", "polygon", "both"),
        default="both",
        help="which body to run; default both",
    )
    parser.add_argument("--re", type=float, default=RE)
    parser.add_argument("--u", type=float, default=U, help="inlet lattice velocity")
    parser.add_argument(
        "--side", type=int, default=SIDE_CELLS, help="body cross-stream extent in cells"
    )
    parser.add_argument(
        "--transient",
        type=float,
        default=TRANSIENT_TC,
        help="convective times (D/U) discarded before measuring (square case)",
    )
    parser.add_argument(
        "--measure",
        type=float,
        default=MEASURE_TC,
        help="convective times (D/U) measured over (square case)",
    )
    parser.add_argument(
        "--outlet-lam",
        type=float,
        default=None,
        help="convective outlet advection speed; default sqrt(cs2) (D-021)",
    )
    parser.add_argument("--scale", type=int, default=1, help="window magnification")
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=f"fixed symmetric colour limit; default {VMAX_FACTOR} U / D",
    )
    args = parser.parse_args(argv)

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    table = cases(
        side_cells=args.side, transient_tc=args.transient, measure_tc=args.measure
    )
    names = ("square", "polygon") if args.case == "both" else (args.case,)

    results: list[tuple[str, bool]] = []
    for name in names:
        res = run_case(
            table[name],
            re=args.re,
            u=args.u,
            outlet_lam=args.outlet_lam,
            headless=args.headless,
            scale=args.scale,
            vmax=args.vmax,
        )
        results.append((name, report(res)))
        print()

    passed = all(ok for _, ok in results)
    print("  " + "   ".join(f"{n}: {'PASS' if ok else 'FAIL'}" for n, ok in results))
    print()
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
