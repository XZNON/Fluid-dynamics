"""Run a `flow` case and draw it the way a wind tunnel photograph looks.

What a wind tunnel actually shows
---------------------------------
A smoke tunnel has a **rake** — a comb of nozzles across the inlet, each
trailing one filament of smoke into the stream. The filaments run parallel in
the undisturbed flow, part around the model, and tangle in the wake. That is
the picture people recognise, and it is a picture of *where the air goes*.

This draws the same thing: one particle line per nozzle, seeded on a fixed row
and advected by the velocity field. Filaments stay on their own row when they
recycle, so the lines stay lines instead of dissolving into speckle.

The vorticity colour is still available underneath, dimmed by ``--tint``, so
the wake structure can be read at the same time. ``--tint 0`` gives the plain
tunnel photograph; ``--tint 1`` is the full false-colour field with smoke over
the top.

Constraint 9 and 10 both hold: the field drawn is vorticity, computed in
``lbm.probe`` and coloured by the one :func:`lbm.render.render`. Dimming and
the smoke are composited onto the RGB that returns.

    myenv/Scripts/python.exe windtunnel.py --shape test2.png --fluid air \
        --speed "3 m/s" --size "0.36 mm" --downstream 2.5 --out tunnel.mp4

Physics — grid, tau, domain, placement, run length — comes from
``flow.autoconfig.plan`` via ``flow.Case``. This script chooses none of it.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
from numpy.typing import NDArray

from flow import Case
from flow.case import KICK_FACTOR, KICK_TIMES, _seed_solid_at_rest
from flow.diagnose import Monitor
from lbm.record import RecordSink
from lbm.render import render
from lbm.runner import Sim, SimConfig, run as _run

#: Smoke colour, RGB. Dark enough to read over the pale background and over
#: both ends of the diverging colormap.
SMOKE_RGB: tuple[int, int, int] = (30, 30, 36)

#: The flat ground the vorticity colour is dimmed toward. render()'s own
#: zero-vorticity grey, so ``--tint 0`` leaves an unbroken background rather
#: than a visible rectangle where the field was.
GROUND_RGB: tuple[int, int, int] = (221, 221, 221)

#: Advection substeps per rendered frame. Velocity is read once per frame
#: (constraint 8), then held fixed while the particles are walked forward in
#: this many pieces. One jump would cut corners and skip vortex cores.
SUBSTEPS: int = 12

#: Particles per filament, as a multiple of the domain length in cells. Above
#: 1.0 the line is drawn denser than one particle per cell, which is what keeps
#: it continuous where the flow stretches it.
DENSITY: float = 1.5


def _advect(
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    home_y: NDArray[np.float64],
    ux: NDArray[np.float32],
    uy: NDArray[np.float32],
    solid: NDArray[np.bool_],
    dt: float,
    u_free: float,
    rng: np.random.Generator,
) -> None:
    """Walk the smoke through a frozen velocity field, in place.

    Nearest-cell sampling: a particle crosses many cells per frame at these
    grid sizes, so bilinear interpolation buys smoothness the eye cannot see.

    Args:
        px, py: particle positions in cells, modified in place.
        home_y: the row each particle's nozzle sits on. A recycled particle
            returns to *its own* row -- that is what makes a rake of filaments
            instead of a cloud.
        ux, uy: velocity components, ``(ny, nx)``, cells per timestep.
        solid: ``(ny, nx)`` bool, ``True`` on the body.
        dt: timesteps to advance across all substeps.
        u_free: free-stream speed in cells per timestep.
        rng: for the respawn spread.
    """
    ny, nx = solid.shape
    h = dt / SUBSTEPS
    for _ in range(SUBSTEPS):
        ix = np.clip(px.astype(np.int64), 0, nx - 1)
        iy = np.clip(py.astype(np.int64), 0, ny - 1)
        px += ux[iy, ix] * h
        py += uy[iy, ix] * h

    # Recycle at the nozzle. The x spread is one frame's travel, not a fixed
    # column: every particle leaving in the same frame returns in the same
    # frame, and in a uniform stream they would then march in lockstep and
    # paint visible vertical bands. Spreading each batch over `u_free * dt`
    # lands it flush against where the previous batch has reached.
    ix = np.clip(px.astype(np.int64), 0, nx - 1)
    iy = np.clip(py.astype(np.int64), 0, ny - 1)
    gone = (px >= nx - 1) | (px < 0) | (py < 0) | (py >= ny - 1) | solid[iy, ix]
    n = int(gone.sum())
    if n:
        px[gone] = rng.uniform(0.0, max(2.0, u_free * dt), n)
        py[gone] = home_y[gone]


def _seed(
    ny: int,
    nx: int,
    solid: NDArray[np.bool_],
    rake: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Lay out the rake: ``rake`` filaments, each already streaming.

    Seeded across the full length rather than at the inlet, so frame 1 shows
    smoke in flight instead of an empty tunnel filling up.

    Returns:
        ``(px, py, home_y)``, all ``(rake * per_line,)`` float64.
    """
    per_line = int(nx * DENSITY)
    rows = np.linspace(ny * 0.04, ny * 0.96, rake)
    px = np.tile(np.linspace(0.0, nx - 1.0, per_line), rake)
    home_y = np.repeat(rows, per_line)
    py = home_y.copy()
    # A nozzle pointed straight at the body would start its line inside it.
    inside = solid[
        np.clip(py.astype(np.int64), 0, ny - 1),
        np.clip(px.astype(np.int64), 0, nx - 1),
    ]
    px[inside] = rng.uniform(0.0, 2.0, int(inside.sum()))
    return px, py, home_y


def _compose(
    frame: NDArray[np.uint8],
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    tint: float,
) -> None:
    """Dim the field toward the flat ground, then lay the smoke on top."""
    if tint < 1.0:
        ground = np.array(GROUND_RGB, dtype=np.float32)
        blended = frame.astype(np.float32) * tint + ground * (1.0 - tint)
        frame[...] = blended.astype(np.uint8)
    ny, nx = frame.shape[:2]
    ix = np.clip(px.astype(np.int64), 0, nx - 1)
    iy = np.clip(py.astype(np.int64), 0, ny - 1)
    frame[iy, ix] = SMOKE_RGB


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", required=True)
    p.add_argument("--fluid", required=True)
    p.add_argument("--speed", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--quality", default="balanced")
    p.add_argument("--backend", default="warp", choices=("numpy", "warp"))
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--downstream", type=float, default=2.5,
                   help="stretch the domain along the flow; the span across it "
                        "is left alone (it sets the blockage ratio)")
    p.add_argument("--rake", type=int, default=30,
                   help="number of smoke filaments across the inlet")
    p.add_argument("--tint", type=float, default=0.25,
                   help="0 = plain tunnel photograph, 1 = full vorticity colour")
    p.add_argument("--seconds", default=None)
    p.add_argument("--out", required=True, help=".mp4 or .gif")
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()

    case = Case.from_image(
        args.shape,
        fluid=args.fluid,
        speed=args.speed,
        size=args.size,
        quality=args.quality,
        backend=args.backend,
    )
    case.explain()
    if not case.runnable:
        return 2

    plan = case.plan
    steps = case._steps(args.seconds)
    if args.downstream != 1.0:
        ny0, nx0 = plan.domain
        plan = dataclasses.replace(plan, domain=(ny0, int(round(nx0 * args.downstream))))
    spf = max(1, steps // args.frames)
    plan = dataclasses.replace(plan, steps_per_frame=spf)
    case.plan = plan

    ny, nx = plan.domain
    u = plan.u_lattice
    solid = case._domain()
    kick_steps = int(round(KICK_TIMES * (case.prepared.mask.shape[0] / u)))

    rng = np.random.default_rng(0)
    px, py, home_y = _seed(ny, nx, solid, args.rake, rng)

    print(f"\ndomain     {ny} x {nx} = {ny * nx / 1e6:.2f}M cells")
    print(f"frames     {steps // spf} at {spf} steps each")
    print(f"rake       {args.rake} filaments, {px.size} particles")
    print(f"tint       {args.tint}\n")

    sim = Sim(
        SimConfig(
            ny=ny, nx=nx, tau=plan.tau, inlet_U=u,
            profile="uniform", inlet_uy=KICK_FACTOR * u,
            use_inlet=True, use_outlet=True, convective_outlet=True,
            inlet_axis="x", check_geometry=True, verbose_mask=False,
            backend=args.backend,
        ),
        solid,
    )
    _seed_solid_at_rest(sim)
    watcher = Monitor()

    def per_step(s: Sim) -> None:
        """Switch the startup kick off on schedule, and watch for divergence."""
        if s.step_count == kick_steps:
            s.u_in[1].fill(0.0)
            s.refresh_inlet_profile()
        watcher(s)

    def field(s: Sim) -> NDArray[np.uint8]:
        """One frame: the one render(), dimmed, with the smoke composited on."""
        frame = render(s.vorticity(), plan.vorticity_limit)
        velocity = s.host_u()
        _advect(px, py, home_y, velocity[0], velocity[1], solid, float(spf), u, rng)
        _compose(frame, px, py, args.tint)
        return frame

    sink = RecordSink(args.out, fps=args.fps)
    print(f"running {steps} steps ...", flush=True)
    start = time.perf_counter()
    with sink:
        _run(sim, sink, steps=steps, steps_per_frame=spf, field=field,
             per_step=per_step, drop=False)  # a file: every frame, in order (D-039)
    print(f"wrote {args.out} in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
