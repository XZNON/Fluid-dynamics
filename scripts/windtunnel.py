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

    myenv/Scripts/python.exe scripts/windtunnel.py --shape examples/shapes/test2.png --fluid air \
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

import sys
from pathlib import Path

# This script lives in ``scripts/`` but drives the packages at the repo root, so
# put the root on the path before importing them. Keeps ``python scripts/x.py``
# working from anywhere without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flow import Case
from flow.case import KICK_FACTOR, KICK_TIMES, _seed_solid_at_rest
from flow.diagnose import Monitor
from lbm.record import RecordSink
from lbm.render import render
from lbm.runner import Sim, SimConfig, run as _run

#: Smoke colour, RGB. Near-black: it has to stay legible over the pale
#: background AND over the deepest red and blue the colormap reaches, so it
#: cannot be a mid grey.
SMOKE_RGB: tuple[int, int, int] = (8, 8, 12)

#: The body's fill and its outline. render() paints solid cells NAN_RGB, a grey
#: close enough to the background that the model reads as a smudge rather than
#: as the thing the flow is going around. Painted explicitly instead, dark, with
#: a black edge so the profile is sharp against both the smoke and the colour.
BODY_RGB: tuple[int, int, int] = (58, 62, 72)
EDGE_RGB: tuple[int, int, int] = (0, 0, 0)

#: The flat ground the vorticity colour is dimmed toward. render()'s own
#: zero-vorticity grey, so ``--tint 0`` leaves an unbroken background rather
#: than a visible rectangle where the field was.
GROUND_RGB: tuple[int, int, int] = (221, 221, 221)

#: Advection substeps per rendered frame. Velocity is read once per frame
#: (constraint 8), then held fixed while the particles are walked forward in
#: this many pieces. One jump would cut corners and skip vortex cores.
SUBSTEPS: int = 12

#: Filament thickness in cells. One cell is a hairline that JPEG-style video
#: compression eats; two survives the encoder and still reads as a line.
SMOKE_WIDTH: int = 2

#: Particles per filament, as a multiple of the domain length in cells. Above
#: 1.0 the line is drawn denser than one particle per cell, which is what keeps
#: it continuous where the flow stretches it.
DENSITY: float = 3.0

#: Fraction of the domain length kept in the picture. The last cells are the
#: convective outlet: smoke bunches there as it waits to be recycled, and the
#: outlet condition itself is the most artificial part of the domain. Cropping
#: is honest -- the flow is still simulated there, it is simply not shown, the
#: way a tunnel photograph does not include the extractor fan.
VISIBLE: float = 0.965


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


def _outline(solid: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The one-cell rim just outside the body.

    A solid cell with at least one fluid neighbour among the four, dilated
    outward by one so the edge sits *on* the background rather than eating a
    cell of the model. Written with shifts rather than ``scipy.ndimage`` —
    ``myenv`` has no scipy and adding one for an outline is not a dependency
    this project should take (``CLAUDE.md`` § Environment).
    """
    rim = np.zeros_like(solid)
    rim[1:, :] |= solid[:-1, :]
    rim[:-1, :] |= solid[1:, :]
    rim[:, 1:] |= solid[:, :-1]
    rim[:, :-1] |= solid[:, 1:]
    return rim & ~solid


def _compose(
    frame: NDArray[np.uint8],
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    solid: NDArray[np.bool_],
    rim: NDArray[np.bool_],
    tint: float,
) -> NDArray[np.uint8]:
    """Dim the field, lay the smoke on, draw the model, crop the outlet.

    Order matters. The body goes on **after** the smoke so no filament is drawn
    across the model — a smoke line crossing a solid object is the one thing
    that would make the picture a lie.

    Returns the visible frame -- a view of ``frame``, not a copy. Particles
    beyond the crop are still advected, just not drawn, so nothing piles up
    against a wall that only exists in the picture.
    """
    if tint < 1.0:
        ground = np.array(GROUND_RGB, dtype=np.float32)
        blended = frame.astype(np.float32) * tint + ground * (1.0 - tint)
        frame[...] = blended.astype(np.uint8)

    ny, nx = frame.shape[:2]
    visible = int(nx * VISIBLE)
    shown = px < visible - 1
    ix = np.clip(px[shown].astype(np.int64), 0, visible - 1)
    iy = np.clip(py[shown].astype(np.int64), 0, ny - 1)
    for dy in range(SMOKE_WIDTH):
        frame[np.minimum(iy + dy, ny - 1), ix] = SMOKE_RGB

    frame[solid] = BODY_RGB
    frame[rim] = EDGE_RGB

    # Flip rows on the way out. render() writes solver y=0 into image row 0,
    # and row 0 displays at the TOP -- so physical "up" ends up at the bottom of
    # a saved file. lbm.render.LiveSink corrects this (frame[::-1], render.py
    # :340); RecordSink and HeadlessSink do not, so every file this project
    # writes is vertically mirrored while the window is right. from_png's own
    # flip_y does NOT cancel it: that one puts the mask into physical
    # orientation, which is exactly what render() then mirrors.
    return frame[::-1, :visible]


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
    p.add_argument("--rake", type=int, default=48,
                   help="number of smoke filaments across the inlet")
    p.add_argument("--tint", type=float, default=0.85,
                   help="0 = plain tunnel photograph, 1 = full vorticity colour")
    p.add_argument("--colour", type=float, default=0.3,
                   help="scale on the vorticity colour limit. The plan's limit "
                        "is 4U/D, sized for the shear layer at the surface; the "
                        "shed vortices are far weaker and come out pale. Below "
                        "1.0 they saturate. Still a FIXED symmetric limit "
                        "(constraint 9) -- just a smaller one, and it is a "
                        "display choice that changes no number")
    p.add_argument("--span", type=float, default=None, metavar="D",
                   help="cross-flow extent in body diameters (plan default 24, "
                        "which is Rung 3's own domain, D-075). Blockage ratio is "
                        "D/span: 24 -> 4.2%%, 14 -> 7.1%%, 10 -> 10%%, constraint "
                        "12's ceiling. MEASURED on test2.png in air at 3 m/s, "
                        "against span 24: at 14 D Cd +0.4%% (inside the run-to-run "
                        "scatter) and St +1.7%%; at 10 D Cd +1.9%% and St +5.0%%. "
                        "St is the sensitive one and rises monotonically -- "
                        "confinement changes how the wake oscillates before it "
                        "changes how hard the body is pushed. 14 D costs 42%% "
                        "fewer cells for a change inside the noise; below 12 D "
                        "you are correcting for blockage, not ignoring it")
    p.add_argument("--u-lattice", type=float, default=None, metavar="U",
                   help="override the lattice velocity (plan default 0.05). "
                        "The ONLY lever that reaches a shape whose sharp corners "
                        "accelerate the flow past the 0.1 compressibility ceiling: "
                        "the speedup is geometric, so neither --speed nor "
                        "--quality touches it. Costs steps in proportion "
                        "(0.05 -> 0.03 is 1.67x the timesteps for the same "
                        "physical time). tau is recomputed and re-checked "
                        "against the 0.54 floor")
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

    if args.u_lattice is not None:
        # tau, run length and the colour limit are all defined against
        # u_lattice, so none of them may be carried over unchanged. tau from
        # constraint 2's own formula; steps because a convective time is
        # D/u steps, so a slower lattice needs proportionally more of them for
        # the same physical time; vorticity_limit because it is 4U/D.
        u_old = plan.u_lattice
        u_new = args.u_lattice
        tau_new = 0.5 + 3.0 * u_new * plan.cells_per_length / plan.Re
        if tau_new < 0.54:
            print(
                f"\nrefused: u_lattice {u_new} gives tau {tau_new:.4f}, under "
                f"the 0.54 bluff-body floor (D-029). Raise u_lattice, or "
                f"raise --quality (tau scales with cells across)."
            )
            return 2
        steps = int(round(steps * u_old / u_new))
        plan = dataclasses.replace(
            plan,
            u_lattice=u_new,
            tau=tau_new,
            vorticity_limit=plan.vorticity_limit * u_new / u_old,
        )
        print(
            f"\nu_lattice  {u_old} -> {u_new}   tau {plan.tau:.4f}   "
            f"steps {steps} ({u_old / u_new:.2f}x)"
        )

    if args.span is not None:
        ny0, nx0 = plan.domain
        ny_new = int(round(args.span * plan.cells_per_length))
        body = case.prepared.mask.shape[0]
        if ny_new <= body + 2:
            print(f"refused: span {args.span} D is {ny_new} cells, not larger "
                  f"than the {body}-cell body.")
            return 2
        blockage = body / ny_new
        flag = "  OVER constraint 12's 10%" if blockage > 0.10 else ""
        print(f"span       {ny0} -> {ny_new} cells across the flow, "
              f"blockage {blockage:.1%}{flag}")
        plan = dataclasses.replace(plan, domain=(ny_new, nx0))

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
    rim = _outline(solid)

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
        frame = render(s.vorticity(), plan.vorticity_limit * args.colour)
        velocity = s.host_u()
        _advect(px, py, home_y, velocity[0], velocity[1], solid, float(spf), u, rng)
        return _compose(frame, px, py, solid, rim, args.tint)

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
