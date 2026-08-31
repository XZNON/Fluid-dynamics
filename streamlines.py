"""Run a `flow` case with the wind made visible — tracer particles in the stream.

Why this exists
---------------
``flow`` draws **vorticity** (``CLAUDE.md`` constraint 9), and that is the right
default: it shows the wake structure crisply where speed magnitude is a grey
smear. But vorticity of a uniform free stream is *zero*, so the air approaching
the body paints one flat colour. Nothing is wrong with the simulation — there is
genuinely no structure upstream to draw. It just does not look like wind
running past a shape.

A scalar field cannot fix that: a constant field is a flat colour whatever
colormap you map it through. The only thing that reads as motion in undisturbed
flow is something that *moves*, so this seeds massless tracer particles across
the domain and advects them with the velocity field. They stream in from the
left, part around the body, and get caught in the vortices behind it.

Constraint 10 — one ``render()`` — is intact: the vorticity field still goes
through :func:`lbm.render.render` and nothing here colours a field. The tracers
are stamped onto the RGB frame that ``render`` returned, which is compositing,
not a second renderer.

    myenv/Scripts/python.exe streamlines.py --shape test2.png --fluid air \
        --speed "3 m/s" --size "0.36 mm" --downstream 2.5 --out wind_tracers.mp4

Everything about the physics — grid, tau, domain, placement, run length — still
comes from ``flow.autoconfig.plan`` by way of ``flow.Case``. This script chooses
none of it.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
from numpy.typing import NDArray

from flow import Case
from flow.case import (
    KICK_FACTOR,
    KICK_TIMES,
    _seed_solid_at_rest,
)
from flow.diagnose import Monitor
from lbm.record import RecordSink
from lbm.render import render
from lbm.runner import Sim, SimConfig, run as _run

#: Tracer colour, RGB. Near-black reads against the pale background and against
#: both ends of the diverging colormap, which a mid grey would not.
TRACER_RGB: tuple[int, int, int] = (25, 25, 30)

#: Advection substeps per rendered frame. The velocity field is read once per
#: frame (constraint 8 — never per step), then held fixed while the particles
#: are walked forward in this many pieces. One big jump would cut corners around
#: the body and step straight through a vortex core.
SUBSTEPS: int = 12


def _advect(
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    ux: NDArray[np.float32],
    uy: NDArray[np.float32],
    solid: NDArray[np.bool_],
    dt: float,
    u_free: float,
    rng: np.random.Generator,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Walk the tracers through a frozen velocity field, in place.

    Nearest-cell sampling: at these grid sizes a particle crosses many cells per
    frame, so bilinear interpolation would buy smoothness the eye cannot see.

    Args:
        px, py: particle positions in cells, modified in place.
        ux, uy: velocity components, ``(ny, nx)``, in cells per timestep.
        solid: ``(ny, nx)`` bool, ``True`` on the body.
        dt: timesteps to advance across all substeps.
        u_free: free-stream speed in cells per timestep, for respawn spread.
        rng: for respawn positions.

    Returns:
        The path walked, one ``(iy, ix)`` pair per substep. Stamping all of them
        draws a streak rather than a dot, and a streak is what reads as *moving*
        air in a still frame — a scattered dot field reads as speckle.
    """
    ny, nx = solid.shape
    h = dt / SUBSTEPS
    trail: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for _ in range(SUBSTEPS):
        ix = np.clip(px.astype(np.int64), 0, nx - 1)
        iy = np.clip(py.astype(np.int64), 0, ny - 1)
        trail.append((iy, ix))
        px += ux[iy, ix] * h
        py += uy[iy, ix] * h

    # Recycle: off the right edge, or swallowed by the body. Both come back at
    # the inlet on a fresh row, which keeps the particle count constant so the
    # stream never thins out over a long run.
    #
    # Respawn x is spread over exactly one frame's worth of travel rather than
    # a fixed column. Every particle leaving in the same frame comes back at the
    # same instant, and in a uniform free stream they then move in lockstep --
    # which paints visible vertical bands marching across the picture. Smearing
    # each batch across `u_free * dt` puts this frame's batch flush against
    # where the last one has got to, and the banding goes away.
    ix = np.clip(px.astype(np.int64), 0, nx - 1)
    iy = np.clip(py.astype(np.int64), 0, ny - 1)
    gone = (px >= nx - 1) | (px < 0) | (py < 0) | (py >= ny - 1) | solid[iy, ix]
    n = int(gone.sum())
    if n:
        px[gone] = rng.uniform(0.0, max(2.0, u_free * dt), n)
        py[gone] = rng.uniform(0.0, ny - 1.0, n)
    return trail


def _stamp(
    frame: NDArray[np.uint8],
    trail: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
) -> None:
    """Draw each tracer's path onto a rendered frame, in place.

    The streak fades from background toward :data:`TRACER_RGB` along its length,
    so the head is darkest and the tail lightest. That gives every particle a
    direction the eye can read without any arrow.
    """
    n = len(trail)
    for k, (iy, ix) in enumerate(trail):
        weight = (k + 1) / n  # 0 at the tail, 1 at the head
        colour = np.array(
            [int(230 + (c - 230) * weight) for c in TRACER_RGB], dtype=np.uint8
        )
        frame[iy, ix] = colour
        if k == n - 1:  # fatten the head so the leading edge is legible
            frame[np.minimum(iy + 1, frame.shape[0] - 1), ix] = colour


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
                        "is left alone (that one sets the blockage ratio)")
    p.add_argument("--tracers", type=int, default=4000)
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
        plan = dataclasses.replace(
            plan, domain=(ny0, int(round(nx0 * args.downstream)))
        )
        case.plan = plan

    spf = max(1, steps // args.frames)
    plan = dataclasses.replace(plan, steps_per_frame=spf)
    case.plan = plan

    ny, nx = plan.domain
    u = plan.u_lattice
    solid = case._domain()
    d_cells = float(case.prepared.mask.shape[0])
    kick_steps = int(round(KICK_TIMES * (d_cells / u)))

    print(f"\ndomain     {ny} x {nx} = {ny * nx / 1e6:.2f}M cells")
    print(f"frames     {steps // spf} at {spf} steps each")
    print(f"tracers    {args.tracers}\n")

    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=plan.tau,
        inlet_U=u,
        profile="uniform",
        inlet_uy=KICK_FACTOR * u,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        inlet_axis="x",
        check_geometry=True,
        verbose_mask=False,
        backend=args.backend,
    )
    sim = Sim(cfg, solid)
    _seed_solid_at_rest(sim)

    rng = np.random.default_rng(0)
    # Seeded across the whole domain, not just at the inlet, so frame 1 already
    # shows a stream in flight rather than an empty field filling up.
    px = rng.uniform(0.0, nx - 1.0, args.tracers)
    py = rng.uniform(0.0, ny - 1.0, args.tracers)
    inside = solid[
        np.clip(py.astype(np.int64), 0, ny - 1),
        np.clip(px.astype(np.int64), 0, nx - 1),
    ]
    px[inside] = rng.uniform(0.0, 2.0, int(inside.sum()))

    watcher = Monitor()

    def per_step(s: Sim) -> None:
        """Switch the startup kick off on schedule, and watch for divergence."""
        if s.step_count == kick_steps:
            s.u_in[1].fill(0.0)
            s.refresh_inlet_profile()
        watcher(s)

    def field(s: Sim) -> NDArray[np.uint8]:
        """One frame: the one render(), with the tracers composited on top."""
        frame = render(s.vorticity(), plan.vorticity_limit)
        velocity = s.host_u()
        trail = _advect(px, py, velocity[0], velocity[1], solid, float(spf), u, rng)
        _stamp(frame, trail)
        return frame

    sink = RecordSink(args.out, fps=args.fps)
    print(f"running {steps} steps ...", flush=True)
    start = time.perf_counter()
    with sink:
        _run(
            sim,
            sink,
            steps=steps,
            steps_per_frame=spf,
            field=field,
            per_step=per_step,
            drop=False,  # a file is being written: every frame, in order (D-039)
        )
    print(f"wrote {args.out} in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
