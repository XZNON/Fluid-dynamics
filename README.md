# Fluid Dynamics

An open-source fluid dynamics engine that lets anyone drop in a shape, set a few
physical numbers, and watch the flow — without learning CFD first.

## The problem

CFD is not short of solvers. It is short of solvers a normal person can use.

| Existing option          | Why it fails a non-specialist                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| OpenFOAM / SU2           | Weeks to learn. Meshing is its own discipline. Config is a directory tree of dictionaries. |
| Ansys Fluent / Star-CCM+ | Thousands of dollars per seat, and still needs a trained user.                            |
| FluidX3D                 | Extremely fast, but you compile it yourself in C++, and the licence forbids commercial use. |
| XLB                      | Genuinely good, Apache-2.0, Python — but it is a library for people who already know LBM.  |

The gap is not the solver. **The gap is everything around the solver.**

## What this is

Not a better solver — a better path to an answer.

```
   user's shape (PNG / SVG / STL)
              |
              v
   [ automatic voxelisation ]        <-- watertightness, thin walls, resolution choice
              |
              v
   [ physical config in real units ] <-- "air, 20 m/s, 1.5 m object"
              |                          NOT tau, NOT lattice velocity
              v
   [ auto-derived lattice params ]   <-- resolution, tau, timestep, stability check
              |
              v
   [ SOLVER — XLB underneath ]       <-- we do not rewrite this
              |
              v
   [ continuous live visual + numbers ]
```

Every box except the solver is where users currently give up. That is the product.

### Where the value sits

1. **Unit translation.** Nobody should ever type a relaxation time. They type
   "air at 20 m/s" and the tool solves for resolution, tau, and timestep so the
   Reynolds number matches and the sim stays stable.
2. **Geometry robustness.** Real STLs are non-watertight, have flipped normals,
   and self-intersect. Handling that gracefully is hard and valuable.
3. **Stability guardrails.** Detect divergence early, explain *why*, propose the
   fix. Not `nan`.
4. **Results that render themselves.** Vorticity fields, drag history, pressure
   maps — in the browser, no ParaView.

## Current state

**Phase 0 — learning the method.** The repo today contains potential-flow
groundwork in NumPy + matplotlib, not yet the LBM engine.

```
Navier-Fluid-Equation/
  potentialFlow.py   # analytic stream functions: uniform, source, doublet, vortex
  cylinderLift.py    # circulation -> lift, symmetric pressure, d'Alembert's paradox
  jukowski.py        # Joukowski airfoil via conformal map, Kutta condition
  panels.py          # source panel method — potential flow past any closed polygon
  polygonsDemo.py    # panel method applied to a set of polygons, surface Cp
idea.md              # the big picture
idea2.md             # Phase 0 spec: D2Q9 LBM in NumPy
```

Each script writes figures into `Navier-Fluid-Equation/images/` (git-ignored).

### Running it

```bash
pip install numpy matplotlib
cd Navier-Fluid-Equation
python potentialFlow.py
python cylinderLift.py
python jukowski.py
python polygonsDemo.py
```

## Next: the D2Q9 lattice Boltzmann engine

Spec lives in [idea2.md](idea2.md). Summary:

- State is `f` of shape `(9, ny, nx)`; per step: macroscopic, equilibrium,
  collide (BGK), stream, bounce-back, boundaries.
- Viscosity is not free — `nu = (tau - 0.5) / 3`. All stability lives there.
- Geometry is one boolean `solid` array, from primitives, PNG, or SVG.
- Simulation is decoupled from rendering: many timesteps per rendered frame,
  ring buffer between them, drop display frames but never simulation steps.
- Draw vorticity, not speed magnitude, with a symmetric diverging colormap.

### Validation ladder — no rung skipped

| Rung | Case                      | Known answer                        |
| ---- | ------------------------- | ----------------------------------- |
| 1    | Poiseuille flow           | `u(y) = (G / 2nu) y (H - y)`, L2 error < 1% |
| 2    | Lid-driven cavity, Re 100/400/1000 | Ghia et al. (1982) centreline profiles |
| 3    | Cylinder, Re 100          | Strouhal ~0.164, Cd ~1.34           |
| 4    | Square cylinder, Re 100   | Cd ~1.5                             |

A wrong sim that looks plausible is the main failure mode of this project.

## Roadmap

- **Phase 0** — D2Q9 LBM in NumPy from scratch, validated against analytic solutions.
- **Phase 1** — 2D engine, continuous. Arbitrary masks from images, live streaming
  visual, drag/lift measurement, config in physical units.
- **Phase 2** — GPU. Port the kernel to Warp or Taichi. Target real-time 2D at 2000x1000.
- **Phase 3** — Swap in XLB. Keep the API; everything above the solver line survives.
- **Phase 4** — 3D + STL. Voxelisation, immersed boundary, turbulence model.

Milestones: **M1** Poiseuille passing · **M2** cavity matches Ghia · **M3** cylinder
shedding live with correct Strouhal · **M4** arbitrary PNG mask, MP4 recording,
physical-unit config · **M5** Warp/Taichi kernel port.

M3 is the demo. M4 is the first thing anyone else can use.

## Positioning

- **Not a competitor to XLB** — a dependent. Apache-2.0 makes that clean.
- **A competitor to the FluidX3D experience**, which is fast and beautiful but
  gated behind a C++ build and a non-commercial licence.
- Licence here should be permissive (MIT or Apache-2.0) precisely because
  FluidX3D's is not.

## Hardware notes

Dev machine: 16 GB RAM, RTX 3050 4 GB.

- 2D: effectively unlimited; even 8000x4000 fits.
- 3D: caps around 192³–224³ — fine for development and validation.
- Real vehicle aero needs 512³+ (24 GB+): a rent-a-GPU-for-two-hours problem later.

Use XLB's **Warp backend, not JAX** — JAX has no native Windows CUDA support.

## Risks

- **Scope.** "Fluid dynamics of anything" is unbounded. Phase 1 stays narrow:
  2D, external flow, incompressible, single fluid.
- **Validation debt.** A pretty wake that is physically wrong is worse than no tool.
- **XLB dependency.** Mitigated by keeping our own kernel working as a fallback.
- **The trap.** Polishing the solver is the fun part. The solver is not the product.

## Definition of success

Someone who has never heard of a Reynolds number opens the tool, drags in a
picture of a shape, and gets a correct, moving, believable answer in under a minute.
