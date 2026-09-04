# Fluid Dynamics

An open-source fluid dynamics engine that lets anyone drop in a shape, set a few
physical numbers, and watch the flow — without learning CFD first.

## Quickstart

A picture, a fluid, a speed and a size. Nothing else — no Reynolds number, no
relaxation time, no grid.

```bash
git clone <this repo> "Fluid Mech" && cd "Fluid Mech"
python -m venv myenv
myenv/Scripts/pip.exe install numpy matplotlib pillow pygame imageio imageio-ffmpeg psutil pytest warp-lang build

myenv/Scripts/python.exe -m flow --shape tests/data/shapes/disc.png \
    --fluid water --speed "5 mm/s" --size "2 cm" \
    --quality fast --backend warp --no-live --quiet
```

That prints a drag coefficient, a lift amplitude, a Strouhal number and the peak
lattice velocity against its stability ceiling. On the machine this was measured
on (NVIDIA RTX 3050 Laptop GPU, driver 592.82; AMD Ryzen 7 5800H at 3201 of 3201
MHz, on mains) it takes **under a minute from a cold shell**, and the drag and
shedding frequency it prints are inside the published bands for a cylinder at
Reynolds 100 — `Cd` **1.4040** against 1.25–1.45, `St` **0.1672** against
0.155–0.175, measured end to end at **49.5 s** by `python -m validate.minute`.

`--quality` is the one knob worth knowing: `fast` is 30 cells across the body,
`balanced` (the default) 40, `accurate` 50 — finer costs roughly the cube of the
ratio in wall clock, so the default takes a few minutes rather than one. Drop
`--no-live --quiet` to watch it in a window instead of printing numbers.

Other things it does:

```bash
# see the plan and run nothing — grid, tau, timestep, run length, and why each
myenv/Scripts/python.exe -m flow --shape tests/data/shapes/disc.png \
    --fluid water --speed "5 mm/s" --size "2 cm" --explain

# write a video instead of opening a window
myenv/Scripts/python.exe -m flow --shape tests/data/shapes/disc.png \
    --fluid water --speed "5 mm/s" --size "2 cm" --out wake.mp4

# a case the solver cannot represent is refused, in your units, with fixes that
# are tested to actually work — add --nearest to run the best of them
myenv/Scripts/python.exe -m flow --shape tests/data/shapes/disc.png \
    --fluid air --speed "20 m/s" --size "1.5 m"

# the claim above, checked: Rung 3's published bands through the product path
myenv/Scripts/python.exe -m validate.minute --backend warp
```

Drop `--backend warp` to run on NumPy — same physics, same answer, slower.

**Installing it rather than cloning it** (Phase 2, T205): the tree builds a wheel named `fengdong`
that ships `lbm`, `flow` and the `fengdong` command into any virtual environment, with no checkout on
the path. It is not on PyPI yet, so the install line is the wheel, not the name:

```bash
myenv/Scripts/python.exe -m build                 # dist/fengdong-0.2.0-py3-none-any.whl
python -m venv elsewhere && elsewhere/Scripts/pip install dist/fengdong-0.2.0-py3-none-any.whl
elsewhere/Scripts/fengdong --version              # fengdong 0.2.0
elsewhere/Scripts/fengdong                        # the window (T207): drop a picture, type three numbers, see the plan
elsewhere/Scripts/python -m flow --shape disc.png --fluid water --speed "5 mm/s" --size "2 cm"
```

The base install runs the NumPy backend and the app; `pip install "dist/fengdong-0.2.0-py3-none-any.whl[gpu]"`
adds the Warp backend and `[video]` adds MP4 recording. `python -m validate.install` (Rung I) does all
of the above into a fresh venv and times it — **52.6 s** on this machine against a 60 s limit. Only
Windows is tested; the code is portable Python and nothing in it is deliberately Windows-only, but no
other platform has been run.
Every command is a validated path: see [the validation ladder](#validation-ladder--no-rung-skipped).

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

**Phase 0 and Phase 1 are complete.** The solver exists, is validated against
published data, runs on the GPU, and has a product layer over it.

```
flow/                # the product layer — physical units in, an answer out
  quantity.py        # "5 mm/s" -> SI, one dimension table
  fluids.py          # the cited fluid library (water, air, oils, honey, helium)
  autoconfig.py      # physics in, every solver parameter out; the guardrails
  diagnose.py        # refusals in prose, suggestions that run, divergence probe
  prepare.py         # picture -> runnable body mask; repair or refusal
  case.py            # the front door: Case.from_image / explain() / run()
  report.py          # Result — Cd/Cl/St/convergence, the summary, the plot
  cli.py             # python -m flow
lbm/                 # the solver — D2Q9 LBM, NumPy and Warp backends
  core.py  boundary.py  geometry.py  probe.py  runner.py
  render.py  record.py  units.py  backends/
validate/            # the validation ladder, one script per rung, PASS/FAIL
Navier-Fluid-Equation/   # prior work: potential flow / panel method (not the solver)
idea.md              # the big picture
DOCS/                # specs, plans, task contracts, live state
```

**Nine rungs are green** (four Phase 0 physics rungs, five Phase 1 product
rungs) and the headline claim is a measured one: a picture and three physical
numbers reach Rung 3's published cylinder bands in **49.5 s from a cold shell**
on a laptop RTX 3050.

The potential-flow scripts in `Navier-Fluid-Equation/` are earlier groundwork,
kept for their polygon geometry: `python potentialFlow.py` and friends write
figures into `Navier-Fluid-Equation/images/` (git-ignored). They are not part of
the LBM solver.

## The D2Q9 lattice Boltzmann engine

Spec lives in [DOCS/IDEA2.md](DOCS/IDEA2.md). Summary:

- State is `f` of shape `(9, ny, nx)`; per step: macroscopic, equilibrium,
  collide (BGK), stream, bounce-back, boundaries.
- Viscosity is not free — `nu = (tau - 0.5) / 3`. All stability lives there.
- Geometry is one boolean `solid` array, from primitives, PNG, or SVG.
- Simulation is decoupled from rendering: many timesteps per rendered frame,
  ring buffer between them, drop display frames but never simulation steps.
- Draw vorticity, not speed magnitude, with a symmetric diverging colormap.

### Validation ladder — no rung skipped

Each rung is a script in `validate/` that prints PASS or FAIL. Rung N+1 is not
started while rung N fails.

| Rung | Command | Known answer | Measured |
| ---- | ------- | ------------ | -------- |
| 1 | `python -m validate.poiseuille` | `u(y) = (G / 2nu) y (H - y)`, L2 under 1% | 0.3650% |
| 2 | `python -m validate.cavity --re 100` | Ghia et al. (1982) centreline profiles | 0.75%, vortex 0.21 cells |
| 3 | `python -m validate.cylinder` | Cylinder Re 100: `St` 0.155–0.175, `Cd` 1.25–1.45 | `St` 0.1731, `Cd` 1.4031 |
| 4 | `python -m validate.polygons` | Square cylinder Re 100, `Cd` ~1.5 | polygon `Cd` 1.4276 |
| A | `python -m validate.parity --backend warp` | The GPU backend reproduces NumPy | whole step 9.6e-06 |
| B | `python -m validate.autoconfig` | Guardrails hold; `Re` reproduced to 0.1% | 24/24 cases |
| C | `python -m validate.shapes` | A committed verdict per corpus image | 15/15 |
| D | `python -m validate.refusals` | The tool's own suggestion must run | 3/3 caught before `nan` |
| E | `python -m validate.minute --backend warp` | Rung 3's bands **through the product path**, under 60 s | `Cd` 1.4040, `St` 0.1672, **49.5 s** |

A wrong sim that looks plausible is the main failure mode of this project.

## Roadmap

- **Phase 0 — done.** D2Q9 LBM in NumPy from scratch, validated against analytic
  solutions and published data. Four rungs green, closed at **M4**.
- **Phase 1 — done.** The product layer: arbitrary masks from images, config in
  physical units, guardrails, refusals that name a working fix, live visual and
  MP4, `flow.Case` and `python -m flow` — on a Warp GPU backend, which was
  pulled forward from Phase 2 because the headline criterion is a wall clock.
  Five more rungs green, closed at **M8**.
- **Phase 2** — Swap in XLB (`idea.md`'s Phase 3). Keep the API; everything
  above the solver line survives, which is what the backend seam was built for.
- **Phase 3** — 3D + STL. Voxelisation, immersed boundary, turbulence model.

Milestones so far: **M1** Poiseuille passing · **M2** cavity matches Ghia ·
**M3** cylinder shedding live with correct Strouhal · **M4** arbitrary PNG mask,
MP4 recording, physical-unit config · **M5** Warp kernel port at parity ·
**M6** auto-configuration · **M7** geometry repair · **M8** the minute.

M3 was the demo. M4 was the first thing anyone else could use. M8 is the first
thing anyone else can use *without reading the solver's vocabulary first*.

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
