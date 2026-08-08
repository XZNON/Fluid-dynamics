# Idea.md — The Big Picture

## One line

An open-source fluid dynamics engine that lets anyone drop in a shape, set a few
physical numbers, and watch the flow — without learning CFD first.

---

## The actual problem

CFD is not short of solvers. It is short of solvers a normal person can use.

Today, someone who wants to know "how does air move around this thing I designed"
has three options, and all of them are bad:

| Option                   | Why it fails them                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| OpenFOAM / SU2           | Weeks to learn. Meshing is a discipline of its own. Config is a directory tree of dictionaries. |
| Ansys Fluent / Star-CCM+ | Thousands of dollars per seat. Still needs a trained user.                                      |
| FluidX3D                 | Extremely fast, but you compile it yourself in C++, and the licence forbids commercial use.     |
| XLB                      | Genuinely good, Apache-2.0, Python. But it is a _library for people who already know LBM_.      |

The gap is not the solver. **The gap is everything around the solver.**

## What we are actually building

Not a better solver. A better _path to an answer_.

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
   [ SOLVER  — XLB underneath ]      <-- we do not rewrite this
              |
              v
   [ continuous live visual + numbers ]
```

Every box except the solver is where users currently give up. That is the product.

## Positioning

- **We do not compete with XLB.** We depend on it. Apache-2.0 makes that clean.
- **We do compete with the experience of FluidX3D** — which is fast and beautiful
  but gated behind a C++ build and a non-commercial licence.
- Our licence should be permissive (MIT or Apache-2.0) precisely because
  FluidX3D's is not. That is a real, stated gap in the ecosystem.

## Why this is defensible

The solver is a commodity — the LBM kernel is a weekend of work and three papers.
The moat is the boring, unglamorous layer:

1. **Unit translation.** Nobody should ever type a relaxation time. They type
   "air at 20 m/s" and we solve for resolution, tau, and timestep such that the
   Reynolds number matches and the sim stays stable.
2. **Geometry robustness.** Real STLs are non-watertight, have flipped normals,
   and self-intersect. Handling that gracefully is genuinely hard and genuinely
   valuable.
3. **Stability guardrails.** Detect divergence early, tell the user _why_, and
   propose the fix. Not `nan`.
4. **Results that render themselves.** Vorticity fields, drag history, pressure
   maps — in the browser, no ParaView.

## Roadmap

**Phase 0 — Learn the method (now).**
Write D2Q9 LBM in NumPy from scratch. Validate against analytic solutions.
Deliverable: `idea2.md`. Non-negotiable, because you cannot design good defaults
for a method you have not debugged yourself.

**Phase 1 — 2D engine, continuous.**
Arbitrary masks from images. Live streaming visual. Drag/lift measurement.
Config in physical units. This is the first thing worth showing anyone.

**Phase 2 — GPU.**
Port the kernel to Warp or Taichi. Same file structure, decorated hot loop.
Target: real-time 2D at 2000x1000.

**Phase 3 — Swap in XLB.**
Keep our API. Replace our kernel with XLB's. Everything above the solver line
in the diagram survives unchanged — that is the whole point of building the
layers separately.

**Phase 4 — 3D + STL.**
Voxelisation, immersed boundary, turbulence model. The hard part.

## Hardware reality check

Dev machine: 16 GB RAM, RTX 3050 4 GB.

- 2D: effectively unlimited. Even 8000x4000 fits.
- 3D: caps around 192^3 - 224^3. Fine for development and validation.
- Real vehicle aero needs 512^3+, i.e. 24 GB+. That is a rent-a-GPU-for-two-hours
  problem later, not a blocker now.

Use XLB's **Warp backend, not JAX** — JAX has no native Windows CUDA support.

## Risks, honestly

- **Scope.** "Fluid dynamics of anything" is unbounded. Phase 1 must be narrow:
  2D, external flow, incompressible, single fluid.
- **Validation debt.** A pretty wake that is physically wrong is worse than no
  tool at all. Every phase ships with a benchmark that has a known answer.
- **XLB dependency.** If it stagnates, we inherit that. Mitigated by keeping our
  own kernel working as a fallback.
- **The trap.** It is very tempting to keep polishing the solver because that
  part is fun. The solver is not the product.

## How we know it worked

Someone who has never heard of a Reynolds number opens the tool, drags in a
picture of a shape, and gets a correct, moving, believable answer in under a
minute.
