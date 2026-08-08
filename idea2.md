# idea2.md — Phase 0: D2Q9 Lattice Boltzmann in NumPy

## Goal

A working, validated, continuously-running 2D fluid simulator in pure NumPy.
Any shape from a boolean mask. Live streaming visual, plus recordable video.

**Explicitly not the product.** This exists so we understand the method well
enough to design the layer above it. Ship it, validate it, then move on.

---

## Scope

| In                           | Out                              |
| ---------------------------- | -------------------------------- |
| 2D, D2Q9                     | 3D                               |
| Incompressible, single phase | Free surface, multiphase         |
| BGK (single relaxation time) | MRT, cumulant (later)            |
| Bounce-back walls            | Curved / interpolated boundaries |
| Re up to ~1000               | High-Re turbulence models        |
| Boolean mask geometry        | STL, meshing                     |

---

## The method, in the order the code runs it

State: `f` of shape `(9, ny, nx)`. Nine numbers per cell — how much fluid at
that cell is moving in each of nine directions.

**Constants**

```
e = [(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)]
w = [4/9,   1/9,   1/9,   1/9,    1/9,    1/36,  1/36,   1/36,    1/36  ]
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]      # index of the reversed direction
cs2 = 1/3                              # lattice speed of sound squared
```

**Per timestep**

1. **Macroscopic** — `rho = f.sum(0)`, `u = (e . f) / rho`
2. **Equilibrium** — `feq_i = w_i * rho * (1 + 3(e_i.u) + 4.5(e_i.u)^2 - 1.5 u^2)`
3. **Collide** — `f -= (f - feq) / tau`
4. **Stream** — `f[i] = np.roll(np.roll(f[i], ey_i, axis=0), ex_i, axis=1)`
5. **Bounce-back** — on solid cells, `f[i] = f_pre_stream[opp[i]]`
6. **Boundaries** — inlet velocity, outlet zero-gradient

Viscosity is not a free parameter. It is fixed by tau:

```
nu = cs2 * (tau - 0.5) = (tau - 0.5) / 3
```

Everything about stability lives in that equation. `tau -> 0.5` means
`nu -> 0` means the sim blows up.

---

## Module layout

```
lbm/
  core.py        # D2Q9 constants, equilibrium, collide, stream
  boundary.py    # bounce-back, inlet, outlet, walls
  geometry.py    # mask from primitives / PNG / polygon
  units.py       # physical <-> lattice conversion
  probe.py       # drag, lift, vorticity, residuals
  runner.py      # the continuous loop (see below)
  render.py      # frame -> RGB, colormaps
  record.py      # MP4 / GIF writer
validate/
  poiseuille.py  # step 1
  cylinder.py    # step 2
  polygons.py    # step 3
```

---

## Continuous simulation — the part that matters most

This is a _stream_, not a batch job. Design the loop for that from the start.

### Decouple simulation from rendering

They run at different rates. One rendered frame is many physical timesteps.

```
while running:
    for _ in range(steps_per_frame):     # e.g. 20
        step()                           # cheap, pure numpy
    frame = render(field())              # expensive-ish, once per display frame
    sink.push(frame)                     # screen, file, or both
```

`steps_per_frame` is the knob that makes the video look like real time
regardless of grid size. Compute it from the target physical playback speed —
do not hardcode 20.

### Three output sinks, same frame source

- **Live** — pygame surface, or matplotlib with `blit=True`. Interactive,
  drop frames if behind.
- **Record** — `imageio` / `ffmpeg` writer, fixed framerate, never drop frames.
- **Headless** — no display, write PNGs or MP4. For long runs on a server.

Same `render()` output feeds all three. Do not write three renderers.

### Never block the sim on the display

If the display is slow, the physics should not stutter. Put a small ring buffer
between them. If it fills, drop display frames, never simulation steps.

### Restartability

Long runs die. `f`, `mask`, and the step count are the entire state — pickle
them every N steps. Resume must produce a bit-identical continuation.

### What to actually draw

Vorticity, not speed.

```
omega = d(uy)/dx - d(ux)/dy
```

Speed magnitude looks like a grey smear. Vorticity with a diverging colormap
(blue/white/red, symmetric limits) makes the vortex street pop immediately.
Clip limits to a fixed range or the colours will flicker frame to frame.

---

## Geometry from a mask

The entire geometry interface is one boolean array, `solid`, shape `(ny, nx)`.

Sources, in order of implementation:

1. Primitives — circle, rectangle, polygon (reuse the vertex code we already have)
2. PNG — load, threshold alpha or luminance, resize to grid
3. SVG path — rasterise
4. (Phase 4) STL slice / voxelise

Rules that will save pain later:

- Solid must be at least 3 cells thick anywhere. Thinner and fluid leaks through
  bounce-back. Detect and warn.
- Keep the object away from the outlet — wake needs at least 8 diameters
  downstream or the outlet reflects.
- Blockage ratio (object height / domain height) under ~10%, otherwise walls
  distort the answer.

---

## Validation ladder — do not skip a rung

**Rung 1 — Poiseuille flow.** Empty channel, no-slip top and bottom, body force.
Exact answer: `u(y) = (G / 2nu) * y * (H - y)`.
Pass condition: L2 error under 1%, and halving `(tau - 0.5)` doubles centreline
velocity. This catches every sign error in collide.

**Rung 2 — Lid-driven cavity, Re 100 / 400 / 1000.**
Compare centreline profiles to Ghia et al. (1982). The standard benchmark;
tabulated values are everywhere. Catches boundary condition errors.

**Rung 3 — Cylinder, Re 100.**
Expect Strouhal ~0.164, Cd ~1.34. Both are well documented. This is the first
run that looks impressive, and the first that measures something real.

**Rung 4 — Square cylinder, Re 100.**
Cd ~1.5. Confirms bluff bodies and sharp corners work.

Each rung gets a script in `validate/` that prints pass/fail. Non-negotiable —
a wrong sim that looks plausible is the main failure mode of this whole project.

---

## Performance budget (NumPy on CPU)

The kernel is memory-bound. Roughly 9 arrays read + 9 written per step.

| Grid     | Cells | Expected                          |
| -------- | ----- | --------------------------------- |
| 400x100  | 40k   | ~500+ steps/s — interactive       |
| 800x200  | 160k  | ~150 steps/s — usable             |
| 2000x500 | 1M    | ~20 steps/s — record, don't watch |

Cheap wins before reaching for a GPU:

- Preallocate everything, never allocate inside the loop
- `float32` not `float64` — halves the bandwidth, accuracy is fine
- Fuse collide+stream into one pass over `f`
- Skip computing `feq` on solid cells

Do not optimise before Rung 3 passes.

---

## Stability — the failures you will actually hit

| Symptom                         | Cause                              | Fix                                 |
| ------------------------------- | ---------------------------------- | ----------------------------------- |
| `nan` after a few hundred steps | `tau` too close to 0.5             | raise `tau`, or raise resolution    |
| Sim fine but wake is wrong      | inlet too close to object          | more upstream space                 |
| Reflections from the right edge | outlet BC reflecting               | proper zero-gradient / sponge layer |
| Checkerboard pattern            | `tau` slightly above 0.5, marginal | raise `tau`                         |
| Flow through the object         | mask too thin                      | thicken, warn at load time          |

Hard rule: **lattice velocity under 0.1.** Above that, compressibility error
(which scales as Mach squared) stops being negligible.

---

## Milestones

- **M1** — `core.py` + Poiseuille passing. Nothing renders yet.
- **M2** — Cavity benchmark matching Ghia. Method is now trusted.
- **M3** — Cylinder shedding, live pygame window, correct Strouhal.
- **M4** — Arbitrary PNG mask, MP4 recording, physical-unit config.
- **M5** — Warp/Taichi port of the kernel, same API.

M3 is the demo. M4 is the first thing anyone else can use.

---

## Deliberately deferred

Multi-relaxation-time, Smagorinsky turbulence, curved boundaries, moving
objects, thermal coupling, adaptive refinement. All real, all later. Getting to
M4 with a correct simple solver beats getting to M2 with a sophisticated one.
