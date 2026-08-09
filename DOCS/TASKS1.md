# TASKS1.md — Phase 0 task contracts

One task per session. Plan and ordering rationale: `DOCS/PLAN1.md`. Live status: `DOCS/STATE1.md`.

**Status vocabulary:** `not_started` · `in_progress` · `blocked` · `done`
A task is `done` only when **every** acceptance criterion is checked. Code written ≠ done.

---

## Backlog index

| ID | Title | Status | Depends on | Gate |
|---|---|---|---|---|
| T001 | D2Q9 constants, macroscopic, equilibrium | `not_started` | — | unit tests |
| T002 | Collide, stream, bounce-back, body force | `not_started` | T001 | **Rung 1** |
| T003 | Moving-lid BC + cavity benchmark | `not_started` | T002 | **Rung 2** |
| T004 | Geometry primitives + mask sanity checks | `not_started` | T002 | unit tests |
| T005 | Inlet / outlet BC + probes | `not_started` | T003, T004 | unit tests |
| T006 | Runner: decoupled loop, ring buffer, restart | `not_started` | T005 | restart test |
| T007 | Render + live sink + cylinder benchmark | `not_started` | T006 | **Rung 3** |
| T008 | Square cylinder benchmark | `not_started` | T007 | **Rung 4** |
| T009 | Physical units + PNG/SVG mask | `not_started` | T004, T007 | unit tests |
| T010 | Performance pass | `not_started` | T007 | all rungs |
| T011 | Recording sinks + CLI | `not_started` | T009 | **M4** |

---

## T001 — D2Q9 constants, macroscopic, equilibrium

**Status:** `not_started`

### Goal

`lbm/core.py` exists and holds the single source of truth for the D2Q9 lattice: the constants, the
macroscopic reduction, and the equilibrium distribution. No time integration yet. At the end of this
session nothing simulates, but the two functions every later task depends on are proven correct.

### Reads / depends on

- `DOCS/IDEA2.md` § The method, in the order the code runs it
- Tasks: none

### Inputs / outputs

**In:** `f: NDArray[np.float32]` shape `(9, ny, nx)`
**Out:** `macroscopic(f) -> (rho (ny,nx), u (2,ny,nx))`; `equilibrium(rho, u) -> feq (9,ny,nx)`
Also exported: `E` (9,2) int, `W` (9,), `OPP` (9,) int, `CS2 = 1/3`, `nu_from_tau(tau)`.

### Acceptance criteria

- [ ] `lbm/__init__.py` and `lbm/core.py` exist; `E`, `W`, `OPP`, `CS2` match `DOCS/IDEA2.md` exactly, in that index order.
- [ ] `W.sum() == 1` to float32 tolerance, and `E[OPP[i]] == -E[i]` for all `i`.
- [ ] `equilibrium(rho, u).sum(axis=0)` equals `rho` to within `1e-5` for random `rho` in `[0.9,1.1]`, random `|u| < 0.1`.
- [ ] First moment holds: `(E.T @ equilibrium(rho,u).reshape(9,-1)).reshape(2,ny,nx)` equals `rho*u` to within `1e-5`.
- [ ] Round trip: `macroscopic(equilibrium(rho, u))` returns the same `rho` and `u` to within `1e-5`.
- [ ] `nu_from_tau(tau)` returns `(tau - 0.5) / 3` and raises `ValueError` naming `tau` when `tau <= 0.5`.
- [ ] All arrays returned are `float32`; asserted in a test.
- [ ] `myenv/Scripts/python.exe -m pytest tests/test_core.py` green.

### Constraints that bite here

- Constraint 4 — `(9, ny, nx)` index order and `float32`. Every later module inherits this; getting it wrong here is expensive.
- Constraint 2 — viscosity only via `tau`. `nu_from_tau` is the only path.
- Constraint 3 — the `|u| < 0.1` bound is why the equilibrium tests only probe that range; document it.

### Notes

`pytest` is not yet installed in `myenv` — install it this session and record it in `DOCS/STATE1.md`
§ Environment. Resist writing `collide` "since it's three lines" — T002 owns it and Rung 1 is what
proves it.

---

## T002 — Collide, stream, bounce-back, body force → Rung 1

**Status:** `not_started`

### Goal

The full timestep runs, and Poiseuille flow in an empty channel matches the analytic parabola. This
is **M1**. The pass condition catches every sign error in `collide`, which is why it comes before
anything visual.

### Reads / depends on

- `DOCS/IDEA2.md` § The method / § Validation ladder Rung 1 / § Stability
- Tasks: T001

### Inputs / outputs

**In:** `f`, `solid` mask, `tau`, body force `G`
**Out:** `lbm/core.py::collide`, `lbm/core.py::stream`, `lbm/boundary.py::bounce_back`,
`lbm/boundary.py::apply_body_force`, `validate/poiseuille.py` printing PASS/FAIL.

### Acceptance criteria

- [ ] `collide(f, feq, tau)` implements `f -= (f - feq) / tau` in place, no allocation.
- [ ] `stream(f, buf)` shifts each `f[i]` by `E[i]` — `roll` on axis 0 by `ey`, axis 1 by `ex` — with the sign convention documented in the docstring and verified by a test that streams a single-cell spike and checks it lands one cell along `E[i]`.
- [ ] `bounce_back` uses the **pre-stream** copy: on solid cells `f[i] = f_pre[OPP[i]]`.
- [ ] `validate/poiseuille.py` runs an empty channel, no-slip top and bottom, constant body force, to steady state, and prints `PASS`/`FAIL` plus the L2 error.
- [ ] **L2 relative error against `u(y) = (G / 2nu) * y * (H - y)` is under 1%.**
- [ ] **Halving `(tau - 0.5)` doubles centreline velocity** to within 2% — asserted in the script, not eyeballed.
- [ ] Mass is conserved: `f.sum()` drifts less than `1e-4` relative over 5000 steps.
- [ ] No `nan` after 20000 steps at `tau = 0.6`.
- [ ] Peak lattice velocity in the run is under 0.1 and the script prints it.

### Constraints that bite here

- Constraint 2 — the pass condition *is* `nu = (tau - 0.5)/3`. If the doubling check fails, the bug is in collide or in the force term, not in the analytic solution.
- Constraint 5 — Rung 1 must be green before T003 starts. No exceptions.
- Constraint 6 — do not fuse collide and stream yet, however tempting.
- Wall placement: decide and document whether walls sit on the last fluid node or halfway between (`H` in the analytic formula differs by one cell). Log the choice in `DOCS/STATE1.md` § Decisions — it will bite Rung 2.

### Notes

If L2 error sits stubbornly near 2–3%, suspect the wall-offset convention before suspecting collide.
The classic fix is `H = ny - 1` vs `H = ny`; try both and record which one the code assumes.

---

## T003 — Moving-lid BC + cavity benchmark → Rung 2

**Status:** `not_started`

### Goal

Lid-driven cavity at Re 100, 400 and 1000 matches Ghia et al. (1982) centreline profiles. This is
**M2** — the point at which the method is trusted. Rung 1 proved collide; this proves boundaries.

### Reads / depends on

- `DOCS/IDEA2.md` § Validation ladder Rung 2
- Tasks: T002

### Inputs / outputs

**In:** square domain, no-slip on three walls, prescribed tangential velocity on the top lid, Re
**Out:** `lbm/boundary.py::moving_wall`, `validate/cavity.py` printing PASS/FAIL per Re

### Acceptance criteria

- [ ] `moving_wall` imposes a tangential wall velocity (momentum-corrected bounce-back or Zou–He — state which in the docstring).
- [ ] Ghia reference values for `ux` along the vertical centreline and `uy` along the horizontal centreline are stored as a literal table in `validate/cavity.py` with the citation, at the standard 17 sample points.
- [ ] `validate/cavity.py --re {100,400,1000}` runs to steady state (residual `max|u_n - u_{n-1}| / U < 1e-6`) and prints per-Re PASS/FAIL.
- [ ] **Max absolute deviation from Ghia is under 5% of the lid velocity at all sampled points, for all three Re.**
- [ ] The primary vortex centre location is printed and lies within 2 cells of Ghia's for each Re.
- [ ] The script prints resolution, `tau`, and peak lattice velocity per case; peak stays under 0.1.
- [ ] Rung 1 re-run and still green.

### Constraints that bite here

- Constraint 3 — at Re 1000 with lid velocity 0.1 the grid must be large enough that `tau` stays comfortably above 0.5. Compute the required resolution from `Re = U*L/nu`, print it, don't guess.
- Constraint 5 — three sub-cases; PASS means all three.
- The wall-offset decision from T002 changes the effective `L` here. Use the same convention and say so.

### Notes

This is the task most likely to need two sessions. `DOCS/PLAN1.md` § Risks defines the valve: timebox
to one session, then log and try Zou–He walls. Corner cells at the lid are the usual culprit —
decide explicitly whether corners are lid or wall.

---

## T004 — Geometry primitives + mask sanity checks

**Status:** `not_started`

### Goal

`lbm/geometry.py` turns primitives into the one boolean array the solver understands, and refuses —
loudly — to hand back a mask that will produce a wrong answer.

### Reads / depends on

- `DOCS/IDEA2.md` § Geometry from a mask
- Tasks: T002
- Prior work to reuse: `Navier-Fluid-Equation/polygonsDemo.py`, `panels.py` (vertex handling)

### Inputs / outputs

**In:** grid shape, primitive parameters (circle centre/radius, rect bounds, polygon vertices)
**Out:** `solid: NDArray[np.bool_]` shape `(ny, nx)`, plus `check_mask(...) -> list[str]` warnings

### Acceptance criteria

- [ ] `circle`, `rectangle`, `polygon` each return a `(ny, nx)` bool array; polygon handles concave shapes and is tested against a known-area convex case to within 2%.
- [ ] `channel_walls(ny, nx)` returns top/bottom no-slip rows, composable with `|`.
- [ ] `check_mask(solid, inlet_axis, ...)` returns a warning string, not silence, when: min solid thickness `< 3` cells; object closer than 8 characteristic lengths to the outlet; blockage ratio `> 10%`.
- [ ] Thickness check verified by a test on a deliberately 1-cell-thick diagonal line and a 4-cell-thick block — warns for the first, not the second.
- [ ] Warnings are emitted through `warnings.warn`, and `check_mask(..., strict=True)` raises instead.
- [ ] Characteristic length used for blockage/downstream checks is derived from the mask bounding box and printed.
- [ ] `pytest tests/test_geometry.py` green; Rungs 1–2 still green.

### Constraints that bite here

- Constraint 12 — all three checks. Skipping the thickness warning is how "flow through the object" reaches Rung 3 and wastes a session.
- Constraint 4 — mask is `(ny, nx)`, matching `f`'s trailing axes. Not `(nx, ny)`.

### Notes

Reuse the prior polygon code rather than rewriting a point-in-polygon test, but keep the import
one-directional: `lbm/` may read from `Navier-Fluid-Equation/` concepts, never the reverse.

---

## T005 — Inlet / outlet BC + probes

**Status:** `not_started`

### Goal

Open-channel flow: velocity in at the left, non-reflecting out at the right — plus the measurement
code that makes Rung 3 checkable. Without probes, "the wake looks right" is all we'd have.

### Reads / depends on

- `DOCS/IDEA2.md` § The method step 6 / § Stability / § What to actually draw
- Tasks: T003, T004

### Inputs / outputs

**In:** `f`, `solid`, inlet velocity `U`
**Out:** `lbm/boundary.py::inlet_velocity`, `::outlet_zero_gradient`;
`lbm/probe.py::vorticity`, `::forces` (Cd, Cl), `::strouhal`, `::residual`

### Acceptance criteria

- [ ] `inlet_velocity` imposes a prescribed profile (uniform or parabolic, selectable) — Zou–He or equilibrium-based, stated in the docstring.
- [ ] `outlet_zero_gradient` copies the second-to-last column; a test confirms a pressure pulse crossing the outlet reflects less than 5% of its amplitude.
- [ ] `vorticity(u)` returns `d(uy)/dx - d(ux)/dy` via central differences, one-sided at edges, masked to `nan` on solid cells.
- [ ] `forces(f_pre, f_post, solid)` computes drag and lift by **momentum exchange** over boundary links, returning dimensionless `Cd`, `Cl` given `U` and characteristic length.
- [ ] `forces` validated on a known case: uniform flow with no obstacle gives `|Cd| < 1e-6`.
- [ ] `strouhal(cl_series, dt)` finds the dominant frequency via FFT, ignores the first 30% of the series as transient, and returns `St = f*D/U`.
- [ ] `strouhal` verified against a synthetic sine of known frequency to within 1%.
- [ ] `residual(u_now, u_prev, U)` returns `max|Δu|/U`.
- [ ] `pytest tests/test_probe.py` green; Rungs 1–2 still green.

### Constraints that bite here

- Constraint 9 — vorticity is the field that gets drawn; it is computed here, not in `render.py`. `render.py` colours arrays, it does not do physics.
- Constraint 3 — inlet `U` under 0.1. `inlet_velocity` warns if asked for more.
- Momentum-exchange drag needs both pre- and post-stream `f`. Decide where the runner keeps that copy now, so T006 doesn't have to reshape the API.

### Notes

`forces` is the single most error-prone function in the project and Rung 3's `Cd ≈ 1.34` is what
audits it. Write it so the link list is precomputed once from the mask, not rebuilt per step.

---

## T006 — Runner: decoupled loop, ring buffer, restart

**Status:** `not_started`

### Goal

The simulation becomes a stream rather than a batch job: physics at its own rate, frames pulled off
at another, and a run that can die and resume bit-identically.

### Reads / depends on

- `DOCS/IDEA2.md` § Continuous simulation — the part that matters most
- Tasks: T005

### Inputs / outputs

**In:** a sim config (grid, `tau`, mask, inlet `U`), target playback speed
**Out:** `lbm/runner.py::Sim` (owns `f`, `solid`, `step_count`), `::run`, `::save_checkpoint`,
`::load_checkpoint`, a `RingBuffer`, and an abstract `Sink` with a no-op implementation

### Acceptance criteria

- [ ] `Sim.step()` runs one full timestep; all buffers preallocated in `__init__`, verified by a test asserting no change in `f.__array_interface__['data']` and (via `tracemalloc`) no growth over 1000 steps.
- [ ] `steps_per_frame` is **computed** from target physical playback speed, grid size and `dt` — a function with a docstring showing the arithmetic, not a constant.
- [ ] `RingBuffer(maxlen)` drops the **oldest frame** when full and increments a `dropped` counter; a test with a deliberately slow sink confirms `dropped > 0` while `step_count` is unaffected.
- [ ] `Sink` is an abstract base with `push(frame)` and `close()`; `NullSink` implemented. Live/record sinks are T007/T011.
- [ ] `save_checkpoint(path)` pickles exactly `f`, `solid`, `step_count`, and the config.
- [ ] **Bit-identical restart is a test:** run 500 steps, checkpoint, run 500 more, record `f`; reload the checkpoint, run 500, and assert `np.array_equal` with the recorded `f`.
- [ ] Auto-checkpoint every N steps, N configurable, off by default.
- [ ] `pytest tests/test_runner.py` green; Rungs 1–2 still green.

### Constraints that bite here

- Constraint 7 — `steps_per_frame` computed. A hardcoded 20 fails this task.
- Constraint 8 — the buffer drops display frames, never steps. That is the assertion above.
- Constraint 11 — bit-identical, hence `float32` determinism: no `float64` intermediates that round differently on resume, and no RNG in the step path.

### Notes

Built before rendering exists, deliberately: it is much easier to prove the buffer drops frames with
a fake slow sink than with a real window in the way.

---

## T007 — Render + live sink + cylinder benchmark → Rung 3

**Status:** `not_started`

### Goal

A live window showing a von Kármán vortex street behind a cylinder at Re 100, with a measured
Strouhal number and drag coefficient that match the literature. This is **M3** — the demo.

### Reads / depends on

- `DOCS/IDEA2.md` § What to actually draw / § Three output sinks / § Validation ladder Rung 3
- Tasks: T006

### Inputs / outputs

**In:** `Sim`, colormap limits
**Out:** `lbm/render.py::render(field, limits) -> RGB uint8 (ny,nx,3)`, `LiveSink` (pygame),
`validate/cylinder.py` printing PASS/FAIL

### Acceptance criteria

- [ ] `render` maps a scalar field to `uint8` RGB with a **diverging** colormap and **symmetric fixed** limits passed in — never computed per frame.
- [ ] A test renders two frames with different data and identical limits and asserts the mapping of a fixed value is byte-identical across them (no flicker).
- [ ] `LiveSink` opens a pygame window, pulls from the ring buffer, and drops frames when behind — no `pygame` call sits inside the physics loop.
- [ ] Measured: opening the window changes steps/s by **less than 10%** versus headless, printed by the validation script.
- [ ] `validate/cylinder.py` sets up circular cylinder at Re 100 with the T004 sanity checks passing (≥8D downstream, <10% blockage), runs past transient, and prints St, Cd, PASS/FAIL.
- [ ] **`St` within 0.155–0.175 (ref 0.164) and `Cd` within 1.25–1.45 (ref 1.34).**
- [ ] Shedding is confirmed present, not assumed: the Cl series amplitude after transient exceeds 1% of Cd.
- [ ] `--headless` flag runs the same validation with no window.
- [ ] Rungs 1–2 still green.

### Constraints that bite here

- Constraint 9 — vorticity, diverging map, fixed symmetric limits. Speed magnitude is not acceptable output.
- Constraint 10 — one `render()`. `LiveSink` consumes its output; it does not colour anything itself.
- Constraint 8 — the <10% steps/s claim is the measurable form of "never block the sim on the display".
- Constraint 6 lifts **after** this task passes, not before.

### Notes

`pygame` needs installing into `myenv`; record it in `DOCS/STATE1.md` § Environment. If shedding
doesn't start, perturb the initial condition slightly or offset the cylinder half a cell — a
perfectly symmetric setup on a symmetric grid can stay symmetric far longer than physics would.

---

## T008 — Square cylinder benchmark → Rung 4

**Status:** `not_started`

### Goal

Confirm bluff bodies with sharp corners work: square cylinder at Re 100, `Cd ≈ 1.5`. The last rung.

### Reads / depends on

- `DOCS/IDEA2.md` § Validation ladder Rung 4
- Tasks: T007

### Inputs / outputs

**In:** square (and one non-trivial polygon) mask from `lbm/geometry.py`
**Out:** `validate/polygons.py` printing PASS/FAIL

### Acceptance criteria

- [ ] `validate/polygons.py` runs a square cylinder at Re 100 and prints Cd, St, PASS/FAIL.
- [ ] **`Cd` within 1.4–1.6 (ref ~1.5).**
- [ ] A second case — an arbitrary convex polygon — runs to completion without `nan` and reports finite Cd/Cl, no reference value asserted.
- [ ] Corner cells behave: a test asserts no fluid velocity inside the solid (`|u| < 1e-6` on solid cells).
- [ ] Rungs 1–3 re-run and still green.

### Constraints that bite here

- Constraint 12 — sharp corners are exactly where a thin-mask warning matters; the sanity checks must pass, not be suppressed.
- Constraint 1 — bounce-back only. A staircased corner is the expected answer here; do not reach for interpolated boundaries to improve `Cd`.

### Notes

Staircase corners give slightly high `Cd` — that's why the window is ±0.1 rather than ±0.02.

---

## T009 — Physical units + PNG/SVG mask

**Status:** `not_started`

### Goal

The user speaks physics ("air, 20 m/s, 1.5 m object, PNG of a wing") and the code derives resolution,
`tau`, and timestep — refusing configs that would be unstable. First step toward **M4**.

### Reads / depends on

- `DOCS/IDEA2.md` § Geometry from a mask / § Stability
- Tasks: T004, T007

### Inputs / outputs

**In:** physical config (fluid `nu_phys`, velocity, characteristic length, target resolution), PNG/SVG path
**Out:** `lbm/units.py::LatticeUnits`, `lbm/geometry.py::from_png`, `::from_svg`

### Acceptance criteria

- [ ] `LatticeUnits.from_physical(...)` returns `dx`, `dt`, `tau`, lattice `U`, and `Re`, with the derivation in the docstring.
- [ ] Round-trip test: physical → lattice → physical reproduces `Re` to within 0.1%.
- [ ] It **raises** with a message naming the offending quantity when the config implies lattice `U >= 0.1` or `tau <= 0.51`, and suggests the resolution that would fix it.
- [ ] `from_png(path, shape)` thresholds alpha (falling back to luminance), resizes to grid, returns a bool mask, and runs `check_mask` automatically.
- [ ] `from_svg(path, shape)` rasterises at least simple closed paths; if a dependency is missing it raises a clear install message rather than failing obscurely.
- [ ] A test PNG committed under `tests/data/` produces a mask with the expected solid-cell count ±2%.
- [ ] Cylinder run reproduced through the physical-units path gives the same `Cd` as T007 to within 2%.
- [ ] Rungs 1–4 still green.

### Constraints that bite here

- Constraint 3 — this is where the `|u| < 0.1` guard actually lives for users. It raises; it does not warn.
- Constraint 2 — `tau` is derived, never set alongside a `nu`. One input path.
- Constraint 12 — `from_png` calls `check_mask` itself; a downscaled PNG is the most likely source of 1-cell-thin walls in the whole project.

### Notes

SVG support may need a new dependency. If it drags, ship PNG and log SVG as a follow-up `/new-task`
rather than burning the session — PNG is what M4 requires.

---

## T010 — Performance pass

**Status:** `not_started`

### Goal

Hit the performance budget with the cheap wins only, and prove correctness survived. **Gated on Rung
3 passing** — see constraint 6.

### Reads / depends on

- `DOCS/IDEA2.md` § Performance budget
- Tasks: T007 (and Rung 3 green)

### Inputs / outputs

**In:** the working solver
**Out:** optimised `lbm/core.py`, a `bench.py` printing steps/s per grid size

### Acceptance criteria

- [ ] Baseline steps/s recorded **before** any change, for 400×100, 800×200, 2000×500, and written into `DOCS/STATE1.md`.
- [ ] Applied, each measured separately: preallocation audit (no allocation in the loop), `float32` end to end, fused collide+stream in one pass over `f`, skip `feq` on solid cells.
- [ ] Post-change numbers meet the budget: **≥400 steps/s at 400×100, ≥120 at 800×200, ≥15 at 1M cells** (budget is ~500/~150/~20; these are the pass floors).
- [ ] `bench.py` prints a before/after table.
- [ ] **All four rungs re-run and still green** — with the same tolerances, not relaxed ones.
- [ ] Restart is still bit-identical (T006's test still passes).
- [ ] No new dependency: no Numba, no Cython, no GPU. Pure NumPy.

### Constraints that bite here

- Constraint 6 — do not start this task if Rung 3 is red.
- Constraint 11 — fusing must not change float ordering in a way that breaks bit-identical restart. If it does, the fusion is reverted, not the test.
- Constraint 1 — optimisation must not quietly change the physics (e.g. skipping `feq` on solid cells must not change fluid-cell results at all: assert bitwise equality against the unoptimised path on a small grid).

### Notes

If a win costs more than ~20 lines of clarity for under 10% speed, drop it. Phase 0's job is
understanding, and M5 replaces this kernel anyway.

---

## T011 — Recording sinks + CLI → M4

**Status:** `not_started`

### Goal

The third and fourth sinks, and one command that takes a PNG plus physical numbers and produces an
MP4. **M4** — the first thing another person can use.

### Reads / depends on

- `DOCS/IDEA2.md` § Three output sinks, same frame source / § Milestones
- Tasks: T009

### Inputs / outputs

**In:** CLI args or a config file (geometry source, fluid, velocity, duration, output path)
**Out:** `lbm/record.py::RecordSink`, `::HeadlessSink`, `python -m lbm.runner` CLI

### Acceptance criteria

- [ ] `RecordSink` writes MP4 via imageio/ffmpeg at a **fixed** framerate and **never drops a frame** — a test writes 50 frames and asserts the file has exactly 50.
- [ ] `HeadlessSink` writes numbered PNGs, no display required.
- [ ] Both consume the same `render()` output as `LiveSink`; a test asserts the three sinks receive byte-identical frames for the same sim state.
- [ ] GIF output works for short clips.
- [ ] `python -m lbm.runner --geometry tests/data/<file>.png --fluid air --velocity 20 --length 1.5 --seconds 5 --out wake.mp4` produces a playable MP4 with a visible vortex street, in one command, from a cold shell.
- [ ] `--live`, `--record`, `--headless` are composable; `--live --record` together works.
- [ ] Missing ffmpeg produces a clear install message, not a traceback.
- [ ] Rungs 1–4 still green. `DOCS/STATE1.md` records M4 as reached with the gate command output.

### Constraints that bite here

- Constraint 10 — one renderer, three sinks. The byte-identical test above is what enforces it.
- Constraint 8 — record must not drop; live may. Different policies, same frame source.
- Phase 0 **ends here.** M5 (Warp/Taichi) is a new plan, not a stretch goal of this task.

### Notes

`imageio[ffmpeg]` needs installing; record it in `DOCS/STATE1.md` § Environment. When this lands,
Phase 0 is closed — the next session should be planning the product layer from root `idea.md`, not
adding solver features.
