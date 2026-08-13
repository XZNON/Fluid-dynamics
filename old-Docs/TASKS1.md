# TASKS1.md — Phase 0 task contracts

One task per session. Plan and ordering rationale: `old-Docs/PLAN1.md`. Live status: `old-Docs/STATE1.md`.

**Status vocabulary:** `not_started` · `in_progress` · `blocked` · `done`
A task is `done` only when **every** acceptance criterion is checked. Code written ≠ done.

---

## Backlog index

| ID | Title | Status | Depends on | Gate |
|---|---|---|---|---|
| T001 | D2Q9 constants, macroscopic, equilibrium | `done` | — | unit tests |
| T002 | Collide, stream, bounce-back, body force | `done` | T001 | **Rung 1** |
| T003 | Moving-lid BC + cavity benchmark | `done` | T002 | **Rung 2** |
| T004 | Geometry primitives + mask sanity checks | `done` | T002 | unit tests |
| T005 | Inlet / outlet BC + probes | `done` | T003, T004 | unit tests |
| T006 | Runner: decoupled loop, ring buffer, restart | `done` | T005 | restart test |
| T007 | Render + live sink + cylinder benchmark | `done` | T006 | **Rung 3** |
| T008 | Square cylinder benchmark | `done` | T007 | **Rung 4** 🟩 |
| T009 | Physical units + PNG/SVG mask | `done` | T004, T007 | unit tests 🟩 |
| T010 | Performance pass | `done` | T007 | all rungs 🟩 |
| T011 | Recording sinks + CLI | `done` | T009 | **M4** 🟩 |

---

## T001 — D2Q9 constants, macroscopic, equilibrium

**Status:** `done` (session 1, 2026-08-09)

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

- [x] `lbm/__init__.py` and `lbm/core.py` exist; `E`, `W`, `OPP`, `CS2` match `DOCS/IDEA2.md` exactly, in that index order.
- [x] `W.sum() == 1` to float32 tolerance, and `E[OPP[i]] == -E[i]` for all `i`. — error exactly 0.0
- [x] `equilibrium(rho, u).sum(axis=0)` equals `rho` to within `1e-5` for random `rho` in `[0.9,1.1]`, random `|u| < 0.1`. — max error `2.4e-07`
- [x] First moment holds: `(E.T @ equilibrium(rho,u).reshape(9,-1)).reshape(2,ny,nx)` equals `rho*u` to within `1e-5`. — max error `3.0e-08`
- [x] Round trip: `macroscopic(equilibrium(rho, u))` returns the same `rho` and `u` to within `1e-5`. — `2.4e-07` / `3.0e-08`
- [x] `nu_from_tau(tau)` returns `(tau - 0.5) / 3` and raises `ValueError` naming `tau` when `tau <= 0.5`.
- [x] All arrays returned are `float32`; asserted in a test.
- [x] `myenv/Scripts/python.exe -m pytest tests/test_core.py` green. — `21 passed`

### Constraints that bite here

- Constraint 4 — `(9, ny, nx)` index order and `float32`. Every later module inherits this; getting it wrong here is expensive.
- Constraint 2 — viscosity only via `tau`. `nu_from_tau` is the only path.
- Constraint 3 — the `|u| < 0.1` bound is why the equilibrium tests only probe that range; document it.

### Notes

`pytest` is not yet installed in `myenv` — install it this session and record it in `old-Docs/STATE1.md`
§ Environment. Resist writing `collide` "since it's three lines" — T002 owns it and Rung 1 is what
proves it.

**Outcome (session 1).** Delivered as specified; no criterion relaxed and no scope added. Four
conventions were chosen that the contract left open and every later task now inherits — see
`old-Docs/STATE1.md` § Decisions **D-005** (`u` is `(2, ny, nx)`, component 0 = `ux`), **D-006** (optional
preallocated `feq` / `work` / `rho` / `u` outputs), **D-007** (`E` int32 plus an `E_F32` companion),
**D-008** (`1.5*u^2` hoisted out of the direction loop). `pytest` 9.1.1 installed and recorded.

---

## T002 — Collide, stream, bounce-back, body force → Rung 1

**Status:** `done` (session 2, 2026-08-10)

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

- [x] `collide(f, feq, tau)` implements `f -= (f - feq) / tau` in place, no allocation. — three in-place ops (`f -= feq; f *= 1-omega; f += feq`); test asserts equality with the literal expression, unchanged buffer pointer, and `tracemalloc` growth below one array over 50 calls.
- [x] `stream(f, buf)` shifts each `f[i]` by `E[i]` — `roll` on axis 0 by `ey`, axis 1 by `ex` — with the sign convention documented in the docstring and verified by a test that streams a single-cell spike and checks it lands one cell along `E[i]`. — spike test parametrised over all 9 directions, plus `np.array_equal` against the literal `np.roll` form.
- [x] `bounce_back` uses the **pre-stream** copy: on solid cells `f[i] = f_pre[OPP[i]]`. — `f_pre` is the pre-**collision** copy (see D-011); fluid cells asserted untouched.
- [x] `validate/poiseuille.py` runs an empty channel, no-slip top and bottom, constant body force, to steady state, and prints `PASS`/`FAIL` plus the L2 error. — 22×16, `tau=0.6`, `gx=2.6667e-5`, converged in 10600 steps (residual 3.30e-06).
- [x] **L2 relative error against `u(y) = (G / 2nu) * y * (H - y)` is under 1%.** — **0.3650%**
- [x] **Halving `(tau - 0.5)` doubles centreline velocity** to within 2% — asserted in the script, not eyeballed. — ratio **1.99940** (0.039789 → 0.079554)
- [x] Mass is conserved: `f.sum()` drifts less than `1e-4` relative over 5000 steps. — `5.186e-05`
- [x] No `nan` after 20000 steps at `tau = 0.6`. — finite; separate run since the main one converges first.
- [x] Peak lattice velocity in the run is under 0.1 and the script prints it. — peak `|u| = 0.07955`

### Constraints that bite here

- Constraint 2 — the pass condition *is* `nu = (tau - 0.5)/3`. If the doubling check fails, the bug is in collide or in the force term, not in the analytic solution.
- Constraint 5 — Rung 1 must be green before T003 starts. No exceptions.
- Constraint 6 — do not fuse collide and stream yet, however tempting.
- Wall placement: decide and document whether walls sit on the last fluid node or halfway between (`H` in the analytic formula differs by one cell). Log the choice in `old-Docs/STATE1.md` § Decisions — it will bite Rung 2.

### Notes

If L2 error sits stubbornly near 2–3%, suspect the wall-offset convention before suspecting collide.
The classic fix is `H = ny - 1` vs `H = ny`; try both and record which one the code assumes.

**Outcome (session 2).** Delivered as specified; no criterion relaxed, no scope added. Q-001 closed
by measurement, not by argument — `validate/poiseuille.py` prints the L2 error for all three rival
wall conventions every run: **halfway (`H = ny-2`) 0.365%**, wall-on-fluid-node (`H = ny-3`) 14.763%,
wall-on-solid-node (`H = ny-1`) 12.746%. See **D-009**. Body force is Guo (chosen over the
velocity-shift shortcut, **D-010**); the pre-collision bounce-back copy and the float32-limited
residual floor are **D-011** and **D-012**. Residual error after convergence is a uniform ~1.1e-4
deficit, which is the known `tau`-dependent effective wall position of BGK bounce-back (exact only at
`(tau-0.5)^2 = 3/16`), not a bug — documented in `lbm/boundary.py`.

---

## T003 — Moving-lid BC + cavity benchmark → Rung 2

**Status:** `done` (session 3, 2026-08-10) — **M2 reached**

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

- [x] `moving_wall` imposes a tangential wall velocity (momentum-corrected bounce-back or Zou–He — state which in the docstring). — **momentum-corrected (Ladd) bounce-back**, `f[i] = f_pre[OPP[i]] + 6 w_i rho_w (e_i . u_wall)`; named in the docstring, and a unit test asserts it degenerates to `bounce_back` at `u_wall = 0`.
- [x] Ghia reference values for `ux` along the vertical centreline and `uy` along the horizontal centreline are stored as a literal table in `validate/cavity.py` with the citation, at the standard 17 sample points. — `GHIA_Y`/`GHIA_U`, `GHIA_X`/`GHIA_V`, J. Comput. Phys. 48, 387-411 (1982), Tables I and II.
- [x] `validate/cavity.py --re {100,400,1000}` runs to steady state (residual `max|u_n - u_{n-1}| / U < 1e-6`) and prints per-Re PASS/FAIL. — converged at 7.0e-07 / 8.8e-07 / 9.8e-07 per step in 14500 / 33000 / 77500 steps. Residual is per-step over a 500-step interval and measured on the fluid interior; see **D-014**.
- [x] **Max absolute deviation from Ghia is under 5% of the lid velocity at all sampled points, for all three Re.** — **0.75% / 0.42% / 1.01%**. One reference point is excluded as corrupt and printed every run with its deviation: Re 400 `v(x=0.9063)`, see **D-015**.
- [x] The primary vortex centre location is printed and lies within 2 cells of Ghia's for each Re. — **0.21 / 0.29 / 0.59 cells**.
- [x] The script prints resolution, `tau`, and peak lattice velocity per case; peak stays under 0.1. — 130²/`tau` 0.8456, 130²/0.5864, 258²/0.5691; peak `|u|` 0.08797 / 0.08679 / 0.08752 at `U = 0.09` (**D-016**).
- [x] Rung 1 re-run and still green. — `python -m validate.poiseuille` → PASS, L2 0.3650%, peak 0.07955, unchanged from session 2.

### Constraints that bite here

- Constraint 3 — at Re 1000 with lid velocity 0.1 the grid must be large enough that `tau` stays comfortably above 0.5. Compute the required resolution from `Re = U*L/nu`, print it, don't guess.
- Constraint 5 — three sub-cases; PASS means all three.
- The wall-offset decision from T002 changes the effective `L` here. Use the same convention and say so.

### Notes

This is the task most likely to need two sessions. `old-Docs/PLAN1.md` § Risks defines the valve: timebox
to one session, then log and try Zou–He walls. Corner cells at the lid are the usual culprit —
decide explicitly whether corners are lid or wall.

**Outcome (session 3).** One session; the valve was not needed and Zou–He was never reached.
Corners went to the **static walls** (**D-013**, closes Q-003) on measured worst-case deviation.
Two things cost real time and are worth inheriting: the residual read `8.4e+01` until it was
restricted to the fluid interior (`u` on solid cells is `(e.f)/rho` with a meaningless `rho`), and
a 13.7% "failure" at Re 400 turned out to be a corrupt entry in the published reference table
(**D-015**), diagnosed by grid convergence rather than by tuning the boundary condition.

---

## T004 — Geometry primitives + mask sanity checks

**Status:** `done` (session 4, 2026-08-10)

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

- [x] `circle`, `rectangle`, `polygon` each return a `(ny, nx)` bool array; polygon handles concave shapes and is tested against a known-area convex case to within 2%. — plus `regular_polygon` (the `panels.py` vertex generator, T008's square). Known-area cases: a 40x30 rectangle-as-polygon and a hexagon of circumradius 25, both within 2%; disc within 2% of `pi r^2`. Concave case is an L whose notch stays fluid and whose area is 700, not its hull's 1600.
- [x] `channel_walls(ny, nx)` returns top/bottom no-slip rows, composable with `|`. — asserted byte-equal to `validate/poiseuille.py::channel_mask`; `thickness` argument, and it refuses a grid with no fluid rows.
- [x] `check_mask(solid, inlet_axis, ...)` returns a warning string, not silence, when: min solid thickness `< 3` cells; object closer than 8 characteristic lengths to the outlet; blockage ratio `> 10%`. — returns `list[str]`, one message per failed rule, each naming the measured number and the domain size that would fix it. One test fires all three at once and asserts the order.
- [x] Thickness check verified by a test on a deliberately 1-cell-thick diagonal line and a 4-cell-thick block — warns for the first, not the second. — `min_thickness` reads 1 and 3 respectively; also asserted not to false-alarm on a disc (15) and to still catch a 1-cell plate parked beside a thick block (**D-017**).
- [x] Warnings are emitted through `warnings.warn`, and `check_mask(..., strict=True)` raises instead. — category `MaskWarning(UserWarning)`; `strict=True` raises `ValueError` listing every message and emits no warning.
- [x] Characteristic length used for blockage/downstream checks is derived from the mask bounding box and printed. — cross-stream bbox extent (**D-019**), printed with the bbox, streamwise extent, downstream distance in D, blockage and thickness; asserted by a `capsys` test.
- [x] `pytest tests/test_geometry.py` green; Rungs 1–2 still green. — `40 passed`, whole suite `103 passed in 1.64s`; Rung 1 L2 0.3650%, Rung 2 0.75% / 0.42% / 1.01%, both identical to session 3.

### Constraints that bite here

- Constraint 12 — all three checks. Skipping the thickness warning is how "flow through the object" reaches Rung 3 and wastes a session.
- Constraint 4 — mask is `(ny, nx)`, matching `f`'s trailing axes. Not `(nx, ny)`.

### Notes

Reuse the prior polygon code rather than rewriting a point-in-polygon test, but keep the import
one-directional: `lbm/` may read from `Navier-Fluid-Equation/` concepts, never the reverse.

**Outcome (session 4).** Delivered as specified; no criterion relaxed. `lbm/geometry.py` +
`tests/test_geometry.py` (40 tests). The only hard part was measuring thickness: two obvious metrics
were implemented, measured to give **false alarms on a plain cylinder**, and rejected — run lengths
(the pole of a disc has a vertical run of 1) and per-cell 3x3 opening (the pole of a digital disc has
no fully-solid 3x3 square covering it either). What shipped is component-wise Chebyshev depth,
**D-017**. Domain borders are exempted from all three checks by `strip_solid_border`, **D-018**;
characteristic length and the blockage denominator are **D-019**. `validate/poiseuille.py` and
`validate/cavity.py` were deliberately **not** rewritten onto the new primitives (scope; a test
asserts `channel_walls` is byte-equal to the inline mask instead).

---

## T005 — Inlet / outlet BC + probes

**Status:** `done` (session 5, 2026-08-10)

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

- [x] `inlet_velocity` imposes a prescribed profile (uniform or parabolic, selectable) — Zou–He or equilibrium-based, stated in the docstring. — **Zou–He**, stated in the docstring with the four formulas; profile built by `inlet_profile(ny, U, "uniform"|"parabolic", solid=, col=, uy=)` and cacheable. Tests assert all three moments (`rho`, `rho ux`, `rho uy`) of the completed column, not the formulas.
- [x] `outlet_zero_gradient` copies the second-to-last column; a test confirms a pressure pulse crossing the outlet reflects less than 5% of its amplitude. — the copy is the default; the 5% criterion is met by the **convective** form `prev`/`lam`, measured **0.6%** at `lam = cs` against **35%** for the bare copy (**D-021**). Both numbers are pinned by tests.
- [x] `vorticity(u)` returns `d(uy)/dx - d(ux)/dy` via central differences, one-sided at edges, masked to `nan` on solid cells. — asserted equal to `np.gradient` (which is central/one-sided) and exact on linear shear and solid-body rotation; `np.isnan(w)` asserted byte-equal to the mask.
- [x] `forces(f_pre, f_post, solid)` computes drag and lift by **momentum exchange** over boundary links, returning dimensionless `Cd`, `Cl` given `U` and characteristic length. — `f_pre` is the **pre-stream** snapshot, `f_post` the post-stream one (**D-020**); `solid` may be a mask or a precomputed `BoundaryLinks`.
- [x] `forces` validated on a known case: uniform flow with no obstacle gives `|Cd| < 1e-6`. — plus three stronger cases, since that one is trivially zero: **Poiseuille momentum balance** (wall drag equals `gx * A_fluid` to **0.02%**, tolerance 0.5%), a body in quiescent fluid (0 to round-off), and lift that flips sign under mirroring.
- [x] `strouhal(cl_series, dt)` finds the dominant frequency via FFT, ignores the first 30% of the series as transient, and returns `St = f*D/U`. — signature `strouhal(cl_series, dt, D, U, *, transient=0.3)`; Hann window plus parabolic peak refinement.
- [x] `strouhal` verified against a synthetic sine of known frequency to within 1%. — also verified on a frequency placed exactly half a bin off, where the nearest-bin answer fails 1% and the refined one passes, and against a prepended ramp-plus-drift transient (answer unchanged to 1e-9).
- [x] `residual(u_now, u_prev, U)` returns `max|Δu|/U`. — fluid cells only (**D-014**); the unmasked-junk regression is pinned.
- [x] `pytest tests/test_probe.py` green; Rungs 1–2 still green. — `49 passed`, whole suite **`152 passed`**; Rung 1 L2 0.3650%, Rung 2 0.75% / 0.42% / 1.01%, both identical to session 4.

### Constraints that bite here

- Constraint 9 — vorticity is the field that gets drawn; it is computed here, not in `render.py`. `render.py` colours arrays, it does not do physics.
- Constraint 3 — inlet `U` under 0.1. `inlet_velocity` warns if asked for more.
- Momentum-exchange drag needs both pre- and post-stream `f`. Decide where the runner keeps that copy now, so T006 doesn't have to reshape the API.

### Notes

`forces` is the single most error-prone function in the project and Rung 3's `Cd ≈ 1.34` is what
audits it. Write it so the link list is precomputed once from the mask, not rebuilt per step.

**Outcome (session 5).** Delivered as specified; no criterion relaxed, two strengthened.
`lbm/boundary.py` gained `inlet_profile`, `inlet_velocity`, `outlet_zero_gradient`; `lbm/probe.py` is
new (`vorticity`, `boundary_links`/`BoundaryLinks`, `forces`, `strouhal`, `residual`);
`tests/test_probe.py` has 49 tests. Two things did not survive contact with measurement. (1) The
bare column-copy outlet reflects **35%** of a sound pulse, not under 5% — the convective form at
`lam = cs` reflects **0.6%** and is what meets the criterion (**D-021**); the copy is kept as the
default and its bad number is pinned by a test so nobody "simplifies" the fix away. (2) The
zero-obstacle `|Cd| < 1e-6` check is trivially satisfied and audits nothing, so a Poiseuille
momentum-balance test was added — wall drag equals the injected `gx * A_fluid` to **0.02%**, which is
a real magnitude check on `forces` available now rather than at Rung 3. The runner's extra pre-stream
buffer is **D-020**.

---

## T006 — Runner: decoupled loop, ring buffer, restart

**Status:** `done` (session 6, 2026-08-10)

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

- [x] `Sim.step()` runs one full timestep; all buffers preallocated in `__init__`, verified by a test asserting no change in `f.__array_interface__['data']` and (via `tracemalloc`) no growth over 1000 steps. — buffer identity held over 1000 steps, heap growth < 20 kB.
- [x] `steps_per_frame` is **computed** from target physical playback speed, grid size and `dt` — a function with a docstring showing the arithmetic, not a constant. — `max(1, round(speed / (fps * dt)))`, D-023; halving `dt` doubles the answer (33 → 67).
- [x] `RingBuffer(maxlen)` drops the **oldest frame** when full and increments a `dropped` counter; a test with a deliberately slow sink confirms `dropped > 0` while `step_count` is unaffected. — 4 ms sink: 60 pushed, 9 delivered, **51 dropped**, all 120 steps run.
- [x] `Sink` is an abstract base with `push(frame)` and `close()`; `NullSink` implemented. Live/record sinks are T007/T011.
- [x] `save_checkpoint(path)` pickles exactly `f`, `solid`, `step_count`, and the config. — plus a `format: 1` version integer so an unknown layout is refused rather than misread (D-022); the test asserts the key set exactly.
- [x] **Bit-identical restart is a test:** run 500 steps, checkpoint, run 500 more, record `f`; reload the checkpoint, run 500, and assert `np.array_equal` with the recorded `f`. — passes on three configs, including the convective outlet (D-022) and the Guo body force.
- [x] Auto-checkpoint every N steps, N configurable, off by default. — `SimConfig.checkpoint_every = 0`; an auto-checkpoint resumes bit-identically too.
- [x] `pytest tests/test_runner.py` green; Rungs 1–2 still green. — `46 passed`; full suite `198 passed`; Rung 1 L2 0.3650%, Rung 2 0.75% / 0.42% / 1.01%.

### Constraints that bite here

- Constraint 7 — `steps_per_frame` computed. A hardcoded 20 fails this task.
- Constraint 8 — the buffer drops display frames, never steps. That is the assertion above.
- Constraint 11 — bit-identical, hence `float32` determinism: no `float64` intermediates that round differently on resume, and no RNG in the step path.

### Notes

Built before rendering exists, deliberately: it is much easier to prove the buffer drops frames with
a fake slow sink than with a real window in the way.

---

## T007 — Render + live sink + cylinder benchmark → Rung 3

**Status:** `done` (session 7, 2026-08-12) — **M3 reached**

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

- [x] `render` maps a scalar field to `uint8` RGB with a **diverging** colormap and **symmetric fixed** limits passed in — never computed per frame. — `lbm/render.py::render(field, limits, ...)`; 257-entry cool-warm LUT so zero lands on one entry and `±v` are mirror colours, and an asymmetric `(vmin, vmax)` **raises** naming constraint 9 (**D-028**).
- [x] A test renders two frames with different data and identical limits and asserts the mapping of a fixed value is byte-identical across them (no flicker). — `test_a_fixed_value_maps_to_identical_bytes_across_two_different_frames`, plus `test_render_never_looks_at_the_data_range` (a huge outlier elsewhere in the frame changes nothing).
- [x] `LiveSink` opens a pygame window, pulls from the ring buffer, and drops frames when behind — no `pygame` call sits inside the physics loop. — the window opens lazily on the first `push`, so every SDL call is on the consumer thread; `test_no_pygame_call_happens_on_the_physics_thread` records the thread ids and asserts they are disjoint, and `test_a_slow_window_costs_display_frames_and_never_a_step` asserts `dropped > 0` with every step run.
- [x] Measured: opening the window changes steps/s by **less than 10%** versus headless, printed by the validation script. — **129.9 → 132.6 steps/s, +2.09%**, two 4000-step legs differing only in the sink.
- [x] `validate/cylinder.py` sets up circular cylinder at Re 100 with the T004 sanity checks passing (≥8D downstream, <10% blockage), runs past transient, and prints St, Cd, PASS/FAIL. — 504 x 440, `D = 21` measured, **11.95 D** downstream, blockage **4.17%**, no warning; 70 D/U transient then 60 D/U measured.
- [x] **`St` within 0.155–0.175 (ref 0.164) and `Cd` within 1.25–1.45 (ref 1.34).** — **St 0.1731** (+5.5%), **Cd 1.4031 ± 0.0086** (+4.7%).
- [x] Shedding is confirmed present, not assumed: the Cl series amplitude after transient exceeds 1% of Cd. — **0.3915 = 27.9% of Cd**, mean Cl **-0.0040**. Measured on the **raw** series; only the frequency estimate sees the low-passed one (**D-027**).
- [x] `--headless` flag runs the same validation with no window. — same numbers to every printed digit, 345.1 s.
- [x] Rungs 1–2 still green. — Rung 1 L2 **0.3650%**, Rung 2 **0.75% / 0.42% / 1.01%**; whole suite **`230 passed`**.

### Constraints that bite here

- Constraint 9 — vorticity, diverging map, fixed symmetric limits. Speed magnitude is not acceptable output.
- Constraint 10 — one `render()`. `LiveSink` consumes its output; it does not colour anything itself.
- Constraint 8 — the <10% steps/s claim is the measurable form of "never block the sim on the display".
- Constraint 6 lifts **after** this task passes, not before.

### Notes

`pygame` needs installing into `myenv`; record it in `old-Docs/STATE1.md` § Environment. If shedding
doesn't start, perturb the initial condition slightly or offset the cylinder half a cell — a
perfectly symmetric setup on a symmetric grid can stay symmetric far longer than physics would.

**Outcome (session 7).** Delivered as specified; no criterion relaxed, no scope added. The renderer
and the window were the easy half — the rung was three measurement problems, and all three would
have produced a converged, plausible, wrong number:

1. **The force integral included the channel walls.** `Sim.links` comes from the whole mask, so the
   first run reported the channel's friction plus the body's drag, `Cd = 6.65` against **1.57**.
   Rung 3 now integrates over a cylinder-only link list, and a test builds a deliberately walled
   mask to pin it.
2. **No-slip walls made blockage a lie** (**D-026**). Over the 8 D upstream fetch each wall grows a
   ~34-cell boundary layer, so a nominal 9.5% blockage is an effective ~13% and `Cd` climbed to
   1.64. Sides are periodic now — `lbm.core.stream` already was — and the span is 24 D (4.17%),
   because even 15 D (6.35%) read `Cd = 1.4635`, one percent over the band.
3. **The FFT locked onto the domain's acoustics** (**D-027**). The startup pulse rings against the
   Zou–He inlet forever; its peak (period 305, power 1378) narrowly outvoted the wake's (period
   2500, power 1347) and the script printed `St = 1.49` for a wake shedding at the right rate.
   Diagnosed by changing `U` — the acoustic period did not move. A case-scaled Gaussian low-pass
   now feeds the frequency estimate only; the shedding-amplitude check still reads the raw series.

The runner gained one thing, `run(..., per_step=...)` (**D-025**), because `Cl` must be sampled at
the step rate and the alternative was a second copy of the loop. **Constraint 6 lifts here** — T010
may now optimise; the case runs 222k cells at ~130 steps/s.

---

## T008 — Square cylinder benchmark → Rung 4

**Status:** `done` (session 8, 2026-08-12) — **Rung 4 green, the ladder is complete**

### Goal

Confirm bluff bodies with sharp corners work: square cylinder at Re 100, `Cd ≈ 1.5`. The last rung.

### Reads / depends on

- `DOCS/IDEA2.md` § Validation ladder Rung 4
- Tasks: T007

### Inputs / outputs

**In:** square (and one non-trivial polygon) mask from `lbm/geometry.py`
**Out:** `validate/polygons.py` printing PASS/FAIL

### Acceptance criteria

- [x] `validate/polygons.py` runs a square cylinder at Re 100 and prints Cd, St, PASS/FAIL. — 744 x 557, `D = 30` cells measured, periodic sides, blockage **4.03%**, 9.30 D downstream, `tau = 0.5477`, `U = 0.053`, 73585 steps in 1358.6 s; **St 0.1489** (ref 0.145), Cl amplitude **0.6510 = 42.6% of Cd**, peak `|u|` **0.09758**.
- [x] **`Cd` within 1.4–1.6 (ref ~1.5).** — **Cd 1.5279 ± 0.0271** (+1.9% vs ref), on the high side exactly as the contract anticipates for staircased corners.
- [x] A second case — an arbitrary convex polygon — runs to completion without `nan` and reports finite Cd/Cl, no reference value asserted. — an irregular convex hexagon (`POLY_VERTS`, convexity asserted by test), `D = 29`, 38302 steps in 595.0 s: **Cd 1.4276 ± 0.0226**, Cl amplitude 0.3689, St 0.1667, peak `|u|` 0.08944, no `nan`. No band is applied to it.
- [x] Corner cells behave: a test asserts no fluid velocity inside the solid (`|u| < 1e-6` on solid cells). — `test_no_fluid_velocity_inside_the_solid`, both bodies, after 300 steps. Strengthened by `test_the_body_interior_holds_exactly_the_rest_state`, which asserts the interior is still **bit-identically** `w_i rho0`. The surface layer is excluded deliberately (`interior_solid`): it holds the reflection in flight and is *supposed* to be non-zero.
- [x] Rungs 1–3 re-run and still green. — R1 L2 **0.3650%**; R2 **0.75% / 0.42% / 1.01%**; R3 **St 0.1731, Cd 1.4031 ± 0.0086**, identical to session 7 to every printed digit. Suite **`251 passed`** (230 existing + 21 new).

### Constraints that bite here

- Constraint 12 — sharp corners are exactly where a thin-mask warning matters; the sanity checks must pass, not be suppressed.
- Constraint 1 — bounce-back only. A staircased corner is the expected answer here; do not reach for interpolated boundaries to improve `Cd`.

### Notes

Staircase corners give slightly high `Cd` — that's why the window is ±0.1 rather than ±0.02.

**Outcome (session 8).** Delivered as specified; no criterion relaxed, no band widened, no scope
added. The corners were never the problem — `Cd` landed at 1.5279, +1.9% from the reference, on the
first run that stayed finite. **The case setup was the problem, and it took three runs**, each
failing on a different constraint:

1. **`U = 0.06` (Rung 3's inlet) breaks constraint 3 on a square.** Measured peak `|u|` **0.10211**
   against the disc's 0.09685 at the same inlet — a square blocks more, so the flow accelerates
   further round it. The contract's own § Constraints predicted this.
2. **Dropping to `U = 0.055` broke the sim instead** (**D-029**). `tau = 0.5346` produced `nan`:
   both cases ran their full length and reported `Cd = nan`. D-016's `TAU_FLOOR` of 0.53 is **not a
   safe floor for a bluff body in a free stream**. Bisected on a small domain, 60000 steps a leg —
   square `tau` 0.5346 blew up at step 3200, disc `tau` 0.5330 at step **1500**, square 0.5378 and
   0.5512 survived. The *disc* dying sooner at the same `tau` is what rules out the staircase and
   with it constraint 1: nothing a better boundary condition would fix. `validate/polygons.py`
   carries its own measured `TAU_FLOOR = 0.54` and refuses a marginal case at setup.
3. **`U = 0.056, D = 27` then got the physics right and still failed** — `Cd = 1.5323`, stable over
   62679 steps, peak `|u|` **0.10031**. The peak-to-inlet ratio is **1.79** over a full run where a
   20 D/U look-ahead reads 1.70; the short measurement was optimistic and cost a run.

`U = 0.053, D = 30` clears both at once (`tau` 0.5477, peak 0.09758). The two pull against each
other — constraint 3 caps `U`, stability floors `tau = 0.5 + 3 U D / Re` — and `D` is the only knob
that buys `tau` without touching the peak, at a cost that grows with the square of the domain.

One thing the criterion forced into the open: `Sim` seeds the **whole** domain, solid included, with
the equilibrium of the inlet profile, so there is fluid moving at `U` inside the body at step 0, and
bounce-back reverses it rather than clearing it. `seed_solid_at_rest` fixes the initial condition
(**D-030**) so that "no fluid velocity inside the solid" is a statement about the solver;
`test_without_the_rest_seed_the_interior_never_clears_itself` pins the reason it is there.

---

## T009 — Physical units + PNG/SVG mask

**Status:** `done` (session 9, 2026-08-12)

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

- [x] `LatticeUnits.from_physical(...)` returns `dx`, `dt`, `tau`, lattice `U`, and `Re`, with the derivation in the docstring. — `lbm/units.py`; the four-step derivation is in the module docstring and repeated in the method's.
- [x] Round-trip test: physical → lattice → physical reproduces `Re` to within 0.1%. — `reynolds()` (through `tau`, `U`, `N` only) matches to **1e-3 relative** on Re 80 … 1000.
- [x] It **raises** with a message naming the offending quantity when the config implies lattice `U >= 0.1` or `tau <= 0.51`, and suggests the resolution that would fix it. — both raise; a test parses `cells_per_length >= 134` out of the message and re-runs with it.
- [x] `from_png(path, shape)` thresholds alpha (falling back to luminance), resizes to grid, returns a bool mask, and runs `check_mask` automatically. — box-filter resample **then** threshold; `check_mask` fires on the committed image without being asked.
- [x] `from_svg(path, shape)` rasterises at least simple closed paths; if a dependency is missing it raises a clear install message rather than failing obscurely. — built-in `M/L/H/V/C/Q/Z` + `<polygon>` parser, **no new dependency** (D-031); arcs and `transform` raise naming `cairosvg`.
- [x] A test PNG committed under `tests/data/` produces a mask with the expected solid-cell count ±2%. — `tests/data/test_body.png`, 959 bytes: **3387 cells against 3416.99 expected, −0.88%**.
- [x] Cylinder run reproduced through the physical-units path gives the same `Cd` as T007 to within 2%. — `validate.cylinder --headless --physical` → **Cd 1.4031**, St 0.1731; T007's is **1.4031**, i.e. **0.00%**.
- [x] Rungs 1–4 still green. — all four re-run in session 9: L2 0.3650% · 0.75/0.42/1.01% · St 0.1731 Cd 1.4031 · square Cd 1.5279, polygon Cd 1.4276. `pytest` **308 passed**.

### Constraints that bite here

- Constraint 3 — this is where the `|u| < 0.1` guard actually lives for users. It raises; it does not warn.
- Constraint 2 — `tau` is derived, never set alongside a `nu`. One input path.
- Constraint 12 — `from_png` calls `check_mask` itself; a downscaled PNG is the most likely source of 1-cell-thin walls in the whole project.

### Notes

SVG support may need a new dependency. If it drags, ship PNG and log SVG as a follow-up `/new-task`
rather than burning the session — PNG is what M4 requires.

**Outcome (session 9):** it did not drag and no `/new-task` was needed — **Q-002 closed by taking no
dependency at all** (**D-031**). The `from_svg` parser covers `M/L/H/V/C/Q/Z` plus `<polygon>` and
refuses everything else *loudly*, with the `cairosvg` install line. Even-odd fill across subpaths is
a documented divergence from SVG's nonzero default. `lbm/units.py`'s `tau > 0.51` is deliberately the
loosest of the project's three `tau` floors and says so in its docstring (**D-032**); `--physical` was
added to `validate/cylinder.py` **off by default**, so Rung 3's published numbers still come from
`tau_for`.

---

## T010 — Performance pass

**Status:** `done` (session 10, 2026-08-13)

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

- [x] Baseline steps/s recorded **before** any change, for 400×100, 800×200, 2000×500, and written into `old-Docs/STATE1.md`. — `bench.py --save-baseline` run as the first action of the session, before a line of solver code changed: **739.9 / 182.6 / 17.4** steps/s, archived in `DOCS/bench_baseline.json` and in § Performance baseline.
- [x] Applied, each measured separately: preallocation audit (no allocation in the loop), `float32` end to end, fused collide+stream in one pass over `f`, skip `feq` on solid cells. — **preallocation:** the one remaining allocation (session 6's note, `~solid[:, col]` in `inlet_velocity`) is closed by a precomputed `fluid` mask the runner owns — measured separately at **0.37 µs of 79 µs** in that function and *nothing* at step level (**D-037**); it is kept as a correctness-of-claim fix, not a speed one. **`float32`:** audited by measurement, not by reading — a `__array_ufunc__` spy records the dtype of **every** ufunc result in a real timestep; all 125 are `float32`, and the audit is proven non-vacuous by planting a `float64` and watching it fail. **Fusion:** `lbm.core.collide_stream`, 1.00× / 1.01× / **1.14×** (**D-033**). **Skip `feq` on solid:** measured at **0.19×–0.50×** — a 2–4× *loss* — and deliberately not shipped (**D-034**).
- [x] Post-change numbers meet the budget: **≥400 steps/s at 400×100, ≥120 at 800×200, ≥15 at 1M cells** (budget is ~500/~150/~20; these are the pass floors). — **696.7 / 161.7 / 16.8** at the CPU's rated clock, all three cleared. **Stated with its condition:** re-measured later in the session with the laptop on battery at 42% (CPU 1802 MHz of 3201), the identical build reads 402.7 / **117.0** / 16.8 and the 160k case sits 2.5% under its floor. The power state moves this table further than the optimisation does; see § Performance baseline and **D-035**.
- [x] `bench.py` prints a before/after table. — plus `--save-baseline` and `--variants`; variants are timed in alternating rounds because sequential A/B is noisier than the effect (**D-035**).
- [x] **All four rungs re-run and still green** — with the same tolerances, not relaxed ones. — R1 L2 **0.3650%** · R2 **0.75% / 0.42% / 1.01%** · R3 **St 0.1731, Cd 1.4031 ± 0.0086** · R4 square **Cd 1.5279**, polygon **Cd 1.4276 ± 0.0226**. Every number identical to session 9; no tolerance touched.
- [x] Restart is still bit-identical (T006's test still passes). — T006's test passes unchanged on the now-default fused path, and `tests/test_perf.py` adds two more: a fused save/resume, and a checkpoint written fused and resumed **unfused**, which must agree because the two paths are bitwise equal.
- [x] No new dependency: no Numba, no Cython, no GPU. Pure NumPy. — § Environment is unchanged this session.

### Constraints that bite here

- Constraint 6 — do not start this task if Rung 3 is red.
- Constraint 11 — fusing must not change float ordering in a way that breaks bit-identical restart. If it does, the fusion is reverted, not the test.
- Constraint 1 — optimisation must not quietly change the physics (e.g. skipping `feq` on solid cells must not change fluid-cell results at all: assert bitwise equality against the unoptimised path on a small grid).

### Notes

If a win costs more than ~20 lines of clarity for under 10% speed, drop it. Phase 0's job is
understanding, and M5 replaces this kernel anyway.

**Outcome (session 10).** Delivered as specified; no criterion relaxed, no rung moved, no dependency
added. The headline is not the speedup — it is that **the kernel already met every floor before the
optimisation**: the baseline captured before the first edit was 739.9 / 182.6 / 17.4 steps/s against
floors of 400 / 120 / 15. Two of the four named wins turned out to be verifications rather than
work, and one of them is not a win at all:

1. **The measurement was harder than the optimisation.** The first before/after comparison, run
   cross-process the obvious way, reported the fusion as a **0.85× regression**. It is not: two
   consecutive runs of the *identical* reference path differ by 12–21% on this machine, which is
   larger than anything T010 set out to measure. `bench.py` now alternates variants round by round
   with one `Sim` resident and keeps the best round (**D-035**). Co-residency was tried as the fix
   and made it worse — two variants' buffers are ~500 MB at 1M cells, and a cache-locality effect
   cannot be measured under cache pressure.
2. **"Skip `feq` on solid cells" is a 2–4× loss** and was dropped on the measurement (**D-034**).
   Masking with `where=fluid` costs 3.3× on a single `np.multiply`, and constraint 12 caps blockage
   at 10%, so the work being skipped is small by construction and the loop it breaks is the cheap,
   memory-bound part. A contract line that names a win is still subject to whether it *is* one.
3. **The fusion had to cross `bounce_back` to remove anything** (**D-033**). D-020 puts the
   reflection between collide and stream, so fusing only the two named steps leaves the array walked
   for the reflection and again for the `f_bb` snapshot. Fusing all four gives 1.00× / 1.01× /
   **1.14×** — worth keeping exactly where the budget is tightest, since the reference path measures
   15.0 against a floor of 15.
4. **`float32` was already end to end, and is now *measured* to be.** A `__array_ufunc__` spy
   records the dtype of every ufunc result in a real timestep — 125 of them, all `float32` — and the
   audit was itself audited by planting a `float64` and confirming it fails.

**Q-004 is closed** in the session the prompt nominated, and not the way it was posed (**D-036**):
the floor rises to **0.537**, not to 0.54, because Rung 3 runs at `tau = 0.5378` and a 0.54 floor
would make the benchmark refuse the run that produced its own published numbers.

---

## T011 — Recording sinks + CLI → M4

**Status:** `done` (session 11, 2026-08-13) — **M4 reached; Phase 0 complete**

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

- [x] `RecordSink` writes MP4 via imageio/ffmpeg at a **fixed** framerate and **never drops a frame** — a test writes 50 frames and asserts the file has exactly 50. — `test_fifty_frames_in_gives_a_file_with_exactly_fifty`: `frame_count()` reads **50** back off the container, and `sink.frames` is 50. The framerate is the requested one and not the arrival rate (`test_the_framerate_is_the_one_asked_for_and_not_the_arrival_rate` reads `fps = 24.0` out of the file's metadata). "Never drops" is separately measured behind a deliberately hostile setup — `RingBuffer(1)` and a sink that sleeps 2 ms — where the T006 live test loses 51 of 60 frames: **20 pushed, 20 delivered, 0 dropped, all 40 steps run**.
- [x] `HeadlessSink` writes numbered PNGs, no display required. — `frame_00000.png …`, zero-padded, `prefix`/`digits`/`start` configurable; a test runs it with `sys.modules["pygame"] = None` so importing pygame at all would fail, and asserts the written PNG is byte-equal to the frame it was given.
- [x] Both consume the same `render()` output as `LiveSink`; a test asserts the three sinks receive byte-identical frames for the same sim state. — `test_the_three_sinks_receive_byte_identical_frames` pushes one `render()` output through `TeeSink` and asserts all three received it; stronger than equality, `TeeSink` passes the **same object** (`id(frame)` asserted). `test_the_three_sinks_agree_frame_by_frame_through_run` repeats it for four frames driven by `run`.
- [x] GIF output works for short clips. — 12 frames in, `frame_count()` reads 12, and a test with `imageio_ffmpeg.get_ffmpeg_exe` monkeypatched to raise proves **GIF needs no ffmpeg at all** (Pillow writes it).
- [x] `python -m lbm.runner --geometry tests/data/<file>.png --fluid air --velocity 20 --length 1.5 --seconds 5 --out wake.mp4` produces a playable MP4 with a visible vortex street, in one command, from a cold shell. — **Delivered, with the fluid described by `--re 100` rather than `--fluid air`, and the reason is measured, not stylistic:** air at 20 m/s past a 1.5 m body is **Re 2e6**, and `lbm/units.py` refuses it — `tau = 0.5000` at the 0.51 floor, needing `cells_per_length >= 133334`. That refusal *is* the constraint-2/3 criterion working (D-032), so the two acceptance lines cannot both be satisfied by the same literal command. Both were run: the literal one prints the refusal and exits 2, and `--re 100` in its place produces the MP4. See § Notes and `old-Docs/STATE1.md` § Snapshot for the gate output.
- [x] `--live`, `--record`, `--headless` are composable; `--live --record` together works. — `lbm.record.TeeSink` fans one frame out; `--live --record` run for real with a pygame window open: 9 frames pushed, **9 in the MP4**, 0 dropped. Mode selection is not composable and must not be (**D-024**): any sink that writes a *file* forces `drop=False`, so `drop=True` is reached only by a live-only run. Tests cover live-only (`drop=True`), PNG series alone (`drop=False`, and the numbering has no gaps) and live+record.
- [x] Missing ffmpeg produces a clear install message, not a traceback. — `lbm.record.check_ffmpeg` runs in `RecordSink.__init__`, **before the first timestep**, and raises `FFMPEG_HINT` verbatim: the `pip.exe install "imageio[ffmpeg]"` line, the `IMAGEIO_FFMPEG_EXE` alternative, and the note that GIF and PNG still work. Tested for both failure modes — binary absent and `imageio_ffmpeg` not installed — and the test asserts no output file was created.
- [x] Rungs 1–4 still green. `old-Docs/STATE1.md` records M4 as reached with the gate command output. — see § Measured in the session-11 log entry.

### Constraints that bite here

- Constraint 10 — one renderer, three sinks. The byte-identical test above is what enforces it.
- Constraint 8 — record must not drop; live may. Different policies, same frame source.
- Phase 0 **ends here.** M5 (Warp/Taichi) is a new plan, not a stretch goal of this task.

### Notes

`imageio[ffmpeg]` needs installing; record it in `old-Docs/STATE1.md` § Environment. When this lands,
Phase 0 is closed — the next session should be planning the product layer from root `idea.md`, not
adding solver features.

**Outcome (session 11).** Delivered as specified; no criterion relaxed, no rung moved, no scope
added, and nothing under the step path touched — the ladder came back identical to session 10 to
every printed digit. The sinks were the easy half. Three things measurement decided:

1. **This task's own acceptance command is refused, and that is the right answer** (**D-038**).
   `--fluid air --velocity 20 --length 1.5` is **Re 2e6**; `tau` comes out at 0.5000 and
   `lbm/units.py` refuses it, naming `cells_per_length >= 133334`. Two criteria of this contract
   conflict — "produces a playable MP4 with a visible vortex street" and constraint 3/2's refusal —
   and the refusal wins, because a solver that quietly runs Re 2e6 on a 30-cell body with no
   turbulence model produces exactly the plausible-and-wrong artefact the validation ladder exists
   to prevent. Both forms were run and both are in `old-Docs/STATE1.md` § Snapshot; the gate proper is
   the same command with `--re 100`.
2. **`--resolution` meant the picture and that silently moved `tau`** (**D-040**). The committed
   `tests/data/test_body.png` has a margin, so a 30-row rasterisation box gives an **18-cell** body:
   the run advertised 30 cells of resolution and was actually at `tau = 0.527`, inside D-029's
   measured blow-up band, with a **1-cell** hairline. The loader now rescales until the measured
   body is the requested size (30 cells, `tau = 0.5465`, thickness 3).
3. **"Record must not drop" is about files, not about MP4** (**D-039**). A gap in a numbered PNG
   series is as wrong and as silent as a missing video frame, so `--headless` takes `drop=False`
   too; `drop=True` is now reached only by a live-*only* run. `TeeSink` makes the flags composable
   without inventing a third mode — it passes each member the **same array**, which is a stronger
   statement of constraint 10 than the byte-equality the criterion asked for.
