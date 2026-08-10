# STATE1.md — live project state

**Read this first, every session, in full.** Updated by `/checkpoint` at the end of every session.
Never rewrite or condense the session log — append only.

---

## Snapshot

| Field | Value |
|---|---|
| **Phase** | Phase 0 — D2Q9 LBM in NumPy (`DOCS/IDEA2.md`) |
| **Current task** | `T005` |
| **Task status** | `not_started` |
| **Completed tasks** | T001, T002, T003, T004 |
| **Milestone reached** | **M2** (2026-08-10, gate run: `python -m validate.cavity --re 100 --re 400 --re 1000` → PASS; max deviation from Ghia 0.75% / 0.42% / 1.01%, vortex centre 0.21 / 0.29 / 0.59 cells) — next: M3 at T007 |
| **Rung status** | R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜ |
| **Last updated** | 2026-08-10 — session 4 (T004 done; geometry primitives + mask checks) |

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

None.

## Open questions

- ~~**Q-001** — wall-offset convention for bounce-back.~~ **Closed in session 2 by measurement** —
  see **D-009**. The wall sits halfway between the last fluid node and the solid node.
- **Q-002** — SVG rasterisation dependency (T009) not chosen. Not blocking; PNG is what M4 needs.
- ~~**Q-003** — do the two lid corner cells belong to the moving lid or to the side walls?~~
  **Closed in session 3 by measurement** — see **D-013**. They belong to the static side walls.

## Environment

Project venv: `myenv/` (gitignored). Python 3.11.15.

| Package | Version | Added by |
|---|---|---|
| numpy | 2.4.6 | pre-existing |
| matplotlib | 3.11.1 | pre-existing |
| pillow | 12.3.0 | pre-existing |
| pytest | 9.1.1 | T001 (session 1) |

Not yet installed, needed later: `pygame` (T007), `imageio[ffmpeg]` (T011).

Tests are run from the repo root so that `import lbm` resolves (no `pip install -e .`; there is no
packaging config in Phase 0). `python -m pytest` from the root works; a script run from elsewhere
needs `PYTHONPATH` set to the repo root.

Install with `myenv/Scripts/pip.exe install <pkg>` and **add a row above in the same session.**

## Performance baseline

Not yet measured. Recorded in T010.

| Grid | Cells | Baseline steps/s | Post-optimisation | Budget floor |
|---|---|---|---|---|
| 400×100 | 40k | — | — | ≥400 |
| 800×200 | 160k | — | — | ≥120 |
| 2000×500 | 1M | — | — | ≥15 |

## Decisions

Anything chosen that wasn't already specified in `DOCS/IDEA2.md` or `CLAUDE.md`. Append; never edit
a past entry — supersede it with a new one that says so.

| ID | Date | Decision | Why |
|---|---|---|---|
| D-001 | 2026-08-09 | Docs live in `DOCS/`, next-session prompts in `PROMPTS/`, agentic config in `.claude/`. `idea2.md` moved to `DOCS/IDEA2.md`. | Keeps the repo root for code and product-level `idea.md` / `README.md`. |
| D-002 | 2026-08-09 | One task per session, enforced by `/start-task` + `/checkpoint`. | Small context per session; every session boundary is a validated state. |
| D-003 | 2026-08-09 | `myenv/` is the canonical interpreter; commands in docs use `myenv/Scripts/python.exe`. | A global-python invocation would silently miss project deps on Windows. |
| D-004 | 2026-08-09 | `Navier-Fluid-Equation/` treated as read-only prior work; only its polygon-vertex logic is reused, in T004. | It is potential-flow, a different method. Mixing the two codebases would confuse both. |
| D-005 | 2026-08-09 | Velocity is `u` of shape `(2, ny, nx)`, component 0 = `ux`, component 1 = `uy` — matching the `(ex, ey)` column order of `E`. | Makes the first-moment reduction a plain `E.T @ f`; any other order needs a transpose nobody would remember. |
| D-006 | 2026-08-09 | Hot functions take optional preallocated outputs: `macroscopic(f, rho=None, u=None)`, `equilibrium(rho, u, feq=None, work=None)` where `work` is `(3, ny, nx)` scratch. Allocating only when the caller passes nothing. | Satisfies "never allocate in the step loop" without forcing T001 to invent the runner's buffer ownership. T006 passes buffers; tests cover both paths. |
| D-007 | 2026-08-09 | `E` is stored `int32`; a companion `E_F32` holds the same table as `float32` for arithmetic. Both live in `lbm/core.py`; this is not a second definition of the constant. | Streaming and geometry need integer offsets; equilibrium needs float without an upcast to `float64` on every call. |
| D-008 | 2026-08-09 | `1.5 * u^2` is folded into the `usq` scratch once per `equilibrium` call rather than per direction. | Nine-fold fewer multiplies, zero clarity cost, and it is not a fused kernel — constraint 6 is about restructuring collide/stream, which is untouched. |
| D-009 | 2026-08-10 | **Closes Q-001.** Bounce-back walls sit **halfway between the last fluid node and the solid node**. For solid rows at `y=0` and `y=ny-1`: fluid rows `1..ny-2`, wall planes at `y=0.5` and `y=ny-1.5`, channel height `H = ny-2`, fluid row `y` evaluated at `y-0.5`. Rung 2's cavity `L` uses the same convention. | Measured, not argued: `validate/poiseuille.py` prints the L2 error for all three rival conventions on every run — halfway **0.365%**, `H=ny-3` 14.763%, `H=ny-1` 12.746%. The remaining 0.365% is a uniform ~1.1e-4 deficit, the known `tau`-dependent effective wall offset of BGK bounce-back (exact only at `(tau-0.5)^2 = 3/16`); fixing it needs TRT/MRT, excluded by constraint 1. |
| D-010 | 2026-08-10 | Body force uses the **Guo (2002) scheme**: velocity correction `u += F/(2 rho)` before equilibrium (`boundary.force_velocity_shift`), plus the source term `S_i = (1 - 1/(2 tau)) w_i [3(e_i.F - u.F) + 9(e_i.u)(e_i.F)]` after collision (`boundary.apply_body_force`). Chosen over the one-line velocity-shift shortcut. | User's call when both were offered. Guo is second-order consistent and stays correct for time-dependent and non-uniform forces, which T005's probes and any future forcing will want; the shortcut would have had to be replaced later. `sum_i S_i = 0` exactly, so mass conservation is structural rather than tuned. |
| D-011 | 2026-08-10 | The "pre-stream copy" that `bounce_back` consumes is the copy taken **before collision** of the current step. Timestep order is: `copy f_pre` → `macroscopic` → `force_velocity_shift` → `equilibrium` → `collide` → `apply_body_force` → `bounce_back` → `stream`. | The contract said "pre-stream", which admits two readings. Only this one reflects the populations that actually arrived at the solid cell during the previous stream; taking the copy after collision instead makes solid cells swap their own values forever and never see the fluid, which reads as a plausible wall but transmits no momentum. Documented in `lbm/boundary.py`'s module docstring so T003/T005 inherit the order. |
| D-012 | 2026-08-10 | Rung 1's steady-state residual tolerance is `5e-6` on `max|du|/peak|u|` per 100 steps, not something tighter. | The `float32` floor: `u` is a near-cancelling sum of `f ~ 0.4` divided by `rho`, so its round-off is about `eps*|f|/|u| ~ 1.2e-6`. Measured floor 1.7e-6. A `1e-9` tolerance is unreachable and simply burns the step cap. Same reasoning sets `atol=1e-6` on the momentum-conservation unit test. This is a dtype limit, not a relaxed physics tolerance — the physics tolerances (1% L2, 2% doubling) are untouched. |
| D-013 | 2026-08-10 | **Closes Q-003.** The two cells where the lid meets the side walls belong to the **static no-slip walls**, not to the moving lid. `validate/cavity.py` defaults to `--corners wall` and keeps `--corners both` so the comparison can be re-run. | Measured, not argued, the same way Q-001 was closed. Max deviation from Ghia (fraction of lid velocity), lid / wall: Re 100 **0.51% / 0.75%**, Re 400 **1.21% / 0.42%**, Re 1000 **1.35% / 1.01%** — worst case across the three is **1.01% for wall against 1.35% for lid**, and the criterion is a worst case. Both choices pass; this is the better one, not the only viable one. It also matches the physics: the corner is where the moving and stationary walls meet and the velocity is genuinely singular, so handing the cell the full lid momentum puts the largest possible source right at the point of worst truncation error. |
| D-014 | 2026-08-10 | Rung 2's steady-state residual is the **per-step** velocity change, `max\|u(n) - u(n-k)\| / (U k)` with `k = 500` and `U` the lid velocity, compared against the contract's `1e-6`. The raw interval difference is printed unscaled next to it. The residual is measured on the **fluid interior only**. | Two separate things. (a) D-012's `float32` floor is on a *per-step* difference; measuring over 500 steps and dividing by 500 keeps the same physical quantity — the rate of change — while pushing the round-off floor down by the interval length, so the contract's `1e-6` is reachable as written instead of needing to be loosened. Measured: every case converges at 7e-7 … 9.7e-7, i.e. the criterion binds and is not floor-limited. (b) `rho` on solid cells is whatever bounce-back left there, so `u = (e.f)/rho` on them is meaningless and made the whole-array residual read `8.4e+01`; the first version of the script "failed to converge" for exactly that reason. |
| D-015 | 2026-08-10 | **One reference value is treated as corrupt**: Ghia Table II, Re 400, `v` at `x = 0.9063`, tabulated `-0.23827`. It is excluded from the pass criterion (`validate/cavity.py` `GHIA_SUSPECT`) and **still printed with its deviation on every run**. No substitute value is invented. | Measured, not argued. (1) This solver matches the other 16 points of that same profile to within 1.2% of the lid velocity, and matches Re 100 and Re 1000 at that *same* station to within 1.2% — a boundary-condition error would not hide at one station of one Reynolds number. (2) Grid convergence at that station, `L = 64/128/256`: `-0.36265 → -0.37522 → -0.37806`, converging (1.26% then 0.28%) and nowhere near `-0.23827`; over the same refinement every other station moves *toward* its tabulated value. (3) The entry has the wrong curvature for its own column — it sits above the chord through its neighbours where Re 100 and Re 1000 both sit below (asserted in `tests/test_lid.py`). A single mistyped digit (`-0.33827` → `-0.23827`) explains it, but guessing the true value would be inventing data, so the point is dropped rather than replaced. |
| D-016 | 2026-08-10 | Rung 2's case setup: lid velocity `U = 0.09` (not 0.1), `L = 128` for Re 100 and 400, `L = 256` for Re 1000, and a hard `TAU_FLOOR = 0.53` enforced in `tau_for` alongside a hard rejection of `U >= 0.1`. Both raise at setup and name the resolution that would fix it. | Constraint 3 says lattice velocity **under** 0.1; `U = 0.1` puts the lid exactly *on* the ceiling, and the lid is the fastest thing in the cavity, so 0.09 is the honest reading (measured peak `\|u\|` 0.0880 at Re 100). `L` then follows from `tau = 0.5 + 3 U L / Re`: Re 1000 at `L = 128` would give `tau = 0.5346`, marginal, so `L = 256` buys `tau = 0.5691` at 258² and 81500 steps (158 s). `TAU_FLOOR = 0.53` refuses the marginal cases outright rather than letting them produce a plausible-looking checkerboard — `DOCS/IDEA2.md` § Stability lists exactly that failure. |
| D-017 | 2026-08-10 | **Minimum solid thickness** (`lbm/geometry.py::min_thickness`) is measured **per 8-connected component** as `2 * max(d) - 1`, where `d` is the Chebyshev distance from a solid cell to the nearest fluid cell — the side of the largest fully-solid square that fits in that component — and the reported value is the minimum over components. | Measured, not argued: the two obvious metrics were written first and both **false-alarm on a plain cylinder**, which would have made every Rung 3 run warn and trained us to ignore the warning. (a) Run lengths: the topmost cell of a disc has a vertical run of 1. (b) Per-cell 3x3 opening ("is every solid cell covered by a fully-solid 3x3 square?"): for a disc of radius 10 centred at (30,30), the pole (20,30) needs (20,29), which is at distance² 101 and therefore fluid — so no square covers it. Digitised curvature always produces locally thin boundary cells and they are not what leaks. Component depth gives disc 15, 4-cell block 3, 2-cell bar 1, 1-cell diagonal 1. Odd-valued and rounds down (a 4-thick block reads 3) — the safe direction for a warning. Known limit, documented in the docstring: a thin appendage **fused** to a thick body shares its component and is not reported; primitives here are convex blobs, but a T009 PNG could hide a hairline behind this. |
| D-018 | 2026-08-10 | **Domain borders are exempt from all three mask checks.** `check_mask` first calls `strip_solid_border`, which peels *entirely* solid edge rows and columns one layer at a time; the checks then run on what remains, the immersed object. A mask with nothing left passes silently. | Constraint 12's 3-cell rule exists because fluid leaks *through* a thin obstacle to the fluid on the far side. A domain border has no far side, and the existing rungs both use one-cell borders on purpose (D-009, `validate/cavity.py::cavity_masks`). Without the exemption, Rung 1's own mask warns — a check that cries wolf on the code's own passing benchmarks gets suppressed, which is exactly the failure mode `DOCS/PLAN1.md` § Risks warns about. Only fully-solid edge layers are peeled, so an object that merely touches the edge is still checked. |
| D-019 | 2026-08-10 | **Characteristic length `D` is the cross-stream extent of the object's bounding box** (bbox height for `inlet_axis="x"`), and the blockage denominator is the **fluid** span — the cross-stream domain minus fully-solid border layers. Both are printed with the bbox, the streamwise extent, the downstream distance in `D`, the blockage and the thickness; `check_mask(verbose=True)` is the default. | `DOCS/IDEA2.md` § Geometry defines blockage as "object height / domain height", and "8 diameters downstream" is the same `D` for a cylinder, so one bbox-derived quantity serves both rules and there is nothing for a caller to pass in inconsistently. Counting the wall rows in the denominator would flatter the blockage ratio by a couple of percent at Rung 1 resolutions. Printing by default is the acceptance criterion ("derived from the mask bounding box and printed") and is what makes a warning diagnosable rather than mysterious. |

## Session log

Append one entry per session. Newest at the bottom.

### 2026-08-09 — Session 0: scaffold

**Task worked:** none (setup)

**Done**
- Read `DOCS/IDEA2.md` in full; confirmed environment (`myenv`, numpy 2.4.6, matplotlib 3.11.1).
- Created `CLAUDE.md` (12 hard constraints, session protocol, conventions, module map).
- Created `DOCS/PLAN1.md` (11 tasks, dependency graph, session map, milestone gates, risks).
- Created `DOCS/TASKS1.md` (full contract per task: goal, depends-on, I/O, acceptance criteria,
  constraints that bite, notes).
- Created `DOCS/STATE1.md` (this file) and `PROMPTS/templates/session-prompt-template.md`.
- Created `.claude/`: `commands/start-task.md`, `checkpoint.md`, `new-task.md`, `validate.md`, plus
  `settings.json`. Adapted from a prior project's command set (`.claude-blueprint/`, since deleted —
  do not look for it).
- Moved `idea2.md` → `DOCS/IDEA2.md`.
- Generated `PROMPTS/001-t001-core-equilibrium.md`.

**Not done / deferred**
- No solver code at all. That is intentional — T001 is session 1.

**Decisions made**
- D-001 through D-004 above.

**Blockers**
- None.

**Next**
- Paste `PROMPTS/001-t001-core-equilibrium.md` into a fresh session. It runs `/start-task T001`.

### 2026-08-09 — Session 1: T001 — D2Q9 constants, macroscopic, equilibrium

**Task worked:** T001 — `done`, every acceptance criterion run and green.

**Done**
- `lbm/__init__.py` — package docstring (everything inside is lattice units), re-exports the public
  names from `lbm.core`.
- `lbm/core.py` — `E (9,2) int32`, `W (9,) float32`, `OPP (9,) int32`, `CS2 = 1/3`, `E_F32`, `Q = 9`;
  `nu_from_tau(tau)`, `macroscopic(f, rho=None, u=None)`, `equilibrium(rho, u, feq=None, work=None)`.
  Docstrings cite `DOCS/IDEA2.md` § The method steps 1–2 and the constraint each obeys.
- `tests/test_core.py` — 21 tests. Beyond the contract they also pin: `sum_i w_i ex_i^2 == cs2`, the
  rest-state equilibrium `feq_i == w_i*rho`, the second moment `sum_i feq_i e_a e_b ==
  rho*(cs2*delta_ab + u_a u_b)`, and the axis convention (`u[0]` is `ux`, and a +x flow puts more
  mass in direction 1 than 3 while directions 2 and 4 stay equal).
- Installed `pytest` 9.1.1 into `myenv`; § Environment row added above.

**Measured** (random `rho` in `[0.9,1.1]`, `|u| <= 0.099`, tolerance `1e-5`):
- `W.sum() - 1` = 0.0 exactly · `E[OPP[i]] == -E[i]` for all `i`
- zeroth moment error `2.4e-07` · first moment error `3.0e-08`
- round trip `rho` `2.4e-07`, `u` `3.0e-08` · `nu_from_tau(0.6) = 0.0333…` · `tau <= 0.5` raises
  naming `tau` · all returned arrays `float32`
- `myenv/Scripts/python.exe -m pytest` → `21 passed in 0.18s`

**Not done / deferred**
- Nothing from the T001 contract. `collide`, `stream`, bounce-back and `validate/poiseuille.py` are
  T002 and were deliberately not started (contract § Notes).

**Decisions made**
- D-005 (velocity component order), D-006 (optional preallocated outputs), D-007 (`E_F32` companion),
  D-008 (`usq` hoisted out of the direction loop). All above.

**Blockers**
- None.

**Rung status after this session**
- R1 ⬜ · R2 ⬜ · R3 ⬜ · R4 ⬜ — `validate/` does not exist yet. Correct for session 1; the first
  rung is built in T002.

**Next**
- Paste `PROMPTS/002-t002-collide-stream-poiseuille.md` into a fresh session. It runs
  `/start-task T002`. Q-001 (wall offset) must be decided and logged there.

### 2026-08-10 — Session 2: T002 — collide, stream, bounce-back, body force → Rung 1

**Task worked:** T002 — `done`, every acceptance criterion run and green. **M1 reached.**

**Done**
- `lbm/core.py` — added `collide(f, feq, tau)` (three in-place ops, algebraically identical to
  `f -= (f - feq)/tau`, no temporary) and `stream(f, buf)` (block-copy periodic shift, equal to the
  spec's double `np.roll` but allocation-free; `f` keeps its buffer identity, which T006's restart
  test will want). Private helper `_shift_blocks`. Module docstring now says steps 1–4.
- `lbm/boundary.py` — new. `bounce_back(f, f_pre, solid)`, `force_velocity_shift(rho, u, g, work)`,
  `apply_body_force(f, rho, u, tau, g, work)`. Module docstring pins the timestep order (D-011);
  `bounce_back`'s docstring carries the wall-offset convention (D-009).
- `lbm/__init__.py` — re-exports the five new names.
- `validate/__init__.py`, `validate/poiseuille.py` — Rung 1. Prints PASS/FAIL, per-row profile vs
  analytic, and the L2 error under all three rival wall conventions.
- `tests/test_step.py` — 26 tests: collide vs the literal expression, allocation-freeness via
  `tracemalloc`, equilibrium as a fixed point, moment conservation, `tau <= 0.5` rejection; the
  single-cell spike test for all 9 directions, equality with `np.roll`, periodic wrap, buffer
  identity; bounce-back reversal on solid with fluid untouched; Guo `sum_i S_i = 0`, first moment
  `(1 - 1/(2 tau)) F`, and a term-by-term float64 comparison against the textbook formula.

**Measured**
- `myenv/Scripts/python.exe -m validate.poiseuille` → **PASS**. 22×16, `tau=0.6`, `gx=2.6667e-5`,
  converged 10600 steps (residual 3.30e-06); halved case `tau=0.55`, 21000 steps.
  - L2 relative error **0.3650%** (limit 1%)
  - halving `(tau-0.5)` doubles centreline: ratio **1.99940** (0.039789 → 0.079554), limit ±2%
  - mass drift over 5000 steps **5.186e-05** (limit 1e-4)
  - finite after 20000 steps at `tau=0.6`
  - peak lattice velocity **0.07955** (limit 0.1)
  - wall-convention sweep: halfway 0.365% · `H=ny-3` 14.763% · `H=ny-1` 12.746%
- `myenv/Scripts/python.exe -m pytest` → **`47 passed in 0.21s`** (T001's 21 still green + 26 new).

**Not done / deferred**
- Nothing from the T002 contract. Moving-lid BC and cavity are T003; `channel_walls` and the mask
  sanity checks are T004, so Rung 1 builds its two solid rows inline with NumPy on purpose.
- No optimisation: collide and stream stay separate passes, and the body force is applied on solid
  cells too rather than masked. Both are T010, gated on Rung 3 (constraint 6).

**Decisions made**
- **D-009** (wall offset — closes Q-001), **D-010** (Guo forcing), **D-011** (pre-collision copy and
  the timestep order), **D-012** (float32 residual floor). All above.
- One tolerance was set at write time and is worth naming: the momentum-conservation unit test uses
  `atol=1e-6`, because momentum is a near-cancelling sum of `f ~ 0.45` values and `float32` gives it
  a ~1e-7 absolute noise floor. Commented in the test. No physics tolerance was relaxed.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 ⬜ · R3 ⬜ · R4 ⬜. R2–R4 have no script yet (T003, T007, T008) — not attempted, not
  failing.

**Next**
- Paste `PROMPTS/003-t003-moving-lid-cavity.md` into a fresh session. It runs `/start-task T003`.
  Q-003 (lid corner cells) should be decided and logged there. `DOCS/PLAN1.md` § Risks flags T003 as
  the task most likely to need two sessions — timebox to one, then log and try Zou–He walls.

### 2026-08-10 — Session 3: T003 — moving-lid BC + cavity benchmark → Rung 2

**Task worked:** T003 — `done`, every acceptance criterion run and green. **M2 reached.**

**Done**
- `lbm/boundary.py` — added `moving_wall(f, f_pre, wall, u_wall, rho_w=1.0)`: momentum-corrected
  (Ladd) bounce-back, `f[i] = f_pre[OPP[i]] + 6 w_i rho_w (e_i . u_wall)`. Zou–He was never needed.
  Its docstring carries the corner-cell decision (D-013) with the measured table.
- `lbm/__init__.py` — re-exports `moving_wall`.
- `validate/cavity.py` — new, Rung 2. Ghia Tables I and II as literal arrays with the citation
  (`GHIA_Y`/`GHIA_U`, `GHIA_X`/`GHIA_V`, `GHIA_VORTEX`), `cavity_masks`, `tau_for`,
  `vortex_centre` (streamfunction extremum with a parabolic sub-cell fit), `run_cavity`, `report`.
  Prints resolution, the `nu = U L / Re` → `tau` arithmetic, steps/s, residual, peak `|u|`, five
  per-case checks, and the excluded reference point. `--corners both` reruns each Re both ways.
- `tests/test_lid.py` — 16 tests: `u_wall = 0` reproduces `bounce_back` exactly; the Ladd term
  term-by-term; fluid cells untouched; `sum_i 6 w_i (e_i.u_w) = 0` so the lid adds momentum and not
  mass; a `+x` lid gives net `+x` momentum in the three re-entering directions; allocation-freeness
  via `tracemalloc`; the two corner masks differ in exactly two cells; `tau_for` inverts
  `Re = U L / nu` and rejects both a too-coarse grid and a lid at 0.1; the vortex finder on a
  planted extremum; the exclusion list is exactly one point and the excluded value has the wrong
  curvature for its own column.

**Measured**
- `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000` → **PASS**
  (`corners = wall`, `U = 0.09`):
  - Re 100 — 130², `tau` 0.8456, 14500 steps, 8.2 s; max deviation **0.75%**, vortex 0.21 cells,
    residual 6.95e-07/step, peak `|u|` 0.08797
  - Re 400 — 130², `tau` 0.5864, 33000 steps, 21.1 s; max deviation **0.42%**, vortex 0.29 cells,
    residual 8.83e-07/step, peak `|u|` 0.08679
  - Re 1000 — 258², `tau` 0.5691, 77500 steps, 148.3 s; max deviation **1.01%**, vortex 0.59 cells,
    residual 9.77e-07/step, peak `|u|` 0.08752
- Corner comparison (`--corners both`), max deviation lid / wall: Re 100 0.51% / 0.75%, Re 400
  1.21% / 0.42%, Re 1000 1.35% / 1.01%. Worst case 1.35% lid vs **1.01% wall** → D-013.
- Grid convergence of the Re 400 `v` profile at `x = 0.9063`, `L = 64/128/256`:
  `-0.36265 → -0.37522 → -0.37806`, against a tabulated `-0.23827` → D-015.
- `myenv/Scripts/python.exe -m validate.poiseuille` → **PASS**, L2 **0.3650%**, ratio 1.99940,
  peak 0.07955 — identical to session 2, so nothing regressed.
- `myenv/Scripts/python.exe -m pytest` → **`63 passed in 0.30s`** (47 existing + 16 new).

**Not done / deferred**
- Nothing from the T003 contract. Zou–He was not implemented — the valve in `DOCS/PLAN1.md` § Risks
  never opened, since momentum-corrected bounce-back passed on the first physics run.
- Geometry still built inline with NumPy in `validate/cavity.py` (`cavity_masks`); the shared
  primitives and the mask sanity checks are T004, deliberately not started.
- No optimisation (constraint 6). The body-force pair is not called at all here (D-010).

**Decisions made**
- **D-013** (lid corner cells — closes Q-003), **D-014** (per-step residual, fluid interior only),
  **D-015** (one corrupt Ghia reference point excluded and printed), **D-016** (case setup: `U`,
  per-Re `L`, `TAU_FLOOR`). All above.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜. R3 and R4 have no script yet (T007, T008) — not attempted, not
  failing.

**Next**
- Paste `PROMPTS/004-t004-geometry-primitives.md` into a fresh session. It runs `/start-task T004`.
  T004 is not gated by a rung; its gate is unit tests. Q-002 (SVG rasteriser) is adjacent to it but
  belongs to T009 and should not be answered early.

### 2026-08-10 — Session 4: T004 — geometry primitives + mask sanity checks

**Task worked:** T004 — `done`, every acceptance criterion run and green. No milestone (M3 is T007).

**Done**
- `lbm/geometry.py` — new. Primitives `circle`, `rectangle`, `polygon`, `regular_polygon`,
  `channel_walls`; measurement `bounding_box`, `min_thickness`, `strip_solid_border`; the three
  sanity checks in `check_mask(solid, inlet_axis, *, min_thickness_cells=3,
  min_downstream_lengths=8.0, max_blockage=0.10, strict=False, verbose=True) -> list[str]`.
  Warnings go through `warnings.warn` with a `MaskWarning(UserWarning)` category; `strict=True`
  raises `ValueError` with every message and emits nothing. Private helpers `_grid`, `_shift`,
  `_erode`, `_wall_distance`, `_label`.
- `polygon` is even-odd ray casting, reimplemented from the `matplotlib.path.Path.contains_points`
  use in `Navier-Fluid-Equation/polygonsDemo.py` and the vertex generator in `panels.py`; nothing is
  imported from that directory and it was not modified (D-004). The loop is over edges with one
  whole-grid NumPy op inside — no vectorisation over edges (constraint 6).
- `lbm/__init__.py` — re-exports the ten new public names.
- `tests/test_geometry.py` — 40 tests: `(ny, nx)` shape and `bool` dtype for every primitive, the
  `(y, x)` index order, three known-area cases within 2% (rectangle-as-polygon, hexagon, disc), the
  concave L (notch fluid, area 700 not the hull's 1600), reversed vertex order, `channel_walls`
  byte-equal to `validate/poiseuille.py::channel_mask` and composable with `|`, the thickness
  criterion (1-cell diagonal vs 4-cell block), the disc non-false-alarm, a thin plate beside a thick
  body, border stripping for both channel and cavity masks, each of the three rules firing alone and
  all three at once in order, `strict`, the warning category, and the printed characteristic length
  via `capsys`.

**Measured**
- `myenv/Scripts/python.exe -m pytest tests/test_geometry.py` → **`40 passed`**
- `myenv/Scripts/python.exe -m pytest` → **`103 passed in 1.64s`** (63 existing + 40 new)
- `myenv/Scripts/python.exe -m validate.poiseuille` → **PASS**, L2 **0.3650%**, ratio 1.99940,
  peak `|u|` 0.07955 — identical to sessions 2 and 3.
- `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000` → **PASS**, max deviation
  **0.75% / 0.42% / 1.01%**, vortex 0.21 / 0.29 / 0.59 cells, peak `|u|` 0.08797 / 0.08679 / 0.08752
  — identical to session 3.
- `min_thickness` readings: 1-cell diagonal **1**, 2-cell bar **1**, 4-cell block **3**, 20-cell
  block **19**, disc of radius 10 **15**, thin plate beside a thick block **1**.
- Worth carrying to T007: a `D = 21` cylinder in a `121`-row channel is **17.6% blockage** and
  `check_mask` warns. Rung 3 needs roughly `ny >= 10 D` (about 240 rows for `D = 21`), plus
  `>= 8 D` downstream. This is the check doing the job `DOCS/PLAN1.md` § Risks assigned it, before
  the run rather than after.

**Not done / deferred**
- Nothing from the T004 contract. PNG and SVG masks are T009 and were not started; Q-002 was left
  open on purpose.
- `validate/poiseuille.py` and `validate/cavity.py` still build their masks inline. Rewriting them
  onto the primitives was allowed only if both rungs reprinted the same numbers, and it buys nothing
  — `test_channel_walls_matches_the_inline_rung_1_mask` asserts byte-equality instead, which proves
  the generalisation without touching a passing benchmark.
- No optimisation (constraint 6). `_label` is plain max-propagation, one pass per cell of component
  diameter; geometry runs once at setup.

**Decisions made**
- **D-017** (thickness metric — component-wise Chebyshev depth, after two rejected metrics),
  **D-018** (domain borders exempt from the checks), **D-019** (characteristic length and the
  blockage denominator). All above.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜. R3 and R4 have no script yet (T007, T008) — not attempted, not
  failing.

**Next**
- Paste `PROMPTS/005-t005-inlet-outlet-probes.md` into a fresh session. It runs `/start-task T005`.
  T005's gate is unit tests; `forces` is the error-prone one and Rung 3 (T007) is what audits it.
