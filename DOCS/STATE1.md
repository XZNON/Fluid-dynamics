# STATE1.md — live project state

**Read this first, every session, in full.** Updated by `/checkpoint` at the end of every session.
Never rewrite or condense the session log — append only.

---

## Snapshot

| Field | Value |
|---|---|
| **Phase** | Phase 0 — D2Q9 LBM in NumPy (`DOCS/IDEA2.md`) |
| **Current task** | `T011` (next; T010 closed) |
| **Task status** | `not_started` |
| **Completed tasks** | T001, T002, T003, T004, T005, T006, T007, T008, T009, T010 |
| **Milestone reached** | **M3** (2026-08-12, gate run: `python -m validate.cylinder` → PASS with the live window open; **St 0.1731** (band 0.155–0.175), **Cd 1.4031 ± 0.0086** (band 1.25–1.45), Cl amplitude 0.3915 = 27.9% of Cd, window costs **+2.09%** of steps/s) — next: M4 at T011 |
| **Rung status** | R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **the ladder is complete** |
| **Last updated** | 2026-08-13 — session 10 (T010 done; fused kernel, ladder re-run green and bit-identical, 329 tests pass) |

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

None.

## Open questions

- ~~**Q-001** — wall-offset convention for bounce-back.~~ **Closed in session 2 by measurement** —
  see **D-009**. The wall sits halfway between the last fluid node and the solid node.
- ~~**Q-002** — SVG rasterisation dependency (T009) not chosen.~~ **Closed in session 9 by choosing
  no dependency** — see **D-031**. `lbm.geometry.from_svg` parses the simple-closed-path subset
  itself and fills it with T004's `polygon`; anything outside that subset raises an `ImportError`
  naming the feature and the `cairosvg` install line.
- ~~**Q-004** — should `validate/cylinder.py::tau_for`'s floor rise from D-016's 0.53 to the 0.54
  that session 8 measured (**D-029**)?~~ **Closed in session 10 — see D-036.** Answer: it rises, but
  to **0.537**, not to 0.54. 0.54 would make Rung 3 refuse its own published case (`tau = 0.5378`)
  and so would force a re-tuned, re-published Rung 3; 0.537 sits inside the measured death bracket
  `(0.5346, 0.5378]`, above everything measured to blow up and below Rung 3's measured-stable
  operating point. Rung 4 keeps its stricter 0.54 and now owns the band between them.
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
| pygame | 2.6.1 | T007 (session 7) |

Not yet installed, needed later: `imageio[ffmpeg]` (T011).

Tests are run from the repo root so that `import lbm` resolves (no `pip install -e .`; there is no
packaging config in Phase 0). `python -m pytest` from the root works; a script run from elsewhere
needs `PYTHONPATH` set to the repo root.

Install with `myenv/Scripts/pip.exe install <pkg>` and **add a row above in the same session.**

## Performance baseline

Measured in T010 (session 10, 2026-08-13) with `bench.py`, on `Sim.step` for an open
channel with a Zou–He inlet, a convective outlet and an immersed disc — the whole step, not
`collide` in isolation. numpy 2.4.6, Python 3.11.15, AMD64.

| Grid | Cells | Baseline steps/s | Post-optimisation | Budget floor | Budget |
|---|---|---|---|---|---|
| 400×100 | 40k | **739.9** | **696.7** | ≥400 🟩 | ~500 |
| 800×200 | 160k | **158.6** | **161.7** | ≥120 🟩 | ~150 |
| 2000×500 | 1M | **15.0** | **16.8** | ≥15 🟩 | ~20 |

**Every floor is cleared, and was already cleared before the optimisation.** The honest
reading of this table is that the kernel was never far off the budget; T010's fusion buys
the 1M case its margin (+12–14%) and is a wash on the two smaller ones.

Two baselines exist and they disagree, which is the measurement being honest rather than a
contradiction:

- **Archived, captured before the first edit** (`DOCS/bench_baseline.json`, the criterion's
  "before any change"): **739.9 / 182.6 / 17.4** steps/s.
- **The `before` column above**, which is the *same code path* (`SimConfig(fused=False)` is
  the T009 solver, kept selectable) re-measured in the same process as the `after` column.

The gap between them is machine drift, not code: two back-to-back runs of the identical
reference path measured **726.8 and 811.5** steps/s at 400×100 (12% apart) and **146.1 and
177.2** at 800×200 (21% apart). Any speed claim smaller than that spread is unmeasurable
by sequential runs, which is why `bench.py` alternates variants round by round and keeps
the best round per variant — see **D-035**. The `after` column is the number to trust; the
archived file is the audit trail the contract asked for.

**The table above holds at the CPU's rated clock, and the machine's power state moves it more than
any optimisation in this task did.** Later in the same session, with the *identical* build, the
numbers read **383.6 / 117.8 / 15.5** (before) and **402.7 / 117.0 / 16.8** (after) — and **800×200
misses its floor of 120 at 117.0**. The cause is not the kernel and was diagnosed, not guessed:

| | early session | late session |
|---|---|---|
| `Win32_Processor.CurrentClockSpeed` | 3201 MHz rated | **1802 MHz** (56%) |
| power | mains | **battery, 42%**, Balanced scheme |
| after-column steps/s | 696.7 / 161.7 / 16.8 | 383.6→402.7 / 117.0 / 16.8 |

So the honest statement of the acceptance criterion is: **the floors are cleared on this machine at
its rated clock; on battery at ~56% clock the 160k case sits 2.5% under its floor.** A future session
re-measuring this table must plug the laptop in and check `CurrentClockSpeed` first, or it will be
measuring the power plan. Nothing in `lbm/` changed between the two readings.

The relative `before`/`after` ratios survive all of it (1.01–1.08× while throttled, 1.00–1.14× cool),
because both columns throttle together — which is the whole reason D-035 alternates them.

Where the time goes at 1M cells (per step, measured): `equilibrium` **39.9 ms**,
`collide` 15.1 ms, `stream` 6.1 ms, `macroscopic` 7.2 ms, the two snapshot copies 5.8 ms,
`bounce_back` 0.8 ms. **`equilibrium` is over half the step** and is the obvious target for
M5's port; it is deliberately untouched here (`DOCS/TASKS1.md` § T010 Notes — Phase 0's job
is understanding, and M5 replaces this kernel).

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
| D-020 | 2026-08-10 | **The two snapshots `lbm.probe.forces` consumes are the pre-stream and post-stream states**, and the runner owns one extra `(9, ny, nx)` buffer for the first: `bounce_back(f, f_pre, solid)` → `np.copyto(f_bb, f)` → `stream(f, buf)` → `forces(f_bb, f, links, ...)`. The D-011 timestep order gains a tail: `stream` → `outlet_zero_gradient` → `inlet_velocity`. | The parameter named `f_pre` in `forces` is **not** the `f_pre` of `bounce_back`, which is the *pre-collision* copy (D-011) — one name, two meanings, and without fixing it now T006 would have had to reshape the API to find out. Momentum exchange needs the population leaving the fluid node (pre-stream) and the reflected one arriving back (post-stream). Because bounce-back is exactly `f[j](x_s) = f_pre_collision[opp(j)](x_s)`, that returning population can equivalently be read off the pre-stream array on the solid side; `tests/test_probe.py::test_the_two_snapshot_form_equals_the_solid_side_form` asserts the identity, so the timing is pinned by a test and not by a comment. The open boundaries go **after** `stream` because `stream` is periodic in `x`: after it, the inlet column's `ex = +1` populations and the outlet column's `ex = -1` populations are precisely the wrap-around garbage those two functions overwrite. |
| D-021 | 2026-08-10 | **The outlet is convective, not a bare copy.** `outlet_zero_gradient(f, *, col, src, prev=None, lam=None)` keeps `f[:, :, -1] = f[:, :, -2]` as the default and adds `f[:, :, -1] = (prev + lam f[:, :, -2]) / (1 + lam)` when the caller supplies the previous outlet column; `lam` defaults to `sqrt(CS2) = 0.577`. The runner supplies `prev`. | Measured, not argued, the way Q-001 and Q-003 were closed. A smooth Gaussian pressure pulse (`sigma = 10` cells, 400-cell domain) fired at the boundary reflects: **plain copy 35%**, `lam = 0.4` 4.7%, **`lam = cs` 0.6%**, `lam = 1.0` 7.5%, `lam = 2 cs` 17%. The criterion is "under 5%", and **the bare copy the contract describes does not meet it**. The copy is the `lam -> inf` limit of the same expression, and the minimum is sharp and sits exactly at the lattice sound speed — which is what an advection boundary tuned to the outgoing wave speed is supposed to do. `DOCS/IDEA2.md` § Stability already anticipates this ("proper zero-gradient / sponge layer"). The copy stays the default because it is the documented behaviour and needs no state; its 35% is pinned by its own test so the convective form cannot be deleted later as redundant. `lam = U` rather than `cs` is the other defensible tuning — wake vorticity is advected, not radiated — and is left exposed for Rung 3 (T007) to measure. |
| D-022 | 2026-08-10 | **`out_prev` is not a fourth piece of state, and the checkpoint stays exactly `f` / `solid` / `step_count` / config** (plus a `format: 1` integer so a future layout change is refused rather than misread). `load_checkpoint` rebuilds the convective outlet's previous column as `f[:, :, outlet_col]`. | D-021's convective outlet carries a `(9, ny)` column across steps, which naively breaks constraint 11's "`f`, `mask` and step count are the entire state". It does not: nothing after `outlet_zero_gradient` writes the outlet column — the inlet is a different column — so at the end of every step `out_prev` is byte-identical to `f[:, :, outlet_col]`. Pinned by `tests/test_runner.py::test_the_outlet_prev_column_equals_f_at_the_end_of_a_step`, asserted every step for 25 steps, and the 500/500/500 restart test runs **with the convective outlet on** so a wrong reconstruction fails on the first resumed step rather than silently drifting. The alternative — pickling the buffer — would have made the checkpoint's contents depend on which boundary conditions were enabled. |
| D-023 | 2026-08-10 | **`steps_per_frame(dt, fps=60.0, speed=1.0) -> max(1, round(speed / (fps * dt)))`**, with `dt` = *seconds of physical time per lattice timestep*, a plain scalar the caller supplies (`lbm/units.py`, T009). Grid size enters only through `dt`. | Constraint 7 wants the number computed from playback speed, grid size and `dt`; the convention forbids physical units inside `lbm/`. Taking `dt` as an already-converted scalar satisfies both — the runner does arithmetic on a number, not a unit conversion — and grid size enters exactly where it enters physically (refine by 2, `dt` halves, the same playback speed asks for twice the steps, which `test_steps_per_frame_halves_with_dt` asserts). Handing the function a `SimConfig` instead would have put a physical quantity into the config, which is the thing the convention exists to prevent. |
| D-024 | 2026-08-10 | **`run(..., drop=True)` drains the ring buffer into the sink from one consumer thread; `drop=False` drains inline.** The physics stays a single un-threaded loop. | Constraint 8 says the sim must never block on the display, and constraint 6 forbids optimising the physics. A thread on the *display* side is the only way a genuinely slow sink cannot stall the producer, and it is not an optimisation of the solver — collide and stream are untouched. Measured with a 4 ms sink against 2 ms of physics per frame: 60 frames pushed, **9 delivered, 51 dropped, all 120 steps executed**, wall clock 0.04 s against 0.026 s for the same 120 steps headless. `drop=False` is the other half of `DOCS/IDEA2.md` § Three output sinks — a fixed-framerate recorder (T011) must see every frame in order, and it is allowed to make the sim wait. |
| D-025 | 2026-08-12 | **`run(..., per_step=callable)`** — an optional probe called with `sim` after every timestep, on the physics thread. `validate/cylinder.py` samples `Cd`/`Cl` through it. | The `Cl` history has to be sampled at the **step** rate: the shedding period is ~2000 steps and one frame is 58 steps, so frame-rate sampling through `field` aliases. The alternative was a second copy of the loop in the validate script, which would have drifted from `run`'s ring-buffer and drop semantics the first time either changed. The hook defaults to `None` and `test_run_without_per_step_is_unchanged` asserts the two paths produce byte-identical `f`. |
| D-026 | 2026-08-12 | **Rung 3's lateral boundaries are periodic, not no-slip walls, and the fluid span is 24 D (4.2% blockage), well past constraint 12's 10% floor.** `validate/cylinder.py::WALL = 0`, `SPAN_D = 24`. | Measured, not argued. (a) With one-cell no-slip walls the free stream grows a boundary layer over the 8 D upstream fetch of thickness `~5 sqrt(nu x / U)` = **34 cells per wall**, so a *nominal* 9.5% blockage presents the cylinder with an effective ~13% and drag climbed straight through the band: `Cd` **1.49 → 1.58 → 1.64** at 5k/10k/15k steps on a 264x524 walled domain. Periodic sides have no boundary layer to grow, and `lbm.core.stream` is already periodic in `y`, so this is a deletion rather than a feature. (b) Blockage is then the only confinement left and it is not free either: at 15 D span (6.35%) the same case measured `Cd = 1.4635`, **1% over the top of the acceptance band**; at 24 D (4.17%) it measures **1.4031**. Constraint 12's 10% is a floor on the domain, not a target, and the reference value being compared against is an unconfined one. |
| D-027 | 2026-08-12 | **The `Cl` series is low-passed (Gaussian, `sigma = 0.5 D/U`) before the FFT, and only for the frequency.** The shedding-amplitude check reads the **raw** series. `validate/cylinder.py::lowpass`. | The force history carries the wake *and* the domain's acoustics: the impulsive start radiates a pressure pulse, the convective outlet absorbs 0.6% of it (D-021) but the Zou–He velocity inlet reflects essentially all of it. Measured `Cl` spectrum on the walled domain: wake peak at period 2500 steps, power **1347**; acoustic peak at period 305 steps, power **1378** — the acoustic one marginally taller, so the unfiltered FFT reported `St = 1.49` for a run whose wake was visibly shedding at the right rate. It is not noise to be tuned away, it is a different real oscillation: its period barely moved (308 → 305) when `U` changed 0.06 → 0.055, which no convected structure does. The cutoff is set by the case (`D/U`), not by the answer — the shedding period is ~6 D/U, so the filter costs the wake peak ~10% and the acoustic peak four orders of magnitude — and the amplitude criterion deliberately still reads the unfiltered series so a filtered-away wake cannot pass as shedding. |
| D-028 | 2026-08-12 | **`render` takes symmetric limits and refuses an asymmetric pair**; the colormap has **257** entries, not 256. | Constraint 9 asks for fixed symmetric limits and a diverging map, and both halves are enforced rather than documented: a lopsided `(vmin, vmax)` raises and names constraint 9, because on a diverging map it moves the neutral colour off zero and makes one sign of rotation look weaker than the other. The odd LUT length is the same argument at one-count resolution — with 256 entries zero falls between two of them and `+v`/`-v` are not mirror colours. `tests/test_render.py` asserts the mirror property and the byte-identical mapping across frames. |
| D-029 | 2026-08-12 | **D-016's `TAU_FLOOR = 0.53` is not a safe floor for a bluff body in a free stream.** `validate/polygons.py` enforces its own measured `TAU_FLOOR = 0.54` through `tau_for_rung4`, which wraps `validate.cylinder.tau_for` rather than replacing it. Rung 3 is **not** changed: it runs at `tau = 0.5378`, which is measured stable. | Measured, not argued, and it cost a full 26728-step run that reported `Cd = nan`. Bisected on a small stand-in domain, 60000 steps a leg: square at `tau` **0.5346** blew up by step 3200, square at 0.5378 survived (peak 0.1177), square at 0.5512 survived (peak 0.1114) — and a **disc** at `tau` 0.5330 blew up by step **1500**. The disc dying *sooner* than the square at the same `tau` is the decisive number: this is a relaxation-time limit, not a staircased-corner one, so constraint 1 is not implicated and no boundary-condition change would help. The threshold sits between 0.5346 and 0.5378; 0.54 is the next round number above it. Rung 3's floor is left alone because raising it would edit a passing benchmark from a task that does not own it — see § Open questions **Q-004**. |
| D-030 | 2026-08-12 | **Solid cells start at the rest equilibrium `w_i rho0`** in `validate/polygons.py` (`seed_solid_at_rest`), applied once at setup, never in the step loop. `Sim` itself is unchanged. | `Sim._init_equilibrium` seeds the whole domain with the equilibrium of the inlet profile, solid included, so at step 0 there is fluid moving at `U` *inside* the body. Bounce-back does not clear it — it writes `f[i] = f_pre[opp[i]]`, which **reverses** the momentum every step — so the interior oscillates at `±U` forever and T008's acceptance criterion ("no fluid velocity inside the solid") would have been measuring the initial condition rather than the boundary condition. Measured both ways: seeded, the interior is still **bit-identically** `w_i rho0` after 300 steps (the rest state is a fixed point of both bounce-back, since `W` is symmetric under `OPP`, and of streaming, since it is uniform); unseeded, it reads `|u| > 1e-3`. Both are tests. It is left out of `Sim` because changing the initial condition inside `lbm/` would perturb T006's bit-identical restart claim and Rung 3's published numbers for no physical gain — nothing on the fluid side of the surface depends on it. |
| D-031 | 2026-08-12 | **Closes Q-002. SVG needs no new dependency.** `lbm.geometry.from_svg` parses `M/L/H/V/C/Q/Z` (absolute and relative, Béziers flattened to 12 segments) plus `<polygon points=...>`, and fills the result with T004's even-odd `polygon`. Arcs (`A`), the smooth shorthands (`S`, `T`), `transform` attributes and anything unfillable **raise an `ImportError`** naming the feature and giving the `myenv/Scripts/pip.exe install cairosvg` line. Subpaths combine even-odd, so a donut has a hole. | The acceptance criterion asks for "at least simple closed paths" and a clear install message if a dependency is missing — it does not ask for a rasteriser. The subset above is ~150 lines on top of code T004 already had and covers what a user exports from a drawing tool as an outline; `cairosvg` would add a Cairo binary dependency to a project whose whole point is pure NumPy, for the half of T009 that M4 does not need (`DOCS/TASKS1.md` § T009 Notes). Refusing loudly rather than partially rendering is the load-bearing half: a silently dropped `transform` produces a plausible mask of the wrong shape, which is the failure mode `DOCS/IDEA2.md` § Validation ladder exists to prevent. Even-odd across subpaths differs from SVG's nonzero default only for self-intersecting or nested same-direction outlines; documented in the docstring as a limit. |
| D-036 | 2026-08-13 | **Closes Q-004. `validate/cylinder.py` grows its own `TAU_FLOOR = 0.537`**, replacing the 0.53 it inherited from Rung 2 (D-016). Rung 4's stricter measured 0.54 is unchanged, and the band 0.537 … 0.54 is now Rung 4's alone, pinned by `test_rung_4_keeps_its_own_stricter_floor_above_rung_3s`. | The floor as written admitted configurations **measured to produce `nan`**: D-029 bisected a square dying at `tau = 0.5346` by step 3200 and a disc at 0.5330 by step 1500, after a 26728-step run reported `Cd = nan`. 0.53 lets both through, and a benchmark that refuses a marginal case at setup rather than after half an hour is the whole point of having a floor. **0.54 was rejected as the answer**, even though it is the measured-safe round number, because Rung 3 itself runs at `tau = 0.5378`: a 0.54 floor makes the benchmark refuse the very run that produced its published `St` and `Cd`, so adopting it means re-tuning `D`/`U` and re-publishing Rung 3's numbers — a physics change, not a performance pass. The measured death threshold is bracketed by `(0.5346, 0.5378]`; 0.537 sits inside that bracket, above everything measured to die and below Rung 3's own measured-stable operating point. T010 re-ran every rung anyway, so the claim that nothing moved is measured rather than asserted. |
| D-035 | 2026-08-13 | **Benchmark protocol: variants are timed in alternating rounds with only one `Sim` resident, and the best round per variant is kept** (`bench.py::compare`). A/B measured by running all of A then all of B is not used, and the pre-change baseline file is kept as an audit artifact rather than as the comparison column. **An absolute steps/s number from this machine is only meaningful with the CPU clock quoted beside it: run straight after the 40-minute ladder, the passing build measured 375.2 / 107.6 / 14.2 and failed all three floors, with the CPU at 1882 MHz of a 3201 MHz maximum. Ratios survive throttling because both columns throttle together; absolute numbers do not.** | Measured, not argued. Two consecutive runs of the **identical** reference path differ by 12% at 400×100 (726.8 vs 811.5 steps/s) and 21% at 800×200 (146.1 vs 177.2) — larger than every effect this task set out to measure, so sequential A/B awards the win to whichever variant ran during the quieter minute. The first cross-process comparison did exactly that and reported the fusion as a **0.85× regression**; alternating rounds put it at 1.00–1.14×. Co-residency was tried as the fix and is worse: holding both variants' buffers at once (~500 MB at 1M cells) dropped *both* to ~16 steps/s from ~21, because a cache-locality change cannot be measured under cache pressure. Best-of-N rather than mean for the same reason session 7 saw Rung 3 take 368.9 s alone and 633.3 s under contention: the slow samples measure Windows, not the kernel. |
| D-037 | 2026-08-13 | **The preallocation audit's one fix — `inlet_velocity` taking a precomputed `fluid` mask — is kept even though it is not measurable.** | Honesty about a number that is not there. Session 6 flagged `~solid[:, col]` as "the one allocation left in the step loop", so T010 closed it; measured separately, it saves **0.37 us of the 79 us** `inlet_velocity` costs at `ny = 500` (0.5% of that function) and nothing detectable at step level (0.996× / 1.054× / 1.012×, which is noise). It stays because the convention it satisfies — "preallocate, never allocate inside the step loop" — is about the loop being *provably* allocation-free rather than about the microseconds, and because `tracemalloc` growth tests structurally cannot see a transient that is freed each step. It is a correctness-of-claim fix, not a performance one, and the § Performance baseline table should not be read as if it contributed. |
| D-034 | 2026-08-13 | **"Skip `feq` on solid cells" is measured and deliberately not shipped.** `lbm.core.equilibrium` stays dense. The contract lists it as one of the four cheap wins; in NumPy it is not a win. | Measured, not argued. A `where=fluid` variant of exactly the same arithmetic runs **2–4× slower** than the dense one at every grid: 400×100 0.24×, 800×200 0.23×, 2000×500 0.42×, and at a deliberately huge 15.9% solid fraction it is still 0.19× / 0.22× / 0.50×. The mechanism is visible one level down — a masked `np.multiply` costs 0.060 ms against 0.018 ms dense, 3.3×, because the mask has to be read and the SIMD loop is broken, while the arithmetic it skips is the cheap part of a memory-bound kernel. Constraint 12 caps blockage at 10%, so the fraction that could ever be skipped is small by construction and the case that would pay does not exist in this project. Shipping it behind a toggle was rejected under `DOCS/TASKS1.md` § T010 Notes: ~25 lines of duplicated equilibrium for a *negative* speed. The measurement is the deliverable. |
| D-033 | 2026-08-13 | **`lbm.core.collide_stream` fuses collide + bounce-back + the `f_bb` snapshot + stream into one pass per direction**, and `SimConfig.fused` (default `True`) selects it. The unfused T009 sequence stays selectable and is what the equality tests compare against. A body force falls back to the unfused path. | `DOCS/IDEA2.md` § Performance budget asks to "fuse collide+stream into one pass over `f`", but D-020's order puts `bounce_back` **between** them, so fusing only the two named steps removes nothing — the array is still walked for the reflection and again for the `f_bb` copy. Fusing across all four is what removes the passes, and it removes the `f_bb` snapshot copy outright. Measured, alternating rounds (D-035): **1.00× at 40k, 1.01× at 160k, 1.14× at 1M** — worth keeping precisely where the budget is tightest (the 1M floor is 15 and the reference path measures 15.0), and never a loss. The arithmetic is unchanged element by element and in order, so both paths are **bitwise** equal — asserted on `f`, on fluid cells alone, on the `f_bb`/`f` snapshots `probe.forces` consumes, and by a checkpoint written on one path and resumed on the other (`tests/test_perf.py`), which is what keeps constraint 11 true. The Guo body force is **not** folded in: its source term goes between collision and bounce-back, and the only case that uses it is Rung 1's 22×16 channel, where speed is irrelevant and the reference arithmetic is worth more than the pass. |
| D-032 | 2026-08-12 | **`lbm/units.py` enforces `tau > 0.51` and `U < 0.1` and nothing stricter**, and ships `LatticeUnits.stability_note()` alongside. It is the third `tau` floor in the project, below Rung 2's 0.53 (D-016) and Rung 4's measured 0.54 (D-029). | D-029 measured a disc dying at `tau = 0.5330` by step 1500, so 0.51 is demonstrably **not** a stability guarantee — but the other two floors were measured on *bluff bodies in a free stream*, and this module converts units for cases it has never seen (a channel at `tau = 0.52` is fine). Refusing everything under 0.54 would reject valid configs; accepting silently would imply a safety the number does not have. So the floor is a floor on nonsense, the rejection message quotes D-029's measured deaths, and `stability_note()` grades the margin for the geometry the caller knows about and the module does not. `BLUFF_BODY_SPEEDUP = 1.8` (session 8's measured `peak = 1.79 U`) is exposed the same way, so the constraint-3 headroom is a number rather than folklore. |

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

### 2026-08-10 — Session 5: T005 — inlet / outlet BC + probes

**Task worked:** T005 — `done`, every acceptance criterion run and green. No milestone (M3 is T007).

**Done**
- `lbm/boundary.py` — added `inlet_profile(ny, U, profile, *, solid, col, uy)` (uniform / parabolic,
  D-009 halfway wall convention, warns at `|u| >= 0.1`), `inlet_velocity(...)` (**Zou–He** on the
  three `ex = +1` unknowns, allocation-free with a cached `u_in` and a `(5, ny)` work buffer, skips
  solid rows) and `outlet_zero_gradient(f, *, col, src, prev, lam)` (copy by default, convective when
  given `prev`). Module docstring now carries the full timestep order including the D-020 tail.
- `lbm/probe.py` — new. `vorticity(u, *, solid, out, work)`, `BoundaryLinks` +
  `boundary_links(solid)`, `forces(f_pre, f_post, solid, *, U, D, rho0)`,
  `strouhal(cl_series, dt, D, U, *, transient)`, `residual(u_now, u_prev, U, *, solid, work)`.
- `lbm/__init__.py` — re-exports the eleven new public names.
- `tests/test_probe.py` — 49 tests covering both BCs and all four probes.

**Measured**
- `myenv/Scripts/python.exe -m pytest tests/test_probe.py` → **`49 passed`**
- `myenv/Scripts/python.exe -m pytest` → **`152 passed in 5.41s`** (103 existing + 49 new)
- `myenv/Scripts/python.exe -m validate.poiseuille` → **PASS**, L2 **0.3650%**, peak `|u|` 0.07955
- `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000` → **PASS**, max deviation
  **0.75% / 0.42% / 1.01%**, vortex 0.21 / 0.29 / 0.59 cells — both rungs identical to session 4.
- Outlet reflection of a `sigma = 10` Gaussian pressure pulse: plain copy **35%**, `lam = 0.4` 4.7%,
  `lam = cs` **0.6%**, `lam = 1.0` 7.5%, `lam = 2 cs` 17% → **D-021**.
- `forces` magnitude audit by Poiseuille momentum balance: wall drag `8.5319e-03` against injected
  `gx * A_fluid = 8.5334e-03`, **ratio 0.99982** (0.02% error), transverse force `1.1e-06`.
- Sign/symmetry harness (41x140 channel, `D = 9` cylinder, `tau = 0.65`, 1200 steps, last 200
  averaged): centred `Cd = 27.3`, `Cl = 1.5e-05`; offset `Cl = +0.0531` against mirrored `-0.0531`.
  `Cd = 27` is not a benchmark number and is not meant to be — that case is Re ~ 9 at 23% blockage,
  well outside what `check_mask` allows. Rung 3 (T007) is the run that measures `Cd ~ 1.34`.

**Not done / deferred**
- Nothing from the T005 contract. No cylinder run and no `validate/cylinder.py` — Rung 3 is T007 and
  the contract says so explicitly.
- No sponge layer. `DOCS/IDEA2.md` § Stability offers "zero-gradient **or** sponge"; a tau-ramp
  sponge was measured (it only improved the copy outlet from 26% to 21% reflection) and dropped in
  favour of the convective boundary, which is one line and ten times better.
- No optimisation (constraint 6). `forces` fancy-indexes per direction, which allocates a small
  temporary per call; the link list is built once, which is what the contract asked for.

**Decisions made**
- **D-020** (which two snapshots `forces` takes, and the extra runner buffer — the API question the
  contract told this session to settle for T006), **D-021** (convective outlet, measured). Both above.

**Two things measurement changed**
- The bare column-copy outlet **fails** the 5% criterion at 35%. Fixed by the convective form rather
  than by relaxing the number; the copy's 35% is now itself a test, so the fix cannot be undone by
  someone tidying up.
- The contract's `forces` validation (uniform flow, no obstacle, `|Cd| < 1e-6`) is trivially true —
  with no obstacle there are no links and the sum is empty. It is kept, and a **Poiseuille momentum
  balance** test was added next to it, which does audit the magnitude: 0.02% against the exact
  answer. Criterion strengthened, not relaxed; the row in `DOCS/TASKS1.md` says so.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜. R3 and R4 still have no script (T007, T008) — not attempted, not
  failing.

**Next**
- Paste `PROMPTS/006-t006-runner-ringbuffer-restart.md` into a fresh session. It runs
  `/start-task T006`. The runner's buffer list is fixed by D-020 (the extra pre-stream `f_bb`, the
  `(9, ny)` outlet `prev`, the cached `u_in` and its `(5, ny)` work array); constraint 11's
  bit-identical restart is the criterion most likely to bite, and `float64` creeping into the step
  path is how it breaks.

### 2026-08-10 — Session 6: T006 — runner, ring buffer, bit-identical restart

**Task worked:** T006 — `done`, every acceptance criterion run and green. No milestone (M3 is T007).

**Done**
- `lbm/runner.py` — new. `SimConfig` (scalars only, so a checkpoint carries it verbatim); `Sim`,
  which owns `f`, `solid`, `step_count` and **every** buffer (`f_pre` pre-collision, `f_bb`
  pre-stream, `buf`, `rho`, `u`, `u_prev`, `feq`, `work`, `omega`, `vort_work`, `res_work`,
  `out_prev (9, ny)`, `inlet_work (5, ny)`, cached `u_in`, and `links` built once); `Sim.step()` in
  the D-020 order; buffer-owning accessors `vorticity()`, `forces()`, `residual()`/`mark_residual()`;
  `steps_per_frame(dt, fps, speed)`; `RingBuffer(maxlen)` with `dropped`/`pushed`; `Sink` (ABC) and
  `NullSink`; `save_checkpoint` / `load_checkpoint`; `run(...)` with `RunStats`.
- `lbm/__init__.py` — re-exports the ten new public names.
- `tests/test_runner.py` — 46 tests, one or more per acceptance criterion.

**Measured**
- `myenv/Scripts/python.exe -m pytest tests/test_runner.py` → **`46 passed in 3.25s`**
- `myenv/Scripts/python.exe -m pytest` → **`198 passed in 8.14s`** (152 existing + 46 new)
- Allocation: `f.__array_interface__['data']` unchanged over 1000 steps; `tracemalloc` heap growth
  over 1000 steps **< 20 kB** (one `(9, 24, 80)` `float32` buffer alone is ~69 kB).
- **Bit-identical restart:** 500 steps → checkpoint → 500 more → reload → 500 →
  `np.array_equal` **True**, on three configs (Zou–He inlet + convective outlet + cylinder; Guo body
  force; plain-copy outlet). Auto-checkpoints resume identically too.
- Slow-sink drop test (4 ms sink, ~0.2 ms of physics per frame, `RingBuffer(2)`): 60 frames pushed,
  **9 delivered, 51 dropped, all 120 steps executed**; 0.04 s wall clock against 0.026 s for the same
  120 steps headless. `step_count` is exactly `frames * steps_per_frame`.
- `steps_per_frame(5e-4, 60) = 33`; halving `dt` gives 67 — the number tracks the grid, not a
  hardcoded 20.
- Runner vs Rung 1's hand-rolled loop, 400 steps, same mask and `tau`: `np.array_equal` **True** —
  the runner's timestep order is byte-for-byte the order the passing rung uses.
- `myenv/Scripts/python.exe -m validate.poiseuille` → **PASS**, L2 **0.3650%**, peak `|u|` 0.07955.
- `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000` → **PASS**, max deviation
  **0.75% / 0.42% / 1.01%**, vortex 0.21 / 0.29 / 0.59 cells. Both rungs identical to session 5.

**Not done / deferred**
- Nothing from the T006 contract. No `render()`, no live sink, no cylinder run — T007/T011 by the
  contract, and the ring buffer was proven with a fake slow sink instead, which is what the contract
  asked for.
- `Sim` has no moving-lid path, so `validate/cavity.py` keeps its own loop. Rung 2 is a closed,
  passing benchmark and rewriting it onto the runner buys nothing this session; the Rung 1
  equivalence test is what pins the shared order.
- No optimisation (constraint 6). Note for T010: `lbm.boundary.inlet_velocity` allocates a small
  `(ny,)` boolean (`fluid = ~solid[:, col]`) per call — transient, freed each step, invisible to the
  growth test, but it is the one allocation left in the loop.

**Decisions made**
- **D-022** (`out_prev` reconstructed from `f`, so the checkpoint stays four things),
  **D-023** (`steps_per_frame` signature and where `dt` comes from), **D-024** (threaded consumer for
  `drop=True`, inline drain for `drop=False`). All above.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜. R3 and R4 still have no script (T007, T008) — not attempted, not
  failing.

**Next**
- Paste `PROMPTS/007-t007-render-live-sink-cylinder.md` into a fresh session. It runs
  `/start-task T007`. `pygame` must be installed into `myenv` first and a row added to § Environment.
  T007 is **M3** and the first rung since session 3; `DOCS/PLAN1.md` § Risks flags "cylinder shows no
  shedding" — session 4 measured that a `D = 21` cylinder in a 121-row channel is 17.6% blockage and
  `check_mask` warns, so Rung 3 needs roughly `ny >= 10 D` plus `>= 8 D` downstream.

### 2026-08-12 — Session 7: T007 — render, live sink, cylinder benchmark → Rung 3

**Task worked:** T007 — `done`, every acceptance criterion run and green. **M3 reached.**

**Done**
- `lbm/render.py` — new. `colormap(n)` / `COOLWARM` (257-entry diverging cool-warm LUT, built once
  at import), `NAN_RGB`, `render(field, limits, *, lut, nan_rgb, out)` and `LiveSink(Sink)`.
  `render` takes the limits and never looks at the data range; a lopsided `(vmin, vmax)` raises
  naming constraint 9. `LiveSink` imports `pygame` inside its methods, opens the window lazily on
  the first `push` (so every SDL call is on the ring buffer's consumer thread), pumps events,
  exposes `quit_requested`, and blits — it colours nothing.
- `lbm/runner.py` — `run(..., per_step=...)` added (**D-025**), the only change to T006's code.
- `lbm/__init__.py` — re-exports the five new public names.
- `validate/cylinder.py` — new, Rung 3. `cylinder_mask` (disc **plus** a separately returned
  cylinder-only mask for the force integral), `tau_for` (refuses `tau <= 0.53` and `U >= 0.1` at
  setup, naming the fix), `lowpass`, `bench_steps_per_second`, `run_cylinder`, `report`, `main`
  with `--headless`.
- `tests/test_render.py` (22 tests) and `tests/test_cylinder.py` (8 tests); two more in
  `tests/test_runner.py` for the `per_step` hook.
- `pygame` 2.6.1 installed into `myenv`; § Environment row added.

**Measured**
- `myenv/Scripts/python.exe -m validate.cylinder` (gate, **window open**) → **PASS**.
  504 x 440, `D = 21` cells measured, periodic sides, blockage **4.17%**, 11.95 D downstream,
  `tau = 0.5378`, `U = 0.06`, 45500 steps in 368.9 s (123 steps/s), 785 frames, 0 dropped.
  - **St 0.1731** (band 0.155–0.175, ref 0.164, +5.5%) — shedding period 2022 steps
  - **Cd 1.4031 ± 0.0086** (band 1.25–1.45, ref 1.34, +4.7%)
  - Cl amplitude **0.3915** = 27.9% of Cd, mean **-0.0040** (the startup kick left nothing behind)
  - peak `|u|` **0.09685** (limit 0.1)
  - **window cost 129.9 → 132.6 steps/s, +2.09%** (limit 10%)
- `--headless` on the same defaults: byte-identical St, Cd, Cl and peak `|u|`; 345.1 s, 132 steps/s.
- `myenv/Scripts/python.exe -m pytest` → **`230 passed`** (198 existing + 32 new).
- `myenv/Scripts/python.exe -m validate.poiseuille` → **PASS**, L2 **0.3650%**, peak 0.07955.
- `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000` → **PASS**, max deviation
  **0.75% / 0.42% / 1.01%**, vortex 0.21 / 0.29 / 0.59 cells. Both identical to session 6.

**Three things measurement changed** — none of them a relaxed tolerance
- **The force integral was measuring the channel.** `Sim.links` is built from the whole mask, so the
  first Rung 3 run reported the walls' friction plus the body's drag: `Cd` **6.65** against the
  body's **1.57**. Fixed by integrating over a cylinder-only link list; `tests/test_cylinder.py`
  pins it with an explicitly walled mask so it cannot come back.
- **No-slip walls made the domain lie about its blockage** (**D-026**). The boundary layer over the
  8 D upstream fetch is ~34 cells per wall, turning a nominal 9.5% blockage into an effective ~13%
  and taking `Cd` to 1.64. Periodic sides removed it; the span then went 15 D → 24 D because 6.35%
  blockage still read `Cd = 1.4635`, 1% over the band.
- **The FFT was locking onto the domain's acoustics** (**D-027**). Wake peak at period 2500 (power
  1347) against an acoustic peak at period 305 (power 1378) — the unfiltered answer was
  `St = 1.49` for a wake that was visibly shedding correctly. Diagnosed by changing `U`: the
  acoustic period moved 308 → 305 while the wake's moved with the flow.

**Not done / deferred**
- Nothing from the T007 contract. No recording sinks and no CLI (T011), no square cylinder (T008),
  no PNG/units path (T009) — all named in the contract as other tasks.
- No optimisation (constraint 6), which **lifts now that Rung 3 is green** — T010 owns it. Note for
  it: 504 x 440 = 222k cells runs at ~130 steps/s, and `validate/cylinder.py` costs an extra
  `forces` call per step on top.
- `lowpass` lives in `validate/cylinder.py`, not `lbm/probe.py`. It is a property of *this domain's*
  acoustics, not of the method; promoting it would make every future caller inherit a filter it may
  not want.

**Decisions made**
- **D-025** (`per_step` probe hook), **D-026** (periodic sides + 24 D span, measured),
  **D-027** (low-pass before the frequency estimate only, measured), **D-028** (symmetric limits
  enforced, 257-entry LUT). All above.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 🟩 · R4 ⬜. R4 has no script yet (T008) — not attempted, not failing.

**Next**
- Paste `PROMPTS/008-t008-square-cylinder.md` into a fresh session. It runs `/start-task T008`.
  Rung 4 reuses `validate/cylinder.py`'s domain sizing wholesale — D-026's periodic sides and 24 D
  span, and the cylinder-only force links — with `lbm.geometry.regular_polygon` in place of
  `circle`. The staircase corners are the expected answer (constraint 1); the band is ±0.1 for
  exactly that reason.

### 2026-08-12 — Session 8: T008 — square cylinder + convex polygon → Rung 4

**Task worked:** T008 — `done`, every acceptance criterion run and green. **The validation ladder is
complete.** No milestone of its own (M4 is T011).

**Done**
- `validate/polygons.py` — new, Rung 4. `square_body` / `convex_body` (thin wrappers over T004's
  `regular_polygon` and `polygon` — **no new geometry code**), `POLY_VERTS`, `Case` + `cases()`,
  `body_mask`, `tau_for_rung4`, `seed_solid_at_rest`, `interior_solid`, `run_case`, `report`, `main`
  with `--headless` / `--case {square,polygon,both}`. Rung 3's setup is imported, not copied:
  `tau_for`, `lowpass`, `make_config`, and the D-026 / D-021 / D-025 / D-027 constants all come from
  `validate.cylinder`.
- `tests/test_polygons.py` — 21 tests: both masks pass `check_mask` silently, the three numeric
  rules, the square is square *and its bbox is full*, `POLY_VERTS` is convex and asymmetric, both
  factories equal the primitives they claim to wrap, `tau` from `Re` alone, the contract band is
  1.4–1.6, the marginal-`tau` refusal, the corner criterion and the rest-state interior, the
  no-seed control, and the walls-in-the-link-list trap re-pinned for Rung 4's mask builder.
- **No change to anything under `lbm/`.** T008 needed no solver code at all.

**Measured**
- `myenv/Scripts/python.exe -m validate.polygons --headless` → **PASS**.
  - **square** — 744 x 557, `D = 30` measured, periodic sides, blockage **4.03%**, 9.30 D
    downstream, `tau = 0.5477`, `U = 0.053`, 73585 steps in 1358.6 s (54 steps/s), 783 frames, 0
    dropped. **Cd 1.5279 ± 0.0271** (band 1.4–1.6, ref 1.5, **+1.9%**), **St 0.1489** (ref 0.145),
    Cl amplitude **0.6510** = 42.6% of Cd, mean Cl +0.0043, peak `|u|` **0.09758**.
  - **polygon** — `D = 29`, 38302 steps in 595.0 s. **Cd 1.4276 ± 0.0226**, Cl amplitude 0.3689,
    St 0.1667, peak `|u|` 0.08944, no `nan`. No reference value asserted, per the contract.
- `myenv/Scripts/python.exe -m pytest` → **`251 passed`** (230 existing + 21 new).
- `myenv/Scripts/python.exe -m validate.poiseuille` → **PASS**, L2 **0.3650%**, peak 0.07955.
- `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000` → **PASS**, max
  deviation **0.75% / 0.42% / 1.01%**, vortex 0.21 / 0.29 / 0.59 cells.
- `myenv/Scripts/python.exe -m validate.cylinder --headless` → **PASS**, **St 0.1731**,
  **Cd 1.4031 ± 0.0086**, peak 0.09685 — identical to session 7 to every printed digit.

**Three runs, three different constraints** — the corners were never the problem
- The staircase was fine on the first run that stayed finite: `Cd = 1.5323` at `D = 27`, +2.2%.
  What took three runs was the **case setup**, and each failure was a different constraint:
  1. `U = 0.06` (Rung 3's inlet) gives peak `|u|` **0.10211** on a square against the disc's 0.09685
     — constraint 3. A square blocks more, so the flow accelerates further round it; the measured
     ratio is **peak = 1.79 U** over a full run (a 20 D/U look-ahead reads 1.70 and is optimistic,
     which is what cost the third run).
  2. Dropping to `U = 0.055` took `tau` to 0.5346 and the sim produced `nan` — **D-029**. D-016's
     0.53 floor is not safe for a bluff body in a free stream, and the *disc* at `tau` 0.5330 blows
     up **sooner** (step 1500) than the square at 0.5346 (step 3200), which is what rules out the
     corners and with them constraint 1.
  3. `U = 0.056, D = 27` was stable and correct and still failed, at peak `|u|` 0.10031.
- `U = 0.053, D = 30` clears both. The two constraints pull against each other and `D` is the only
  knob that buys `tau` without moving the peak — at a cost that grows with the area of the domain.

**Not done / deferred**
- Nothing from the T008 contract. No physical units or PNG/SVG (T009), no performance pass (T010),
  no recording sinks or CLI (T011). No live-window run: T008 sets no window criterion, and T007
  already measured `--headless` to be byte-identical.
- No optimisation, though constraint 6 is lifted. Note for T010: 414k cells runs at ~54–64 steps/s,
  and this rung's `per_step` probe costs an extra `forces` call per step on top.
- `validate/cylinder.py::tau_for` keeps its 0.53 floor — see **Q-004**. Editing a passing benchmark
  was out of scope; Rung 3 at 0.5378 is measured stable, so nothing is red.

**Decisions made**
- **D-029** (Rung 4's measured `TAU_FLOOR = 0.54`, and why the disc's earlier death rules out the
  corners), **D-030** (solid cells seeded at rest, and why it stays out of `lbm/`). Both above.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **the ladder is complete.** Every rung has been run in this
  session and every one is green.

**Next**
- Paste `PROMPTS/009-t009-units-png-mask.md` into a fresh session. It runs `/start-task T009`.
  Q-002 (the SVG rasteriser) has to be answered there; PNG is what M4 actually needs, so SVG is the
  half to cut if the session runs long. T009's own acceptance criterion — the cylinder reproduced
  through the units path within 2% of T007's `Cd` — means Rung 3 gets re-run, and its numbers are
  above to compare against.

### 2026-08-12 — Session 9: T009 — physical units + PNG / SVG mask

**Task worked:** T009 — `done`, every acceptance criterion run and green. No milestone (M4 is T011,
which this task unblocks).

**Done**
- `lbm/units.py` — new, and the **only** module in `lbm/` that holds metres or seconds.
  `LatticeUnits` (frozen dataclass: `dx`, `dt`, `tau`, `U`, `Re`, `cells_per_length`, and the three
  physical inputs) with `from_physical(...)`, the conversions
  `to_lattice_/to_physical_ velocity|length|time|viscosity`, `reynolds()` (the round trip),
  `resolution_for_tau`, `peak_velocity_estimate`, `stability_note()` and `summary()`. Module
  docstring derives `Re -> dx -> dt -> nu -> tau` in four steps and states what is refused and why.
  Constants `U_LATTICE_MAX = 0.1`, `TAU_FLOOR = 0.51`, `U_LATTICE_DEFAULT = 0.05`,
  `BLUFF_BODY_SPEEDUP = 1.8`.
- `lbm/geometry.py` — `from_png(path, shape, ...)` (alpha when present and non-uniform, luminance
  fallback, box-filter resample **then** threshold, `flip_y` so a picture loads upright against
  `LiveSink`'s row-0-at-bottom, `fit="stretch"|"contain"`, and `check_mask` run automatically) and
  `from_svg(path, shape, ...)` (built-in `M/L/H/V/C/Q/Z` + `<polygon>` parser, Béziers flattened,
  even-odd across subpaths, `check_mask` automatic). Private helpers `_fit_to_grid`,
  `_parse_path_d`, `_parse_svg`, `_flatten_bezier`, `_svg_dependency_error`. **No new dependency**
  (D-031); Pillow was already present.
- `lbm/__init__.py` — re-exports `LatticeUnits`, the three units constants, `from_png`, `from_svg`.
- `validate/cylinder.py` — added `--physical` (and `run_cylinder(physical=...)`), which derives
  `tau`, the lattice `U` and `dt` through `LatticeUnits.from_physical` instead of `tau_for`. **Off
  by default**, so Rung 3's published numbers still come from the code that produced them; the
  default path is otherwise untouched.
- `tests/test_units.py` — 57 tests. `tests/data/test_body.png` — committed, **959 bytes**, generated
  by `write_test_png()` *in that test file*, so the expected solid-cell count is derived arithmetic
  (`pi r^2 + w h`, scaled) rather than a magic number; a test asserts the committed bytes match a
  fresh regeneration.

**Measured** — every acceptance criterion run, not read
- `LatticeUnits.from_physical` returns `dx`, `dt`, `tau`, lattice `U` and `Re`; derivation is in the
  module and method docstrings.
- Round trip: `reynolds()` (which goes through `tau`, `U` and the resolution only — no metres)
  reproduces `Re` within **1e-3 relative** on four cases spanning Re 80 … 1000, criterion 0.1%.
- Rejections, both with the fixing number in the message: `u_lattice = 0.1` raises naming the
  lattice velocity and constraint 3; `Re 1000, U 0.05, N 40` gives `tau = 0.5` and raises naming
  `tau` and `cells_per_length >= 134`, and running at the suggested 134 clears the floor. A test
  parses the number out of the message and re-runs with it.
- `tests/data/test_body.png` on a 128x128 grid: **3387 solid cells against 3416.99 expected,
  −0.88%** (criterion ±2%). A 97x97 non-integer downscale is inside 2% too.
- `from_png` runs `check_mask` itself: the committed image is 50% blockage and warns without the
  caller asking; `strict=True` raises. A 6-px bar downscaled 4x reads `min_thickness` **1 < 3** and
  warns — the hairline case constraint 12 exists for.
- `from_svg`: a `M/L/Z` square, the same square as `m/h/v/z` (byte-identical masks), the same square
  as `<polygon>` (byte-identical), a cubic-Bezier circle within **5%** of `pi r^2`, and a two-subpath
  donut with a hole. Arcs and `transform` raise `ImportError` naming `cairosvg`.
- **Cylinder through the units path** — `myenv/Scripts/python.exe -m validate.cylinder --headless
  --physical` → **PASS**: **Cd 1.4031 ± 0.0086**, **St 0.1731**, Cl amplitude 0.3915, peak `|u|`
  0.09685, 45500 steps, 633.3 s. T007's `Cd` is **1.4031** — identical to every printed digit,
  **0.00%** against a 2% criterion, because `tau` and `dt` come out of `LatticeUnits` equal to
  `tau_for`'s to float precision (pinned in milliseconds by
  `test_the_physical_path_sets_rung_3_up_identically`).
- **The ladder, all four, re-run this session:**
  - R1 `validate.poiseuille` → **PASS**, L2 **0.3650%**, peak `|u|` 0.07955
  - R2 `validate.cavity --re 100 --re 400 --re 1000` → **PASS**, max deviation
    **0.75% / 0.42% / 1.01%**, vortex 0.21 / 0.29 / 0.59 cells
  - R3 `validate.cylinder --headless` → **PASS**, **St 0.1731**, **Cd 1.4031 ± 0.0086**, peak 0.09685
  - R4 `validate.polygons --headless` → **PASS**, square **Cd 1.5279** (+1.9% vs 1.5), St 0.1489,
    peak 0.09758; polygon **Cd 1.4276 ± 0.0226**, St 0.1667, peak 0.08944
  - Every number identical to session 8. Nothing regressed.
- `myenv/Scripts/python.exe -m pytest` → **`308 passed in 11.62s`** (251 existing + 57 new).

**One thing measurement changed**
- The first `summary()` printed `peak |u| ~ 0.1080 (limit 0.1)` for Rung 3 — a case whose *measured*
  peak is 0.09685 and which passes. The estimate is `BLUFF_BODY_SPEEDUP * U` with the **square**
  cylinder's 1.79 applied to a disc, so it reads as a violation of constraint 3 on a config that
  does not violate it. Reworded to `peak |u| <= ...` and explicitly labelled an upper bound whose
  arbiter is the run's own measured peak. The number was not softened — only stopped from lying
  about what it is.

**Not done / deferred**
- Nothing from the T009 contract. All eight criteria are green, SVG included, so the `/new-task`
  escape hatch in § Notes ("if SVG drags, ship PNG") was not needed.
- No demo script putting a PNG body in a flow. Not in the contract; the CLI is **T011**, and D-030's
  `seed_solid_at_rest` is where that work should start when it happens.
- No optimisation (T010). Geometry and unit conversion are setup code and run once.
- SVG's `fill-rule` is even-odd across subpaths where SVG's default is nonzero. They differ only for
  self-intersecting or nested same-direction outlines; documented in the `from_svg` docstring as a
  known limit rather than fixed, since the fix is a full rasteriser (**D-031**).
- `validate/cylinder.py::tau_for` still has its 0.53 floor — **Q-004**, still open, still not this
  task's to edit. `lbm/units.py`'s 0.51 is a *third* floor and says so in its docstring (**D-032**).

**Decisions made**
- **D-031** (SVG needs no new dependency — closes **Q-002**), **D-032** (`lbm/units.py` enforces
  0.51 and not 0.54, plus `stability_note()` and `BLUFF_BODY_SPEEDUP`). Both in § Decisions.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — the ladder is complete and every rung was re-run here.

**Next**
- Paste `PROMPTS/010-t010-performance-pass.md` into a fresh session. It runs `/start-task T010`.
  Constraint 6 is lifted (Rung 3 has been green since session 7). The budget table in
  `DOCS/STATE1.md` § Performance baseline is still empty and T010 fills it. Known starting points:
  504x440 = 222k cells runs at ~130 steps/s, 414k cells at ~54–64 steps/s, and
  `lbm.boundary.inlet_velocity` allocates a `(ny,)` boolean per call — the one allocation left in
  the step loop. T010 re-runs **all four** rungs, and `DOCS/PLAN1.md` § Risks says revert rather
  than debug a fused kernel.

### 2026-08-13 — Session 10: T010 — performance pass

**Task worked:** T010 — `done`, every acceptance criterion run and green. No milestone (M4 is T011).

**Done**
- `bench.py` — new, repo root. `GRIDS` / `STEPS` / `FLOORS` / `BUDGET`, `make_sim`, `measure`
  (best-of-N), `compare` (alternating rounds, one `Sim` resident — **D-035**), `print_table`,
  `print_variants`, and `--save-baseline` / `--variants`. Benchmarks the **whole** `Sim.step`
  (Zou–He inlet, convective outlet, immersed disc), not `collide` in isolation.
- `lbm/core.py` — new `collide_stream(f, feq, tau, buf, *, f_pre, solid, f_bb)`: collide,
  bounce-back, the `f_bb` snapshot and the shift, **one pass per direction** (**D-033**). The
  unfused sequence is untouched and still exported.
- `lbm/runner.py` — `SimConfig.fused` (default `True`); `Sim` gained `_fused`, `_has_solid` and
  `_inlet_fluid` (the precomputed inlet row mask); `Sim.step` branches to the fused kernel unless a
  body force is present, in which case the T009 sequence runs so Rung 1's arithmetic is untouched.
- `lbm/boundary.py` — `inlet_velocity(..., fluid=None)`, closing session 6's "one allocation left in
  the step loop" (**D-037**).
- `validate/cylinder.py` — `TAU_FLOOR = 0.537` as a documented module constant, replacing the
  inherited literal `0.53` inside `tau_for`; the rejection message now quotes D-029's measured
  deaths. **Closes Q-004** (**D-036**). One stale docstring paragraph (`D = 24`, `254 x 552`, 9.5%
  blockage) corrected — it had described the pre-D-026 walled domain.
- `tests/test_perf.py` — new, 18 tests: fused vs unfused bitwise on `f`, on fluid cells alone, on the
  two force snapshots, with an empty domain and with a body force; `collide_stream` against
  `collide`+`bounce_back`+`stream` directly; buffer identity; restart bit-identical on the fused path
  **and across the toggle**; the `float32` audit; the preallocation checks.
- `tests/test_cylinder.py` (+2), `tests/test_polygons.py` (+1) — pin the new floor, that it still
  admits Rung 3's own `tau = 0.5378`, and that the 0.537–0.54 band is Rung 4's alone.
- `CLAUDE.md` § Current state — refreshed to T010.

**Measured** — every acceptance criterion run, not read
- **Baseline, captured before the first edit:** **739.9 / 182.6 / 17.4** steps/s at 40k / 160k / 1M,
  archived in `DOCS/bench_baseline.json`.
- **After:** **696.7 / 161.7 / 16.8** against floors of 400 / 120 / 15 — all cleared, at the CPU's
  rated clock. See § Performance baseline for the power-state caveat, which is large.
- Each win separately: **fusion** 1.00× / 1.01× / **1.14×** · **skip `feq` on solid** 0.24× / 0.23× /
  0.42× (0.19×–0.50× even at 15.9% solid) — a 2–4× **loss**, not shipped · **preallocation** 0.37 µs
  of 79 µs in `inlet_velocity`, nothing at step level · **`float32`** already end to end, now proven:
  **125 ufunc results in a real timestep, every one `float32`**.
- Where the time goes at 1M cells: `equilibrium` **39.9 ms**, `collide` 15.1, `macroscopic` 7.2,
  `stream` 6.1, the snapshot copies 5.8, `bounce_back` 0.8.
- **The ladder, all four, re-run on the optimised build:**
  - R1 `validate.poiseuille` → **PASS**, L2 **0.3650%**, peak `|u|` 0.07955
  - R2 `validate.cavity --re 100 --re 400 --re 1000` → **PASS**, **0.75% / 0.42% / 1.01%**,
    vortex 0.21 / 0.29 / 0.59 cells
  - R3 `validate.cylinder --headless` → **PASS**, **St 0.1731**, **Cd 1.4031 ± 0.0086**, peak
    0.09685, 45500 steps in 359.7 s (127 steps/s)
  - R4 `validate.polygons --headless` → **PASS**, square **Cd 1.5279 ± 0.0271**, St 0.1489, peak
    0.09758; polygon **Cd 1.4276 ± 0.0226**, St 0.1667, peak 0.08944
  - **Every number identical to session 9 to every printed digit** — which is the fusion's real
    acceptance test, not the stopwatch.
- `myenv/Scripts/python.exe -m pytest` → **`329 passed`** (308 existing + 21 new).

**Three things measurement changed**
- **The first before/after run said the fusion was a 0.85× regression.** It is not. Two consecutive
  runs of the *identical* reference path differ by 12–21% on this machine, so cross-process
  sequential A/B measures the quiet minute, not the code. Alternating rounds put the fusion at
  1.00–1.14× (**D-035**). Co-residency was tried as the fix and is worse — two variants' buffers are
  ~500 MB at 1M cells, and a cache-locality effect cannot be measured under cache pressure.
- **One of the contract's four named wins is not a win.** Skipping `feq` on solid cells via
  `where=fluid` is 2–4× slower at every grid, because a masked ufunc costs 3.3× and the arithmetic
  it saves is the cheap part of a memory-bound kernel. Measured and dropped (**D-034**), rather than
  shipped behind a toggle for ~25 lines of duplicated equilibrium.
- **The laptop's power state moves the budget table further than the optimisation does.** Re-measured
  late in the session on battery at 42% (CPU **1802 MHz of 3201**), the passing build reads
  402.7 / **117.0** / 16.8 and the 160k case sits 2.5% under its floor. Nothing in `lbm/` changed
  between the readings. Recorded in full rather than quietly reporting the good run.

**Not done / deferred**
- Nothing from the T010 contract. No recording sinks, no CLI — T011.
- `equilibrium` is **over half** the step at 1M cells and is deliberately untouched: restructuring it
  is not one of the four cheap wins, and § Notes settles it — Phase 0's job is understanding and M5
  replaces this kernel. It is the obvious first target when M5 starts.
- The Guo body force is not folded into the fused kernel, so Rung 1 runs the T009 sequence. Its
  22×16 grid makes the pass worthless and its arithmetic is worth more untouched.
- `f_pre` is still a full `(9, ny, nx)` copy every step whose only consumer is solid cells; gathering
  just those cells would remove one of the seven whole-array passes. Not one of the four wins, not
  attempted, noted for M5.
- `DOCS/ISSUES.jsonl` is **untracked in git** despite `CLAUDE.md` saying the queue is committed —
  noticed, not acted on, since committing is the user's call.

**Decisions made**
- **D-033** (fused collide/bounce-back/snapshot/stream, and why it must cross `bounce_back`),
  **D-034** (skip-`feq`-on-solid measured and dropped), **D-035** (benchmark protocol, and that an
  absolute steps/s needs its CPU clock quoted), **D-036** (**closes Q-004** — `TAU_FLOOR = 0.537`,
  not 0.54, and why), **D-037** (the preallocation fix is kept although unmeasurable). All in
  § Decisions.

**Blockers**
- None.

**Rung status after this session**
- R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run this session on the optimised kernel, all four
  bit-identical to session 9.

**Next**
- Paste `PROMPTS/011-t011-recording-sinks-cli.md` into a fresh session. It runs `/start-task T011`,
  which is **M4** and the last task of Phase 0. `imageio[ffmpeg]` must be installed into `myenv`
  first and a row added to § Environment. Constraint 10 (one `render()`, three sinks) and constraint
  8 (record must **not** drop; live may) are the two that bite.
