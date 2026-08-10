# Session 7 — T007: render + live sink + cylinder benchmark → Rung 3 (M3)

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-024), session log. **D-019 (characteristic length `D`), D-021 (convective outlet) and
   D-024 (how `run()` feeds a sink) are the ones your window and your benchmark sit on.**
3. `DOCS/TASKS1.md` § T007 — the task contract, in full. Also read § T006's outcome note: the
   runner, the ring buffer and the restart are done, and your job is to hang a renderer and a window
   off them without touching the loop.
4. `DOCS/IDEA2.md` § **What to actually draw**, § **Three output sinks, same frame source**,
   § **Never block the sim on the display**, and § **Validation ladder** Rung 3 (the whole rung).
5. `DOCS/PLAN1.md` § Session map and § Risks — T007 is session 7 of 11 and it is **M3, the demo**.
   § Risks assigns it "cylinder shows no shedding": almost always insufficient upstream/downstream
   space or blockage ratio, which is exactly what T004's `check_mask` exists to catch *before* the
   run.
6. `lbm/core.py`, `lbm/boundary.py`, `lbm/geometry.py`, `lbm/probe.py`, `lbm/runner.py` — all exist
   and all work. `lbm/runner.py`'s module docstring holds the timestep order and the buffer list.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 6: **T006 is `done`**, every acceptance criterion run and green.
  `lbm/runner.py` is new: `SimConfig`, `Sim` (owns `f`, `solid`, `step_count` and every buffer),
  `Sim.step()` in the D-020 order, `Sim.vorticity()` / `Sim.forces()` / `Sim.residual()`,
  `steps_per_frame(dt, fps, speed)`, `RingBuffer`, `Sink` (ABC) + `NullSink`,
  `save_checkpoint`/`load_checkpoint`, and `run(...)`.
  `myenv/Scripts/python.exe -m pytest` → `198 passed`. Restart is bit-identical on three configs;
  a 4 ms sink dropped 51 of 60 frames while all 120 steps ran.
- **Rung status:** R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜ — **R3 is yours.** It is the first new rung since
  session 3.
- **Milestone:** **M2** reached (2026-08-10). **M3 is this session** — its gate is
  `python -m validate.cylinder` printing PASS with `St` in 0.155–0.175 and `Cd` in 1.25–1.45, and the
  live window running without stuttering the physics.
- **Completed tasks:** T001, T002, T003, T004, T005, T006.

## Your task this session

**T007 — Render + live sink + cylinder benchmark → Rung 3.** One task, this session only.

Run this first:

    /start-task T007

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: a live window showing a von Kármán vortex street behind a cylinder at Re 100, with a measured
Strouhal number and drag coefficient that match the literature.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `render` maps a scalar field to `uint8` RGB with a **diverging** colormap and **symmetric fixed** limits passed in — never computed per frame.
- [ ] A test renders two frames with different data and identical limits and asserts the mapping of a fixed value is byte-identical across them (no flicker).
- [ ] `LiveSink` opens a pygame window, pulls from the ring buffer, and drops frames when behind — no `pygame` call sits inside the physics loop.
- [ ] Measured: opening the window changes steps/s by **less than 10%** versus headless, printed by the validation script.
- [ ] `validate/cylinder.py` sets up circular cylinder at Re 100 with the T004 sanity checks passing (≥8D downstream, <10% blockage), runs past transient, and prints St, Cd, PASS/FAIL.
- [ ] **`St` within 0.155–0.175 (ref 0.164) and `Cd` within 1.25–1.45 (ref 1.34).**
- [ ] Shedding is confirmed present, not assumed: the Cl series amplitude after transient exceeds 1% of Cd.
- [ ] `--headless` flag runs the same validation with no window.
- [ ] Rungs 1–2 still green.

### Constraints that bite on this task

- **Constraint 9** — draw **vorticity**, diverging colormap, symmetric **fixed** limits. Speed
  magnitude is a grey smear and per-frame autoscaling flickers; the byte-identical-mapping test is
  the assertion form of that. `Sim.vorticity()` already computes the field and puts `nan` on solid
  cells so the renderer can skip them.
- **Constraint 10** — **one** `render()`, three sinks. `LiveSink` consumes `render`'s output; it does
  not colour anything itself. Record sinks are T011.
- **Constraint 8** — never block the sim on the display. The ring buffer already exists
  (`lbm/runner.py`); the <10% steps/s claim is its measurable form. `run(..., drop=True)` drains from
  a consumer thread (D-024) — `LiveSink.push` runs on that thread, so no pygame call belongs in the
  physics loop.
- **Constraint 7** — one frame is many timesteps and `steps_per_frame` is **computed**
  (`lbm.runner.steps_per_frame`, D-023). A hardcoded 20 fails this task too.
- **Constraint 12** — geometry is one boolean `solid` array; the object must be ≥8 D from the outlet,
  blockage under ~10%, solid ≥3 cells thick. `lbm.geometry.check_mask` enforces all three and `Sim`
  runs it at setup.
- **Constraint 3** — lattice velocity under 0.1. Rung 2 settled on `U = 0.09` as the honest ceiling
  (D-016); Rung 3 wants a slower free stream than that anyway.
- **Constraint 6 lifts only *after* this task passes.** No fused kernels, no Numba, no GPU until the
  cylinder shows the right Strouhal number. T010 is the performance pass.
- **Constraint 5** — do not start Rung 4 (T008) in this session even if R3 goes green early.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-002** (open) — SVG rasterisation dependency for T009 not chosen. Not yours, not blocking.
- **D-019** — **characteristic length `D` is the cross-stream extent of the object's bounding box**
  (`lbm.geometry.bounding_box`), and the blockage denominator is the fluid span. `Sim` derives `D`
  this way already; `probe.forces` and `probe.strouhal` both take it. Do not invent a second
  definition — that is how a 10% error in `Cd` gets blamed on the solver.
- **D-021** — the outlet is **convective** (`lam = sqrt(CS2)`, 0.6% reflection against 35% for the
  bare copy). `SimConfig.convective_outlet` is `True` by default. `lam = U` rather than `cs` is the
  other defensible tuning and **this rung is the run that can measure which leaves the wake
  cleaner** — `SimConfig.outlet_lam` is exposed for exactly that.
- **D-024** — `run(sim, sink, drop=True)` drains the ring buffer from one consumer thread; the
  physics stays a single un-threaded loop. `drop=False` drains inline for a recorder. Measured with a
  4 ms sink: 51 of 60 frames dropped, every step executed.
- **D-020** — `Sim.forces()` already uses the two correct snapshots (`f_bb` pre-stream, `f`
  post-stream). Collect the `Cl` series from it each step and hand it to `lbm.probe.strouhal`.
- **D-012 / D-014** — `float32` puts a per-step floor near `1.7e-6` on any residual, and `u` on solid
  cells is meaningless, so residuals are fluid-cells-only and measured over an interval.
- **From session 4, worth carrying:** a `D = 21` cylinder in a 121-row channel is **17.6% blockage**
  and `check_mask` warns. Rung 3 needs roughly `ny >= 10 D` (about 240 rows for `D = 21`) plus
  `>= 8 D` downstream. Size the domain from that before the first run, not after.

### Before you start

- **`pygame` must be installed:** `myenv/Scripts/pip.exe install pygame`, and **add a row to
  `DOCS/STATE1.md` § Environment in the same session** (D-003: the interpreter is always
  `myenv/Scripts/python.exe`, never bare `python`).
- `myenv` currently has numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1, Python 3.11.15.
  (`imageio[ffmpeg]` lands in T011.)
- Run tests and validation scripts **from the repo root** so `import lbm` resolves — there is no
  packaging config in Phase 0.
- Rungs 1 and 2 must be green before you start and again before you finish. Rung 2's Re 1000 case
  takes about 155 s; the whole ladder is roughly 3 minutes.
- If shedding does not start, the contract's own note applies: perturb the initial condition slightly
  or offset the cylinder half a cell — a perfectly symmetric setup on a symmetric grid can stay
  symmetric far longer than physics would.

## Scope discipline

Work only what's in the contract. The square cylinder is **T008**; PNG/SVG masks and physical units
are **T009**; the performance pass is **T010**; MP4/GIF sinks and the CLI are **T011**. Something
else genuinely needs doing? `/new-task` it. If it is under `DOCS/IDEA2.md` § Deliberately deferred,
the answer is no.

The temptation this session is to optimise — the cylinder run is the longest yet. Constraint 6 says
no, and it lifts the moment `validate/cylinder.py` prints PASS, not before.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it. `St` and `Cd` inside their bands is
   the pair that earns M3.
2. Run `/validate`. All of R1, R2 and R3 must be reported with their measured numbers; R4 stays ⬜.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 198 existing tests
   must still pass.
4. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T007 `in_progress`. Do not claim M3 without the
   gate command printing PASS.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes
   `PROMPTS/008-t008-*.md` for the next session. Do not end the session without it.
