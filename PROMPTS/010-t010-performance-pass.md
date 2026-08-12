# Session 10 — T010: performance pass

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map. **Constraint
   6 ("do not optimise before Rung 3 passes") is the one this task exists to spend, and it has been
   lifted since session 7.** Constraint 11 (bit-identical restart) is the one most likely to break.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-032), session log. The decisions that bear on this task are **D-006** (hot functions
   take preallocated outputs), **D-011** and **D-020** (the exact timestep order, which a fused
   kernel must preserve), **D-022** (`out_prev` is reconstructed, so the checkpoint stays four
   things) and **D-012** (the `float32` residual floor — do not mistake it for a tolerance to
   tighten).
3. `DOCS/TASKS1.md` § T010 — the task contract, in full, plus the backlog index row. Also read
   § T009's outcome note: nothing under the hot path changed there, so the baseline you measure is
   session 7/8's solver unchanged.
4. `DOCS/IDEA2.md` § **Performance budget** (the whole section — the table and the four cheap wins
   are the contract) and § **Stability**.
5. `DOCS/PLAN1.md` § Session map and § Risks. T010 is session 10 of 11, carries no milestone, and
   § Risks names its failure mode explicitly: *"Performance pass breaks correctness — a rung goes
   red after T010 — revert rather than debug a fused kernel."*
6. `lbm/core.py` (`collide`, `stream`, `equilibrium`, `macroscopic`) and `lbm/runner.py::Sim.step` —
   the loop you are optimising, and the buffer ownership you must not duplicate.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 9: **T009 is `done`**, every acceptance criterion run and green.
  `lbm/units.py` is new (`LatticeUnits.from_physical` — physical case in, `dx`/`dt`/`tau`/lattice
  `U`/`Re` out, raising on `U >= 0.1` or `tau <= 0.51`); `lbm/geometry.py` gained `from_png` and
  `from_svg`; `tests/test_units.py` and `tests/data/test_body.png` are new;
  `validate/cylinder.py` gained an off-by-default `--physical` flag. **No change to the hot path.**
- **The ladder was re-run in full in session 9 and all four rungs are green:**
  R1 L2 **0.3650%**, peak `|u|` 0.07955 · R2 **0.75% / 0.42% / 1.01%** max deviation ·
  R3 **St 0.1731, Cd 1.4031 ± 0.0086**, peak 0.09685 · R4 square **Cd 1.5279**, St 0.1489, peak
  0.09758; polygon **Cd 1.4276 ± 0.0226**. `myenv/Scripts/python.exe -m pytest` → **`308 passed`**.
  **Those are the numbers your optimised solver has to reproduce.**
- **Milestone:** **M3** reached (2026-08-12). T010 carries no milestone; **M4 is T011**.
- **Completed tasks:** T001 … T009.

## Your task this session

**T010 — Performance pass.** One task, this session only.

Run this first:

    /start-task T010

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: hit the performance budget with the **cheap wins only**, and prove correctness survived.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] Baseline steps/s recorded **before** any change, for 400×100, 800×200, 2000×500, and written into `DOCS/STATE1.md`.
- [ ] Applied, each measured separately: preallocation audit (no allocation in the loop), `float32` end to end, fused collide+stream in one pass over `f`, skip `feq` on solid cells.
- [ ] Post-change numbers meet the budget: **≥400 steps/s at 400×100, ≥120 at 800×200, ≥15 at 1M cells** (budget is ~500/~150/~20; these are the pass floors).
- [ ] `bench.py` prints a before/after table.
- [ ] **All four rungs re-run and still green** — with the same tolerances, not relaxed ones.
- [ ] Restart is still bit-identical (T006's test still passes).
- [ ] No new dependency: no Numba, no Cython, no GPU. Pure NumPy.

### Constraints that bite on this task

- **Constraint 6** — the gate is *lifted*: Rung 3 has been green since session 7 and was re-run green
  in session 9. This is the one session where restructuring the kernel is in scope. It is also the
  only one — T011 is the CLI and recording sinks, not more tuning.
- **Constraint 11** — restart must stay **bit-identical**. Fusing changes float ordering, and
  `tests/test_runner.py`'s 500 → checkpoint → 500 → reload → 500 test is what catches it.
  `DOCS/PLAN1.md` § Risks and the contract agree: **if a fusion breaks it, revert the fusion, never
  the test.**
- **Constraint 1** — the physics must not change. "Skip `feq` on solid cells" in particular must
  leave fluid-cell results **bitwise** identical to the unoptimised path on a small grid; assert
  that, do not eyeball a `Cd`.
- **Constraint 4** — `(9, ny, nx)`, `float32`, constants only from `lbm/core.py`. A fused kernel that
  upcasts to `float64` anywhere in the step path both loses the speed and breaks constraint 11.
- **Constraint 5** — the ladder stays complete. All four rungs get re-run at the end. Budget from
  session 9's measured wall clock: R1 ~15 s, R2 ~155 s, R3 ~370 s (633 s under contention), R4
  ~**33 min** — about 40 minutes for the full ladder, so **start it early and in the background.**
- **Constraint 7 / 8** — `steps_per_frame` and the ring buffer are already correct; do not
  "optimise" the display path. The physics loop is the target.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-004** (open, **and this is the natural place to answer it**) — should
  `validate/cylinder.py::tau_for`'s floor rise from D-016's 0.53 to the 0.54 that session 8 measured
  (**D-029**)? Nothing is red: Rung 3 runs at `tau = 0.5378`, measured stable. But the floor as
  written would admit a future case at 0.5346, which is measured to produce `nan`. T008 and T009 both
  declined to edit a passing benchmark they did not own; **T010 re-runs every rung anyway**, so it is
  the cheapest session in which to change it and prove nothing moved. Decide, log it, or say plainly
  why you left it.
- **D-029** — measured stability data, so you do not have to re-measure: `tau` **0.5330** (disc) blows
  up by step 1500, **0.5346** (square) by step 3200, **0.5378** and **0.5512** survive 60000 steps.
- **D-020 / D-011** — the timestep order is `copy f_pre` → `macroscopic` → `force_velocity_shift` →
  `equilibrium` → `collide` → `apply_body_force` → `bounce_back` → `copy f_bb` → `stream` →
  `outlet_zero_gradient` → `inlet_velocity`. A fused collide+stream has to keep the two snapshots
  `lbm.probe.forces` consumes (`f_bb` pre-stream, `f` post-stream) or Rung 3 and Rung 4 measure
  nothing.
- **D-022** — the checkpoint is exactly `f` / `solid` / `step_count` / config plus a `format` integer,
  and `out_prev` is reconstructed from `f[:, :, outlet_col]`. If a fusion writes the outlet column
  after `outlet_zero_gradient`, that reconstruction silently stops being true — the assertion in
  `tests/test_runner.py::test_the_outlet_prev_column_equals_f_at_the_end_of_a_step` is what tells you.
- **D-006** — `macroscopic` and `equilibrium` already take preallocated outputs and `Sim` already
  owns every buffer. The preallocation audit is a *verification* task, not a rewrite; `tracemalloc`
  growth over 1000 steps was measured at **< 20 kB** in session 6.

### Known starting points, measured

- 504 × 440 = **222k cells at ~130 steps/s** (Rung 3, session 7), and Rung 3 pays an extra `forces`
  call per step on top.
- 744 × 557 = **414k cells at ~54–64 steps/s** (Rung 4, session 8).
- `lbm.boundary.inlet_velocity` allocates a `(ny,)` boolean (`fluid = ~solid[:, col]`) on every
  call — transient, freed each step, invisible to the growth test, and **the one allocation left in
  the step loop** (noted in session 6).
- `DOCS/IDEA2.md` § Performance budget expects ~500 / ~150 / ~20 steps/s at 40k / 160k / 1M cells;
  the contract's pass floors are 400 / 120 / 15.

### Before you start

- **Nothing to install.** `myenv` has numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  pygame 2.6.1, Python 3.11.15. The contract forbids adding Numba, Cython or a GPU dependency, so
  this session should end with § Environment unchanged. (`imageio[ffmpeg]` is **T011's** install.)
- Run tests and validation scripts **from the repo root** so `import lbm` resolves.
- `bench.py` is new this session. The contract says it prints a before/after table — so capture the
  baseline *first*, into a file, before the first optimisation lands.

## Scope discipline

Work only what's in the contract. MP4/GIF sinks and the CLI are **T011**. Something else genuinely
needs doing? `/new-task` it. If it is under `DOCS/IDEA2.md` § Deliberately deferred, the answer is no.

The temptation this session is to keep going past the budget. `DOCS/TASKS1.md` § T010 Notes settles
it: *if a win costs more than ~20 lines of clarity for under 10% speed, drop it* — Phase 0's job is
understanding, and M5 replaces this kernel anyway.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it. Each of the four wins is measured
   **separately**, so a win that does nothing can be dropped rather than carried.
2. Run `/validate`. R1, R2, R3 and R4 must all be reported with their measured numbers and compared
   against session 9's above. A rung that moves is a **revert**, not a debugging session.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 308 existing tests
   must still pass, and the bit-identical restart test is the one that matters most.
4. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T010 `in_progress`.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes
   `PROMPTS/011-t011-*.md` for the final session (**M4**). Do not end the session without it.
