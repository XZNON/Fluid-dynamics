# Session 9 — T009: physical units + PNG/SVG mask

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map. Note the
   convention this task exists to enforce: **physical units never reach the solver.** `lbm/units.py`
   converts at the boundary; everything inside `lbm/` is lattice units.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-030), session log. **D-019 (characteristic length `D`), D-023 (`steps_per_frame` takes
   `dt` as an already-converted scalar) and D-018 (`check_mask` exempts fully-solid borders) are the
   three your API sits on.** D-023 in particular already fixes half of this task's interface: `dt` is
   *seconds of physical time per lattice timestep*, a plain scalar the caller supplies, and
   `lbm/units.py` is named in it as where that arithmetic moves.
3. `DOCS/TASKS1.md` § T009 — the task contract, in full. Also read § T008's outcome note: it is a
   case study in the *setup* being wrong while the solver is right, and T009 is a task whose entire
   job is producing correct setups from physical inputs.
4. `DOCS/IDEA2.md` § **Geometry from a mask** (sources 2 and 3 are yours) and § **Stability**.
5. `DOCS/PLAN1.md` § Session map and § Risks — T009 is session 9 of 11 and carries no milestone;
   M4 is T011, and T011 **depends on this task**.
6. `lbm/geometry.py` — `from_png` and `from_svg` land beside the T004 primitives and must reuse
   `check_mask`, `bounding_box` and `min_thickness` rather than re-deriving them.
7. `validate/cylinder.py` — the case you must reproduce through the units path, and the source of
   the `Cd` you are compared against.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 8: **T008 is `done`**, every acceptance criterion run and green, and
  **the validation ladder is complete**. `validate/polygons.py` is new (square cylinder plus an
  arbitrary convex polygon); `tests/test_polygons.py` is new; **nothing under `lbm/` changed at
  all.**
  `myenv/Scripts/python.exe -m validate.polygons --headless` → **PASS**: square **Cd 1.5279 ±
  0.0271** (band 1.4–1.6), St 0.1489, Cl amplitude 0.6510 = 42.6% of Cd, peak `|u|` 0.09758, on a
  744 x 557 domain at `D = 30`, `U = 0.053`, `tau = 0.5477`; polygon **Cd 1.4276**, finite, peak
  0.08944.
  `myenv/Scripts/python.exe -m pytest` → `251 passed`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **all four green, every one run in session 8.**
  R1 L2 **0.3650%** · R2 **0.75% / 0.42% / 1.01%** · R3 **St 0.1731, Cd 1.4031 ± 0.0086** ·
  R4 **Cd 1.5279**.
- **Milestone:** **M3** reached (2026-08-12). T009 carries no milestone gate of its own; **M4 is
  T011, and T011 cannot start until this task lands.**
- **Completed tasks:** T001 … T008.

## Your task this session

**T009 — Physical units + PNG/SVG mask.** One task, this session only.

Run this first:

    /start-task T009

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: the user speaks physics ("air, 20 m/s, 1.5 m object, PNG of a wing") and the code derives
resolution, `tau` and timestep — **refusing** configs that would be unstable.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `LatticeUnits.from_physical(...)` returns `dx`, `dt`, `tau`, lattice `U`, and `Re`, with the derivation in the docstring.
- [ ] Round-trip test: physical → lattice → physical reproduces `Re` to within 0.1%.
- [ ] It **raises** with a message naming the offending quantity when the config implies lattice `U >= 0.1` or `tau <= 0.51`, and suggests the resolution that would fix it.
- [ ] `from_png(path, shape)` thresholds alpha (falling back to luminance), resizes to grid, returns a bool mask, and runs `check_mask` automatically.
- [ ] `from_svg(path, shape)` rasterises at least simple closed paths; if a dependency is missing it raises a clear install message rather than failing obscurely.
- [ ] A test PNG committed under `tests/data/` produces a mask with the expected solid-cell count ±2%.
- [ ] Cylinder run reproduced through the physical-units path gives the same `Cd` as T007 to within 2%.
- [ ] Rungs 1–4 still green.

### Constraints that bite on this task

- **Constraint 3** — this is where the `|u| < 0.1` guard lives *for users*. It **raises**; it does
  not warn. Session 8 is the argument for strictness: a square cylinder at Rung 3's inlet measured
  peak `|u|` 0.10211 and the run was wasted.
- **Constraint 2** — `nu = cs2 (tau - 0.5)`. `tau` is **derived**, never set alongside a `nu`. One
  input path, no `nu` setter that bypasses `tau`.
- **Constraint 12** — `from_png` calls `check_mask` itself. A downscaled PNG is the most likely
  source of a 1-cell-thin wall in this whole project, and `min_thickness` (D-017) is what catches it.
- **The units convention** (`CLAUDE.md` § Coding conventions) — physical quantities stop at
  `lbm/units.py`. Nothing inside `lbm/` other than that module may hold metres or seconds. `SimConfig`
  takes lattice scalars only, and D-023 kept it that way on purpose.
- **Constraint 4** — `(9, ny, nx)`, `float32`, constants only from `lbm/core.py`.
- **Constraint 5** — the ladder is complete and must **stay** complete. All four rungs are green
  now and all four are re-run at the end. Budget: R1 ~15 s, R2 ~155 s, R3 ~370 s, R4 ~**33 min**
  (two cases, 112k steps at 414k cells) — about 40 minutes for the full ladder, so start it early
  and in the background.
- **Constraint 6 is lifted** but optimisation is **T010**, not this session.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-002** (open, **and it is yours to answer**) — the SVG rasterisation dependency is not chosen.
  `DOCS/TASKS1.md` § T009 Notes: **if SVG drags, ship PNG and `/new-task` the SVG half.** PNG is
  what M4 requires; SVG is not. Decide early, log it, do not let it eat the session.
- **Q-004** (open) — should `validate/cylinder.py::tau_for`'s floor rise from D-016's 0.53 to the
  0.54 that session 8 measured (D-029)? Not yours and not blocking, but **your `tau <= 0.51`
  rejection is the third floor in this project**, and it is worth one sentence in the docstring
  saying why it differs from the other two. Nothing is red: Rung 3 runs at 0.5378, measured stable.
- **D-029** — measured stability data you can cite instead of re-measuring: `tau` **0.5330** (disc)
  blows up by step 1500, **0.5346** (square) by step 3200, **0.5378** and **0.5512** survive 60000
  steps. So `tau <= 0.51` is a *floor on nonsense*, not a stability guarantee — say so, because a
  user handed a config at `tau = 0.52` will get `nan` and the message must not have implied safety.
- **D-023** — `steps_per_frame(dt, fps, speed)` already takes `dt` as seconds per timestep and names
  `lbm/units.py` as its source. Produce exactly that scalar; do not change the signature.
- **D-019** — characteristic length `D` is the cross-stream extent of the object's bounding box, and
  the blockage denominator is the fluid span. `Re` in your conversion must use the same `D`, measured
  from the mask, or the units path will disagree with `validate/cylinder.py` by a few percent and the
  2% criterion will fail for a reason that is not a bug.
- **D-018** — `check_mask` peels fully-solid border rows/columns before checking. A PNG that
  rasterises to a solid frame will pass silently; that is intended, not a hole.
- **D-030** — session 8 found that `Sim` seeds solid cells with the *inlet* equilibrium, so there is
  fluid moving inside a body at step 0. `validate/polygons.py::seed_solid_at_rest` fixes it locally.
  Not your task, but if T009's demo puts a PNG body in a flow, reach for that function rather than
  rediscovering it.

### Before you start

- **PNG needs nothing new** — `pillow` 12.3.0 is already in `myenv` (numpy 2.4.6, matplotlib 3.11.1,
  pytest 9.1.1, pygame 2.6.1, Python 3.11.15). **SVG probably does** — that is Q-002. Anything you
  install goes in `DOCS/STATE1.md` § Environment **in the same session**, via
  `myenv/Scripts/pip.exe install <pkg>`.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves.
- The test PNG under `tests/data/` is a committed binary; keep it small (a few kB) and generate it
  with a script whose source is in the test file, so the expected solid-cell count is derivable and
  not a magic number.

## Scope discipline

Work only what's in the contract. The performance pass is **T010**; MP4/GIF sinks and the CLI are
**T011**. Something else genuinely needs doing? `/new-task` it. If it is under `DOCS/IDEA2.md`
§ Deliberately deferred, the answer is no.

The temptation this session is SVG. It is one bullet of eight and the only one M4 does not need.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it. The cylinder-through-units `Cd` is
   compared against T007's **1.4031**, and 2% of that is ±0.028.
2. Run `/validate`. R1, R2, R3 and R4 must all be reported with their measured numbers — the ladder
   is complete now and this session must not be the one that quietly breaks it.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 251 existing tests
   must still pass.
4. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T009 `in_progress`.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes
   `PROMPTS/010-t010-*.md` for the next session. Do not end the session without it.
