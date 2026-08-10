# Session 4 — T004: geometry primitives + mask sanity checks

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-016), session log. D-009 and D-013 in particular describe how walls and corners are
   already handled, and your masks feed both.
3. `DOCS/TASKS1.md` § T004 — the task contract, in full. Also read § T003's outcome note; it says
   what the two validation scripts currently build inline and therefore what they want from you.
4. `DOCS/IDEA2.md` § **Geometry from a mask** (in full — it is short, and all three sanity checks
   come from it) and § **Stability** (the "flow through the object" row is the failure T004 exists
   to prevent).
5. `DOCS/PLAN1.md` § Session map and § Risks — T004 is session 4 of 11. § Risks names the cylinder
   showing no shedding as a live risk and says **T004's sanity checks exist to catch it before the
   run**.
6. `lbm/core.py`, `lbm/boundary.py`, `validate/poiseuille.py`, `validate/cavity.py` — all exist and
   all work. `validate/poiseuille.py::channel_mask` and `validate/cavity.py::cavity_masks` are the
   two inline masks whose job you are generalising.
7. `Navier-Fluid-Equation/polygonsDemo.py` and `panels.py` — **read-only prior work** (D-004). Reuse
   the vertex / point-in-polygon logic; never import from that directory into `lbm/`.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 3: **T003 is `done`**, every acceptance criterion run and green.
  `lbm/boundary.py` now has `moving_wall` (momentum-corrected Ladd bounce-back); `validate/cavity.py`
  is Rung 2 and passes at Re 100, 400 and 1000.
  `myenv/Scripts/python.exe -m pytest` → `63 passed`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜ — **you build neither.** T004's gate is unit tests.
- **Milestone reached:** **M2** (gate run 2026-08-10:
  `python -m validate.cavity --re 100 --re 400 --re 1000` → PASS; max deviation from Ghia
  0.75% / 0.42% / 1.01%, vortex centre 0.21 / 0.29 / 0.59 cells, peak `|u|` under 0.088 throughout).
  **M3 is T007, not this session.**
- **Completed tasks:** T001, T002, T003.

## Your task this session

**T004 — geometry primitives + mask sanity checks.** One task, this session only.

Run this first:

    /start-task T004

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: `lbm/geometry.py` turns primitives into the one boolean array the solver understands, and
refuses — loudly — to hand back a mask that will produce a wrong answer.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `circle`, `rectangle`, `polygon` each return a `(ny, nx)` bool array; polygon handles concave shapes and is tested against a known-area convex case to within 2%.
- [ ] `channel_walls(ny, nx)` returns top/bottom no-slip rows, composable with `|`.
- [ ] `check_mask(solid, inlet_axis, ...)` returns a warning string, not silence, when: min solid thickness `< 3` cells; object closer than 8 characteristic lengths to the outlet; blockage ratio `> 10%`.
- [ ] Thickness check verified by a test on a deliberately 1-cell-thick diagonal line and a 4-cell-thick block — warns for the first, not the second.
- [ ] Warnings are emitted through `warnings.warn`, and `check_mask(..., strict=True)` raises instead.
- [ ] Characteristic length used for blockage/downstream checks is derived from the mask bounding box and printed.
- [ ] `pytest tests/test_geometry.py` green; Rungs 1–2 still green.

Gate for this task is **unit tests plus both existing rungs still green** — there is no Rung 3 yet.

### Constraints that bite on this task

- **Constraint 12** — geometry is one boolean array, `solid`, shape `(ny, nx)`. Solid at least 3
  cells thick (detect and warn — thinner leaks through bounce-back), object ≥ 8 diameters from the
  outlet, blockage ratio under ~10%. All three checks, not two: skipping the thickness warning is
  how "flow through the object" reaches Rung 3 and wastes a session.
- **Constraint 4** — the mask is `(ny, nx)`, matching `f`'s trailing axes, index order `(y, x)`.
  Not `(nx, ny)`. The nine lattice constants come from `lbm/core.py` and are never redefined.
- **Constraint 5** — the ladder is ordered. Rungs 1 and 2 must both still be green at the end of
  this session: `myenv/Scripts/python.exe -m validate.poiseuille` and
  `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000`.
- **Constraint 6** — do not optimise before Rung 3 passes. No clever vectorisation of the
  point-in-polygon test.
- **Coding conventions** — type hints with array shapes documented, docstrings citing the
  `DOCS/IDEA2.md` section, `float32` where arrays are numeric, no physical units inside `lbm/`.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-002** (open) — SVG rasterisation dependency for T009 not chosen. It is *adjacent* to this task
  and it is **not yours**: T004 is primitives only. PNG and SVG are T009. Do not answer it early.
- **D-009** — **wall offset.** Bounce-back walls sit **halfway between the last fluid node and the
  solid node**. A one-cell solid border at `y = 0` and `y = ny - 1` therefore gives fluid rows
  `1..ny-2` and a characteristic length of `ny - 2`. `channel_walls` must follow this and say so;
  Rung 1 measured the rivals at 14.8% and 12.7% error against 0.365%, so it is settled.
- **D-013** — the two cells where a moving lid meets the side walls belong to the **static walls**,
  not the lid. If any helper you add composes a cavity, keep that convention.
- **D-005** — velocity is `u` of shape `(2, ny, nx)`, component 0 = `ux`, component 1 = `uy`.
- **D-006** — hot functions take optional preallocated outputs. Geometry is not hot — it runs once at
  setup — so plain returns are fine here; don't invent buffer plumbing that nothing needs.
- **D-004** — `Navier-Fluid-Equation/` is read-only prior work. Reuse its polygon-vertex logic by
  reading it and reimplementing inside `lbm/geometry.py`; never import from it, never modify it.
- **D-003** — `myenv/Scripts/python.exe` is the canonical interpreter. Never bare `python`.

### Before you start

- No new package needed. `myenv` has: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  Python 3.11.15. Pillow is already present and will matter in T009, not now. If you do install
  something, add a row to `DOCS/STATE1.md` § Environment in the same session.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves — there is no
  packaging config in Phase 0.
- Rungs 1 and 2 must be green before you start and again before you finish. Rung 2's Re 1000 case
  takes about 150 s; the whole ladder is roughly 3 minutes.

## Scope discipline

Work only what's in the contract. PNG and SVG masks are **T009**; inlet/outlet and probes are
**T005**; the runner and rendering are **T006/T007**. Something else genuinely needs doing?
`/new-task` it. If it's under `DOCS/IDEA2.md` § Deliberately deferred, the answer is no.

A tempting piece of scope creep here is rewriting `validate/poiseuille.py` and `validate/cavity.py`
to use the new primitives. That is allowed only if both rungs are re-run and still print the same
numbers (Rung 1 L2 0.3650%; Rung 2 0.75% / 0.42% / 1.01%) — otherwise leave them alone and say why.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate`. Rungs 1 and 2 must both be reported with their measured numbers; R3–R4 stay ⬜.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 63 existing tests
   must still pass.
4. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T004 `in_progress`.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes
   `PROMPTS/005-t005-*.md` for the next session. Do not end the session without it.
