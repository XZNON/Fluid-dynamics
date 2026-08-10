# Session 5 — T005: inlet / outlet BC + probes

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-019), session log. D-005 (velocity component order), D-006 (preallocated outputs) and
   D-011 (the timestep order) are the ones your code sits inside.
3. `DOCS/TASKS1.md` § T005 — the task contract, in full. Also read § T004's outcome note: the
   geometry and its sanity checks are done and your inlet/outlet work on the masks it produces.
4. `DOCS/IDEA2.md` § **The method, in the order the code runs it** (step 6 — inlet/outlet),
   § **Stability** (the "reflections from the right edge" and "sim fine but wake is wrong" rows are
   the two failures this task exists to prevent) and § **What to actually draw** (vorticity, which
   you compute here — `render.py` only colours it).
5. `DOCS/PLAN1.md` § Session map and § Risks — T005 is session 5 of 11. It is the last task before
   the runner; nothing in § Risks is assigned to it directly, but the cylinder-shows-no-shedding
   risk is measured by the probes you write.
6. `lbm/core.py`, `lbm/boundary.py`, `lbm/geometry.py` — all exist and all work. `lbm/boundary.py`'s
   module docstring pins the timestep order; you are adding to that file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 4: **T004 is `done`**, every acceptance criterion run and green.
  `lbm/geometry.py` has `circle`, `rectangle`, `polygon`, `regular_polygon`, `channel_walls`,
  `bounding_box`, `min_thickness`, `strip_solid_border` and `check_mask`.
  `myenv/Scripts/python.exe -m pytest` → `103 passed`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜ — **you build neither.** T005's gate is unit tests.
  Rung 3 is T007 and is what actually audits `forces`.
- **Milestone reached:** **M2** (2026-08-10). **M3 is T007, not this session.**
- **Completed tasks:** T001, T002, T003, T004.

## Your task this session

**T005 — inlet / outlet BC + probes.** One task, this session only.

Run this first:

    /start-task T005

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: open-channel flow — velocity in at the left, non-reflecting out at the right — plus the
measurement code that makes Rung 3 checkable. Without probes, "the wake looks right" is all we'd
have.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `inlet_velocity` imposes a prescribed profile (uniform or parabolic, selectable) — Zou–He or equilibrium-based, stated in the docstring.
- [ ] `outlet_zero_gradient` copies the second-to-last column; a test confirms a pressure pulse crossing the outlet reflects less than 5% of its amplitude.
- [ ] `vorticity(u)` returns `d(uy)/dx - d(ux)/dy` via central differences, one-sided at edges, masked to `nan` on solid cells.
- [ ] `forces(f_pre, f_post, solid)` computes drag and lift by **momentum exchange** over boundary links, returning dimensionless `Cd`, `Cl` given `U` and characteristic length.
- [ ] `forces` validated on a known case: uniform flow with no obstacle gives `|Cd| < 1e-6`.
- [ ] `strouhal(cl_series, dt)` finds the dominant frequency via FFT, ignores the first 30% of the series as transient, and returns `St = f*D/U`.
- [ ] `strouhal` verified against a synthetic sine of known frequency to within 1%.
- [ ] `residual(u_now, u_prev, U)` returns `max|Δu|/U`.
- [ ] `pytest tests/test_probe.py` green; Rungs 1–2 still green.

Gate for this task is **unit tests plus both existing rungs still green** — there is no Rung 3 yet.

### Constraints that bite on this task

- **Constraint 9** — vorticity is the field that gets drawn, and it is computed **here**, not in
  `render.py`. `render.py` colours arrays; it does not do physics.
- **Constraint 3** — inlet `U` stays under 0.1. `inlet_velocity` **warns** if asked for more (T009's
  `units.py` is where it raises).
- **Constraint 4** — `f` is `(9, ny, nx)`, `float32`, index order `(direction, y, x)`; the mask is
  `(ny, nx)`. The nine constants come from `lbm/core.py` and are never redefined.
- **Constraint 6** — do not optimise before Rung 3 passes. Precompute the boundary-link list once
  from the mask (that is correctness and clarity, not optimisation); leave collide/stream alone.
- **Constraint 1** — bounce-back only. Momentum exchange is computed over the bounce-back links; do
  not reach for interpolated boundaries to make `Cd` prettier.
- **Coding conventions** — type hints with array shapes documented, docstrings citing the
  `DOCS/IDEA2.md` section, `float32`, no allocation in the step loop, no physical units inside
  `lbm/`.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-002** (open) — SVG rasterisation dependency for T009 not chosen. Not yours, not blocking.
- **D-011** — **the timestep order is fixed** and your new boundaries slot into it:
  `copy f_pre` → `macroscopic` → `force_velocity_shift` → `equilibrium` → `collide` →
  `apply_body_force` → `bounce_back` → `stream`. `f_pre` is the copy taken **before collision**.
  Momentum-exchange drag needs both the pre- and post-stream populations on the boundary links —
  decide where the runner will keep that copy **now**, and write it down, so T006 does not have to
  reshape the API.
- **D-005** — velocity is `u` of shape `(2, ny, nx)`, component 0 = `ux`, component 1 = `uy`.
- **D-006** — hot functions take optional preallocated outputs (`macroscopic(f, rho=None, u=None)`,
  `equilibrium(rho, u, feq=None, work=None)`). The probes run every step in T007, so follow the same
  pattern for anything on the per-step path; a once-per-run link list is built eagerly.
- **D-012 / D-014** — `float32` gives `u` a round-off floor near `1.7e-6` per step, and `u` on solid
  cells is `(e.f)/rho` with a meaningless `rho`. `residual` must therefore be documented against
  that floor and computed on **fluid cells only** — Rung 2's residual read `8.4e+01` until it was.
- **D-017 / D-018 / D-019** — `check_mask` measures thickness per connected component, exempts
  fully-solid domain borders, and takes the characteristic length from the object's cross-stream
  bounding-box extent. `forces` needs the same `D`; take it from `lbm.geometry.bounding_box` rather
  than inventing a second definition.
- **Measured last session, and it will bite T007:** a `D = 21` cylinder in a 121-row channel is
  **17.6% blockage** and `check_mask` warns. Rung 3 wants roughly `ny >= 10 D` and `>= 8 D`
  downstream. Any demo case you set up while testing should already obey that.

### Before you start

- No new package needed. `myenv` has: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  Python 3.11.15. (`pygame` lands in T007, `imageio[ffmpeg]` in T011.) If you do install something,
  add a row to `DOCS/STATE1.md` § Environment in the same session.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves — there is no
  packaging config in Phase 0.
- `myenv/Scripts/python.exe` is the canonical interpreter (**D-003**). Never bare `python`.
- Rungs 1 and 2 must be green before you start and again before you finish. Rung 2's Re 1000 case
  takes about 150 s; the whole ladder is roughly 3 minutes.

## Scope discipline

Work only what's in the contract. The runner, the ring buffer and restart are **T006**; rendering
and the cylinder benchmark are **T007**; PNG/SVG masks and physical units are **T009**. Something
else genuinely needs doing? `/new-task` it. If it's under `DOCS/IDEA2.md` § Deliberately deferred,
the answer is no.

`forces` is the single most error-prone function in the project and Rung 3's `Cd ≈ 1.34` is what
audits it — but that is T007. Do not build a cylinder run this session to "check"; unit tests plus
the zero-obstacle `|Cd| < 1e-6` case are the gate.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate`. Rungs 1 and 2 must both be reported with their measured numbers; R3–R4 stay ⬜.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 103 existing tests
   must still pass.
4. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T005 `in_progress`.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes
   `PROMPTS/006-t006-*.md` for the next session. Do not end the session without it.
