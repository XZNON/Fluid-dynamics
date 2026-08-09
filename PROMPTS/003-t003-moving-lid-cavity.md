# Session 3 — T003: moving-lid BC + cavity benchmark → Rung 2

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-012 all constrain your API), session log.
3. `DOCS/TASKS1.md` § T003 — the task contract, in full. Also read § T002's outcome note.
4. `DOCS/IDEA2.md` § **Validation ladder** (Rung 2), § **The method, in the order the code runs it**,
   § **Stability** — all three in full.
5. `DOCS/PLAN1.md` § Session map and § Risks — T003 is session 3 of 11, it is **M2**, and § Risks
   names it as the task most likely to need two sessions.
6. `lbm/core.py` and `lbm/boundary.py` — both exist and both work. Read them before adding to them.
   The timestep order you must reuse is in `lbm/boundary.py`'s module docstring.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 2: **T002 is `done`**, every acceptance criterion run and green.
  `lbm/core.py` now has `collide` and `stream`; `lbm/boundary.py` has `bounce_back`,
  `force_velocity_shift` and `apply_body_force` (Guo). `validate/poiseuille.py` is Rung 1 and passes.
  `myenv/Scripts/python.exe -m pytest` → `47 passed`.
- **Rung status:** R1 🟩 · R2 ⬜ · R3 ⬜ · R4 ⬜ — **you build Rung 2.**
- **Milestone reached:** **M1** (gate run 2026-08-10: `python -m validate.poiseuille` → PASS,
  L2 0.3650%, doubling ratio 1.99940, peak |u| 0.07955). **M2 is this session's gate.**
- **Completed tasks:** T001, T002.

## Your task this session

**T003 — moving-lid BC + cavity benchmark → Rung 2.** One task, this session only.

Run this first:

    /start-task T003

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: lid-driven cavity at Re 100, 400 and 1000 matches Ghia et al. (1982) centreline profiles. This
is **M2** — the point at which the method is trusted. Rung 1 proved collide; this proves boundaries.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `moving_wall` imposes a tangential wall velocity (momentum-corrected bounce-back or Zou–He — state which in the docstring).
- [ ] Ghia reference values for `ux` along the vertical centreline and `uy` along the horizontal centreline are stored as a literal table in `validate/cavity.py` with the citation, at the standard 17 sample points.
- [ ] `validate/cavity.py --re {100,400,1000}` runs to steady state (residual `max|u_n - u_{n-1}| / U < 1e-6`) and prints per-Re PASS/FAIL.
- [ ] **Max absolute deviation from Ghia is under 5% of the lid velocity at all sampled points, for all three Re.**
- [ ] The primary vortex centre location is printed and lies within 2 cells of Ghia's for each Re.
- [ ] The script prints resolution, `tau`, and peak lattice velocity per case; peak stays under 0.1.
- [ ] Rung 1 re-run and still green.

Gate command for **M2**: `myenv/Scripts/python.exe -m validate.cavity --re 100 --re 400 --re 1000`
prints PASS against Ghia et al. (1982).

### Constraints that bite on this task

- **Constraint 1** — D2Q9, BGK single relaxation time, bounce-back walls. No MRT, no cumulant, no
  curved/interpolated boundaries, no turbulence model. Zou–He is a legitimate alternative *wall
  formulation* if momentum-corrected bounce-back misses Ghia; it is not a change of collision model.
- **Constraint 2** — `nu = (tau - 0.5)/3`, via `nu_from_tau` only. Derive the grid from
  `Re = U*L/nu`: pick `U` and `Re`, and `tau` follows. Print the arithmetic, don't guess it.
- **Constraint 3** — lattice velocity under 0.1. At Re 1000 with lid velocity 0.1 the grid must be
  large enough that `tau` stays comfortably above 0.5. Compute the required resolution, print it,
  and warn at setup rather than at `nan` time.
- **Constraint 4** — state is `f` of shape `(9, ny, nx)`, `(direction, y, x)`, `float32`. The nine
  constants come from `lbm/core.py` — never redefined in `lbm/boundary.py` or in `validate/`.
- **Constraint 5** — the ladder is ordered. Rung 1 must still be green at the end of this session,
  and Rung 2 must be green before T004 starts. No relaxing the 5% tolerance to get there.
- **Constraint 6** — do not optimise before Rung 3 passes. No fused collide+stream, no Numba, no
  clever vectorisation. Three Reynolds numbers will feel slow; that is fine.
- **Coding conventions** — type hints with array shapes documented, docstrings citing the
  `DOCS/IDEA2.md` section, preallocate (never allocate inside the step loop), `float32` throughout.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-003 (open — and it is yours to close this session).** Do the two lid corner cells belong to
  the moving lid or to the side walls? `DOCS/TASKS1.md` § T003 § Notes names corner cells as the
  usual cause of a Ghia mismatch. **Decide it explicitly, document it in the `moving_wall`
  docstring, and log it in `DOCS/STATE1.md` § Decisions** — the same way Q-001 was closed in T002:
  by measuring both and printing the comparison, not by arguing.
- **Q-002** (open) — SVG rasterisation dependency for T009. Not yours; not blocking.
- **D-009** — **wall offset.** Bounce-back walls sit **halfway between the last fluid node and the
  solid node**. For solid rows at `y=0` and `y=ny-1`: fluid rows `1..ny-2`, wall planes at `y=0.5`
  and `y=ny-1.5`, so the cavity's characteristic length is `L = n - 2`, not `n` or `n - 1`. Use the
  same convention here and say so — Rung 1 measured the rivals at 14.8% and 12.7% error against
  0.365%, so this is settled, not a preference.
- **D-011** — **timestep order**, and `bounce_back` consumes the copy taken **before collision**:
  `copy f_pre` → `macroscopic` → `force_velocity_shift` → `equilibrium` → `collide` →
  `apply_body_force` → `bounce_back` → `stream`. Reuse this order; changing it silently breaks the
  wall reflection.
- **D-012** — the `float32` residual floor is about `1.7e-6` on `max|du|/peak|u|`. The contract asks
  for `< 1e-6`, which is **at or below that floor**. Expect to have to average over more steps, use
  a longer check interval, or state plainly that the achievable residual is floor-limited and record
  the number you actually reach. Do not silently substitute a looser criterion — log it as a
  decision if you change it.
- **D-005** — velocity is `u` of shape `(2, ny, nx)`, component 0 = `ux`, component 1 = `uy`.
- **D-006** — hot functions take optional preallocated outputs (`work` is `(3, ny, nx)` scratch).
  `moving_wall` should follow the same convention.
- **D-007** — `E` is `int32`; `E_F32` is the float companion. Both in `lbm/core.py`; no third.
- **D-010** — the body force is Guo's scheme, a matched pair: `force_velocity_shift` **and**
  `apply_body_force`. The cavity has no body force, so you will call neither — but if you add any
  forcing, both halves go together.
- **D-003** — `myenv/Scripts/python.exe` is the canonical interpreter. Never bare `python`.
- **D-004** — `Navier-Fluid-Equation/` is read-only prior work. Don't import from it.

### Before you start

- No new package needed. `myenv` has: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  Python 3.11.15. If you do install something, add a row to `DOCS/STATE1.md` § Environment in the
  same session.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves — there is no
  packaging config in Phase 0.
- `validate/` exists now and has an `__init__.py`; `validate/cavity.py` is the new file.
- Rung 1 must be green before you start and again before you finish:
  `myenv/Scripts/python.exe -m validate.poiseuille`.

## Scope discipline

Work only what's in the contract. Geometry primitives are **T004**; inlet/outlet and probes are
**T005**; the runner and rendering are **T006/T007**. A cavity here means three no-slip walls, one
moving lid, and the Ghia comparison — nothing more. Something else genuinely needs doing?
`/new-task` it. If it's under `DOCS/IDEA2.md` § Deliberately deferred, the answer is no.

`DOCS/PLAN1.md` § Risks sets the valve for this task explicitly: **timebox to one session.** If the
cavity centreline is off by more than 5% at the end of it, log it in `DOCS/STATE1.md` § Decisions,
try Zou–He walls, and only then consider proceeding with Rung 2 flagged red — never silently.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate`. Rung 1 and Rung 2 must both be reported with their measured numbers; R3–R4 stay ⬜.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 47 existing tests
   must still pass.
4. If Rung 2 is red at session end, say so plainly, record the measured deviation and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T003 `in_progress`. Do not mark M2 reached.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md` (snapshot, rung status, the Q-003 decision,
   session log), syncs `DOCS/TASKS1.md`, and writes `PROMPTS/004-t004-*.md` for the next session. Do
   not end the session without it.
