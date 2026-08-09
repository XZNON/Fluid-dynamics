# Session 2 — T002: collide, stream, bounce-back, body force → Rung 1

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-008 landed last session and constrain your API), session log.
3. `DOCS/TASKS1.md` § T002 — the task contract, in full. Also read § T001's outcome note.
4. `DOCS/IDEA2.md` § **The method, in the order the code runs it**, § **Validation ladder** (Rung 1),
   § **Stability** — all three in full.
5. `DOCS/PLAN1.md` § Session map and § Milestone gates — T002 is session 2 of 11 and it is **M1**.
6. `lbm/core.py` — it exists now. Read it before adding to it.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 1: **T001 is `done`.** `lbm/core.py` holds `E`, `W`, `OPP`, `CS2`,
  `E_F32`, `Q`, plus `nu_from_tau`, `macroscopic`, `equilibrium`. `tests/test_core.py` has 21 tests,
  `myenv/Scripts/python.exe -m pytest` → `21 passed`. Moments verified to `~1e-7` against a `1e-5`
  tolerance.
- **Rung status:** R1 ⬜ · R2 ⬜ · R3 ⬜ · R4 ⬜ — `validate/` does not exist yet. **You build Rung 1.**
- **Milestone reached:** none. **M1 is this session's gate.**
- **Completed tasks:** T001.

## Your task this session

**T002 — collide, stream, bounce-back, body force → Rung 1.** One task, this session only.

Run this first:

    /start-task T002

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: the full timestep runs, and Poiseuille flow in an empty channel matches the analytic parabola.
This is **M1**. The pass condition catches every sign error in `collide`, which is why it comes
before anything visual.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `collide(f, feq, tau)` implements `f -= (f - feq) / tau` in place, no allocation.
- [ ] `stream(f, buf)` shifts each `f[i]` by `E[i]` — `roll` on axis 0 by `ey`, axis 1 by `ex` — with the sign convention documented in the docstring and verified by a test that streams a single-cell spike and checks it lands one cell along `E[i]`.
- [ ] `bounce_back` uses the **pre-stream** copy: on solid cells `f[i] = f_pre[OPP[i]]`.
- [ ] `validate/poiseuille.py` runs an empty channel, no-slip top and bottom, constant body force, to steady state, and prints `PASS`/`FAIL` plus the L2 error.
- [ ] **L2 relative error against `u(y) = (G / 2nu) * y * (H - y)` is under 1%.**
- [ ] **Halving `(tau - 0.5)` doubles centreline velocity** to within 2% — asserted in the script, not eyeballed.
- [ ] Mass is conserved: `f.sum()` drifts less than `1e-4` relative over 5000 steps.
- [ ] No `nan` after 20000 steps at `tau = 0.6`.
- [ ] Peak lattice velocity in the run is under 0.1 and the script prints it.

Gate command for **M1**: `myenv/Scripts/python.exe -m validate.poiseuille` prints PASS, L2 under 1%,
and the halving-`(tau-0.5)` doubling check passes.

### Constraints that bite on this task

- **Constraint 2** — viscosity is not a free parameter: `nu = (tau - 0.5) / 3`, via `nu_from_tau`
  only. The pass condition *is* that relation. If the doubling check fails, the bug is in `collide`
  or in the force term, **not** in the analytic solution.
- **Constraint 5** — the ladder is ordered. Rung 1 must be green before T003 starts. No exceptions,
  and no relaxing the 1% tolerance to get there.
- **Constraint 6** — do not optimise before Rung 3 passes. Do **not** fuse collide and stream, however
  tempting; fusion is T010 and only after the cylinder shows the right Strouhal number.
- **Constraint 4** — state is `f` of shape `(9, ny, nx)`, `(direction, y, x)`, `float32`. The nine
  constants come from `lbm/core.py` — never redefined in `lbm/boundary.py` or in `validate/`.
- **Constraint 3** — peak lattice velocity under 0.1; the script prints it, so choose `G` accordingly.
- **Coding conventions** — type hints with array shapes documented, docstrings citing the
  `DOCS/IDEA2.md` section, preallocate (never allocate inside the step loop).

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-001 (open — and it is yours to close this session).** The wall-offset convention for
  bounce-back: does the wall sit on the last fluid node, or halfway between nodes? It changes `H` in
  `u(y) = (G/2nu)*y*(H-y)` by one cell, and it changes `L` in the Rung 2 cavity benchmark. **Decide
  it, document it in the `bounce_back` docstring, and log it in `DOCS/STATE1.md` § Decisions.** From
  the contract notes: if L2 error sits stubbornly near 2–3%, suspect this before suspecting
  `collide` — the classic fix is `H = ny - 1` vs `H = ny`; try both and record which one the code
  assumes.
- **Q-002** (open) — SVG rasterisation dependency for T009. Not yours; not blocking.
- **D-005** — velocity is `u` of shape `(2, ny, nx)`, component 0 = `ux`, component 1 = `uy`, matching
  the `(ex, ey)` column order of `E`. The body force term must use the same order.
- **D-006** — `macroscopic(f, rho=None, u=None)` and `equilibrium(rho, u, feq=None, work=None)` accept
  optional preallocated outputs (`work` is `(3, ny, nx)` scratch). Use that path in the step loop so
  nothing allocates; `collide` and `stream` should follow the same convention.
- **D-007** — `E` is `int32` (use it for the `roll` offsets); `E_F32` is the float companion. Both are
  in `lbm/core.py`; do not create a third.
- **D-003** — `myenv/Scripts/python.exe` is the canonical interpreter. Never bare `python`.
- **D-004** — `Navier-Fluid-Equation/` is read-only prior work (potential flow, a different method).
  Don't import from it.

### Before you start

- No new package needed. `myenv` has: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  Python 3.11.15. If you do install something, add a row to `DOCS/STATE1.md` § Environment in the
  same session.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves — there is no
  packaging config in Phase 0.
- `validate/` does not exist yet. It needs an `__init__.py` for `python -m validate.poiseuille` to
  work.

## Scope discipline

Work only what's in the contract. The moving-lid BC and the cavity benchmark are **T003**; geometry
primitives are **T004**; inlet/outlet and probes are **T005**. A channel here means two no-slip rows
and a body force — nothing more. Something else genuinely needs doing? `/new-task` it. If it's under
`DOCS/IDEA2.md` § Deliberately deferred, the answer is no.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate`. Rung 1 must be green and reported with its measured numbers; R2–R4 stay ⬜.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — T001's 21 tests must
   still pass.
4. If Rung 1 is red at session end, say so plainly, record the measured L2 error and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T002 `in_progress`. Do not mark M1 reached.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md` (snapshot, rung status, the Q-001 decision,
   session log), syncs `DOCS/TASKS1.md`, and writes `PROMPTS/003-t003-*.md` for the next session. Do
   not end the session without it.
