# Session 1 — T001: D2Q9 constants, macroscopic, equilibrium

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions,
   session log.
3. `DOCS/TASKS1.md` § T001 — the task contract, in full.
4. `DOCS/IDEA2.md` § **The method, in the order the code runs it** — the constants and the
   equilibrium formula are specified there literally; use them verbatim.
5. `DOCS/PLAN1.md` § Session map and § Why this order — T001 is session 1 of 11; it proves the two
   functions every later task depends on, before any time integration exists.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 0: scaffold only. `CLAUDE.md`, `DOCS/{IDEA2,PLAN1,TASKS1,STATE1}.md`,
  `PROMPTS/`, and `.claude/commands/` were created. **No solver code exists.**
- **Rung status:** R1 ⬜ · R2 ⬜ · R3 ⬜ · R4 ⬜ (none attempted)
- **Milestone reached:** none. Next is M1, at T002.
- **Completed tasks:** none.

## Your task this session

**T001 — D2Q9 constants, macroscopic, equilibrium.** One task, this session only.

Run this first:

    /start-task T001

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: `lbm/core.py` exists and holds the single source of truth for the D2Q9 lattice — constants,
macroscopic reduction, equilibrium distribution. **No time integration this session** (`collide` and
`stream` are T002, and Rung 1 is what proves them). At the end, nothing simulates, but the two
functions everything else depends on are proven correct.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `lbm/__init__.py` and `lbm/core.py` exist; `E`, `W`, `OPP`, `CS2` match `DOCS/IDEA2.md` exactly, in that index order.
- [ ] `W.sum() == 1` to float32 tolerance, and `E[OPP[i]] == -E[i]` for all `i`.
- [ ] `equilibrium(rho, u).sum(axis=0)` equals `rho` to within `1e-5` for random `rho` in `[0.9,1.1]`, random `|u| < 0.1`.
- [ ] First moment holds: `(E.T @ equilibrium(rho,u).reshape(9,-1)).reshape(2,ny,nx)` equals `rho*u` to within `1e-5`.
- [ ] Round trip: `macroscopic(equilibrium(rho, u))` returns the same `rho` and `u` to within `1e-5`.
- [ ] `nu_from_tau(tau)` returns `(tau - 0.5) / 3` and raises `ValueError` naming `tau` when `tau <= 0.5`.
- [ ] All arrays returned are `float32`; asserted in a test.
- [ ] `myenv/Scripts/python.exe -m pytest tests/test_core.py` green.

For reference, the constants as specified in `DOCS/IDEA2.md`:

```
e   = [(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)]
w   = [4/9,   1/9,   1/9,   1/9,    1/9,    1/36,  1/36,   1/36,    1/36  ]
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]
cs2 = 1/3
feq_i = w_i * rho * (1 + 3(e_i·u) + 4.5(e_i·u)^2 - 1.5 u^2)
```

### Constraints that bite on this task

- **Constraint 4** — state is `f` of shape `(9, ny, nx)`, index order `(direction, y, x)`, `float32`.
  Every later module inherits this convention; getting it wrong here is the most expensive possible
  mistake in the project.
- **Constraint 2** — viscosity is not a free parameter: `nu = (tau - 0.5) / 3`. `nu_from_tau` is the
  only path to it. Never write a `nu` setter.
- **Constraint 3** — lattice velocity stays under 0.1 (compressibility error scales as Mach squared).
  That's why the equilibrium tests only probe `|u| < 0.1`; document that in the docstring.
- **Constraint 4 (second half)** — the nine constants live in `lbm/core.py` and are imported from
  there. Never redefined locally, in any later module.
- Docstrings cite the `DOCS/IDEA2.md` section they implement.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-001** (open) — the wall-offset convention for bounce-back (wall on the last fluid node vs
  halfway between). **Not yours to resolve** — it's a T002 decision. Don't pre-empt it.
- **D-003** — `myenv/` is the canonical interpreter. Use `myenv/Scripts/python.exe`, never bare
  `python`, or the project deps silently won't be there.
- **D-004** — `Navier-Fluid-Equation/` is read-only prior work (potential flow, a different method).
  Don't import from it this session.

### Before you start

- `pytest` is **not installed** in `myenv`. Install it: `myenv/Scripts/pip.exe install pytest`, then
  add a row to `DOCS/STATE1.md` § Environment in the same session. A missing row means a future
  session hits an ImportError it can't explain.
- Available already: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, Python 3.11.15.

## Scope discipline

Work only what's in the contract. `collide`, `stream`, bounce-back, and `validate/poiseuille.py` are
**T002** — resist writing `collide` "since it's three lines"; Rung 1 is what proves it and that needs
a full session. Something else genuinely needs doing? `/new-task` it. If it's under `DOCS/IDEA2.md`
§ Deliberately deferred, the answer is no.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. No rungs exist yet, so `/validate` will report all ⬜ — that's the correct result this session.
   Run `myenv/Scripts/python.exe -m pytest` and report the actual output.
3. **Run `/checkpoint`** — it updates `DOCS/STATE1.md` (snapshot, environment row, decisions, session
   log), syncs `DOCS/TASKS1.md`, and writes `PROMPTS/002-t002-*.md` for the next session. Do not end
   the session without it.
