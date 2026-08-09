# STATE1.md — live project state

**Read this first, every session, in full.** Updated by `/checkpoint` at the end of every session.
Never rewrite or condense the session log — append only.

---

## Snapshot

| Field | Value |
|---|---|
| **Phase** | Phase 0 — D2Q9 LBM in NumPy (`DOCS/IDEA2.md`) |
| **Current task** | `T003` |
| **Task status** | `not_started` |
| **Completed tasks** | T001, T002 |
| **Milestone reached** | **M1** (2026-08-10, gate run: `python -m validate.poiseuille` → PASS, L2 0.3650%, doubling ratio 1.99940) — next: M2 at T003 |
| **Rung status** | R1 🟩 · R2 ⬜ · R3 ⬜ · R4 ⬜ |
| **Last updated** | 2026-08-10 — session 2 (T002 done, M1 reached) |

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

None.

## Open questions

- ~~**Q-001** — wall-offset convention for bounce-back.~~ **Closed in session 2 by measurement** —
  see **D-009**. The wall sits halfway between the last fluid node and the solid node.
- **Q-002** — SVG rasterisation dependency (T009) not chosen. Not blocking; PNG is what M4 needs.
- **Q-003** (new, T003) — do the two lid corner cells belong to the moving lid or to the side walls?
  `DOCS/TASKS1.md` § T003 § Notes flags corner cells as the usual cause of a Ghia mismatch. Decide it
  explicitly in T003 and log it, the same way Q-001 was closed. Not blocking anything before T003.

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
