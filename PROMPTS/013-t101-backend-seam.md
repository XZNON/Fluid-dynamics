# Session 13 — T101: backend seam, NumPy behind it

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator in
pure NumPy, spec `DOCS/IDEA2.md`, closed at M4 on 2026-08-13 with all four validation rungs green.

**Phase 1 is live and this is its first task.** Phase 1 builds the product layer above that solver:
a `flow/` package plus a CLI, on a Warp GPU backend, ten tasks `T101` → `T110`, five new validation
rungs A–E, milestones M5 → M8. Its spec is `DOCS/IDEA3.md`, planned in session 12.

**This session writes no product code and makes nothing faster.** It introduces the seam that the
GPU backend will later slot into, and its acceptance criterion is that every Phase 0 number comes
back *identical*.

## Read these first, in this order

1. `CLAUDE.md` — hard constraints, session protocol, conventions. Note § Session protocol now points
   at the Phase 1 documents, and § Current state says the 12 constraints' Phase 1 fates live in
   `DOCS/STATE2.md` **D-046**.
2. `DOCS/STATE2.md` — **in full**. It is short (one session old): snapshot, blockers, open questions,
   environment, performance baseline, decisions D-041 … D-048 with the constraint fate table, and
   the session-12 log entry.
3. `DOCS/TASKS2.md` § T101 — the task contract, in full.
4. `DOCS/IDEA3.md` § What Phase 1 is, concretely · § Performance budget · § Deliberately deferred.
5. `old-Docs/STATE1.md` § Decisions — **only these five entries**, which this task must not break:
   **D-011** (timestep order), **D-020** (which two snapshots `probe.forces` consumes, and the extra
   runner buffer), **D-022** (the checkpoint is exactly `f` / `solid` / `step_count` / config plus a
   `format` integer), **D-033** (`collide_stream` fuses collide + bounce-back + snapshot + stream,
   and why it must cross `bounce_back`), **D-035** (how any speed number must be measured).
   STATE1 is **frozen** — read it, never edit it (**D-041**).
6. `DOCS/PLAN2.md` § Why this order and § Risks — T101 exists so that the session introducing Warp
   introduces *only* Warp.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 12: it planned Phase 1 and wrote four documents (`DOCS/IDEA3.md`,
  `DOCS/PLAN2.md`, `DOCS/TASKS2.md`, `DOCS/STATE2.md`) plus a pointer edit to `CLAUDE.md`. **No
  code was changed.**
- **Phase 0 rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — the ladder is complete. Session 11's
  published numbers, which this task must reproduce **to every printed digit**:
  R1 L2 **0.3650%** · R2 **0.75% / 0.42% / 1.01%** · R3 **St 0.1731, Cd 1.4031 ± 0.0086** ·
  R4 square **Cd 1.5279 ± 0.0271**, polygon **Cd 1.4276 ± 0.0226**.
- **Phase 1 rung status:** A ⬜ · B ⬜ · C ⬜ · D ⬜ · E ⬜ — no script exists yet.
- **Milestone reached:** M4 (2026-08-13). Phase 1 targets M5 → M8.
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: none.
- `myenv/Scripts/python.exe -m pytest` → **367 passed** (session 12, re-run and confirmed).

## Your task this session

**T101 — Backend seam, NumPy behind it.** One task, this session only.

Run this first:

    /start-task T101

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** `Sim` stops calling `lbm.core` directly and calls a **backend** instead. Only the NumPy
backend exists at the end of this session, and every Phase 0 rung prints the same numbers it printed
in session 11. Nothing gets faster; the seam is the deliverable.

**Outputs:** `lbm/backends/__init__.py` with a `Backend` protocol; `lbm/backends/numpy_backend.py`
implementing it over the existing `lbm.core` functions; `SimConfig.backend: str = "numpy"`;
`lbm/backends/registry.py` mapping a name to an implementation and raising a message naming the
install line for an unavailable one.

The protocol covers, at minimum: `equilibrium(rho, u, feq, work)`, `collide(f, feq, tau)`,
`stream(f, buf)`, `collide_stream(f, feq, tau, solid, f_bb, buf)`, `macroscopic(f, rho, u)`,
`bounce_back(f, f_pre, solid)`, `to_host(f) -> NDArray[np.float32] (9, ny, nx)`,
`from_host(arr) -> backend array`.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `Backend` is a `typing.Protocol` (or ABC) with every method above, each documented with its array shapes; `lbm/core.py`'s functions are unchanged and the NumPy backend delegates to them.
- [ ] `SimConfig(backend="numpy")` is the default and `Sim` reaches every kernel through `self.backend`; a test asserts no module in `lbm/runner.py` imports `lbm.core`'s kernels directly any more (import-level assertion, not a comment).
- [ ] `to_host` / `from_host` round-trip a `(9, ny, nx)` `float32` array bit-identically on the NumPy backend; asserted with `np.array_equal`.
- [ ] An unknown backend name raises `ValueError` naming the requested backend and listing the available ones.
- [ ] **Restart is still bit-identical** — T006's test passes unchanged, plus a new one that checkpoints and resumes through the seam.
- [ ] `myenv/Scripts/python.exe -m pytest` green with **no existing test modified**.
- [ ] **All four Phase 0 rungs re-run and print numbers identical to session 11 to every printed digit** — R1 L2 0.3650% · R2 0.75% / 0.42% / 1.01% · R3 St 0.1731, Cd 1.4031 · R4 square Cd 1.5279, polygon Cd 1.4276.

### Constraints that bite on this task

From `CLAUDE.md` § Hard constraints, as amended by `DOCS/STATE2.md` **D-046**:

- **Constraint 4, in its Phase 1 form** — the backend owns its state layout, but `to_host` must
  produce `(9, ny, nx)` `float32`. That is what makes checkpoints and cross-backend parity portable
  later. The nine constants (`e`, `w`, `opp`, `cs2`) still come from `lbm/core.py` only — never
  redefined, in any backend.
- **Constraint 11** — restart is bit-identical **within** a backend. If the seam changes float
  ordering anywhere, the seam is wrong, not the test. (Cross-backend agreement becomes a printed
  tolerance in T103; it does not apply this session, because there is only one backend.)
- **Constraint 5** — the validation ladder is ordered and non-negotiable. All four Phase 0 rungs are
  this task's gate, and a rung that goes red blocks T102.
- **Constraint 1, in its Phase 1 form** — the physics is unchanged through Phase 1. This task adds
  indirection and **no new physics**. D2Q9, BGK, bounce-back; nothing from `DOCS/IDEA2.md`
  § Deliberately deferred gets un-deferred here.
- **Retired: constraint 6** ("do not optimise before Rung 3 passes") — spent in session 7, replaced
  by *no backend optimisation before its parity rung passes*. There is no parity rung yet, so **do
  not optimise anything this session**.

### Blockers, open questions and decisions that affect you

**Blockers:** none.

**Open questions** (from `DOCS/STATE2.md`) — none of them are yours to close, but Q-103 is shaped by
what you build:

- **Q-103** — what tolerance does cross-backend whole-step agreement actually need? T103 sets
  `max|Δu|/U < 1e-4` at 1000 steps as the contract; whether that is achievable or generous is
  unknown until the port runs. Design `to_host`/`from_host` so that question is *measurable* — it is
  the hook the parity rung hangs off.

**Decisions that constrain this session:**

- **D-042** — the product layer will be a new top-level package `flow/`; `flow/` may import `lbm/`,
  `lbm/` may **never** import `flow/`. Nothing in this task creates `flow/`, but do not add anything
  to `lbm/` that anticipates it.
- **D-043** — the Warp port is T102–T103, and **NumPy is kept as the reference oracle, not
  replaced**. The seam exists to make both first-class. If a design choice makes NumPy feel like a
  legacy path, it is the wrong choice.
- **D-011** — the timestep order is: copy `f_pre` → `macroscopic` → `force_velocity_shift` →
  `equilibrium` → `collide` → `apply_body_force` → `bounce_back` → `stream`, with a tail of
  `outlet_zero_gradient` → `inlet_velocity` after `stream` (**D-020**). The seam must not reorder it.
- **D-033** — `collide_stream` fuses collide + bounce-back + the `f_bb` snapshot + stream, because
  D-020's order puts the reflection *between* collide and stream. A body force falls back to the
  unfused path. Both paths are bitwise equal and are asserted so; keep both selectable through the
  seam.
- **D-022** — the checkpoint is exactly `f`, `solid`, `step_count`, config, plus a `format: 1`
  integer. Adding the backend name to it is a real decision with a real consequence (a checkpoint
  written on one backend must be loadable on another) — if you make it, record it as **D-049**.
- **D-035** — if you quote any steps/s number at all, quote `Win32_Processor.CurrentClockSpeed` and
  the power state beside it, and A/B by alternating rounds. This task is not expected to change
  performance; a measurable *slowdown* from the indirection is worth reporting.

### Before you start

- **Nothing to install.** `myenv` has numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  pygame 2.6.1, imageio 2.37.4, imageio-ffmpeg 0.6.0, psutil 7.2.2, Python 3.11.15. **Warp is
  T102's**, not this session's — installing it early is scope creep.
- Confirm the starting point is what the last session left: `myenv/Scripts/python.exe -m pytest`
  should print **367 passed** before you change anything.
- The full ladder is ~55 minutes and R4 alone is ~40 of them. **Start it early**, in the background,
  the way session 11 did — you need it green at the end, and a rung discovered red in the last ten
  minutes costs the session.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no.

Specifically for this task: **do not make the protocol general enough for XLB "while we're here."**
Two implementations is the number that reveals the right seam; one plus a guess is not.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate` — all four Phase 0 rungs — and confirm every number is identical to session 11,
   not merely inside its band. "Identical" is the criterion here precisely because this task should
   change nothing.
3. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
