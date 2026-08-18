# Session 14 — T102: Warp kernels — equilibrium, collide, stream

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator in
pure NumPy, spec `DOCS/IDEA2.md`, closed at M4 on 2026-08-13 with all four validation rungs green.

**Phase 1 is live**: the product layer above that solver — a `flow/` package plus a CLI, on a Warp
GPU backend, ten tasks `T101` → `T110`, five new validation rungs A–E, milestones M5 → M8. Spec
`DOCS/IDEA3.md`, plan `DOCS/PLAN2.md`.

**This is the session that introduces Warp, and Warp is the only new thing in it.** T101 already
built the seam it slots into, precisely so this session is not also a refactor.

## Read these first, in this order

1. `CLAUDE.md` — hard constraints, session protocol, conventions. **The constraint list is now the
   Phase 1 one** (16 entries, folded in from D-046 by T101); it is the authority, not `DOCS/STATE2.md`
   D-046.
2. `DOCS/STATE2.md` — **in full**: snapshot, blockers, open questions, environment, performance
   baseline, decisions D-041 … D-051, and the session 12 and 13 log entries.
3. `DOCS/TASKS2.md` § T102 — the task contract, in full. Also read § T103, which is where the rest of
   the port lands, so you know what is *not* yours this session.
4. `DOCS/IDEA3.md` § Performance budget · § Validation ladder (Rung A's row) · § Deliberately
   deferred.
5. `old-Docs/STATE1.md` § Performance baseline, and **only these decisions**: **D-008** (`usq`
   hoisting in `equilibrium`), **D-035** (how any speed number must be measured). STATE1 is
   **frozen** — read it, never edit it (**D-041**).
6. `DOCS/PLAN2.md` § Dependency graph, § Session map and § Risks — the first two risk rows are yours.
7. `lbm/backends/__init__.py` — the `Backend` protocol you are implementing, and its module docstring
   on what a backend owns and what it does not.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 13: **T101 is done.** `lbm/backends/` now holds the `Backend`
  protocol (`typing.Protocol`, `@runtime_checkable`), `registry.py` and `numpy_backend.py`;
  `SimConfig.backend` defaults to `"numpy"`; `Sim` reaches every kernel through `self.backend` and
  `lbm/runner.py` imports **no** kernel directly. Nothing got faster and no physics moved.
- **Phase 0 rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — re-run at the end of session 13 and
  identical to session 11 to every printed digit: R1 L2 **0.3650%** · R2 **0.75% / 0.42% / 1.01%** ·
  R3 **St 0.1731, Cd 1.4031 ± 0.0086** · R4 square **Cd 1.5279 ± 0.0271**, polygon
  **Cd 1.4276 ± 0.0226**. These are the numbers the GPU must land inside the *bands* of (not
  reproduce exactly — see below).
- **Phase 1 rung status:** A ⬜ · B ⬜ · C ⬜ · D ⬜ · E ⬜. **Rung A's harness is this session's
  deliverable**, built before the code it validates.
- **Milestone reached:** M4 (2026-08-13). M5 is T103's, not yours.
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101.
- `myenv/Scripts/python.exe -m pytest` → **389 passed** at the end of session 13.

## Your task this session

**T102 — Warp kernels: equilibrium, collide, stream.** One task, this session only.

Run this first:

    /start-task T102

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** the three hot kernels run on the GPU and produce what NumPy produces. `equilibrium` is
**39.9 ms of a ~75 ms step at 1M cells** — over half — so it is written first.

**Outputs:** `lbm/backends/warp_backend.py` implementing `equilibrium`, `collide`, `stream`,
`macroscopic`, `to_host`, `from_host`; `validate/parity.py` with a `--kernels` mode printing
PASS/FAIL per kernel.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `warp` installed into `myenv` and recorded in `DOCS/STATE2.md` § Environment with its version and the CUDA/driver version it found.
- [ ] The nine D2Q9 constants come from `lbm/core.py` and are uploaded to the device once at backend construction — **not redefined in a Warp kernel** (constraint 4 / "no physics constant twice").
- [ ] `validate/parity.py --kernels` compares each of `equilibrium`, `collide`, `stream`, `macroscopic` against the NumPy backend on random `rho ∈ [0.9, 1.1]`, `|u| ≤ 0.099`, at 3 grid sizes, and prints the max absolute difference per kernel.
- [ ] **Per-kernel agreement: max absolute difference ≤ 1e-6 in `f` units** — and the script prints the number, so a later regression is visible rather than merely passing. A difference that is not explainable by float ordering fails the task; do not widen the tolerance.
- [ ] `stream` is verified independently of parity by the Phase 0 spike test: a single-cell spike lands one cell along `E[i]`, for all 9 directions, on the GPU.
- [ ] No allocation per call: the backend preallocates its device buffers at construction; a test runs 1000 steps' worth of kernel calls and asserts device memory is flat.
- [ ] `myenv/Scripts/python.exe -m pytest` green; Phase 0 rungs unaffected (they still run `--backend numpy`).

### Constraints that bite on this task

From `CLAUDE.md` § Hard constraints, in their Phase 1 form:

- **Constraint 1** — *the physics does not change in Phase 1.* The arithmetic is a transcription of
  `lbm/core.py`, **term for term and in the same order**. If a kernel reads "clearer" written
  differently, write it the same way and note the difference in a comment. Nothing from
  `DOCS/IDEA2.md` § Deliberately deferred gets un-deferred here.
- **Constraint 4** — the backend owns its device layout, but `to_host` must yield `(9, ny, nx)`
  `float32`. The nine constants (`e`, `w`, `opp`, `cs2`) come from `lbm/core.py` and are **uploaded
  once at construction**, never redefined in a kernel.
- **Constraint 2** — `nu = (tau - 0.5)/3` still lives in `nu_from_tau`. The kernel takes `omega` and
  does **not** re-derive it.
- **Constraint 5** — the ladder is ordered. Rung A is this task's gate; the four Phase 0 rungs must
  stay green on NumPy (they are unaffected — they run the default backend).
- **Constraint 6's replacement** — *no backend optimisation before its parity rung passes.* Get the
  Warp kernels **correct** first; the performance table is T103's, and tuning a kernel that has no
  green Rung A is how a wrong answer gets fast.
- **Constraint 11** — bit-identical restart holds **within** a backend. GPU/CPU bit-identity is
  neither achievable nor the goal: **explainable** agreement is. Where a fused multiply-add changes
  the last bits, say so *with the measured magnitude*. Cross-backend tolerance becomes a printed
  number in T103.

### Blockers, open questions and decisions that affect you

**Blockers:** none.

**Open questions:**

- **Q-103** — what tolerance does cross-backend whole-step agreement actually need? T103 sets
  `max|Δu|/U < 1e-4` at 1000 steps as the contract; whether that is achievable or generous is
  unknown until the port runs. **It is a pass condition to be met, not adjusted.** Your per-kernel
  ≤ 1e-6 is the evidence that will decide whether it is easy or tight — measure and print it, do not
  round it.

**Decisions that constrain this session:**

- **D-051** (new, session 13) — **the `Backend` protocol covers kernels and the two host transfers
  and nothing else.** Buffer allocation, the open boundaries, the Guo body force and the probes are
  outside it, and `Sim` still allocates its own `(9, ny, nx)` NumPy buffers with `np.empty`. **You
  will have to widen the seam, starting with allocation** — that is expected, budgeted work, not a
  defect of T101, and it is the first thing you will hit. Widen it as narrowly as two
  implementations now justify.
- **D-050** (new, session 13) — the checkpoint is still exactly `f` / `solid` / `step_count` /
  config + `format: 1` (**D-022**). `f` is written through `backend.to_host`, so it is always the
  portable host layout, and the backend name rides inside the pickled config;
  `load_checkpoint(path, backend=...)` overrides it. A Warp checkpoint must therefore stay loadable
  on NumPy. **Checkpointing on the GPU is T103's criterion**, not yours — do not break it either.
- **D-043** — **NumPy is kept as the reference oracle, not replaced.** A GPU that disagrees with
  NumPy is a broken backend, never a new answer. If a design choice makes NumPy feel like a legacy
  path, it is the wrong choice.
- **D-035** — any speed number you quote comes with `Win32_Processor.CurrentClockSpeed`, the power
  state and the GPU name beside it, and A/B by alternating rounds. Session 13's machine state, for
  comparison: AMD Ryzen 7 5800H at **3201 MHz** of a 3201 MHz maximum, on **mains**. **You are not
  expected to quote a speed number at all this session** — that is T103's.
- **D-008** — `usq` is hoisted out of the direction loop in `equilibrium`; the transcription keeps
  that structure.
- **D-042** — nothing in this task creates `flow/`, and `lbm/` may never import it.

### Before you start

- **Install `warp-lang`**: `myenv/Scripts/pip.exe install warp-lang`, then **add a row to
  `DOCS/STATE2.md` § Environment in the same session** with its version and the CUDA/driver version
  it reports. A missing row is the next session's unexplainable ImportError. The registry already
  knows the name — `lbm/backends/registry.py` maps `"warp"` to
  `lbm.backends.warp_backend.WarpBackend` and prints that install line when it is missing, so
  **you do not edit the registry table**; you write the module it already points at.
- Everything else is present: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  pygame 2.6.1, imageio 2.37.4, imageio-ffmpeg 0.6.0, psutil 7.2.2, Python 3.11.15.
- Machine: RTX 3050 4 GB, 16 GB RAM.
- Confirm the starting point: `myenv/Scripts/python.exe -m pytest` should print **389 passed**
  before you change anything.
- The four Phase 0 rungs do **not** need re-running this session — they run the NumPy backend and
  nothing you write touches it. Re-run `pytest` instead, and Rung A.

**The pressure valve, read it before you start, not at hour three:** `DOCS/PLAN2.md` § Risks — if
Warp will not install or will not run a kernel **in the first half of this session**, log the
blocker in `DOCS/STATE2.md` and fall through to **T104** (physical quantities + fluid library, which
is independent of the GPU work) in the same session. Try Taichi in the next GPU session rather than
spending a third one on the install. And the first risk row is `idea.md`'s "The trap": Phase 1
becoming a solver-optimisation project. **If T102 or T103 overruns by one session, the port is
demoted back to Phase 2** and Phase 1 continues on NumPy — T101's seam makes that a config change.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no.

Specifically for this task: the **boundaries** (`bounce_back`, the inlet, the outlet), the fused
`collide_stream`, checkpointing on the GPU and the performance table are all **T103**. Four kernels
and a parity script is the whole job.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `myenv/Scripts/python.exe -m validate.parity --kernels` — **Rung A (kernels)** — and print the
   measured max absolute difference per kernel, not just PASS.
3. Run `myenv/Scripts/python.exe -m pytest` and confirm it is green.
4. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
