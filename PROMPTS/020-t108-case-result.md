# Session 20 — T108: `flow.Case` / `flow.Result` — the front door

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator in
pure NumPy, spec `DOCS/IDEA2.md`, closed at M4 with all four validation rungs green. **Phase 1 is
live**: the product layer above the solver — a `flow/` package plus a CLI, on a Warp GPU backend.
Spec `DOCS/IDEA3.md`, plan `DOCS/PLAN2.md`, backlog `DOCS/TASKS2.md`, live status `DOCS/STATE2.md`.

## Read these first, in this order

1. `CLAUDE.md` — the 16 hard constraints (Phase 1 list), session protocol, conventions.
2. `DOCS/STATE2.md` — **in full**: snapshot, the one live blocker, open questions, decisions
   D-041 … D-066, and at minimum the session 18 and 19 log entries.
3. `DOCS/TASKS2.md` § **T108** — the task contract, in full. Also the backlog index row: **T109
   depends on you**, and you are its only open dependency.
4. `DOCS/IDEA3.md` § **What Phase 1 is, concretely** (the three lines that must actually run) and
   § The five things Phase 1 must get right, **item 4 in full** ("Results render themselves").
5. `DOCS/PLAN2.md` § Session map (you are session 20) and § Risks.
6. `old-Docs/STATE1.md` **D-023**, **D-024**, **D-028**, **D-030**, **D-039** — read the entry each
   is cited for, not the whole file.
7. The four modules you are the front door for, **none of which you should duplicate**:
   `flow/quantity.py` + `flow/fluids.py` (T104), `flow/autoconfig.py` (T105 — `plan()`, `Plan`,
   `Unrepresentable`, `Suggestion`), `flow/diagnose.py` (T106 — `explain`, `suggest`,
   `apply_suggestion`, `Monitor`), `flow/prepare.py` (T107 — `prepare()`, `Prepared`, `Fix`,
   `apply_fix`).
8. `lbm/runner.py` — `Sim`, `SimConfig`, `demo_domain`, and the `__main__` CLI. **This is the
   reference for behaviour that already works**: geometry placed at body scale in a domain sized in
   its own diameters, the startup kick, the printed summary. Port the *behaviour*; the assembly now
   lives in `Case`.
9. `lbm/record.py` and `lbm/render.py` — the sinks and the one `render()`. `flow/` composes them and
   **colours nothing**.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 19: **T107 is done and M7 is reached.** `flow/prepare.py` exists —
  `prepare(source, cells_across, *, repair=...) -> Prepared`, four individually switchable repairs
  each reporting itself in `actions`, refusals carrying an executable `Fix` — plus a committed
  15-image corpus in `tests/data/shapes/` with its generator and `expectations.json`, and
  `validate/shapes.py` = **Rung C, PASS in 7.8 s**. `pytest` prints **663 passed, 1 skipped**.
- **Q-102 was closed by measurement (D-064).** `flow.prepare.thin_branch_depth` detects a hairline
  *fused* to a thick body (where `min_thickness` reports the body and is blind) and does not
  false-alarm on a plain disc at any radius or sub-cell offset.
- **Phase 1 rung status: A 🟩 · B 🟩 on numpy / 🟥 on warp · C 🟩 · D 🟩 · E ⬜.**
- **Phase 0 rung status: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩** — all four re-run in session 19, printing
  session 11/15/18's digits: R1 L2 0.3650%, R2 0.75% / 0.21 cells, R3 St 0.1731 Cd 1.4031 ± 0.0086,
  R4 square Cd 1.5279 ± 0.0271 and polygon Cd 1.4276 ± 0.0226.
- **Milestones reached: M5, M6, M7.** M8 is the last one in Phase 1 and belongs to T110.
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101, T102, T103, T104, T105, T106, **T107**.

## Your task this session

**T108 — `flow.Case` / `flow.Result` API.** One task, this session only. Gate: unit tests (no new
rung). It is the front door over everything T104–T107 built.

Run this first:

    /start-task T108

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** the three lines from `DOCS/IDEA3.md` § What Phase 1 is, concretely actually run.
Everything decided in T104–T107 gets one front door.

**Inputs:** an image path plus physical strings.
**Outputs:** `flow/case.py::Case` (`from_image`, `from_array`, `explain()`, `plan`, `run(...)`) and
`flow/report.py::Result` (`cd`, `cl`, `strouhal`, `convergence`, `peak_u`, `elapsed`, `substituted`,
`frames`, `save(path)`, `summary()`, `plot()`).

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `Case.from_image("x.png", fluid="air", speed="5 m/s", size="0.1 m")` builds without running anything, and `explain()` prints the plan, every `why` line, the geometry verdict and actions, and the estimated wall clock.
- [ ] **No lattice quantity appears in any public signature of `flow/`** — asserted by an introspection test over every public callable's annotations and defaults (Phase 1 constraint 13).
- [ ] `run()` accepts `live=`, `record=`, `headless=` and composes them through `TeeSink`, selecting `drop` by **D-039** (any file-writing sink ⇒ `drop=False`); a test asserts the mode chosen for each combination.
- [ ] `Result.save("wake.mp4")` and `.save("frames/")` both work and go through `lbm.record`; `flow/` **colours nothing** — asserted by a test that no `flow` module imports a colormap or builds RGB (constraint 10).
- [ ] Solid cells are seeded at rest (**D-030**) and a test asserts the body interior holds the rest state after 300 steps.
- [ ] `Result.summary()` prints Cd (mean ± std), Cl amplitude, St with its confidence, peak `|u|` against 0.1, convergence, elapsed, backend, and — if applicable — the substitution banner.
- [ ] `Result.strouhal` is `None`, not a number, when shedding is not detected (Cl amplitude below 1% of Cd); a test covers a steady case at Re 10.
- [ ] **Never a silent substitution** (constraint 16) — **inherited from T106, see D-062**: a test asserts that a `Result` produced from a `flow.diagnose` suggestion carries `substituted=True` and that the flag reaches the printed summary *and* the recorded video's metadata. T106 shipped the half that could exist without `Result` (every case-changing suggestion carries "not your case" on the object, asserted by `tests/test_diagnose.py`); this is the other half.
- [ ] `pytest tests/test_case.py tests/test_report.py` green; Phase 0 rungs still green.

### Constraints that bite on this task

- **Constraint 13** — no lattice quantity in any public `flow/` signature. `tests/test_flow_package.py`
  scans every public callable reachable through each module's `__all__` against `LATTICE_NAMES`
  (`tau`, `u_lattice`, `nx`, `ny`, `dx`, `dt`, `cells`, `cells_per_length`, `steps_per_frame`,
  `n_steps`, …). **D-060** limits the frozen-dataclass exemption to *auto-generated* constructors —
  a hand-written `__init__` is still scanned. `Result` should therefore be a frozen dataclass if it
  is to carry `dx`/`dt`-shaped fields; `Case` is **not** a result and gets no exemption, so its
  constructor and `run()` must speak physics only. The one number T107 was allowed to cross with is
  `cells_across`, and it is `prepare`'s argument, not `Case`'s.
- **Constraint 10** — one `render()`, three sinks, and `flow/` colours nothing. `Case.run()`
  composes `lbm.render` / `lbm.record` sinks; it does not build RGB. `Result.plot()` is a matplotlib
  figure, which is a *different* kind of output from a frame, and must say so rather than becoming a
  fourth renderer.
- **Constraint 16 / D-045 / D-062** — `substituted` is carried **on `Result`**, not in one rendering
  of it, so the printed summary, the report and the video metadata all get it. This is the criterion
  D-062 deliberately carried forward from T106 to you.
- **Constraint 8 / D-039** — never block the sim on the display; drop *display* frames, never steps,
  and **any file-writing sink takes `drop=False`**.
- **Constraint 7 / D-023** — `steps_per_frame` is computed from `dt` through the `Plan`, never
  hardcoded.
- **Constraint 15** — `flow/` may import `lbm/`; `lbm/` may never import `flow/`. An AST scan and a
  runtime scan assert it.
- **Constraint 5** — the ladder stays a gate. T108 adds no rung, but all four Phase 0 rungs plus A,
  C and D must still be green when you finish.

### Blockers, open questions and decisions that affect you

**Blocker — one rung is red, and it is not yours to fix:**

- **Rung B fails its accuracy check on `--backend warp`.** `Plan.estimated_seconds` predicts
  **5.61 s** against an actual **3.19 s** — **75.7%** against the rung's 25% limit, measured on an
  idle machine. The 24-case sweep passes on **both** backends with identical digits, and **Rung B on
  numpy is PASS** (predicted 61.05 s, actual 56.46 s, **8.1%**). Pre-existing: Rung B had never been
  run on warp before session 19, and **D-059** calibrated the estimator against the numpy column of
  `DOCS/STATE2.md` § Performance baseline. Queued as `e4874a146490`. **It does not block you** —
  T108 times nothing — but note that your `explain()` prints an estimated wall clock, and on warp
  that estimate is currently ~1.8x too long. Say so honestly in the output if you print it; do not
  fix the estimator here (that is **T110**'s, because M8's gate is a wall clock on warp).

**Open questions:**

- **Q-101** — does `python -m lbm.runner` survive as a working entry point once `python -m flow`
  exists, or become a pointer to it? **T109 decides**, not you — but your `Case` is what T109 will
  wrap, so build it so either answer stays cheap.
- ~~Q-102~~ and ~~Q-103~~ are closed (**D-064**, **D-056**).

**Decisions that constrain this session:**

- **D-062** — the `substituted=True` criterion was carried from T106 to **you**, explicitly and with
  a named owner. T106 shipped the half that could exist: every case-changing `Suggestion` carries
  *"This is a different flow from the one you asked for -- not your case."* on the object. Your half
  is `Result.substituted` reaching the summary and the video metadata.
- **D-060** — the constraint-13 scan exempts a frozen dataclass's auto-generated constructor and
  nothing else. `test_the_frozen_dataclass_exemption_is_narrow` proves it.
- **D-039** — two run modes; any file-writing sink takes `drop=False`.
- **D-030** — solid cells are seeded at rest before the first timestep.
- **D-028** — colour limits are fixed and symmetric, and come from the `Plan`
  (`Plan.vorticity_limit`), not from per-frame autoscaling.
- **D-023** — `steps_per_frame` is computed from target playback speed via `dt`.
- **D-024** — the two run modes Phase 0 settled on; `Case.run()` composes them rather than inventing
  a third.
- **D-065** (session 19) — `prepare()` **refuses** a body size the picture cannot deliver rather
  than quietly returning a different one. `Case` must surface that refusal through the same path as
  an `Unrepresentable`, not swallow it: a `Prepared` with `verdict="refused"` carries `reason` and an
  executable `fix`, and `flow.prepare.apply_fix` is its `apply_suggestion`.
- **D-063** — a suggestion is a testable claim. Anything `Case.explain()` offers as a way forward
  should be executable by the same machinery Rung D and Rung C use.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` was the last addition).
  If you want `scipy` or anything else, that is a real decision and needs a row in
  `DOCS/STATE2.md` § Environment **in the same session**.
- Confirm the starting point: `myenv/Scripts/python.exe -m pytest` should print **663 passed,
  1 skipped**, `myenv/Scripts/python.exe -m validate.shapes` should print **PASS** (~8 s), and
  `myenv/Scripts/python.exe -m validate.refusals` should print **PASS** (~9 min).
- **`Case` is a facade.** If logic accumulates in it that is not delegation, it belongs in
  `autoconfig`, `diagnose` or `prepare`, where it can be tested without a run. This is the T108
  Notes' one instruction and it is the thing most likely to go wrong.
- Watch `.gitignore`: it drops `*/__init__.py` and `tools/` (queued issue `495777c58269`, still
  open). Session 19's new files were checked and are clean, but run `git status` on anything you add
  and confirm it is actually tracked.
- **Long commands:** session 19 found that any single command over ~10 minutes gets moved to the
  background and may be killed, and that killed runs can leave orphaned processes that contaminate
  later timings (**D-035** caught it — Rung B read 92 s dirty vs 56 s clean). Run long rungs
  detached with `nohup ... &` and `python -u`, check `ps -W | grep "Fluid Mech/myenv"` before
  trusting any timing, and kill orphans.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no. **This session does not write the CLI** (`python -m flow` is T109) and does not fix Rung B's
warp estimator (T110). A change `lbm/` genuinely needs is a `/new-task` naming the Phase 0 rung it
must re-prove (`DOCS/PLAN2.md` § Risks, last row) — never folded into a product task.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `pytest tests/test_case.py tests/test_report.py`, then the whole suite.
3. Run `/validate` for every rung at or below this task — all four Phase 0 rungs, Rung A, Rung C,
   Rung D. Nothing may regress. (Rung B's warp accuracy check is already red and is recorded as
   such; do not let it hide a *new* failure — check the sweep line, not just PASS/FAIL.)
4. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
