# STATE2.md — live project state, Phase 1

**Read this first, every session, in full.** Updated by `/checkpoint` at the end of every session.
Never rewrite or condense the session log — append only.

**Phase 0's state file is `old-Docs/STATE1.md` and it is frozen** (**D-041**). Its § Decisions
(D-005 … D-040) remain in force and are cited by number throughout this file; its session log is
history and is never edited. Decision numbering continues here at **D-041**.

---

## Snapshot

| Field | Value |
|---|---|
| **Phase** | Phase 1 — the product layer (`DOCS/IDEA3.md`) |
| **Current task** | `T101` — backend seam, NumPy behind it (`DOCS/TASKS2.md`) |
| **Task status** | `not_started` |
| **Completed tasks** | none in Phase 1. Phase 0: T001 … T011, all done |
| **Milestone reached** | **M4** (2026-08-13, Phase 0 complete). Phase 1 targets M5 → M8 |
| **Phase 0 rung status** | R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — the ladder is complete and stays a gate for every Phase 1 task |
| **Phase 1 rung status** | A ⬜ · B ⬜ · C ⬜ · D ⬜ · E ⬜ — no script exists yet; each is built in its own task |
| **Last updated** | 2026-08-13 — session 12 (Phase 1 planned; `IDEA3`, `PLAN2`, `TASKS2`, this file; D-041 … D-048; no code changed) |

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

None.

## Open questions

- **Q-101** — does `python -m lbm.runner` (the M4 gate command) survive as a working entry point once
  `python -m flow` exists, or become a pointer to it? T109 decides and records it. Both are
  defensible; the M4 gate output in `old-Docs/STATE1.md` § Snapshot must remain reproducible or be
  explicitly marked historical.
- **Q-102** — is D-017's documented limit (a thin appendage **fused** to a thick body shares its
  component and is not reported) closable without false-alarming on a plain disc? T107 must measure
  this, not inherit it. If no metric clears both, the limit stays with the measurement recorded.
- **Q-103** — what tolerance does cross-backend whole-step agreement actually need? T103 sets
  `max|Δu|/U < 1e-4` at 1000 steps as the contract; whether that is achievable or generous is
  unknown until the port runs. It is a pass condition to be met, not adjusted.

## Environment

Project venv: `myenv/` (gitignored). Python 3.11.15. Unchanged this session — **nothing was
installed**.

| Package | Version | Added by |
|---|---|---|
| numpy | 2.4.6 | pre-existing |
| matplotlib | 3.11.1 | pre-existing |
| pillow | 12.3.0 | pre-existing |
| pytest | 9.1.1 | T001 (session 1) |
| pygame | 2.6.1 | T007 (session 7) |
| imageio | 2.37.4 | T011 (session 11) |
| imageio-ffmpeg | 0.6.0 | T011 (session 11) |
| psutil | 7.2.2 | T011 (session 11) |

**Expected in Phase 1:** `warp-lang` in T102 (**D-043**). Install it in the session that first needs
it and add a row here with the CUDA and driver versions it reports. No XLB, no Taichi, no UI
framework — see `DOCS/IDEA3.md` § Deliberately deferred.

Install with `myenv/Scripts/pip.exe install <pkg>` and **add a row above in the same session.**

## Performance baseline

Phase 0's measured table is in `old-Docs/STATE1.md` § Performance baseline and stands: **696.7 / 161.7 /
16.8** steps/s at 40k / 160k / 1M cells at the CPU's rated 3201 MHz on mains, against floors of
400 / 120 / 15. At 1M cells `equilibrium` is **39.9 ms of a ~75 ms step** — over half — which is
T102's first target.

Phase 1's budget (`DOCS/IDEA3.md` § Performance budget), to be filled in by T103:

| Grid | Cells | NumPy measured | Warp floor | Warp measured |
|---|---|---|---|---|
| 400×100 | 40k | 696.7 | ≥2000 | — |
| 2000×500 | 1M | 16.8 | ≥250 | — |
| 2000×1000 | 2M | ~8 (estimated, not measured) | ≥150 | — |

**D-035 still governs every number here**: alternating-round A/B, best round per variant, and no
absolute steps/s figure without the CPU clock, the power state and now the GPU name beside it.

## Decisions

Anything chosen that wasn't already specified in `DOCS/IDEA3.md`, `idea.md` or `CLAUDE.md`. Append;
never edit a past entry — supersede it with a new one that says so. Numbering continues from
`old-Docs/STATE1.md` (D-005 … D-040), which remains in force.

| ID | Date | Decision | Why |
|---|---|---|---|
| D-041 | 2026-08-13 | **Phase 1 starts a new state file, `DOCS/STATE2.md`. `old-Docs/STATE1.md` is frozen** — read for history, never edited, never condensed. Decision numbering continues unbroken at D-041, and Phase 0 decisions are cited by number rather than copied. | The session protocol says read the state file **in full** at session start, and STATE1 is 1106 lines of Phase 0 history whose operative content is § Decisions. A second file keeps that cost bounded while a continuous decision numbering keeps the citations unambiguous — there is exactly one D-029 in the project. Copying Phase 0's decisions forward was rejected: two copies drift, and the append-only rule then protects the wrong one. |
| D-042 | 2026-08-13 | **The product layer is a new top-level package, `flow/`, beside `lbm/`. `flow/` may import `lbm/`; `lbm/` may never import `flow/`, and a test asserts it.** | `idea.md`'s pipeline diagram has a solver line, and Phase 3 plans to replace everything below it with XLB. A one-directional import is what makes that a substitution rather than a rewrite. It also keeps two invariants that only held in Phase 0 because nothing was allowed to break them: constraint 10 ("one `render()`" — `flow/` colours nothing) and the units boundary (`lbm/` is lattice units only, `flow/` never speaks them — Phase 1 constraint 13). Growing `lbm/case.py` instead was rejected for blurring both. |
| D-043 | 2026-08-13 | **The GPU port (Warp) moves from Phase 2 into Phase 1, ahead of the product layer, as tasks T102–T103. NumPy is kept as the reference oracle, not replaced** — every GPU claim is checked against it (Rung A), and a GPU that disagrees with NumPy is a broken backend, never a new answer. **This is a deliberate deviation from `idea.md` § Roadmap**, which puts the port at Phase 2. | User's call when the alternatives were put side by side. The arithmetic behind it: Phase 1's headline criterion is a wall clock — "under a minute" — and the M4 gate took **335 s** for 5 physical seconds at 185k cells at 16.8 steps/s (1M cells). No product polish closes that gap; only the kernel does. Building the product layer on a backend about to be replaced would mean measuring every product-level timing twice. **The known risk is `idea.md` § Risks "The trap"** — a port is solver work, and solver work is the fun part — so `DOCS/PLAN2.md` § Risks carries a hard valve: if T102 or T103 overruns by one session, the port is demoted back to Phase 2 and Phase 1 continues on NumPy with M8's wall clock restated honestly against it. T101's seam makes that demotion a config change. |
| D-044 | 2026-08-13 | **Phase 1's deliverable is a library (`flow/`) plus a CLI. No UI** — no browser, no server, no TUI framework. Phase 1's success test replaces `idea.md`'s "drags in a picture" with "three lines of Python or one command, from a cold shell, under a minute", and is otherwise identical. | `idea.md` § Risks — Scope demands a narrow Phase 1, and the UI is the one part of the pipeline that can be added later **without** invalidating anything below it, precisely because everything below it will have been tested. The reverse is not true: a UI built over untested judgement hides the judgement. `python -m lbm.runner` already proves the CLI shape works end to end (M4). The cost is stated rather than hidden — Phase 1 does **not** meet `idea.md`'s literal success sentence, and `DOCS/IDEA3.md` § Scope says so in as many words. |
| D-045 | 2026-08-13 | **The answer to a case the solver cannot represent is: refuse it, explain it in the user's units, and offer the nearest runnable case — never substitute silently.** Refusals carry structured fields (`reason`, `quantity`, `value`, `limit`, `suggestions`), and **each suggestion is a testable claim**: Rung D feeds the tool's own top suggestion back through the planner and runs it, so a suggestion that does not fix its case is a failing test. Any run that differs from what was asked carries `substituted=True` into the printed summary, the report and the video metadata (Phase 1 constraint 16). | This is the question **D-038** left open and the one the target user meets first: air at 20 m/s past a 1.5 m body is Re 2e6, `tau` reads 0.5000, and BGK with bounce-back and no turbulence model cannot represent it at any resolution this project will run. Refusing is correct (**D-032**) and, as Phase 0 leaves it, a dead end. Silently running a dynamically-dissimilar case was rejected as exactly the artefact the validation ladder exists to prevent — a converged, plausible, wrong video — and a loud banner was rejected as a defence that does not survive a screenshot. Making the suggestion executable is what stops the explanation from decaying into a wording exercise. |
| D-046 | 2026-08-13 | **Fate of `CLAUDE.md`'s 12 hard constraints, decided rather than left to rot** (table below). Permanent: **2, 3, 5, 7, 8, 9, 10, 11, 12**. Rewritten for Phase 1: **1** (physics unchanged, implementation may move to a GPU backend), **4** (the backend owns its layout; `to_host` must yield `(9, ny, nx)` `float32`), **11** (bit-identical *within* a backend; cross-backend is a printed tolerance). Retired: **6** (spent — Rung 3 passed in session 7), replaced by "no backend optimisation before its parity rung passes". Added: **13** (no lattice quantity in any public `flow/` signature), **14** (every refusal names a fix, and the fix is machine-checked), **15** (`flow/` may import `lbm/`, never the reverse), **16** (no silent substitution). | Constraint 6 is spent by its own terms and constraint 1 is explicitly a Phase 0 constraint that Phase 3 plans to break; leaving either in place would train us to read the list as decoration, which is how a checked constraint becomes an ignored one. The nine kept unchanged are kept because each is load-bearing above the solver line too, and three of them (8, 9, 10) are the only reason the rendering path stayed coherent through four sinks. The four added constraints are the Phase 1 equivalents: each has a test that fails when it is violated, which is the property that made the original twelve useful. |
| D-047 | 2026-08-13 | **Phase 1's validation ladder is five rungs — A parity, B auto-config, C shapes, D refusals, E the minute — ordered and non-negotiable, and they live in the existing `validate/` package** (`validate/parity.py`, `autoconfig.py`, `shapes.py`, `refusals.py`, `minute.py`), each printing PASS/FAIL. **A rung's harness is built in the task that needs it, before the code it validates.** For a usability layer, the "known answer" is a **committed verdict or an executable suggestion**: a shape's expected verdict plus measured properties (Rung C), a suggestion that must actually run (Rung D), and — for Rung E — Phase 0's own published cylinder bands reached through the product path. | `idea.md` § Risks: *"Every phase ships with a benchmark that has a known answer."* Phase 0's ladder caught three wrong-but-plausible answers (the force integral measuring the channel walls, the blockage lie from no-slip walls, the FFT locking onto the domain's acoustics) precisely because PASS meant a number inside a published band. Judgement can be held to the same standard if the claims are made falsifiable — a verdict is checkable, a suggestion is runnable — and Rung E deliberately reuses Rung 3's cylinder so a regression in *judgement* surfaces as a regression in *physics*, which is the only kind this project knows how to detect reliably. A new `validate2/` package was rejected: `python -m validate.<rung>` is already the documented command shape. |
| D-049 | 2026-08-13 | **Phase 0's three session-management documents move to `old-Docs/`** — `STATE1.md`, `TASKS1.md`, `PLAN1.md`, by `git mv`, with an `old-Docs/README.md` pointer. **`DOCS/IDEA2.md` stays in `DOCS/`**, and so do `bench_baseline.json` (read by `bench.py`) and `ISSUES.jsonl` (read by `tools/issues.py`). Phase 0's session prompts `001` … `011` are deleted; `012` onward stay. Every reference to the three moved paths was rewritten across code, docs, `.claude/commands/` and the prompt template — **a path rewrite is not a content edit, and D-041's freeze covers content and history, not the file's location**. | The archive is worth having and the cost was measured before paying it: `DOCS/IDEA2.md` is cited by **~100 docstrings** in `lbm/`, `validate/`, `tests/` and `bench.py`, because `CLAUDE.md` § Coding conventions requires docstrings to name the spec section. Moving it would have invalidated every one of those citations for a tidier directory listing — the convention is load-bearing and the listing is not. The three session documents are cited ~90 times but **only in prose**, where a mechanical rewrite is safe and verifiable; `pytest` re-run afterwards confirms **367 passed**, unchanged. The `.claude/commands/` rewrite needed a second pass in the opposite direction: the first pass pointed `/checkpoint` and `/start-task` at `old-Docs/STATE1.md`, which would have had the next session **writing into the frozen file**. They now target `DOCS/STATE2.md` / `DOCS/TASKS2.md` / `DOCS/PLAN2.md` / `DOCS/IDEA3.md` and `T1XX`. Prompts 001–011 are all committed, so deletion is recoverable from git history — that is what made it a cheap call rather than an irreversible one. |
| D-048 | 2026-08-13 | **Phase 1 tasks are numbered `T101` … `T110`**, and the phase's documents are `DOCS/IDEA3.md` (spec), `DOCS/PLAN2.md` (plan), `DOCS/TASKS2.md` (backlog), `DOCS/STATE2.md` (state). | `T012` would have read as a continuation of a closed backlog, and `/start-task` resolves a bare ID against a task file — two files with a `T0xx` range each is an ambiguity waiting for a tired session. The `IDEA3`/`PLAN2` mismatch is inherited (`idea2.md` was Phase 0's spec) and is kept rather than renamed, because `DOCS/IDEA2.md` is cited by name in eleven session-log entries. |

### Constraint fate table (D-046)

| # | Phase 0 constraint | Phase 1 status |
|---|---|---|
| 1 | D2Q9, BGK, bounce-back; no MRT/cumulant/curved boundaries/turbulence model | **Rewritten.** The *physics* is unchanged through Phase 1; the *implementation* may move to a GPU backend. Deferred items stay deferred. |
| 2 | `nu = (tau - 0.5)/3`; no `nu` setter that bypasses `tau` | **Permanent.** |
| 3 | Lattice velocity under 0.1, warned at setup | **Permanent**, and now enforced by `flow/autoconfig.py` for users who never see `u`. |
| 4 | State is `f` `(9, ny, nx)` `float32`; constants only from `lbm/core.py` | **Rewritten.** The backend owns its device layout; `to_host` must yield `(9, ny, nx)` `float32`, and the constants still come from `lbm/core.py` only — uploaded, never redefined. |
| 5 | The validation ladder is ordered and non-negotiable | **Permanent**, extended to Phase 1's five rungs (**D-047**). |
| 6 | Do not optimise before Rung 3 passes | **Retired** — spent in session 7. Replaced by: *no backend optimisation before its parity rung passes.* |
| 7 | Simulation and rendering decoupled; `steps_per_frame` computed | **Permanent.** |
| 8 | Never block the sim on the display; drop display frames, never steps | **Permanent**, with **D-039**'s refinement (any file-writing sink takes `drop=False`). |
| 9 | Draw vorticity; diverging map, fixed symmetric limits | **Permanent.** |
| 10 | One `render()`, three sinks | **Permanent**, and now also means `flow/` colours nothing. |
| 11 | Restart is bit-identical | **Rewritten.** Bit-identical **within** a backend; **across** backends it is a printed tolerance (T103), because float ordering differs on a GPU and no test should pretend otherwise. |
| 12 | Geometry is one bool `solid`; ≥3 cells thick, ≥8 D downstream, <10% blockage | **Permanent**, and Phase 1 *repairs* where it can rather than only warning (T107). |
| **13** | — | **New.** No lattice quantity in any public `flow/` signature. |
| **14** | — | **New.** Every refusal names a fix, and the fix is machine-checked (Rung D). |
| **15** | — | **New.** `flow/` may import `lbm/`; `lbm/` may never import `flow/`. |
| **16** | — | **New.** No silent substitution — a run that differs from the request says so in every artifact. |

## Session log

Append one entry per session. Newest at the bottom.

### 2026-08-13 — Session 12: Phase 1 planning

**Task worked:** none — this session plans a phase and writes documents. There is no `T012`
(**D-048**).

**Done**
- Read, in this order: root `idea.md`, root `README.md`, `CLAUDE.md`, `old-Docs/STATE1.md`
  (§ Snapshot, § Blockers, § Open questions, § Environment, § Performance baseline, the whole of
  § Decisions D-005 … D-040, every "Not done / deferred" block, and the session 0, 1, 2 and 11
  entries in full), `old-Docs/TASKS1.md` in full, `old-Docs/PLAN1.md` in full, and `DOCS/IDEA2.md`
  § Stability / § Milestones / § Deliberately deferred.
- Wrote **`DOCS/IDEA3.md`** — the Phase 1 spec: goal, scope with an explicit out-list, the pipeline
  made concrete as modules, the five things Phase 1 must get right, the five-rung ladder, the
  performance budget with the bandwidth arithmetic behind its floors, and § Deliberately deferred.
- Wrote **`DOCS/PLAN2.md`** — 10 tasks T101 → T110, dependency graph, session map (sessions 13–22),
  four milestone gates **with literal gate commands**, and a risks table whose first row is
  `idea.md`'s "The trap" with a hard pressure valve.
- Wrote **`DOCS/TASKS2.md`** — full contract per task: goal, reads/depends-on, inputs and outputs
  with types and array shapes, acceptance criteria as a checklist, the constraints that bite, notes.
- Wrote **this file**, including the constraint fate table and D-041 … D-048.
- Edited `CLAUDE.md` — **pointers only** (§ Session protocol → the Phase 1 documents and
  `/start-task T1XX`; § Current state → Phase 1 live, T101 current, D-046 the constraint authority;
  the header's conflict rule → "the spec of the live phase wins"). The 12 hard constraints
  themselves are untouched.
- Wrote `PROMPTS/013-t101-backend-seam.md` — paste-ready, with T101's acceptance criteria verbatim,
  the constraints that bite in their D-046 form, and the five Phase 0 decisions T101 must not break.
- **Archived Phase 0** (**D-049**), after the checkpoint, at the user's request: `git mv` of
  `STATE1.md` / `TASKS1.md` / `PLAN1.md` into **`old-Docs/`** with an `old-Docs/README.md` pointer;
  `git rm` of the eleven Phase 0 session prompts (`PROMPTS/001-…` … `011-…`). `DOCS/IDEA2.md`,
  `bench_baseline.json` and `ISSUES.jsonl` deliberately stayed in `DOCS/`. ~90 references to the
  moved paths rewritten across `lbm/`, `validate/`, `tests/`, `bench.py`, `CLAUDE.md`, the Phase 1
  docs, `.claude/commands/` and the prompt template — the slash commands and the template were then
  re-pointed **forward** to `DOCS/STATE2.md` / `TASKS2.md` / `PLAN2.md` / `IDEA3.md` and `T1XX`,
  because the mechanical pass had aimed `/checkpoint` at the frozen file.
- `myenv/Scripts/python.exe -m pytest` re-run after the rewrite → **`367 passed, 7 warnings in
  55.69s`**. The rewrite touched docstrings and prose only; the suite proves it.

**Measured**
- `myenv/Scripts/python.exe -m pytest` → **`367 passed, 7 warnings in 35.18s`**, unchanged from
  session 11 — the cheap confirmation that no solver code moved since Phase 0 closed. The full
  ladder (~55 min) was **not** re-run: nothing has changed to invalidate it, and session 11 ran all
  four.

**Decisions made**
- **D-041** (STATE2, STATE1 frozen), **D-042** (`flow/` package, one-directional import),
  **D-043** (Warp port pulled into Phase 1, NumPy kept as oracle — the one deviation from
  `idea.md`'s roadmap), **D-044** (library + CLI, no UI), **D-045** (refuse, explain, offer a
  runnable alternative; never substitute silently), **D-046** (constraint fates), **D-047**
  (the five-rung ladder and what a "known answer" means for judgement), **D-048** (numbering and
  file names). All in § Decisions.

**Not done / deferred**
- **No code.** Nothing under `lbm/` was touched, no `flow/` package was created, no rung script was
  written — `DOCS/PLAN2.md` § Session map puts T101 in session 13 and `CLAUDE.md`'s one-task-per-
  session rule makes planning the task.
- `CLAUDE.md` got a **pointer edit only**, not a rewrite: § Session protocol now sends a cold
  session to `DOCS/STATE2.md` / `DOCS/TASKS2.md` / `/start-task T1XX`, § Current state names Phase 1
  and its four documents, and the header's conflict rule now names the live phase's spec. This was
  not optional — `CLAUDE.md` loads automatically every session, so leaving it pointing at the frozen
  Phase 0 files would have misdirected session 13 before it read anything. **The 12 hard constraints
  themselves are untouched**; D-046's table is marked as the authority where they disagree, and
  folding it in is T101's documentation work, when the first Phase 1 code exists to obey it.
- `DOCS/ISSUES.jsonl` tracking in git — noted in sessions 10 and 11, still the user's call, still
  not acted on.
- The five-question list in `PROMPTS/012-phase1-planning.md` is fully answered: kernel (**D-043**),
  deliverable (**D-044**), ladder (**D-047**), nasty geometry (T107 + **Q-102**), where the product
  layer lives (**D-042**).

**Blockers**
- None.

**Rung status after this session**
- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — unchanged, confirmed by the test suite rather than
  re-run.
- Phase 1: A ⬜ · B ⬜ · C ⬜ · D ⬜ · E ⬜ — correct for a planning session; each rung is built in
  the task that needs it.

**Next**
- Paste `PROMPTS/013-t101-backend-seam.md` into a fresh session. It runs `/start-task T101`.
- T101 is deliberately the least glamorous task in the phase: it adds indirection, changes no
  physics, makes nothing faster, and its acceptance criterion is that **every Phase 0 number comes
  back identical to session 11**. It exists so that T102's Warp backend is the only new thing in the
  session that introduces Warp.
