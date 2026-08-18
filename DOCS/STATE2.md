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
| **Current task** | `T103` — Warp boundaries, checkpoint, performance (`DOCS/TASKS2.md`) |
| **Task status** | `not_started` |
| **Completed tasks** | Phase 1: **T101**, **T102**. Phase 0: T001 … T011, all done |
| **Milestone reached** | **M4** (2026-08-13, Phase 0 complete). Phase 1 targets M5 → M8 |
| **Phase 0 rung status** | R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — the ladder is complete and stays a gate for every Phase 1 task |
| **Phase 1 rung status** | **A 🟨** · B ⬜ · C ⬜ · D ⬜ · E ⬜ — Rung A's **kernel** half is green (`validate/parity.py --kernels`, worst 5.96e-08 against a 1e-6 bar); its whole-step half, the boundaries and the four Phase 0 rungs on GPU are T103's |
| **Last updated** | 2026-08-18 — session 14 (**T102 done**: `lbm/backends/warp_backend.py`, `validate/parity.py --kernels` green, warp-lang 1.16.0 installed; D-052, D-053) |

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
  **Session 14 evidence, not an answer:** per *kernel*, one step's disagreement is at most
  **5.96e-08** in `f` units (**D-053**) — `macroscopic` and `stream` are bitwise identical, so only
  `equilibrium` and `collide` inject anything at all. That makes 1e-4 look generous *if* the error
  does not compound; whether it compounds over 1000 steps is exactly what T103 measures, and the
  question stays open until it prints the 10 / 100 / 1000-step growth.

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
| warp-lang | 1.16.0 | T102 (session 14) — CUDA Toolkit **12.9**, Driver **13.1**; device `cuda:0` = NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB, sm_86, mempool enabled); `nvidia-smi` driver **592.82** |

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
| D-050 | 2026-08-18 | **The checkpoint's contents are unchanged by the backend seam. `f` is written through `backend.to_host`, so it is always the portable host layout `(9, ny, nx)` `float32`, and the backend *name* rides inside the pickled `SimConfig` rather than as a new top-level key — so D-022 stays literally true: `f` / `solid` / `step_count` / config plus `format: 1`.** `load_checkpoint(path, backend=...)` overrides the saved name, which is how a checkpoint written on one backend is resumed on another. A checkpoint written before T101 has no `backend` field and picks up the dataclass default, `"numpy"`, which is the backend it ran on — tested, not assumed. | `PROMPTS/013-t101-backend-seam.md` asked for this to be recorded as D-049; that number was already spent on session 12's archival decision, so it is D-050 — logged here rather than silently renumbered. D-022 asks for this decision by name and warns that adding the backend to the checkpoint has a real consequence. Putting it in the config is what makes the consequence benign: the backend is *configuration*, like `tau` or `fused`, and the config was already pickled verbatim, so nothing is added and nothing new can be misread. A top-level `backend` key would have broken D-022's four-things rule for no information gain, and pinning the file to the backend that wrote it would have made the D-043 pressure valve (demote the port, continue on NumPy) discard every checkpoint in flight. `to_host` on the write side is what makes the file portable at all — constraint 4 in its D-046 form is the only reason a Warp checkpoint will be readable by NumPy in T103. |
| D-051 | 2026-08-18 | **The `Backend` protocol covers kernels and the two host transfers, and nothing else. Buffer allocation, the open boundaries (`inlet_velocity`, `outlet_zero_gradient`), the Guo body force and the probes stay outside it** — `Sim` still allocates its own `(9, ny, nx)` buffers with `np.empty` and still calls `lbm.boundary`'s open-boundary functions directly. **T102/T103 will therefore have to widen the seam** (allocation first), and that widening is expected work, not a defect of T101. | `DOCS/TASKS2.md` § T101 Notes: two implementations is the number that reveals the right seam; one plus a guess is not. Every method guessed at now would be shaped by NumPy alone and rewritten in T102 anyway, at the cost of a protocol nobody could read. The contract's minimum list is exactly the set `Sim.step` calls per timestep, which is the set with a measurable cost, and Rung A can already be built on `to_host`/`from_host` alone (**Q-103**). Recorded here so T102 budgets for it rather than discovering it. |
| D-052 | 2026-08-18 | **The Warp backend takes *host* arrays at its boundary and owns preallocated *device* buffers keyed by grid shape.** `Sim` is untouched: it still owns its `(9, ny, nx)` NumPy buffers (**D-051**), and each kernel call uploads its inputs, launches, and downloads its outputs. Device buffers are allocated once per `(ny, nx)` — at construction with `WarpBackend(shape=(ny, nx))`, otherwise on the first call for that shape — and never again. `WarpBackend(device=None)` takes `warp.get_preferred_device()`, so a machine with no CUDA runs Rung A on the CPU rather than not running it. | The `Backend` protocol has no allocation method and T102's contract is four kernels and a parity script, not a seam redesign. Every T102 acceptance criterion — per-kernel parity, the spike test, no allocation per call — is measurable through a host boundary, so widening the seam here would have been a guess made against **one** call site (the mistake **D-051** exists to avoid) *and* a change to `Sim` inside the session that introduces Warp, which `DOCS/PLAN2.md` § Why this order spends T101 to prevent. The cost is stated rather than hidden: the per-call copies make this backend **slower than NumPy** at these sizes and that is why T102 quotes **no speed number at all** (constraint 6's replacement — no backend optimisation before its parity rung passes). T103 removes them by moving the state onto the device, which is where whole-step parity forces the seam into its real shape. |
| D-053 | 2026-08-18 | **Cross-backend kernel agreement is a measured, explained number, not a hope: worst case 5.96e-08 in `f` units against the task's 1e-6 bar, on 32×64, 200×400 and 500×1000.** Per kernel: `macroscopic` **0.000e+00** (bitwise, both `rho` and `u`), `stream` **0.000e+00** (bitwise — a permutation has no arithmetic to round), `collide` **1.49e-08**, `equilibrium` **5.96e-08**. The two non-zero ones are a **fused multiply-add**: the GPU contracts `x * a + b` into one rounding where NumPy rounds twice, which is a half-ulp at `f ~ 0.2` for `collide` and one ulp at `f ~ 0.44` for `equilibrium`. `validate/parity.py --kernels` prints all of it, including a bitwise column, so a later regression shows in the digits rather than hiding under PASS. | `DOCS/TASKS2.md` § T102: "the script prints the number, so a later regression is visible rather than merely passing", and `DOCS/PLAN2.md` § Risks: a parity failure is bisected **by kernel**, which requires the per-kernel numbers to exist before anything fails. Recording *which* kernels are bitwise matters as much as the magnitudes: it means a future disagreement in `macroscopic` or `stream` is a **bug**, not float ordering, and the tolerance argument does not apply to it. The tolerance itself was not widened and must not be — a difference above 1e-6 is not reachable by reordering `float32` arithmetic at these magnitudes. |

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

### 2026-08-18 — Session 13: T101, the backend seam

**Task worked:** `T101` — backend seam, NumPy behind it. **Done**, every acceptance criterion run
rather than read.

**Done**
- Read, in the prompt's order: `CLAUDE.md`, this file in full, `DOCS/TASKS2.md` § T101 in full,
  `DOCS/IDEA3.md` § What Phase 1 is, concretely / § The five things / § Validation ladder /
  § Deliberately deferred, `old-Docs/STATE1.md` **D-011**, **D-020**, **D-022**, **D-033**,
  **D-035**, and `DOCS/PLAN2.md` § Why this order / § Dependency graph / § Session map / § Risks.
- **`lbm/backends/__init__.py`** — the `Backend` protocol (`typing.Protocol`, `@runtime_checkable`):
  `macroscopic`, `equilibrium`, `collide`, `bounce_back`, `stream`, `collide_stream`, `to_host`,
  `from_host`, plus a `name` attribute. Every method documents its array shapes; a test asserts the
  docstrings actually contain them, because a seam whose shapes live in a comment is a seam nobody
  can port to. Its module docstring states what a backend owns (kernels + layout) and what it does
  **not** (the D-011/D-020 timestep order, which stays in `Sim.step` because the order is physics).
- **`lbm/backends/numpy_backend.py`** — `NumpyBackend`, a pure delegation to the **unchanged**
  `lbm.core` functions and `lbm.boundary.bounce_back`. `to_host` / `from_host` are the identity with
  the `(9, ny, nx)` `float32` contract *checked*, so a layout mistake in a future backend fails at
  the seam rather than three rungs later.
- **`lbm/backends/registry.py`** — `get_backend` / `available_backends` / `known_backends` and
  `BackendUnavailableError(ValueError)`. Two failures, two messages: an unknown name lists what
  exists, a known-but-uninstalled one names the install line. **The `warp` row already exists** —
  T102 does not touch this table, it only writes the module the row points at.
- **`SimConfig.backend: str = "numpy"`**; `Sim.backend` is resolved *before* any allocation (a bad
  name should cost nothing); `Sim.step` and `Sim._init_equilibrium` reach every kernel through it.
  `lbm/runner.py` now imports `Q, W` from `lbm.core` and **no kernels at all** — the constants still
  come from there, which constraint 4 requires rather than merely allows.
- **`tests/test_backends.py`** — 22 tests, one per acceptance criterion plus the seam's own
  invariants: protocol completeness, documented shapes, bit-for-bit delegation against `lbm.core`,
  fused-vs-unfused bitwise equality through the seam (**D-033**), an AST scan asserting no backend
  module redefines `E`/`W`/`OPP`/`CS2`, an AST + namespace scan asserting `lbm/runner.py` imports no
  kernel, a counting-backend test proving `Sim` reaches all six kernels through `self.backend`, a
  `tracemalloc` check that the indirection did not put an allocation back in the step loop, the
  `to_host`/`from_host` round trip under `np.array_equal`, the two registry errors, and the restart
  pair.
- **`CLAUDE.md`: D-046's constraint table folded in**, which § Current state and session 12's log
  both assign to this task. The list is now 16 Phase 1 constraints — 1, 4 and 11 rewritten, 6
  **retired but struck rather than deleted**, 13–16 added with a note that their tests land with
  `flow/`. § Current state and § Module map updated (`lbm/backends/` added). D-046 is no longer "the
  authority"; it is now the record of *why* each constraint reads the way it does.

**Measured**
- `myenv/Scripts/python.exe -m pytest` → **`389 passed, 7 warnings in 15.40s`** — 367 before,
  22 added, **no existing test modified** (`git status` shows `tests/test_backends.py` as the only
  change under `tests/`).
- **The full ladder, re-run after the change, printing session 11's digits exactly:**

  | Rung | Session 11 | Session 13 |
  |---|---|---|
  | R1 Poiseuille | L2 **0.3650%** | L2 **0.3650%** |
  | R2 cavity | **0.75% / 0.42% / 1.01%** | **0.75% / 0.42% / 1.01%** |
  | R3 cylinder | St **0.1731**, Cd **1.4031 ± 0.0086** | St **0.1731**, Cd **1.4031 ± 0.0086** |
  | R4 square | Cd **1.5279 ± 0.0271** | Cd **1.5279 ± 0.0271** |
  | R4 polygon | Cd **1.4276 ± 0.0226** | Cd **1.4276 ± 0.0226** |

  Identical, not merely inside the band — which is the criterion for a task that is supposed to
  change nothing.
- **Performance: no measurable cost from the indirection**, and deliberately not benchmarked.
  Per **D-035** the numbers below are quoted with their conditions: AMD Ryzen 7 5800H,
  `Win32_Processor.CurrentClockSpeed` **3201 MHz** of a 3201 MHz maximum, on **mains**
  (`BatteryStatus 2`). R3 ran 45500 steps in **365.0 s (125 steps/s)** against session 7's
  368.9 s standalone for the same rung. That is a rung wall clock, **not** an A/B benchmark — D-035
  requires alternating rounds and one resident `Sim`, and none of that was done — so it is recorded
  as "no slowdown large enough to show up", not as a speed claim. Six Python attribute lookups per
  timestep against an 8–75 ms step was never going to be visible, and constraint 6's replacement
  forbids optimising a backend before its parity rung exists.

**Decisions made**
- **D-050** — the checkpoint's contents are unchanged: `f` is written through `backend.to_host` so
  it is always the portable `(9, ny, nx)` `float32` layout, and the backend *name* rides inside the
  pickled `SimConfig` rather than as a new top-level key, so **D-022** stays literally true.
  `load_checkpoint(path, backend=...)` overrides it. A pre-T101 checkpoint has no `backend` field
  and picks up the dataclass default `"numpy"` — tested, not assumed.
  `PROMPTS/013-t101-backend-seam.md` asked for this to be D-049; that number was spent on session
  12's archival decision, so the conflict is logged in the D-050 row rather than silently
  renumbered.
- **D-051** — the protocol covers kernels and the two host transfers **and nothing else**. Buffer
  allocation, the open boundaries, the Guo body force and the probes stay outside it; `Sim` still
  allocates its own buffers with `np.empty`. **T102/T103 will have to widen the seam, starting with
  allocation** — recorded so the port budgets for it rather than discovering it.

**Not done / deferred**
- **No `--backend` flag on any CLI or rung script.** `python -m lbm.runner` and `validate/*` still
  run the default. M5's gate needs `validate.<rung> --backend warp`; that wiring belongs to T103,
  which owns the full-ladder-on-GPU gate. Adding it now would have been a flag with one legal value.
- **No Warp, no `warp-lang` install.** T102's, by the contract and by `DOCS/PLAN2.md` § Session map.
- **The seam does not yet cover allocation** — see **D-051**. This is the one place a reader could
  mistake T101 for finished-in-the-larger-sense: `Sim` still owns `(9, ny, nx)` NumPy buffers, so a
  device backend cannot yet hold device memory. Deliberate (two implementations reveal the seam; one
  plus a guess does not), and the first thing T102 will hit.
- **Q-103 not closed** — it is not this task's to close. What T101 gives it is the measurable hook:
  `to_host`/`from_host` are the only path state takes in or out, so the parity rung compares host
  arrays and nothing else.
- `DOCS/ISSUES.jsonl` tracking in git — still the user's call, still not acted on. Nothing was
  queued this session; nothing failed.

**Blockers**
- None.

**Rung status after this session**
- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four **re-run**, not inherited, and identical.
- Phase 1: A ⬜ · B ⬜ · C ⬜ · D ⬜ · E ⬜ — Rung A is T102's first deliverable.

**Next**
- Paste `PROMPTS/014-t102-warp-kernels.md` into a fresh session. It runs `/start-task T102`.
- T102 installs `warp-lang`, writes `lbm/backends/warp_backend.py` for `equilibrium`, `collide`,
  `stream`, `macroscopic`, `to_host`, `from_host`, and builds `validate/parity.py --kernels` —
  **Rung A's harness before the code it validates**. The nine constants are uploaded from
  `lbm/core.py`, never redefined in a kernel. The first thing it will meet is **D-051**.
- `DOCS/PLAN2.md` § Risks: if Warp will not install or run in the first half of that session, log
  the blocker and fall through to **T104**, which is independent of the GPU work.

### 2026-08-18 — Session 14: T102, the Warp kernels

**Task worked:** `T102` — Warp kernels: equilibrium, collide, stream. **Done**, every acceptance
criterion run rather than read.

**Done**
- Read, in the prompt's order: `CLAUDE.md`, this file in full, `DOCS/TASKS2.md` § T102 **and**
  § T103 (so the boundary between them was known before any code), `DOCS/IDEA3.md` § Performance
  budget / § Validation ladder (Rung A) / § Deliberately deferred, `old-Docs/STATE1.md`
  § Performance baseline, **D-008** and **D-035**, `DOCS/PLAN2.md` § Dependency graph / § Session
  map / § Risks, and `lbm/backends/__init__.py`.
- **`warp-lang` 1.16.0 installed** into `myenv` and recorded in § Environment: Warp reports **CUDA
  Toolkit 12.9, Driver 13.1**; device `cuda:0` is the **NVIDIA GeForce RTX 3050 Laptop GPU**
  (4 GiB, sm_86, mempool enabled); `nvidia-smi` reports driver **592.82**. It installed and ran a
  kernel inside the first twenty minutes, so `DOCS/PLAN2.md` § Risks' fall-through to T104 was never
  reached.
- **`lbm/backends/warp_backend.py`** — `WarpBackend` with four kernels (`_macroscopic_kernel`,
  `_equilibrium_kernel`, `_collide_kernel`, `_stream_kernel`), the two host transfers, and
  `bounce_back` / `collide_stream` as stubs raising
  `NotImplementedError("see DOCS/TASKS2.md T103")`. Each kernel is a **transcription** of its
  `lbm.core` counterpart, operation for operation and in the same order (constraint 1): `usq` is
  still hoisted and premultiplied by 1.5 (**D-008**) even though a thread has registers; core's
  `work` scratch has no analogue and `equilibrium` documents why it accepts and ignores it. The one
  deliberate structural difference is `stream`, which **gathers** (`dst[y,x] = src[y-ey, x-ex]`)
  where core **scatters**, because one thread per destination cell has no write conflict — the same
  assignment read backwards, noted in the kernel's docstring, and checked by the spike test rather
  than by argument.
- **The nine constants are uploaded once at construction** from `lbm.core` — `E` (int32), `E_F32`,
  `W`, `OPP` and `CS2` — and no lattice constant appears in a kernel. `OPP` and `CS2` are uploaded
  though T102's kernels do not read them: T103's boundaries do, and the alternative was uploading
  the constant set twice. The three numeric literals in `_equilibrium_kernel` (`1.5`, `3.0`, `4.5`)
  are the ones `lbm.core.equilibrium` itself writes; rewriting them through `CS2` would change the
  emitted arithmetic, which constraint 1 forbids.
- **`validate/parity.py`** — **Rung A's harness, written before the code it validates.** `--kernels`
  mode: random `rho ∈ [0.9, 1.1]`, `|u| ≤ 0.099` (sampled in a disc, so the *magnitude* obeys
  constraint 3 rather than each component), a near-equilibrium `f`, at three grids, printing a
  per-kernel table of max absolute difference **and a bitwise column**, the worst case per kernel,
  and the nine-direction spike result. `--backend` selects the device under test; NumPy is always the
  reference (**D-043**). An uninstalled backend exits **2** and prints `SKIP`, so "no GPU here" never
  reads as a physics failure. The whole-step mode is deliberately absent and says so — T103's.
- **`tests/test_warp_backend.py`** — 20 tests, one per acceptance criterion plus the guards: the
  constants read back **off the device** and compared to `lbm.core`'s (an AST scan proves nothing was
  assigned; this proves what the kernels index), device pointers stable across calls, per-kernel
  parity at three grids reusing the rung's own `compare_kernels` (so rung and tests cannot drift),
  `stream` bitwise-equal and `collide` bounded at one ulp, the spike test, 1000 steps' worth of
  kernel calls with buffer pointers *and* free device memory flat, the bit-exact host round trip,
  `to_host` on a real device array, `tau <= 0.5` refused with core's message, a non-contiguous host
  array refused rather than silently copied, and both T103 stubs naming their task. The whole file
  skips cleanly where `warp-lang` is absent.

**Measured**
- **Rung A (kernels): PASS.** `myenv/Scripts/python.exe -m validate.parity --kernels`, on `cuda:0`:

  | kernel | quantity | 32×64 | 200×400 | 500×1000 | bitwise |
  |---|---|---|---|---|---|
  | macroscopic | rho | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes |
  | macroscopic | u | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes |
  | equilibrium | feq | 5.960e-08 | 5.960e-08 | 5.960e-08 | no |
  | collide | f | 1.490e-08 | 1.490e-08 | 1.490e-08 | no |
  | stream | f | 0.000e+00 | 0.000e+00 | 0.000e+00 | yes |

  Worst case **5.96e-08** against a **1e-6** bar — 17× inside it, and the tolerance was not touched.
  Spike test **9/9**. The two non-zero rows are fused multiply-add and nothing else (**D-053**).
- `myenv/Scripts/python.exe -m pytest` → **`408 passed, 1 skipped, 7 warnings in 17.17s`**. 389
  passed before; 20 added. The one skip is
  `test_a_known_but_uninstalled_backend_names_its_install_line`, which skips **by its own design**
  now that `warp` is installed — session 13 wrote it to do exactly that.
- **Phase 0 rungs unaffected, checked twice.** `git status` shows **no Phase 0 file modified** — the
  change is three new files plus the two Phase 1 documents — and the two cheap rungs were re-run
  anyway: **R1 L2 0.3650%, peak |u| 0.07955** and **R2 Re 100 max deviation 0.75%, vortex centre
  0.21 cells** — session 11 and 13's digits exactly. R3 and R4 were **not** re-run (~45 minutes for
  a NumPy path nothing touched); `PROMPTS/014` explicitly does not ask for them.
- **No speed number, deliberately** (**D-052**, constraint 6's replacement). The backend copies host
  to device and back per kernel call, so it is currently *slower* than NumPy at these sizes;
  measuring that would be measuring the seam, not the kernels. The performance table is T103's, and
  **D-035** governs it.

**Decisions made**
- **D-052** — the Warp backend takes host arrays at its boundary and owns preallocated device buffers
  keyed by grid shape; `Sim` is untouched, and the **D-051** seam widening happens in T103 where
  whole-step parity forces its real shape. The per-call copies are why no speed number is quoted.
- **D-053** — cross-backend kernel agreement is a measured, explained number: worst 5.96e-08,
  `macroscopic` and `stream` **bitwise**, `collide` and `equilibrium` off by one FMA rounding. Which
  kernels are bitwise is recorded because it means a *future* difference there is a bug, not float
  ordering.

**Not done / deferred**
- **The boundaries are not on the GPU.** `bounce_back`, the inlet, the outlet, `moving_wall` and the
  fused `collide_stream` all raise `NotImplementedError("see DOCS/TASKS2.md T103")`. This is the
  contract's scope line, not an overrun: "Four kernels and a parity script is the whole job."
- **`Sim` cannot run on the Warp backend yet**, and nothing in `lbm/runner.py` changed. A
  `SimConfig(backend="warp")` run reaches `bounce_back` and raises. T103's first job.
- **No `--backend` flag on the four Phase 0 rung scripts** — still T103's, as session 13 left it.
- **No whole-step parity, no GPU checkpoint, no performance table, no `bench.py --backend warp`** —
  all four are T103 acceptance criteria, listed there.
- **Q-103 is not closed**, only fed: the per-kernel evidence is recorded against it in § Open
  questions, and whether the error compounds over 1000 steps is what T103 measures.
- `DOCS/ISSUES.jsonl` tracking in git — still the user's call, still not acted on. Nothing was queued
  this session; nothing failed.

**Blockers**
- None.

**Rung status after this session**
- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — R1 and R2 **re-run** and identical; R3 and R4 inherited
  from session 13, with `git status` as the argument that nothing they exercise moved.
- Phase 1: **A 🟨** · B ⬜ · C ⬜ · D ⬜ · E ⬜ — the kernel half of Rung A is green; the rung is not
  fully green until T103's whole-step mode, the boundaries and the four Phase 0 rungs run on GPU.

**Next**
- Paste `PROMPTS/015-t103-warp-boundaries.md` into a fresh session. It runs `/start-task T103`.
- T103 is **M5**: the boundaries and the fused `collide_stream` on the GPU, whole-step parity printed
  at 10 / 100 / 1000 steps, a checkpoint written on `warp` that resumes on `numpy`, all four Phase 0
  rungs inside their published bands with `--backend warp`, and `bench.py --backend warp` clearing
  2000 / 250 / 150 steps/s at 40k / 1M / 2M cells.
- **The first thing it does is widen the seam** (**D-051**, **D-052**): the state has to live on the
  device, or the copies T102 accepted will eat the entire budget.
- `DOCS/PLAN2.md` § Risks, the hard valve: **T102 did not overrun.** If **T103** does, the port is
  demoted back to Phase 2 and Phase 1 continues on NumPy — T101's seam makes that a config change.
