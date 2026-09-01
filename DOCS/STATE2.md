# STATE2.md — project state, Phase 1 — **FROZEN**

> **FROZEN 2026-09-01, session 23 (D-084). Read for history; never edited, never condensed.**
> Phase 1 is complete (**M8**). The live state file is **`DOCS/STATE3.md`** and decision
> numbering continues there at **D-080**. This file's § Decisions (**D-041 … D-079**) remain in
> force and are cited by number. Phase 1's documents stay at these paths rather than moving to
> `old-Docs/` — D-084 priced the move at ~470 citations, ~120 of them docstring paths, and
> rejected it on D-049's own threshold.


**Read this first, every session, in full.** Updated by `/checkpoint` at the end of every session.
Never rewrite or condense the session log — append only.

**Phase 0's state file is `old-Docs/STATE1.md` and it is frozen** (**D-041**). Its § Decisions
(D-005 … D-040) remain in force and are cited by number throughout this file; its session log is
history and is never edited. Decision numbering continues here at **D-041**.

---

## Snapshot

| Field | Value |
|---|---|
| **Phase** | **Phase 1 — complete.** The product layer (`DOCS/IDEA3.md`) is built, validated and closed |
| **Current task** | none — `T110` was the last. **Next is Phase 2**, the XLB swap (`idea.md`'s Phase 3), which needs its own spec, plan, backlog and state file |
| **Task status** | `done` |
| **Completed tasks** | Phase 1: **T101 … T110**, all ten. Phase 0: T001 … T011, all eleven |
| **Milestone reached** | **M8** (2026-08-27, session 22) — `myenv/Scripts/python.exe -m validate.minute --backend warp` prints **PASS**: a committed PNG of a disc plus water at 5 mm/s past a 2 cm body, through `flow.Case` with no lattice quantity in the invocation, gives `Cd` **1.4040 ± 0.0173** (band 1.25–1.45) and `St` **0.1672** (band 0.155–0.175) in **49.5 s of wall clock from process start** against a 60 s limit. Conditions (**D-035**): AMD Ryzen 7 5800H at 3201 of 3201 MHz, on **mains**; NVIDIA RTX 3050 Laptop GPU, driver 592.82. **All four Phase 0 milestones and all four Phase 1 milestones are now reached: M1 … M8** |
| **Phase 0 rung status** | R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run in session 22 **on both backends**, printing session 11/15's digits: R1 L2 **0.3650%** numpy / **0.3649%** warp, R2 **0.75%** / **0.21 cells** on both, R3 St **0.1731** Cd **1.4031** on both, R4 square Cd **1.5279** and polygon Cd **1.4276** on both |
| **Phase 1 rung status** | **A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 — the whole ladder is green, and B is now green on _both_ backends.** A: worst kernel 5.960e-08, whole step 9.611e-06, checkpoint 8.196e-06, restart bit-identical. B: warp predicted 33.22 s vs actual 34.43 s (**3.5%**), numpy predicted 610.45 s vs actual 719.91 s (**15.2%**), sweep 24/24 on both. C: PASS in 16.2 s, 15/15. D: PASS, caught/`nan` at 1525/1650, 75/325, 50/59275, Monitor cost **-0.69%**. E: PASS on warp in **49.5 s**, and PASS on physics on numpy with the *same printed digits* |
| **Last updated** | 2026-08-27 — session 22 (**T110 done, M8 reached, Phase 1 complete**: `validate/minute.py` is Rung E; **D-075** answers **Q-104** by giving the chooser Rung 3's own domain, 24 D span / 8 D upstream, which is what puts `Cd` inside the published band; **D-076** drops the force probe to 10 samples per convective time, which is how the widened domain still fits in the minute; **D-077** adds measured 160k / 400k rate anchors and closes the Rung B warp blocker; **D-078** re-samples Rung D's cost check; **D-079** raises the default run length to 80 convective times after the README quickstart printed `nan`; `pytest` **772 passed, 1 skipped**) |

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

**None. Both of session 21's entries were T110's to answer and both are closed by measurement.**

- ~~**Rung B fails its accuracy check on `--backend warp`**~~ — **closed, D-077.** The estimator
  was interpolating log-log between two *bandwidth-bound* anchors (40k and 1M) straight through the
  region where this card is *kernel-launch*-bound, and over-predicted the wall clock at 160k cells
  by **1.78x**. Two measured anchors at 160k and 400k on both backends fixed it without touching
  the model or the 25% tolerance: Rung B on warp now reads **predicted 33.22 s, actual 34.43 s,
  error 3.5%**, and the sweep is 24/24 on both backends.
- ~~**The product path's own domain puts `Cd` and `St` outside Rung 3's published bands**~~ —
  **closed, D-075.** The chooser now uses Rung 3's own domain (24 D span, 8 D upstream), measured
  through `flow.Case` to give `Cd` **1.4030** where Rung 3 itself prints **1.4031**. The bands were
  not widened, and no domain flag was added to `flow/` or the CLI — `SPAN_D` and `UPSTREAM_D` are
  module constants (constraint 13). Two follow-on defects that change caused were found by
  *running* things and fixed in the same session: Rung D's cost check went noisy at the larger
  domain (**D-078**) and the default run length stopped being long enough to measure anything at all
  (**D-079**, found by pasting the README's own quickstart into a fresh shell).

## Open questions

- ~~**Q-104** — does `flow/autoconfig.py`'s domain widen to Rung 3's (24 D span, 8 D upstream), or
  do Rung E's bands get restated for a confined case?~~ **Closed in session 22 by measurement — see
  D-075. It widens, to Rung 3's own 24 D / 8 D.** The measurement that decided it is one case with
  one thing changing: 12 D / 3 D gives `Cd` **1.5943**, 16 D / 6 D **1.4523** (still outside),
  20 D / 6 D 1.4426, 24 D / 6 D 1.4360, **24 D / 8 D 1.4030** against Rung 3's own 1.4031. Two
  things that were not obvious before measuring: the **upstream fetch is not second-order** — at a
  fixed 24 D span it is the whole remaining distance to the benchmark's digit — and the cheapest
  in-band option (20 D / 6 D, 288k cells) was rejected rather than taken, because a `Cd` sitting
  0.5% under the band's top with a `cd_std` of 0.0177 is a pass a future session would have to
  re-argue. The wall clock the widening spends was paid back by **D-076**, not by a shorter run:
  Rung E lands at **49.5 s** of the 60.

- ~~**Q-101** — does `python -m lbm.runner` (the M4 gate command) survive as a working entry point
  once `python -m flow` exists, or become a pointer to it?~~ **Closed in session 21 — see D-072.
  It survives, working, plus a pointer line.** Constraint 15 removed one of the three options
  before judgement got a say: `flow/` may import `lbm/` and `lbm/` may **never** import `flow/`, so
  "delegating" was never available and the choice was only keep-vs-pointer. Keeping it costs one
  `--help` line and preserves the knobs `flow` deliberately has not got — `--re` / `--nu`,
  `--resolution` in cells, `--span-d` / `--upstream-d` / `--downstream-d`, `--u-lattice`,
  `--tau-floor`, `--checkpoint` — one of which (`--span-d`) is the knob **Q-104** is a question
  about. The M4 gate command was re-run in full and reproduces to the digit: **301 frames, 40033
  steps, peak |u| 0.06554**, 0.32 MB, h264 528x368 at 60 fps, so `old-Docs/STATE1.md` § Snapshot
  needs no historical marking.
- ~~**Q-102** — is D-017's documented limit (a thin appendage **fused** to a thick body shares its
  component and is not reported) closable without false-alarming on a plain disc?~~
  **Closed in session 19 by measurement — see D-064. Yes.** `flow.prepare.thin_branch_depth` reads
  **2** on every plain disc tested (radius 3..60, four sub-cell offsets each), **1** on squares and
  on Rung 4's two bodies, **2** on Rung 3's cylinder — and **`L+1`** on a hairline of length `L`
  fused to a disc, where `min_thickness` reads the disc throughout. Threshold 3. What D-017's two
  predecessors were missing is the *question*: they asked whether a given cell is thin, which
  digitised curvature answers yes to on every convex body, where this asks how far the thin part
  reaches — and curvature's answer to that is bounded while a hairline's is its own length. The
  measurement is not archived: `validate/shapes.py` section 4 re-runs it every time Rung C runs.
- ~~**Q-103** — what tolerance does cross-backend whole-step agreement actually need?~~
  **Closed in session 15 by measurement — see D-056.** The contract's `max|Δu|/U < 1e-4` at 1000
  steps was met without being touched, and the interesting half of the answer is that the
  disagreement **does not compound**: 2.459e-06 at 10 steps, 1.743e-05 at 100, **9.611e-06 at
  1000** — it rises through the startup transient and then settles an order of magnitude inside the
  bar. Session 14's per-kernel evidence (**D-053**) was the right prior; what it could not say is
  whether 1000 steps of a nonlinear system would amplify it, and the answer is no.

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

Phase 1's budget (`DOCS/IDEA3.md` § Performance budget), **filled in by T103 (session 15)** —
`myenv/Scripts/python.exe bench.py --backend warp`, alternating rounds, best round per backend, one
`Sim` resident, 5 rounds. Conditions, per **D-035**: AMD Ryzen 7 5800H at
`Win32_Processor.CurrentClockSpeed` **3201 MHz of 3201 MHz**, on **mains**; NVIDIA GeForce RTX 3050
Laptop GPU, **driver 592.82**, CUDA Toolkit 12.9 / Driver 13.1, `cuda:0`.

| Grid | Cells | NumPy measured | Warp floor | **Warp measured** | Speedup | Budget target |
|---|---|---|---|---|---|---|
| 400×100 | 40k | 775.1 | ≥2000 | **4155.0** 🟩 | 5× | ~5000 |
| 2000×500 | 1M | 23.0 | ≥250 | **757.3** 🟩 | 33× | ~600 |
| 2000×1000 | 2M | 8.3 | ≥150 | **441.0** 🟩 | 53× | ~400 |

**Every floor is cleared, and 1M and 2M clear the budget's *targets* too.** The NumPy column is
measured in the same alternating rounds rather than quoted from Phase 0, which is what makes the
speedup column mean anything (**D-035**).

**Device memory at 2M cells: 391 MiB in 13 `Sim`-owned arrays** (69 MiB per `(9, ny, nx)` buffer),
independently observed as a **384 MiB** drop in free device memory — the two agree, so it is a
measurement and not an accounting exercise. **2882 MiB of the card's 4096 MiB stay free**, and the
display path does not compete for them: the vorticity field (8 MiB) and its RGB frame (6 MiB) both
live on the host.

Phase 0's own table also moved, and the reason is **D-055** rather than the port. Re-run this
session, `bench.py` (no `--backend`) prints **730.8 / 179.9 / 19.6** steps/s at 40k / 160k / 1M
against floors of 400 / 120 / 15 — all still cleared — with the fused/unfused ratio up from session
10's **1.00 / 1.01 / 1.14×** to **1.16 / 1.20 / 1.14×**. That is exactly what dropping the
pre-collision copy from the fused path predicts, and the change is bitwise identical by test.

**Calibration anchors between 40k and 1M, added in session 22 (D-077).** Same protocol —
`bench.compare_backends`, backends alternated inside each round, one `Sim` resident, warmup, best
round of 3 — at round grids near the sizes the product path actually runs at, rather than at any
one case's exact size. Same conditions: AMD Ryzen 7 5800H at **3201 MHz of 3201 MHz**, on **mains**;
NVIDIA GeForce RTX 3050 Laptop GPU, driver **592.82**.

| Grid | Cells | NumPy measured | Warp measured | us/step (warp) |
|---|---|---|---|---|
| 800×200 | 160k | **185.6** | **3560.4** | 280.9 |
| 800×500 | 400k | **76.5** | **1403.9** | 712.3 |

These two rows are the whole of the Rung B fix. The warp curve is **nearly flat from 40k to 160k**
(log-log slope **-0.11** — the kernel launches, not the bytes, are the cost at that size) and then
turns over sharply (**-1.02** from 160k to 400k), so a model interpolating between 40k and 1M ran
straight through the knee and over-predicted the wall clock at 160k by **1.78x**.

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
| D-054 | 2026-08-18 | **The `Backend` protocol widens to cover the whole timestep, and the state moves onto the device. Supersedes D-052.** Added: `empty`, `zeros`, `copy`, `upload`, `download` (allocation and the two general transfers), `moving_wall`, `inlet_velocity`, `outlet_zero_gradient` (the three boundaries T101 left out), and `force_velocity_shift` / `apply_body_force` (both halves of the Guo scheme). **Backend arrays are opaque handles**: on `NumpyBackend` they *are* `numpy.ndarray` and every transfer is the identity, so the reference path is untouched; on `WarpBackend` they are device arrays and a timestep moves **no bytes across the bus**. `Sim` allocates every buffer through the backend and exposes host views through `host_f()` / `host_u()` / `host_rho()` / `host_f_bb()` — one preallocated mirror each, downloaded on **frame and probe cadence, never per step** — plus `load_f()` and `refresh_inlet_profile()` for the two places a caller writes state at setup. Boolean masks are held as `uint8` on the device and round-trip as `bool`. | **D-051** predicted this widening and **D-052** priced the alternative: T102's host boundary copied in and out per kernel call and was therefore *slower than NumPy*, which is why it quoted no speed number. Nothing in `DOCS/IDEA3.md` § Performance budget is reachable without removing those copies, and removing them means the state has to live where the kernels do — so the seam's real shape was decided by whole-step parity, exactly as **D-051** said it would be. The alternative considered and rejected was a single `backend.step()`: it would have moved the D-011/D-020 order behind the seam, duplicating the one thing `lbm/backends/__init__.py` says a backend does **not** own, and two copies of a timestep order drift. Keeping `NumpyBackend`'s handles as plain `ndarray` is what let the four Phase 0 rungs re-print session 11's digits through a rewritten `Sim`. |
| D-055 | 2026-08-18 | **The pre-collision copy (D-011) is dropped on the fused path, on every backend, and the Warp fused pass streams `f_bb` straight into `f`.** `lbm.core.collide_stream` with `f_bb` supplied stages every direction there and never writes `f` until the stream lands, so `f` is still the pre-collision state when the reflection reads it: passing `f` where D-011's copy would go is **bitwise identical**. It is valid *only* because `f_bb` is supplied — with `f_bb=None` the pass stages in `f` itself and the alias would read values it had already overwritten. Asserted, not argued, by `tests/test_backends.py::test_the_fused_path_needs_no_pre_collision_copy_and_says_so_in_bits` and its Warp twin. | Bandwidth. A naive port moves about **850 MB per step at 2M cells**; `copy(f_pre, f)` is 144 MB of that and the fused path's final `copy(f, buf)` another 144 MB, and the 150 steps/s floor is not reachable while paying both. Removing them is a **removal, not an optimisation** — no arithmetic changes, which is what keeps constraint 1 and **D-033**'s bitwise fused/unfused equality true. The measured side effect on NumPy is recorded rather than hidden: the fused/unfused ratio rose from 1.00–1.14× to 1.14–1.20×, and Rung 3 on `numpy` was re-run to confirm the reference path still prints its published digits. |
| D-056 | 2026-08-18 | **Q-103's answer: cross-backend whole-step agreement is bounded and does not compound.** Measured on Rung 3's case shape (256×64, disc D 16, Zou–He inlet, convective outlet, U 0.05), from a bit-identical start: `max|Δu|/U` = **2.459e-06 at 10 steps, 1.743e-05 at 100, 9.611e-06 at 1000**, against a contract of 1e-4 that was met without being touched. A `warp` checkpoint resumed on `numpy` and run 100 further steps agrees to **8.196e-06**. `validate/parity.py` prints the ladder and the growth factor beside each row, permanently. | **Q-103** asked whether 1e-4 was achievable or generous, and the honest answer needed the *shape* of the curve, not one number: an error that grows linearly from **D-053**'s 5.96e-08 per kernel lands near 6e-05 at 1000 steps, and one that grows geometrically lands nowhere. It does neither — it rises through the startup transient and then settles, because the case is a converging flow and BGK is dissipative. Recording the growth rate rather than only the bound is what makes a *future* regression visible: a port that starts compounding will show it in the 10/100/1000 column long before it crosses 1e-4. |
| D-057 | 2026-08-18 | **Any scalar NumPy computes in `float64` and rounds once to `float32` is computed host-side, in NumPy's own expression order, and uploaded — never recomputed per thread in `float32`.** That covers the Ladd wall correction `6 w_i rho_w (e_i . u_wall)`, Guo's `(1 - 1/(2 tau)) w_i`, `9 (e_i.F)` and `3 (e_i.F)`, and the convective outlet's `1/(1 + lam)`. Each lands in a `(9,)` device array allocated at `WarpBackend` construction, so no boundary allocates inside the step loop. | Constraint 1 says the arithmetic the implementation transcribes may not change, and a scalar recomputed in `float32` on the device is **three extra roundings** where NumPy has one — a difference introduced for tidiness, not forced by the hardware. The measurement is the argument: with the scalars uploaded, `bounce_back`, `moving_wall`, `outlet(copy)` and `moving_wall(u=0)` are **bitwise**, and no boundary exceeds one ulp (worst 5.96e-08). This is the boundary-side counterpart of **D-053**, and it has the same payoff: knowing which comparisons are bitwise turns a future difference there into a bug rather than float ordering. |
| D-058 | 2026-08-19 | **The fluid library carries the numbers its cited sources actually give, and the ordering test asserts the order those numbers produce — which is *not* the order `DOCS/TASKS2.md` § T104's acceptance criterion parenthesises.** The criterion asks for `helium < air < water < oil < glycerine`; the measured ascending order of kinematic viscosity at 20 °C is **water 1.004e-6 < air 1.516e-5 < olive oil 8.4e-5 < helium 1.178e-4 < glycerine 1.120e-3 < honey 7.042e-3 m²/s**. Two of the criterion's four inequalities are false. The criterion's *intent* — the table is ordered by physics, not by typing — is kept and strengthened: `tests/test_fluids.py` asserts the measured order **and** checks every entry's `nu` against its independently cited `mu` and `rho` (`nu = mu / rho`, to 0.2%), which is the check that actually catches a transcription error. A second test, `test_the_ordering_the_contract_asked_for_is_not_physical`, pins the disagreement so a future session cannot quietly edit the data to satisfy the parenthetical. | `nu = mu / rho`, and the densities span four orders of magnitude while the dynamic viscosities span six — so kinematic order is not dynamic order and neither is intuitive. Helium's `mu` (1.96e-5 Pa s) is *larger* than air's (1.825e-5) and its density is 7.2x smaller, so helium's `nu` is ~7.8x air's; water's `mu` is 55x air's but its density is 829x, so water has the **smallest** `nu` of the six. Ordering by `mu` instead would not rescue the criterion either — `helium < air` is false there too, narrowly. Editing the data to fit the sentence was rejected outright: constraint 5 names *"a wrong sim that looks plausible"* as this project's main failure mode, and a fabricated viscosity is that failure mode at its source, one layer below the solver where no rung would ever catch it. `CLAUDE.md` § Session protocol says a spec conflict is logged here rather than silently resolved; this is that log entry, and the spec's sentence is the half that yields because the sources are checkable and the sentence is not. |
| D-059 | 2026-08-19 | **`flow/autoconfig.py`'s generic constants, each cited or measured.** `TAU_FLOOR = 0.54` is applied to **every** case regardless of shape — the strictest of the project's three floors (D-029 bluff body / D-032 generic 0.51 / D-036 Rung 3's 0.537) — because an arbitrary immersed mask cannot be classified disc-vs-square at plan time and the guardrail should be pessimistic. `QUALITY_CELLS = {fast: 30, balanced: 40, accurate: 50}` is the smallest round numbers such that even `"fast"` clears `TAU_FLOOR` for Rung 3's own Re 100 benchmark (`N >= Re (TAU_FLOOR-0.5)/(3U)` ~ 26.7 at the fixed `U_LATTICE_DEFAULT = 0.05`). Domain margins are deliberately **smaller** than Rung 3's tuned values, because Rung B needs guardrails satisfied and 5000 clean steps, not a published force coefficient: `SPAN_D = 12` (8.33% blockage, under the 10% ceiling with a 1.67-point margin, vs Rung 3's 24 D / 4.17%), `UPSTREAM_D = 3`, `DOWNSTREAM_D = 9` (constraint 12's floor is 8, so this keeps a 1 D margin). `RUN_CONVECTIVE_TIMES = 20` (vs Rung 3's 70+60=130, which is sized to *publish* a coefficient) is enough to clear the startup transient; Rung B's own 5000-step-per-case criterion is independent of it. Measured cost, `myenv/Scripts/python.exe -m validate.autoconfig`: the accuracy check (quality="fast", water, Re 100) predicted 61.05 s against an actual 63.75 s (4.2% error, well inside the 25% bar); the whole rung — the accuracy check plus the 24-case sweep (2 fluids x 2 speeds x 2 sizes x 3 quality, 5000 steps each) — ran in **~23 minutes** wall clock on the numpy backend, minutes rather than hours as `PROMPTS/017` asked. | `DOCS/PLAN2.md` § Risks names "auto-config becomes a pile of tuned constants nobody can defend" at this task by name; every number above traces to a Phase 0 decision or is measured in this row rather than picked. The floor choice is the one genuine judgement call the task's Notes flagged (`tau_for`/`tau_for_rung4` disagree with each other by geometry) — resolved by always taking the stricter of the two rather than building a shape classifier neither hand-tuned function needed. |
| D-060 | 2026-08-19 | **`tests/test_flow_package.py`'s constraint-13 scan is precise about input vs. output: a frozen dataclass's auto-generated `__init__` is not scanned.** `inspect.signature(SomeClass)` resolves to `__init__`, and for a frozen dataclass returned as a *result* (`flow.autoconfig.Plan`, and retroactively `flow.fluids.Fluid`) nothing in the product ever calls that constructor with the user's input — only the function that builds one (`plan()`, `fluid()`) does, and its return value is exactly where `DOCS/IDEA3.md` § 1 says `tau` / `dx` / `dt` / `cells_per_length` belong: "derived and printed." A class that is **not** a frozen dataclass, or one with a hand-written `__init__`, is still scanned — `test_the_frozen_dataclass_exemption_is_narrow` proves the exemption is that specific, not "skip classes." | `DOCS/TASKS2.md` § T105 Notes predicted this exact collision: `Plan` is the first object whose fields are legitimately lattice-named, and the Notes said fix the scan's precision rather than weaken the assertion. Confirmed by running it: before this fix, `test_no_public_signature_in_flow_takes_a_lattice_quantity[flow.autoconfig]` failed on `Plan.__init__`'s `tau`/`u_lattice`/`dx`/`dt`/`steps_per_frame`/`cells_per_length` parameters. |
| D-061 | 2026-08-23 | **`Monitor`'s three tripwires, and the measured gap between catching a run and losing it.** Tripwires: peak `|u|` at or above the constraint-3 ceiling on **3 consecutive samples** (sustained, not instantaneous), total fluid mass more than **1%** away from its starting value, and a last-resort "already not finite". Cause is attributed *separately* from symptom: a case at or below **D-029**'s 0.54 floor is blamed on the relaxation time whatever tripwire fired, because `DOCS/IDEA2.md` § Stability row 1 says everything else is downstream of it. **Measured, `numpy`, `SAMPLE_EVERY = 25`:** `tau` 0.533 disc — caught **1525**, `nan` **1650** (7.6% of the run's life earlier); inlet `U` 0.25 at `tau` 0.6 — caught **75**, `nan` **325** (76.9%); inlet-fed, sealed downstream — caught **50**, `nan` **59275** (99.9%). The T106 criterion's *"within 10% of the steps the run would have taken to produce `nan`"* is therefore met by the mode that develops mid-run and beaten by the two whose defect is present from step one. | Two measurements forced each half. **(a) Sustained, not instantaneous**: the D-029 case crosses 0.1 briefly around step **155** and recovers, then dies at 1650 — sampling every 5 steps sees the transient, every 25 does not. A tripwire that fires on it is a false-alarm generator, and a probe users learn to ignore is worse than no probe, so the crossing must persist. **(b) The 10% band cannot be met by all three without making the probe worse.** An over-driven inlet is over the ceiling at step 1 and a sealed outlet leaks mass from the first sample; the only way to delay those to 90% of the run's life is to raise the thresholds past the physics that defines them (constraint 3's 0.1, and incompressibility). Separately measured and worth recording: **peak `|u|` over 0.1 usually does not produce `nan` at all** — a body-force channel ran 30000 steps at peak **7.75** without one, and inlet `U` 0.15 and 0.20 both survived 30000 steps. It produces a *plausible, invalid* answer instead, which is constraint 5's named failure mode, so `Monitor` trips on it regardless of whether a `nan` would ever arrive. Cost, alternating rounds, best round per variant (**D-035**), AMD Ryzen 7 5800H at 3201 of 3201 MHz on mains: **+1.10% / +0.98% / -1.09% / -2.10%** across four measurements — every one inside the 2% limit and inside the machine's own run-to-run spread, so the honest reading is "below the noise floor", not "free". |
| D-062 | 2026-08-23 | **T106's `substituted=True` acceptance criterion is carried to T108 rather than satisfied against a stub.** The criterion names `Result`, the printed summary and the recorded video's metadata; `flow/case.py`, `flow/report.py` and the CLI are T108/T109 and none exist. It is now an explicit acceptance criterion in `DOCS/TASKS2.md` § T108, citing this entry. **The half that could exist shipped in T106**: every suggestion that changes the flow carries *"This is a different flow from the one you asked for -- not your case."* **on the `Suggestion` object**, not in one rendering of it, asserted by `tests/test_diagnose.py::test_a_suggestion_that_changes_the_flow_says_so`. | User's call when the three options were put side by side (defer whole / partial now / build a `Result` stub). `PROMPTS/018` predicted this exact gap and said to raise it at confirmation time rather than reinterpret the criterion or build a stub T108 would then have to reconcile. Constraint 16 is about *every artifact* saying so, which is precisely why the label belongs on the object rather than on the pretty-printer: a note only `explain()` knows about is a note the report and the video metadata will not carry. The dependency order in `DOCS/PLAN2.md` is not wrong — T108 depends on T106 — but one T106 criterion reaches forward past that edge, and this is the record of it. |
| D-063 | 2026-08-23 | **Two of `flow/autoconfig.py`'s suggestions did not fix their own case, and were repaired in T106.** The empty-mask refusal offered `change="size"` with a 1.0 m length — a physics knob for a problem no physics knob touches — and a mask whose thinnest feature no resolution can save offered `quality="accurate"`, which re-raises the identical refusal. Both are now `change="mask"`. The `Suggestion` vocabulary therefore grows from `{speed, size, quality}` to `{quality, mask, fluid, speed, size}` = `flow.diagnose.SUGGESTION_ORDER`, applied by `flow.diagnose.apply_suggestion`; `"fluid"` is created only by `flow/diagnose.py` and only after the library entry's cited `nu` is checked to actually clear the floor. Since the tool cannot invent the user's geometry, a `"mask"` suggestion's `value` is a *description* and Rung D substitutes `flow.diagnose.EXAMPLE_MASK` (the same disc `validate/autoconfig.py` holds fixed) to execute the claim. | T106's own acceptance criterion is *"a suggestion that does not fix its case is a failing test"*, so finding two was the rung working, and leaving them would have meant a Rung D that passes by not looking. `Suggestion.value` stays a `str`/`Quantity` rather than gaining an `ndarray` variant: `Suggestion` is a frozen dataclass, and an array field breaks its generated `__eq__` and `__hash__` for no gain. **Ranking** is by `SUGGESTION_ORDER` and is not cosmetic: `tau = 0.5 + 3 U N / Re` means resolution is the *only* knob that raises `tau` without changing the Reynolds number, so `quality` is the one fix that answers the user's own question and every other fix answers a different one — which is what D-045's *"clearly labelled as not your case"* is about. |
| D-064 | 2026-08-23 | **Q-102 is closed by measurement, not restated: `flow.prepare.thin_branch_depth` detects a hairline *fused* to a thick body and does not false-alarm on a plain disc.** The metric: with `d` the Chebyshev wall distance D-017 already uses, a **core** cell has `d >= 2` (its whole 3x3 is solid) and the **shell** is `d == 1`; the depth is the 8-connected geodesic distance *within the shell* from the nearest core cell. Threshold `THIN_BRANCH_DEPTH = 3`. **Measured this session**, and re-measured by `validate/shapes.py` section 4 on every run rather than quoted: discs of radius 3..60 at four sub-cell offsets each, worst **2**; squares 3..60, worst **1**; straight 3-cell plates **1**; Rung 3's cylinder body **2**; Rung 4's square and polygon bodies **1** and **1**; a 1- or 2-cell hairline of length `L` fused to a disc reads **`L+1`** (3 at L=2, 6 at 5, 11 at 10, 21 at 20) while `min_thickness` reads the *disc* (21) throughout. **The cost is recorded rather than hidden**: the metric also fires on shapes whose thinness is intended — a rotated triangle's apex **4**, a 40x3 ellipse's tip **5**, a three-cell-stroke ring of radius 60 **6**. Every one of those genuinely has a one-cell-thick region, so they are true positives under constraint 12's own rule, but a user meets them as their shape being blunted — which is why `thicken` reports itself in `actions`. | D-017 states the bar in as many words: its own two candidate metrics (run lengths, per-cell 3x3 opening) were written first and *both false-alarm on a plain cylinder*, and a check that cries wolf on the project's own passing benchmarks gets suppressed. So the disc is not one test among many, it is **the** test — which is why the sweep covers sub-cell offsets: `validate/cylinder.py` deliberately offsets its disc by half a cell to break mirror symmetry, the offset disc reads 2 where the centred one reads 1, and a threshold of 2 would therefore have failed on the project's own Rung 3 body. The core/shell split is what the two predecessors were missing. They asked *is this cell thin*, which digitised curvature answers yes to on every convex body; this asks *how far does the thin part reach*, and curvature's answer to that is bounded while a hairline's is its own length. Recording the tapered-shape hits as measured cost rather than filing them as false alarms is the honest reading: an ellipse tapering to one cell leaks through bounce-back exactly as a hairline does. |
| D-065 | 2026-08-23 | **A body size the picture cannot deliver is a refusal, not a quieter resolution.** When no raster box makes the *repaired* body `cells_across` cells across to within `RESOLUTION_TOL = 1`, `prepare()` returns `verdict="refused"` with a `Fix(change="resolution")` naming a size the picture **can** reach — and that size is iterated to a **fixed point** (`_reachable_size`), because the size the search happened to land on is not automatically one that comes back when it is asked for. | Found by a test, not by reasoning: a two-pixel-wide plate in a 200-pixel picture, asked for 40 cells across, was returning a **61-cell** body, silently. That is constraint 16's named failure mode at the worst possible place — **D-040** exists because `tau = 0.5 + 3 U N / Re` is computed from the number that comes back, so a substituted resolution is substituted *physics*, not a cosmetic difference, and D-040's own evidence is a body that ran at `tau = 0.527` while the printed summary claimed 30 cells of resolution. The fixed-point iteration is not fussiness either: the first attempt named 75 cells, and asking for 75 landed somewhere else again, because a feature that only survives at a finer raster changes what `thicken` does. A fix that fails its own case is a failing test (**D-063**), and this one did until it was iterated. |
| D-066 | 2026-08-23 | **`flow/prepare.py`'s remaining constants, and the order the repairs run in.** The order is fixed at `REPAIRS = (fill_holes, drop_specks, largest_component, thicken)` whatever order a caller passes them in, so two callers asking for the same set get the same mask. `MIN_BODY_CELLS = 8` — under it a body cannot be `MIN_THICKNESS_CELLS = 3` thick *and* have an outline, and every quantity derived from `D` (**D-019**) becomes a rounding artefact. A speck is `min(1% of the largest component's area, SPECK_MAX_CELLS = 12)`; the cap is what stops a *small* body having a proportionally small neighbour dropped out from under it. `MAX_THICKEN_PASSES = 4`, where one pass turns a one-cell feature into three, so a shape still failing after four is refused rather than fattened indefinitely into a different shape. **And every repair runs on the solid's bounding box plus `_REPAIR_PAD = 6` cells of fluid, not on the picture** — which is what makes Rung C **10.6 s** rather than over two minutes. | `DOCS/PLAN2.md` section Risks names *a pile of tuned constants nobody can defend* at this layer, and **D-059** set the standard: cite a decision or measure it in the session that adds it. The cropping is the one that was measured into existence — the first working version timed out at two minutes on a corpus of fifteen small images, because `lbm.geometry._label`'s max-propagation costs one pass per component diameter and a 400x400 picture of a 50-pixel body pays for all the whitespace. It is not an approximation: every repair here is local to the solid, and the pad leaves room for the only one that grows outward, which is what makes it a free 20x rather than a trade. **D-059** asks Rung C to be seconds so B, C and D together stay minutes; this is how. |

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
| D-067 | 2026-08-27 | **A refused `Case` is built, not raised: the refusal rides on the object and `run()` raises it.** `Case.from_image(...)` always returns a `Case`; `case.runnable` is `False`, `case.refusal` holds the `Unrepresentable` (or the `Prepared` with `verdict="refused"`), `case.explain()` prints the refusal and its way forward, and `case.run()` raises before a single timestep. **D-065**'s picture refusal is raised as the *same* `Unrepresentable` a physics refusal is, built from the `Prepared`'s `reason` and `fix`, so one `except Unrepresentable` catches both. | The T108 contract's first criterion is that building "runs nothing" and that `explain()` prints the plan **or** the way forward — and a constructor that raises can print neither: the caller has an exception, not an object to interrogate. T109's `--explain` has to print a refusal and exit 2, and doing that through exception plumbing in a CLI is how the explanation ends up living in two places. The rejected alternative was raising at construction and offering a separate `flow.explain_request(...)` free function, which is the same information reachable two ways — the thing **D-062** and **D-045** exist to prevent for `substituted`. What is *not* softened: nothing runs. `run()` raises, and it raises the structured refusal with its suggestions, so constraint 16 holds where it matters. |
| D-068 | 2026-08-27 | **A geometry `Fix` is translated into the user's vocabulary before it is applied: `change="resolution"` becomes the *finest quality level the picture can actually resolve*, and a picture that cannot reach even `"fast"` is substituted by `flow.diagnose.EXAMPLE_MASK` with a sentence saying so.** `Case.nearest()` is where that happens; `flow.prepare.apply_fix` is still what executes the `"picture"` branch. | `flow.prepare.apply_fix` speaks `cells_across`, and the T108 contract says by name that `cells_across` is `prepare`'s argument and **not** `Case`'s (constraint 13) — so a fix cannot be handed back through this layer in the units it was written in. The user-facing knob for body size *is* `quality`, so that is what the fix becomes. **Finest that fits, not coarsest**: the nearest runnable case should be the one nearest to what was asked, and the corpus's `tiny_body.png` shows the difference — refused at `"accurate"` (50 cells) with a reachable size of **41**, it comes back as `"balanced"` (40) rather than `"fast"` (30). When nothing fits, inventing the user's geometry is not an option, so the worked example is run and labelled — exactly the escape `apply_fix` and Rung D already use, rather than a new one. |
| D-069 | 2026-08-27 | **Nothing is measured until the startup kick has switched off *and* washed out of the domain: the measurement window starts at `max(50% of the run, kick_steps + one flow-through)`, and a run shorter than that reports nothing at all rather than reporting the kick.** `flow.case.SETTLE_FLOW_THROUGHS = 1.0`; the shortfall is a `Result.warnings` line naming the step count that would work. | Found by the Re 10 acceptance criterion, and measured rather than reasoned: on a steady Re 10 disc, a window that opens the instant the kick stops reports a lift **amplitude of 0.5463 against a `Cd` of 3.6115** — 15% — which is the kick's shutdown transient decaying, and which reads to everything downstream as a shedding wake. The full decay, 30000 steps, lift amplitude as a fraction of `Cd` per 3000-step block: **0.890, 0.151, 0.041, 0.0123, 0.0043, 0.0018, 0.0009, 0.0005, 0.00012, 0.00004** — monotone through the 1% shedding gate at ~12000 steps. One domain flow-through (`nx / U`) is the time for the perturbed fluid to leave, which is the physical quantity involved; it is also a *floor* rather than a rival to Rung 3's choice, which discards 70 convective times where its kick stops at 3 (~2.5 flow-throughs). Reporting `nan` with a sentence beats reporting a number nobody measured. |
| D-070 | 2026-08-27 | **`Result.strouhal` survives three gates or it is `None`: (1) lift amplitude over `CL_AMPLITUDE_MIN = 1%` of `\|Cd\|`, (2) the window long enough to hold `MIN_PERIODS = 2` of the **longest plausible** period `D / (U * 0.05)`, (3) the estimate inside `ST_PLAUSIBLE = (0.05, 0.5)`.** Gate 2 is deliberately *not* counted against the period the estimate itself implies. | The contract names gate 1, and gate 1 alone is not enough: measured, a synthetic sine planted at `St = 0.17` with **one** period in the window returns **0.459** with a peak 56x the next distinct one — a confident wrong answer, and the artefact constraint 5 exists to prevent. The first attempt at gate 2 counted periods of the *estimate*, which is self-referential and passes exactly when it should fire: the wrong 0.459 implies a short period and therefore "2.7 periods observed". Counting against the slowest shedding we would believe (St 0.05, 20 convective times) is a question whose answer does not depend on the answer being guarded, and it costs nothing real — Rung 3 measures over 60 convective times, and the 130-convective-time product run in this session reports `St` 0.1841 with 12.0 periods. Gate 3's band cites `validate/cylinder.py::lowpass`'s own measurement, an unfiltered FFT reporting `St = 1.49` from the domain's acoustics. |
| D-071 | 2026-08-27 | **Frames kept in memory are budgeted at `FRAME_MEMORY_BUDGET = 512 MB` and the shortfall is *reported*, and a run with no sink and `keep_frames=False` renders nothing at all.** The budget's overflow adds a `Result.warnings` line naming how many of how many frames were kept and what to pass instead; the un-drawn run pushes a shared 1x1 placeholder, never `None`. | `Plan.steps_per_frame` is computed for 60 fps playback, so the default 20-convective-time run at `quality="fast"` asks for ~6000 frames of 421 KB — **2.5 GB**, which is not a default anyone chose. A silent cap was the obvious fix and is the wrong one: a truncated list reads as "these are your frames" when it is the first third of them, so the count kept and the count seen both go in the summary. The placeholder is not cosmetic: `None` is `lbm.runner.RingBuffer`'s "the buffer is empty" sentinel, so returning it from `field` would make a pushed frame indistinguishable from an empty ring and leave `RunStats.delivered` counting frames that were never delivered. |
| D-072 | 2026-08-27 | **`python -m lbm.runner` survives as a working entry point, plus a one-line pointer at `python -m flow` (`lbm.runner.PHASE1_CLI_POINTER`, printed by `main` and repeated in `--help`). Closes Q-101.** The M4 gate command in `old-Docs/STATE1.md` § Snapshot stays literally reproducible and is **not** marked historical. | The T109 contract offered three options and **constraint 15 deleted one of them before judgement got a say**: `flow/` may import `lbm/` and `lbm/` may never import `flow/`, so `lbm/runner.py` cannot delegate to `flow/cli.py` and the pointer has to be a string rather than a call. That left keep-vs-pointer, and keeping wins on what it costs versus what it preserves: one `--help` line against `--re` / `--nu`, `--resolution` in cells, `--span-d` / `--upstream-d` / `--downstream-d`, `--u-lattice`, `--tau-floor` and `--checkpoint` — every solver-level knob `flow` deliberately has not got (constraint 13), and `--span-d` is precisely the knob **Q-104** is a question about, so deleting it would have removed a tool **T110** may want in the very next session. Verified rather than assumed: the gate command re-run in full prints **301 frames, 40033 steps, peak |u| 0.06554**, 0.32 MB, h264 528x368 at 60 fps — session 11's digits exactly, with only the wall clock differing (304.5 s vs 334.9 s), which is not a published band. `tests/test_cli.py` pins all three halves: that it still writes an MP4, that the pointer is in `--help`, and that the nine knobs are still there and still absent from `flow`. |
| D-073 | 2026-08-27 | **`--live` is three-valued, not a switch: absent means Phase 0's rule (a window opens only when no file sink was asked for), `--live` forces one, and `--no-live` suppresses it — and `--no-live` with neither `--out` nor `--frames-dir` is the only way to reach D-071's un-drawn run.** `--quiet` prints the summary and nothing else; it does **not** hide the result. | Both halves were found by running the CLI, not by reading it. **(a)** Preserving `lbm.runner`'s "no sink means open a window" rule is right — it is the behaviour the T109 contract says to preserve — but it left `Case.run`'s un-drawn path (**D-071**: no vorticity field computed, no frame coloured, a 1x1 placeholder to `NullSink`) unreachable from the command line, which is exactly the shape a script or a CI job wants and the shape a person asking "what is the drag coefficient" wants. Phase 0 had no way to ask for it because Phase 0's CLI had no numbers to print at the end. Measured through it: 75000 steps on warp in **50.7 s**, `Cd` **1.5955 ± 0.0157** and `St` **0.1838** — session 20's numpy digits, through the CLI, drawing nothing. **(b)** `--quiet` first passed `quiet=True` straight through to `Case.run`, which suppresses `Result.summary()` as well, so the command printed **nothing at all** while `--help` promised "numbers only". The first version of the test asserted only that the *plan* was absent, which is how it got past; it now asserts both halves. A flag that hides the result is not a quieter answer, it is no answer. |
| D-074 | 2026-08-27 | **The T109 contract's own example command is refused, and that is the correct outcome — D-038 repeating one layer up.** `--shape wing.png --fluid air --speed "5 m/s" --size "10 cm"` is `Re = 5 * 0.1 / 1.516e-5 = 32982`; `tau` reads **0.500182** against the 0.54 bluff-body floor (**D-029**). The criterion is met through `--nearest`, which runs the tool's own top suggestion and marks it. | Identical in shape to **D-038** and settled the same way: two acceptance criteria of one task cannot both be met by one literal command — "writes a playable file" and "refuses a case it cannot represent" — and the refusal wins, because a solver that quietly runs Re 33000 on a 40-cell body with no turbulence model produces exactly the artefact constraint 5 exists to prevent. What T109 has that T011 did not is a **way through**: `flow.diagnose` exists now, so the CLI does not dead-end at the refusal. Run end to end from a cold shell with `--nearest`, the literal command exits **0** and writes a playable 6-frame MP4 whose container comment reads `substituted=True; fluid -> honey: ...` — constraint 16 surviving into the file, checked with `ffmpeg -i` rather than from the counter. Recorded as a decision rather than silently substituting a working command into the criteria, because the next session reading "criterion 1 passed" should know which command actually passed it. |
| D-075 | 2026-08-27 | **Q-104 answered: `flow/autoconfig.py` adopts Rung 3's own domain — `SPAN_D` 12 -> **24**, `UPSTREAM_D` 3 -> **8**, `DOWNSTREAM_D` unchanged at 9. Supersedes D-059's domain choice** (and only that; D-059's `TAU_FLOOR`, `QUALITY_CELLS` and `RUN_CONVECTIVE_TIMES` stand). The bands were **not** widened and no domain flag was added to `flow/` or to the CLI — these are module constants, which is what constraint 13 allows. | Measured, not argued: the finished product path on Rung 3's own case (water, 5 mm/s, 2 cm, `quality="fast"`, Re 99.6, 48000 steps, `warp`, one thing changing at a time) reads `Cd` **1.5943** at 12 D / 3 D against a published band of 1.25..1.45, and **1.4030** at 24 D / 8 D against Rung 3's own **1.4031** on the same machine. The intermediate rows are recorded in `flow/autoconfig.py`'s `SPAN_D` docstring: 16 D / 6 D **1.4523** (still outside), 20 D / 6 D 1.4426, 22 D / 6 D 1.4394, 24 D / 6 D 1.4360. Two things fall out of that table and both are decisions in themselves. **(a)** The upstream fetch is not second-order — at a fixed 24 D span, 6 D gives 1.4360 and 8 D gives 1.4030, which is the whole remaining distance to the benchmark, so restoring `SPAN_D` alone would have bought a pass with 0.014 of margin instead of a reproduction. **(b)** 20 D / 6 D *does* land inside the band at 288k cells rather than 389k, and was rejected: 1.4426 sits 0.5% under the band's top with a `cd_std` of 0.0177, which is a pass that a future session would have to re-argue, and D-059's own failure mode was choosing the cheapest domain that still cleared a rule. The alternative the T110 contract explicitly floated — restating the bands for a confined case — was rejected on the same evidence: the confined number is not a different-but-valid answer, it is 14% of drag from the walls, and `validate/cylinder.py`'s `SPAN_D` docstring had already measured that mechanism at 15 D. **The cost is stated rather than hidden**: 389k cells against 140k, `pytest` from 108 s to 208 s, and Rung B's 24-case sweep proportionally slower on `numpy` — all of it charged to M8's wall clock, where **D-076** paid it back. `DOWNSTREAM_D` stayed at 9 because 9 reproduces Rung 3's digit while Rung 3 spends 12, and three diameters of cells that change nothing are three diameters not worth spending. |
| D-076 | 2026-08-27 | **`flow.case.FORCE_SAMPLES_PER_TIME` 50 -> 10.** The force/residual/peak-speed probe samples ten times per convective time, not fifty. | **D-075** roughly tripled the cell count and the probe is what pays for it on a device backend: one sample is **five host reads** (`forces()` downloads `f` and `f_bb`, `residual()` / `mark_residual()` / the peak-speed check download `u` three times), which at Rung E's 720x540 domain is ~37 MB across the bus, and at 50 samples per convective time that was **37.6 s of a 76.9 s run** — nearly as much wall clock as the timesteps themselves, solving the pair of measurements for a fixed cost plus a per-sample one. Measured on Rung E's own case, `warp`, 48000 steps, everything else identical: 50 samples/tc -> **76.9 s**, `Cd` 1.4030, `St` 0.1676; 10 samples/tc -> **46.8 s**, `Cd` **1.4030**, `St` **0.1676**. Identical to four decimals on both published quantities, 1.64x the speed. The one number that moves is peak `|u|`, 0.09761 -> 0.09725 — 0.4% *downward*, because a coarser sampler sees a slightly smaller maximum, which is a small real loss of pessimism in a constraint-3 check and is recorded rather than hidden. 10 is a floor and not a knob to keep turning: the acoustic ringing `flow.report._lowpass` exists to reject has a period of ~305 steps here, which is ~5 samples per period at this cadence and ~2.5 at half of it — the second margin goes first. |
| D-077 | 2026-08-27 | **`flow.autoconfig._RATE_TABLE` gains measured anchors at 160k and 400k cells on both backends — `numpy` 185.6 / 76.5 steps/s, `warp` 3560.4 / 1403.9 — and that is the whole fix for Rung B on `warp`. The model (log-log interpolation) and the 25% tolerance are unchanged.** | The blocker's own diagnosis, confirmed by measurement. With only 40k / 1M / 2M anchors the model interpolated **between two bandwidth-bound points** straight through the region where this card is *kernel-launch*-bound: at 160k cells it predicted 1996 steps/s against a measured **3560.4**, i.e. a wall clock **1.78x** the real one — session 19's 75.7% error almost exactly, since Rung B measures the error on time. The measured slopes say it plainly — 40k -> 160k is **-0.11** in log-log (nearly flat: the launches, not the bytes, are the cost) and 160k -> 400k is **-1.02**. Adding an explicit launch-overhead term was the alternative and was rejected: a least-squares `t = a + b * cells` over all five measured points (a = 208.8 us, b = 1.049 us per 1000 cells) is **+34.1% at 160k** — worse than the 25% tolerance it would have to satisfy — because the launch-bound to bandwidth-bound transition is a knee, not a sum of two regimes. A model with a fitted constant nobody measured is also exactly what `DOCS/PLAN2.md` § Risks warns about at this layer. Anchors are measurements. Both are taken with `bench.compare_backends`'s own protocol (backends alternated inside each round, one `Sim` resident, warmup, best round) at 800x200 and 800x500 — **round grids near the product's sizes rather than at any case's exact size**, so the table is calibration and not curve-fitting to the rung that checks it. The `numpy` column was measured in the same alternating rounds for the same reason the speedup column exists (**D-035**). |
| D-078 | 2026-08-27 | **Rung D's `Monitor` cost check samples nine rounds of 300 steps instead of five of 600 — same protocol, same total timesteps, same 2% limit, better statistics under thermal drift. The tolerance was not touched and `Monitor` was not changed.** | **D-075** tripled the domain the check runs on; a round went from ~2 s to ~8 s, and the machine's own drift across the resulting 80 seconds is larger than the 2% being measured, so each variant's *maximum* was landing on a different part of a falling curve. Measured, three repeats each on an idle machine: **5x600** gave **+0.79%, +5.82%, -4.32%** — a 10-point spread against a 2-point gate — and **9x300** gave **+0.49%, +1.17%, -1.25%**. What says which of those is the artefact is an independent, contention-free number: `Monitor`'s sample is **0.45 ms** of NumPy at this domain (0.159 ms for the speed field, 0.294 ms for the masked `float64` mass sum) against a **13.2 ms** timestep at a 25-step cadence — **0.14%**, an upper bound no honest measurement can exceed by 40x. This is `71a74d08789c` (queued in session 21 at a -2.10%..+2.57% spread) diagnosed rather than re-observed: the check was reporting the machine, and D-035's own answer to that is more alternation, not a wider gate. Rung D re-run after the change: **-0.69%**, with all three detections at their published steps (1525/1650, 75/325, 50/59275). |
| D-079 | 2026-08-27 | **`flow.autoconfig.RUN_CONVECTIVE_TIMES` 20 -> 80. Supersedes D-059's run length.** The default run is now long enough that the report can report everything it reports. | Found by running the README's own quickstart command in a fresh shell, which is what that acceptance criterion is for: after **D-075** it printed `Cd` **`nan`** and *"the run ended after 16000 steps, before the startup kick had switched off (4000) and washed out of the domain (18400)"*. Two rules set the floor and both are arithmetic. **D-069** opens the window only after the kick has switched off (5 convective times) and washed out (one flow-through = `UPSTREAM_D + 1 + DOWNSTREAM_D` = **18**), so nothing is measured before 23 and — the window being the last 50% — a run under **46** measures nothing at all. **D-070**'s gate 2 wants two of the slowest plausible shedding periods in the window, 20 convective times each, which is 40 of window and **80** of run. 80 is the larger, so the default satisfies both. The interesting part is *why 20 had been passing*: the narrower domain made a flow-through shorter, so the constant D-059 chose for cheapness was silently coupled to the domain D-075 has now changed — exactly the failure mode a constant with a citation but no derivation has. It is written as arithmetic in the docstring now rather than as a number. **The cost is stated**: Rung B's numpy accuracy case goes from ~2 minutes to ~10, and the whole rung on numpy from ~23 minutes to ~65. A shorter run is still reachable, and reachable *honestly*, through `Case.run(seconds=...)`. |


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

### 2026-08-18 — Session 15: T103, the Warp boundaries and the whole timestep — **M5**

**Task worked:** `T103` — Warp boundaries, checkpoint, performance. **Done**, every acceptance
criterion run rather than read.

**Done**
- Read, in the prompt's order: `CLAUDE.md`, this file in full, `DOCS/TASKS2.md` § T103 **and** § T102,
  `DOCS/IDEA3.md` § Performance budget / § Validation ladder (Rung A), `old-Docs/STATE1.md`
  § Performance baseline and **D-011**, **D-020**, **D-021**, **D-022**, **D-033**, **D-035**,
  `DOCS/PLAN2.md` § Dependency graph / § Session map / § Milestone gates / § Risks, and
  `lbm/backends/warp_backend.py` + `validate/parity.py` as session 14 left them.
- **The seam widened first** (**D-054**), because nothing in the budget is reachable while the state
  lives on the host. `lbm/backends/__init__.py` gains ten methods: `empty`, `zeros`, `copy`,
  `upload`, `download`, `moving_wall`, `inlet_velocity`, `outlet_zero_gradient`,
  `force_velocity_shift`, `apply_body_force`. `lbm/backends/numpy_backend.py` implements them as
  allocation plus delegation to the **unchanged** `lbm.boundary` functions.
- **`lbm/backends/warp_backend.py` rewritten device-native.** Six new kernels — `_bounce_back_kernel`,
  `_moving_wall_kernel`, `_inlet_kernel`, `_outlet_copy_kernel`, `_outlet_convective_kernel`,
  `_force_shift_kernel`, `_body_force_kernel` — plus `_collide_bb_kernel`, which with `_stream_kernel`
  is the fused `collide_stream`. Each is a transcription of its `lbm.boundary` counterpart term for
  term (constraint 1). The reflections loop the nine directions inside **one thread per cell** so the
  mask is read once rather than nine times. `_GridBuffers` and the per-call host copies are gone;
  `WarpBackend(shape=...)` is accepted and ignored.
- **The scalars NumPy rounds once are computed host-side and uploaded** (**D-057**) into `(9,)`
  device arrays allocated at construction, so no boundary allocates in the step loop.
- **`lbm/runner.py`: `Sim` owns backend state.** Every buffer comes from `backend.empty`; `step()`
  reaches the boundaries and both Guo halves through the seam, so the module now imports **no kernel
  and no boundary**. Host views are `host_f()` / `host_u()` / `host_rho()` / `host_f_bb()` — one
  preallocated mirror each, on frame and probe cadence (constraint 8) — with `load_f()` and
  `refresh_inlet_profile()` for the two setup-time writes. `vorticity()`, `forces()`, `residual()`
  and `mark_residual()` go through the accessors; `save_checkpoint` still goes through `to_host`
  (**D-050**), which is the only reason a Warp checkpoint is readable on NumPy.
- **The pre-collision copy is gone from the fused path** (**D-055**), on both backends, and the Warp
  fused pass streams `f_bb` straight into `f` — two `(9, ny, nx)` copies per step removed, bitwise
  identical, with the argument written as a test rather than a comment.
- **`validate/parity.py` is now the whole of Rung A**: `--kernels` (T102's), `--boundaries`,
  `--whole-step`, `--checkpoint`, and no flag runs all four, which is M5's gate command. The
  whole-step mode prints the 10 / 100 / 1000 ladder **with the growth factor beside each row**.
- **`--backend` on all four Phase 0 rung scripts.** `cylinder.py` and `polygons.py` pass it through
  `make_config`; `poiseuille.py` and `cavity.py` — which run their own loops rather than using `Sim`
  — now allocate and step through the backend too, with their periodic `nan` / residual checks on
  the **check** cadence rather than the step cadence. Rung 1 is the only case that exercises the Guo
  body force on the device and Rung 2 the only one that exercises `moving_wall`.
- **`bench.py --backend warp`**: `machine_state()` collects the CPU clock, the power state, the GPU
  name and the driver for **D-035**; `compare_backends()` alternates backends round by round with one
  `Sim` resident and per-backend step counts; `measure_footprint()` reports both an accounted and an
  observed device footprint, and runs **before** the timed rounds because Warp's memory pool makes
  the observed figure read zero afterwards.
- **Tests**: `tests/test_warp_backend.py` rewritten for the T103 contract (28 tests — boundaries,
  fused-vs-unfused bitwise, the `f_pre` aliasing, device-resident state, cross-backend checkpoint,
  bitwise restart, 1000 timesteps with pointers and free memory flat, mask round trips); eight added
  to `tests/test_backends.py` for the widened seam on the NumPy side.
- `CLAUDE.md` § Current state and § Module map updated.

**Measured**
- **Rung A, full — PASS.** `myenv/Scripts/python.exe -m validate.parity --backend warp`, `cuda:0`:

  | section | worst | note |
  |---|---|---|
  | kernels | **5.960e-08** (`equilibrium`) | `macroscopic` and `stream` bitwise — session 14's digits, reproduced through a rewritten seam |
  | boundaries | **5.960e-08** (`outlet(conv)`) | `bounce_back`, `moving_wall`, `moving_wall(u=0)`, `outlet(copy)` all **bitwise**; `inlet_velocity` 1.490e-08, `body_force` 1.490e-08, `force_shift` 7.451e-09 |
  | whole step | **9.611e-06** at 1000 steps | 2.459e-06 at 10, 1.743e-05 at 100 — bounded, not compounding (**D-056**) |
  | checkpoint | **8.196e-06** | `warp` → `numpy`, 100 steps on; restart within `warp` bit-identical; contents `config, f, format, solid, step_count` |

  Bar is 1e-6 per kernel and boundary, 1e-4 for the whole step. Neither was touched. Spike test 9/9.
- **All four Phase 0 rungs on the GPU, inside their published bands, printing session 11's digits:**

  | Rung | session 11 / 13 (numpy) | session 15 (numpy) | session 15 (**warp**) |
  |---|---|---|---|
  | R1 Poiseuille | L2 0.3650% | L2 **0.3650%** | L2 **0.3649%** |
  | R2 cavity Re 100 | 0.75%, vortex 0.21 cells | **0.75%**, **0.21** cells | **0.75%**, **0.21** cells |
  | R3 cylinder | St 0.1731, Cd 1.4031 ± 0.0086 | St **0.1731**, Cd **1.4031 ± 0.0086** | St **0.1731**, Cd **1.4031 ± 0.0086** |
  | R4 square | Cd 1.5279 ± 0.0271 | not re-run | Cd **1.5279 ± 0.0271** |
  | R4 polygon | Cd 1.4276 ± 0.0226 | not re-run | Cd **1.4276 ± 0.0226** |

  R3 wall clock **158.1 s (288 steps/s)** on warp against session 13's 365.0 s (125 steps/s); R4
  square 470.6 s, polygon 229.2 s. Those rung rates are held down by the per-step force probe, which
  downloads `f_bb` and `f` **every step** because that is what the rung measures — the kernel rate is
  `bench.py`'s.
- **Performance budget — PASS, every floor, with margin.** See § Performance baseline for the table
  and its **D-035** conditions: **4155.0 / 757.3 / 441.0** steps/s at 40k / 1M / 2M against floors of
  2000 / 250 / 150, which is **5× / 33× / 53×** NumPy measured in the same alternating rounds. 1M and
  2M clear the budget's *targets* as well. Device footprint at 2M cells **391 MiB**, leaving 2882 MiB
  of 4096 MiB free.
- `myenv/Scripts/python.exe -m pytest` → **`428 passed, 1 skipped, 7 warnings`**. 408 before; the one
  skip is `test_a_known_but_uninstalled_backend_names_its_install_line`, unchanged and by design.

**M5 gate output, pasted verbatim** (`DOCS/PLAN2.md` § Milestone gates asks for three things:
`validate.parity --backend warp` printing PASS, all four Phase 0 rungs re-run with
`--backend warp` inside their published bands, and `bench.py --backend warp` clearing
2000 / 250 / 150 steps/s. The rung digits are in the table above; the other two are here.)

```
$ myenv/Scripts/python.exe -m validate.parity --backend warp
Rung A - backend parity, warp vs numpy
  random state: rho in [0.9, 1.1], |u| <= 0.099, tau = 0.6, seed = 20260818
  tolerance: max|numpy - warp| <= 1e-06 in f units
  whole step: max|du| / U < 1e-04 at 1000 steps
  device: cuda:0

1. kernels
  kernel            quantity          grid   max abs diff   bitwise
  -----------------------------------------------------------------
  macroscopic       rho              32x64      0.000e+00   yes  [ok]
  macroscopic       u                32x64      0.000e+00   yes  [ok]
  equilibrium       feq              32x64      5.960e-08   no   [ok]
  collide           f                32x64      1.490e-08   no   [ok]
  stream            f                32x64      0.000e+00   yes  [ok]
  macroscopic       rho            200x400      0.000e+00   yes  [ok]
  macroscopic       u              200x400      0.000e+00   yes  [ok]
  equilibrium       feq            200x400      5.960e-08   no   [ok]
  collide           f              200x400      1.490e-08   no   [ok]
  stream            f              200x400      0.000e+00   yes  [ok]
  macroscopic       rho           500x1000      0.000e+00   yes  [ok]
  macroscopic       u             500x1000      0.000e+00   yes  [ok]
  equilibrium       feq           500x1000      5.960e-08   no   [ok]
  collide           f             500x1000      1.490e-08   no   [ok]
  stream            f             500x1000      0.000e+00   yes  [ok]

  worst per kernel:

  stream spike test, all 9 directions land on cell + E[i]: 9/9

2. boundaries
  kernel            quantity          grid   max abs diff   bitwise
  -----------------------------------------------------------------
  bounce_back       f                32x64      0.000e+00   yes  [ok]
  moving_wall       f                32x64      0.000e+00   yes  [ok]
  moving_wall(u=0)  f                32x64      0.000e+00   yes  [ok]
  inlet_velocity    f                32x64      1.490e-08   no   [ok]
  outlet(copy)      f                32x64      0.000e+00   yes  [ok]
  outlet(conv)      f                32x64      5.960e-08   no   [ok]
  outlet(conv)      prev             32x64      5.960e-08   no   [ok]
  force_shift       u                32x64      0.000e+00   yes  [ok]
  body_force        f                32x64      0.000e+00   yes  [ok]
  bounce_back       f              200x400      0.000e+00   yes  [ok]
  moving_wall       f              200x400      0.000e+00   yes  [ok]
  moving_wall(u=0)  f              200x400      0.000e+00   yes  [ok]
  inlet_velocity    f              200x400      1.490e-08   no   [ok]
  outlet(copy)      f              200x400      0.000e+00   yes  [ok]
  outlet(conv)      f              200x400      5.960e-08   no   [ok]
  outlet(conv)      prev           200x400      5.960e-08   no   [ok]
  force_shift       u              200x400      3.725e-09   no   [ok]
  body_force        f              200x400      7.451e-09   no   [ok]
  bounce_back       f             500x1000      0.000e+00   yes  [ok]
  moving_wall       f             500x1000      0.000e+00   yes  [ok]
  moving_wall(u=0)  f             500x1000      0.000e+00   yes  [ok]
  inlet_velocity    f             500x1000      1.490e-08   no   [ok]
  outlet(copy)      f             500x1000      0.000e+00   yes  [ok]
  outlet(conv)      f             500x1000      5.960e-08   no   [ok]
  outlet(conv)      prev          500x1000      5.960e-08   no   [ok]
  force_shift       u             500x1000      7.451e-09   no   [ok]
  body_force        f             500x1000      1.490e-08   no   [ok]

  worst per kernel:

3. whole step - 256x64, disc D = 16, Zou-He inlet, convective outlet, U = 0.05
  ----------------------------------------------
  the disagreement is bounded, not compounding: a growth factor well under the
  step-count factor beside it means the error is not accumulating (Q-103).

4. checkpoint
  contents: config, f, format, solid, step_count
  f written in the host layout (9, ny, nx) float32 (constraint 4): yes
  written on warp, resumed on numpy, 100 steps on:   max|du| / U = 8.196e-06  [ok]
  restart within warp is bit-identical (constraint 11): yes

PASS
```

```
$ myenv/Scripts/python.exe bench.py --backend warp
backend warp   numpy 2.4.6   python 3.11.15
  cpu:   AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
  clock: 3201 MHz of 3201 MHz   power: mains   (D-035)
  gpu:   NVIDIA GeForce RTX 3050 Laptop GPU   driver 592.82
measuring Sim.step (inlet + convective outlet + immersed disc)
  alternating rounds, best round per backend, one Sim resident (D-035); 5 rounds

       grid      cells      numpy       warp   speedup   floor   target  result
------------------------------------------------------------------------------------
400x100       40,000      775.1     4155.0        5x    2000     5000  PASS
2000x500    1,000,000       23.0      757.3       33x     250      600  PASS
2000x1000   2,000,000        8.3      441.0       53x     150      400  PASS
------------------------------------------------------------------------------------
budget: PASS  (floors from DOCS/IDEA3.md S Performance budget)

device memory at 2,000,000 cells: 391 MiB in 13 Sim-owned arrays (69 MiB per (9, ny, nx) buffer), 384 MiB observed as a drop in free memory
  2882 MiB still free of 4096 MiB - room for the display path (one (ny, nx) vorticity field and its RGB frame are 8 + 6 MiB, and both live on the host)
  the observed drop reads 0 once Warp's memory pool has retained an earlier
  allocation of the same size; the accounted figure is the one to trust.
```


**Decisions made**
- **D-054** — the seam widens to the whole timestep and the state moves onto the device; supersedes
  **D-052**. **D-055** — the pre-collision copy is dropped on the fused path, bitwise identically.
  **D-056** — Q-103's answer: bounded, not compounding. **D-057** — scalars NumPy rounds once are
  computed host-side and uploaded, never recomputed per thread.

**Not done / deferred**
- **No `--backend` on `python -m lbm.runner`.** T103's outputs name the four rung scripts, `bench.py`
  and `validate/parity.py`; the CLI is T109's, and **Q-101** is still open about what that CLI
  becomes. `python -m lbm.runner` is unchanged and still runs on `numpy`.
- **R4 on `numpy` was not re-run** (~40 minutes for a path whose only change is **D-055**). R3 on
  `numpy` **was**, precisely because D-055 touches the fused path on the *reference* backend too, and
  it printed **St 0.1731, Cd 1.4031 ± 0.0086 in 397.4 s (115 steps/s)** — session 11 and 13's digits
  exactly. The evidence standing in for R4 is specific rather than hand-waving: the D-055 removal is
  asserted **bitwise** by
  `tests/test_backends.py::test_the_fused_path_needs_no_pre_collision_copy_and_says_so_in_bits` and
  its Warp twin, `tests/test_perf.py`'s fused-vs-unfused equality still passes, R3 on `numpy` is
  unchanged to four decimals, and R4 on `warp` printed session 11's `numpy` digits to four decimals
  as well.
- **T102's non-contiguous-host-array guard is gone.** It existed to stop a silent host copy on the
  hot path; since the step loop never touches a host array it protects nothing, and `upload` now
  makes its input contiguous, symmetrically with `NumpyBackend.upload`. The shape and dtype guards on
  `to_host` / `from_host` — the ones constraint 4 needs — are unchanged and still tested.
- **No `flow/` package.** T104 onward, and `lbm/` may never import it (**D-042**, constraint 15).
- `DOCS/ISSUES.jsonl` tracking in git — still the user's call, still not acted on. Nothing was queued
  this session; nothing failed.

**Blockers**
- None.

**Rung status after this session**
- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — R1, R2 and R3 re-run on **both** backends and identical;
  R4 re-run on `warp` only (see § Not done for why, and for what stands in its place).
- Phase 1: **A 🟩** · B ⬜ · C ⬜ · D ⬜ · E ⬜ — Rung A green in full.

**Next**
- Paste `PROMPTS/016-t104-quantities-fluids.md` into a fresh session. It runs `/start-task T104`.
- T104 is the first `flow/` task and the first that is **not** solver work: `flow/quantity.py` and
  `flow/fluids.py`, physical units parsed and a fluid library, with constraint 13 (no lattice
  quantity in any public `flow/` signature) and constraint 15 (`flow/` may import `lbm/`, never the
  reverse) live from the first file.
- `DOCS/PLAN2.md` § Risks, the hard valve on "the trap": **it was not needed.** T102 and T103 both
  landed on schedule and the port is done. Phase 1's remaining seven tasks are product work, and the
  next mention of a kernel should be Phase 2's XLB swap.

---

### Session 16 — 2026-08-19 — T104: physical quantities + fluid library

**Task:** T104 — Physical quantities + fluid library. **Status: done.** Every acceptance criterion
run, not read. `pytest` **547 passed, 1 skipped** (428 → 547; 119 new). **This was the first session
of the phase that is not solver work**, and `lbm/` was not touched — `git status -- lbm` is empty,
which is the honest argument that the Phase 0 rungs are unaffected, and R1/R2 were re-run anyway.

**Done**

- **`flow/` exists** (**D-042**), with three files:
  - `flow/__init__.py` — the package docstring states the two rules that govern everything landing
    here (constraint 13, constraint 10) and re-exports the T104 surface.
  - `flow/quantity.py` — `Quantity`, `parse(spec, *, expect=None, default_unit=None)`, `to_si`,
    six dimensions (`LENGTH`, `SPEED`, `TIME`, `VISCOSITY`, `TEMPERATURE`, `DENSITY`), ~90 unit
    spellings in one `(dimension, factor, offset)` table. Affine conversion throughout so temperature
    is not a special case. `Quantity` is immutable (`__slots__` + blocked `__setattr__`), carries
    `given` verbatim for reports, and accepts `str | float | int | Quantity` so T105 needs no type
    switch. **No dependency** — `flow/quantity.py` imports `re` and nothing else, asserted by an AST
    scan against `sys.stdlib_module_names` (**D-031**'s precedent; `pint` was not adopted).
  - `flow/fluids.py` — `Fluid(name, nu, rho, T, source, mu_pa_s)` and `FLUIDS` with six cited
    entries, `fluid(name)` resolving case, padding, `_`/`-` separators and the aliases
    glycerol/glycerin/H2O/He, `known_fluids()`.
- **Constraint 15 is enforced rather than aspirational**, in
  `tests/test_flow_package.py::test_no_module_under_lbm_imports_flow`: an AST scan over every file
  under `lbm/` (module level *and* inside function bodies), plus a runtime scan that imports every
  `lbm.*` module and asserts nothing in its namespace came from `flow/`. A companion test,
  `test_the_constraint_15_scan_would_actually_catch_a_violation`, feeds the scanner three synthetic
  violations and the legal direction — **a guard that never fires is not a guard.**
- **Constraint 13 is enforced the same way**: `inspect.signature` over every public function, class
  and public method reachable through each `flow` module's `__all__`, refusing any parameter in
  `LATTICE_NAMES` (`tau`, `u_lattice`, `steps_per_frame`, `cells_per_length`, `nx`, `ny`, `dx`,
  `dt`, …), plus a check that no public *name* is one. Also teeth-tested.
- **Refusals follow D-045's shape one layer down**: every `ValueError` names **what was given**,
  **what dimension was expected**, and **one valid example**, and `tests/test_quantity.py` asserts
  all three parts rather than asserting that *an* exception was raised. A separate test asserts the
  example in every refusal itself parses to the dimension it illustrates — a fix that does not work
  is not a fix.
- **The silent-substitution case is pinned**: `parse("20 kg/m^3", expect=SPEED, default_unit="m/s")`
  raises. A declared default unit is for *unitless* input only and never reinterprets a wrong unit
  (constraint 16, in miniature).
- Error-message text is ASCII (`--`, not an em dash): a Windows console at its default codepage
  mojibakes U+2014, and these strings are what T109's CLI prints.

**Rungs re-run this session** (all on the code as shipped, not from memory)

| Rung | Command | Result |
|---|---|---|
| A | `validate.parity --backend warp` | **PASS** — kernels/boundaries worst 5.96e-08, whole step 9.611e-06 at 1000 steps, checkpoint cross-backend 8.196e-06 |
| R1 | `validate.poiseuille` | **PASS** — L2 0.3650%, peak `|u|` 0.07955 |
| R2 | `validate.cavity --re 100` | **PASS** — max dev vs Ghia 0.75%, vortex 0.21 cells, peak `|u|` 0.08797 |
| R3 | `validate.cylinder --backend warp --headless` | **PASS** — St **0.1731**, Cd **1.4031 ± 0.0086**, blockage 4.17%, peak `|u|` 0.09685 |
| R4 | `validate.polygons --backend warp --headless` | **PASS** — square PASS; polygon Cd **1.4276**, Cl amplitude 0.3689, blockage 3.90%, 9.62 D downstream, 19 cells thick, peak `|u|` 0.08944 |

Every digit matches session 11's published values, which is the expected result for a session that
did not touch `lbm/` — the point of running them is that "expected" is not "checked". Unlike session
15, **R4 was run rather than argued for**, so the git-status argument and the measurement now agree.

**Measurements and evidence**

- Ascending kinematic viscosity at 20 °C, from the cited numbers:
  **water 1.004e-6 · air 1.516e-5 · olive oil 8.4e-5 · helium 1.178e-4 · glycerine 1.120e-3 ·
  honey 7.042e-3 m²/s.** Every entry's `nu` agrees with its independently cited `mu / rho` to
  better than 0.2%.
- `flow/fluids.py`'s air (1.516e-5) and water (1.004e-6) agree with the values `lbm/units.py`
  documents (1.5e-5, 1.0e-6) to 2% — asserted, because the two faces of the units boundary
  disagreeing about what air is would be exactly the failure constraint 13 exists to prevent.

**Not done / deferred**

- **Nothing in the contract is outstanding.** All seven acceptance criteria are checked in
  `DOCS/TASKS2.md` § T104.
- `Fluid.temperature_note()` returns a *sentence*, not a structured refusal: the T104 Notes require
  that a user asking for water at 80 °C is told the value is the 20 °C one, and doing more than
  telling them is judgement, which is T106's. The structured version belongs with
  `flow/diagnose.py`.
- Dynamic viscosity is carried as a plain `float` in Pa s (`Fluid.mu_pa_s`) rather than as a
  `Quantity`: Pa s is not a unit anyone describes a case in, so it stays out of the
  `flow.quantity` vocabulary. It exists so `nu = mu / rho` is a **checkable identity** rather than a
  retyped number.

**Decisions made**

- **D-058** — the fluid library carries the numbers its cited sources actually give, and the ordering
  test asserts the order those numbers produce, which is **not** the order the T104 acceptance
  criterion parenthesises (`helium < air < water < oil < glycerine`). Two of that sentence's four
  inequalities are false for kinematic viscosity, and one is false for dynamic viscosity too. The
  criterion's *intent* is kept and strengthened: measured order asserted, `nu = mu / rho` checked per
  entry, and `test_the_ordering_the_contract_asked_for_is_not_physical` pins the disagreement so the
  data cannot be quietly edited to fit the sentence. Logged per `CLAUDE.md` § Session protocol rather
  than silently resolved; `DOCS/TASKS2.md` § T104 carries a "Deviation recorded" note pointing here.

**Blockers:** none.

**Found while closing, not fixed here**

- **`.gitignore` lines 14-15 (`*/__init__.py`, `__init__.py`) and line 17 (`tools`) drop real source
  from the repo.** Verified with `git ls-files`: **`lbm/backends/__init__.py` has been untracked
  since T101** — the `Backend` protocol is not in the repository, so a fresh clone cannot
  `import lbm.backends` — and **the whole of `tools/` is untracked**, although `CLAUDE.md` documents
  `python -m tools.issues` and calls `DOCS/ISSUES.jsonl` committed. `lbm/__init__.py` and
  `validate/__init__.py` are tracked only because they predate the rule. `flow/__init__.py` hit the
  same rule this session and was **force-added** (`git add -f`) so T104's deliverable is actually in
  the repo. The rest is **queued as issue `495777c58269`**, not fixed: it touches `lbm/` and
  repo-wide config, which `DOCS/PLAN2.md` § Risks (last row) says is a `/new-task`, never folded into
  a product task.

**Housekeeping**

- Session 15's T103 work was still uncommitted at session start (last commit `238b404`, 14 files,
  +3387/−627). It was committed as `816db29` **before** any T104 file was created, so that
  "nothing under `lbm/` moved" is a readable claim rather than an assertion buried in a mixed diff.

**Next:** **T105 — auto-configuration**, session 17, gate **Rung B**, milestone **M6**. It is the
module `DOCS/IDEA3.md` § 1 calls the most product-defining in the phase, and `DOCS/PLAN2.md` § Risks
aims its "pile of tuned constants nobody can defend" row directly at it. Prompt written to
`PROMPTS/017-t105-autoconfig.md`. The one thing that session should read before writing code is
`validate/cylinder.py::tau_for` and `validate/polygons.py::tau_for_rung4` — hand-tuned instances of
the function it is about to write.

### 2026-08-19 — Session 17: T105, auto-configuration — **M6**

**Task:** T105 — Auto-configuration. **Status: done.** Every acceptance criterion run, not read.
`pytest` **565 passed, 1 skipped** (547 → 565; 18 new: 15 in `tests/test_autoconfig.py`, 3 in
`tests/test_flow_package.py`). `lbm/` untouched — `git status -- lbm` empty, and R1/R2/R3/R4 plus
Rung A were all re-run anyway, on `--backend warp`.

**Done**

- **`flow/autoconfig.py`** — `Plan` (frozen dataclass: `cells_per_length`, `tau`, `u_lattice`,
  `domain`, `steps`, `steps_per_frame`, `vorticity_limit`, `dx`, `dt`, `Re`, `warnings`, `why`),
  `plan(*, fluid, speed, size, mask, quality) -> Plan`, `Suggestion`, `Unrepresentable`
  (structured: `reason`, `quantity`, `value`, `limit`, `suggestions`), `QUALITY_LEVELS`.
  `mask` is read only for the immersed object's proportions (bounding box, via `lbm.geometry`) — a
  `Plan` carries no `solid` array; rasterising the body into a grid stays T107's job. Every guardrail
  is enforced *before* `Plan` is built and every constant cites a decision or is measured this
  session (**D-059**): `TAU_FLOOR = 0.54` applied to every case regardless of shape (the strictest of
  the project's three floors — D-029/D-032/D-036 — since an arbitrary mask cannot be classified
  disc-vs-square at plan time), `QUALITY_CELLS = {fast: 30, balanced: 40, accurate: 50}` (the
  smallest resolutions such that even `"fast"` clears the floor for Rung 3's own Re 100 benchmark),
  `SPAN_D = 12` / `UPSTREAM_D = 3` / `DOWNSTREAM_D = 9` (smaller margins than Rung 3's tuned 24 D —
  Rung B needs guardrails satisfied and 5000 clean steps, not a published force coefficient),
  `RUN_CONVECTIVE_TIMES = 20`. `Plan.estimated_seconds(backend)` predicts wall clock from a log-log
  interpolation of the measured `DOCS/STATE2.md` § Performance baseline table.
- **`flow/__init__.py`** re-exports `Plan`, `Suggestion`, `Unrepresentable`, `plan`, `QUALITY_LEVELS`.
- **`tests/test_flow_package.py` fixed the collision the T105 contract's Notes predicted**: the
  constraint-13 scan's `inspect.signature(SomeClass)` resolves to `__init__`, and for a frozen
  dataclass returned as a *result* (`Plan`) that flagged the contract's own acceptance criterion —
  `tau`/`u_lattice`/`dx`/`dt`/`steps_per_frame`/`cells_per_length` are supposed to be `Plan` fields.
  `_is_frozen_output_record` exempts a frozen dataclass's auto-generated constructor from the scan and
  nothing else; `test_the_frozen_dataclass_exemption_is_narrow` proves a hand-written `__init__` (or a
  non-frozen class) is still caught (**D-060**).
- **`validate/autoconfig.py`** — Rung B's harness, `python -m validate.autoconfig [--backend]`: (1)
  the accuracy check, `Plan.estimated_seconds` against a real timed run of the committed cylinder
  case (water, Re 100, `quality="fast"`); (2) the 24-case sweep, 2 fluids x 2 speeds x 2 sizes x 3
  quality levels, each run 5000 steps through a real `Sim`, checked against `lbm.geometry.check_mask`
  independently (not trusted from `Plan`), for `nan`, for peak `|u| < 0.1`, and for `Re` reproduced to
  0.1% through `LatticeUnits.reynolds()`.
- **`tests/test_autoconfig.py`** — 15 tests: every `Plan` field has a `why` entry (and `why` has no
  extra keys), each guardrail's refusal cites its decision (`D-029`/`D-032`/`D-036` for `tau`,
  `D-017` for thickness, `D-019`/`D-026` for blockage/downstream), a real thin-mask case (a vertical
  plate whose bounding-box height *is* its thickness, so scaling never clears the floor) and an empty
  mask both raise `Unrepresentable`, the D-038 case's suggestions are fed back through `plan()` and
  proven to actually fix it, quality levels are a strict refinement (`cells_per_length`, `steps` and
  `estimated_seconds` all strictly increase fast → balanced → accurate) while sharing one `u_lattice`,
  an unknown quality or backend raises `ValueError`, and `Re` matches the physical inputs to 1e-9.

**Measured**

- **Rung B — PASS.** `myenv/Scripts/python.exe -m validate.autoconfig`:
  - accuracy: predicted **61.05 s**, actual **63.75 s**, error **4.2%** (limit 25%).
  - sweep: **24/24 cases pass** — every guardrail holds on the *rasterised* geometry, no `nan`, worst
    peak `|u|` **0.0695** (water/accurate, well under 0.1), worst `Re` reproduction error
    **0.0000%** (limit 0.1%). Whole rung (accuracy check plus the 24-case sweep) ran in **~23
    minutes** wall clock on `numpy` — minutes, not hours, as `PROMPTS/017` asked.
- **Rungs re-run this session, all inside their published bands, printing session 11/15's digits:**

  | Rung | Command | Result |
  |---|---|---|
  | A | `validate.parity --backend warp` | **PASS** — kernels/boundaries worst 5.96e-08, whole step 9.611e-06, checkpoint cross-backend 8.196e-06 |
  | R1 | `validate.poiseuille` | **PASS** — L2 0.3650%, peak `|u|` 0.07955 |
  | R2 | `validate.cavity --re 100` | **PASS** — max dev vs Ghia 0.75%, vortex 0.21 cells, peak `|u|` 0.08797 |
  | R3 | `validate.cylinder --backend warp --headless` | **PASS** — St **0.1731**, Cd **1.4031 ± 0.0086**, peak `|u|` 0.09685 |
  | R4 | `validate.polygons --backend warp --headless` | **PASS** — square PASS; polygon Cd **1.4276 ± 0.0226**, Cl amplitude 0.3689, peak `|u|` 0.08944 |

**Decisions made**

- **D-059** — `flow/autoconfig.py`'s constants, each cited or measured this session (see § Decisions
  for the full table: `TAU_FLOOR`, `QUALITY_CELLS`, `SPAN_D`/`UPSTREAM_D`/`DOWNSTREAM_D`,
  `RUN_CONVECTIVE_TIMES`, and the measured Rung B cost).
- **D-060** — the constraint-13 scan is precise about input vs. output: a frozen dataclass's
  auto-generated constructor is not scanned, because nothing calls `Plan(tau=..., ...)` directly —
  only `plan()` does, and its return value is exactly where those numbers are supposed to live.

**Not done / deferred**

- **Nothing in the T105 contract is outstanding.** All seven acceptance criteria are checked in
  `DOCS/TASKS2.md` § T105.
- The Notes' question — does `plan()` reproduce `validate/cylinder.py::tau_for` /
  `validate/polygons.py::tau_for_rung4`'s choices "within a factor"? — is answered by design rather
  than by matching their numbers cell for cell: `plan()` does not try to reproduce Rung 3's specific
  `(D=20, U=0.06)` point, it produces a *safer* one for the same Re (at `quality="fast"`,
  `N=30, U=0.05` gives `tau=0.545` for Re 100, comfortably above both Rung 3's 0.5378 and this
  module's own 0.54 floor). The two hand-tuned functions disagree with each other by *geometry*
  (0.537 disc vs. 0.54 bluff body) — `plan()` resolves that disagreement by always taking the
  stricter one, which is recorded as the judgement call in **D-059** rather than reproducing either
  function's exact arithmetic.
- `flow/diagnose.py` (turning `Unrepresentable` into prose, Rung D) is explicitly **T106's**, not
  written here — `plan()` raises the structured exception; it does not format it.
- `DOCS/ISSUES.jsonl` — one stale entry queued by `pytest`'s auto-capture during an in-session test
  fix (a test-construction bug, not a product defect) was dropped (`835809f057bb`) once the real fix
  landed. The pre-existing `.gitignore` entry (`495777c58269`, session 16) is untouched — still not
  this task's to fix.

**Blockers:** none.

**Rung status after this session**

- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run on `--backend warp`, session 11/15's
  digits exactly.
- Phase 1: **A 🟩** · **B 🟩** · C ⬜ · D ⬜ · E ⬜ — Rung B green in full. **M6.**

**Next:** **T106 — diagnosis, refusal, nearest runnable case**, session 18, gate **Rung D**. It turns
`flow.autoconfig.Unrepresentable` into plain-language explanation plus a suggestion that is *proven*
to work (feed the tool's own top suggestion back through `plan()` and run it), and adds `Monitor` for
live divergence detection. Prompt written to `PROMPTS/018-t106-diagnose.md`.

### 2026-08-23 — Session 18: T106, diagnosis and refusal — **Rung D**

**Task:** T106 — Diagnosis, refusal, nearest runnable case. **Status: done**, with **one criterion
deliberately carried to T108** (**D-062**, the user's call) and **one deviation recorded**
(**D-061**). Every other criterion was run, not read. `pytest` **610 passed, 1 skipped** (565 → 610;
45 new, all in `tests/test_diagnose.py`). `lbm/` untouched — `git status -- lbm` is empty — and all
four Phase 0 rungs plus A and B were re-run anyway.

**Done**

- **`flow/diagnose.py`** — the other half of **D-045**:
  - `explain(exc, *, request=None) -> str`. Three sections: one paragraph a non-specialist can act
    on, `What would work`, and `Details`. The first paragraph carries **no lattice quantity** and no
    `Reynolds`, `Mach`, `grid` or `constraint` either — `tests/test_diagnose.py` greps a superset of
    the criterion's list, which is how the word `grid` was caught and reworded.
  - `suggest(*, fluid, speed, size, mask, quality) -> list[Suggestion]`, ranked by
    `SUGGESTION_ORDER = (quality, mask, fluid, speed, size)`, each note rewritten into plain language
    and each case-changing one labelled *"not your case"* **on the object** (**D-062**). A case that
    plans returns `[]` — an empty list, never an invented alternative.
  - `apply_suggestion(suggestion, **request) -> dict` — the modified request, ready for `plan()`.
    This is what turns a suggestion into an executable claim; Rung D is
    `plan(**apply_suggestion(...))` followed by 2000 steps.
  - `classify(exc)` over `REFUSAL_CLASSES`, raising loudly on an unregistered refusal — an
    unexplained refusal is the dead end D-045 exists to remove.
  - `Monitor` (a `per_step` probe, **D-025**) and `Diverging`. Sampled every 25 steps, three
    whole-array reductions into buffers allocated once, cause attributed separately from symptom
    (**D-061**).
- **`validate/refusals.py`** — **Rung D**, `python -m validate.refusals [--backend] [--steps]`, four
  sections: the refusal classes, the D-038 case printed in full, `Monitor` against the three
  `DOCS/IDEA2.md` § Stability failure modes, and `Monitor`'s cost by alternating rounds (**D-035**).
- **`flow/autoconfig.py`** — two suggestions repaired because they did not fix their own case
  (**D-063**), and `Suggestion`'s docstring now names the whole five-value vocabulary.
- **`tests/test_diagnose.py`** — 43 tests, including the **D-038 golden string** pinned in full, a
  teeth-test that an unregistered refusal class is loud, and `_FakeSim`, which drives `Monitor` by
  hand because a real sim cannot be asked to cross the ceiling for exactly one sample.
- **`CLAUDE.md`** § Commands gained the three Phase 1 rung commands and § Module map gained the four
  `flow/` rows; `.claude/commands/validate.md` gained the Rung A–E table, which it had been missing
  since session 13.

**Measured**

- **Rung D — PASS**, `myenv/Scripts/python.exe -m validate.refusals`, ~9 minutes on `numpy`:

  | Refusal class | top suggestion | replanned | 2000 steps |
  |---|---|---|---|
  | relaxation (D-038: air, 20 m/s, 1.5 m) | speed → 0.0012128 m/s | 480x520 | clean, peak 0.0727 |
  | thickness, 3-cell plate | quality → balanced | 480x483 | clean, peak 0.0669 |
  | thickness, 1-cell plate | mask → redraw thicker | 360x390 | clean, peak 0.0670 |
  | empty_mask | mask → a picture with a body | 360x390 | clean, peak 0.0714 |
  | speed_ceiling | speed → halved | *unreachable through `plan()`* | clean on the base case |
  | blockage | quality → fast | *unreachable through `plan()`* | clean on the base case |

  `speed_ceiling` and `blockage` are **unreachable by construction** — `flow/autoconfig.py` fixes the
  lattice velocity and the domain span (**D-059**) — so the rung explains them from a synthetic
  refusal built with the same fields and *prints that it is doing so*, rather than silently covering
  five classes and calling it six.
- **`Monitor`, caught vs `nan`** (see **D-061** for the full reasoning): `tau` below the floor
  **1525 / 1650**; past the 0.1 ceiling **75 / 325**; mass drift **50 / 59275**. No false alarm — a
  healthy case ran 2000 steps untouched at peak 0.0714, drift 5.66e-05.
- **`Monitor`'s cost: +1.10% / +0.98% / -1.09% / -2.10%** across four alternating-round
  measurements, all inside the 2% limit and inside the machine's run-to-run spread. AMD Ryzen 7
  5800H at 3201 MHz of 3201 MHz, on mains.
- **Rungs re-run this session, every one green:**

  | Rung | Command | Result |
  |---|---|---|
  | A | `validate.parity --backend warp` | **PASS** — worst 5.96e-08, whole step 9.611e-06, checkpoint 8.196e-06 |
  | B | `validate.autoconfig` | **PASS** — predicted 61.05s vs actual **53.11s**, error **15.0%** (limit 25%); 24/24; worst peak 0.0695; worst `Re` error 0.0000% |
  | D | `validate.refusals` | **PASS** — the table above |
  | R1 | `validate.poiseuille` | **PASS** — L2 0.3650% |
  | R2 | `validate.cavity --re 100` | **PASS** — 0.75%, vortex 0.21 cells |
  | R3 | `validate.cylinder --backend warp --headless` | **PASS** — St **0.1731**, Cd **1.4031 ± 0.0086**, peak 0.09685 |
  | R4 | `validate.polygons --backend warp --headless` | **PASS** — square Cd **1.5279 ± 0.0271**, polygon Cd **1.4276 ± 0.0226**, Cl amplitude 0.3689, peak 0.08944 |

  Rung B's *accuracy* number moved from session 17's 4.2% to **15.0%** and the prediction did not:
  `Plan.estimated_seconds` said **61.05 s** both times, and the timed run came in at **53.11 s**
  rather than 63.75 s. That is the machine, not the model — **D-035**'s whole point — and 15.0% is
  well inside the 25% bar. Recorded rather than smoothed over.

**Decisions made**

- **D-061** — `Monitor`'s tripwires and the measured catch-vs-`nan` gap per failure mode, including
  the two measurements that forced the design: the sustained-crossing rule (the D-029 case crosses
  0.1 at step 155 and recovers) and the finding that **peak lattice velocity over 0.1 usually does
  not produce `nan` at all** — 30000 steps at peak 7.75 without one. It produces a plausible,
  invalid answer, which is why the probe trips on it anyway.
- **D-062** — the `substituted=True` criterion is carried to T108, which now has it as an explicit
  acceptance criterion. The half that could exist shipped here.
- **D-063** — two T105 suggestions did not fix their own case and were repaired; the `Suggestion`
  vocabulary grew to five values with `flow.diagnose.SUGGESTION_ORDER` as its authority.

**Not done / deferred**

- **The `substituted=True` criterion** — see **D-062**. This is the one T106 criterion left
  unchecked in `DOCS/TASKS2.md`, deliberately and with a named owner, not forgotten.
- **`Monitor` was measured on `numpy` only.** The probe reads through `host_u()` / `host_rho()`,
  which are a device download on `warp`, so its cost there is a *different* number and is not
  claimed. `validate/refusals.py` already takes `--backend warp`; the measurement belongs to whoever
  first runs a real product case on the GPU (T110's Rung E is the natural place).
- **Rasterising the user's own mask** is Rung C's (T107). Rung D runs the canonical disc at the
  planned resolution and says so in its module docstring — the suggestion changed the *parameters*,
  and those are what the 2000 steps exercise.

**Blockers:** none.

**Housekeeping**

- `DOCS/ISSUES.jsonl` — nothing new queued this session. The `.gitignore` entry from session 16
  (`495777c58269`) is still open and still not this task's to fix.

**Rung status after this session**

- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run, session 11/15's digits.
- Phase 1: **A 🟩** · **B 🟩** · C ⬜ · **D 🟩** · E ⬜.

**Next:** **T107 — geometry preparation + shape corpus**, session 19, gate **Rung C**, milestone
**M7**. It is the last dependency T108 is waiting on, and it owns **Q-102** (D-017's documented
limit: a thin appendage fused to a thick body shares its component and is not reported). Prompt
written to `PROMPTS/019-t107-prepare.md`.

### 2026-08-23 — Session 19: T107, geometry preparation and the shape corpus — **Rung C**, **M7**

**Task:** T107 — Geometry preparation + shape corpus. **Status: done.** Every acceptance criterion
was run, not read. `pytest` **663 passed, 1 skipped** (610 -> 663; 53 new: 51 in
`tests/test_prepare.py`, 2 auto-parametrised in `tests/test_flow_package.py` for the new module).
`lbm/` untouched — `git status -- lbm` is empty — and every rung at or below this task was re-run
anyway.

**Done**

- **`flow/prepare.py`** — `prepare(source, cells_across, *, repair=True, verbose=False) -> Prepared`
  (frozen output record: `mask` `(ny, nx)` bool cropped to the body, `verdict` `"ok" | "repaired" |
  "refused"`, `actions`, `properties`, `reason`, `fix`, `warnings`), plus `Fix`, `apply_fix`,
  `measure`, `thin_branch_depth`, `REPAIRS`, `VERDICTS`. Accepts a PNG, an SVG (the **D-031**
  subset) or a bool array. It builds **on** `lbm/geometry.py` and forks nothing — `check_mask`,
  `min_thickness`, `_label` and `_wall_distance` are imported, asserted by
  `test_prepare_builds_on_lbm_geometry_rather_than_forking_it`.
- **Four repairs, each individually switchable (`repair=False` / `repair=["fill_holes", ...]`) and
  each reporting itself** as one keyed line in `actions` (`"fill_holes: filled 1 interior hole
  (37 cells) that the flow could never reach"`): `fill_holes`, `drop_specks`, `largest_component`,
  `thicken`. Order is fixed at `REPAIRS` whatever order a caller passes (**D-066**), so two callers
  asking for the same set get the same mask.
- **`thin_branch_depth` — the Q-102 metric** (**D-064**), which closes D-017's documented limit.
  `prepare` also records `min_thickness_before` / `thin_branch_depth_before` / `components_before` /
  `holes_before` in `properties`, so the corpus records what went **in** broken, not only what came
  out clean.
- **`tests/data/shapes/`** — **15 committed images (46 KB in total) plus `generate.py`, the
  generator that produces them**, and `expectations.json`.
  `test_the_committed_images_are_the_ones_the_generator_produces` compares the committed PNGs to the
  generator pixel for pixel, so a hand-edited binary is a failing test.
- **`validate/shapes.py`** — **Rung C**, `python -m validate.shapes [--write-expectations]`, four
  sections: the corpus against the committed table; constraint 12 after repair; **D-040** (measured
  body vs requested, within 1 cell); refusals whose own fix is executed; and the **Q-102
  measurement, re-run every time rather than quoted**.
- **`flow/__init__.py`** re-exports the new surface. Note the deliberate shadowing: `flow.prepare`
  the *callable* wins over `flow.prepare` the *module* on the package, because `flow.prepare(picture,
  40)` is the API the product wants; the module is reached as `from flow.prepare import ...` or
  `sys.modules["flow.prepare"]`.
- **`CLAUDE.md`** § Commands gained the Rung C line and § Module map gained the `flow/prepare.py`
  row.

**Measured**

- **Rung C — PASS in 7.8 s** (`myenv/Scripts/python.exe -m validate.shapes`), 15/15 images. Seconds,
  not minutes, as **D-059** asks. The corpus, with the defect each image exists for:

  | image | verdict | thickness before -> after | branch before -> after | repair |
  |---|---|---|---|---|
  | `disc`, `square`, `antialiased`, `huge_margin` | ok | 27->27 / 37->37 | 1->1 | — |
  | `hairline_appendage` | repaired | 27->27 (**blind**) | **29**->1 | thicken |
  | `self_touching` | repaired | 29->29 (**blind**) | **4**->1 | thicken |
  | `specks` | repaired | 1->27 | 1->1 | drop_specks |
  | `two_bodies` | repaired | 15->27 | 1->1 | largest_component |
  | `donut` | repaired | 11->27 (1 hole -> 0) | 1->1 | fill_holes |
  | `unclosed_outline`, `diagonal_line`, `extreme_aspect` | repaired | 1->3 | 0->1 | thicken |
  | `tiny_body` (asked 6 across) | refused | — | — | fix `resolution` -> 8, **runs** |
  | `all_white`, `all_black` | refused | — | — | fix `picture`, runs against `EXAMPLE_MASK` |

  The two rows marked **blind** are the whole of Q-102: `min_thickness` reports 27 and 29 — the
  *body* — while the hairline and the pinched waist are what is actually wrong.
- **Q-102 closed (D-064).** `thin_branch_depth` over discs of radius 3..60 at four sub-cell offsets
  each: worst **2**. Squares 3..60: **1**. Straight 3-cell plates: **1**. Rung 3's cylinder body:
  **2**. Rung 4's square and polygon bodies: **1** and **1**. A 1- or 2-cell hairline of length `L`
  fused to a disc: **`L+1`** (3 at L=2, 6 at 5, 11 at 10, 21 at 20). Threshold 3 separates them.
  Cost recorded rather than hidden: it also fires on a rotated triangle's apex (**4**), a 40x3
  ellipse's tip (**5**) and a stroke-3 ring of radius 60 (**6**) — all genuinely one cell thick, so
  true positives under constraint 12's own rule.
- **Every rung at or below this task, re-run, all on an idle machine:**

  | Rung | Command | Result |
  |---|---|---|
  | R1 | `validate.poiseuille` | **PASS** — L2 **0.3650%**, peak lattice velocity 0.07955 |
  | R2 | `validate.cavity --re 100` | **PASS** — max dev vs Ghia **0.75%**, vortex **0.21 cells** |
  | R3 | `validate.cylinder --backend warp --headless` | **PASS** — St **0.1731**, Cd **1.4031 +- 0.0086**, peak 0.09685 |
  | R4 | `validate.polygons --backend warp --headless` | **PASS** — square Cd **1.5279 +- 0.0271**; polygon Cd **1.4276 +- 0.0226**, Cl amplitude 0.3689, peak 0.08944 |
  | A | `validate.parity --backend warp` | **PASS** — worst 5.960e-08, whole step **9.611e-06**, checkpoint **8.196e-06** |
  | **C** | `validate.shapes` | **PASS in 7.8 s** — 15/15, all four repairs exercised by the corpus |
  | D | `validate.refusals` | **PASS** — caught / `nan` at **1525/1650**, **75/325**, **50/59275**; cost 0.01% |
  | B | `validate.autoconfig` (numpy) | **PASS** — predicted 61.05 s, actual **56.46 s**, **8.1%**; 24/24; worst peak 0.0695; worst Re error 0.0000% |
  | B | `validate.autoconfig --backend warp` | **FAIL** — accuracy check only: predicted 5.61 s, actual **3.19 s**, **75.7%**. Sweep 24/24, identical digits |

  Every Phase 0 digit is session 11/15/18's exactly.

**A measurement this session got wrong first, and the correction**

Rung B's numpy run was first measured at **92.12 s actual against the same 61.05 s prediction —
33.7%, a FAIL** — and the warp run at 48.3%. Both were taken while **this session's own orphaned
processes were still running**: several `validate` runs survived their background tasks being
killed, and one (PID 16358) was alive through most of Rung B's window. Re-run on an idle machine,
numpy came in at **56.46 s / 8.1% — PASS**, and warp got *worse*, **3.19 s / 75.7%**. The first
warp issue queued this session was **dropped** for that reason and re-filed with the clean numbers
(`e4874a146490`). **D-035** is why this was caught rather than published: it requires the machine
state beside every timing, and a rung whose prediction is byte-identical across three sessions while
the measurement swings 53 -> 92 s is reporting the machine, not the model.

**Decisions made**

- **D-064** — Q-102 closed by measurement: the core/shell geodesic metric, the threshold, the disc
  false-alarm bar it clears, and the tapered-shape hits recorded as measured cost.
- **D-065** — a body size the picture cannot deliver is a **refusal** naming a reachable size
  (iterated to a fixed point), not a quieter resolution. Found by a test: a two-pixel plate asked
  for 40 cells across was silently returning **61**.
- **D-066** — `flow/prepare.py`'s constants and repair order, including the bounding-box cropping
  that is the difference between Rung C at 7.8 s and Rung C at over two minutes.

**Not done / deferred**

- **Nothing in the T107 contract is outstanding.** All eight acceptance criteria are checked in
  `DOCS/TASKS2.md` § T107.
- **Rung B on `--backend warp`** — see § Blockers. Pre-existing, queued `e4874a146490`, owned by
  **T110** because M8's gate is a wall clock on warp. T108 does not time anything.
- **A feature too small for the requested raster is dropped silently by thresholding**, not by
  repair — `from_png` area-averages and thresholds at 0.5 (**D-040**), so a 1-pixel whisker at
  5 px/cell simply is not there, and `prepare` returns `verdict="ok"` because the mask it was handed
  is clean. `prepare` refuses when the *body size* is unreachable (**D-065**) but says nothing about
  a *feature* lost to resolution. Not in the T107 contract; worth a `/new-task` if T108's report is
  meant to speak to it.
- **`Monitor` on warp** is still unmeasured (session 18's deferral, unchanged).

**Blockers:** none for T108. One red rung, recorded in § Blockers with what would unblock it.

**Housekeeping**

- `DOCS/ISSUES.jsonl` — `d58a860250c5` queued and then **dropped** (contaminated measurement);
  `e4874a146490` queued in its place with the idle-machine numbers. The `.gitignore` entry from
  session 16 (`495777c58269`) is still open and still not this task's to fix; it was checked against
  the new files and does **not** catch them — `git check-ignore` is clean on all 15 PNGs, the
  generator, `expectations.json`, `flow/prepare.py`, `validate/shapes.py` and
  `tests/test_prepare.py`.

**Rung status after this session**

- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run, session 11/15/18's digits.
- Phase 1: **A 🟩** · **B 🟩 numpy / 🟥 warp (accuracy check only)** · **C 🟩** · **D 🟩** · E ⬜.
  **M7 reached.**

**Next:** **T108 — `flow.Case` / `flow.Result` API**, session 20, gate unit tests. It is the front
door over everything T104–T107 built, and it carries **D-062**'s forward-referenced criterion: the
`substituted=True` flag must reach `Result`, the printed summary and the recorded video's metadata.
Prompt written to `PROMPTS/020-t108-case-result.md`.

### 2026-08-27 — Session 20: T108, the `Case` / `Result` front door

**Task:** T108 — `flow.Case` / `flow.Result` API. **Status: done.** Every acceptance criterion was
run, not read. `pytest` **721 passed, 1 skipped** in 63.9 s (663 -> 721; 58 new: 29 in
`tests/test_case.py`, 25 in `tests/test_report.py`, 4 auto-parametrised in
`tests/test_flow_package.py` for the two new modules). `lbm/` untouched — `git status -- lbm` is
empty — and every rung at or below this task was re-run anyway.

**Done**

- **`flow/case.py`** — `Case`, a facade and nothing more. `Case.from_image(path, *, fluid, speed,
  size, quality="balanced", repair=True, backend="numpy")` and `Case.from_array(mask, ...)`;
  `case.plan` (the `Plan`, or `None`), `case.prepared` (the `Prepared`), `case.runnable`,
  `case.refusal`, `case.suggestions`, `case.fix`, `case.nearest()`, `case.explain()`, `case.run()`,
  and `case.sim` (the `Sim` of the most recent run — the escape hatch T110 will want). Building runs
  `prepare` and `plan` and **no timesteps**.
- **`flow/report.py`** — `Result`, a frozen output record (**D-060**'s exemption applies to its
  generated constructor and to nothing else): `cd` / `cd_std` / `cd_amplitude`, `cl` / `cl_mean`,
  `strouhal` / `strouhal_confidence` / `periods`, `convergence` (+ its history), `peak_u`,
  `elapsed`, `substituted` / `substitution`, `frames`, `backend`, `steps`, `stable`, the `Cd`/`Cl`
  histories, the `Plan` and the `Prepared`. Plus `summary()` (prints and returns), `as_dict()`,
  `save(path)`, `plot(path=None)`, `metadata()`, and the shared `metadata_entries()` builder.
- **The three `DOCS/IDEA3.md` lines run end to end.** Measured, water at Re 99.6 through
  `disc.png`, `quality="fast"`, 78000 steps = 130 convective times, numpy, idle machine:
  **`Cd` 1.5955 +- 0.0157, `Cl` amplitude 0.416, `St` 0.1841** (peak 44x the next distinct one, 12.0
  periods), peak `|u|` **0.09761**, **478.2 s**. That is a working product path — and its `Cd`/`St`
  sit *outside* Rung 3's bands for a reason that is the chooser's domain and not this task's:
  **Q-104**, § Blockers, queued `a924f78acc32`.
- **Assembly ported from `lbm/runner.py`'s CLI**, which is the behaviour that already works (M4):
  the body placed at `UPSTREAM_D` diameters with one cell of asymmetry, the solid seeded at rest
  (**D-030**), the startup kick at 0.20 for 5 convective times, `steps_per_frame` taken **from the
  plan** and never computed here (constraint 7 / **D-023** — asserted by reading the source).
- **`Result.save()` writes both sinks** — `wake.mp4` through `RecordSink`, `frames/` through
  `HeadlessSink` — and a video written *either* way (by `save()` afterwards or by `run(record=...)`
  as it goes) carries the same provenance comment from one builder, into the container's comment
  atom. `flow/` colours nothing: the only name it imports from `lbm.render` is `render` (and
  `LiveSink`), asserted by an AST scan.
- **`flow/__init__.py`** re-exports `Case` and `Result`; `CLAUDE.md` § Module map gained the two
  rows.

**Measured**

- **Every rung at or below this task, re-run:**

  | Rung | Command | Result |
  |---|---|---|
  | R1 | `validate.poiseuille` | **PASS** — L2 **0.3650%**, peak lattice velocity 0.07955 |
  | R2 | `validate.cavity --re 100` | **PASS** — max dev vs Ghia **0.75%**, vortex **0.21 cells** |
  | R3 | `validate.cylinder --backend warp --headless` | **PASS** — St **0.1731**, Cd **1.4031 +- 0.0086**, peak 0.09685 |
  | R4 | `validate.polygons --backend warp --headless` | **PASS** — polygon Cd **1.4276 +- 0.0226**, Cl amplitude 0.3689, peak 0.08944 |
  | A | `validate.parity --backend warp` | **PASS** — whole step **9.611e-06**, checkpoint **8.196e-06** |
  | C | `validate.shapes` | **PASS in 6.6 s** — 15/15 |
  | D | `validate.refusals` | **PASS** — caught / `nan` at 1525/1650, 75/325, 50/59275; Monitor cost **-0.53%** |

  Every digit is session 11/15/18/19's exactly. **Rung B was not re-run** — it costs ~23 min, T108
  times nothing, and its warp accuracy failure is recorded in § Blockers as session 19 left it.
  **Disclosure (D-035):** `pytest` overlapped the tail of R3/R4, so R3's 294 steps/s is not a clean
  timing; every pass condition there is a physics band, and Rung D's cost check ran clean and in the
  safe direction (a negative cost cannot be manufactured by contention).
- **Re 10 is steady, and settles slowly.** 30000 steps, lift amplitude as a fraction of `Cd` per
  3000-step block: **0.890, 0.151, 0.041, 0.0123, 0.0043, 0.0018, 0.0009, 0.0005, 0.00012,
  0.00004** — through the 1% gate at ~12000 steps. This is the measurement behind **D-069**, and the
  reason the suite's Re 10 test asserts the cheap half (no `St`, a run that says it measured
  nothing, a visibly decaying history) rather than paying ~110 s for a post-settling window.
- **The one-period trap, measured**: a synthetic sine planted at `St = 0.17` with one period in the
  window returns **0.459** with a peak 56x the next distinct one (**D-070**).

**Not done / deferred**

- **Nothing in the T108 contract is outstanding.** All nine acceptance criteria are checked in
  `DOCS/TASKS2.md` § T108, with five deviations recorded there and carrying D-ids.
- **`Case` has no checkpoint/resume surface.** `Sim` has one (**D-022**, **D-050**) and `case.sim`
  reaches it, but nothing in `flow/` wraps it. Not in the contract; T109 or a `/new-task` if the CLI
  wants `--checkpoint`.
- **`live=True` is composed and mode-tested but never opened in a test** — a window in a test suite
  is a hang waiting to happen, so `LiveSink` is stubbed and the *rule* (**D-039**) is what is
  asserted. T109's manual gate is where a real window gets opened.
- **Rung B on `--backend warp`** — unchanged, § Blockers, **T110's**.
- **`Monitor` on warp** is still unmeasured (session 18's deferral, unchanged).

**Decisions made**

- **D-067** — a refused `Case` is carried, not raised at construction; `run()` raises it, and a
  picture refusal is raised as the same `Unrepresentable` a physics refusal is (**D-065**'s path).
- **D-068** — a geometry `Fix` is translated into a quality level, never into `cells_across`
  (constraint 13); finest that fits, and the worked example when nothing fits.
- **D-069** — the measurement window starts after the kick has washed out; a run too short for that
  measures nothing and says so.
- **D-070** — the three Strouhal gates, and why the period count is taken against the slowest
  plausible period rather than against the estimate's own.
- **D-071** — the frame-memory budget, reported rather than silently capped, and the 1x1 placeholder
  that keeps the ring buffer's `None` sentinel meaning what it says.

**Blockers:** none for T109. Two things are recorded in § Blockers, both **T110's**: Rung B's warp
estimator, and **Q-104** — the product path's domain versus Rung 3's bands, which is M8's gate.

**Housekeeping**

- `DOCS/ISSUES.jsonl` — `a924f78acc32` queued (the domain-versus-bands finding). The `.gitignore`
  entry from session 16 (`495777c58269`) is still open and still not this task's to fix; it was
  checked against the new files and does **not** catch them — `git check-ignore` is clean on
  `flow/case.py`, `flow/report.py`, `tests/test_case.py` and `tests/test_report.py`.
- The suite's wall clock roughly doubled, 32 s -> 63.9 s, and that is deliberate: four of T108's
  criteria (**D-030** after 300 steps, the tee writing two files, the Re 10 case, the substituted
  run recording an MP4) cannot be asserted without running the solver. The Re 10 test is the
  expensive one at ~27 s.

**Rung status after this session**

- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run, session 11/15's digits.
- Phase 1: **A 🟩** · **B 🟩 numpy / 🟥 warp (accuracy check only, not re-run this session)** ·
  **C 🟩** · **D 🟩** · E ⬜. **M7 stands; M8 is T110's.**

**Next:** **T109 — CLI on `flow`, live + record wiring**, session 21, gate manual + tests.
`python -m flow` replaces `python -m lbm.runner` as the thing a person runs, built on `Case`. It
also decides **Q-101** (does `python -m lbm.runner` survive as an entry point, or become a pointer).
Prompt written to `PROMPTS/021-t109-cli.md`.

### 2026-08-27 — Session 21: T109, `python -m flow` — the CLI on `Case`

**Task:** T109 — CLI on `flow`, live + record wiring. **Status: done.** Every acceptance criterion
was run, not read — including the cold-shell command and both exit codes. `pytest` **772 passed,
1 skipped** in 87.5 s (721 -> 772; 51 new: 47 in `tests/test_cli.py`, 4 auto-parametrised in
`tests/test_flow_package.py` for `flow.cli` and `flow.__main__`). The only change under `lbm/` is
**+22 lines in `lbm/runner.py`** — a constant, a `--help` line and one `print` — and the diff was
read to confirm it before any rung was blamed on it.

**Done**

- **`flow/cli.py`** — `build_parser()` and `main(argv) -> int`, and nothing else. Flags:
  `--shape` / `--fluid` / `--speed` / `--size` (all required), `--quality`, `--seconds`,
  `--no-repair`, `--out` / `--frames-dir` / `--live` / `--no-live` / `--record` / `--headless`,
  `--backend`, `--explain` (`--dry-run`), `--nearest`, `--quiet`. Exit codes **0** ran or
  explained, **1** started and did not survive (`nan`, or `Monitor` tripped), **2** nothing ran.
- **`flow/__main__.py`** — the `python -m flow` entry point, with a `__name__` guard that is
  load-bearing: `tests/test_flow_package.py` imports every module under `flow.` to scan it for
  constraint 13, so without it collecting the suite would run the CLI.
- **Every decision is relayed, not made here.** `Case` prepares, plans, refuses and composes the
  sinks; `Result` prints the numbers. The CLI does not re-derive `drop` — **D-039**'s rule stays in
  `flow.case._resolve_sinks`, and the eight-combination test reads `drop` back out of it rather
  than out of a second copy.
- **`lbm/runner.py` keeps its CLI** and gained `PHASE1_CLI_POINTER` (**D-072**, closing **Q-101**).
- **`CLAUDE.md`** § Commands gained the two `python -m flow` lines and § Module map the
  `flow/cli.py` row; `flow/__init__.py`'s docstring names the CLI and says why it is deliberately
  **not** re-exported.

**Measured**

- **Every rung at or below this task, re-run on an idle machine:**

  | Rung | Command | Result |
  |---|---|---|
  | R1 | `validate.poiseuille` | **PASS** — L2 **0.3650%**, peak 0.07955 |
  | R2 | `validate.cavity --re 100` | **PASS** — max dev vs Ghia **0.75%**, vortex **0.21 cells** |
  | R3 | `validate.cylinder --backend warp --headless` | **PASS** — St **0.1731**, Cd **1.4031**, peak 0.09685 |
  | R4 | `validate.polygons --backend warp --headless` | **PASS** — polygon Cd **1.4276**, Cl amplitude 0.3689, peak 0.08944 |
  | A | `validate.parity --backend warp` | **PASS** — whole step **9.611e-06**, checkpoint **8.196e-06**, restart bit-identical |
  | C | `validate.shapes` | **PASS in 6.7 s** — 15/15 |
  | D | `validate.refusals` | **PASS** — 1525/1650, 75/325, 50/59275; Monitor cost **0.57%** |

  Every digit is session 11/15/18/19's exactly. **Rung B was not re-run** — ~23 min, T109 times
  nothing, and its warp failure is § Blockers as session 19 left it.
- **Rung D failed once and passed on re-measurement, and both numbers are recorded.** As the third
  rung of a chained `R4 -> A -> D` run its `Monitor` cost check read **2.57%** against a 2% limit
  (bare 253.7 / watched 247.2); re-run **alone** on a confirmed-idle machine twenty minutes later
  it read **0.57%** (bare 250.1 / watched 248.6). All three detections were identical in both runs.
  The tolerance was **not** touched. The finding is that the check's spread across sessions
  (**-2.10% ... +2.57%**, D-061's four readings plus 0.01% in session 19, -0.53% in session 20 and
  both of this session's) is **wider than its own ±2% gate**, so it fails intermittently on noise —
  queued as `71a74d08789c`. Note the direction: the failing run's *bare* figure was the faster of
  the pair, so the machine was not slow, the ratio was noisy.
- **The M4 gate command reproduces to the digit** (**D-072**): 301 frames, 40033 steps, peak |u|
  **0.06554**, 0.32 MB, verified from the file as h264 528x368 at 60 fps. Only the wall clock moved
  (304.5 s vs 334.9 s), which is not a published band.
- **The product path through the CLI, un-drawn, on warp:** 75000 steps in **50.7 s**, `Cd`
  **1.5955 ± 0.0157**, `St` **0.1838** (peak 196.3x, 11.5 periods), peak |u| 0.09761 — session 20's
  numpy digits (1.5955 ± 0.0157, 0.1841) reached through `python -m flow`. **This is evidence for
  T110, not a verdict on M8:** the run is at the chooser's present 12 D span, which is exactly what
  **Q-104** asks about, and widening it roughly doubles the cell count and the wall clock with it.
- **A 45000-step live-window run reported `St = None` and was right to.** Cl amplitude was 27.7% of
  Cd so gate 1 passed; **D-070**'s gate 2 wants 2 periods of the slowest plausible shedding
  (`D / (U * 0.05)` = 12000 steps, so 24000) and the window held 22500. The 75000-step run above
  recovers `St`. Recorded so a future session does not read that `None` as a regression.
- **Constraint 16 survives into the container:** the substituted run's MP4 comment atom reads
  `substituted=True; fluid -> honey: ...`, read back with `ffmpeg -i`, not from a counter.

**Two findings, queued rather than fixed — both are T108 code and outside this contract**

- **`2fd69b874c32` — `Case.explain()` prints a different suggestion list than `Case.nearest()`
  acts on.** `explain` renders the `Unrepresentable`'s own suggestions; `nearest` takes the top of
  `Case.suggestions`, which is `flow.diagnose.suggest`'s list and which **D-063** prepends a
  `"fluid"` option to. Measured on the criterion's own case: `explain` shows *speed, size* while
  `nearest` runs *fluid -> honey*, so `flow/case.py:548`'s "applies the first of these" is false
  whenever a fluid substitution exists. The two agree on **D-038**'s air/20/1.5 case (no fluid is
  thick enough at Re 2e6), which is why T108's tests did not catch it. **The CLI prints
  `case.suggestions` — the list it will actually execute — so its own output is honest**, and
  `test_the_suggestions_the_cli_prints_are_the_ones_nearest_would_run` pins the divergence.
- **`71a74d08789c`** — Rung D's cost-check spread, above.

**Not done / deferred**

- **Nothing in the T109 contract is outstanding.** All seven acceptance criteria are checked in
  `DOCS/TASKS2.md` § T109, with **D-074** recording that criterion 1's *literal* command is a
  refusal and is met through `--nearest`.
- **No `--checkpoint` / `--resume` on `python -m flow`.** `Sim` has one (**D-022**, **D-050**) and
  `case.sim` reaches it; nothing in `flow/` wraps it. Session 20 flagged this as "T109 or a
  `/new-task`"; T109 did not take it, because it is not in the contract and a checkpoint flag on a
  CLI whose run length is a physical duration needs a decision about what resuming *means* first.
- **Rung B on `--backend warp`** — unchanged, § Blockers, **T110's**.
- **`Monitor` on warp** is still unmeasured (session 18's deferral, unchanged).

**Decisions made**

- **D-072** — `python -m lbm.runner` survives, working, plus a pointer line. **Closes Q-101.**
  Constraint 15 removed the "delegating" option before judgement got a say.
- **D-073** — `--live` is three-valued; `--no-live` with no file sink is the only route to
  **D-071**'s un-drawn run; and `--quiet` prints the summary rather than swallowing it.
- **D-074** — the T109 contract's own example command is refused, as T011's was under **D-038**,
  and is met through `--nearest` with the substitution marked in every artifact.

**Blockers:** none stopping T110 from starting, but **both § Blockers entries are now T110's to
answer** — Rung B's warp estimator, and **Q-104**, the product path's domain versus Rung 3's bands.
M8 gates on both.

**Housekeeping**

- `DOCS/ISSUES.jsonl` — `2fd69b874c32` and `71a74d08789c` queued. The `.gitignore` entry from
  session 16 (`495777c58269`) is still open and still not this task's to fix; checked against the
  new files and it does **not** catch them — `git check-ignore` is clean on `flow/cli.py`,
  `flow/__main__.py` and `tests/test_cli.py`, `flow/__main__.py` being the one worth checking
  because `.gitignore` drops `*/__init__.py`.

**Rung status after this session**

- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run, session 11/15's digits.
- Phase 1: **A 🟩** · **B 🟩 numpy / 🟥 warp (accuracy check only, not re-run this session)** ·
  **C 🟩** · **D 🟩** · E ⬜. **M7 stands; M8 is T110's.**

**Next:** **T110 — The minute: end to end, timed**, session 22, gate **Rung E** → **M8**. It is the
last task of Phase 1 and it owns both blockers. Prompt written to `PROMPTS/022-t110-the-minute.md`.

### 2026-08-27 — Session 22: T110, the minute — **Rung E**, **M8**, and the end of Phase 1

**Task:** T110 — The minute: end to end, timed. **Status: done. M8 reached. Phase 1 is complete.**
Every acceptance criterion was run, not read — including the timed gate on both backends and the
README quickstart pasted into a fresh venv built by its own `pip install` line. `lbm/` untouched:
`git status -- lbm` is empty. Five decisions, **D-075** … **D-079**, and both of session 21's
blockers are closed by measurement.

**The gate, first, because it is what the phase is judged on**

```
myenv/Scripts/python.exe -m validate.minute --backend warp

1. physics - Rung 3's published bands, unwidened
   grid 720x540 = 389k cells, blockage 4.17%, Re 99.6, tau 0.5452, 50400 steps
   Cd  1.4040 +- 0.0173   band 1.25..1.45   [ok]
   St  0.1672   band 0.155..0.175   [ok]
   peak |u| 0.09725 of the 0.1 ceiling (constraint 3)

2. the minute - process start to Result.summary()
   49.5 s wall clock (48.0 s of it the run itself), limit 60 s   [ok]

Rung E: PASS
```

Conditions (**D-035**): AMD Ryzen 7 5800H at **3201 MHz of 3201 MHz**, on **mains**; NVIDIA GeForce
RTX 3050 Laptop GPU, driver **592.82**, CUDA Toolkit 12.9 / Driver 13.1. The wall clock is taken
from `psutil`'s own `create_time`, so the interpreter, every import and the Warp context are inside
the number — it is the shell's minute, not the solver's.

**Done**

- **`validate/minute.py`** — Rung E. A committed PNG (`tests/data/shapes/disc.png`, already Rung C's
  corpus disc, so no new binary), `fluid="water"`, `speed="5 mm/s"`, `size="2 cm"`,
  `quality="fast"` — no lattice quantity anywhere in the invocation (constraint 13). The two bands
  are **imported** from `validate/cylinder.py` rather than copied, so this rung cannot drift from
  Rung 3's published numbers even by a typo. The run length is **derived** from **D-070**'s gate 2
  rather than chosen (`RUN_MARGIN * MIN_PERIODS / (ST_PLAUSIBLE[0] * (1 - TRANSIENT_FRACTION))` = 84
  convective times), so it follows `flow/report.py` if those constants move. `machine_state()` is
  collected *after* the run: it shells out to WMI three times and to `nvidia-smi`, which is seconds
  a user never spends, and the clock starts at process start.
- **`README.md`** — a quickstart at the top: clone, venv, one `pip install` line, and the product
  command. Plus `--explain`, `--out wake.mp4`, the refusal path, and `python -m validate.minute`.
  § Current state and § Roadmap were rewritten because they still said the LBM engine did not exist,
  and the ladder table now carries all nine rungs with their measured numbers.
- **`CLAUDE.md`** — § Commands gained the Rung E line, § Module map the `validate/minute.py` row, and
  § Current state was rewritten for a finished Phase 1.
- **Five decisions**, each measured in this session: **D-075** the domain, **D-076** the probe
  cadence, **D-077** the rate-table anchors, **D-078** Rung D's cost sampling, **D-079** the default
  run length.

**Q-104, answered by measuring one case with one thing changing at a time**

The product path's `Cd` was 1.5955 against a band of 1.25–1.45, and the question was whether to
widen the chooser's domain or restate the bands. Measured on Rung E's own case (`warp`, 48000 steps,
`quality="fast"`):

| span / upstream | cells | blockage | `Cd` | `St` | wall clock |
|---|---|---|---|---|---|
| 12 D / 3 D | 140k | 8.33% | **1.5943** | 0.1835 | 32.0 s |
| 16 D / 6 D | 230k | 6.25% | **1.4523** | 0.1729 | 46.8 s |
| 20 D / 6 D | 288k | 5.00% | 1.4426 | 0.1713 | 58.6 s |
| 22 D / 6 D | 317k | 4.55% | 1.4394 | 0.1705 | 69.9 s |
| 24 D / 6 D | 346k | 4.17% | 1.4360 | 0.1696 | 72.1 s |
| **24 D / 8 D** | **389k** | **4.17%** | **1.4030** | **0.1676** | 76.9 s |

Rung 3 itself prints **1.4031** and 0.1731 on this machine on the same day, so the last row is the
benchmark's own drag coefficient to four decimals. Two things only the table could say: the
**upstream fetch is not second-order** — at a fixed 24 D span it is the whole remaining distance to
Rung 3's digit — and the cheapest in-band option (20 D / 6 D) was **rejected rather than taken**,
because a `Cd` 0.5% under the band's top with a `cd_std` of 0.0177 is a pass a future session would
have to re-argue. Restating the bands for a confined case was rejected on the same evidence: the
confined number is not a different-but-valid answer, it is 14% of drag contributed by the walls.

**And the widening had to be paid for, twice, in ways that were only found by running things**

- **The minute.** 24 D / 8 D at the old probe cadence is 76.9 s — over. The lever was
  **D-076**: one force sample is five host reads (~37 MB across the bus at this domain), and at 50
  samples per convective time that was 37.6 s of the 76.9. At 10 samples it is 7.5 s, and `Cd` and
  `St` are **identical to four decimals** (1.4030 / 0.1676 either way). The one number that moves is
  peak `|u|`, 0.09761 → 0.09725: a coarser sampler sees a slightly smaller maximum, which is a real
  if small loss of pessimism in a constraint-3 check.
- **Rung D went noisy.** The `Monitor` cost check read **5.86%** against its 2% limit on the bigger
  domain — and `Monitor`'s arithmetic is 0.45 ms against a 13.2 ms timestep at a 25-step cadence,
  which is **0.14%**, so the measurement was reporting the machine. Three repeats of the old
  sampling: +0.79%, +5.82%, −4.32%. Three of the new: +0.49%, +1.17%, −1.25% (**D-078**). Same
  protocol, same total timesteps, same tolerance — more alternation, which is D-035's own answer to
  drift. Re-run: **−0.69%**.
- **The default run stopped measuring anything.** The README quickstart, pasted into a fresh shell,
  printed `Cd` **`nan`** and *"the run ended after 16000 steps, before the startup kick had ...
  washed out (18400)"*. D-059's 20 convective times had been silently coupled to the old domain's
  shorter flow-through. **D-079** raises it to 80 — the larger of D-069's floor (46) and D-070's
  gate (80) — and writes it as arithmetic rather than as a number.

**Rung B on warp, closed**

`Plan.estimated_seconds` was interpolating log-log between two *bandwidth-bound* anchors (40k and
1M) straight through the region where this card is *kernel-launch*-bound. Measured this session with
`bench.compare_backends`'s own protocol at two round grids near the product's sizes:

| Grid | Cells | NumPy | Warp | µs/step (warp) |
|---|---|---|---|---|
| 800×200 | 160k | **185.6** | **3560.4** | 280.9 |
| 800×500 | 400k | **76.5** | **1403.9** | 712.3 |

The old model predicted 1996 steps/s at 160k where the card does 3560.4 — a wall clock **1.78x** the
real one, which is session 19's 75.7% almost exactly. The warp curve is nearly flat from 40k to 160k
(log-log slope **−0.11**) and turns over at **−1.02** beyond it. A least-squares `t = a + b·cells`
was tried and rejected: it is +34.1% at 160k, worse than the tolerance it would have to satisfy.
Two anchors, no new model, no widened tolerance (**D-077**).

**Measured — every rung, both backends where both apply**

| Rung | Command | Result |
|---|---|---|
| R1 | `validate.poiseuille` | **PASS** — L2 **0.3650%**, peak 0.07955 |
| R1 | `validate.poiseuille --backend warp` | **PASS** — L2 **0.3649%** |
| R2 | `validate.cavity --re 100` | **PASS** — max dev vs Ghia **0.75%**, vortex **0.21 cells** |
| R2 | `validate.cavity --re 100 --backend warp` | **PASS** — **0.75%**, **0.21 cells** |
| R3 | `validate.cylinder --backend numpy` | **PASS** — St **0.1731**, Cd **1.4031**, peak 0.09685 |
| R3 | `validate.cylinder --backend warp` | **PASS** — St **0.1731**, Cd **1.4031 +- 0.0086**, peak 0.09685 |
| R4 | `validate.polygons --backend numpy` | **PASS** — square Cd **1.5279 +- 0.0271**, polygon Cd **1.4276 +- 0.0226**, peak 0.08944 — warp's digits exactly |
| R4 | `validate.polygons --backend warp` | **PASS** — square Cd **1.5279 +- 0.0271**, polygon Cd **1.4276 +- 0.0226**, Cl amplitude 0.3689, peak 0.08944 |
| A | `validate.parity --backend warp` | **PASS** — worst kernel 5.960e-08, whole step **9.611e-06**, checkpoint **8.196e-06**, restart bit-identical |
| B | `validate.autoconfig --backend warp` | **PASS** — predicted **33.22 s**, actual **34.43 s**, **3.5%**; sweep 24/24 |
| B | `validate.autoconfig` (numpy) | **PASS** — predicted **610.45 s**, actual **719.91 s**, **15.2%**; sweep 24/24, worst peak |u| 0.0656, worst Re error 0.0000% |
| C | `validate.shapes` | **PASS in 16.2 s** — 15/15 |
| D | `validate.refusals` | **PASS** — caught/`nan` at 1525/1650, 75/325, 50/59275; Monitor cost **-0.69%** |
| **E** | `validate.minute --backend warp` | **PASS** — Cd **1.4040**, St **0.1672**, **49.5 s** |
| **E** | `validate.minute --backend numpy` | **PASS on physics** — Cd **1.4040**, St **0.1672**, the *same printed digits* as warp; 1454.2 s, not a gate on this backend |

Every Phase 0 digit is session 11/15/18/19/21's exactly, on both backends. **Disclosure (D-035):**
R3/R4 and R1/R2 on `warp`, and R3/R4 on `numpy`, were run while Rung B's 24-case sweep was still
running, so no steps/s figure from them is a clean timing. Every pass condition in those four rungs
is a physics band, and the three checks that *are* timings — Rung B's accuracy check, Rung D's cost
check and Rung E's wall clock — were each run with nothing else on the machine.

**A re-run that failed, and why it is evidence rather than a regression**

The final confirmation run of `validate.minute --backend warp` printed **FAIL at 71.9 s** — because
the laptop had been unplugged: `Win32_Battery.BatteryStatus` 1, `CurrentClockSpeed` **1882 MHz of
3201**. The physics is bit-for-bit the mains run's (`Cd` 1.4040 ± 0.0173, `St` 0.1672, peak `|u|`
0.09725, 50400 steps), so nothing in the code moved; the CPU did. **This is exactly what D-035
exists for** — an absolute timing without the power state beside it is not a measurement — and it is
recorded here rather than quietly re-run, because a future session comparing 49.5 s to 71.9 s should
find the reason next to the number. **M8's gate output is the mains run**: 49.5 s, 3201 of 3201 MHz,
on mains, same 50400 steps, same digits. The suite's own wall clock moved the same way and for the
same reason: 208 s on mains, 333 s on battery, `772 passed, 1 skipped` either way.

**The README quickstart, actually pasted**

A copy of the working tree, a **fresh venv created and populated by the quickstart's own
`pip install` line**, and then the quickstart command:

```
Cd            1.4030 +- 0.0175      St            0.1676 (peak 3.5x, 6.7 periods observed)
peak |u|      0.09725               elapsed       48.8 s for 48000 steps on backend 'warp'
```

50.7 s of total wall clock from the shell. That paste is what found **D-079**: the first attempt,
before it, printed `nan`.

**Decisions made**

- **D-075** — the chooser's domain becomes Rung 3's own (24 D span, 8 D upstream); supersedes
  D-059's domain choice. **Closes Q-104.**
- **D-076** — the force probe drops to 10 samples per convective time, which is how the widened
  domain still fits inside the minute.
- **D-077** — `_RATE_TABLE` gains measured 160k / 400k anchors on both backends. **Closes the Rung B
  warp blocker.**
- **D-078** — Rung D's cost check samples nine rounds of 300 rather than five of 600.
- **D-079** — the default run length rises from 20 to 80 convective times, derived from D-069 and
  D-070 rather than chosen.

**Not done / deferred**

- **Nothing in the T110 contract is outstanding.** All seven acceptance criteria are checked in
  `DOCS/TASKS2.md` § T110.
- **`Monitor` on warp is still unmeasured** — session 18's deferral, carried through five sessions
  and now the oldest open thread in the product layer. Rung D runs on `numpy` by design (the
  failure modes it provokes are cheaper there), so the probe's device-side cost has never been
  timed. Worth a `/new-task` in Phase 2 rather than a silent carry.
- **`2fd69b874c32`** — `Case.explain()` prints a different suggestion list than `Case.nearest()`
  acts on. Still open, still a T108 defect, still not user-facing because `flow/cli.py` prints the
  list it will actually run.
- **The suite is slower**: 108 s → **208 s** on mains. That is D-075 and D-079 charged to every test
  that runs the solver, and it is the honest price of the domain being right.

**Blockers:** none. Phase 1 is closed.

**Housekeeping**

- `DOCS/ISSUES.jsonl` — `a924f78acc32` (Q-104) and `e4874a146490` (Rung B on warp) dropped as
  **fixed**, each with the decision that fixed it named in the reason; `71a74d08789c` (Rung D's cost
  spread) dropped as fixed by **D-078**. The `.gitignore` entry from session 16 (`495777c58269`) is
  still open and still not this task's to fix; `git check-ignore` is clean on `validate/minute.py`
  and on every file changed this session.
- An empty stray directory `d/` in the repo root, created 2026-08-27 by an earlier session, was
  left alone — it is untracked, empty, and not this task's.

**Rung status after this session**

- Phase 0: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — all four re-run **on both backends**.
- Phase 1: **A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩**. **M8 reached. Phase 1 is complete.**

**Next:** **Phase 2 — the XLB swap** (`idea.md`'s Phase 3), which is what the T101 backend seam was
built to make a substitution rather than a rewrite. It needs its own spec, plan, backlog and state
file, planned the way session 12 planned this one. `DOCS/STATE2.md` becomes history the moment that
happens. Prompt written to `PROMPTS/023-phase2-planning.md`, and it asks the next session to decide
*whether* Phase 2 is the XLB swap rather than to inherit it — the current backend is validated and
fast, and `idea.md`'s ordering has been deliberately deviated from once already (**D-043**).
