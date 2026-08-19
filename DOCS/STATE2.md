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
| **Current task** | `T105` — Auto-configuration (`DOCS/TASKS2.md`) — **Rung B**, **M6** |
| **Task status** | `not_started` |
| **Completed tasks** | Phase 1: **T101**, **T102**, **T103**, **T104**. Phase 0: T001 … T011, all done |
| **Milestone reached** | **M5** (2026-08-18, the whole timestep on the GPU, Rung A green, the budget cleared). Phase 1 targets M6 → M8 |
| **Phase 0 rung status** | R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — the ladder is complete and stays a gate for every Phase 1 task |
| **Phase 1 rung status** | **A 🟩** · B ⬜ · C ⬜ · D ⬜ · E ⬜ — Rung A is green **in full**: kernels and boundaries worst **5.96e-08** against a 1e-6 bar, whole step **9.611e-06** at 1000 steps against 1e-4, a `warp` checkpoint resumed on `numpy` at **8.196e-06**, restart bitwise within a backend, and all four Phase 0 rungs re-run with `--backend warp` inside their published bands |
| **Last updated** | 2026-08-19 — session 16 (**T104 done**: the `flow/` package exists — `quantity.py`, `fluids.py`; constraints 13 and 15 enforced by test rather than aspiration; **D-058**, the fluid-library ordering conflict; `pytest` **547 passed, 1 skipped**) |

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
