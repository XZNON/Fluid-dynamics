# Session 22 — T110: the minute, end to end, timed → Rung E, M8

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator in
pure NumPy, spec `DOCS/IDEA2.md`, closed at M4 with all four validation rungs green. **Phase 1 is
live and this is its last task**: the product layer above the solver — a `flow/` package plus a CLI,
on a Warp GPU backend. Spec `DOCS/IDEA3.md`, plan `DOCS/PLAN2.md`, backlog `DOCS/TASKS2.md`, live
status `DOCS/STATE2.md`.

## Read these first, in this order

1. `CLAUDE.md` — the 16 hard constraints (Phase 1 list), session protocol, conventions.
2. `DOCS/STATE2.md` — **in full**: snapshot, the **two blockers, both of which are now yours**,
   open question **Q-104** (yours to decide), decisions D-041 … D-074, and at minimum the session
   19, 20 and 21 log entries.
3. `DOCS/TASKS2.md` § **T110** — the task contract, in full. It is the **last** row of the backlog
   index; nothing depends on you, and you close the phase.
4. `DOCS/IDEA3.md` § **Goal**, § **Validation ladder** (Rung E), § **Performance budget**, and
   `idea.md` § **Definition of success** — Rung E is that sentence minus the drag-and-drop, plus
   the word *correct*.
5. `DOCS/PLAN2.md` § **Milestone gates** (M8's wording is exact and is the bar), § Session map (you
   are session 22), § Risks — in particular the last row: a change `lbm/` genuinely needs is a
   `/new-task` naming the Phase 0 rung it must re-prove.
6. `old-Docs/STATE1.md` **D-029** (the 0.54 bluff-body floor), **D-026** (why `validate/cylinder.py`
   picks the domain it picks), **D-019** (the measured `D`), **D-035** (no timing without the CPU
   clock, power state and GPU beside it) — read the entry each is cited for, not the whole file.
7. **`flow/autoconfig.py`** — specifically `SPAN_D = 12`, `UPSTREAM_D = 3`, `DOWNSTREAM_D = 9`,
   `QUALITY_CELLS`, `TAU_FLOOR` and `estimated_seconds`. **These constants are what Q-104 is
   about.** Their justification is **D-059**; `validate/cylinder.py`'s own `SPAN_D` docstring is
   the counter-evidence.
8. **`flow/case.py`, `flow/report.py`, `flow/cli.py`** — the product path you are timing.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 21: **T109 is done.** `flow/cli.py` and `flow/__main__.py` exist;
  `python -m flow` is the thing a person runs, built on `Case`. `pytest` prints **772 passed,
  1 skipped** in 87.5 s. `tests/test_cli.py` has 47 tests.
- **No milestone was reached in session 21** — T109 has none. **M7** stands (2026-08-23). **M8 is
  yours and it is the last of Phase 1.**
- **Phase 1 rung status: A 🟩 · B 🟩 on numpy / 🟥 on warp · C 🟩 · D 🟩 · E ⬜.**
- **Phase 0 rung status: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩** — all four re-run in session 21, printing
  session 11/15's digits: R1 L2 0.3650%, R2 0.75% / 0.21 cells, R3 St 0.1731 Cd 1.4031, R4 polygon
  Cd 1.4276.
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101 … **T109**.

## Your task this session

**T110 — The minute: end to end, timed.** One task, this session only. Gate: **Rung E** → **M8**.
Phase 1 closes here.

Run this first:

    /start-task T110

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** from a cold shell, a picture and three physical numbers give a **correct** answer in
**under 60 seconds**.

**Inputs:** the finished product path.
**Outputs:** `validate/minute.py` printing PASS/FAIL with the elapsed time; a `README.md` quickstart
section; `DOCS/STATE2.md` recording M8 with the gate output.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] **Rung E — `validate/minute.py --backend warp`:** a committed PNG of a disc, plus fluid/speed/size chosen to give Re 100, driven through `flow.Case` with no lattice quantity anywhere in the invocation, produces **St in 0.155–0.175 and Cd in 1.25–1.45** — Rung 3's published bands, unwidened.
- [ ] **Wall clock under 60 s**, printed, from process start to `Result.summary()`, on the dev machine with GPU name, driver, CPU clock and power state quoted (**D-035**). If it misses, the number is recorded honestly and the shortfall becomes a blocker, not a widened criterion.
- [ ] The same rung run with `--backend numpy` **also passes on physics** (bands only, no time limit), so the product path is proven independent of the port.
- [ ] All five Phase 1 rungs (A–E) re-run in this session and print PASS; all four Phase 0 rungs re-run and print numbers inside their published bands on both backends.
- [ ] `README.md` gains a quickstart that is **copy-pasteable and was actually pasted** into a fresh shell this session, with its output recorded.
- [ ] `myenv/Scripts/python.exe -m pytest` green.
- [ ] `DOCS/STATE2.md` records **M8** with the gate output, and § Snapshot says Phase 1 is complete.

### The two blockers, and they are both yours

**This is the session that has to answer them.** T108 and T109 shipped past both because neither
times anything and neither chooses a domain. M8 gates on exactly those two things.

- **Q-104 — the product path's own domain puts `Cd` and `St` outside Rung 3's bands.** Measured
  twice now, and the second time through the finished CLI: session 20 on numpy got **`Cd` 1.5955 ±
  0.0157, `St` 0.1841**; session 21 through `python -m flow` on warp got **`Cd` 1.5955 ± 0.0157,
  `St` 0.1838**. The bands are **1.25–1.45** and **0.155–0.175**. **The physics is not in
  question** — R3 and R4 print session 11/15's digits on the same machine on the same day — the
  **domain** is: `flow/autoconfig.py` picks `SPAN_D = 12` (8.3% blockage) and `UPSTREAM_D = 3`
  where `validate/cylinder.py` picks **24** (4.2%) and **8** *deliberately*, and cylinder's own
  `SPAN_D` docstring records that 15 D of span already gave `Cd` 1.4635, "just over the top of the
  acceptance band". 12 D is tighter still.
  **As things stand M8's first criterion cannot be met however long the run is**, because the
  domain and not the run length is the reason. Queued `a924f78acc32`.
  **The choice, and note that it is a real trade:** widen `SPAN_D`/`UPSTREAM_D` toward Rung 3's and
  the bands come into reach — at roughly double the cell count, which is charged directly to the
  criterion *below* it. Or state the bands for a confined case and say why, which the contract's
  own Notes come close to forbidding ("Widening the bands is not an option") — read that line
  carefully and decide whether a *restated* band for a *different domain* is the same thing as a
  widened one. **Decide this before writing Rung E, not after it fails.**

- **Rung B fails its accuracy check on `--backend warp`** — `Plan.estimated_seconds` predicts
  **5.61 s** against an actual **3.19 s**, **75.7%** against a 25% limit. The 24-case sweep passes
  on **both** backends with identical digits, and Rung B on numpy is PASS (8.1%). Diagnosis already
  done: the case is 140k cells, between § Performance baseline's measured warp rows at 40k (4155
  steps/s) and 1M (757 steps/s), and a GPU at that size is partly **kernel-launch-bound** rather
  than bandwidth-bound, so a log-log interpolation between two bandwidth-bound points
  over-predicts. **What would unblock it:** a warp calibration point near 100–200k cells in
  § Performance baseline, or an explicit launch-overhead term in `estimated_seconds` for GPU
  backends. Queued `e4874a146490`. It matters to you because M8's gate is a wall clock on warp and
  criterion 4 re-runs Rung B.

### What session 21 measured that is evidence for you

- **The product path through the CLI, drawing nothing, on warp: 75000 steps in 50.7 s**, `Cd`
  1.5955 ± 0.0157, `St` 0.1838 (peak 196.3x the next distinct one, 11.5 periods), peak |u| 0.09761.
  Reached with `python -m flow ... --no-live --quiet` (**D-073** — that is the un-drawn run,
  **D-071**, and it is the only CLI route to it).
  **Read this carefully: it is evidence, not a verdict.** It says the *minute* is reachable **at the
  present 12 D domain** — which is the very domain Q-104 says cannot hit the bands. Widening the
  domain to reach the bands roughly doubles the cell count and spends that headroom. The two
  criteria pull against each other and that is the actual shape of this task.
- `Result.strouhal` is `None` unless **D-070**'s three gates all pass, and gate 2 wants the window
  to hold 2 periods of the *slowest plausible* shedding (`D / (U * 0.05)` steps each). A
  45000-step run reported `None` for exactly that reason and a 75000-step run did not. Size the
  rung's run length against gate 2, not against the period you expect.

### Constraints that bite on this task

- **Constraint 5 — the ladder is the point.** This is the session that proves the phase, not the
  one that finishes the last feature. Every rung, in order, re-run.
- **Constraint 3 — lattice velocity under 0.1.** The product path currently peaks at **0.09761**,
  which clears it with very little room. Anything that raises `u_lattice` to buy wall clock spends
  that margin, and `Result.summary()` prints the peak for this reason.
- **Constraint 13 — no lattice quantity in any public `flow/` signature.** Rung E's own criterion
  says "no lattice quantity anywhere in the invocation", and two scanners auto-parametrise over
  every `flow/` module. If Q-104 is answered by changing `SPAN_D`, that is a **module constant**,
  not a new argument — do not add a domain flag to `flow/` or to the CLI.
- **Constraint 16 — no silent substitution.** If the rung ends up running something other than what
  it claims (a coarser quality, a smaller domain), it says so in the printed output.
- **Constraint 1 — the physics does not change.** A domain is not physics; `tau`, the collision
  operator and the boundaries are. If reaching the bands seems to need a change to any of those,
  stop: that is a `/new-task` naming the Phase 0 rung it must re-prove, per `DOCS/PLAN2.md` § Risks.
- **`DOCS/IDEA3.md` § Deliberately deferred** — XLB, a UI, 3D, STL, packaging. Phase 1 ends at M8.

### Decisions that constrain this session

- **D-059** — every `flow/autoconfig.py` constant cites a Phase 0 decision or was measured in the
  session that added it. If you change `SPAN_D`, the new value needs the same standard, in a new
  decision that says it supersedes D-059's choice rather than editing it.
- **D-069 / D-070** — the measurement window opens only after the startup kick has washed out
  (`max(50% of the run, kick_steps + one flow-through)`), and `St` survives three gates or is
  `None`. Both bear directly on how long Rung E has to run.
- **D-035** — no absolute timing without `Win32_Processor.CurrentClockSpeed`, the power state and
  the GPU name beside it. **And the operational half, learned twice:** run long rungs detached with
  `python -u`, check `ps -W | grep "Fluid Mech/myenv"` before trusting any timing, kill orphans,
  and do not run `pytest` while a rung is timing itself. Session 19 published a Rung B FAIL that
  was its own orphaned processes; session 21 got a Rung D FAIL at 2.57% as the third rung of a
  chained run and 0.57% re-running it alone.
- **D-072** — `python -m lbm.runner` is kept working *and* points at `python -m flow`; the M4 gate
  command still reproduces to the digit. Do not remove it.
- **D-073** — `--live` is three-valued; `--no-live` with no file sink is the un-drawn run, which is
  what a timed rung wants.
- **D-074** — the T109 criterion's own example command is refused (`Re 32982`), as T011's was under
  **D-038**. Expect Rung E's own case to need a fluid/speed/size that actually gives Re 100 —
  water at 0.005 m/s past a 0.02 m body is the one the project already uses.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` was the last
  addition). Anything new is a real decision and needs a row in `DOCS/STATE2.md` § Environment **in
  the same session**.
- Confirm the starting point: `myenv/Scripts/python.exe -m pytest` should print **772 passed,
  1 skipped** (~88 s), `myenv/Scripts/python.exe -m validate.shapes` should print **PASS** (~7 s).
- **Two open queue entries you may touch, and one you should not.** `a924f78acc32` (Q-104) and
  `e4874a146490` (Rung B on warp) are yours to close. `2fd69b874c32` — `Case.explain()` prints a
  different suggestion list than `Case.nearest()` acts on — is a real T108 defect found in session
  21; it is **not** in your contract, and `flow/cli.py` already prints the executed list so no
  user-facing output is currently lying. `71a74d08789c` (Rung D's cost check fails intermittently
  on noise) is likewise not yours, but **know about it before you read a Rung D FAIL**: re-measure
  on an idle machine before believing it.
- **`.gitignore` drops `*/__init__.py` and `tools/`** (queued `495777c58269`, still open). Run
  `git status` and `git check-ignore -v` on everything you add and confirm it is tracked.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no. A change `lbm/` genuinely needs is a `/new-task` naming the Phase 0 rung it must re-prove
(`DOCS/PLAN2.md` § Risks, last row) — never folded into a product task.

**One exception, and it is the point of the session:** Q-104 *is* in your contract by implication,
because M8's first criterion cannot be met without answering it. Answering it is not scope creep.
Answering it by widening Rung 3's bands is not answering it.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it — including the pasted README
   quickstart and the timed gate on both backends.
2. Run the whole suite.
3. Run `/validate` for **every** rung — all four Phase 0 rungs and all five Phase 1 rungs A–E.
   This is the session where "all of them, in order" is the deliverable rather than a precaution.
4. If M8's wall clock misses, record the number honestly and make the shortfall a blocker. Do not
   widen a band, and do not quote a timing without the machine state beside it.
5. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it. This one also records
   **M8** and marks Phase 1 complete.
