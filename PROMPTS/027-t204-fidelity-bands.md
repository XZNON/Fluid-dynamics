# Session 27 — T204: `flow/fidelity.py`, the bands wired through, Rung H, and M10

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer.
**Phase 2 is live** and its spec is `DOCS/IDEA4.md`. The solver is not the product: see `idea.md`
§ Risks — *"The trap"*, which names the standing temptation to keep polishing it because that part is
fun. **This task is not solver work at all** — it is the judgement layer above a solver that is now
validated by eleven rungs, and the temptation this session has to resist is a different one, named
under "The one thing this task must not do" below.

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
T201 landed the closure on NumPy, T202 landed it on Warp, T203 closed Rung G — **and with it M9, the
phase's first milestone. T204 is the bands, Rung H, and M10.**

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (**18**, **13**, **14**, **16** and **5** are the ones
   that govern this session), the session protocol, the conventions, the module map, § Current state.
2. `DOCS/STATE3.md` — **in full.** § Snapshot, § Blockers, § Open questions (**Q-203** is yours;
   Q-201 and Q-202 are now both closed), § Environment, § Performance baseline, § Decisions
   **D-080 … D-092**, the constraint fate table, and all four session-log entries. The session-26
   entry matters most to you, and **D-091** in particular is a warning about how easily a
   threshold-based check can be made to pass without testing anything — which is exactly the risk
   this task's three bands carry.
3. `DOCS/TASKS3.md` § **T204** — the task contract, in full. Also skim § T201, § T202 and § T203 (all
   `done`, every box ticked) so you know what the closure is, how it is switched, and what is already
   known about the eddy viscosity it generates. Then the backlog index.
4. `DOCS/IDEA4.md` § **The five things Phase 2 must get right (1)** — which contains the band table
   verbatim and is what the contract cites — and § **Validation ladder** (Rung H's row).
5. `DOCS/PLAN3.md` § Why this order, § Session map, § **Milestone gates** (M10's gate command is
   literal), and § **Risks** — the row *"A plausible wrong answer at high Re"* is this task's own, and
   its pressure valve is armed for you.
6. The code you are extending: `flow/autoconfig.py` (**D-059**, **D-075**, **D-079**),
   `flow/report.py` (**D-069**, **D-070**, **D-071**), `flow/case.py` (**D-067**, **D-068**),
   `flow/diagnose.py` (**D-045**, **D-061**, **D-063**), and `lbm/probe.py::eddy_viscosity` — the
   field every band is decided from. Then `validate/les.py` and `validate/taylorgreen.py` for the
   shape of a Phase 2 rung and how it takes `--backend`.

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 26 and it worked **T203 — the Taylor–Green harness. It is `done`:
  every acceptance criterion was run and passed, and M9 was claimed on printed output.** What landed:
  `validate/taylorgreen.py` (Rung G) and `tests/test_taylorgreen.py` (24 tests), plus a `--backend
  warp` line in `CLAUDE.md` § Commands. Nothing in `lbm/` or `flow/` was touched.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 · A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 · **F 🟩 · G 🟩**
  — **eleven green, all re-run in session 26 on BOTH backends with no physics digit moved.**
  H ⬜ · I ⬜ · J ⬜.
- **Milestone reached: M9** (2026-09-03). **M10 is yours.**
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110; Phase 2 **T201**, **T202**, **T203**.
- `pytest`: **827 passed, 2 skipped** (224.3 s).
- **Numbers to keep still.** Rung G: at `Cs = 0`, measured `nu` returns `(tau-0.5)/3` to **0.2303%**;
  at `Cs = 0.17` it returns `nu + <nu_t>` to **1.1547%** while bare `nu` misses by **3.0178%**; the
  excess equals the dissipation-weighted `<nu_t>` to **0.9972**; cross-backend `max|du|/u0`
  **1.150e-05**. Rung F: `cs_smag = 0` bitwise on both backends, worst |diff| **0.000e+00**; Rung 3 at
  `Cs = 0.17` printing **Cd 1.4143**, **St 0.1719** on both; `max(nu_t)/nu` **0.1910** on that wake.
  The LES budget: warp **3504.0 / 661.6 / 403.7** steps/s at 40k / 1M / 2M against floors
  **3116 / 568 / 331**.
- **The three calibration points for `max(nu_t)/nu` you will be banding against**, all measured:
  **0.1910** on Rung 3's shedding wake · **9.011e-02** on Rung A's smooth channel · **1.8418%** (as a
  domain mean) on a resolved Taylor–Green. Note the band table's boundaries are **0.1** and **1**, so
  Rung 3's own case sits just above the quantitative boundary on `nu_t` — check that early, because it
  decides whether the closure is on at all for the validated case.

## Your task this session

**T204 — `flow/fidelity.py` → Rung H → M10.** One task, this session only.

Run this first:

    /start-task T204

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** every `Result` carries a band, and the band's claim is machine-checked. This is the task
that makes the closure safe to ship, and it is the most product-defining module of the phase the way
`autoconfig` was of Phase 1.

**In:** a planned case and, after a run, the `nu_t` field.
**Out:** `flow/fidelity.py::Band` (`quantitative` / `qualitative` / `illustrative`),
`::band_for(plan, nu_t_max=None) -> Band`, `::sentence(band) -> str`; `flow.autoconfig.plan` gains the
closure as a planned parameter; `Result` gains `fidelity`; `validate/fidelity.py` (Rung H).

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `band_for` implements `DOCS/IDEA4.md`'s table exactly: **quantitative** iff `Re <= 200` and `max(nu_t)/nu < 0.1`; **qualitative** iff `max(nu_t)/nu < 1`; **illustrative** otherwise. The `Re <= 200` gate cites Williamson (1996) in the docstring.
- [ ] Before a run, `band_for` returns the band the plan **expects**, from `Re` alone; after a run it returns the band the run **earned**, from the measured `nu_t`. When they differ, the earned band wins and `Result.warnings` says so. A plan that expected `quantitative` and earned `qualitative` is a finding, not a footnote.
- [ ] **Constraint 18, machine-checked:** for every case outside the quantitative band, `Result` emits **no unqualified `Cd`** — asserted by inspecting the object and the rendered summary, not by reading the prose. A test tries to obtain a bare `Cd` from an `illustrative` result and fails if it succeeds.
- [ ] `flow/autoconfig.py` turns the closure **on** only when the plan needs it to satisfy the `TAU_FLOOR` (**D-059**), and the plan records `cs_smag` and *why* — printed by `--explain`. A case that fits under BGK is still run under BGK, bitwise as Phase 1 ran it.
- [ ] **Constraint 13 holds:** `cs_smag` never appears in a public `flow/` signature. It is a planned, printed quantity like `tau`, not an input.
- [ ] **Constraint 16 holds:** a run that engaged the closure says so in the printed summary, the report and the video metadata, alongside its band.
- [ ] **D-038's own case runs.** `--fluid air --speed "20 m/s" --size "1.5 m"` completes, reports `illustrative`, prints no `Cd`, and says in the user's units what it is and is not showing. The Phase 1 refusal is superseded for this case and the supersession is recorded as a decision.
- [ ] Refusals that remain refusals still name a working fix (**constraint 14**); `myenv/Scripts/python.exe -m validate.refusals` re-run and prints **PASS**.
- [ ] `myenv/Scripts/python.exe -m validate.fidelity` prints **PASS** over a Re sweep spanning all three bands.
- [ ] `myenv/Scripts/python.exe -m validate.autoconfig` and `-m validate.minute --backend warp` re-run and print **PASS** with their published digits.
- [ ] `pytest` green.

### Constraints that bite on this task

- **Constraint 18** (new in Phase 2, **D-082**) — *no unqualified quantitative claim outside the
  validated band.* **This task is where it stops being a design rule and becomes code.** Rung H
  asserts it by inspecting the `Result` object, not by reading the prose.
- **Constraint 5** — *the validation ladder is non-negotiable and ordered*, and *"a wrong sim that
  looks plausible is the main failure mode of this project."* This task exists **because the closure
  makes that failure mode reachable on purpose.** All eleven green rungs stay a gate; Rung I (T205)
  does not depend on this one, but Rung H must be green before T208 wires a band into the app.
- **Constraint 13** — *no lattice quantity in any public `flow/` signature.* `Cs` is **planned and
  printed**, never a user knob, and the fidelity band is what surfaces instead.
- **Constraint 14** — *every refusal names a fix, and the fix is machine-checked.* Fewer cases refuse
  after this task; the ones that still do are held to exactly the same bar.
- **Constraint 16** — *no silent substitution.* A run that engaged the closure is a substituted run
  and says so in **every** artifact: the printed summary, the report, and the video metadata.
- **Constraint 2** — `nu_t` is derived through `tau_eff` by `lbm.probe.eddy_viscosity` and never
  assigned. The bands read that field; they do not compute a viscosity.
- **Constraint 19** — the closure still defaults **off**, and a case that fits under BGK must still
  run bitwise as Phase 1 ran it. Rung F is what proves that and it must stay green.
- **Coding conventions** — type hints with intent, shapes in docstrings, `float32`, docstrings cite
  `DOCS/IDEA4.md`, and `flow/` may import `lbm/` but never the reverse (constraint 15, asserted by a
  test).

### The one thing this task must not do

`DOCS/PLAN3.md` § Risks, this task's own row, and `DOCS/TASKS3.md` § T204 Notes both say it:
**widening a band to make a number reportable is out of bounds.** The pressure valve is armed and it
is the correct fallback — *if the bands cannot be made falsifiable, the closure ships
**stability-only**: it stabilises the run so the user sees a picture, and the tool declines to report
`Cd` at all outside the quantitative band.* That is a worse product and an honest one. Choosing it is
a decision in `DOCS/STATE3.md` § Decisions with its reasoning, not a failure.

`sentence(band)` is prose, and prose is **not** what the rung tests (**D-047**'s posture). What is
tested is the verdict and the **absence of the number**.

### Blockers, open questions and decisions that affect you

**Blockers: none.**

- **Q-203 (open — this task answers it)** — *"Can the fidelity bands be made falsifiable enough to
  report a qualified `Cd` outside the quantitative band, or does the closure ship **stability-only**
  (a picture, and no `Cd` at all outside the validated band)? This is the phase's central product
  question."* **Answer it with a decision, and record which way you went and why.**
- **D-082** — the bands themselves, and the reasoning behind the two boundaries. `Re <= 200` is the
  physics gate and it is **cited, not chosen** (Williamson 1996, the mode-A instability at Re ≈ 190);
  `max(nu_t)/nu < 1` is the outer boundary because it is *the point where the model supplies more
  viscosity than the fluid does*, which is a statement about **this run** that a test can evaluate.
- **D-091 (session 26) — read this one as a warning, not as background.** Rung G's `Cs = 0.17` clause
  would have passed *with the closure term deleted* at any comfortably resolved operating point,
  because `<nu_t>/nu` was 0.14% against a 2% bar. The fix was to size the case and add a
  discriminator clause that fails if the term is removed. **Rung H carries the identical risk in a
  worse form**: a band assertion that no case in the sweep can violate is a green rung that proves
  nothing. Make the sweep straddle both boundaries, and make sure at least one case *would* land in
  the wrong band if a comparison were inverted.
- **D-092 (session 26)** — a wall-clock A/B that fails under load is re-run on an idled machine and
  **both** readings are recorded. Rung D's `Monitor` cost has now read −0.69%, +1.02%, +0.17%, +2.11%
  and −0.55% across five sessions against a **2%** limit. You re-run Rung D as an acceptance
  criterion, so expect to need this.
- **D-038 / D-074** — the refused case this whole phase exists for (air, 20 m/s, 1.5 m body,
  Re ≈ 2e6). It currently refuses, correctly. After this task it **runs and reports `illustrative`**,
  and superseding that refusal is itself a recorded decision.
- **D-059 / D-075 / D-079** — the chooser's `TAU_FLOOR`, its domain (24 D span, 8 D upstream) and the
  80-convective-time default run length. The closure comes on **only** where `TAU_FLOOR` would
  otherwise refuse.
- **D-085 / D-086 / D-088** — the closure's normalisation, and that `cs_smag = 0` is an explicit
  branch on both backends into Phase 1's own arithmetic. Do not re-derive the normalisation.
- **`lbm.probe.eddy_viscosity` has no device-side implementation.** Rung F and Rung G both read it
  host-side off `Sim.host_f()`, which is what your banding will also do — at **frame or report
  cadence, never step cadence** (constraint 8). Nothing needs it on the GPU and nothing should be
  written speculatively.
- One issue queued since session 25 and not a blocker: **`022ac461c920`** — `Sim` allocates
  `smag_work (4, ny, nx)` on the Warp backend, which never reads it (32 MiB of dead device memory at
  2M cells with the closure on). Fixing it crosses the T101 seam, so it is a decision, not a
  tidy-up. **T204 or T208 is a natural place**; taking it is optional and is a `/new-task` if you do.
- Two older queued issues, neither a blocker: **`2fd69b874c32`** (`Case.explain()` prints a different
  suggestion list than `Case.nearest()` acts on — folded into T207) and **`495777c58269`**
  (`.gitignore` drops `*/__init__.py` and `tools/` — will bite T205).

### Three environment facts sessions 24–26 paid for, so you do not have to

- **A foreground command is hard-capped at ten minutes and is killed on the dot.** Run long rungs
  **detached, in the background, redirected to a file under `outputs/ladder/`, and poll the file.**
  Note that a `nohup … &` inside a tool call returns immediately and the wrapper reports success for
  the *shell*, not the work — poll the output file, not the exit code. Python block-buffers a
  redirected stdout unless you pass **`-u`**; with `-u` you can watch progress, without it a long
  rung shows zero bytes for most of its life and silence is not a stall. Measured wall clocks:
  **Rung 4 ~35 min per backend**, **Rung B numpy ~3 h 15 m** (and 11.9 h in session 26 when the
  machine slept and then ran on battery).
- **Idle the machine before any timing gate; do not merely read the clock.**
  `Win32_Processor.CurrentClockSpeed` is instantaneous and reports 3201 while sustained load is being
  clocked well below it. Session 25's Rung E read **68.2 s** against a 60 s limit after hours of load
  and **57.2 s** after a seven-minute idle; session 26 read **55.6 s** after idling first, and had to
  re-run Rung D for the same reason (**D-092**). `bench.py --les`, Rung E and Rung D's `Monitor` cost
  are all timing gates.
- **Before timing anything, check for stray processes**
  (`Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' }`) and kill them. Also
  **check the power state** — session 26 lost most of a night to the laptop running on battery at
  **1882 of 3201 MHz**. `Win32_Battery.BatteryStatus` is `2` on mains, `1` discharging.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` 1.16.0 was the last
  addition); T201, T202 and T203 added nothing. Anything new is a real decision and needs a row in
  `DOCS/STATE3.md` § Environment **in the same session**.
- **Check early whether Rung 3's own case wants the closure on.** Its `max(nu_t)/nu` is **0.1910**,
  which is above the quantitative band's 0.1 boundary — but Rung 3 runs at Re 100 under BGK and
  `TAU_FLOOR` does not refuse it, so the closure should never come on there and it should band
  `quantitative`. If your wiring turns the closure on for it, Rung 3, Rung E and Rung B all move and
  M9 breaks. That is the first thing to verify, not the last.

## Scope discipline

Work only what's in the contract. Packaging is **T205** and the app is **T206**–**T209** — none of
them is this session. If something else needs doing, `/new-task` it; do not expand this one. If it is
listed under `DOCS/IDEA4.md` § Deliberately deferred (XLB, 3D, STL, KBC, MRT, curved boundaries, wall
models, dynamic `Cs`, a web UI), the answer is no.

## Before the session ends

Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
`PROMPTS/028-…` for the next session. Do not end the session without it.
