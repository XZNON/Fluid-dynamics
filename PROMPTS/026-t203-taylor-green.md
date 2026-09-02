# Session 26 — T203: the Taylor–Green harness, Rung G, and M9

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer.
**Phase 2 is live** and its spec is `DOCS/IDEA4.md`. The solver is not the product: see `idea.md`
§ Risks — *"The trap"*, which names the standing temptation to keep polishing it because that part is
fun. This task is a validation harness for solver physics, so that warning is pointed at this session
too — the harness is the deliverable, not a better closure.

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
T201 landed the closure on NumPy, T202 landed it on Warp and closed Rung F on both. **T203 is Rung G,
and Rung F plus Rung G is milestone M9.**

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (**5**, **3** and **12** are the ones that govern this
   session), the session protocol, the conventions, the module map, § Current state.
2. `DOCS/STATE3.md` — **in full.** § Snapshot, § Blockers, § Open questions (**Q-202** is yours;
   Q-201 is now closed), § Environment, § Performance baseline (the LES table is filled in),
   § Decisions **D-080 … D-090**, the constraint fate table, and all three session-log entries. The
   session-25 entry is the one that matters most to you, and it carries two warnings about *timing*
   measurements on this machine that will cost you an hour if you ignore them.
3. `DOCS/TASKS3.md` § **T203** — the task contract, in full. Also read § T201 and § T202 (both now
   `done`, every box ticked) so you know what the closure is and how it is switched, and skim the
   backlog index.
4. `DOCS/IDEA4.md` § **Validation ladder** (Rung G's row, which is what the contract cites) and
   § The five things Phase 2 must get right (2).
5. `DOCS/PLAN3.md` § Why this order, § Session map, § **Milestone gates** (M9's gate command is
   literal and you have to run all of it), and § Risks — the phase's hard pressure valve is aimed at
   T201/T202, which landed on schedule, so it is not armed.
6. `validate/poiseuille.py` — **Rung 1, the existing analytic-solution harness and the source of the
   1% bar you must meet.** Read it as the model for what you are writing. Then `lbm/core.py`'s
   `smagorinsky_tau_eff` / `smagorinsky_omega` and the `cs_smag == 0.0` branches (**D-085**,
   **D-086**), `lbm/probe.py::eddy_viscosity`, and `validate/les.py` (Rung F) for how a Phase 2 rung
   is shaped and how it takes `--backend`.

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 25 and it worked **T202 — the Smagorinsky closure on the Warp
  backend. It is `done`: every acceptance criterion was run and passed.** What landed: three Warp
  kernels (`_smag_scale_kernel`, `_collide_smag_kernel`, and a `_collide_bb_smag_kernel` that folds
  the second-moment reduction into the fused pass), `WarpBackend._smag_scalars`, `--backend` and a
  frozen Phase 1 *warp* oracle and a cross-backend clause in `validate/les.py`, a defaulted
  `cs_smag` on `validate/parity.py::step_case` / `::whole_step`, and `bench.py --les`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 · A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 · **F 🟩** — **ten
  green, all re-run in session 25 on BOTH backends with no physics digit moved.** G ⬜ · H ⬜ · I ⬜ ·
  J ⬜.
- **Milestone reached:** **M8** (2026-08-27). **M9 is one rung away**: it needs Rung F on both
  backends (done) *and* Rung G (you), *and* all nine existing rungs re-run, *and*
  `bench.py --backend warp --les` clearing its floors. **This session can reach M9.**
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110; Phase 2 **T201**, **T202**.
- `pytest`: **803 passed, 2 skipped** (286.1 s).
- **Numbers to keep still.** Rung F: `cs_smag = 0` bitwise on both backends, worst |diff|
  **0.000e+00** after 1000 steps of Rung 3's case; Rung 3 at `Cs = 0.17` printing **Cd 1.4143**,
  **St 0.1719** on both; `max(nu_t)/nu` **0.1910** on that wake; cross-backend with the closure on,
  worst kernel **2.980e-08** and whole step **9.611e-06**. The LES budget: warp
  **3914.2 / 660.0 / 406.0** steps/s at 40k / 1M / 2M against floors **3116 / 568 / 331**, the
  closure costing **4.6% / 8.9% / 8.8%**; numpy **81.8% / 80.5% / 78.1%** of its own BGK column.

## Your task this session

**T203 — Taylor–Green harness → Rung G → M9.** One task, this session only.

Run this first:

    /start-task T203

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** prove the closure adds the viscosity it claims to add and no more, against an **exact
analytic solution** rather than against a benchmark table.

**In:** a doubly periodic domain, no bodies.
**Out:** `validate/taylorgreen.py` printing PASS/FAIL, taking `--backend` and `--cs`.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] The harness initialises the exact 2D Taylor–Green vortex, `u = u0 cos(kx) sin(ky)`, `v = -u0 sin(kx) cos(ky)`, on a periodic domain, and measures the decay rate of the kinetic energy against `exp(-2 nu k^2 t)`.
- [ ] With `--cs 0`: the measured viscosity returns `nu = (tau - 0.5)/3` to **under 1%**, the bar Rung 1 already meets. This is a fourth independent check on the base solver and it must pass before any LES number is believed.
- [ ] With `--cs 0.17`: the measured viscosity returns `nu + <nu_t>` to **under 2%**, where `<nu_t>` is the domain average of `lbm.probe.eddy_viscosity` **computed from the model during the run**, not fitted to the decay curve afterwards. A fitted `nu_t` proves nothing and the test says so.
- [ ] The peak lattice velocity stays under 0.1 throughout (constraint 3) and the harness prints it.
- [ ] Both backends pass, and the printed digits agree to the **D-056** whole-step tolerance.
- [ ] `myenv/Scripts/python.exe -m validate.taylorgreen` and `--backend warp` both print **PASS**.
- [ ] **M9 gate run in full** — Rungs F and G on both backends, all nine existing rungs re-run, and `bench.py --backend warp --les` clearing its floors. Milestone claimed only on printed output.
- [ ] `pytest` green.

### Constraints that bite on this task

- **Constraint 5** — *the validation ladder is non-negotiable and ordered.* This is a rung: it is a
  script in `validate/` that prints pass/fail, and **Rung H (T204) does not start while it fails**.
  All ten green rungs stay a gate for every later task. *A wrong sim that looks plausible is the main
  failure mode of this project.*
- **Constraint 3** — *lattice velocity stays under 0.1.* The ceiling is **checked and printed**, not
  assumed. A Taylor–Green at too high a `u0` is a compressibility measurement wearing a viscosity
  measurement's clothes.
- **Constraint 12** — geometry is one boolean `solid`. There are **no bodies** here, so the geometry
  checks are vacuous; **say so in the docstring** rather than leaving a reader to wonder whether they
  were skipped or forgotten.
- **Constraint 2** — `nu = cs2 (tau - 0.5) = (tau - 0.5)/3`, and it governs `tau_eff` exactly as it
  governs `tau`. `<nu_t>` comes from `lbm.probe.eddy_viscosity`, which derives it through `tau_eff`;
  it is never assigned and never fitted.
- **Constraint 1 (D-081 form)** — D2Q9, BGK, bounce-back, plus exactly one closure, Smagorinsky,
  named. This session **measures** the closure; it does not tune it, and `Cs = 0.17` is not a knob.
- **Constraint 19** — the closure defaults off and `Cs = 0` is bitwise Phase 1 on every backend. Rung
  G's `--cs 0` clause runs on that path and must not need a tolerance to do so.
- **Constraint 4** — `to_host` yields `(9, ny, nx)` `float32`; the constants come from `lbm/core.py`
  only.
- **Coding conventions** — type hints with intent, shapes in docstrings, **preallocate, never allocate
  inside the step loop**, `float32` throughout, docstrings cite `DOCS/IDEA4.md`.

### Blockers, open questions and decisions that affect you

**Blockers: none.**

- **Q-202 (open — this task answers it)** — *"What is `<nu_t>` on a resolved 2D Taylor–Green at
  `Cs = 0.17`, as a fraction of `nu`? Expected small. If the model fires hard on a smooth flow, that
  is a finding about the implementation rather than about turbulence, and it belongs in
  § Decisions with its measurement."* For calibration: on Rung 3's **shedding cylinder wake**,
  `max(nu_t)/nu` is **0.1910**, and on Rung A's smooth Re ~ 24 channel it is **9.011e-02** — both
  measured in session 25. A resolved Taylor–Green should be well under either. **Answer it with a
  number and record it.**
- **D-085** (frozen normalisation) — filter width `Delta = 1` lattice unit, strain norm
  `|S| = sqrt(2 S_ab S_ab)`, `Q_ab = sum_i e_ia e_ib (f_i - feq_i)` with no factor of two, giving
  `tau_eff = 0.5 (tau + sqrt(tau^2 + 18 sqrt(2) Cs^2 |Q| / rho))`. **Do not re-derive it and do not
  adopt XLB's `36`** — that differs by exactly `sqrt(2)` because of the strain-norm convention, and
  `tests/test_smagorinsky.py` pins ours. `smagorinsky_tau_eff` is the primitive and
  `smagorinsky_omega` its reciprocal, in that order, because `1 / (1 / tau)` is not `tau` in
  `float32` and `nu_t` must be *exactly* zero when the closure is off.
- **D-086 / D-088** — constraint 19 is an explicit branch on both backends, and on warp it is **two
  separately compiled kernels**: `cs_smag = 0` launches the *unedited* Phase 1 kernel. So your
  `--cs 0` clause is running Phase 1's arithmetic exactly, on both backends, and any deviation it
  finds is a finding about the base solver, not about the closure.
- **D-089** — on the fused warp path the reduction is folded into `_collide_bb_smag_kernel`; the
  unfused `collide` keeps two kernels. Same arithmetic either way. Relevant only if you find yourself
  reading GPU kernels.
- **D-090** — Rung F takes `--backend` and carries **two** frozen Phase 1 oracles, one per backend,
  each transcribed exactly once in `validate/les.py` and marked *do not edit*. If Rung G ever needs an
  oracle, the same one-copy rule applies. Rung A stays closure-off.
- **D-053 / D-056** (frozen) — cross-backend agreement: per-kernel worst **5.960e-08** against a 1e-6
  bar, whole step **9.611e-06** at 1000 steps against 1e-4. Your "both backends agree" criterion is
  **D-056**'s number.
- **D-082** — the closure is a **stability device, not a fidelity device**. Rung G measures that it
  adds a known small viscosity to a *resolved* flow; it does not license any claim about turbulence.
  The bands are **T204** and nothing in this session should start banding anything.
- **Q-203 / Q-204** — open, and neither is yours (T204 and T205 answer them).
- One issue queued in session 25 and not a blocker: **`022ac461c920`** — `Sim` allocates
  `smag_work (4, ny, nx)` on the Warp backend, which never reads it (32 MiB of dead device memory at
  2M cells with the closure on). Fixing it crosses the T101 seam, so it is a decision, not a tidy-up.
  **Not yours** unless you choose to `/new-task` it.

### Three environment facts sessions 24 and 25 paid for, so you do not have to

- **A foreground command is hard-capped at ten minutes and is killed on the dot.** The clean
  workaround, found in session 25 and now the default: **run the long rungs detached, in the
  background, redirected to a file under `outputs/ladder/`, and poll the file.** A backgrounded run
  is not capped and leaves no orphan. Expect two things: Python **block-buffers a redirected stdout**,
  so a long rung shows **zero bytes** for most of its life and silence is *not* a stall — check
  `UserModeTime` on the worker instead, and note `myenv\Scripts\python.exe` is a **trampoline** whose
  real worker is a separate `uv` `python.exe`; and the measured wall clocks are **Rung 4 ~35 min per
  backend** and **Rung B numpy ~3 h 15 m**. Budget the M9 gate accordingly — it is most of a session
  on its own.
- **`Win32_Processor.CurrentClockSpeed` is an instantaneous reading and does not detect thermal
  throttling under sustained load.** Session 25's Rung E read **68.2 s** against a 60 s limit after
  ~2.5 hours of continuous full-load compute and **57.2 s** after a seven-minute idle, with the clock
  reading **3201 of 3201 on mains** both times and identical physics digits. **Idle the machine before
  any timing gate**, do not merely read the clock. `bench.py --les` and Rung E are both timing gates.
- **Before timing anything, check for stray processes**
  (`Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' }`) and kill them.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` 1.16.0 was the last
  addition); T201 and T202 added nothing. Anything new is a real decision and needs a row in
  `DOCS/STATE3.md` § Environment **in the same session**.
- A Taylor–Green case is **doubly periodic with no inlet, no outlet and no solid**, which is a
  configuration no existing rung runs. Check early that `lbm.runner.SimConfig` will build it — Rung 1
  is the closest thing in the tree and it still has walls.

## Scope discipline

Work only what's in the contract. The fidelity bands are **T204**, packaging is **T205**, and the app
is **T206**–**T209** — none of them is this session. If something else needs doing, `/new-task` it; do
not expand this one. If it is listed under `DOCS/IDEA4.md` § Deliberately deferred (XLB, 3D, STL, KBC,
MRT, curved boundaries, wall models, dynamic `Cs`, a web UI), the answer is no.

`DOCS/TASKS3.md` § T203 Notes is explicit about the one temptation here: Taylor–Green is chosen over
decaying 2D turbulence **because it has an exact solution**, and an enstrophy-cascade check is a
better test of a turbulence model and a worse test of *this* claim. Do not write one.

## Before the session ends

Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
`PROMPTS/027-…` for the next session. Do not end the session without it.
