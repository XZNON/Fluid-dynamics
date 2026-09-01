# Session 25 — T202: the Smagorinsky closure on the Warp backend

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer.
**Phase 2 is live** and its spec is `DOCS/IDEA4.md`. The solver is not the product: see `idea.md`
§ Risks — *"The trap"*, which names the standing temptation to keep polishing it because that part is
fun. This task is solver work on a GPU, so that warning is pointed directly at this session.

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
T201 landed the closure's NumPy half last session; **T202 is its Warp half**, and it is the task that
closes Rung F.

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (constraint 1 was rewritten by **D-081**; **19** is the
   one that governs this session), the session protocol, the conventions, the module map,
   § Current state.
2. `DOCS/STATE3.md` — **in full.** § Snapshot, § Blockers, § Open questions (**Q-201 … Q-204**),
   § Environment, § Performance baseline, § Decisions **D-080 … D-087**, the constraint fate table,
   and both session-log entries. It is still short, and the session-24 entry is the one that matters
   most to you.
3. `DOCS/TASKS3.md` § **T202** — the task contract, in full. Also read § T201 (now `done`, every box
   ticked) so you know exactly what you are porting, and skim the backlog index.
4. `DOCS/IDEA4.md` § **Performance budget** and § **Validation ladder** (Rung F). These are what the
   contract cites.
5. `DOCS/PLAN3.md` § Why this order, § Session map and § Risks — T202 is session 25 of nine, and the
   phase's hard pressure valve is aimed at exactly this task and the one before it.
6. `lbm/core.py` — `smagorinsky_tau_eff`, `smagorinsky_omega`, and the `cs_smag == 0.0` branch in
   `collide` / `collide_stream` (**D-086**). `lbm/backends/warp_backend.py` — the kernels, and the
   two `NotImplementedError` stubs that name this task. `validate/les.py` — the Rung F harness you
   are extending.

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 24 and it worked **T201 — the Smagorinsky closure in `lbm/core.py` and
  the NumPy backend. It is `done`: every acceptance criterion was run and passed.** What landed:
  `lbm.core.smagorinsky_tau_eff` / `smagorinsky_omega`, `CS_SMAG_LITERATURE = 0.17`,
  `SMAG_Q_COEFF = 18 sqrt(2)`, keyword-only `cs_smag` on `collide` / `collide_stream`,
  `lbm.probe.eddy_viscosity`, `SimConfig.cs_smag`, the seam carried through `Backend` and the NumPy
  backend, `validate/les.py` (Rung F), and 28 tests in `tests/test_smagorinsky.py`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 · A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 — **nine green,
  re-run in session 24 with no physics digit moved** — and **F 🟨: green on numpy, unattempted on
  warp, which is you.** G ⬜ · H ⬜ · I ⬜ · J ⬜.
- **Milestone reached:** **M8** (2026-08-27). Phase 2's are M9 … M12 and none is reached. **M9 needs
  Rung F on both backends *and* Rung G**, so this session does not reach it on its own.
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110; Phase 2 **T201**.
- `pytest`: **800 passed, 1 skipped** (317.1 s).
- **Rung F's numbers to beat:** `cs_smag = 0` bitwise on numpy, worst |diff| **0.000e+00** after 1000
  steps of Rung 3's case on both paths; Rung 3 at `Cs = 0.17` printing **Cd 1.4143**, **St 0.1719**;
  `max(nu_t)/nu` **0.191** on that wake.

## Your task this session

**T202 — The closure on the Warp backend.** One task, this session only.

Run this first:

    /start-task T202

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** the same arithmetic on the GPU, agreeing with NumPy to the tolerance **D-053** and **D-056**
already established, and costing less than 25% of the BGK step rate.

**In:** the T201 signatures, unchanged.
**Out:** `cs_smag` support in the Warp backend's `collide` and `collide_stream` kernels;
`validate/les.py` gains `--backend`; `bench.py` gains `--les`; the LES row of
`DOCS/STATE3.md` § Performance baseline.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] **Bitwise degeneracy on warp too:** `cs_smag=0.0` gives `f` bitwise identical to the Phase 1 warp kernel after 1000 steps. Not "within tolerance" — identical. The closure must compile out or multiply by zero without touching the result.
- [ ] Cross-backend agreement with the closure **on** meets the existing contract: per-kernel worst under **1e-6** in `f` units (**D-053**'s bar), whole step under **1e-4** in `max|Δu|/U` at 1000 steps (**D-056**'s bar). The measured numbers are printed and recorded, not just compared.
- [ ] Any `float64`-then-rounded scalar the closure needs is computed **host-side in NumPy's own expression order** and uploaded (**D-057**). No per-thread `float32` recomputation of a constant.
- [ ] `myenv/Scripts/python.exe -m validate.les --backend warp` prints **PASS**.
- [ ] `myenv/Scripts/python.exe -m validate.parity --backend warp` re-run and prints **PASS** with the closure off; its published numbers unmoved.
- [ ] `bench.py --backend warp --les` clears **≥3116 / ≥568 / ≥331** steps/s at 40k / 1M / 2M cells, quoted with the CPU clock, power state and GPU name (**D-035**), by alternating rounds. The NumPy column takes the same 25% rule against its own measured baseline.
- [ ] **All nine existing rungs re-run on both backends** and print their published digits.
- [ ] `pytest` green.

### Constraints that bite on this task

- **Constraint 19 (D-081)** — *the closure defaults off, and `Cs = 0` is bitwise identical to Phase 1
  on every backend.* This session is the "every backend" half of that sentence. A closure you cannot
  switch off is a closure you cannot validate against.
- **Constraint 1 (rewritten by D-081)** — D2Q9, BGK, bounce-back, **plus exactly one turbulence
  closure, Smagorinsky, and it is named**. No MRT, no cumulant, no KBC, no curved or interpolated
  boundaries, no wall model, no dynamic `Cs`. The *implementation* may move backend; the **base**
  arithmetic it transcribes may not change.
- **Constraint 2** — `nu = cs2 (tau - 0.5) = (tau - 0.5)/3`, and that governs `tau_eff` exactly as it
  governs `tau`. Smagorinsky modifies the **relaxation time**, never a viscosity directly.
- **Constraint 4** — the backend owns its layout but `to_host` still yields `(9, ny, nx)` `float32`;
  the nine constants (`E`, `W`, `OPP`, `CS2`) come from `lbm/core.py` only, uploaded to the device and
  **never redefined**. `SMAG_Q_COEFF` is now one of the constants that rule covers.
- **Constraint 5** — the ladder is ordered and non-negotiable; **all nine existing rungs stay a gate**.
  Rung G (T203) is not started while Rung F fails.
- **Constraint 6's replacement** — no backend optimisation before its parity rung passes. Rung F on
  warp is that rung for this feature: make it correct, then make it fast.
- **Constraint 11** — restart is bit-identical within a backend, a printed tolerance across. `tau_eff`
  is *derived*, so the closure adds nothing to the checkpoint: `f`, `mask`, `step_count` stay the
  entire state (**D-022**, **D-050**). Assert it.
- **D-035** — no absolute steps/s figure without `Win32_Processor.CurrentClockSpeed`, the power state
  and the GPU name beside it; alternating rounds, best round per variant.
- **Coding conventions** — type hints with intent, shapes in docstrings, **preallocate, never allocate
  inside the step loop**, `float32` throughout, docstrings cite `DOCS/IDEA4.md`.

### Blockers, open questions and decisions that affect you

**Blockers: none.**

- **Q-201 (open — this task answers it)** — "Does bitwise degeneracy on the Warp backend survive
  multiplying the closure term by zero, or does `cs_smag = 0` need a separately compiled kernel?
  **D-053** documents that the GPU contracts `x * a + b` into one rounding where NumPy does two, so a
  term that is algebraically zero is not automatically bitwise inert." **T201 narrowed it**: **D-086**
  already made `cs_smag == 0.0` an explicit branch on the reference path, so you inherit a shape to
  port rather than a tolerance to argue about. What is still genuinely open is whether Warp needs
  *two compiled kernels* or one guarded branch suffices. **Answer it by measurement and record the
  answer.**
- **D-085** — the closure's normalisation is fixed and written down: filter width `Delta = 1` lattice
  unit, strain norm `|S| = sqrt(2 S_ab S_ab)`, `Q_ab = sum_i e_ia e_ib (f_i - feq_i)` with no factor of
  two, giving `tau_eff = 0.5 (tau + sqrt(tau^2 + 18 sqrt(2) Cs^2 |Q| / rho))`. **Do not re-derive it
  and do not adopt XLB's `36`** — that differs by exactly `sqrt(2)` because of the strain-norm
  convention, and `tests/test_smagorinsky.py` pins ours. `smagorinsky_tau_eff` is the primitive and
  `smagorinsky_omega` its reciprocal, in that order, because `1 / (1 / tau)` is not `tau` in
  `float32` and `nu_t` must be *exactly* zero when the closure is off.
- **D-086** — constraint 19 is an explicit `cs_smag == 0.0` branch, not a zero-valued term. **This is
  the design you are porting.** `DOCS/TASKS3.md` § T202 Notes is explicit: if the closure cannot be
  made bitwise-degenerate on warp, **the branch is the fix, not the tolerance.**
- **D-087** — Rung F reuses Rung 3's own harness (`validate/cylinder.py` takes a defaulted `cs_smag`)
  and the **frozen Phase 1 collision lives in exactly one place**, `validate/les.py`, imported by
  `tests/test_smagorinsky.py`. It is marked *do not edit*. When you add `--backend`, the warp
  bitwise clause needs its own oracle — the Phase 1 **warp** kernel — and the same one-copy rule
  applies to it.
- **D-053 / D-056** (frozen) — what cross-backend agreement measures: per-kernel worst 5.96e-08
  against a 1e-6 bar, whole-step 9.611e-06 at 1000 steps and not compounding. The closure is held to
  the same two numbers.
- **D-057** (frozen) — a `float64`-then-rounded scalar is computed host-side in NumPy's own expression
  order and uploaded. `SMAG_Q_COEFF * cs_smag * cs_smag` and `tau32 * tau32` are exactly such
  scalars; `lbm/core.py` already folds them host-side and keeps `SMAG_Q_COEFF` in `float64` for this
  reason.
- **D-082** — the closure is a **stability device, not a fidelity device**. `lbm.probe.eddy_viscosity`
  exists and T204 consumes it; nothing in this session should start banding anything.
- **Q-202 / Q-203 / Q-204** — open, and none of them is yours (T203, T204 and T205 answer them).

### Two environment facts session 24 paid for, so you do not have to

- **A process running longer than roughly ten minutes under the agent's own tooling is liable to be
  killed on this machine.** Session 24 lost several Rung B attempts that way; worse, one killed
  attempt left an **orphaned child still running**, which then competed for CPU with its replacement
  and made everything after it look stalled. **Before timing anything, check for stray processes**
  (`Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*validate*' }`) and kill
  them. Note that `myenv\Scripts\python.exe` is a *trampoline*: the real worker is a separate
  `uv` `python.exe`, so the parent showing 0 CPU means nothing. **Rung B (~23 min) and Rung 4
  (~36 min) are both past that ten-minute line** and you will hit this.
- **The CPU throttles.** It sat at `CurrentClockSpeed` **1990 of 3201 MHz on mains** for most of
  session 24 and recovered to **3201 of 3201** later. Every timing figure that moved last session
  moved because of this and no physics digit did. **Read the clock immediately before any D-035
  measurement**, and re-read it after; if it is not at 3201, the number is not quotable.
  `bench.py --les` and Rung E are both timing gates and both need the clean state.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` 1.16.0 was the last
  addition), and T201 added nothing. Anything new is a real decision and needs a row in
  `DOCS/STATE3.md` § Environment **in the same session**.
- **Rung E's 60 s gate needs the machine on mains at full clock.** Session 24 measured **55.7 s**
  against the 60 s limit at 3201 of 3201 MHz; session 23's attempt at 1882 MHz read 71.9 s on
  identical physics. **D-035**: no absolute timing without the clock, the power state and the GPU name
  beside it.
- A cross-check exists but is **not** a source: XLB's own 2D Smagorinsky is at
  `Autodesk/XLB:xlb/operator/collision/smagorinsky_les_bgk.py`, and it has a Warp implementation.
  Session 23 measured their closure costing **17%** of their BGK step rate at 100k cells, which is
  only useful as a sanity check that our own 25% floor is generous. Two implementations agreeing is
  evidence; one copied is not — and their normalisation is **not** ours (**D-085**).

## Scope discipline

Work only what's in the contract. The Taylor–Green harness is **T203**, the fidelity bands are
**T204**, and packaging is **T205** — none of them is this session. If something else needs doing,
`/new-task` it; do not expand this one. If it is listed under `DOCS/IDEA4.md` § Deliberately deferred
(XLB, 3D, STL, KBC, MRT, curved boundaries, wall models, dynamic `Cs`, a web UI), the answer is no.

`DOCS/PLAN3.md` § Risks names the hard valve for this task: **if T201 and T202 together overrun by one
session, `Cs` freezes at the literature 0.17, no dynamic procedure is attempted, and the phase moves
to T205** — which depends on none of it. T201 landed on schedule, so the valve is not yet armed; it
arms if this session does not finish.

## Before the session ends

Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
`PROMPTS/026-…` for the next session. Do not end the session without it.
