# Session 24 — T201: the Smagorinsky closure in `lbm/core.py` + the NumPy backend

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer.
**Phase 2 is live** and its spec is `DOCS/IDEA4.md`. The solver is not the product: see `idea.md`
§ Risks — *"The trap"*, which names the standing temptation to keep polishing it because that part is
fun. This task is solver work, so that warning is pointed directly at this session.

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
T201 is the first task and it is the closure's NumPy half.

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (constraint 1 was rewritten by **D-081** and constraints
   17–20 are new), the session protocol, the conventions, the module map, § Current state.
2. `DOCS/STATE3.md` — **in full.** It is short: § Snapshot, § Blockers, § Open questions
   (**Q-201 … Q-204**), § Environment, § Performance baseline, § Decisions **D-080 … D-084**, the
   constraint fate table, and one session-log entry. This is the whole of Phase 2's accumulated state.
3. `DOCS/TASKS3.md` § **T201** — the task contract, in full. Also skim the backlog index so you know
   what T202 inherits from you.
4. `DOCS/IDEA4.md` § **The five things Phase 2 must get right** (1) and (2), and § **Validation
   ladder** (Rung F). These are what the contract cites.
5. `DOCS/PLAN3.md` § Why this order, § Session map and § Risks — T201 is session 24 of nine, and the
   phase's hard pressure valve is aimed at exactly this task and T202.
6. `lbm/core.py` — `collide`, `collide_stream`, and the D-011 / D-020 / D-033 ordering the fused path
   depends on. `lbm/backends/__init__.py` — the `Backend` protocol (**D-054**).

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 23: a **planning session**, no code. Phase 2 was chosen against XLB and
  3D on measured evidence (**D-080**), and `DOCS/IDEA4.md`, `DOCS/PLAN3.md`, `DOCS/TASKS3.md` and
  `DOCS/STATE3.md` were written. Phase 1's documents were frozen in place (**D-084**).
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 · A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 —
  **nine green**, last measured in session 22 on both backends. **F ⬜ · G ⬜ · H ⬜ · I ⬜ · J ⬜.**
- **Milestone reached:** **M8** (2026-08-27). Phase 2's are M9 … M12 and none is reached.
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110. Phase 2: none.
- `pytest`: **772 passed, 1 skipped** (302.6 s).

## Your task this session

**T201 — Smagorinsky closure: `lbm/core.py` + NumPy backend.** One task, this session only.

Run this first:

    /start-task T201

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** BGK gains an optional per-cell effective relaxation time computed from the second moment of
`f - feq`. With `Cs = 0` it is **bitwise** the collision Phase 1 shipped, on every path. The NumPy
backend implements it; Warp is T202. Nothing gets faster and nothing gets more accurate — **the
switchability is the deliverable.**

**Write the bitwise degeneracy test before the model does anything.** It is the cheapest possible
test and it is what protects all nine existing green rungs.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `smagorinsky_omega` cites its source in the docstring — Hou et al. (1996) or equivalent — and the **exact algebra is pinned by a test**, not just the shape. The filter width is one lattice unit and the docstring says so.
- [ ] **Bitwise degeneracy, both paths, asserted first:** with `cs_smag=0.0`, `collide` and `collide_stream` produce `f` `numpy.array_equal` to the Phase 1 functions after 1000 steps of Rung 3's case. Fused and unfused agree bitwise with each other, as **D-055** already requires.
- [ ] The closure **defaults off**: no call site in `lbm/`, `flow/` or `validate/` passes `cs_smag` unless it is testing the closure. `git grep cs_smag` in those trees returns only the closure's own definitions and tests.
- [ ] `nu_t = cs2 * (tau_eff - tau)` is derived through `tau`, never as a viscosity assigned directly — **constraint 2** applies to the effective relaxation time exactly as it applies to the base one, and a test asserts `nu_t >= 0` everywhere and `nu_t == 0` when `cs_smag == 0`.
- [ ] `tau_eff >= tau` for every cell, always (the closure adds viscosity and never removes it); asserted on a case with strong shear.
- [ ] **No allocation inside the step loop.** `smagorinsky_omega` takes an `out=` buffer, `Sim` preallocates it only when the closure is on, and a test asserts the allocation count is unchanged when it is off.
- [ ] `validate/les.py` exists and prints PASS/FAIL. On numpy it asserts (a) the bitwise clause above and (b) Rung 3's case with `cs_smag=0.17` still prints `Cd` in 1.25–1.45 and `St` in 0.155–0.175.
- [ ] `myenv/Scripts/python.exe -m validate.les` prints **PASS**.
- [ ] **All nine existing rungs re-run and print their published digits.** R1–R4, A–E. Any moved digit is a stop-work.
- [ ] `pytest` green, with the new tests counted in `DOCS/STATE3.md`.

### Inputs / outputs the contract names

**In:** `f`, `feq` `(9, ny, nx)` `float32`; `tau` float; `cs_smag` float, default `0.0`.
**Out:** `lbm/core.py::smagorinsky_omega(f, feq, tau, cs_smag, out=None) -> NDArray[np.float32]`
returning `(ny, nx)` `float32`; `lbm/core.py::collide` and `::collide_stream` gain a keyword-only
`cs_smag: float = 0.0`; `Backend.collide` and `Backend.collide_stream` gain the same in the protocol;
`lbm/backends/numpy_backend.py` implements it; `lbm/probe.py::eddy_viscosity(f, feq, tau, cs_smag)
-> NDArray[np.float32]` `(ny, nx)` — the `nu_t` field Rung G and constraint 18 both need;
`validate/les.py` (the Rung F harness).

### Constraints that bite on this task

- **Constraint 1 (rewritten by D-081)** — the physics is D2Q9, BGK, bounce-back, **plus exactly one
  turbulence closure, Smagorinsky, and it is named**. No MRT, no cumulant, no KBC, no curved or
  interpolated boundaries, no wall model, no dynamic `Cs`. The base arithmetic may not change, and
  with the closure **off** it must be bitwise what Phase 1 shipped.
- **Constraint 19 (new, D-081)** — the closure defaults off, and `Cs = 0` is bitwise identical to
  Phase 1 on every backend. *A closure you cannot switch off is a closure you cannot validate
  against.* This is the criterion that protects the other nine rungs; write it first.
- **Constraint 2** — `nu = cs2 * (tau - 0.5) = (tau - 0.5)/3`, and that governs `tau_eff` exactly as
  it governs `tau`. Smagorinsky modifies the **relaxation time**, never a viscosity directly.
- **Constraint 4** — the nine D2Q9 constants (`E`, `W`, `OPP`, `CS2`) come from `lbm/core.py` only,
  never redefined. `to_host` still yields `(9, ny, nx)` `float32`.
- **Constraint 5** — the ladder is ordered and non-negotiable; **all nine existing rungs stay a gate
  for every Phase 2 task.** Rung G is not started while Rung F fails.
- **Constraint 11** — restart is bit-identical within a backend. `tau_eff` is *derived*, so it adds
  nothing to the checkpoint: `f`, `mask`, `step_count` stay the entire state (**D-022**, **D-050**).
- **Coding conventions** — type hints with intent (`NDArray[np.float32]`), shapes in docstrings,
  **preallocate, never allocate inside the step loop**, `float32` throughout, docstrings cite the
  `DOCS/IDEA4.md` section.

### Blockers, open questions and decisions that affect you

**Blockers: none.**

- **Q-201** (open, T202 answers it, but *design for it now*) — "Does bitwise degeneracy on the Warp
  backend survive multiplying the closure term by zero, or does `cs_smag = 0` need a separately
  compiled kernel? **D-053** documents that the GPU contracts `x * a + b` into one rounding where
  NumPy does two, so a term that is algebraically zero is not automatically bitwise inert." Write the
  NumPy version so that a guarded branch is easy for T202 to add.
- **Q-202** (open, T203 answers it) — "What is `<nu_t>` on a *resolved* 2D Taylor–Green at
  `Cs = 0.17`, as a fraction of `nu`? Expected small. If the model fires hard on a smooth flow, that
  is a finding about the implementation rather than about turbulence."
- **D-081** — constraint 1's rewrite: one closure, additive and switchable; the base arithmetic still
  may not change. Supersedes D-046's rewrite of constraint 1.
- **D-082** — the closure is a **stability device, not a fidelity device**. Three bands, decided from
  the eddy viscosity a run actually generated. This is why `lbm/probe.py::eddy_viscosity` is part of
  *your* contract even though T204 is what consumes it.
- **D-055** (frozen, still in force) — "The pre-collision copy (D-011) is dropped on the fused path,
  on every backend, and the Warp fused pass streams `f_bb` straight into `f`… passing `f` where
  D-011's copy would go is **bitwise identical**." Your fused path must keep that property.
- **D-033** (frozen) — the Guo source term goes between collision and bounce-back, so a **forced run
  takes the unfused path**. Your closure has to work on both routes and they must agree bitwise.
- **D-053 / D-056** (frozen) — what cross-backend agreement measures: per-kernel worst 5.96e-08
  against a 1e-6 bar, whole-step 9.611e-06 at 1000 steps and not compounding. T202 holds the closure
  to the same two numbers.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` was the last addition).
  Anything new is a real decision and needs a row in `DOCS/STATE3.md` § Environment **in the same
  session**.
- **All nine rungs are green** as of session 22 and must still be green when you finish. Re-running
  the full ladder is expensive — Rung B alone is ~23 min — so budget for it, and note that Rung E's
  60 s gate needs the machine **on mains**: session 22's identical-physics re-run printed **FAIL at
  71.9 s** purely because the laptop was at 1882 of 3201 MHz. **D-035**: no absolute timing without
  `Win32_Processor.CurrentClockSpeed`, the power state and the GPU name beside it.
- A cross-check exists but is **not** a source: XLB's own 2D Smagorinsky is at
  `Autodesk/XLB:xlb/operator/collision/smagorinsky_les_bgk.py`. Their normalisation bakes in choices
  about filter width and strain norm that your docstring must state and your test must pin. Two
  implementations agreeing is evidence; one copied is not.

## Scope discipline

Work only what's in the contract. The Warp port is **T202**, the Taylor–Green harness is **T203**, and
the fidelity bands are **T204** — none of them is this session. If something else needs doing,
`/new-task` it; do not expand this one. If it is listed under `DOCS/IDEA4.md` § Deliberately deferred
(XLB, 3D, STL, KBC, MRT, curved boundaries, wall models, dynamic `Cs`, a web UI), the answer is no.

`DOCS/PLAN3.md` § Risks names the hard valve for this task: **if T201 and T202 together overrun by one
session, `Cs` freezes at the literature 0.17, no dynamic procedure is attempted, and the phase moves
to T205** — which depends on none of it.

## Before the session ends

Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
`PROMPTS/025-…` for the next session. Do not end the session without it.
