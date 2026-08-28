# Session 23 — Phase 2 planning: design the XLB swap

**This is not a solver task and it is not a product task. There is no `T111`.** Phase 1 finished in
session 22 at **M8** and its backlog (`DOCS/TASKS2.md`, T101 → T110) is closed. This session
**plans Phase 2** and writes the planning documents; it should end with a spec, a plan and a task
list — not with a new feature in `lbm/` or `flow/`.

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis, in
its own words: *"The gap is not the solver. The gap is everything around the solver."*

**Phase 0** — complete, spec `DOCS/IDEA2.md`, closed at **M4** — was a validated, continuously
running 2D D2Q9 lattice-Boltzmann simulator in pure NumPy, with four physics rungs green against
published data.

**Phase 1** — complete, spec `DOCS/IDEA3.md`, closed at **M8** in session 22 — is the product layer
above it: `flow/` (units, fluids, auto-configuration, diagnosis and refusal, geometry repair, the
`Case`/`Result` API, a CLI) on a Warp GPU backend behind `lbm/backends/`. Five more rungs green.
The claim it closed on, measured: **a picture and three physical numbers reach Rung 3's published
cylinder bands in 49.5 s from a cold shell** — `Cd` 1.4040 against 1.25–1.45, `St` 0.1672 against
0.155–0.175.

Phase 2, per `idea.md` § Roadmap (where it is called Phase 3), is **the XLB swap**: replace our own
kernel with XLB underneath, keeping everything above the solver line — which is exactly what the
T101 backend seam was built to make possible. Whether that is still the right next move is the
first question this session has to answer honestly, not assume.

## Read these first, in this order

1. **Root `idea.md`** — in full. The product, the positioning, the roadmap, the risks. Especially
   § What we are actually building (the pipeline diagram), § Why this is defensible, § Roadmap
   (**Phase 3 — swap in XLB**, **Phase 4 — 3D + STL**), and § Risks, where **"The trap"** names
   this session's failure mode: *"It is very tempting to keep polishing the solver because that
   part is fun. The solver is not the product."* An XLB swap is solver work by definition, so the
   plan needs a pressure valve the way `DOCS/PLAN2.md` § Risks gave one to the Warp port (**D-043**).
2. **Root `README.md`** — rewritten in session 22; its § Quickstart, § Current state and the
   nine-rung validation-ladder table are the product-level statement of where things actually are.
3. `CLAUDE.md` — the **16 hard constraints** (the Phase 1 list, and the authority — **D-046** is
   only the record of *why* each reads as it does), the session protocol, the conventions, the
   module map, and § Current state.
4. `DOCS/STATE2.md` — **in full**, and it is long (1734 lines). § Snapshot, § Blockers (empty —
   Phase 1 closed with none), § Open questions (all closed), § Environment, § Performance baseline
   including session 22's calibration anchors, § Decisions **D-041 … D-079**, and the eleven session
   entries. This is the accumulated understanding Phase 2 is designed *from*.
   `old-Docs/STATE1.md` § Decisions (**D-005 … D-040**) is still in force and cited by number; read
   the entry a decision names, not the whole file.
5. `DOCS/PLAN2.md` — **as a model, not as content.** It is what a good phase plan looked like: a
   task graph, a one-task-per-session map, milestone gates with **literal gate commands**, and a
   risks table with pressure valves. Phase 2's plan should have the same shape.
6. `DOCS/IDEA3.md` § Deliberately deferred — what Phase 1 said no to and why: **XLB**, a UI, 3D,
   STL, packaging, multi-body studies, drag polars, parameter sweeps. Un-deferring one is a
   decision to record.
7. `lbm/backends/__init__.py` — the `Backend` protocol as it actually stands after **D-054**
   (allocation, the two general transfers, four kernels, four boundaries, both halves of the Guo
   body force, opaque array handles). **This is the seam an XLB backend would have to satisfy**, and
   reading it is how you find out whether the swap is a substitution or a rewrite.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 22: **T110 done, M8 reached, Phase 1 complete.** `validate/minute.py`
  is Rung E; five decisions **D-075 … D-079** landed; both of session 21's blockers are closed.
- **Phase 0 rungs: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩** — all four re-run in session 22 **on both
  backends**: R1 L2 0.3650% (numpy) / 0.3649% (warp), R2 0.75% and 0.21 cells on both, R3 `St`
  0.1731 `Cd` 1.4031 on both, R4 square `Cd` 1.5279 and polygon `Cd` 1.4276 on both.
- **Phase 1 rungs: A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩** — the whole ladder, and B is green on *both*
  backends now (warp 3.5%, numpy 15.2%).
- **Milestones M1 … M8 are all reached.** No blockers. No open questions.
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110.
- `pytest`: **772 passed, 1 skipped**.

## Your task this session

**Plan Phase 2.** There is no `/start-task` to run — there is no task file yet; writing one is the
deliverable. Work in this order:

1. **Decide what Phase 2 actually is, before planning it.** `idea.md` says XLB. Session 22 leaves
   a working, validated, *fast* kernel of our own (4155 / 757 / 441 steps/s at 40k / 1M / 2M cells,
   5x / 33x / 53x NumPy) and a product layer that meets its headline claim on it. So the honest
   question is what XLB buys that the current backend does not, and what it costs — and the
   alternatives on the table are named in `idea.md` § Roadmap and `DOCS/IDEA3.md` § Deliberately
   deferred: **3D + STL**, **a UI**, **packaging and distribution**, **parameter sweeps / drag
   polars**. Put the options side by side with what each is worth and what each risks, and get the
   user's call rather than inheriting the roadmap by default. Record it as a decision either way —
   **D-043** is the precedent for deliberately deviating from `idea.md`'s ordering, and it says so
   in as many words.
2. **Write the phase spec** — `DOCS/IDEA4.md`, modelled on `DOCS/IDEA3.md`: goal in one sentence,
   scope with an explicit **out** list, the pipeline made concrete as modules, the things the phase
   must get right, the validation ladder with its known answers, the performance budget with the
   arithmetic behind its floors, and § Deliberately deferred.
3. **Write the plan** — `DOCS/PLAN3.md`, modelled on `DOCS/PLAN2.md`: why this order, a dependency
   graph, a one-task-per-session map, milestone gates **with literal gate commands**, and a risks
   table where every risk has a signal and a pressure valve.
4. **Write the backlog** — `DOCS/TASKS3.md`, modelled on `DOCS/TASKS2.md`: one contract per task
   with goal, reads/depends-on, inputs and outputs with types and array shapes, acceptance criteria
   as a checklist, the constraints that bite, and notes.
5. **Write the state file** — `DOCS/STATE3.md`, and **freeze `DOCS/STATE2.md`** the way **D-041**
   froze `STATE1.md`: read for history, never edited, decision numbering continuing unbroken at
   **D-080**. Decide whether Phase 1's documents move to `old-Docs/` as Phase 0's did (**D-049**),
   and **price the move before making it** — D-049 records that `DOCS/IDEA2.md` deliberately stayed
   put because ~100 docstrings cite it by path, and `DOCS/IDEA3.md` is now cited the same way.
6. **Update `CLAUDE.md`** — pointers only: § Session protocol to the Phase 2 documents, § Current
   state to say Phase 2 is live. **Decide the fate of each of the 16 hard constraints explicitly**,
   as **D-046** did for Phase 0's twelve, and record the table. Constraint 1 in particular says the
   arithmetic may not change — an XLB backend is exactly the thing that would test it.

### Acceptance criteria (this is what marks the session done)

- [ ] A decision, recorded with its reasoning and the alternatives that were rejected, on **what
      Phase 2 is** — XLB, or something else, chosen deliberately rather than inherited.
- [ ] `DOCS/IDEA4.md` exists: goal, scope with an out-list, modules, the validation ladder with a
      known answer per rung, a performance budget with its arithmetic, § Deliberately deferred.
- [ ] `DOCS/PLAN3.md` exists: task graph, session map, milestone gates **with literal gate
      commands**, risks with pressure valves.
- [ ] `DOCS/TASKS3.md` exists: a full contract per task, acceptance criteria as checklists.
- [ ] `DOCS/STATE3.md` exists with § Snapshot, § Blockers, § Open questions, § Environment,
      § Performance baseline, § Decisions (continuing at **D-080**) and an empty § Session log;
      `DOCS/STATE2.md` is marked frozen.
- [ ] `CLAUDE.md` § Session protocol and § Current state point at the Phase 2 documents, and the
      fate of all 16 hard constraints is decided and recorded.
- [ ] `myenv/Scripts/python.exe -m pytest` still prints **772 passed, 1 skipped** — this session
      writes documents, so anything else is a regression it caused.
- [ ] `PROMPTS/024-...` written by `/checkpoint`.

### Constraints that bite on this session

- **Constraint 1 — the physics is D2Q9, BGK, bounce-back, and the arithmetic an implementation
  transcribes may not change.** Its D-046 rewrite already allows the *implementation* to move to a
  GPU backend. An XLB swap is the first thing that would test where that line actually is, so if
  Phase 2 is XLB, this constraint needs deciding in the spec rather than discovered in a task.
- **Constraint 5 — the validation ladder is non-negotiable and ordered**, and all nine existing
  rungs stay a gate for every Phase 2 task. Phase 2's own rungs are added to them, not instead of
  them. *"A wrong sim that looks plausible is the main failure mode of this project."*
- **Constraint 4 — `to_host` must yield `(9, ny, nx)` `float32`**, and the nine constants live in
  `lbm/core.py` and are imported from there. That shape is the portability contract; an XLB backend
  either satisfies it or the swap is not a swap.
- **Constraint 15 — `flow/` may import `lbm/`; `lbm/` may never import `flow/`, and a test asserts
  it.** This is the property that makes the swap a substitution. Whatever Phase 2 does must not
  weaken it.
- **Constraint 11 — restart is bit-identical within a backend, a printed tolerance across.** A new
  backend inherits both halves.
- **Constraint 6's replacement — no backend optimisation before its parity rung passes.**

### Decisions that constrain this session

- **D-043** — the Warp port was moved from Phase 2 into Phase 1 as a deliberate, recorded deviation
  from `idea.md` § Roadmap, *with a hard pressure valve*: if the port overran by one session it was
  to be demoted and the phase continued on NumPy. That is the shape a Phase 2 decision about XLB
  should take.
- **D-054** — the `Backend` protocol's real surface: `empty`, `zeros`, `copy`, `upload`, `download`,
  the four kernels, `bounce_back` / `moving_wall` / `inlet_velocity` / `outlet_zero_gradient`, and
  `force_velocity_shift` / `apply_body_force`. Arrays are opaque handles. **This is the list an XLB
  backend has to implement**, and the honest first question of the phase is how much of it XLB
  exposes.
- **D-053 / D-056 / D-057** — what cross-backend agreement actually measures: per-kernel worst
  5.96e-08 against a 1e-6 bar, whole-step 9.611e-06 at 1000 steps and *not compounding*, and every
  `float64`-then-rounded scalar computed host-side so the arithmetic is NumPy's. A third backend
  gets held to the same three, which is Rung A run against it.
- **D-041 / D-049** — how a phase boundary is handled: a new state file, the old one frozen, the
  decision numbering continuing unbroken, and a path move priced before it is made because
  docstrings cite specs by path.
- **D-075 … D-079** (session 22) — the newest and the ones a Phase 2 task is most likely to trip
  over: the chooser's domain is Rung 3's (24 D span, 8 D upstream), the force probe samples 10 times
  per convective time, `_RATE_TABLE` has measured 160k / 400k anchors, Rung D's cost check samples
  nine rounds of 300, and the default run is 80 convective times. Every one of them is a *measured*
  constant with its measurement in its own docstring; a backend change that invalidates any of them
  invalidates a number, not a preference.

### Blockers and open questions

**None.** Phase 1 closed with an empty § Blockers and every open question resolved. Two entries stay
in the local issue queue and neither is a blocker:

- `2fd69b874c32` — `Case.explain()` prints a different suggestion list than `Case.nearest()` acts
  on. A real T108 defect; not user-facing today because `flow/cli.py` prints the list it will
  actually execute. A candidate for an early Phase 2 task.
- `495777c58269` — `.gitignore` drops `*/__init__.py` and `tools/`. Open since session 16.

One thing carried five sessions and worth naming in the plan: **`Monitor` on `warp` has never been
timed.** Rung D runs on `numpy` by design, so the divergence probe's device-side cost is unmeasured.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` was the last addition).
  Anything new is a real decision and needs a row in the live state file's § Environment **in the
  same session**. If Phase 2 is XLB, note that installing it is itself a task with a risk row, not
  a preliminary.
- **The machine's power state matters and has already bitten once.** Session 22's final Rung E
  re-run printed **FAIL at 71.9 s** purely because the laptop was on battery at 1882 of 3201 MHz —
  identical physics, different CPU. **D-035**: no absolute timing without
  `Win32_Processor.CurrentClockSpeed`, the power state and the GPU name beside it. Check the mains
  before quoting a number.
- Confirm the starting point: `myenv/Scripts/python.exe -m pytest` prints **772 passed, 1 skipped**,
  and `myenv/Scripts/python.exe -m validate.minute --backend warp` prints **PASS** in under 60 s on
  mains.

## Scope discipline

This session writes documents. **It should not write solver or product code**, and it should not
start Phase 2's first task — the plan says one task per session for a reason, and a phase planned
and begun in the same session is a phase whose first task shaped its plan. If something small and
real turns up, `/new-task` it against the new backlog.

## Verify, then close

1. Check every acceptance criterion above by reading the file you wrote, not from memory.
2. Re-run `pytest` to confirm the document work broke nothing.
3. **Run `/checkpoint`** — it updates the live state file, syncs the backlog, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
