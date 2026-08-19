# Session 17 — T105: Auto-configuration → Rung B → M6

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator,
spec `DOCS/IDEA2.md`, closed at M4 on 2026-08-13 with all four validation rungs green. Phase 0 is
not the product; it exists so we understand LBM well enough to design the layer above it.

**Phase 1 is live**: that layer — a `flow/` package plus a CLI, on a Warp GPU backend, ten tasks
`T101` → `T110`, five validation rungs A–E, milestones M5 → M8. Spec `DOCS/IDEA3.md`, plan
`DOCS/PLAN2.md`.

**This is the session the phase is named after.** `DOCS/IDEA3.md` § 1 calls `flow/autoconfig.py`
*"the single most product-defining module in the phase"*, and `DOCS/TASKS2.md` § T105 calls it
**the moat**. Everything before it was plumbing; this is judgement.

## Read these first, in this order

1. `CLAUDE.md` — the 16 hard constraints (the Phase 1 list), session protocol, conventions.
   **Constraints 2, 3 and 12 are yours, and this is the module where they are enforced for users
   rather than documented.**
2. `DOCS/STATE2.md` — **in full**: snapshot, blockers, open questions, environment, performance
   baseline, decisions D-041 … D-058, and the session 13 → 16 log entries.
3. `DOCS/TASKS2.md` § T105 — the task contract, in full. Also the backlog index row: **T106 depends
   on you directly**, and T108 through it.
4. `DOCS/IDEA3.md` § The five things Phase 1 must get right, **item 1 in full** ("The user never
   types a lattice quantity") · § Validation ladder (the Rung B row) · § Deliberately deferred.
5. `lbm/units.py` **in full** — `LatticeUnits.from_physical` is the arithmetic you are choosing
   inputs *for*. It converts and it refuses; it does not choose. You are the chooser.
6. `validate/cylinder.py::tau_for` and `validate/polygons.py::tau_for_rung4` — **read these before
   writing anything**. They are hand-tuned instances of exactly the function you are writing, and
   the contract's Notes say so: if `plan()` cannot reproduce their choices within a factor, one of
   the two is wrong and finding out which is the session's real work.
7. `flow/quantity.py` and `flow/fluids.py` (T104, last session) — your inputs arrive as `Quantity`
   and `Fluid`, already in SI, already dimension-checked.
8. `DOCS/PLAN2.md` § Session map (T105 is session 17) and § Risks — the row *"Auto-config becomes a
   pile of tuned constants nobody can defend"* is aimed at this session by name.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 16: **T104 is done.** The `flow/` package exists —
  `flow/__init__.py`, `flow/quantity.py` (a `Quantity` that parses `"20 m/s"`, `"72 km/h"`,
  `"1.5e-5 m^2/s"`, `"20 °C"` and bare numbers with a declared default unit), and `flow/fluids.py`
  (six cited fluids: water, air, olive oil, helium, glycerine, honey). `pytest` prints **547 passed,
  1 skipped** (119 new). Nothing under `lbm/` was touched.
- **Constraints 13 and 15 are now enforced by test rather than aspiration**, in
  `tests/test_flow_package.py`: an AST scan asserts no module under `lbm/` imports `flow` (and a
  companion test proves the scan has teeth against three synthetic violations), and a signature scan
  asserts no public `flow/` name or parameter is a lattice quantity — `tau`, `u_lattice`,
  `steps_per_frame`, `cells_per_length`, `nx`, `ny`, `dx`, `dt` and friends. **Both scans will run
  against the code you write this session, and `Plan` is the first object with a real chance of
  tripping them** — see "Constraints that bite", below, because the answer is not "weaken the test".
- **Phase 1 rung status: A 🟩 · B ⬜ · C ⬜ · D ⬜ · E ⬜.** Rung A is green in full: kernels and
  boundaries worst **5.96e-08** against a 1e-6 bar, whole step **9.611e-06** at 1000 steps against
  1e-4 and *not compounding*, a `warp` checkpoint resumed on `numpy` at **8.196e-06**, restart
  bit-identical within a backend.
- **Phase 0 rung status: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩** — **all four re-run in session 16**, on
  `warp` for R3/R4: R1 L2 **0.3650%**, R2 **0.75%** vs Ghia and vortex **0.21 cells**, R3 St
  **0.1731** and Cd **1.4031 ± 0.0086**, R4 square PASS and polygon Cd **1.4276**. Every digit
  matches session 11's published values. All four stay a gate for every Phase 1 task.
- **Performance:** `bench.py --backend warp` prints **4155 / 757 / 441 steps/s** at 40k / 1M / 2M
  cells against floors of 2000 / 250 / 150 — 5× / 33× / 53× NumPy — using 391 MiB of the 4 GB card at
  2M cells. Measured at 3201 MHz of 3201 MHz on mains, RTX 3050 Laptop GPU, driver 592.82 (**D-035**).
  **`Plan.estimated_seconds` predicts against these numbers**, so read § Performance baseline in
  `DOCS/STATE2.md` before inventing a rate model.
- **Milestone reached:** **M5** (2026-08-18). **M6 is this session** — Rung B is its gate.
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101, T102, T103, T104.

## Your task this session

**T105 — Auto-configuration.** One task, this session only. **M6.**

Run this first:

    /start-task T105

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** given physics, choose everything the solver needs — resolution, `tau`, lattice `U`, domain
size and shape, run length, frame cadence, colour limits.

**Inputs:** a `Fluid`, a speed `Quantity`, a size `Quantity`, a mask, and a quality level
(`"fast" | "balanced" | "accurate"`).

**Outputs:** `flow/autoconfig.py::Plan` — a frozen dataclass carrying `cells_per_length: int`,
`tau: float`, `u_lattice: float`, `domain: tuple[int, int]` (ny, nx), `steps: int`,
`steps_per_frame: int`, `vorticity_limit: float`, `dx`, `dt`, `Re`, plus `warnings: list[str]` and a
`why: dict[str, str]` giving one sentence per chosen number; `::plan(...) -> Plan`;
`validate/autoconfig.py` printing PASS/FAIL.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `plan(...)` returns a `Plan` whose every field has an entry in `why`, and a test asserts that (a field with no explanation is a field the user cannot check).
- [ ] Every guardrail is enforced at plan time, each citing its decision: `u_lattice < 0.1` with the **1.8× bluff-body speed-up headroom** (D-032), `tau` above the floor appropriate to the geometry (D-029/D-032/D-036), blockage < 10% and ≥ 8 D downstream (constraint 12, D-019, D-026), min solid thickness ≥ 3 cells (D-017).
- [ ] The three quality levels differ **only** in `cells_per_length` and run length, and `accurate` is a strict refinement of `fast` — a test asserts monotonicity of resolution and of wall-clock estimate.
- [ ] `Plan.estimated_seconds` predicts wall clock from the measured backend rate; on the committed cylinder case the prediction is **within 25%** of the real run, and the script prints predicted vs actual.
- [ ] **Rung B — `validate/autoconfig.py`:** a sweep of at least 24 cases (fluids × speeds × sizes × quality) where every planned case (a) satisfies every guardrail, (b) runs 5000 steps with **no `nan`** and peak `|u|` under 0.1, (c) reproduces its requested `Re` to **0.1%** through `LatticeUnits.reynolds()`. Prints PASS/FAIL and the worst case of each.
- [ ] Cases that *cannot* be planned raise `Unrepresentable` (T106 turns it into prose) carrying structured fields — `reason`, `quantity`, `value`, `limit`, `suggestions` — rather than a formatted string.
- [ ] `pytest tests/test_autoconfig.py` green; Phase 0 rungs still green.

### Constraints that bite on this task

From `CLAUDE.md` § Hard constraints, in their Phase 1 form:

- **Constraint 2** — *viscosity is not a free parameter*: `nu = cs2 (tau - 0.5) = (tau - 0.5)/3`.
  Never expose a `nu` setter that does not go through `tau`. You *choose* `tau`, and you choose it
  by choosing a resolution — `tau = 0.5 + 3 U N / Re` is the only expression, and it lives in
  `lbm/units.py`. Do not re-derive it here; call it.
- **Constraint 3** — *lattice velocity stays under 0.1.* Any config path that can produce
  `|u| >= 0.1` must warn **at setup, not at `nan` time**, and this module is where that is enforced
  for users who never see `u`. The free stream is not what the ceiling caps — the **peak** is, and
  the flow accelerates round a body: 1.79× measured on a square cylinder, 1.61× on a disc
  (`lbm/units.py::BLUFF_BODY_SPEEDUP = 1.8`). **It raises; it does not warn.**
- **Constraint 12** — *geometry is one boolean array* `solid`, shape `(ny, nx)`: solid at least 3
  cells thick, object ≥ 8 diameters from the outlet, blockage under ~10%. **The domain is chosen
  here**, so there is no excuse for a domain that violates it.
- **Constraint 13** — *no lattice quantity in any public `flow/` signature.* This is the sharp one
  for T105 and it needs reading carefully: `Plan` is *output*, and the contract requires it to carry
  `tau`, `u_lattice`, `cells_per_length`, `domain`, `steps_per_frame`, `dx` and `dt`. That is
  **derived and printed**, which is exactly what `DOCS/IDEA3.md` § 1 asks for — "everything else is
  derived and **printed**, because a derived number the user cannot see is a number they cannot
  check." The constraint bans lattice quantities from **signatures** — what the user *types* — not
  from results. `tests/test_flow_package.py` currently scans public *parameters* and public
  *exported names*, and will not flag a dataclass field; if it does flag something, the fix is to
  make the scan precise about the input/output distinction **and say so in `DOCS/STATE2.md`
  § Decisions**, not to quietly delete the assertion.
- **Constraint 15** — `flow/` may import `lbm/`; `lbm/` may never import `flow/`. You *will* import
  `lbm.units` and `lbm.geometry` this session; that is legal and there is a test asserting it stays
  legal. The reverse is what the scan forbids.
- **Constraint 5** — the ladder is ordered and non-negotiable. **Rung B is your gate**, and its
  harness (`validate/autoconfig.py`) is built in this task, *before* the code it validates
  (**D-047**). All four Phase 0 rungs must still be green.
- **Constraint 16** — no silent substitution. A case you cannot plan is refused, not quietly
  down-scaled.

### Blockers, open questions and decisions that affect you

**Blockers:** none.

**Open questions:**

- **Q-101** — does `python -m lbm.runner` (the M4 gate command) survive as a working entry point once
  `python -m flow` exists, or become a pointer to it? **T109 decides it, not you** — but do not do
  anything in `flow/` that forecloses either answer.
- **Q-102** — is D-017's documented limit (a thin appendage *fused* to a thick body shares its
  component and is not reported) closable without false-alarming on a plain disc? **T107's**, not
  yours. Your thickness guardrail uses `min_thickness` as it stands, limit and all.

**Decisions that constrain this session.** `DOCS/PLAN2.md` § Risks: *"every constant in `autoconfig`
cites a Phase 0 decision or gets measured in the session that adds it and recorded in
`DOCS/STATE2.md` § Decisions."* Treat that as a hard rule — a magic number with no citation is a
defect in this module, not a style issue. The ones the contract names:

- **D-017** — minimum solid thickness is measured **per 8-connected component** as `2*max(d) - 1`,
  `d` the Chebyshev distance from a solid cell to the nearest fluid cell, reported as the minimum
  over components. Its known limit: a thin appendage *fused* to a thick body shares that body's
  component and is never reported (that is Q-102).
- **D-019** — the characteristic length `D` is the **cross-stream extent of the object's bounding
  box** (bbox height for `inlet_axis="x"`), and the blockage denominator is the **fluid** span — the
  cross-stream domain minus fully-solid border layers.
- **D-023** — `steps_per_frame(dt, fps=60.0, speed=1.0) = max(1, round(speed / (fps * dt)))`, with
  `dt` in *seconds of physical time per lattice timestep*. Grid size enters only through `dt`.
  Constraint 7: `steps_per_frame` is **computed**, never hardcoded to 20.
- **D-026** — Rung 3's lateral boundaries are **periodic, not no-slip walls**, and the fluid span is
  **24 D** (4.2% blockage), well inside constraint 12's 10% limit. `validate/cylinder.py::WALL = 0`,
  `SPAN_D = 24`.
- **D-028** — `render` takes **symmetric** colour limits and refuses an asymmetric pair; the
  colormap has 257 entries. Your `vorticity_limit` is one number, applied ±.
- **D-029** — D-016's `TAU_FLOOR = 0.53` is **not** safe for a bluff body in a free stream. Rung 4
  enforces a measured **0.54**. The measurements behind it: a disc at `tau = 0.5330` blew up by step
  1500, a square at 0.5346 by step 3200.
- **D-032** — `lbm/units.py` enforces `tau > 0.51` and `U < 0.1` **and nothing stricter**, and ships
  `LatticeUnits.stability_note()` beside it. It is the loosest of the project's three `tau` floors —
  a floor on nonsense, not a stability guarantee.
- **D-036** — `validate/cylinder.py` has its own `TAU_FLOOR = 0.537`; the band 0.537 … 0.54 is Rung
  4's alone, pinned by a test. **Three floors, and picking the right one per geometry is the
  criterion**: 0.51 generic (D-032), 0.537 Rung 3 (D-036), 0.54 bluff body (D-029).
- **D-040** — `--resolution N` means **N cells across the body**, not across the picture.
  `lbm.runner._body_mask` rasterises, measures the solid bbox, rescales by the shortfall, repeats
  (at most three passes). Your `cells_per_length` means the same thing.
- **D-045** — refuse, explain in the user's units, and offer the nearest runnable case; never
  substitute silently. Refusals carry `reason`, `quantity`, `value`, `limit`, `suggestions`. **Your
  `Unrepresentable` is the structured half of that** — T106 turns it into prose and Rung D feeds its
  top suggestion back through you and runs it, so **a suggestion that does not fix its case is a
  failing test**. Build `suggestions` as data (a quantity and a value), not as a sentence.
- **D-047** — a rung's harness is built in the task that needs it, before the code it validates, and
  the rungs live in `validate/` as `python -m validate.<rung>` printing PASS/FAIL.
- **D-058** (session 16) — the fluid library carries what its cited sources give, and the T104
  contract's parenthetical ordering was physically wrong and was not honoured. Relevant to you as
  precedent: **when a contract sentence and a measurement disagree, the measurement wins and the
  disagreement is logged as a decision.** The contract's Notes say the same thing about
  `tau_for` / `tau_for_rung4`.
- **D-035** — no absolute steps/s figure without the CPU clock, the power state and the GPU name
  beside it. `Plan.estimated_seconds` is a timing claim; quote its conditions.

### Before you start

- **Nothing to install.** `myenv` (Python 3.11.15) has numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0,
  pytest 9.1.1, pygame 2.6.1, imageio 2.37.4, imageio-ffmpeg 0.6.0, psutil 7.2.2, warp-lang 1.16.0.
  Anything new is a **recorded decision with a reason** in `DOCS/STATE2.md` § Decisions plus a row in
  § Environment — not a quiet `pip install`.
- Confirm the starting point before changing anything: `myenv/Scripts/python.exe -m pytest` should
  print **547 passed, 1 skipped**, and `myenv/Scripts/python.exe -m validate.parity --backend warp`
  should print **PASS**.
- Rung timings, for planning the session: `validate.poiseuille` and `validate.cavity --re 100` are
  ~1 minute each and both take `--backend`; `validate.cylinder` is ~3 minutes on
  `warp --headless`; `validate.polygons` about 12. **Rung B's own sweep runs 5000 steps × 24 cases**
  — size the cases so the script is minutes, not hours, and say what it cost.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no. **This session does not write `flow/diagnose.py`** — turning `Unrepresentable` into prose and a
nearest-runnable-case offer is T106, and Rung D is its gate. You raise the structured exception; you
do not format it.

Two specific traps for this task:

1. **Tuned constants with no citation.** `DOCS/PLAN2.md` § Risks names this as the failure mode of
   T105 specifically. Every number in `flow/autoconfig.py` cites a D-id or is measured this session
   and recorded.
2. **Reproducing `tau_for` by accident rather than by agreement.** The contract's Notes are explicit:
   if `plan()` disagrees with the two hand-tuned Phase 0 instances by more than a factor, **one of
   them is wrong**, and deciding which is the real work. Do not tune `plan()` until it matches and
   call that validation.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `myenv/Scripts/python.exe -m validate.autoconfig` — **Rung B is the gate and M6 is claimed
   only when it prints PASS.**
3. Run `pytest tests/test_autoconfig.py`, then the whole suite.
4. Run `/validate` for every rung at or below this task — all four Phase 0 rungs plus Rung A.
   Nothing may regress.
5. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
