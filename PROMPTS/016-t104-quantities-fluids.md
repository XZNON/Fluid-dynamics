# Session 16 — T104: Physical quantities + fluid library

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator,
spec `DOCS/IDEA2.md`, closed at M4 on 2026-08-13 with all four validation rungs green.

**Phase 1 is live**: the product layer above that solver — a `flow/` package plus a CLI, on a Warp
GPU backend, ten tasks `T101` → `T110`, five validation rungs A–E, milestones M5 → M8. Spec
`DOCS/IDEA3.md`, plan `DOCS/PLAN2.md`.

**The GPU port is finished.** T101 built the backend seam, T102 put the kernels on the device, T103
put the whole timestep there and cleared the performance budget — **M5 was reached on 2026-08-18**.
**This session is the first one that is not solver work.** `T104` is the first `flow/` module, and
from here to M8 the subject is judgement, units and refusals, not kernels.

## Read these first, in this order

1. `CLAUDE.md` — the 16 hard constraints (the Phase 1 list), session protocol, conventions.
   **Constraints 13 and 15 are yours and they start being enforced by tests in this session.**
2. `DOCS/STATE2.md` — **in full**: snapshot, blockers, open questions, environment, performance
   baseline, decisions D-041 … D-057, and the session 13, 14 and 15 log entries.
3. `DOCS/TASKS2.md` § T104 — the task contract, in full. Also read the backlog index row so you know
   what depends on it (T105 does, directly).
4. `DOCS/IDEA3.md` § The five things Phase 1 must get right, **item 1 in full** ("The user never
   types a lattice quantity") · § Scope · § Deliberately deferred.
5. `lbm/units.py` **in full** — the conversion arithmetic already exists and this task is the layer
   *above* it, not a replacement. `lbm/units.py` is the units boundary's inner face; `flow/quantity.py`
   is its outer face.
6. `old-Docs/STATE1.md` **D-031** only — `from_svg` parsing a subset itself rather than taking a
   dependency. It is the precedent your "no new dependency" criterion cites.
7. `DOCS/PLAN2.md` § Dependency graph and § Session map — T104 is session 16 and is the root of the
   T105 → T106 chain.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 15: **T103 is done and M5 is reached.** The whole timestep runs on
  `cuda:0`. The `Backend` protocol now covers allocation, the two host transfers, every kernel, all
  four boundaries and both halves of the Guo body force (**D-054**); `Sim` owns device state and
  exposes `host_f()` / `host_u()` / `host_rho()` / `host_f_bb()` for host reads on frame and probe
  cadence.
- **Rung A is green in full.** Every kernel and every boundary within **5.96e-08** of NumPy against a
  1e-6 bar (`bounce_back`, `moving_wall`, `outlet(copy)`, `macroscopic`, `stream` all **bitwise**);
  whole step **9.611e-06** at 1000 steps against 1e-4, and *not compounding*; a checkpoint written on
  `warp` resumes on `numpy` at **8.196e-06**; restart within a backend is bit-identical.
- **Phase 0 rung status: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩**, printing session 11's digits — R1 L2 0.3650%
  (numpy) / 0.3649% (warp), R2 0.75% and vortex 0.21 cells on both, R3 St 0.1731 and
  Cd 1.4031 ± 0.0086 on both, R4 square Cd 1.5279 ± 0.0271 and polygon Cd 1.4276 ± 0.0226 on `warp`.
  R1, R2 and R3 were re-run on **both** backends; R4 on `numpy` was not (~40 min) — see
  `DOCS/STATE2.md` § session 15 "Not done" for what stands in its place.
- **Phase 1 rung status: A 🟩 · B ⬜ · C ⬜ · D ⬜ · E ⬜.**
- **Performance:** `bench.py --backend warp` prints **4155 / 757 / 441 steps/s** at 40k / 1M / 2M
  cells against floors of 2000 / 250 / 150 — 5× / 33× / 53× NumPy — using 391 MiB of the 4 GB card
  at 2M cells. Measured at 3201 MHz of 3201 MHz on mains, RTX 3050 Laptop GPU, driver 592.82
  (**D-035**).
- **Milestone reached:** **M5** (2026-08-18). M6 is Rung B, two tasks away.
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101, T102, T103.
- `myenv/Scripts/python.exe -m pytest` → **428 passed, 1 skipped** at the end of session 15. (The
  skip is `test_a_known_but_uninstalled_backend_names_its_install_line`, which skips by design now
  that `warp` is installed.)

## Your task this session

**T104 — Physical quantities + fluid library.** One task, this session only.

Run this first:

    /start-task T104

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** `"20 m/s"` becomes `20.0` m/s and `"air"` becomes a kinematic viscosity with a citation.
The first `flow/` module, and the one that makes every later signature physical.

**Outputs:** `flow/__init__.py`; `flow/quantity.py::Quantity`, `::parse` (`str | float → Quantity`),
`::to_si`; `flow/fluids.py::FLUIDS` (name → `Fluid(nu, rho, T, source)`), `::fluid(name) -> Fluid`.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `parse("20 m/s")`, `parse("72 km/h")`, `parse("20")` (with a declared default unit) and `parse(20.0)` all give the same SI value; a table-driven test covers m, cm, mm, in, ft, m/s, km/h, mph, knots, and both `°C`/`K` for temperature.
- [ ] An unparseable or dimensionally wrong string raises `ValueError` naming **what was given, what dimension was expected, and one valid example**. No silent default-unit assumption when a unit is present and wrong.
- [ ] `FLUIDS` has at least air, water, honey, olive oil, glycerine and helium, each with `nu` in m²/s at a stated temperature and a **cited source string**; a test asserts every entry has a non-empty source and a physically ordered `nu` (helium < air < water < oil < glycerine).
- [ ] `fluid("Air")`, `fluid("air")`, `fluid(" air ")` resolve; an unknown name raises listing the known ones.
- [ ] A custom fluid can be given directly as a viscosity (`fluid=Quantity("1.5e-5 m^2/s")`) without touching `FLUIDS`.
- [ ] **No new dependency.** Parsing is ~150 lines, in the spirit of **D-031** (`from_svg` took no dependency for the same reason). If `pint` is adopted instead, that is a recorded decision with the reason.
- [ ] `pytest tests/test_quantity.py tests/test_fluids.py` green; Phase 0 rungs untouched.

### Constraints that bite on this task

From `CLAUDE.md` § Hard constraints, in their Phase 1 form:

- **Constraint 13** — *no lattice quantity in any public `flow/` signature.* No `tau`, no lattice `U`,
  no `steps_per_frame`, no cell counts. **This module is the boundary's outer face**: it speaks
  metres, seconds and m²/s and nothing else, while `lbm/units.py` is the inner face that turns those
  into lattice numbers. Constraint 13's test lands with the code it governs — that is this session.
- **Constraint 15** — *`flow/` may import `lbm/`; `lbm/` may never import `flow/`, and a test asserts
  it.* Write that test **this session**, in the session that creates `flow/`, not later: it is what
  makes the Phase 3 XLB swap a substitution rather than a rewrite (**D-042**). An AST or import scan
  over `lbm/` is the shape `tests/test_backends.py` already uses for "runner imports no kernel".
- **Constraint 2** — `nu = (tau - 0.5)/3` still lives only in `lbm.core.nu_from_tau`. A *fluid's*
  `nu` is a physical viscosity in m²/s and has nothing to do with `tau`; do not let the two names
  meet in this module.
- **Constraint 5** — the ladder is ordered and stays a gate. T104 has no rung of its own (its gate is
  unit tests), but the four Phase 0 rungs must still be green, and nothing in this task should touch
  `lbm/` at all. If it turns out something must, that is a `/new-task`, not scope creep
  (`DOCS/PLAN2.md` § Risks, last row).

### Blockers, open questions and decisions that affect you

**Blockers:** none.

**Open questions:**

- **Q-101** — does `python -m lbm.runner` (the M4 gate command) survive as a working entry point once
  `python -m flow` exists, or become a pointer to it? **T109 decides it, not you** — but do not do
  anything in `flow/` that forecloses either answer.
- **Q-102** — is D-017's documented limit (a thin appendage *fused* to a thick body shares its
  component and is not reported) closable without false-alarming on a plain disc? **T107's**, not
  yours.
- **Q-103 is closed** (**D-056**), by measurement in session 15.

**Decisions that constrain this session:**

- **D-042** — the product layer is a new top-level package, `flow/`, beside `lbm/`. `flow/` may
  import `lbm/`; `lbm/` may never import `flow/`, and a test asserts it. It also keeps two invariants
  that only held in Phase 0 because nothing was allowed to break them: constraint 10 (one `render()`
  — **`flow/` colours nothing**) and the units boundary. **You are creating this package**, so its
  first two files set the tone for the seven tasks after it.
- **D-031** — `lbm.geometry.from_svg` parses a subset of SVG itself rather than taking a dependency,
  and raises an `ImportError` naming the feature and the install line for anything outside that
  subset. Your "no new dependency" criterion cites it: ~150 lines of parsing beats a dependency for
  a bounded problem, **and the honest failure message is part of the deal**.
- **D-045** — the answer to a case the solver cannot represent is: refuse it, explain it in the
  user's units, and offer the nearest runnable case — never substitute silently. Refusals carry
  `reason`, `quantity`, `value`, `limit`, `suggestions`. **T106 builds that machinery**, but your
  `ValueError`s are its first instance: "what was given, what dimension was expected, and one valid
  example" is the same shape, one layer down. Do not invent a second refusal vocabulary.
- **D-044** — Phase 1 ships a library plus a CLI, **no UI**. Nothing in `flow/quantity.py` needs to
  format for a screen.
- **D-054** (session 15) — the `Backend` protocol now covers the whole timestep and `Sim` owns
  backend-allocated state. You will not touch it, but know that `Sim`'s arrays are *backend arrays*
  now: anything reading state does it through `sim.host_f()` / `host_u()` / `host_rho()` /
  `host_f_bb()`, not by indexing `sim.f`. T108 is where `flow/` first meets that.

### Before you start

- **Nothing to install**, and the criterion says keep it that way. `myenv` has numpy 2.4.6,
  matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1, pygame 2.6.1, imageio 2.37.4, imageio-ffmpeg 0.6.0,
  psutil 7.2.2, warp-lang 1.16.0, Python 3.11.15. If you decide `pint` is worth it after all, that is
  a **recorded decision with a reason** in `DOCS/STATE2.md` § Decisions plus a row in § Environment —
  not a quiet `pip install`.
- Confirm the starting point before changing anything: `myenv/Scripts/python.exe -m pytest` should
  print **428 passed, 1 skipped**, and `myenv/Scripts/python.exe -m validate.parity --backend warp`
  should print **PASS**.
- The two cheap Phase 0 rungs are ~1 minute each (`validate.poiseuille`, `validate.cavity --re 100`)
  and both now take `--backend`. R3 is ~3 minutes on `warp --headless`, R4 about 12. **This task
  should not touch `lbm/` at all**, so `git status` showing nothing under `lbm/` is the honest
  argument that the Phase 0 rungs are unaffected — but say so explicitly rather than leaving it
  implied.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no. **This session does not write `flow/autoconfig.py` or `flow/diagnose.py`** — they are T105 and
T106 and they are where the judgement lives; T104 is deliberately the boring, testable half that
they stand on.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run the two tests the contract names — `pytest tests/test_quantity.py tests/test_fluids.py` — and
   then the whole suite.
3. Run the constraint-15 import test and say in the log that it is now enforced rather than
   aspirational.
4. Confirm nothing under `lbm/` moved, and re-run the two cheap Phase 0 rungs anyway.
5. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
