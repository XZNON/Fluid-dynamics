# Session 8 — T008: square cylinder benchmark → Rung 4

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-028), session log. **D-019 (characteristic length `D`), D-026 (periodic sides and the
   24 D span) and D-027 (the low-pass before the frequency estimate) are the three your case setup
   sits on** — Rung 4 is Rung 3's domain with a different mask.
3. `DOCS/TASKS1.md` § T008 — the task contract, in full. Also read § T007's outcome note: it lists
   the three measurement traps Rung 3 fell into, and **all three apply to Rung 4 unchanged**.
4. `DOCS/IDEA2.md` § **Validation ladder** Rung 4 (and Rung 3 above it), § **Geometry from a mask**.
5. `DOCS/PLAN1.md` § Session map and § Risks — T008 is session 8 of 11 and carries no milestone;
   M4 is T011.
6. `validate/cylinder.py` — **read it before writing `validate/polygons.py`.** It already contains
   the domain sizing, the `tau` derivation, the cylinder-only force links, the startup kick, the
   low-pass and the report format. Rung 4 is that script with `regular_polygon` in place of
   `circle`; reuse it rather than reinventing it.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 7: **T007 is `done`**, every acceptance criterion run and green, and
  **M3 was reached**. `lbm/render.py` is new (`render`, `COOLWARM`, `LiveSink`), `lbm/runner.py`
  gained `run(..., per_step=...)`, and `validate/cylinder.py` is Rung 3.
  `myenv/Scripts/python.exe -m validate.cylinder` → **PASS**: St **0.1731**, Cd **1.4031 ± 0.0086**,
  Cl amplitude 0.3915 (27.9% of Cd), peak `|u|` 0.09685, window costs **+2.09%** of steps/s.
  `myenv/Scripts/python.exe -m pytest` → `230 passed`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 ⬜ — **R4 is yours, and it is the last rung.**
- **Milestone:** **M3** reached (2026-08-12). T008 carries no milestone gate of its own.
- **Completed tasks:** T001, T002, T003, T004, T005, T006, T007.

## Your task this session

**T008 — Square cylinder benchmark → Rung 4.** One task, this session only.

Run this first:

    /start-task T008

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: confirm bluff bodies with sharp corners work — a square cylinder at Re 100 with `Cd ≈ 1.5`.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `validate/polygons.py` runs a square cylinder at Re 100 and prints Cd, St, PASS/FAIL.
- [ ] **`Cd` within 1.4–1.6 (ref ~1.5).**
- [ ] A second case — an arbitrary convex polygon — runs to completion without `nan` and reports finite Cd/Cl, no reference value asserted.
- [ ] Corner cells behave: a test asserts no fluid velocity inside the solid (`|u| < 1e-6` on solid cells).
- [ ] Rungs 1–3 re-run and still green.

### Constraints that bite on this task

- **Constraint 1** — bounce-back only. A staircased corner **is** the expected answer; do not reach
  for interpolated or curved boundaries to improve `Cd`. That is why the band is ±0.1 and not ±0.02.
- **Constraint 12** — sharp corners are exactly where a thin-mask warning matters. `check_mask` must
  **pass**, not be suppressed. `Sim` runs it at setup; a warning is a domain to fix, not a line to
  silence.
- **Constraint 5** — the ladder is ordered. Rungs 1–3 must be green before you start and again
  before you finish; R3 takes ~6 minutes, the whole ladder about 10.
- **Constraint 3** — lattice velocity under 0.1. Rung 3 settled at `U = 0.06` with a measured peak of
  **0.09685** — a square blocks more than a disc of the same width, so the peak will be *higher* for
  the same `U`. Check it, and drop `U` if it crosses.
- **Constraint 2** — `nu = U D / Re`, `tau = 0.5 + 3 nu`. `validate/cylinder.py::tau_for` refuses
  `tau <= 0.53` and `U >= 0.1` at setup and names the fix; reuse it.
- **Constraint 6 is lifted** (Rung 3 is green) — but optimisation is **T010**, not this session.
- **Constraint 4** — `(9, ny, nx)`, `float32`, constants only from `lbm/core.py`.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-002** (open) — SVG rasterisation dependency for T009 not chosen. Not yours, not blocking.
- **D-019** — characteristic length `D` is the **cross-stream extent of the object's bounding box**,
  and the blockage denominator is the fluid span. For a square this is the side length, and it is
  what `Cd` is divided by. Do not invent a second definition.
- **D-026** — **Rung 3's lateral boundaries are periodic (`WALL = 0`), and the fluid span is 24 D
  (4.17% blockage).** Measured: one-cell no-slip walls grow a ~34-cell boundary layer over the 8 D
  upstream fetch, which turned a nominal 9.5% blockage into an effective ~13% and took `Cd` from
  1.40 to 1.64; and even 15 D of span (6.35%) read `Cd = 1.4635`, over the band. Constraint 12's 10%
  is a floor on the domain, not a target. **Start Rung 4 from these numbers.**
- **D-027** — the `Cl` series is **low-passed** (Gaussian, `sigma = 0.5 D/U`) before the FFT, and
  only for the frequency; the shedding-amplitude check reads the raw series. The domain's acoustics
  (period ~305 steps, from the startup pulse ringing against the Zou–He inlet) otherwise outvoted
  the wake (period ~2500) and produced `St = 1.49`. `validate/cylinder.py::lowpass` is the code.
- **D-025** — `run(sim, sink, ..., per_step=probe)` samples every timestep on the physics thread.
  `Cl` must be sampled at the step rate; frame-rate sampling aliases.
- **From session 7, the trap most likely to bite you again:** `Sim.links` is built from the **whole**
  mask. Integrate the force over a **body-only** link list (`boundary_links(square)`), or `Cd` will
  include whatever else is in the mask. On the walled Rung 3 domain that read 6.65 against 1.57.
- **D-021** — the outlet is convective at `lam = sqrt(cs2)`; `SimConfig.outlet_lam` is exposed.
  Rung 3 ran at the default and the wake left cleanly.

### Before you start

- **Nothing to install.** `myenv` has numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  pygame 2.6.1, Python 3.11.15. (`imageio[ffmpeg]` lands in T011.)
- `lbm.geometry.regular_polygon` and `lbm.geometry.polygon` already exist and are tested (T004) —
  the square and the arbitrary convex polygon both come from there. No new geometry code should be
  needed.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves.
- Rungs 1–3 must be green before you start and again before you finish. Rung 2's Re 1000 case takes
  ~155 s and Rung 3 ~370 s; budget ~10 minutes for the full ladder.
- Expect `Cd` on the **high** side of the reference: staircased corners over-resist. The contract
  anticipates it. If it lands above 1.6, suspect the domain (blockage, upstream fetch) and the link
  list before suspecting the solver — in that order, because that is the order they went wrong in
  session 7.

## Scope discipline

Work only what's in the contract. Physical units and PNG/SVG masks are **T009**; the performance
pass is **T010**; MP4/GIF sinks and the CLI are **T011**. Something else genuinely needs doing?
`/new-task` it. If it is under `DOCS/IDEA2.md` § Deliberately deferred, the answer is no.

The temptation this session is to "fix" a high `Cd` with a better boundary condition. Constraint 1
forbids it and the ±0.1 band exists because staircasing is the expected physics here.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it. `Cd` in 1.4–1.6 is what earns R4.
2. Run `/validate`. R1, R2, R3 and R4 must all be reported with their measured numbers.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 230 existing tests
   must still pass.
4. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T008 `in_progress`.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes
   `PROMPTS/009-t009-*.md` for the next session. Do not end the session without it.
