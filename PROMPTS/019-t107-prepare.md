# Session 19 — T107: Geometry preparation + shape corpus → Rung C, **M7**

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator in
pure NumPy, spec `DOCS/IDEA2.md`, closed at M4 with all four validation rungs green. **Phase 1 is
live**: the product layer above the solver — a `flow/` package plus a CLI, on a Warp GPU backend.
Spec `DOCS/IDEA3.md`, plan `DOCS/PLAN2.md`, backlog `DOCS/TASKS2.md`, live status `DOCS/STATE2.md`.

## Read these first, in this order

1. `CLAUDE.md` — the 16 hard constraints (Phase 1 list), session protocol, conventions.
2. `DOCS/STATE2.md` — **in full**: snapshot, blockers, open questions, decisions D-041 … D-063, and
   at minimum the session 17 and 18 log entries.
3. `DOCS/TASKS2.md` § **T107** — the task contract, in full. Also the backlog index row: **T108
   depends on you**, and you are the last of its three dependencies still open.
4. `DOCS/IDEA3.md` § The five things Phase 1 must get right, **item 3 in full** ("Real shapes are
   not convex blobs") · § Validation ladder (the Rung C row).
5. `DOCS/PLAN2.md` § Session map (you are session 19) and § Risks — the row *"Shape repair silently
   changes the user's geometry"* is aimed at this task by name.
6. `old-Docs/STATE1.md` **D-017**, **D-018**, **D-019**, **D-031**, **D-040** — read the entry each
   is cited for, not the whole file. **D-017 is the one that matters most**: it is the metric you
   are asked to either improve or restate with evidence.
7. The code you are building on, not forking: `lbm/geometry.py` — `min_thickness`,
   `strip_solid_border`, `bounding_box`, `check_mask`, `from_png`, `from_svg`, `circle`, `polygon` —
   and `lbm/runner.py::_body_mask`, which already implements D-040's rescale-until-measured loop.
8. `flow/diagnose.py` (T106, last session) — `explain` / `suggest` / `apply_suggestion` /
   `REFUSAL_CLASSES`. Your `verdict="refused"` cases should reach a user the same way T106's do.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 18: **T106 is done.** `flow/diagnose.py` exists — `explain()`,
  `suggest()`, `apply_suggestion()`, `classify()`, `Monitor`, `Diverging` — and
  `validate/refusals.py` is **Rung D, green**: every refusal class's own top suggestion plans and
  runs 2000 steps clean, `Monitor` catches all three `DOCS/IDEA2.md` § Stability failure modes
  before `nan` with the cause named, and costs under 2%. `pytest` prints **610 passed, 1 skipped**.
- **Phase 1 rung status: A 🟩 · B 🟩 · C ⬜ (yours) · D 🟩 · E ⬜.**
- **Phase 0 rung status: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩** — all four re-run in session 18, printing
  session 11/15's digits: R1 L2 0.3650%, R2 0.75% / 0.21 cells, R3 St 0.1731 Cd 1.4031 ± 0.0086,
  R4 square Cd 1.5279 ± 0.0271 and polygon Cd 1.4276 ± 0.0226.
- **Milestone reached: M6** (2026-08-19). **M7 is yours** — `DOCS/PLAN2.md` § Milestone gates:
  *"`myenv/Scripts/python.exe -m validate.shapes` prints PASS: every corpus image gets its committed
  verdict and its measured properties, with no manual step."*
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101, T102, T103, T104, T105, **T106**.

## Your task this session

**T107 — Geometry preparation + shape corpus.** One task, this session only. Gate: **Rung C**.
Milestone: **M7**.

Run this first:

    /start-task T107

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** real user pictures — not convex blobs — become masks the solver can run, or get refused
with a reason.

**Inputs:** a PNG/SVG path or a bool array, plus a target body resolution.
**Outputs:** `flow/prepare.py::prepare(source, cells_across, *, repair=...) -> Prepared`
(`mask: NDArray[np.bool_] (ny, nx)`, `verdict: "ok" | "repaired" | "refused"`, `actions: list[str]`,
`properties: dict`); a `tests/data/shapes/` corpus with `expectations.json`; `validate/shapes.py`
printing PASS/FAIL.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] **The corpus is committed**: at least 12 images covering hairline appendage fused to a thick body, detached specks, interior hole (donut), unclosed outline, heavy anti-aliasing, huge margin, tiny body, extreme aspect ratio, diagonal 1-cell line, self-touching shape, all-white, all-black. Each has a committed expected verdict and expected properties.
- [ ] **D-017's documented limit is closed or restated with evidence:** a thin appendage fused to a thick body is *detected* (it currently shares the component and is not reported). If a metric that catches it also false-alarms on a plain disc, the measurement is recorded and the limit stays — but it is measured this session, not inherited.
- [ ] Repairs are individually switchable and individually reported in `actions`: fill interior holes, drop specks below a stated area, thicken sub-3-cell features, keep the largest component. Nothing is repaired silently.
- [ ] After repair, the mask satisfies constraint 12's rules — min thickness ≥ 3, and the checks in `check_mask` produce no warning — asserted for every corpus image whose verdict is `repaired`.
- [ ] Refusal cases refuse: an all-white, an all-black and a body smaller than 8 cells across return `verdict="refused"` with a reason naming what is wrong and what would fix it.
- [ ] **Rung C — `validate/shapes.py`:** every corpus image, no manual step, verdict and measured properties compared against `expectations.json`. PASS/FAIL printed, with a per-image line.
- [ ] Body resolution obeys **D-040**: the *measured* body is the requested size after preparation, within 1 cell, for every corpus image that is not refused.
- [ ] `pytest tests/test_prepare.py` green; Phase 0 rungs still green.

### Constraints that bite on this task

- **Constraint 12** — geometry is one boolean array `solid` `(ny, nx)`; solid at least 3 cells
  thick, object ≥8 diameters from the outlet, blockage under ~10%. **This is the module that makes
  constraint 12 true rather than merely warned about** — Phase 1 *repairs* where it can.
- **Constraint 13** — no lattice quantity in any public `flow/` signature. `cells_across` is the
  body's resolution in the D-040 sense and is the one number that may cross; `nx`, `ny`, `dx`, `dt`,
  `tau`, `cells_per_length` may not. `tests/test_flow_package.py` scans every public callable
  reachable through each module's `__all__`, and **D-060** limits the frozen-dataclass exemption to
  auto-generated constructors — a hand-written `__init__` is still scanned. If `Prepared` is a
  frozen dataclass returned as a *result*, it is exempt; anything else is not.
- **Constraint 15** — `flow/` may import `lbm/`; `lbm/` may never import `flow/`, and an AST scan
  plus a runtime scan assert it. `flow/prepare.py` builds **on top of** `lbm/geometry.py` and does
  not fork `check_mask` or `min_thickness`.
- **Constraint 16 / D-045** — no silent substitution. A repair *is* a substitution of the user's
  geometry, so every repair is named in `actions`, and a refusal names a fix. T106's
  `flow/diagnose.py` is the pattern to follow, not to reinvent.
- **Constraint 5** — the ladder is ordered. Rung C does not start while a rung below it is red, and
  all of R1–R4, A, B and D are currently green.
- **Constraint 10** — `flow/` colours nothing. A corpus image is data, not a rendering.

### Blockers, open questions and decisions that affect you

**Blockers:** none.

**Open questions:**

- **Q-102 — this one is yours.** *"Is D-017's documented limit (a thin appendage fused to a thick
  body shares its component and is not reported) closable without false-alarming on a plain disc?
  T107 must measure this, not inherit it. If no metric clears both, the limit stays with the
  measurement recorded."* Closing it or restating it with evidence is an acceptance criterion.
- **Q-101** — does `python -m lbm.runner` survive as a working entry point once `python -m flow`
  exists? T109's, not yours.

**Decisions that constrain this session:**

- **D-017** — `min_thickness` is measured **per 8-connected component** as `2 * max(d) - 1`, where
  `d` is the Chebyshev distance from a solid cell to the nearest fluid cell, reported as the minimum
  over components. **The two obvious alternatives were written first and both false-alarm on a plain
  cylinder** — run lengths (a disc's topmost cell has a vertical run of 1) and per-cell 3x3 opening
  (digitised curvature always produces a pole no fully-solid 3x3 covers). Any new metric you propose
  has to clear that same bar, and the disc false-alarm test is the one that kills candidates.
- **D-018** — domain borders are exempt from all three mask checks; `check_mask` peels *entirely*
  solid edge rows and columns first. Without the exemption Rung 1's own mask warns, and a check that
  cries wolf on the project's own passing benchmarks gets suppressed.
- **D-040** — `--resolution N` is N cells across the **body**, not the picture, and
  `lbm/runner.py::_body_mask` already rescales (at most three passes) until the *measured* body
  matches. Measured on `tests/data/test_body.png`: rasterised into a 30-row box the body is **18**
  cells, which runs at `tau = 0.527` while the summary claims 30 — and 0.527 is inside the band
  D-029 measured a disc dying in. **A repair that changes the body's extent must re-derive, not
  adjust.**
- **D-031** — `from_svg` handles `M/L/H/V/C/Q/Z` and `<polygon>`; arcs, smooth shorthands and
  `transform` raise `ImportError` naming `cairosvg`. Do not add a rasteriser dependency; a corpus
  image that needs one is the wrong corpus image.
- **D-063** (session 18) — a suggestion is a testable claim, and two of T105's were repaired when
  they turned out not to fix their own case. Your refusals carry the same burden: whatever
  `verdict="refused"` names as the fix should be checkable, and Rung C is where that gets checked.
- **D-059** — Rung B costs ~23 minutes and Rung D ~9. **Size Rung C so the three together stay
  minutes.** A corpus rung should be seconds, not minutes — it is image processing, not simulation.

### Before you start

- **Nothing to install.** `myenv` is unchanged since session 14 (`warp-lang` was the last addition).
  If you find you want `scipy` for connected components or a distance transform, that is a real
  decision: `DOCS/STATE2.md` § Environment needs a row **in the same session**, and `lbm/geometry.py`
  got its component labelling without it — check what is already there before adding a dependency to
  a project whose stated point is pure NumPy (**D-031**'s reasoning).
- Confirm the starting point: `myenv/Scripts/python.exe -m pytest` should print **610 passed,
  1 skipped**, and `myenv/Scripts/python.exe -m validate.refusals` should print **PASS**.
- **Generate the corpus programmatically where possible** and commit the generator alongside the
  images, so they are auditable and small (the T107 Notes ask for this).
- Watch `.gitignore`: session 16 found it drops `*/__init__.py` and `tools/` (queued issue
  `495777c58269`, still open). If you add `flow/prepare.py` or a `tests/data/shapes/` tree, run
  `git status` and confirm they are actually tracked — `flow/__init__.py` needed `git add -f`.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no. This session does not write `flow/case.py` / `flow/report.py` (T108) or the CLI (T109). A change
`lbm/geometry.py` genuinely needs is a `/new-task` naming the Phase 0 rung it must re-prove
(`DOCS/PLAN2.md` § Risks, last row) — never folded into a product task.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `myenv/Scripts/python.exe -m validate.shapes` — **Rung C is the gate, and M7 is claimed only
   when it prints PASS.**
3. Run `pytest tests/test_prepare.py`, then the whole suite.
4. Run `/validate` for every rung at or below this task — all four Phase 0 rungs, Rung A, Rung B,
   Rung D. Nothing may regress.
5. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
