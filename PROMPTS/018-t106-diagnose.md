# Session 18 — T106: Diagnosis, refusal, nearest runnable case → Rung D

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator,
spec `DOCS/IDEA2.md`, closed at M4 with all four validation rungs green. **Phase 1 is live**: the
product layer above the solver — `flow/` package plus a CLI, on a Warp GPU backend, spec
`DOCS/IDEA3.md`, plan `DOCS/PLAN2.md`, backlog `DOCS/TASKS2.md`, live status `DOCS/STATE2.md`.

## Read these first, in this order

1. `CLAUDE.md` — the 16 hard constraints (Phase 1 list), session protocol, conventions.
2. `DOCS/STATE2.md` — **in full**: snapshot, blockers, open questions, decisions D-041 … D-060, and
   at minimum the session 16 and 17 log entries.
3. `DOCS/TASKS2.md` § T106 — the task contract, in full. Also the backlog index row: **T108 depends
   on you** (alongside T105 and T107).
4. `DOCS/IDEA3.md` § The five things Phase 1 must get right, **item 2 in full** ("Refusal is a
   feature, and it comes with a way forward") · § Validation ladder (the Rung D row).
5. `DOCS/IDEA2.md` § Stability — the symptom/cause/fix table. This is the seed for `Monitor`'s three
   failure modes.
6. `old-Docs/STATE1.md` **D-025** (the `per_step` probe hook), **D-032**, **D-038**, **D-029** — read
   the entry each is cited for, not the whole file.
7. `flow/autoconfig.py` (T105, last session) — `Unrepresentable` is what you diagnose. Read its
   docstring and every `raise Unrepresentable(...)` site: each already carries `reason`, `quantity`,
   `value`, `limit`, `suggestions` (a list of `Suggestion(change, value, note)`), and the tau-floor
   path already builds a real `Suggestion` for the Re-2e6-style case. You are not inventing the
   structured data — you are turning it into prose and proving the suggestions work.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 17: **T105 is done — M6 reached.** `flow/autoconfig.py` exists:
  `Plan`, `plan()`, `Suggestion`, `Unrepresentable`. Every guardrail (tau floors, `u_lattice <0.1`
  with 1.8x headroom, blockage <10%, ≥8D downstream, min thickness ≥3 cells) is enforced at plan
  time and cites its decision. `pytest` prints **565 passed, 1 skipped** (18 new).
- **Rung B is green in full**: `validate/autoconfig.py` — the accuracy check predicted 61.05s against
  an actual 63.75s (4.2% error, limit 25%), and the 24-case sweep (2 fluids x 2 speeds x 2 sizes x 3
  quality) passed all 24: every guardrail held on the *rasterised* geometry, no `nan`, worst peak
  `|u|` 0.0695, worst `Re` reproduction error 0.0000% (limit 0.1%). Whole rung ran in ~23 minutes.
- **Phase 1 rung status: A 🟩 · B 🟩 · C ⬜ · D ⬜ · E ⬜.** Rung A unchanged this session (green in
  full, worst 5.96e-08 kernels/boundaries, 9.611e-06 whole-step, 8.196e-06 cross-backend checkpoint).
- **Phase 0 rung status: R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩** — all four re-run this session on
  `--backend warp`, printing session 11/15's digits exactly: R1 L2 0.3650%, R2 0.75% / 0.21 cells, R3
  St 0.1731 Cd 1.4031 ± 0.0086, R4 square PASS and polygon Cd 1.4276 ± 0.0226.
- **Milestone reached: M6** (2026-08-19). **M7 is T107's**, not this session's — T106 has no
  milestone of its own; it is a gate inside itself (Rung D) and is re-run by M8.
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101, T102, T103, T104, T105.

## Your task this session

**T106 — Diagnosis, refusal, nearest runnable case.** One task, this session only. Gate: **Rung D**.

Run this first:

    /start-task T106

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** a case the tool cannot run produces a plain-language explanation and a concrete alternative
that is **tested to work**. A case that goes unstable mid-run is caught early and explained, not
`nan`.

**Inputs:** an `Unrepresentable` from `flow.autoconfig`, or a live `Sim` during a run.
**Outputs:** `flow/diagnose.py::explain(exc) -> str`, `::suggest(request) -> list[Suggestion]`,
`::Monitor` (a `per_step`-compatible probe, **D-025**) raising `Diverging` with a cause;
`validate/refusals.py` printing PASS/FAIL.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `explain()` output contains **no lattice quantity** — no `tau`, no lattice `U`, no cell counts — in its first paragraph; the numbers are available in a second "details" section. Asserted by a test that greps the first paragraph.
- [ ] `suggest()` returns at least one `Suggestion` for each refusal class, each carrying a modified request plus one sentence on what it changes physically (slower, smaller, more viscous fluid, or "the same shape at a Reynolds number we can represent — **not your case**").
- [ ] **Rung D — `validate/refusals.py`:** for every refusal class, take the tool's own top suggestion, feed it back through `plan()`, and **run 2000 steps**. PASS requires every suggestion to produce a case that plans and runs without `nan`. A suggestion that does not fix its case is a failing test.
- [ ] The Re-2e6 case from **D-038** (air, 20 m/s, 1.5 m) is a named case in the rung, with its full user-facing output pinned in the test as a golden string, so a later reword is a deliberate edit.
- [ ] `Monitor` detects divergence **before** `nan`: on a deliberately under-resolved case, it raises within 10% of the steps the run would have taken to produce `nan`, naming the cause and the fix. Measured on at least three failure modes from `DOCS/IDEA2.md` § Stability — `tau` too near 0.5, peak `|u|` crossing 0.1, and a mass-drift blow-up.
- [ ] `Monitor` costs **under 2%** of steps/s, measured with the sim otherwise identical, quoted with CPU clock (**D-035**).
- [ ] **Never a silent substitution** (Phase 1 constraint 16): a test asserts that a `Result` produced from a suggestion carries `substituted=True` and that the flag reaches the printed summary and the recorded video's metadata.
- [ ] `pytest tests/test_diagnose.py` green.

**A note on the `Result` criterion above** — `flow.Result` / `flow.report` do not exist yet (T108,
not this session). If the literal `Result`-carries-`substituted=True` check cannot be built against
real code, that is a real dependency gap in the contract's ordering, not something to paper over:
raise it explicitly at `/start-task T106` confirmation time rather than silently reinterpreting the
criterion or silently building a `Result` stub that T108 then has to reconcile with. Whatever gets
decided, log it in `DOCS/STATE2.md` § Decisions per `CLAUDE.md` § Session protocol (a spec conflict
is logged, not silently resolved) — same pattern as this session's **D-058**.

### Constraints that bite on this task

- **D-045** — refuse, explain, offer; never run something else and call it the answer. This is the
  policy `flow.diagnose` turns into working code.
- Constraint 9's spirit — a wrong answer that looks plausible is the failure mode. A *substituted*
  answer that looks like the requested one is the same failure with a friendlier face.
- Constraint 8 — `Monitor` runs on the physics thread through `per_step` (**D-025**); keep it
  arithmetic-cheap and sampled, not per-cell every step.
- Constraint 16 — no silent substitution, carried all the way to the printed summary and the video
  metadata.

### Blockers, open questions and decisions that affect you

**Blockers:** none.

**Open questions:**

- **Q-101** — does `python -m lbm.runner` survive as a working entry point once `python -m flow`
  exists? T109's, not yours.
- **Q-102** — is D-017's documented limit (a thin appendage fused to a thick body shares its
  component and is not reported) closable without false-alarming on a plain disc? T107's, not yours.

**Decisions that constrain this session:**

- **D-045** — the refusal policy in full: refuse, explain in the user's units, offer the nearest
  runnable case, never substitute silently. Suggestions are testable claims.
- **D-032 / D-029 / D-036** — the three `tau` floors. `flow.autoconfig` always applies D-029's 0.54
  (this session's **D-059**), so every `Unrepresentable` you diagnose that came from the tau guardrail
  will cite D-029 in its `reason` — `explain()` can rely on that rather than re-deriving which floor
  applied.
- **D-059** — `flow.autoconfig`'s constants (`TAU_FLOOR`, `QUALITY_CELLS`, `SPAN_D`/`UPSTREAM_D`/
  `DOWNSTREAM_D`, `RUN_CONVECTIVE_TIMES`) and the measured Rung B cost (~23 minutes). Relevant to you
  because Rung D's 2000-step-per-suggestion runs are on top of Rung B's own ~23 minutes — size Rung D
  so the two stay minutes, not hours, together.
- **D-060** — `tests/test_flow_package.py`'s constraint-13 scan now exempts a frozen dataclass's
  auto-generated constructor (an *output* record) but still scans everything else, including
  `flow.diagnose`'s functions and any non-frozen or hand-written class you add. If `Diverging` or
  `Suggestion`-adjacent types trip the scan, the fix is the same precision, not a weakened assertion.

### Before you start

- **Nothing to install.** `myenv` is unchanged from session 17 — nothing was added.
- Confirm the starting point: `myenv/Scripts/python.exe -m pytest` should print **565 passed,
  1 skipped**, and `myenv/Scripts/python.exe -m validate.autoconfig` should print **PASS**.
- Rung timings for planning the session: Rung B is ~23 minutes if you re-run it; Rung D's own budget
  is per-refusal-class x 2000 steps, which should be much cheaper — say so if it isn't.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no. This session does not write `flow/prepare.py` (T107) or `flow/case.py` / `flow/report.py`
(T108) — if T106's `Result`/`substituted` criterion genuinely needs either, that is the dependency
gap to raise at confirmation time (see the note under Acceptance criteria above), not scope to quietly
absorb.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `myenv/Scripts/python.exe -m validate.refusals` — **Rung D is the gate.**
3. Run `pytest tests/test_diagnose.py`, then the whole suite.
4. Run `/validate` for every rung at or below this task — all four Phase 0 rungs, Rung A, Rung B.
   Nothing may regress.
5. **Run `/checkpoint`** — it updates `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
