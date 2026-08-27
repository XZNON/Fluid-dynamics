# TASKS2.md — Phase 1 task contracts

One task per session. Plan and ordering rationale: `DOCS/PLAN2.md`. Live status: `DOCS/STATE2.md`.
Phase 0's closed backlog is `old-Docs/TASKS1.md` (T001 → T011) — read it, never edit it.

**Status vocabulary:** `not_started` · `in_progress` · `blocked` · `done`
A task is `done` only when **every** acceptance criterion is checked. Code written ≠ done.

**Numbering:** Phase 1 tasks are `T1xx` so they can never collide with Phase 0's `T0xx`.

---

## Backlog index

| ID | Title | Status | Depends on | Gate |
|---|---|---|---|---|
| T101 | Backend seam, NumPy behind it | `done` | — | Phase 0 rungs 1–4 |
| T102 | Warp kernels: equilibrium, collide, stream | `done` | T101 | **Rung A** (kernels) |
| T103 | Warp boundaries, checkpoint, performance | `done` | T102 | **Rung A** (full) → **M5** |
| T104 | Physical quantities + fluid library | `done` | — | unit tests |
| T105 | Auto-configuration | `done` | T104 | **Rung B** → **M6** |
| T106 | Diagnosis, refusal, nearest runnable case | `done` | T105 | **Rung D** |
| T107 | Geometry preparation + shape corpus | `done` | — | **Rung C** → **M7** |
| T108 | `flow.Case` / `flow.Result` API | `done` | T105, T106, T107 | unit tests |
| T109 | CLI on `flow`, live + record wiring | `not_started` | T108 | manual gate + tests |
| T110 | The minute: end to end, timed | `not_started` | T109, T103 | **Rung E** → **M8** |

---

## T101 — Backend seam, NumPy behind it

**Status:** `done` — session 13, 2026-08-18. All four Phase 0 rungs re-printed session 11's digits.

### Goal

`Sim` stops calling `lbm.core` directly and calls a **backend** instead. Only the NumPy backend
exists at the end of this session, and every Phase 0 rung prints the same numbers it printed in
session 11. Nothing gets faster; the seam is the deliverable.

### Reads / depends on

- `DOCS/IDEA3.md` § What Phase 1 is, concretely / § Performance budget
- `old-Docs/STATE1.md` **D-033** (the fused path), **D-020** (which snapshots `forces` takes),
  **D-011** (timestep order), **D-022** (what the checkpoint contains)
- Tasks: none (first Phase 1 task)

### Inputs / outputs

**In:** the working solver.
**Out:** `lbm/backends/__init__.py` with a `Backend` protocol; `lbm/backends/numpy_backend.py`
implementing it over the existing `lbm.core` functions; `SimConfig.backend: str = "numpy"`;
`lbm/backends/registry.py` mapping a name to an implementation and raising a message naming the
install line for an unavailable one.

The protocol covers, at minimum:
`equilibrium(rho, u, feq, work)`, `collide(f, feq, tau)`, `stream(f, buf)`,
`collide_stream(f, feq, tau, solid, f_bb, buf)`, `macroscopic(f, rho, u)`,
`bounce_back(f, f_pre, solid)`, `to_host(f) -> NDArray[np.float32] (9, ny, nx)`,
`from_host(arr) -> backend array`.

### Acceptance criteria

- [x] `Backend` is a `typing.Protocol` (or ABC) with every method above, each documented with its array shapes; `lbm/core.py`'s functions are unchanged and the NumPy backend delegates to them.
- [x] `SimConfig(backend="numpy")` is the default and `Sim` reaches every kernel through `self.backend`; a test asserts no module in `lbm/runner.py` imports `lbm.core`'s kernels directly any more (import-level assertion, not a comment).
- [x] `to_host` / `from_host` round-trip a `(9, ny, nx)` `float32` array bit-identically on the NumPy backend; asserted with `np.array_equal`.
- [x] An unknown backend name raises `ValueError` naming the requested backend and listing the available ones.
- [x] **Restart is still bit-identical** — T006's test passes unchanged, plus a new one that checkpoints and resumes through the seam.
- [x] `myenv/Scripts/python.exe -m pytest` green with **no existing test modified**.
- [x] **All four Phase 0 rungs re-run and print numbers identical to session 11 to every printed digit** — R1 L2 0.3650% · R2 0.75% / 0.42% / 1.01% · R3 St 0.1731, Cd 1.4031 · R4 square Cd 1.5279, polygon Cd 1.4276.

### Constraints that bite here

- Constraint 4 (Phase 1 form) — the backend owns its state layout, but `to_host` must produce `(9, ny, nx)` `float32`. That is what makes checkpoints and parity portable.
- Constraint 11 — bit-identical restart within a backend. If the seam changes float ordering anywhere, the seam is wrong, not the test.
- `DOCS/IDEA3.md` § Scope — **no new physics**. This task adds indirection and nothing else.

### Notes

Resist making the protocol general enough for XLB "while we're here". Two implementations is the
number that reveals the right seam; one plus a guess is not. The Guo body force stays on the unfused
path (D-033) and the protocol should not pretend otherwise.

---

## T102 — Warp kernels: equilibrium, collide, stream

**Status:** `done` (session 14, 2026-08-18)

### Goal

The three hot kernels run on the GPU and produce what NumPy produces. `equilibrium` is over half the
step at 1M cells (`old-Docs/STATE1.md` § Performance baseline), so it is written first.

### Reads / depends on

- `DOCS/IDEA3.md` § Performance budget / § Validation ladder Rung A
- `old-Docs/STATE1.md` § Performance baseline, **D-035** (how to measure), **D-008** (`usq` hoisting)
- Tasks: T101

### Inputs / outputs

**In:** the `Backend` protocol from T101.
**Out:** `lbm/backends/warp_backend.py` implementing `equilibrium`, `collide`, `stream`,
`macroscopic`, `to_host`, `from_host`; `validate/parity.py` with a `--kernels` mode printing
PASS/FAIL per kernel.

### Acceptance criteria

- [x] `warp` installed into `myenv` and recorded in `DOCS/STATE2.md` § Environment with its version and the CUDA/driver version it found.
- [x] The nine D2Q9 constants come from `lbm/core.py` and are uploaded to the device once at backend construction — **not redefined in a Warp kernel** (constraint 4 / "no physics constant twice").
- [x] `validate/parity.py --kernels` compares each of `equilibrium`, `collide`, `stream`, `macroscopic` against the NumPy backend on random `rho ∈ [0.9, 1.1]`, `|u| ≤ 0.099`, at 3 grid sizes, and prints the max absolute difference per kernel.
- [x] **Per-kernel agreement: max absolute difference ≤ 1e-6 in `f` units** — and the script prints the number, so a later regression is visible rather than merely passing. A difference that is not explainable by float ordering fails the task; do not widen the tolerance.
- [x] `stream` is verified independently of parity by the Phase 0 spike test: a single-cell spike lands one cell along `E[i]`, for all 9 directions, on the GPU.
- [x] No allocation per call: the backend preallocates its device buffers at construction; a test runs 1000 steps' worth of kernel calls and asserts device memory is flat.
- [x] `myenv/Scripts/python.exe -m pytest` green; Phase 0 rungs unaffected (they still run `--backend numpy`).

### Constraints that bite here

- **No new physics** — the arithmetic is a transcription of `lbm/core.py`, term for term. If a kernel is "clearer" written differently, write it the same and note the difference.
- Constraint 2 — `nu = (tau - 0.5)/3` still lives in `nu_from_tau`; the kernel takes `omega` and does not re-derive it.
- `DOCS/PLAN2.md` § Risks — if Warp will not install or run in the first half of the session, log the blocker and fall through to T104 rather than burning the session.

### Notes

Bit-identical GPU/CPU agreement is not achievable and is not the goal; **explainable** agreement is.
Where a fused multiply-add changes the last bits, say so with the measured magnitude. The parity
script is the deliverable that outlives this task.

**Outcome (session 14).** Every criterion run. `warp-lang` **1.16.0**, CUDA Toolkit 12.9 / Driver
13.1, `cuda:0` = RTX 3050 Laptop GPU. Measured worst-case difference **5.96e-08** in `f` units
against the 1e-6 bar: `macroscopic` and `stream` **bitwise identical**, `collide` 1.49e-08,
`equilibrium` 5.96e-08 — one fused multiply-add each, magnitudes recorded in **D-053**. Spike test
9/9 on the GPU. `pytest` **408 passed, 1 skipped**. The backend takes host arrays at its boundary and
owns preallocated device buffers per grid shape (**D-052**); moving the state onto the device, and
therefore any speed number at all, is T103's.

---

## T103 — Warp boundaries, checkpoint, performance → M5

**Status:** `done` (session 15, 2026-08-18)

### Goal

The whole timestep runs on the GPU, the four Phase 0 rungs pass on it, and the performance budget is
met. **M5.**

### Reads / depends on

- `DOCS/IDEA3.md` § Performance budget
- `old-Docs/STATE1.md` **D-011**, **D-020**, **D-021**, **D-022**, **D-033**, **D-035**
- Tasks: T102

### Inputs / outputs

**In:** the Warp backend's kernels.
**Out:** `bounce_back`, `moving_wall`, `inlet_velocity`, `outlet_zero_gradient` and the fused
`collide_stream` on the Warp backend; `bench.py --backend warp`; `validate/parity.py` full mode;
`--backend` flag on all four Phase 0 rung scripts.

### Acceptance criteria

- [x] Every boundary condition Phase 0 ships runs on the GPU, and `validate/parity.py` compares each separately against NumPy at ≤ 1e-6 in `f` units.
- [x] **Whole-step parity:** starting from identical state, 1000 steps on each backend agree to `max|Δu| / U < 1e-4`, printed. The number is expected to grow with step count — the script prints it at 10 / 100 / 1000 steps so the growth rate is visible and not merely bounded.
- [x] `save_checkpoint` on the GPU backend writes the same four things plus `format` (**D-022**), via `to_host`; a checkpoint written on `warp` **resumes on `numpy`** and continues within the whole-step parity tolerance. Within a backend, restart stays **bit-identical**.
- [x] **All four Phase 0 rungs pass with `--backend warp`, inside their published bands** — R1 L2 < 1%, R2 max deviation < 5%, R3 St 0.155–0.175 and Cd 1.25–1.45, R4 Cd 1.4–1.6. Bands are not widened; the printed numbers are recorded beside session 11's.
- [x] **`bench.py --backend warp` clears ≥2000 / ≥250 / ≥150 steps/s at 40k / 1M / 2M cells**, measured by alternating rounds (**D-035**), with GPU name, driver, CPU clock and power state quoted.
- [x] The GPU memory footprint at 2M cells is printed and fits the 4 GB card with room for the display path.
- [x] `myenv/Scripts/python.exe -m pytest` green; **M5 recorded in `DOCS/STATE2.md` with the gate output pasted in.**

### Constraints that bite here

- Constraint 5 — Rung A is the gate. A rung that fails on GPU blocks T104 onward; it is a `DOCS/STATE2.md` § Blockers entry, not a queued issue.
- Constraint 11 — bit-identical *within* a backend; cross-backend is the tolerance above, and that distinction gets written into `CLAUDE.md`.
- Constraint 8 — the live path must still not block the physics. Device-to-host transfer for a frame happens on the frame cadence, not the step cadence.

### Notes

R4 alone is ~40 minutes on NumPy; on GPU it should be minutes, which makes the full ladder cheap for
the first time. Do not use that to justify running it less carefully.

**Outcome (session 15).** Every criterion run. The seam had to widen first (**D-054**, superseding
**D-052**): allocation, the three remaining boundaries and both halves of the Guo body force joined
the `Backend` protocol, backend arrays became opaque handles, and `Sim` now owns **device** state
with host reads through `host_f` / `host_u` / `host_rho` / `host_f_bb` on frame and probe cadence.
**Rung A is green in full** — every kernel *and every boundary* within **5.96e-08** of NumPy against
the 1e-6 bar, with `bounce_back`, `moving_wall`, `outlet(copy)`, `macroscopic` and `stream` all
**bitwise**; whole-step **9.611e-06** at 1000 steps against 1e-4, and *not compounding* (2.459e-06 at
10, 1.743e-05 at 100), which closes **Q-103** as **D-056**. A checkpoint written on `warp` resumes on
`numpy` and continues at **8.196e-06**; restart within `warp` is bit-identical. All four Phase 0
rungs pass with `--backend warp` printing session 11's digits — R1 L2 **0.3649%**, R2 **0.75%** /
**0.21** cells, R3 St **0.1731** Cd **1.4031 ± 0.0086**, R4 square Cd **1.5279 ± 0.0271** and polygon
Cd **1.4276 ± 0.0226**. `bench.py --backend warp` clears every floor with margin —
**4155 / 757 / 441 steps/s** at 40k / 1M / 2M against 2000 / 250 / 150, which is **5x / 33x / 53x**
NumPy measured in the same alternating rounds — and 1M and 2M clear the budget's *targets* too. The
2M footprint is **391 MiB in 13 Sim-owned arrays**, leaving 2882 MiB of the card's 4096 MiB free.
`pytest` **428 passed, 1 skipped**. **M5.**

---

## T104 — Physical quantities + fluid library

**Status:** `done` — session 16, 2026-08-19. `pytest` **547 passed, 1 skipped** (119 new); constraints 13 and 15 enforced by test; `lbm/` untouched and R1/R2 re-run green.

### Goal

`"20 m/s"` becomes `20.0` m/s and `"air"` becomes a kinematic viscosity with a citation. The first
`flow/` module, and the one that makes every later signature physical.

### Reads / depends on

- `DOCS/IDEA3.md` § The five things Phase 1 must get right (1)
- `lbm/units.py` (the conversion arithmetic already exists — this is the layer above it)
- Tasks: none

### Inputs / outputs

**In:** user strings and numbers.
**Out:** `flow/__init__.py`; `flow/quantity.py::Quantity`, `::parse` (`str | float → Quantity`),
`::to_si`; `flow/fluids.py::FLUIDS` (name → `Fluid(nu, rho, T, source)`), `::fluid(name) -> Fluid`.

### Acceptance criteria

- [x] `parse("20 m/s")`, `parse("72 km/h")`, `parse("20")` (with a declared default unit) and `parse(20.0)` all give the same SI value; a table-driven test covers m, cm, mm, in, ft, m/s, km/h, mph, knots, and both `°C`/`K` for temperature.
- [x] An unparseable or dimensionally wrong string raises `ValueError` naming **what was given, what dimension was expected, and one valid example**. No silent default-unit assumption when a unit is present and wrong.
- [x] `FLUIDS` has at least air, water, honey, olive oil, glycerine and helium, each with `nu` in m²/s at a stated temperature and a **cited source string**; a test asserts every entry has a non-empty source and a physically ordered `nu` (helium < air < water < oil < glycerine).
- [x] `fluid("Air")`, `fluid("air")`, `fluid(" air ")` resolve; an unknown name raises listing the known ones.
- [x] A custom fluid can be given directly as a viscosity (`fluid=Quantity("1.5e-5 m^2/s")`) without touching `FLUIDS`.
- [x] **No new dependency.** Parsing is ~150 lines, in the spirit of **D-031** (`from_svg` took no dependency for the same reason). If `pint` is adopted instead, that is a recorded decision with the reason.
- [x] `pytest tests/test_quantity.py tests/test_fluids.py` green; Phase 0 rungs untouched.

### Deviation recorded

The third criterion's parenthetical — `nu` ordered `helium < air < water < oil < glycerine` — is not
physical and the library does not reproduce it (**D-058**). Measured ascending order at 20 °C is
**water 1.004e-6 < air 1.516e-5 < olive oil 8.4e-5 < helium 1.178e-4 < glycerine 1.120e-3 <
honey 7.042e-3 m²/s**. The criterion's intent is kept and strengthened: the ordering test asserts the
measured order, every entry's `nu` is checked against its independently cited `mu` and `rho`
(`nu = mu / rho`, to 0.2%), and a second test pins the disagreement so the data cannot be quietly
edited to fit the sentence.

### Constraints that bite here

- **Phase 1 constraint 13** — nothing in `flow/` speaks lattice units. This module is the boundary's outer face; `lbm/units.py` is its inner face.
- **Phase 1 constraint 15** — `flow/` may import `lbm/`; `lbm/` may never import `flow/`. Assert it in a test.

### Notes

Temperature dependence of `nu` is real and Phase 1 does **not** model it: each fluid carries one
value at one stated temperature and says so. A user asking for water at 80 °C should get told that,
not silently given the 20 °C number.

---

## T105 — Auto-configuration → Rung B

**Status:** `done` — session 17, 2026-08-19. Rung B PASS: 24/24 cases, accuracy 4.2% error (limit 25%), worst Re error 0.0000%, worst peak |u| 0.0695. `pytest` 565 passed, 1 skipped (18 new: 15 in `tests/test_autoconfig.py`, 3 in `tests/test_flow_package.py`). All four Phase 0 rungs and Rung A re-run green with `--backend warp`, printing session 11/15's published digits. **M6.**

### Goal

Given physics, choose everything the solver needs: resolution, `tau`, lattice `U`, domain size and
shape, run length, frame cadence, colour limits. This is the moat. **M6.**

### Reads / depends on

- `DOCS/IDEA3.md` § The five things Phase 1 must get right (1) / § Validation ladder Rung B
- `old-Docs/STATE1.md` **D-019** (characteristic length), **D-023** (`steps_per_frame`), **D-026**
  (periodic sides, 24 D span), **D-029** / **D-032** / **D-036** (the three `tau` floors),
  **D-028** (symmetric colour limits), **D-040** (resolution means the body)
- Tasks: T104

### Inputs / outputs

**In:** `Fluid`, speed `Quantity`, size `Quantity`, a mask, and a quality level
(`"fast" | "balanced" | "accurate"`).
**Out:** `flow/autoconfig.py::Plan` — a frozen dataclass carrying `cells_per_length: int`,
`tau: float`, `u_lattice: float`, `domain: tuple[int, int]` (ny, nx), `steps: int`,
`steps_per_frame: int`, `vorticity_limit: float`, `dx`, `dt`, `Re`, plus `warnings: list[str]` and a
`why: dict[str, str]` giving one sentence per chosen number; `::plan(...) -> Plan`;
`validate/autoconfig.py` printing PASS/FAIL.

### Acceptance criteria

- [x] `plan(...)` returns a `Plan` whose every field has an entry in `why`, and a test asserts that (a field with no explanation is a field the user cannot check).
- [x] Every guardrail is enforced at plan time, each citing its decision: `u_lattice < 0.1` with the **1.8× bluff-body speed-up headroom** (D-032), `tau` above the floor appropriate to the geometry (D-029/D-032/D-036), blockage < 10% and ≥ 8 D downstream (constraint 12, D-019, D-026), min solid thickness ≥ 3 cells (D-017).
- [x] The three quality levels differ **only** in `cells_per_length` and run length, and `accurate` is a strict refinement of `fast` — a test asserts monotonicity of resolution and of wall-clock estimate.
- [x] `Plan.estimated_seconds` predicts wall clock from the measured backend rate; on the committed cylinder case the prediction is **within 25%** of the real run, and the script prints predicted vs actual.
- [x] **Rung B — `validate/autoconfig.py`:** a sweep of at least 24 cases (fluids × speeds × sizes × quality) where every planned case (a) satisfies every guardrail, (b) runs 5000 steps with **no `nan`** and peak `|u|` under 0.1, (c) reproduces its requested `Re` to **0.1%** through `LatticeUnits.reynolds()`. Prints PASS/FAIL and the worst case of each.
- [x] Cases that *cannot* be planned raise `Unrepresentable` (T106 turns it into prose) carrying structured fields — `reason`, `quantity`, `value`, `limit`, `suggestions` — rather than a formatted string.
- [x] `pytest tests/test_autoconfig.py` green; Phase 0 rungs still green.

### Constraints that bite here

- Constraint 3 and constraint 2 — this module is where they are actually enforced for users. It raises; it does not warn.
- Constraint 12 — the domain is *chosen* here, so there is no excuse for a domain that violates it.
- `DOCS/PLAN2.md` § Risks — every constant cites a Phase 0 decision or is measured this session and recorded.

### Notes

The Phase 0 rung setups (`validate/cylinder.py::tau_for`, `validate/polygons.py::tau_for_rung4`) are
hand-tuned instances of exactly this function. Read them first; if `plan()` cannot reproduce their
choices within a factor, one of the two is wrong and finding out which is the session's real work.

---

## T106 — Diagnosis, refusal, nearest runnable case → Rung D

**Status:** `done` — session 18, 2026-08-23. **Rung D green.** One criterion (`substituted=True`)
carried to T108 by decision, not omission — see **D-062** and the Deviations below.

### Goal

A case the tool cannot run produces a plain-language explanation and a concrete alternative that is
**tested to work**. A case that goes unstable mid-run is caught early and explained, not `nan`.

### Reads / depends on

- `DOCS/IDEA3.md` § The five things Phase 1 must get right (2) — the refusal policy (**D-045**)
- `DOCS/IDEA2.md` § Stability (the symptom/cause/fix table is the seed)
- `old-Docs/STATE1.md` **D-032**, **D-038**, **D-029**
- Tasks: T105

### Inputs / outputs

**In:** an `Unrepresentable` from T105, or a live `Sim` during a run.
**Out:** `flow/diagnose.py::explain(exc) -> str`, `::suggest(request) -> list[Suggestion]`,
`::Monitor` (a `per_step`-compatible probe, **D-025**) raising `Diverging` with a cause;
`validate/refusals.py` printing PASS/FAIL.

### Acceptance criteria

- [x] `explain()` output contains **no lattice quantity** — no `tau`, no lattice `U`, no cell counts — in its first paragraph; the numbers are available in a second "details" section. Asserted by a test that greps the first paragraph.
- [x] `suggest()` returns at least one `Suggestion` for each refusal class, each carrying a modified request plus one sentence on what it changes physically (slower, smaller, more viscous fluid, or "the same shape at a Reynolds number we can represent — **not your case**").
- [x] **Rung D — `validate/refusals.py`:** for every refusal class, take the tool's own top suggestion, feed it back through `plan()`, and **run 2000 steps**. PASS requires every suggestion to produce a case that plans and runs without `nan`. A suggestion that does not fix its case is a failing test.
- [x] The Re-2e6 case from **D-038** (air, 20 m/s, 1.5 m) is a named case in the rung, with its full user-facing output pinned in the test as a golden string, so a later reword is a deliberate edit.
- [x] `Monitor` detects divergence **before** `nan`: on a deliberately under-resolved case it raises within 10% of the steps the run would have taken to produce `nan`, naming the cause and the fix. Measured on at least three failure modes from `DOCS/IDEA2.md` § Stability — `tau` too near 0.5, peak `|u|` crossing 0.1, and a mass-drift blow-up. — **met by measurement, with the 10% band met by one of the three modes and beaten by the other two; deviation recorded below (D-061).**
- [x] `Monitor` costs **under 2%** of steps/s, measured with the sim otherwise identical, quoted with CPU clock (**D-035**).
- [ ] **Never a silent substitution** (Phase 1 constraint 16): a test asserts that a `Result` produced from a suggestion carries `substituted=True` and that the flag reaches the printed summary and the recorded video's metadata. — **carried to T108 (D-062)**: `flow.Result`, `flow.report` and the CLI do not exist yet, so the literal check cannot be run against real code. The half that can exist shipped here: every case-changing suggestion carries "not your case" **on the object**, asserted by `tests/test_diagnose.py::test_a_suggestion_that_changes_the_flow_says_so`.
- [x] `pytest tests/test_diagnose.py` green — **43 tests**, suite **610 passed, 1 skipped**.

### Deviations recorded

- **D-061** — the divergence criterion's *"within 10% of the steps the run would have taken to
  produce `nan`"* is met by the failure mode that develops mid-run and beaten, by a lot, by the two
  whose defect is present from the start. Measured, `--backend numpy`: `tau` below the floor —
  caught at **1525**, `nan` at **1650** (**7.6%** earlier, inside the band); driven past the 0.1
  ceiling — caught at **75**, `nan` at **325** (76.9% earlier); mass drift — caught at **50**,
  `nan` at **59275** (99.9% earlier). Tightening the tripwires to land inside 10% for all three
  would mean *degrading* the probe.
- **D-063** — two of T105's suggestions did not fix their own case and were repaired here, since
  "a suggestion that does not fix its case is a failing test" is this task's own criterion.

### Constraints that bite here

- **D-045** — refuse, explain, offer; never run something else and call it the answer.
- Constraint 9's spirit — a wrong answer that looks plausible is the failure mode. A *substituted* answer that looks like the requested one is the same failure with a friendlier face.
- Constraint 8 — `Monitor` runs on the physics thread through `per_step` (D-025); keep it arithmetic-cheap and sampled, not per-cell every step.

### Notes

The wording is not what Rung D tests, and it is still the part users meet. Write the messages as if
to someone who has never heard of a Reynolds number — that is the stated target user — and put the
numbers underneath for someone who has.

---

## T107 — Geometry preparation + shape corpus → Rung C

**Status:** `done`

### Goal

Real user pictures — not convex blobs — become masks the solver can run, or get refused with a
reason. **M7.**

### Reads / depends on

- `DOCS/IDEA3.md` § The five things Phase 1 must get right (3)
- `old-Docs/STATE1.md` **D-017** (thickness metric and its documented limit), **D-018** (border
  exemption), **D-019**, **D-031**, **D-040**
- Tasks: none (independent of T104–T106)

### Inputs / outputs

**In:** a PNG/SVG path or a bool array, plus a target body resolution.
**Out:** `flow/prepare.py::prepare(source, cells_across, *, repair=...) -> Prepared`
(`mask: NDArray[np.bool_] (ny, nx)`, `verdict: "ok" | "repaired" | "refused"`, `actions: list[str]`,
`properties: dict`); `tests/data/shapes/` corpus with `expectations.json`;
`validate/shapes.py` printing PASS/FAIL.

### Acceptance criteria

- [x] **The corpus is committed**: at least 12 images covering hairline appendage fused to a thick body, detached specks, interior hole (donut), unclosed outline, heavy anti-aliasing, huge margin, tiny body, extreme aspect ratio, diagonal 1-cell line, self-touching shape, all-white, all-black. Each has a committed expected verdict and expected properties.
- [x] **D-017's documented limit is closed or restated with evidence:** a thin appendage fused to a thick body is *detected* (it currently shares the component and is not reported). If a metric that catches it also false-alarms on a plain disc, the measurement is recorded and the limit stays — but it is measured this session, not inherited.
- [x] Repairs are individually switchable and individually reported in `actions`: fill interior holes, drop specks below a stated area, thicken sub-3-cell features, keep the largest component. Nothing is repaired silently.
- [x] After repair, the mask satisfies constraint 12's rules — min thickness ≥ 3, and the checks in `check_mask` produce no warning — asserted for every corpus image whose verdict is `repaired`.
- [x] Refusal cases refuse: an all-white, an all-black and a body smaller than 8 cells across return `verdict="refused"` with a reason naming what is wrong and what would fix it.
- [x] **Rung C — `validate/shapes.py`:** every corpus image, no manual step, verdict and measured properties compared against `expectations.json`. PASS/FAIL printed, with a per-image line.
- [x] Body resolution obeys **D-040**: the *measured* body is the requested size after preparation, within 1 cell, for every corpus image that is not refused.
- [x] `pytest tests/test_prepare.py` green; Phase 0 rungs still green.

### Constraints that bite here

- Constraint 12 — this is the module that makes it true rather than merely warned about.
- **D-040** — every derived number comes from the same measured `D`. A repair that changes the body's extent must re-derive, not adjust.
- `DOCS/PLAN2.md` § Risks — repair must not turn the user's shape into a different shape; the rung asserts measured properties, not merely the absence of warnings.

### Deviations recorded

- **Q-102 is closed, not restated** — `flow.prepare.thin_branch_depth` catches a hairline fused to a
  thick body and does **not** false-alarm on a plain disc at any radius or sub-cell offset. The
  measurement is **D-064** and `validate/shapes.py` § 4 re-runs it every time rather than quoting it.
- **A fourteenth and fifteenth refusal class, neither in the contract.** A picture that cannot
  produce the requested body size at any raster (a two-pixel plate asked for 40 cells) is
  **refused** with the size it can reach, rather than quietly returning a different resolution —
  **D-065**. Returning 61 cells for a request of 40 was the original behaviour and it is a silent
  substitution of `tau`, not just of the picture.
- **`flow.prepare` the function shadows `flow.prepare` the module** on the package, deliberately:
  `flow.prepare(picture, 40)` is the API the product wants. The module is reachable as
  `from flow.prepare import ...` and `sys.modules["flow.prepare"]`, which is what
  `tests/test_prepare.py` uses.

### Notes

Generate the corpus programmatically where possible so the images are auditable and small, and
commit the generator alongside. `check_mask` and `min_thickness` stay in `lbm/geometry.py` — `flow/`
adds judgement and repair on top and does not fork them.

**Done in session 19.** 15 corpus images (`tests/data/shapes/`, generator committed beside them),
`flow/prepare.py`, `validate/shapes.py` = **Rung C, PASS in 10.6 s**, `tests/test_prepare.py` 48
tests, `pytest` **660 passed, 1 skipped**.

---

## T108 — `flow.Case` / `flow.Result` API

**Status:** `done`

### Goal

The three lines from `DOCS/IDEA3.md` § What Phase 1 is, concretely actually run. Everything decided
in T104–T107 gets one front door.

### Reads / depends on

- `DOCS/IDEA3.md` § What Phase 1 is, concretely / § The five things (4)
- `old-Docs/STATE1.md` **D-024** / **D-039** (two run modes), **D-028** (colour limits), **D-030**
  (seed solid at rest), **D-023**
- Tasks: T105, T106, T107

### Inputs / outputs

**In:** an image path plus physical strings.
**Out:** `flow/case.py::Case` (`from_image`, `from_array`, `explain()`, `plan`, `run(...)`),
`flow/report.py::Result` (`cd`, `cl`, `strouhal`, `convergence`, `peak_u`, `elapsed`, `substituted`,
`frames`, `save(path)`, `summary()`, `plot()`).

### Acceptance criteria

- [x] `Case.from_image("x.png", fluid="air", speed="5 m/s", size="0.1 m")` builds without running anything, and `explain()` prints the plan, every `why` line, the geometry verdict and actions, and the estimated wall clock.
- [x] **No lattice quantity appears in any public signature of `flow/`** — asserted by an introspection test over every public callable's annotations and defaults (Phase 1 constraint 13).
- [x] `run()` accepts `live=`, `record=`, `headless=` and composes them through `TeeSink`, selecting `drop` by **D-039** (any file-writing sink ⇒ `drop=False`); a test asserts the mode chosen for each combination.
- [x] `Result.save("wake.mp4")` and `.save("frames/")` both work and go through `lbm.record`; `flow/` **colours nothing** — asserted by a test that no `flow` module imports a colormap or builds RGB (constraint 10).
- [x] Solid cells are seeded at rest (**D-030**) and a test asserts the body interior holds the rest state after 300 steps.
- [x] `Result.summary()` prints Cd (mean ± std), Cl amplitude, St with its confidence, peak `|u|` against 0.1, convergence, elapsed, backend, and — if applicable — the substitution banner.
- [x] `Result.strouhal` is `None`, not a number, when shedding is not detected (Cl amplitude below 1% of Cd); a test covers a steady case at Re 10.
- [x] **Never a silent substitution** (constraint 16) — **inherited from T106, see D-062**: a test asserts that a `Result` produced from a `flow.diagnose` suggestion carries `substituted=True` and that the flag reaches the printed summary *and* the recorded video's metadata. T106 shipped the half that could exist without `Result` (every case-changing suggestion carries "not your case" on the object, asserted by `tests/test_diagnose.py`); this is the other half.
- [x] `pytest tests/test_case.py tests/test_report.py` green; Phase 0 rungs still green.

### Constraints that bite here

- Constraint 10 — one `render()`, three sinks. `flow/` composes them; it does not add a fourth renderer for plots — matplotlib figures are a separate, non-frame output and say so.
- Constraint 7 / **D-023** — `steps_per_frame` comes from `dt`, through the plan.
- **D-045 / constraint 16** — `substituted` is carried by `Result` and printed, never dropped on the way out.

### Deviations recorded

- **`Case` does not raise for a refused case at construction; it carries the refusal and `run()`
  raises it** (**D-067**). The contract says `from_image(...)` "builds without running anything" and
  that `explain()` prints the way forward — both of which a constructor that raises makes
  impossible, and T109's `--explain` needs to print a refusal and exit 2 without exception plumbing.
  Nothing is ever run in place of a refused case: `run()` raises, and **D-065**'s picture refusal is
  raised as the same `Unrepresentable` a physics refusal is.
- **A geometry `Fix` is translated into `quality`, never into `cells_across`** (**D-068**). The
  contract names `cells_across` as `prepare`'s argument and not `Case`'s (constraint 13), so
  `flow.prepare.apply_fix`'s `change="resolution"` arrives as the finest quality level the picture
  can actually resolve — and when even `"fast"` is too much for it, the worked example is
  substituted and says so, exactly as `apply_fix` does for a `"picture"` fix.
- **The measurement window starts after the startup kick has washed out, not when it switches off**
  (**D-069**), and a run too short for that reports **nothing** rather than the kick. Found by the
  Re 10 criterion: measured, the window that opens at kick-off reports a lift "amplitude" of 0.55
  against a `Cd` of 3.6, which is a decaying transient reading as a shedding wake.
- **`Result.strouhal` passes three gates, not one** (**D-070**). The contract names the lift-amplitude
  gate; two more were needed because a window can hold an amplitude and still hold no frequency —
  measured, a one-period synthetic sine planted at `St = 0.17` returns **0.459**.
- **The Re 10 criterion is covered end to end at 6000 steps plus a hand-measured trace**, not by a
  full post-settling window: at this domain size a measurable window does not open until ~21600
  steps (~110 s), which is more than the whole suite costs. The full trace is in `DOCS/STATE2.md`
  § session 20 and in the test's docstring.

### Notes

`Case` is a facade. If logic accumulates in it that is not delegation, it belongs in `autoconfig`,
`diagnose` or `prepare`, where it can be tested without a run.

---

## T109 — CLI on `flow`, live + record wiring

**Status:** `not_started`

### Goal

`python -m flow` replaces the Phase 0 `python -m lbm.runner` as the thing a person runs, built on
`Case` rather than on hand-assembled solver calls.

### Reads / depends on

- `DOCS/IDEA3.md` § Scope (a CLI is the Phase 1 UI, D-044)
- `lbm/runner.py`'s existing CLI (`main`, `_build_parser`, `_resolve_sinks`, `_body_mask`) — the
  behaviour to preserve, the implementation to replace
- Tasks: T108

### Inputs / outputs

**In:** command-line arguments.
**Out:** `flow/cli.py`, `flow/__main__.py`; `lbm/runner.py`'s CLI kept, delegating or deprecated
with a one-line pointer (decide and record).

### Acceptance criteria

- [ ] `python -m flow --shape wing.png --fluid air --speed "5 m/s" --size "10 cm" --out wake.mp4` runs end to end from a cold shell and writes a playable file.
- [ ] `--explain` (or `--dry-run`) prints the full plan and exits **0 without simulating**; a refused case prints the explanation and its suggestions and exits **2** (matching Phase 0's convention, D-038).
- [ ] `--quality fast|balanced|accurate`, `--seconds`, `--backend numpy|warp`, `--live`, `--record`, `--headless` are all wired; `--live --record` works together (**D-039**).
- [ ] The old `python -m lbm.runner` invocation from the M4 gate still produces an MP4, or prints a one-line pointer to the new command — whichever is chosen is recorded as a decision and tested.
- [ ] `--help` states the Re limit in plain words, as Phase 0's does (**D-038**), so the arithmetic is met before the run and not after.
- [ ] Missing ffmpeg still produces `lbm.record.FFMPEG_HINT` before the first timestep, not a traceback.
- [ ] `pytest tests/test_cli.py` green; Phase 0 rungs still green.

### Constraints that bite here

- Constraint 8 / **D-039** — a file-writing sink never drops; live-only may.
- `DOCS/IDEA3.md` § Scope — no UI. A prettier terminal output is fine; a TUI framework is not.

### Notes

The Phase 0 CLI is the reference for behaviour that already works — geometry placed at body scale in
a domain sized in its own diameters, the startup kick, the printed summary. Port the *behaviour*;
the assembly now lives in `Case`.

---

## T110 — The minute: end to end, timed → Rung E, M8

**Status:** `not_started`

### Goal

From a cold shell: a picture and three physical numbers give a **correct** answer in **under 60
seconds**. Phase 1 closes here. **M8.**

### Reads / depends on

- `DOCS/IDEA3.md` § Goal / § Validation ladder Rung E
- `DOCS/PLAN2.md` § Milestone gates
- Tasks: T109, T103

### Inputs / outputs

**In:** the finished product path.
**Out:** `validate/minute.py` printing PASS/FAIL with the elapsed time; a `README.md` quickstart
section; `DOCS/STATE2.md` recording M8 with the gate output.

### Acceptance criteria

- [ ] **Rung E — `validate/minute.py --backend warp`:** a committed PNG of a disc, plus fluid/speed/size chosen to give Re 100, driven through `flow.Case` with no lattice quantity anywhere in the invocation, produces **St in 0.155–0.175 and Cd in 1.25–1.45** — Rung 3's published bands, unwidened.
- [ ] **Wall clock under 60 s**, printed, from process start to `Result.summary()`, on the dev machine with GPU name, driver, CPU clock and power state quoted (**D-035**). If it misses, the number is recorded honestly and the shortfall becomes a blocker, not a widened criterion.
- [ ] The same rung run with `--backend numpy` **also passes on physics** (bands only, no time limit), so the product path is proven independent of the port.
- [ ] All five Phase 1 rungs (A–E) re-run in this session and print PASS; all four Phase 0 rungs re-run and print numbers inside their published bands on both backends.
- [ ] `README.md` gains a quickstart that is **copy-pasteable and was actually pasted** into a fresh shell this session, with its output recorded.
- [ ] `myenv/Scripts/python.exe -m pytest` green.
- [ ] `DOCS/STATE2.md` records **M8** with the gate output, and § Snapshot says Phase 1 is complete.

### Constraints that bite here

- Constraint 5 — the whole ladder, in order, re-run. This is the session that proves the phase, not the session that finishes the last feature.
- `idea.md` § Definition of success — this rung is that sentence, minus the drag-and-drop, plus the word *correct*.

### Notes

If the wall clock misses by a little, the honest options are a lower default quality level or a
smaller default domain — **both of which change the physics** and therefore must be re-validated
against the bands, in this rung, in this session. Widening the bands is not an option.
