# Session 15 — T103: Warp boundaries, checkpoint, performance → M5

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis:
*"The gap is not the solver. The gap is everything around the solver."*

**Phase 0 is complete** — a validated, continuously-running 2D D2Q9 lattice-Boltzmann simulator in
pure NumPy, spec `DOCS/IDEA2.md`, closed at M4 on 2026-08-13 with all four validation rungs green.

**Phase 1 is live**: the product layer above that solver — a `flow/` package plus a CLI, on a Warp
GPU backend, ten tasks `T101` → `T110`, five new validation rungs A–E, milestones M5 → M8. Spec
`DOCS/IDEA3.md`, plan `DOCS/PLAN2.md`.

**This is the session that finishes the port.** T101 built the seam, T102 put four kernels on the
GPU and proved they agree with NumPy. T103 moves the *whole timestep* there and measures it. It is
**M5**, and it is the last GPU session Phase 1 gets — see the pressure valve below.

## Read these first, in this order

1. `CLAUDE.md` — hard constraints (16 of them, the Phase 1 list), session protocol, conventions.
2. `DOCS/STATE2.md` — **in full**: snapshot, blockers, open questions, environment, performance
   baseline, decisions D-041 … D-053, and the session 12, 13 and 14 log entries.
3. `DOCS/TASKS2.md` § T103 — the task contract, in full. Also read § T102 (just finished) so you
   know what the Warp backend already does and what it deliberately does not.
4. `DOCS/IDEA3.md` § Performance budget (in full — the bandwidth arithmetic behind the floors is
   what you argue with if a floor is missed) · § Validation ladder, Rung A's row.
5. `old-Docs/STATE1.md` § Performance baseline, and **only these decisions**: **D-011** (the `f_pre`
   copy and why the timestep order is what it is), **D-020** (the pre-stream snapshot
   `probe.forces` consumes), **D-021**, **D-022** (the checkpoint is exactly four things plus
   `format`), **D-033** (the fused `collide_stream`, and that fused and unfused must agree bitwise),
   **D-035** (how any speed number must be measured — alternating rounds, CPU clock, power state,
   GPU name). STATE1 is **frozen**: read it, never edit it (**D-041**).
6. `DOCS/PLAN2.md` § Dependency graph, § Session map, § Milestone gates (M5's literal gate command)
   and § Risks — the first three risk rows are yours.
7. `lbm/backends/warp_backend.py` and `validate/parity.py` — what session 14 left you, including
   the two stubs that raise `NotImplementedError("see DOCS/TASKS2.md T103")`.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 14: **T102 is done.** `lbm/backends/warp_backend.py` has
  `equilibrium`, `collide`, `stream`, `macroscopic`, `to_host`, `from_host` running on `cuda:0`;
  `validate/parity.py --kernels` is Rung A's kernel half and **passes**. `warp-lang` **1.16.0** is
  installed (CUDA Toolkit 12.9, Driver 13.1, RTX 3050 Laptop GPU, 4 GiB, sm_86).
- **Measured parity, session 14** — worst **5.96e-08** in `f` units against a 1e-6 bar:
  `macroscopic` **bitwise**, `stream` **bitwise**, `collide` **1.49e-08**, `equilibrium`
  **5.96e-08** (one fused multiply-add each, **D-053**). Spike test 9/9 on the GPU.
- **Phase 0 rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩. Session 14 re-ran R1 (**L2 0.3650%**) and
  R2 (**Re 100, 0.75%, vortex 0.21 cells**), identical to sessions 11 and 13. R3 (**St 0.1731,
  Cd 1.4031 ± 0.0086**) and R4 (**square Cd 1.5279 ± 0.0271**, **polygon Cd 1.4276 ± 0.0226**) were
  not re-run because no Phase 0 file was touched. **These are the bands the GPU must land inside**
  — inside, not identical: cross-backend bit-identity is neither achievable nor the goal.
- **Phase 1 rung status:** **A 🟨** (kernels green, whole step not written) · B ⬜ · C ⬜ · D ⬜ · E ⬜.
- **Milestone reached:** M4 (2026-08-13). **M5 is yours this session.**
- **Completed tasks:** Phase 0 T001 … T011. Phase 1: T101, T102.
- `myenv/Scripts/python.exe -m pytest` → **408 passed, 1 skipped** at the end of session 14. (The
  skip is `test_a_known_but_uninstalled_backend_names_its_install_line`, which skips by design now
  that `warp` is installed.)

## Your task this session

**T103 — Warp boundaries, checkpoint, performance → M5.** One task, this session only.

Run this first:

    /start-task T103

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** the whole timestep runs on the GPU, the four Phase 0 rungs pass on it, and the performance
budget is met.

**Outputs:** `bounce_back`, `moving_wall`, `inlet_velocity`, `outlet_zero_gradient` and the fused
`collide_stream` on the Warp backend; `bench.py --backend warp`; `validate/parity.py` full mode;
`--backend` flag on all four Phase 0 rung scripts.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] Every boundary condition Phase 0 ships runs on the GPU, and `validate/parity.py` compares each separately against NumPy at ≤ 1e-6 in `f` units.
- [ ] **Whole-step parity:** starting from identical state, 1000 steps on each backend agree to `max|Δu| / U < 1e-4`, printed. The number is expected to grow with step count — the script prints it at 10 / 100 / 1000 steps so the growth rate is visible and not merely bounded.
- [ ] `save_checkpoint` on the GPU backend writes the same four things plus `format` (**D-022**), via `to_host`; a checkpoint written on `warp` **resumes on `numpy`** and continues within the whole-step parity tolerance. Within a backend, restart stays **bit-identical**.
- [ ] **All four Phase 0 rungs pass with `--backend warp`, inside their published bands** — R1 L2 < 1%, R2 max deviation < 5%, R3 St 0.155–0.175 and Cd 1.25–1.45, R4 Cd 1.4–1.6. Bands are not widened; the printed numbers are recorded beside session 11's.
- [ ] **`bench.py --backend warp` clears ≥2000 / ≥250 / ≥150 steps/s at 40k / 1M / 2M cells**, measured by alternating rounds (**D-035**), with GPU name, driver, CPU clock and power state quoted.
- [ ] The GPU memory footprint at 2M cells is printed and fits the 4 GB card with room for the display path.
- [ ] `myenv/Scripts/python.exe -m pytest` green; **M5 recorded in `DOCS/STATE2.md` with the gate output pasted in.**

### Constraints that bite on this task

From `CLAUDE.md` § Hard constraints, in their Phase 1 form:

- **Constraint 1** — *the physics does not change in Phase 1.* The boundaries are transcriptions of
  `lbm/boundary.py`, term for term and in the same order, exactly as T102's four kernels were. No
  curved boundaries, no MRT, no turbulence model — `DOCS/IDEA2.md` § Deliberately deferred stays
  deferred.
- **Constraint 4** — the backend owns its device layout, but `to_host` must yield `(9, ny, nx)`
  `float32`. **This is the constraint that makes the checkpoint criterion possible at all**: a
  Warp checkpoint is readable on NumPy only because `f` goes out through `to_host`. The nine
  constants still come from `lbm/core.py`, uploaded once — `OPP` and `CS2` are already on the device
  from T102 precisely for your boundaries.
- **Constraint 5** — the ladder is ordered. **Rung A is your gate, and it is currently 🟨.** A Phase 0
  rung that fails on GPU blocks T104 onward and is a `DOCS/STATE2.md` § Blockers entry, not a queued
  issue. Do not start on a failing rung.
- **Constraint 6's replacement** — *no backend optimisation before its parity rung passes.* Order
  within this session matters: boundaries correct → whole-step parity green → **then** the fused
  path and the performance table. Tuning a kernel whose Rung A is red is how a wrong answer gets
  fast.
- **Constraint 8** — the live path must not block the physics. Device-to-host transfer for a frame
  happens on the **frame** cadence, not the step cadence.
- **Constraint 11** — restart is **bit-identical within a backend**; **across** backends it is the
  printed tolerance above. This distinction is already written into `CLAUDE.md`; your job is to make
  the number real and print it.
- **Constraint 2** — `nu = (tau - 0.5)/3` still lives only in `nu_from_tau`. Kernels take `omega`
  or `tau` and never re-derive viscosity.

### Blockers, open questions and decisions that affect you

**Blockers:** none.

**Open questions:**

- **Q-103** — what tolerance does cross-backend whole-step agreement actually need? **You close
  this.** The contract is `max|Δu|/U < 1e-4` at 1000 steps and it is *a pass condition to be met,
  not adjusted*. Session 14's evidence: per kernel the disagreement is at most **5.96e-08** in `f`
  units, and two of the four kernels are bitwise identical — so 1e-4 looks generous **if the error
  does not compound**. Whether it compounds is what your 10 / 100 / 1000-step print measures.
  Record the answer as a decision.

**Decisions that constrain this session:**

- **D-052** (session 14) — the Warp backend currently takes **host** arrays at its boundary and owns
  preallocated **device** buffers keyed by grid shape; `Sim` still owns its `(9, ny, nx)` NumPy
  buffers, so every kernel call copies host→device→host. That makes it *slower than NumPy* today,
  which is why T102 quotes no speed number. **Removing those copies — moving the state onto the
  device — is the first thing this session does**, and nothing in the performance budget is
  reachable without it.
- **D-051** (session 13) — the `Backend` protocol covers kernels and the two host transfers and
  nothing else. Buffer allocation, the open boundaries, the Guo body force and the probes are
  outside it. **You widen the seam**, allocation first; that is budgeted work, not a defect.
- **D-050** (session 13) — the checkpoint is exactly `f` / `solid` / `step_count` / config plus
  `format: 1` (**D-022**). `f` is written through `backend.to_host`; the backend name rides inside
  the pickled config; `load_checkpoint(path, backend=...)` overrides it. That override is the
  mechanism your "written on warp, resumed on numpy" criterion uses.
- **D-053** (session 14) — the per-kernel differences and their cause (FMA contraction), with the
  bitwise column recorded: a *future* disagreement in `macroscopic` or `stream` is a **bug**, not
  float ordering, and the tolerance argument does not cover it.
- **D-043** — **NumPy is the reference oracle, not a legacy path.** A GPU that disagrees is a broken
  backend, never a new answer.
- **D-033** — fused and unfused `collide_stream` must agree **bitwise on a given backend**, and the
  Guo body force stays on the unfused path. Both properties have to survive on the GPU too.
- **D-035** — every speed number comes with `Win32_Processor.CurrentClockSpeed`, the power state and
  the GPU name beside it, and A/B by alternating rounds with one `Sim` resident. Session 13's
  machine state for comparison: AMD Ryzen 7 5800H at **3201 MHz** of 3201 MHz, on **mains**.
  **This session is the one that must quote numbers**, so re-read D-035 before running `bench.py`.

### Before you start

- **Nothing to install.** `warp-lang` 1.16.0 is in `myenv` (CUDA Toolkit 12.9, Driver 13.1, `cuda:0`
  = NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB, sm_86; `nvidia-smi` driver 592.82). Also present:
  numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1, pygame 2.6.1, imageio 2.37.4,
  imageio-ffmpeg 0.6.0, psutil 7.2.2, Python 3.11.15.
- Machine: RTX 3050 **4 GB**, 16 GB RAM. The 2M-cell criterion is deliberately near that ceiling —
  `9 × 4 bytes × 2M` is 72 MB per `(9, ny, nx)` buffer, and Phase 0's `Sim` holds four of them.
- Confirm the starting point: `myenv/Scripts/python.exe -m pytest` should print **408 passed,
  1 skipped**, and `myenv/Scripts/python.exe -m validate.parity --kernels` should print **PASS** with
  a worst case of 5.96e-08, before you change anything.
- R4 alone is ~40 minutes on NumPy; on GPU it should be minutes, which makes the full ladder cheap
  for the first time. **Do not use that to justify running it less carefully.**

**The pressure valve, read it before you start, not at hour three:** `DOCS/PLAN2.md` § Risks, first
row — `idea.md`'s "The trap", Phase 1 becoming a solver-optimisation project. **T102 landed on
schedule; if T103 overruns by one session, the port is demoted back to Phase 2** and Phase 1
continues on NumPy with M8's wall clock restated honestly against it. T101's seam makes that a
config change, not a rewrite. If a *floor* is missed but the rungs are green, that is a number to
record and argue with using § Performance budget's bandwidth arithmetic — not a reason to keep
tuning into a second session.

## Scope discipline

Work only what's in the contract. If something else needs doing, `/new-task` it against
`DOCS/TASKS2.md` — do not expand this one. If it is listed under `DOCS/IDEA2.md` § Deliberately
deferred or `DOCS/IDEA3.md` § Deliberately deferred (XLB, a UI, 3D, STL, packaging), the answer is
no. **Nothing in this session creates `flow/`** — that is T104 onward, and `lbm/` may never import it
(**D-042**, constraint 15).

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `myenv/Scripts/python.exe -m validate.parity` (full mode) **and** `--kernels`, and print the
   measured numbers — per boundary, and whole-step at 10 / 100 / 1000 steps.
3. Run all four Phase 0 rungs with `--backend warp` and record the printed digits **beside session
   11's**, inside the published bands.
4. Run `myenv/Scripts/python.exe -m pytest` and confirm it is green.
5. **Run `/checkpoint`** — it updates `DOCS/STATE2.md` (with **M5** and the gate output pasted in),
   syncs `DOCS/TASKS2.md`, and writes the next session's prompt into `PROMPTS/`. Do not end the
   session without it.
