# CLAUDE.md — Fluid Mech / Phase 2 — FengDong

**Project, one line:** A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp —
under a product layer that takes a picture and three physical numbers, and, from Phase 2, under a
window you drop the picture onto.

**The solver is not the product.** Phase 0 built and validated it so we understood LBM well enough to
design the layers above it; Phase 1 built those layers; Phase 2 ships them to a person. See root
`idea.md` and `README.md`, and `idea.md` § Risks — *"The trap"*, which names the standing temptation
to go back to polishing the solver because that part is fun.

**Specs by phase: Phase 0 `DOCS/IDEA2.md` · Phase 1 `DOCS/IDEA3.md` · Phase 2 `DOCS/IDEA4.md` (live).**
Don't re-derive decisions already made there — cite them. If anything here conflicts with the spec of
the live phase, **the spec wins**; log the conflict in `DOCS/STATE3.md` § Decisions rather than
silently picking one.

The existing `Navier-Fluid-Equation/` directory is **prior work** — potential-flow / panel-method
scripts. It is not part of the LBM solver. Reuse its polygon-vertex code (`polygonsDemo.py`,
`panels.py`) for `lbm/geometry.py` primitives; do not modify it otherwise.

---

## Hard constraints

Load-bearing decisions, not optimizations. A design that drifts from these is wrong even if it runs.

**These are the Phase 2 constraints.** Phase 1's sixteen are all carried forward: **D-081** rewrote
constraint 1 (one turbulence closure, named and switchable), **D-082** and **D-083** added **17–20**,
and every other constraint is kept verbatim. The fate of all sixteen is decided and recorded in
`DOCS/STATE3.md` § Constraint fate table — the same exercise D-046 did for Phase 0's twelve. **This
list is the authority**; D-046 and D-081 / D-082 / D-083 are the record of *why* each one reads the
way it does, and the retired one is kept below, struck, rather than deleted.

1. **The physics is D2Q9, BGK single relaxation time, bounce-back walls, plus exactly one turbulence
   closure — Smagorinsky, and it is named.** No MRT, no cumulant, no KBC, no curved or interpolated
   boundaries, no wall model, no dynamic `Cs`. The *implementation* may move to another backend; the
   **base** arithmetic it transcribes may not change, and with the closure **off** it must be bitwise
   what Phase 1 shipped (constraint 19). Deferred is not the same as forgotten. *(Rewritten by
   **D-081**, superseding D-046's rewrite.)*
2. **Viscosity is not a free parameter.** `nu = cs2 * (tau - 0.5) = (tau - 0.5) / 3`. Never expose a
   `nu` setter that doesn't go through `tau` — and that governs the closure's `tau_eff` exactly as it
   governs `tau`: Smagorinsky modifies the relaxation time, never a viscosity directly.
   `tau -> 0.5` means `nu -> 0` means the sim blows up.
3. **Lattice velocity stays under 0.1.** Compressibility error scales as Mach squared. Any config
   path that can produce `|u| >= 0.1` must warn at setup, not at `nan` time — and `flow/autoconfig.py`
   enforces it for users who never see `u`.
4. **The backend owns its state layout; `to_host` must yield `(9, ny, nx)` `float32`**, index order
   `(direction, y, x)`. That shape is the portability contract — checkpoints and cross-backend parity
   both go through it and nothing else. The nine constants (`e`, `w`, `opp`, `cs2`) live in
   `lbm/core.py` and are imported from there — uploaded to a device, **never redefined**, in any
   backend. *(Rewritten by D-046.)*
5. **The validation ladder is non-negotiable and ordered.** Phase 0: Rung 1 Poiseuille, Rung 2 cavity
   vs Ghia, Rung 3 cylinder Re 100, Rung 4 square cylinder — all four stay a gate for every Phase 1
   task. Phase 1 adds Rung A parity, B auto-config, C shapes, D refusals, E the minute (**D-047**). Phase 2
   adds F LES degeneracy, G Taylor–Green, H the fidelity bands, I the install, J the drop — fourteen
   in all, and every earlier rung stays a gate for every later task.
   Each rung is a script in `validate/` that prints pass/fail. **A wrong sim that looks plausible is
   the main failure mode of this project.** Do not start rung N+1 while rung N fails.
6. ~~**Do not optimise before Rung 3 passes.**~~ **Retired** — spent in session 7 when Rung 3 went
   green. Replaced by: **no backend optimisation before its parity rung passes.** A backend with no
   green Rung A is a backend to make correct, not fast.
7. **Simulation and rendering are decoupled.** One rendered frame is many timesteps.
   `steps_per_frame` is **computed** from target playback speed — never hardcoded to 20.
8. **Never block the sim on the display.** Ring buffer between them. If it fills, drop *display*
   frames, never simulation steps — and any file-writing sink takes `drop=False` (**D-039**).
9. **Draw vorticity, not speed.** Diverging colormap, symmetric **fixed** limits. Speed magnitude is
   a grey smear; per-frame autoscaled limits flicker.
10. **One `render()`, three sinks** (live / record / headless). Do not write three renderers, and
    `flow/` colours nothing — nor does `fengdong/`, whose live view is a **fourth sink on the existing
    ring buffer**, never a new path to the screen.
11. **Restart must be bit-identical within a backend.** `f`, `mask`, and step count are the entire
    state. Pickle every N steps; resume produces a bit-identical continuation, and that is a tested
    claim. **Across** backends it is a printed tolerance (T103), because float ordering differs on a
    GPU and no test should pretend otherwise. *(Rewritten by D-046.)*
12. **Geometry is one boolean array**, `solid`, shape `(ny, nx)`. Solid at least 3 cells thick
    (detect and warn — thinner leaks through bounce-back), object ≥8 diameters from the outlet,
    blockage ratio under ~10%. Phase 1 *repairs* where it can rather than only warning (T107).
13. **No lattice quantity in any public `flow/` signature, and none in a `fengdong/` widget.** No
    `tau`, no lattice `U`, no `steps_per_frame`, no cell counts, and `Cs` is not a user knob — it is
    planned and printed, and the fidelity band is what surfaces instead. The inputs are a picture, a fluid, a speed, a size. Everything
    else is derived and **printed**. *(New in Phase 1.)*
14. **Every refusal names a fix, and the fix is machine-checked.** A refusal carries `reason`,
    `quantity`, `value`, `limit`, `suggestions`; Rung D feeds the tool's own top suggestion back
    through the planner and runs it. A suggestion that does not fix its case is a failing test.
    *(New in Phase 1.)*
15. **`flow/` may import `lbm/`; `lbm/` may never import `flow/`,** and a test asserts it. That
    one-directional import is what keeps an eventual XLB swap a substitution rather than a rewrite
    (deferred past Phase 2 by **D-080**, and the seam is why deferring it costs nothing).
    *(New in Phase 1.)*
16. **No silent substitution.** A run that differs from what was asked says so in every artifact it
    produces — the printed summary, the report, and the video metadata — via `substituted=True`.
    *(New in Phase 1.)* A run that engaged the closure is such a run, and carries its band with it.
17. **`fengdong/` may import `flow/`; `flow/` may never import `fengdong/`,** and a test asserts it.
    Same shape as constraint 15 and for the same reason: **the app is a view, not a second brain.**
    Every solver parameter it displays comes from `flow.autoconfig.plan`. *(New in Phase 2, D-083.)*
18. **No unqualified quantitative claim outside the validated band.** Every `Result` carries a
    `fidelity` band — `quantitative` / `qualitative` / `illustrative`, decided from the eddy viscosity
    the run actually generated — and outside `quantitative` there is no bare `Cd`. Rung H asserts it
    by inspecting the object, not by reading the prose. *(New in Phase 2, D-082.)*
19. **The closure defaults off, and `Cs = 0` is bitwise identical to Phase 1 on every backend.**
    A closure you cannot switch off is a closure you cannot validate against, and nine green rungs are
    what this phase puts at risk. *(New in Phase 2, D-081.)*
20. **One `pip install`, one command.** The distribution is `fengdong`; Rung I installs a built wheel
    into a fresh venv with no repository on the path. A package that only installs from the
    developer's tree is not distributed. *(New in Phase 2, D-083.)*

Constraints 13–16 are enforced by tests that live with the code they govern (`flow/`). **Constraint
19 landed with T201/T202** (Rung F, `validate/les.py`) and **constraint 18 landed with T204** —
`flow/fidelity.py` decides the band, `flow.report.Result.__post_init__` withholds every claim the band
forbids, and Rung H (`validate/fidelity.py`) asserts it by inspecting the object. Constraints **17**
and **20** land with the code *they* govern (T205 onward); until then they are the design rule code
has to be written to satisfy, not a dead letter.

---

## Session protocol

**Follow this every session. No exceptions.**

**Phase 2 is live. The live documents are `DOCS/STATE3.md` and `DOCS/TASKS3.md`** — Phase 1's
`DOCS/STATE2.md` / `DOCS/TASKS2.md` / `DOCS/PLAN2.md` / `DOCS/IDEA3.md` and Phase 0's
`old-Docs/STATE1.md` / `old-Docs/TASKS1.md` / `old-Docs/PLAN1.md` are **frozen**: read for history,
never edited (**D-041**, **D-084**). Phase 1's stay at their `DOCS/` paths rather than moving to
`old-Docs/` — D-084 priced the move at ~470 citations and rejected it. Everywhere below that names a
Phase 0 or Phase 1 file, read the Phase 2 one instead.

1. **At session start** — read `DOCS/STATE3.md` (all of it) and `DOCS/TASKS3.md` (the row for the
   live task) **before touching any code**. They say what task is live, what's blocked, what the
   previous session actually left behind. `old-Docs/STATE1.md` § Decisions (D-005 … D-040) and
   `DOCS/STATE2.md` § Decisions (D-041 … D-079) are still in force and are cited by number; read the
   entry a task names, not the whole file.
2. **Before working a task** — run `/start-task T2XX`. It reads the task contract and the
   `DOCS/IDEA4.md` section it cites, then restates goal + acceptance criteria for confirmation.
3. **One task per session.** `DOCS/PLAN3.md` maps one task to one session deliberately — context
   stays small and each session ends at a validated boundary. Work that turns up mid-stream gets
   `/new-task`, not scope creep.
4. **At session end** — run `/checkpoint`. Never end a session without it. It updates
   `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes the next paste-ready prompt into
   `PROMPTS/`.

## Coding conventions

- **Type hints everywhere.** Arrays annotated with intent: `NDArray[np.float32]`, and document
  shape in the docstring (`(9, ny, nx)`).
- **Docstrings cite the spec** — name the section of the spec that owns the task (`DOCS/IDEA2.md`,
  `DOCS/IDEA3.md` or `DOCS/IDEA4.md`) so the reasoning is one hop away.
- **Preallocate. Never allocate inside the step loop.** Buffers are created once by the runner and
  passed in, or held on the sim object.
- **`float32` throughout.** Halves the bandwidth; accuracy is fine for this.
- **No physics constant twice.** `e`, `w`, `opp`, `cs2` from `lbm/core.py` only.
- **Physical units never reach the solver.** `lbm/units.py` converts at the boundary; everything
  inside `lbm/` is lattice units.
- Stubs raise `NotImplementedError("see DOCS/TASKS3.md T2XX")` until their task lands.
- `pytest` for unit tests; `validate/` scripts are the integration tests and print pass/fail.

## Commands

```bash
myenv/Scripts/python.exe -m pytest                        # unit tests
myenv/Scripts/python.exe -m validate.poiseuille           # Rung 1
myenv/Scripts/python.exe -m validate.cavity --re 100      # Rung 2
myenv/Scripts/python.exe -m validate.cylinder             # Rung 3
myenv/Scripts/python.exe -m validate.polygons             # Rung 4
myenv/Scripts/python.exe -m validate.parity --backend warp  # Rung A
myenv/Scripts/python.exe -m validate.autoconfig           # Rung B (~23 min)
myenv/Scripts/python.exe -m validate.shapes               # Rung C (~10 s)
myenv/Scripts/python.exe -m validate.refusals             # Rung D
myenv/Scripts/python.exe -m validate.minute --backend warp  # Rung E (~50 s), the M8 gate
myenv/Scripts/python.exe -m lbm.runner --demo cylinder    # live window (T007+)

# Phase 2 rungs — each lands with the task that needs it (DOCS/TASKS3.md)
myenv/Scripts/python.exe -m validate.les                  # Rung F on numpy (T201)
myenv/Scripts/python.exe -m validate.les --backend warp   # Rung F on warp (T202) — both are the rung
myenv/Scripts/python.exe -m validate.taylorgreen          # Rung G — analytic decay (T203)
myenv/Scripts/python.exe -m validate.taylorgreen --backend warp   # Rung G on warp — both are the rung
myenv/Scripts/python.exe -m validate.fidelity             # Rung H — the bands (T204), ~50 min on numpy
myenv/Scripts/python.exe -m validate.fidelity --backend warp      # Rung H on warp (~5 min) — both are the rung
myenv/Scripts/python.exe -m validate.fidelity --skip-sweep --skip-cylinder  # the table clauses alone, instant
myenv/Scripts/python.exe -m validate.install              # Rung I — fresh-venv wheel (T205)
myenv/Scripts/python.exe -m validate.drop                 # Rung J — the drop, timed (T209)
myenv/Scripts/python.exe bench.py --backend warp --les    # the closure's cost against BGK (T202)

# the Phase 2 product command (T207+). `python -m flow` and `python -m lbm.runner`
# both survive underneath it, with the knobs it deliberately has not got (D-072).
fengdong                                                  # the window; pip install fengdong

# the product command (T109). python -m lbm.runner is kept for the solver-level
# knobs this one deliberately does not have (D-072).
myenv/Scripts/python.exe -m flow --shape tests/data/shapes/disc.png \
    --fluid water --speed "5 mm/s" --size "2 cm" --out wake.mp4
myenv/Scripts/python.exe -m flow --shape ... --explain     # plan, exit 0, runs nothing

# D-038's own case, refused by Phase 1 and RUN by Phase 2 (T204, D-093): the
# closure engages, the run completes, and it reports `illustrative` — a moving
# picture and no Cd at all (constraint 18).
myenv/Scripts/python.exe -m flow --shape tests/data/shapes/disc.png \
    --fluid air --speed "20 m/s" --size "1.5 m" --no-live --backend warp
```

### Issue queue

Problems found while testing or running are **queued locally, never filed automatically**.

```bash
myenv/Scripts/python.exe -m tools.issues list                     # the queue
myenv/Scripts/python.exe -m tools.issues add --title "[core] ..." --body "..." --location "lbm/core.py:42"
myenv/Scripts/python.exe -m tools.issues drop <id> --reason "..."  # not worth filing
myenv/Scripts/python.exe -m tools.issues sync --dry-run            # what would be pushed
myenv/Scripts/python.exe -m tools.issues sync                      # push to GitHub via gh

# wrap any run so a non-zero exit queues an issue with the output tail
myenv/Scripts/python.exe -m tools.issues capture --source validate -- myenv/Scripts/python.exe -m validate.cylinder
```

- Queue is `DOCS/ISSUES.jsonl`, one JSON object per line, **committed**. Entries are deduped by a
  `sha1(source|location|title)` fingerprint with numbers and paths folded out, so re-running a
  failing rung bumps `count` instead of appending.
- `pytest` queues every distinct failure by itself (`tests/conftest.py`). Disable for a run with
  `--no-issue-capture` or `LBM_ISSUE_CAPTURE=0`.
- `sync` is the only thing that talks to GitHub. It needs the `gh` CLI authenticated
  (`winget install --id GitHub.cli -e && gh auth login`); without it, entries stay queued.
- Slash commands: `/file-issue <description>` to queue, `/sync-issues` to review then push.
- **A failing rung that blocks the live task is a `DOCS/STATE3.md` § Blockers entry, not a queued
  issue.** The queue is for things the work continues without.

`myenv/` is the project venv (Python 3.11, numpy 2.4, matplotlib 3.11, pillow). It is gitignored.
Adding a dependency (pygame, imageio, pytest) means `myenv/Scripts/pip.exe install <pkg>` **and** a
line in `DOCS/STATE3.md` § Environment — and, from T205, a matching entry in `pyproject.toml`.

## Module map

| Module | Responsibility | Lands in |
|---|---|---|
| `lbm/backends/` | `Backend` protocol (kernels, boundaries, allocation, transfers), the registry, one module per compute target | T101, T102, T103 |
| `lbm/core.py` | D2Q9 constants, macroscopic, equilibrium, collide, stream | T001, T002 |
| `lbm/boundary.py` | bounce-back, walls, body force, inlet, outlet | T002, T005 |
| `lbm/geometry.py` | mask from primitives / PNG / SVG, sanity checks | T004, T009 |
| `lbm/probe.py` | vorticity, drag, lift, Strouhal, residuals | T005 |
| `lbm/runner.py` | continuous loop, ring buffer, checkpoint / restart, `python -m lbm.runner` CLI | T006, T011 |
| `lbm/render.py` | field -> RGB, diverging colormap, fixed limits | T007 |
| `lbm/record.py` | MP4 / GIF writer, headless sink, tee | T011 |
| `lbm/units.py` | physical <-> lattice conversion | T009 |
| `flow/quantity.py` | units the user types -> SI, one dimension table | T104 |
| `flow/fluids.py` | the cited fluid library | T104 |
| `flow/autoconfig.py` | physics in, every solver parameter out; the guardrails | T105 |
| `flow/diagnose.py` | refusals in prose, suggestions that run, live divergence probe | T106 |
| `flow/prepare.py` | picture -> runnable body mask; repair, refusal, the Q-102 thin-branch metric | T107 |
| `flow/case.py` | the front door: `Case.from_image` / `from_array`, `explain()`, `plan`, `run()` | T108 |
| `flow/report.py` | `Result` — Cd/Cl/St/convergence, the printed summary, the plot, `save()` | T108 |
| `flow/cli.py` | `python -m flow` — the flags, the exit codes; `flow/__main__.py` is the entry point | T109 |
| `flow/fidelity.py` | the three bands, decided from the eddy viscosity a run generated | T204 |
| `fengdong/widgets.py` | the closed widget set: label, text field, dropdown, button, drop target, panel | T206 |
| `fengdong/app.py` | the window, the event loop, the panels; `fengdong/__main__.py` is the entry point | T207, T208 |
| `validate/*.py` | the rungs, each printing pass/fail; all take `--backend` | T002, T003, T007, T008, T103 |
| `validate/minute.py` | Rung E — the whole product path, timed from process start | T110 |
| `validate/les.py` | Rung F — `Cs = 0` is bitwise BGK; Rung 3 survives the closure | T201, T202 |
| `validate/taylorgreen.py` | Rung G — the closure adds the viscosity it claims, against an exact solution | T203 |
| `validate/fidelity.py` | Rung H — every band's claim, machine-checked | T204 |
| `validate/install.py` | Rung I — a built wheel into a fresh venv, no repo on the path | T205 |
| `validate/drop.py` | Rung J — a dropped picture to Rung 3's bands, timed | T209 |

### Everything else at the root

Not modules — the places experiments and their leavings go, so the repo root stays
the map and not the dumping ground. Nothing here is imported by `lbm/` or `flow/`.

| Path | What belongs there |
|---|---|
| `bench.py` | the steps/s table; stays at the root, cited above. `--les` (T202) A/Bs the closure against plain BGK on `--backend`, in alternating rounds |
| `pyproject.toml` | the `fengdong` distribution (T205): packages `lbm`, `flow`, `fengdong`; console entry point `fengdong`. Its runtime dependencies must match `DOCS/STATE3.md` § Environment exactly, and a test asserts it |
| `scripts/` | visualisation drivers on top of `flow` — `slowmo`, `streamlines`, `windtunnel`. They change how a run is **drawn or paced**, never what it computes; every solver parameter still comes from `flow.autoconfig.plan`. Each puts the repo root on `sys.path` itself, so cwd does not matter. See `scripts/README.md` |
| `examples/shapes/` | ad-hoc geometry for demos and issue repros. **Not** `tests/data/shapes/` — `validate/shapes.py` (Rung C) iterates every image in that one and `tests/test_prepare.py` cross-checks it against its `generate.py`, so a picture added there changes what a rung measures |
| `outputs/` | rendered videos, GIFs and frame dumps. Gitignored; nothing reads from it |

A throwaway probe script is scratch, not a root file — it does not get committed.

## Current state

**Phase 0 is complete.** T001 -> T011 done; **M4 reached** (2026-08-13). `lbm/` has `core`,
`boundary`, `geometry`, `probe`, `runner`, `render`, `record`, `units`, `backends/`; `validate/` has
`poiseuille`, `cavity`, `cylinder`, `polygons`; `bench.py` at the root prints the steps/s table.
Rungs R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **the ladder is complete**, and it stays a gate for every
task above it.

**Phase 1 is complete. M8 reached (2026-08-27, session 22).** T101 -> T110 all done. The product
layer is `flow/` — `quantity`, `fluids`, `autoconfig`, `diagnose`, `prepare`, `case`, `report`,
`cli` — on a Warp GPU backend behind `lbm/backends/`, and `python -m flow` is the thing a person
runs. Rungs **A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩**, on both backends where both apply. Spec
`DOCS/IDEA3.md` · plan `DOCS/PLAN2.md` · backlog `DOCS/TASKS2.md` · **live status
`DOCS/STATE2.md`**.

**The claim the phase is judged on, measured**: `python -m validate.minute --backend warp` reaches
`Cd` **1.4040** (band 1.25–1.45) and `St` **0.1672** (band 0.155–0.175) — Rung 3's published bands,
unwidened, through `flow.Case` from a committed PNG and three physical numbers — in **49.5 s of
wall clock from process start**, against a 60 s limit. Conditions, per D-035: AMD Ryzen 7 5800H at
3201 of 3201 MHz on mains, NVIDIA RTX 3050 Laptop GPU, driver 592.82.

**What session 22 changed under the product, and why it matters to anything built next**: the
chooser's domain is now Rung 3's own (24 D span, 8 D upstream — **D-075**, superseding D-059),
which is ~2.8x the cells, so the probe cadence dropped to 10 samples per convective time
(**D-076**) and the default run length rose to 80 convective times (**D-079**). `flow/` is
therefore *slower per case and correct*, where before it was fast and 14% high on drag. Two
measurement harnesses were repaired rather than re-tuned: `_RATE_TABLE` gained 160k / 400k anchors
(**D-077**) and Rung D's cost check gained rounds (**D-078**).

**Phase 2 is live — FengDong** (风洞, *wind tunnel*). Planned in session 23; **T201 landed in
session 24, T202 in session 25, T203 in session 26 — and with it M9 — and T204 in session 27, with
M10**. **T205 is next.** Three deliverables: a **Smagorinsky closure** on the existing BGK collision
(both backends, defaulting off), the **fidelity bands** that make it safe to ship, and a **pygame
desktop application** shipped as `pip install fengdong`. Two of the three are done. Rungs
**F 🟩 · G 🟩 · H 🟩 · I ⬜ · J ⬜**, milestones **M9 🟩 · M10 🟩** – **M12**.
Spec `DOCS/IDEA4.md` · plan `DOCS/PLAN3.md` · backlog `DOCS/TASKS3.md` · **live status
`DOCS/STATE3.md`**.

**T201, done (session 24):** the closure is in `lbm/core.py` — `smagorinsky_tau_eff` (the primitive)
and `smagorinsky_omega` (its reciprocal), `CS_SMAG_LITERATURE = 0.17`, `SMAG_Q_COEFF = 18 sqrt(2)` —
with keyword-only `cs_smag` on `collide` / `collide_stream`, `lbm.probe.eddy_viscosity`,
`SimConfig.cs_smag`, and the NumPy backend implementing it. **D-085** fixes the normalisation,
**D-086** makes constraint 19 an explicit branch rather than a zero-valued term, **D-087** keeps one
frozen Phase 1 oracle.

**T202, done (session 25): Rung F is green on both backends and the closure is on the GPU.** The
Warp backend has `_smag_scale_kernel`, `_collide_smag_kernel` and a `_collide_bb_smag_kernel` that
folds the reduction into the fused pass; `validate/les.py` takes `--backend` and gained a
cross-backend clause; `bench.py` takes `--les`. **Q-201's answer, by measurement: two compiled
kernels, not one guarded branch** — `cs_smag = 0` launches the untouched Phase 1 kernel, so bitwise
degeneracy is by construction (**D-088**), and the fold is **D-089**. Measured: `cs_smag = 0` bitwise
on both paths after 1000 steps of Rung 3's case (worst |diff| **0.000e+00**); Rung 3 at `Cs = 0.17`
prints Cd **1.4143**, St **0.1719** on *both* backends; cross-backend with the closure **on**, worst
kernel **2.980e-08** against 1e-6 and whole step **9.611e-06** against 1e-4 — Rung A's own bars,
unwidened. The closure costs **1.6% / 9.3% / 9.8%** of the BGK step rate at 40k / 1M / 2M cells.

**T203, done (session 26): Rung G is green on both backends, and F + G is M9.**
`validate/taylorgreen.py` decays an exact 2D Taylor–Green vortex on a doubly periodic domain with no
bodies and fits `ln E` against `t`. Measured on 64x64 at `tau = 0.52`, `u0 = 0.08`: at `Cs = 0` the
decay returns `(tau - 0.5)/3` to **0.2303%** against Rung 1's own 1% bar, with `<nu_t>` exactly zero;
at `Cs = 0.17` it returns `nu + <nu_t>` to **1.1547%** against 2%, and **bare `nu` misses by
3.0178%** — so deleting the closure term breaks the clause instead of passing it (**D-091**).
**Q-202's answer: `<nu_t>/nu` = 1.8418%** on a resolved flow, an order of magnitude below Rung 3's
wake. The trap D-091 records: that ratio is a *design output* — it scales as `u0 / (L nu)`, and on a
more resolved case the closure is more inert, not less, so the case had to be **sized** for the bar
to have teeth. Taylor–Green has `S_xy = 0`, so the decay responds to the **dissipation-weighted**
`<nu_t^3>/<nu_t^2>`, which is **1.7780x** the domain mean; against *that* the measured excess is
**0.9972**. The M9 gate was run in full — all nine existing rungs re-run on both backends with every
published digit unmoved, and `bench.py --backend warp --les` reading **3504.0 / 661.6 / 403.7** against
floors **3116 / 568 / 331**.

**T204, done (session 27): the bands, and with them M10 — and the wall this phase exists to remove
is gone.** `flow/fidelity.py` is the judgement layer: `Band`, `band_for(plan, nu_t_max=None)`,
`sentence(band)` and `Qualified`, with `RE_3D_ONSET = 200` cited to Williamson (1996) rather than
chosen. `flow.autoconfig.plan` engages the closure below `TAU_FLOOR` instead of refusing (**D-093**),
`Plan` gains `cs_smag` and its `why`, `Result` gains `fidelity` — and **constraint 18 is implemented
in exactly one place**, `Result.__post_init__`, which withholds every claim the band forbids so that
no printer, dict or MP4 comment has anything to leak. **Q-203's answer (D-095): a *qualified* `Cd` in
the qualitative band, stability-only in the illustrative one**, and the evidence is measured, not
argued — Rung 3's own case with the closure on sits at `max(nu_t)/nu` **0.1057** (inside the
qualitative band, whose boundary is 0.1) and still prints Cd **1.4143**, St **0.1719** against the
published, unwidened bands, identically on both backends. **D-094** is the price: on a closure-on run
`Monitor`'s speed and mass wires move from the accuracy bound to the meaning bound — `1/sqrt(3)` and
half the domain's mass — because D-038's case is **finite, flat and linear** for all 48000 steps
(peak `|u|` 0.20 from step 4000 on, mass leaking 0.11% per 1000 steps to 5.24%) while the Phase 1
wires called that "growing without bound" at step 75. Every crossing of the narrow bounds is still
counted and printed. **Every one of the eleven existing rungs was re-run and no published digit
moved** — R1 · R2 · R3 · R4 · A · B · C · D · E · F · G, on both backends where both apply, with
Rung B's numpy half still running at checkpoint time and its warp half green (`DOCS/STATE3.md`
§ Provenance).

**The measured claim T204 is judged on**: `python -m flow --shape disc.png --fluid air --speed
"20 m/s" --size "1.5 m"` — Re 2e6, the case **D-038** refused and **D-074** re-refused — now **exits
0**, reports **illustrative**, and prints **no `Cd` at all**. Rung H runs that literal command.

**What Phase 2 is for, in one sentence**: `idea.md`'s success test says *"opens the tool, drags in a
picture"* and D-044 deferred that; everything beneath it is now validated by twelve rungs, so this
phase spends them. The closure existed because **D-038** and **D-074** — the first case any plausible
user asks for — *were* refused; since T204 they are not, and the fidelity band is what stands in the
refusal's place.

**The thing this phase can most easily get wrong**, stated so a future session cannot claim it was
not warned: the closure buys **stability, not fidelity**. The cylinder wake is three-dimensional above
Re ≈ 190, so a 2D answer beyond that is wrong about the flow and no 2D closure repairs it
(**D-082**). Constraint 18 and Rung H are the interlock. Widening a band to make a number reportable
is the one move that is out of bounds.

**Why not XLB and why not 3D**, both decided in session 23 by measurement rather than by inheriting
`idea.md`'s roadmap (**D-080**): XLB 0.3.1 installs and runs on this box but does not import against
warp-lang ≥ 1.14 (ours is 1.16), its monolithic stepper is the wrong shape for D-054's per-kernel
seam, and its 2D-relevant gift is ~20 lines of our own `collide`. 3D at our own quality floor is
28.4 GB per buffer on a 4 GB card. Both stay deferred; both keep their reasons on file.

**The 20 hard constraints above are the Phase 2 list.** Phase 1's sixteen all carry: constraint 1 was
rewritten by **D-081**, constraints **17–20** were added by **D-082** and **D-083**, and the fate of
all sixteen is recorded in `DOCS/STATE3.md` § Constraint fate table. The constraints section is the
authority; the decisions are the record of why each reads the way it does.
