# CLAUDE.md — Fluid Mech / Phase 0

**Project, one line:** A validated, continuously-running 2D fluid simulator in pure NumPy —
D2Q9 lattice Boltzmann, any shape from a boolean mask, live streaming visual plus recordable video.

**Phase 0 is not the product.** It exists so we understand LBM well enough to design the layer
above it (see root `idea.md` and `README.md` for that product). Ship Phase 0, validate it, move on.

**Full Phase 0 spec is `DOCS/IDEA2.md`; Phase 1's is `DOCS/IDEA3.md`.** Don't re-derive decisions
already made there — cite them. If anything here conflicts with the spec of the live phase, **the
spec wins**; log the conflict in `DOCS/STATE2.md` § Decisions rather than silently picking one.

The existing `Navier-Fluid-Equation/` directory is **prior work** — potential-flow / panel-method
scripts. It is not part of the LBM solver. Reuse its polygon-vertex code (`polygonsDemo.py`,
`panels.py`) for `lbm/geometry.py` primitives; do not modify it otherwise.

---

## Hard constraints

Load-bearing decisions, not optimizations. A design that drifts from these is wrong even if it runs.

**These are the Phase 1 constraints.** D-046 decided the fate of each of Phase 0's twelve — nine kept
verbatim, three rewritten, one retired, four added — and T101 folded that table in here, which is why
`DOCS/STATE2.md` **D-046** is no longer the authority: this list is. D-046 remains the record of *why*
each one reads the way it does, and the retired one is kept below, struck, rather than deleted.

1. **The physics is D2Q9, BGK single relaxation time, bounce-back walls, and it does not change in
   Phase 1.** No MRT, no cumulant, no curved/interpolated boundaries, no turbulence model. The
   *implementation* may move to a GPU backend; the arithmetic it transcribes may not change. Deferred
   is not the same as forgotten. *(Rewritten by D-046.)*
2. **Viscosity is not a free parameter.** `nu = cs2 * (tau - 0.5) = (tau - 0.5) / 3`. Never expose a
   `nu` setter that doesn't go through `tau`. `tau -> 0.5` means `nu -> 0` means the sim blows up.
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
   task. Phase 1 adds Rung A parity, B auto-config, C shapes, D refusals, E the minute (**D-047**).
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
    `flow/` colours nothing.
11. **Restart must be bit-identical within a backend.** `f`, `mask`, and step count are the entire
    state. Pickle every N steps; resume produces a bit-identical continuation, and that is a tested
    claim. **Across** backends it is a printed tolerance (T103), because float ordering differs on a
    GPU and no test should pretend otherwise. *(Rewritten by D-046.)*
12. **Geometry is one boolean array**, `solid`, shape `(ny, nx)`. Solid at least 3 cells thick
    (detect and warn — thinner leaks through bounce-back), object ≥8 diameters from the outlet,
    blockage ratio under ~10%. Phase 1 *repairs* where it can rather than only warning (T107).
13. **No lattice quantity in any public `flow/` signature.** No `tau`, no lattice `U`, no
    `steps_per_frame`, no cell counts. The inputs are a picture, a fluid, a speed, a size. Everything
    else is derived and **printed**. *(New in Phase 1.)*
14. **Every refusal names a fix, and the fix is machine-checked.** A refusal carries `reason`,
    `quantity`, `value`, `limit`, `suggestions`; Rung D feeds the tool's own top suggestion back
    through the planner and runs it. A suggestion that does not fix its case is a failing test.
    *(New in Phase 1.)*
15. **`flow/` may import `lbm/`; `lbm/` may never import `flow/`,** and a test asserts it. That
    one-directional import is what makes the Phase 3 XLB swap a substitution rather than a rewrite.
    *(New in Phase 1.)*
16. **No silent substitution.** A run that differs from what was asked says so in every artifact it
    produces — the printed summary, the report, and the video metadata — via `substituted=True`.
    *(New in Phase 1.)*

Constraints 13–16 are enforced by tests that land with the code they govern (`flow/`, T104 onward);
until then they are the design rule that code has to be written to satisfy, not a dead letter.

---

## Session protocol

**Follow this every session. No exceptions.**

**Phase 1 is live. The live documents are `DOCS/STATE2.md` and `DOCS/TASKS2.md`** — Phase 0's
`old-Docs/STATE1.md` / `old-Docs/TASKS1.md` / `old-Docs/PLAN1.md` are **frozen**: read for history, never edited
(**D-041**). Everywhere below that names a Phase 0 file, read the Phase 1 one instead.

1. **At session start** — read `DOCS/STATE2.md` (all of it) and `DOCS/TASKS2.md` (the row for the
   live task) **before touching any code**. They say what task is live, what's blocked, what the
   previous session actually left behind. `old-Docs/STATE1.md` § Decisions (D-005 … D-040) is still in
   force and is cited by number; read the entry a task names, not the whole file.
2. **Before working a task** — run `/start-task T1XX`. It reads the task contract and the
   `DOCS/IDEA3.md` section it cites, then restates goal + acceptance criteria for confirmation.
3. **One task per session.** `DOCS/PLAN2.md` maps one task to one session deliberately — context
   stays small and each session ends at a validated boundary. Work that turns up mid-stream gets
   `/new-task`, not scope creep.
4. **At session end** — run `/checkpoint`. Never end a session without it. It updates
   `DOCS/STATE2.md`, syncs `DOCS/TASKS2.md`, and writes the next paste-ready prompt into
   `PROMPTS/`.

## Coding conventions

- **Type hints everywhere.** Arrays annotated with intent: `NDArray[np.float32]`, and document
  shape in the docstring (`(9, ny, nx)`).
- **Docstrings cite the spec** — name the `DOCS/IDEA2.md` section so the reasoning is one hop away.
- **Preallocate. Never allocate inside the step loop.** Buffers are created once by the runner and
  passed in, or held on the sim object.
- **`float32` throughout.** Halves the bandwidth; accuracy is fine for this.
- **No physics constant twice.** `e`, `w`, `opp`, `cs2` from `lbm/core.py` only.
- **Physical units never reach the solver.** `lbm/units.py` converts at the boundary; everything
  inside `lbm/` is lattice units.
- Stubs raise `NotImplementedError("see DOCS/TASKS2.md T0XX")` until their task lands.
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
myenv/Scripts/python.exe -m validate.refusals             # Rung D
myenv/Scripts/python.exe -m lbm.runner --demo cylinder    # live window (T007+)
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
- **A failing rung that blocks the live task is a `DOCS/STATE2.md` § Blockers entry, not a queued
  issue.** The queue is for things the work continues without.

`myenv/` is the project venv (Python 3.11, numpy 2.4, matplotlib 3.11, pillow). It is gitignored.
Adding a dependency (pygame, imageio, pytest) means `myenv/Scripts/pip.exe install <pkg>` **and** a
line in `DOCS/STATE2.md` § Environment.

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
| `validate/*.py` | the rungs, each printing pass/fail; all take `--backend` | T002, T003, T007, T008, T103 |

## Current state

**Phase 0 is complete.** T001 → T011 done; **M4 reached** (2026-08-13). `lbm/` has `core`,
`boundary`, `geometry`, `probe`, `runner`, `render`, `record`, `units`; `validate/` has
`poiseuille`, `cavity`, `cylinder`, `polygons`; `bench.py` at the root prints the steps/s
before/after table.
Rungs R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **the ladder is complete**. The performance budget is met
(696.7 / 161.7 / 16.8 steps/s at 40k / 160k / 1M cells, floors 400 / 120 / 15). `python -m
lbm.runner` turns a PNG plus physical numbers into an MP4 in one command.

**Phase 1 is live** (planned in session 12, 2026-08-13): the product layer above the solver —
`flow/` package plus CLI, on a Warp GPU backend, ten tasks `T101` → `T110`, five new rungs
A–E, milestones M5 → M8. Spec `DOCS/IDEA3.md` · plan `DOCS/PLAN2.md` · backlog `DOCS/TASKS2.md` ·
**live status `DOCS/STATE2.md`**. Rungs **A 🟩** · B ⬜ · C ⬜ · D ⬜ · E ⬜; **M5 reached**
(2026-08-18).

**T101 → T103 are done; M5 is reached** (session 15, 2026-08-18). `lbm/backends/` holds the
`Backend` protocol, the registry, `numpy_backend` and `warp_backend`; `SimConfig.backend` selects
between them and `lbm/runner.py` imports no kernel and no boundary. **The whole timestep runs on the
GPU**: the protocol now covers allocation and the two general transfers as well as the kernels, the
four boundaries and both halves of the Guo body force, backend arrays are opaque handles, and `Sim`
owns device state — host reads go through `host_f()` / `host_u()` / `host_rho()` / `host_f_bb()` on
frame and probe cadence, never per step.

**Rung A is green in full**: every kernel and every boundary within **5.96e-08** of NumPy against a
1e-6 bar, whole-step **9.6e-06** at 1000 steps against 1e-4 and *not compounding*, a checkpoint
written on `warp` resuming on `numpy`, and restart bit-identical within a backend. All four Phase 0
rungs pass with `--backend warp` printing session 11's digits. `bench.py --backend warp` clears
**4155 / 757 / 441 steps/s** at 40k / 1M / 2M against floors of 2000 / 250 / 150 — 5x / 33x / 53x
NumPy — using 391 MiB of the 4 GB card at 2M cells. The current task is `T104` (physical quantities
and the fluid library), the first of the `flow/` package.

**The 16 hard constraints above are the Phase 1 list**, folded in from `DOCS/STATE2.md` **D-046** by
T101 — the constraints section is now the authority, and D-046 is the record of why each one reads
the way it does.
