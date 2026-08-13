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

1. **D2Q9, BGK single relaxation time, bounce-back walls.** No MRT, no cumulant, no curved/
   interpolated boundaries, no turbulence model in Phase 0. Deferred is not the same as forgotten.
2. **Viscosity is not a free parameter.** `nu = cs2 * (tau - 0.5) = (tau - 0.5) / 3`. Never expose a
   `nu` setter that doesn't go through `tau`. `tau -> 0.5` means `nu -> 0` means the sim blows up.
3. **Lattice velocity stays under 0.1.** Compressibility error scales as Mach squared. Any config
   path that can produce `|u| >= 0.1` must warn at setup, not at `nan` time.
4. **State is `f` of shape `(9, ny, nx)`**, index order `(direction, y, x)`. `float32`. The nine
   constants (`e`, `w`, `opp`, `cs2`) live in `lbm/core.py` and are imported from there — never
   redefined locally.
5. **The validation ladder is non-negotiable and ordered.** Rung 1 Poiseuille, Rung 2 cavity vs
   Ghia, Rung 3 cylinder Re 100, Rung 4 square cylinder. Each rung is a script in `validate/` that
   prints pass/fail. **A wrong sim that looks plausible is the main failure mode of this project.**
   Do not start rung N+1 while rung N fails.
6. **Do not optimise before Rung 3 passes.** No fused kernels, no Numba, no GPU, no clever
   vectorisation tricks until the cylinder shows the right Strouhal number.
7. **Simulation and rendering are decoupled.** One rendered frame is many timesteps.
   `steps_per_frame` is **computed** from target playback speed — never hardcoded to 20.
8. **Never block the sim on the display.** Ring buffer between them. If it fills, drop *display*
   frames, never simulation steps.
9. **Draw vorticity, not speed.** Diverging colormap, symmetric **fixed** limits. Speed magnitude is
   a grey smear; per-frame autoscaled limits flicker.
10. **One `render()`, three sinks** (live / record / headless). Do not write three renderers.
11. **Restart must be bit-identical.** `f`, `mask`, and step count are the entire state. Pickle every
    N steps; resume produces a bit-identical continuation, and that is a tested claim.
12. **Geometry is one boolean array**, `solid`, shape `(ny, nx)`. Solid at least 3 cells thick
    (detect and warn — thinner leaks through bounce-back), object ≥8 diameters from the outlet,
    blockage ratio under ~10%.

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
| `lbm/core.py` | D2Q9 constants, macroscopic, equilibrium, collide, stream | T001, T002 |
| `lbm/boundary.py` | bounce-back, walls, body force, inlet, outlet | T002, T005 |
| `lbm/geometry.py` | mask from primitives / PNG / SVG, sanity checks | T004, T009 |
| `lbm/probe.py` | vorticity, drag, lift, Strouhal, residuals | T005 |
| `lbm/runner.py` | continuous loop, ring buffer, checkpoint / restart, `python -m lbm.runner` CLI | T006, T011 |
| `lbm/render.py` | field -> RGB, diverging colormap, fixed limits | T007 |
| `lbm/record.py` | MP4 / GIF writer, headless sink, tee | T011 |
| `lbm/units.py` | physical <-> lattice conversion | T009 |
| `validate/*.py` | the four rungs, each printing pass/fail | T002, T003, T007, T008 |

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
**live status `DOCS/STATE2.md`**. The current task is `T101` (backend seam).

**The 12 hard constraints above are Phase 0's.** Their Phase 1 fates are decided in
`DOCS/STATE2.md` **D-046** — nine permanent, three rewritten (1, 4, 11), one retired (6), four
added (13–16). Until T101 folds that table into this file, **D-046 is the authority** where the two
disagree.
