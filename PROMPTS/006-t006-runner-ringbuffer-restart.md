# Session 6 — T006: runner — decoupled loop, ring buffer, restart

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). Ship Phase 0, validate it, move on.

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-021), session log. **D-020 is your task's spine** — it fixes the timestep order and
   names every buffer the runner has to own. D-006 (preallocated outputs), D-011 (the original
   timestep order) and D-012/D-014 (the `float32` residual floor) are the others you sit inside.
3. `DOCS/TASKS1.md` § T006 — the task contract, in full. Also read § T005's outcome note: the
   boundaries and probes are done and the runner's job is to call them in the right order with the
   right buffers.
4. `DOCS/IDEA2.md` § **Continuous simulation — the part that matters most** (the whole section), and
   § **The method, in the order the code runs it** so the loop body has a reference.
5. `DOCS/PLAN1.md` § Session map and § Risks — T006 is session 6 of 11. § Risks assigns it "live
   display drags the physics down": the ring buffer is built **before** rendering exists, on purpose,
   because a fake slow sink proves frame-dropping far more cleanly than a real window does.
6. `lbm/core.py`, `lbm/boundary.py`, `lbm/geometry.py`, `lbm/probe.py` — all exist and all work.
   `lbm/boundary.py`'s module docstring holds the timestep order you are implementing.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 5: **T005 is `done`**, every acceptance criterion run and green.
  `lbm/boundary.py` gained `inlet_profile`, `inlet_velocity` (Zou–He) and `outlet_zero_gradient`
  (copy by default, convective when given `prev`); `lbm/probe.py` is new with `vorticity`,
  `boundary_links`/`BoundaryLinks`, `forces`, `strouhal` and `residual`.
  `myenv/Scripts/python.exe -m pytest` → `152 passed`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 ⬜ · R4 ⬜ — **you build neither.** T006's gate is unit tests,
  with the bit-identical restart test as the one that matters. Rung 3 is T007.
- **Milestone reached:** **M2** (2026-08-10). **M3 is T007, not this session.**
- **Completed tasks:** T001, T002, T003, T004, T005.

## Your task this session

**T006 — Runner: decoupled loop, ring buffer, restart.** One task, this session only.

Run this first:

    /start-task T006

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: the simulation becomes a stream rather than a batch job — physics at its own rate, frames
pulled off at another, and a run that can die and resume bit-identically.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `Sim.step()` runs one full timestep; all buffers preallocated in `__init__`, verified by a test asserting no change in `f.__array_interface__['data']` and (via `tracemalloc`) no growth over 1000 steps.
- [ ] `steps_per_frame` is **computed** from target physical playback speed, grid size and `dt` — a function with a docstring showing the arithmetic, not a constant.
- [ ] `RingBuffer(maxlen)` drops the **oldest frame** when full and increments a `dropped` counter; a test with a deliberately slow sink confirms `dropped > 0` while `step_count` is unaffected.
- [ ] `Sink` is an abstract base with `push(frame)` and `close()`; `NullSink` implemented. Live/record sinks are T007/T011.
- [ ] `save_checkpoint(path)` pickles exactly `f`, `solid`, `step_count`, and the config.
- [ ] **Bit-identical restart is a test:** run 500 steps, checkpoint, run 500 more, record `f`; reload the checkpoint, run 500, and assert `np.array_equal` with the recorded `f`.
- [ ] Auto-checkpoint every N steps, N configurable, off by default.
- [ ] `pytest tests/test_runner.py` green; Rungs 1–2 still green.

Gate for this task is **unit tests plus both existing rungs still green** — there is no Rung 3 yet.

### Constraints that bite on this task

- **Constraint 7** — simulation and rendering are decoupled; one rendered frame is many timesteps and
  `steps_per_frame` is **computed** from the target playback speed. **A hardcoded 20 fails this task.**
- **Constraint 8** — never block the sim on the display. The ring buffer sits between them, and when
  it fills it drops *display frames*, never simulation steps. That is the assertion, not a comment.
- **Constraint 11** — restart must be bit-identical. `f`, `mask` and step count are the entire state.
  It is a **tested claim**, so: no `float64` intermediates that round differently on resume, no RNG
  anywhere in the step path, and no reliance on buffer contents that `__init__` leaves uninitialised.
- **Constraint 4** — `f` is `(9, ny, nx)`, `float32`, index order `(direction, y, x)`; the mask is
  `(ny, nx)`. Constants come from `lbm/core.py` and are never redefined.
- **Constraint 6** — do not optimise before Rung 3 passes. The loop calls the existing functions in
  order; no fusing, no Numba, no threading the physics.
- **Constraint 10** — one `render()`, three sinks. You build the **abstract** `Sink` and `NullSink`
  only; the live sink is T007 and the record sinks are T011.
- **Coding conventions** — type hints with array shapes documented, docstrings citing the
  `DOCS/IDEA2.md` section, `float32`, **preallocate — never allocate inside the step loop**, no
  physical units inside `lbm/`.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Q-002** (open) — SVG rasterisation dependency for T009 not chosen. Not yours, not blocking.
- **D-020** — **this is the decision T005 made for you.** The timestep order, in full:
  `copy f_pre` → `macroscopic` → `force_velocity_shift` → `equilibrium` → `collide` →
  `apply_body_force` → `bounce_back` → `copy f_bb` → `stream` → `outlet_zero_gradient` →
  `inlet_velocity`. `f_pre` is the **pre-collision** copy that `bounce_back` consumes; `f_bb` is a
  **second** `(9, ny, nx)` buffer holding the **pre-stream** state, which is what
  `lbm.probe.forces(f_pre=f_bb, f_post=f, ...)` needs. Two buffers, two meanings, do not merge them.
  The open boundaries go after `stream` because `stream` is periodic in `x`.
- **D-021** — `outlet_zero_gradient` reflects **35%** of a pressure pulse in its default copy mode and
  **0.6%** in convective mode. The runner should own the `(9, ny)` `prev` buffer and pass it, i.e.
  use the convective form. `lam` defaults to `sqrt(CS2)`.
- **D-006** — hot functions take optional preallocated outputs. Everything the loop calls already
  supports it: `macroscopic(f, rho, u)`, `equilibrium(rho, u, feq, work)`,
  `force_velocity_shift(rho, u, g, work)`, `apply_body_force(f, rho, u, tau, g, work)`,
  `inlet_velocity(f, u_in=..., work=...)` (a `(5, ny)` buffer), `vorticity(u, out=, work=)`,
  `residual(u_now, u_prev, U, solid=, work=)`. The runner is the buffer owner.
- **D-011** — `bounce_back`'s `f_pre` is the copy taken **before collision**. Superseded by nothing;
  D-020 only adds the second copy.
- **D-012 / D-014** — `float32` gives `u` a per-step round-off floor near `1.7e-6`, and `u` on solid
  cells is meaningless, so any residual the runner reports must be fluid-cells-only and compared over
  an interval of `k` steps if a tolerance below the floor is wanted.
- **D-003** — `myenv/Scripts/python.exe` is the canonical interpreter. Never bare `python`.
- **D-019** — characteristic length `D` comes from `lbm.geometry.bounding_box`; if the config carries
  a `D`, derive it there rather than inventing a second definition.

### Before you start

- No new package needed. `myenv` has: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  Python 3.11.15. (`pygame` lands in T007, `imageio[ffmpeg]` in T011.) If you do install something,
  add a row to `DOCS/STATE1.md` § Environment in the same session.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves — there is no
  packaging config in Phase 0.
- Rungs 1 and 2 must be green before you start and again before you finish. Rung 2's Re 1000 case
  takes about 150 s; the whole ladder is roughly 3 minutes.

## Scope discipline

Work only what's in the contract. Rendering, the live pygame sink and the cylinder benchmark are
**T007**; the square cylinder is **T008**; PNG/SVG masks and physical units are **T009**; MP4/GIF
sinks and the CLI are **T011**. Something else genuinely needs doing? `/new-task` it. If it is under
`DOCS/IDEA2.md` § Deliberately deferred, the answer is no.

The temptation this session is to draw something. Resist it — `NullSink` and a fake slow sink are
what prove constraint 8, and a window in the way makes that proof harder, not easier.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it. The restart test is the one that
   earns the session.
2. Run `/validate`. Rungs 1 and 2 must both be reported with their measured numbers; R3–R4 stay ⬜.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 152 existing tests
   must still pass.
4. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T006 `in_progress`.
5. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes
   `PROMPTS/007-t007-*.md` for the next session. Do not end the session without it.
