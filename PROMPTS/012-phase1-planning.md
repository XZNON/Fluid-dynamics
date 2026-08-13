# Session 12 — Phase 1 planning: design the product layer

**This is not a solver task. There is no `T012`.** Phase 0 finished in session 11 and its backlog
(`old-Docs/TASKS1.md`, T001 → T011) is closed. This session **plans Phase 1** and writes the planning
documents; it should end with a plan and a task list, not with a new feature in `lbm/`.

## What this project is

The product (root `idea.md`) is an open-source fluid-dynamics engine that lets anyone drop in a
shape, set a few physical numbers, and watch the flow — without learning CFD first. Its thesis, in
its own words: *"The gap is not the solver. The gap is everything around the solver."*

**Phase 0** — now complete — was a validated, continuously-running 2D fluid simulator in pure NumPy:
D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual plus
recordable video. Its full spec is `DOCS/IDEA2.md`. It existed *so that we understand LBM well
enough to design the layer above it*, and `idea.md` § Roadmap is explicit that this was
non-negotiable: "you cannot design good defaults for a method you have not debugged yourself."

That understanding now exists, and it is written down as **40 numbered decisions** in
`old-Docs/STATE1.md` § Decisions, most of them settled by measurement rather than argument. Phase 1's
job is to turn it into a product layer.

## Read these first, in this order

1. **Root `idea.md`** — in full. The product, the positioning, the roadmap, the risks. Especially
   § What we are actually building (the pipeline diagram), § Why this is defensible (the four moat
   items), § Roadmap (**Phase 1 — 2D engine, continuous**), and § Risks, honestly — where "The
   trap" names exactly the failure mode of this session: *"It is very tempting to keep polishing the
   solver because that part is fun. The solver is not the product."*
2. **Root `README.md`** — the product-level statement of the same thing.
3. `CLAUDE.md` — the 12 hard constraints, the session protocol, the conventions, the module map.
   Note that its § Current state now says Phase 0 is complete. **Some of those constraints are
   Phase 0's and some are permanent** — deciding which is which is part of this session's work.
4. `old-Docs/STATE1.md` — **in full**, and it is long. § Snapshot (the M4 gate output), § Decisions
   D-005 … D-040, § Performance baseline, § Environment, and the eleven session-log entries. This is
   the accumulated understanding Phase 1 is supposed to be designed *from*.
5. `old-Docs/PLAN1.md` — **as a model, not as content.** It is what a good phase plan looked like for
   Phase 0: a task graph, a one-task-per-session map, milestone gates with literal gate commands,
   and a risks table with pressure valves. Phase 1's plan should have the same shape.
6. `DOCS/IDEA2.md` § Deliberately deferred — the list of things Phase 0 said no to (MRT,
   Smagorinsky, curved boundaries, moving objects, thermal coupling, adaptive refinement, 3D, STL,
   GPU, UI). Several of them are Phase 1 or Phase 2 candidates now. Deferred is not forgotten, but
   un-deferring one is a decision to make deliberately and record.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 11: **T011 done**, and with it **M4 — Phase 0 is complete.**
  `lbm/record.py` (`RecordSink`, `HeadlessSink`, `TeeSink`) and the `python -m lbm.runner` CLI
  landed; `imageio[ffmpeg]` was added to `myenv`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **the ladder is complete**, re-run in session 11
  and identical to session 10 to every printed digit: R1 L2 **0.3650%** · R2 **0.75% / 0.42% /
  1.01%** · R3 **St 0.1731, Cd 1.4031 ± 0.0086** · R4 square **Cd 1.5279 ± 0.0271**, polygon
  **Cd 1.4276 ± 0.0226**.
- **Milestones:** M1 (T002) · M2 (T003) · M3 (T007) · **M4 (T011, 2026-08-13)**. M5 (a Warp/Taichi
  port of the kernel, same API) is explicitly *not* Phase 0 and gets its own plan.
- **Completed tasks:** T001 … T011 — all of Phase 0.
- `myenv/Scripts/python.exe -m pytest` → **`367 passed`**.
- **Blockers:** none. **Open questions:** none open — Q-001 … Q-004 are all closed (D-009, D-031,
  D-013, D-036).

### What Phase 0 actually hands you

The API the product layer sits on top of, all lattice units inside, physical units converted at the
boundary by `lbm/units.py`:

| Module | What it gives you |
|---|---|
| `lbm/core.py` | `E`, `W`, `OPP`, `CS2`, `Q`; `macroscopic`, `equilibrium`, `collide`, `stream`, and the fused `collide_stream` |
| `lbm/boundary.py` | `bounce_back`, `moving_wall`, `inlet_profile`/`inlet_velocity` (Zou–He), `outlet_zero_gradient` (convective), the Guo body-force pair |
| `lbm/geometry.py` | `circle`, `rectangle`, `polygon`, `regular_polygon`, `channel_walls`; `from_png`, `from_svg`; `check_mask`, `min_thickness`, `bounding_box`, `strip_solid_border` |
| `lbm/probe.py` | `vorticity`, `forces` (Cd/Cl by momentum exchange), `strouhal`, `residual`, `boundary_links` |
| `lbm/runner.py` | `SimConfig`, `Sim` (owns `f`, `solid`, `step_count` and every buffer), `run`, `RingBuffer`, `Sink`, `steps_per_frame`, `save_checkpoint`/`load_checkpoint`, and the `python -m lbm.runner` CLI |
| `lbm/render.py` | `render(field, limits) -> uint8 (ny, nx, 3)`, `COOLWARM` (257 entries), `LiveSink` |
| `lbm/record.py` | `RecordSink` (MP4/GIF, fixed fps, never drops), `HeadlessSink` (numbered PNGs), `TeeSink`, `check_ffmpeg`, `frame_count` |
| `lbm/units.py` | `LatticeUnits.from_physical(...)` → `dx`, `dt`, `tau`, `U`, `Re`; `stability_note()`, `resolution_for_tau`, `summary()` |
| `validate/` | `poiseuille`, `cavity`, `cylinder`, `polygons` — the four rungs, each printing PASS/FAIL |
| `bench.py` | steps/s per grid, alternating-round A/B (**D-035**) |

Performance, at the CPU's rated 3201 MHz on mains: **696.7 / 161.7 / 16.8 steps/s** at 40k / 160k /
1M cells against floors of 400 / 120 / 15. On battery at ~56% clock the same build reads
402.7 / 117.0 / 16.8 and the 160k case misses its floor — quote `Win32_Processor.CurrentClockSpeed`
with any number you measure (**D-035**). At 1M cells `equilibrium` is **over half** the step
(39.9 ms of ~66 ms), which is the obvious first target for a GPU port.

## Your task this session

**Plan Phase 1.** One session, and its deliverable is documents.

There is no `/start-task` to run — that command reads a task contract out of `old-Docs/TASKS1.md`, and
Phase 1 has no task file yet. This session *creates* it. `/create-spec` exists and is the closest
fit if you want a spec file and a branch; use it or write the documents directly, but decide with
the user before generating a whole plan.

### What the session should produce

1. **A Phase 1 specification** — the analogue of `DOCS/IDEA2.md` for the product layer (suggest
   `DOCS/IDEA3.md`; confirm the name with the user). What Phase 1 *is*, in enough detail that a task
   list falls out of it, and — as important — what it deliberately is not.
2. **A phase plan** — the analogue of `old-Docs/PLAN1.md`: task graph, one task per session, milestone
   gates **with literal gate commands**, and a risks table with pressure valves. `old-Docs/PLAN1.md`
   § Risks and § Milestone gates are the model; copy the shape, not the content.
3. **A task backlog** — the analogue of `old-Docs/TASKS1.md`: per task, goal, depends-on, inputs and
   outputs with types and array shapes, acceptance criteria as a checklist, the constraints that
   bite, and notes.
4. **A decision on state-file continuity** — does Phase 1 continue `old-Docs/STATE1.md` or start
   `DOCS/STATE2.md`? Either is defensible; the session log is append-only and must not be rewritten
   or condensed either way. Record the choice as a decision.
5. **A decision on which of `CLAUDE.md`'s 12 constraints survive into Phase 1**, and what replaces
   the ones that do not. Constraint 6 ("do not optimise before Rung 3 passes") is spent. Constraint
   1 (D2Q9/BGK/bounce-back only) is a *Phase 0* constraint and Phase 3 explicitly plans to replace
   the kernel with XLB. Constraints 2, 3, 8, 9, 10, 11 and 12 look permanent. Say so explicitly
   rather than letting them rot.

### Questions this session has to answer, not dodge

These are the ones where `idea.md` and the Phase 0 evidence pull in different directions. Answer them
with the user, and record each as a numbered decision with its reasoning:

- **Does Phase 1 keep our kernel or adopt XLB now?** `idea.md` § Roadmap puts XLB at **Phase 3** and
  the GPU port at Phase 2, with our kernel as the stated fallback if XLB stagnates. Phase 0's kernel
  is validated and meets its budget; that is an asset with a known value.
- **What is the actual Phase 1 deliverable — a library, a CLI, or a UI?** `python -m lbm.runner`
  already does "PNG plus physical numbers to an MP4 in one command". `idea.md` § How we know it
  worked describes someone dragging in a picture and getting an answer in under a minute, which the
  current CLI does not reach. The gap between those two is the honest scope of Phase 1.
- **What is Phase 1's validation ladder?** `idea.md` § Risks: *"A pretty wake that is physically
  wrong is worse than no tool at all. Every phase ships with a benchmark that has a known answer."*
  Phase 0's ladder was four rungs against published data and it caught three wrong-but-plausible
  answers (the force integral measuring the channel walls, the blockage lie from no-slip walls, the
  FFT locking onto the domain's acoustics — all in the session 7 log). Phase 1 needs its own, and
  what a "known answer" means for a *usability* layer is a real question.
- **What breaks when the geometry stops being a nice convex blob?** `from_png` already warns on
  1-cell hairlines (**D-031**, **D-040**), and `min_thickness`'s documented limit is that a thin
  appendage fused to a thick body shares its component and is not reported (**D-017**). Real user
  shapes are exactly that case.
- **Where does the product layer live?** A new package beside `lbm/`, or on top of it? Phase 0's
  module map is clean and constraint 10's "one `render()`" only held because nothing was allowed to
  colour anything itself.

### Constraints that bite on *this* session

- **`idea.md` § Risks — "The trap."** Named in the source document: *"It is very tempting to keep
  polishing the solver because that part is fun. The solver is not the product."* If this session
  ends with a change under `lbm/` and no plan, it failed.
- **`idea.md` § Risks — "Scope."** *"'Fluid dynamics of anything' is unbounded. Phase 1 must be
  narrow: 2D, external flow, incompressible, single fluid."* A Phase 1 plan that is not narrow is
  not a plan.
- **`CLAUDE.md` § Session protocol — one task per session.** Planning is the task. Do not start the
  first Phase 1 task in the same session that plans it.
- **Constraint 5's spirit — validation is ordered and non-negotiable.** Whatever ladder Phase 1
  gets, it is built before the thing it validates, not after.

### Decisions from Phase 0 that the product layer inherits

Quoted because they are load-bearing above the solver line, not merely historical:

- **D-032 / D-038** — `lbm/units.py` enforces `U < 0.1` and `tau > 0.51` and **raises**, naming the
  resolution that would fix it. This is the "stability guardrails" moat item of `idea.md` in its
  current form, and it is why the CLI **refuses** the acceptance command written into T011's own
  contract: air at 20 m/s past a 1.5 m body is Re 2e6, `tau` reads 0.5000, and BGK with bounce-back
  and no turbulence model cannot represent it at any resolution this project will run. A product
  aimed at people who have never heard of a Reynolds number **will** be handed that case on day one.
  *How the product answers it is a genuine Phase 1 design question and probably the most important
  one in this list.*
- **D-024 / D-039** — two run modes and no more: `drop=True` drains from a consumer thread and drops
  the oldest *display* frames; `drop=False` drains inline and the sim waits. Any sink that writes a
  file takes the second.
- **D-028** — `render` takes fixed symmetric limits and **raises** on an asymmetric pair; the LUT has
  257 entries so zero lands on one. Two frames with different data and identical limits map a fixed
  value to byte-identical output.
- **D-023** — `steps_per_frame` is computed from `dt`, never hardcoded.
- **D-030** — solid cells must be seeded at rest; `Sim` seeds the whole domain with the inlet
  equilibrium and bounce-back reverses that junk rather than clearing it.
- **D-019 / D-040** — the characteristic length `D` is the *measured* cross-stream extent of the
  object's bounding box, and every derived number (blockage, `tau`, `Cd`) must come from the same
  one.
- **D-029 / D-036** — the project carries three `tau` floors on purpose (0.51 in `units`, 0.537 in
  Rung 3, 0.54 in Rung 4) because a floor that refuses valid configs is as wrong as one that admits
  `nan`.
- **D-035** — an absolute steps/s number means nothing without the CPU clock beside it; A/B by
  alternating rounds, never sequentially.

### Before you start

- **Nothing to install.** `myenv` has numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  pygame 2.6.1, imageio 2.37.4, imageio-ffmpeg 0.6.0, psutil 7.2.2, Python 3.11.15. If Phase 1 picks
  XLB, Warp or Taichi, that is a decision to *record* this session and install in the session that
  first needs it.
- **Confirm the ladder is still green if any solver code has changed since session 11** — it should
  not have. `myenv/Scripts/python.exe -m pytest` (367 passed) is the cheap check; the full ladder is
  about 55 minutes and R4 alone is ~40 of them.
- `DOCS/ISSUES.jsonl` is still untracked in git although `CLAUDE.md` says the queue is committed.
  Noted twice now (sessions 10 and 11); committing it is the user's call.

## Scope discipline

The output of this session is **documents**. No new modules under `lbm/`, no new rungs, no
refactoring of Phase 0 code because the plan suggests a nicer shape — the plan can say what to change
and the task that does it gets its own session. If something genuinely must be fixed in Phase 0 code,
`/new-task` it against `old-Docs/TASKS1.md` rather than folding it into the plan.

## Verify, then close

1. Read the plan back as if you were the next cold session: does each task have a gate you could
   actually run and fail?
2. Check every acceptance criterion you wrote is *measurable*. Phase 0's ladder worked because "PASS"
   meant a number inside a published band, not "the wake looks right".
3. **Run `/checkpoint`** — it updates the state file, syncs the task file, and writes the next
   session's prompt into `PROMPTS/`. Do not end the session without it.
