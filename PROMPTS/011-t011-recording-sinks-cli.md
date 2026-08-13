# Session 11 — T011: recording sinks + CLI → **M4**

## What this project is

Phase 0 of a fluid-dynamics project: a validated, continuously-running 2D fluid simulator in pure
NumPy — D2Q9 lattice Boltzmann, BGK collision, geometry from a boolean mask, live streaming visual
plus recordable video. Full Phase 0 spec is `DOCS/IDEA2.md`.

Phase 0 is **not** the product. It exists so we understand LBM well enough to design the layer above
it (root `idea.md` / `README.md` describe that product). **This task ends Phase 0.**

## Read these first, in this order

1. `CLAUDE.md` — 12 hard constraints, session protocol, coding conventions, module map. **Constraint
   10 ("one `render()`, three sinks — do not write three renderers") is the one this task exists to
   satisfy**, and constraint 8 is the one most likely to be got backwards.
2. `DOCS/STATE1.md` — **in full**. Snapshot, blockers, open questions, environment, decisions
   (D-005 … D-037), session log. The decisions that bear on this task are **D-024** (`drop=True`
   threads the consumer, `drop=False` drains inline — the recorder is the `drop=False` half and it
   is *allowed* to make the sim wait), **D-028** (`render` refuses asymmetric limits; the LUT has
   257 entries), **D-023** (`steps_per_frame` is computed from `dt`, never hardcoded), and
   **D-031/D-032** (the PNG/SVG loader and `LatticeUnits`, which the CLI wires together).
3. `DOCS/TASKS1.md` § **T011** — the task contract, in full, plus the backlog index row. Also read
   § T010's outcome note: the hot path changed (a fused kernel is now the default), but the sink and
   frame path did **not**.
4. `DOCS/IDEA2.md` § **Three output sinks, same frame source** and § **Milestones**.
5. `DOCS/PLAN1.md` § Session map and § Risks. T011 is session 11 of 11 and carries **M4**.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 10: **T010 is `done`**, every acceptance criterion run and green.
  `bench.py` is new at the repo root; `lbm/core.py` gained `collide_stream` (collide + bounce-back +
  the `f_bb` snapshot + stream fused into one pass per direction) and `SimConfig.fused` selects it,
  defaulting on. `lbm/boundary.py::inlet_velocity` takes a precomputed `fluid` mask.
  `validate/cylinder.py` gained `TAU_FLOOR = 0.537`. **The frame path, the ring buffer and
  `render()` were not touched.**
- **Performance:** 696.7 / 161.7 / 16.8 steps/s at 40k / 160k / 1M cells against floors of
  400 / 120 / 15 — cleared at the CPU's rated clock. Note the caveat in § Performance baseline: on
  battery the same build reads 402.7 / 117.0 / 16.8. Quote the clock with any number you measure.
- **The ladder was re-run in full in session 10 and all four rungs are green, bit-identical to
  session 9:** R1 L2 **0.3650%** · R2 **0.75% / 0.42% / 1.01%** · R3 **St 0.1731,
  Cd 1.4031 ± 0.0086** · R4 square **Cd 1.5279 ± 0.0271**, polygon **Cd 1.4276 ± 0.0226**.
  `myenv/Scripts/python.exe -m pytest` → **`329 passed`**.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — the ladder is complete.
- **Milestone reached:** **M3** (2026-08-12). **This task is M4.**
- **Completed tasks:** T001 … T010.

## Your task this session

**T011 — Recording sinks + CLI.** One task, this session only.

Run this first:

    /start-task T011

It re-reads the contract, restates goal and acceptance criteria, and waits for your confirmation
before implementing.

Goal: the third and fourth sinks, and one command that takes a PNG plus physical numbers and
produces an MP4. **M4 — the first thing another person can use.**

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `RecordSink` writes MP4 via imageio/ffmpeg at a **fixed** framerate and **never drops a frame** — a test writes 50 frames and asserts the file has exactly 50.
- [ ] `HeadlessSink` writes numbered PNGs, no display required.
- [ ] Both consume the same `render()` output as `LiveSink`; a test asserts the three sinks receive byte-identical frames for the same sim state.
- [ ] GIF output works for short clips.
- [ ] `python -m lbm.runner --geometry tests/data/<file>.png --fluid air --velocity 20 --length 1.5 --seconds 5 --out wake.mp4` produces a playable MP4 with a visible vortex street, in one command, from a cold shell.
- [ ] `--live`, `--record`, `--headless` are composable; `--live --record` together works.
- [ ] Missing ffmpeg produces a clear install message, not a traceback.
- [ ] Rungs 1–4 still green. `DOCS/STATE1.md` records M4 as reached with the gate command output.

### Constraints that bite on this task

- **Constraint 10 — one `render()`, three sinks.** `lbm/render.py::render` already exists and
  returns `uint8` RGB `(ny, nx, 3)`; the recording sinks consume its output and colour nothing
  themselves. The byte-identical-frames test is what enforces this, so write it early.
- **Constraint 8 — the buffer drops display frames, never simulation steps** — but **record must not
  drop**. Different policies, same frame source: `run(..., drop=False)` drains inline and is the
  recorder's mode (**D-024**); `drop=True` is live. Getting these backwards produces a video with
  missing frames that still looks plausible.
- **Constraint 9 — vorticity, diverging colormap, fixed symmetric limits.** Recording does not get
  to autoscale per frame; `render` raises on an asymmetric `(vmin, vmax)` (**D-028**).
- **Constraint 7 — `steps_per_frame` is computed**, from `dt` via `lbm.runner.steps_per_frame`
  (**D-023**). `--seconds` is physical time and must go through `lbm/units.py`, not through a
  guessed step count.
- **Constraint 3 / 2 — the CLI is where a user can ask for something unstable.** `lbm/units.py`
  already raises on lattice `U >= 0.1` or `tau <= 0.51` and names the resolution that fixes it
  (**D-032**); surface that message, do not catch and soften it.
- **Constraint 12 — `from_png` runs `check_mask` itself.** A downscaled PNG is the most likely
  source of a 1-cell-thin wall in the project; do not suppress the warning to make the demo quiet.
- **Constraint 5 — the ladder stays complete.** All four rungs get re-run at the end. Budget from
  session 10's measured wall clock: R1 ~15 s, R2 ~195 s, R3 ~360 s, R4 ~42 min — about **50 minutes**
  for the full ladder, so **start it early and in the background**, and note that R4's wall clock
  varies by nearly 25% with the laptop's power state.

### Blockers, open questions and decisions that affect you

- **Blockers:** none.
- **Open questions:** none open. Q-001, Q-002, Q-003 and Q-004 are all closed (D-009, D-031, D-013,
  D-036).
- **D-024** — `run(sim, sink, drop=True)` drains the ring buffer from a consumer thread and drops the
  oldest display frames when it fills; `drop=False` drains inline and the sink sees every frame in
  order. The second is explicitly there for "a fixed-framerate recorder (T011) must see every frame
  in order, and it is allowed to make the sim wait". Do not add a third mode.
- **D-028** — `render(field, limits, ...)` takes symmetric limits and **raises** on a lopsided pair;
  the colormap has 257 entries so that zero lands on one and `±v` are mirror colours. Two frames with
  different data and identical limits map a fixed value to byte-identical output — that test exists
  and the recording sinks inherit the guarantee.
- **D-030** — `Sim` seeds the whole domain, solid included, with the equilibrium of the inlet
  profile, so at step 0 there is fluid moving at `U` *inside* the body and bounce-back reverses
  rather than clears it. `validate/polygons.py::seed_solid_at_rest` is the fix and is where a
  PNG-body demo should start.
- **D-031** — `lbm.geometry.from_png(path, shape)` thresholds alpha (luminance fallback), resamples
  then thresholds, flips `y` so a picture loads upright, and runs `check_mask` automatically.
  `from_svg` covers `M/L/H/V/C/Q/Z` and `<polygon>` and raises an `ImportError` naming `cairosvg`
  for anything else.
- **D-033 / D-035** — the step is now fused by default and is bitwise identical to the old path;
  `SimConfig(fused=False)` selects the T009 sequence if you ever need to bisect. If you benchmark
  anything, alternate the variants and quote `Win32_Processor.CurrentClockSpeed`.

### Before you start

- **Install `imageio[ffmpeg]` into `myenv` first** and add a row to `DOCS/STATE1.md` § Environment in
  the same session — a missing row is the one thing § Environment says is non-negotiable:

      myenv/Scripts/pip.exe install "imageio[ffmpeg]"

  Everything else is present: numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1,
  pygame 2.6.1, Python 3.11.15.
- `tests/data/test_body.png` already exists (959 bytes, committed in T009) and is what the CLI
  acceptance criterion's `--geometry` argument can point at.
- Run tests and validation scripts **from the repo root** so `import lbm` resolves.

## Scope discipline

Work only what's in the contract. **Phase 0 ends here** — M5 (a Warp/Taichi port of the kernel) is a
new plan, not a stretch goal of this task, and `DOCS/IDEA2.md` § Deliberately deferred (STL, 3D, GPU,
UI) stays deferred. Something else genuinely needs doing? `/new-task` it.

The temptation this session is to build the product layer because M4 makes it feel close. Don't. When
this lands, the next session plans the product from root `idea.md` with a working, validated,
understood solver underneath it — which was the entire point of Phase 0.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it — including producing an actual MP4 and
   opening it, and the missing-ffmpeg message on a path where ffmpeg is not available.
2. Run `/validate`. R1, R2, R3 and R4 must all be reported with their measured numbers and compared
   against session 10's above. A rung that moves is a revert, not a debugging session.
3. Re-run `myenv/Scripts/python.exe -m pytest` and report the actual output — the 329 existing tests
   must still pass.
4. **Record M4 in `DOCS/STATE1.md` § Snapshot with the gate command and its output**, per
   `DOCS/PLAN1.md` § Milestone gates: "An arbitrary PNG becomes a mask, runs in physical units, and
   records an MP4 — end to end, one command." A milestone is claimed only when its gate is run.
5. If anything is red at session end, say so plainly, record the measured numbers and the suspected
   cause in `DOCS/STATE1.md` § Blockers, and leave T011 `in_progress`.
6. **Run `/checkpoint`** — it updates `DOCS/STATE1.md`, syncs `DOCS/TASKS1.md`, and writes the next
   prompt. Since Phase 0 closes here, that next prompt is a **planning** session for the product
   layer from root `idea.md`, not another solver task.
