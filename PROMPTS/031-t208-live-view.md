# Session 31 — T208: Live view, numbers panel, save, refusal UI

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer
with a fidelity band attached, shipped as a wheel named `fengdong`. **Phase 2 is live** and its spec is
`DOCS/IDEA4.md`. The solver is not the product: see `idea.md` § Risks — *"The trap"*, which names the
standing temptation to keep polishing it because that part is fun. **This task is the moving
picture** — the window exists and shows the plan; now it has to run it — and `DOCS/IDEA4.md` § The
five things Phase 2 must get right (4) names its rule: *"Watching stays smooth, and never at the sim's
expense."*

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
The closure (T201, T202), the bands (T204), the box (T205), the widgets (T206) and **the window
(T207)** are done. What remains: **T208** the live view, T209 the drop and Rung J.

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (**7**, **8**, **9**, **10**, **11**, **16**, **17**, **18**
   govern this session), the session protocol, the conventions, the module map (the four `fengdong/`
   rows), and § Current state (the T206 and T207 paragraphs say what the widgets and the window look
   like).
2. `DOCS/STATE3.md` — **in full.** § Snapshot, § Blockers, § Open questions (**Q-204** is the user's,
   **Q-205** is *yours* — the trend wire on `Monitor`), § Environment (nothing gets installed this
   session), § Decisions **D-080 … D-098** (**D-094** moved `Monitor`'s wires on a closure-on run and
   is what Q-205 is about; **D-097** is the widgets' event model; **D-098** is why the window rebuilds
   a `Case` on commit and not per keystroke — the same argument applies to anything else that costs
   more than a frame), and the session-30 log entry (**the manual gate record and the `Monitor`-on-warp
   number, 9.82%, are both there**).
3. `DOCS/TASKS3.md` § **T208** — the task contract, in full. Also § T207 (what you are consuming —
   `App.handle` returns whether the case changed; `open` / `run` / `close` are separate on purpose),
   § T209 (what will consume the running window: a synthesised `DROPFILE` through the real `App` to
   Rung 3's bands, timed), and the backlog index.
4. `DOCS/IDEA4.md` § **The five things Phase 2 must get right (4)**, § **Performance budget** (the
   app: 30 fps at `quality="balanced"` on warp with zero dropped simulation steps), § What Phase 2 is,
   concretely.
5. `DOCS/PLAN3.md` § Why this order, § Session map (session 31 is T208), and § **Risks** — the row
   *"The window blocks the sim"* is this task's; *"The app becomes a second brain"* still applies.
6. **The code you build on:** `fengdong/app.py` in full (`App.__init__`, `layout`, `handle`,
   `rebuild`, `_apply_nearest`, `_show_case`, `plan_text`, `draw`, `open` / `run` / `close`, and the
   `_TextPane` helper), `tests/test_app.py` (the headless driving pattern — `fill`, `drop`, `click`,
   `typed`; the subprocess test with `-W error::ResourceWarning`; the `open`-then-post-`QUIT` pattern,
   because initialising the display starts a fresh event queue), `fengdong/widgets.py` (`Panel`,
   `Button`, `Label`), `lbm/runner.py` (`run`, `RingBuffer`, `Sim`, checkpoint / restart — **D-022**,
   **D-024**, **D-050**), `lbm/render.py` (`render`, `Sink`, `LiveSink` — the shape of a sink and the
   one-thread rule for SDL; **the app's view is a fourth sink on the existing ring buffer, never a new
   path to the screen**), `lbm/record.py` (**D-039**: file sinks take `drop=False`), `flow/case.py::run`
   (how sinks are resolved — `_resolve_sinks` — and how `Monitor` is attached), `flow/report.py`
   (`Result`, `GATED_QUANTITIES`, `Qualified`, `save`, `summary`), `flow/fidelity.py`,
   `flow/diagnose.py::Monitor` and `Diverging` (**D-061**, **D-094**), `validate/refusals.py` (how
   `Monitor`'s cost is measured, **D-078**, **D-092**).

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 30 and it worked **T207 — the window. It is `done`: every acceptance
  criterion was run and passed, and the manual gate was run and recorded.** What landed:
  `fengdong/app.py` — `App` (the resizable window titled FengDong; one `Panel` of T206 widgets: drop
  target, fluid, speed, size, quality, *Preview the plan*, *Use the nearest case that runs*, a status
  line; the body preview as two flat colours; a scrollable plan pane whose text is
  `Case.explain(quiet=True)` verbatim plus, for a refusal, the list `Case.nearest()` acts on) and
  `fengdong/__main__.py::main` now opens it (`--version` still first, without pygame / numpy / `flow`;
  `--backend` passed through). `tests/test_app.py` (32 tests). `tests/test_widgets.py`'s
  constraint-13 scans now walk every file in `fengdong/`. **Nothing in `lbm/` or `flow/` changed.**
- **Rung status:** R1–R4 🟩 · A–E 🟩 · F 🟩 · G 🟩 · H 🟩 · **I 🟩** · J ⬜. Session 30 re-banked
  Rung I on mains first thing — **PASS, 47.4 s** (venv 13.6, pip 28.5, `--version` 0.14, smoke 5.2)
  at 3201 of 3201 MHz — and re-ran it after `app.py` joined the wheel three more times — 62.2 s FAIL straight after a four-minute `pytest`, 63.4 s FAIL with the smoke child at 18.7 s against its usual 5 s, then **42.1 s PASS** idled, every content clause `[ok]` in all four; **D-092**: all four readings are in the session-30 log; re-ran `pytest`
  (**1012 passed, 2 skipped** in 259.7 s), Rung 1 (L2 **0.3650%**, unmoved) and Rung C (**15/15**, 27.5 s) as spot checks. The physics rungs were **not**
  re-run: no file under `lbm/`, `flow/` or `validate/` changed, and the session log says so.
- **Milestones reached: M9, M10, M11.** **M12 is T209's.**
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110; Phase 2 **T201 … T207**.
- **Numbers to keep still.** Rung I **47.4 s** (session 30, mains). Rung H: quantitative `Cd`
  **1.4030** on both backends; qualitative `max(nu_t)/nu` **0.6906** / **0.6886** with **no `Cd`**;
  illustrative **3.374e4** / **3.797e4**; Q-203's **0.1057** with `Cd` **1.4143**, `St` **0.1719**.
  Rung E `Cd` **1.4040**, `St` **0.1672**, 55.0 s. Rung 3 St **0.1731** Cd **1.4031**; Rung 4
  **1.5279** / **1.4276**; Rung 1 **0.3650%** / **0.3649%**; Rung 2 **0.75%** / **0.21 cells**; Rung A
  **5.960e-08** / **9.611e-06**; Rung F **0.000e+00**; Rung G **0.2303%** / **1.1547%** / **3.0178%** /
  **0.9972**. Rung D's Monitor cost on numpy: −0.39% (limit 2%); **on warp, measured once in session
  26 outside any gate: bare 1274.3 steps/s, watched 1149.2, cost 9.82%** — that is the number this
  task exists to own.

## Your task this session

**T208 — Live view, numbers panel, save, refusal UI.** One task, this session only.

Run this first:

    /start-task T208

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** Press the button and watch it. The sim runs, the vorticity streams into the window, the
numbers update, and the result can be saved — without the display ever costing a simulation step.

**In:** a runnable `Case`. **Out:** the live view, the numbers panel, save actions, the in-window
refusal path.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] The sim runs on a **worker thread**; the window consumes frames from the **existing ring buffer** as a fourth sink. `steps_per_frame` still comes from `flow.autoconfig.plan` (constraint 7).
- [ ] **Zero simulation steps are dropped, ever.** Display frames may drop; the app **counts them and shows the count** (constraint 8, **D-039**'s posture). A test drives a deliberately slow consumer and asserts the step count is unaffected.
- [ ] The view draws **vorticity, diverging colormap, fixed symmetric limits**, through `lbm.render` (constraints 9 and 10). `fengdong/` colours nothing — asserted by test.
- [ ] The numbers panel shows `Cd`, `Cl`, `St`, peak `|u|` against the 0.1 ceiling, convergence, elapsed — **and the fidelity band**, with `Cd` withheld or qualified exactly as `Result` does it (**constraint 18**). The panel cannot show a number `Result` declines to give.
- [ ] Pause, resume and restart work; restart from a checkpoint reproduces bit-identically within the backend (**constraint 11**), asserted by test rather than by eye.
- [ ] Save writes the MP4, the plot and the summary through `flow.report.Result.save` — no second writer. The video metadata carries `substituted` and the band (**constraint 16**).
- [ ] The **30 fps at `quality="balanced"` on warp** budget is measured and printed, with **D-035** conditions beside it.
- [ ] A run that diverges is caught by the existing `Monitor` and shown in the window with its cause and fix (**D-061**), not as a frozen picture. ***And `Monitor` is finally timed on warp*** — the oldest open thread in the product layer, carried since session 18.
- [ ] `pytest` green.

### Constraints that bite on this task

- **Constraints 7 and 8** — simulation and rendering decoupled, `steps_per_frame` computed, never
  block the sim on the display, drop *display* frames only. The app is a **fourth sink on the
  existing ring buffer** (`lbm.runner.RingBuffer`, driven by `lbm.runner.run` with `drop=True` from a
  consumer thread, **D-024**). Do not write a second loop, a second buffer, or a second thread model.
  `LiveSink` shows the one-thread rule: every SDL call on the consumer side; the physics thread never
  touches pygame. The app's window is *already* the SDL thread — the sink pushes frames into a slot the
  app's loop blits, and it counts what it dropped.
- **Constraints 9 and 10** — vorticity, diverging map, fixed symmetric limits from
  `Plan.vorticity_limit`, through `lbm.render.render`. `fengdong/` still defines no `render`, `to_rgb`
  or `colormap` (`tests/test_widgets.py::test_fengdong_defines_no_renderer_of_its_own` already walks
  every file in `fengdong/`). **T207's body preview is a stencil of a bool mask and stays one**; the
  moment a field reaches the screen it is `render`'s pixels.
- **Constraint 11** — restart is bit-identical within a backend: `f`, `mask`, `step_count` are the
  whole state (**D-022**, **D-050**); a Restart button is a claim about that and a test asserts it.
- **Constraint 16** — `Result.save` is the one writer; the MP4 metadata carries `substituted` and the
  band. No second MP4 path in the app.
- **Constraint 17** — still a view. The numbers come from `Result` / `Case`; the run length,
  `steps_per_frame` and the colour limit come from `Case.plan`. `tests/test_app.py::test_app_computes_no_solver_parameter`
  asserts `app.py` reaches for no `.plan` attribute and names no lattice quantity — **if T208 needs
  `plan.vorticity_limit` or `plan.steps_per_frame`, it reads them where `flow.case.Case.run` already
  does, or extends that test's exemption deliberately and says why in the log**; do not silently
  delete the assertion. The constraint-13 identifier and string scans walk every file in `fengdong/`
  (docstrings exempt).
- **Constraint 18** — the numbers panel is a second surface for the same claim and must not be a
  laxer one: read `Result.cd` / `Result.cd_qualified` / the band and show exactly what the object
  holds. `Result.__post_init__` already withholds; the panel has nothing to leak if it reads the
  object.
- **Constraint 13** — no new input widget carries a lattice quantity. Pause / Resume / Restart / Save
  are `Button`s; the set stays closed (**D-083**). A sixth widget is `/new-task`.
- **Constraint 20** — `fengdong --version` still answers without pygame; Rung I must still pass with
  whatever T208 adds to the wheel.
- **Constraint 5** — the ladder is a gate. This task touches no solver code; `pytest` and Rung I are
  the minimum re-run, and say which physics rungs were not re-run and why. If `Monitor` or anything
  under `flow/` is edited to time it on warp, **that is a `flow/` change and Rung D re-runs.**

### Blockers, open questions and decisions that affect you

**Blockers: none.**

- **Q-205 (open — yours)** — **D-094** moved `Monitor`'s speed and mass wires to the meaning bound on
  a closure-on run, giving up early warning for a run that runs away *between* the bounds. Does a
  live run need a **trend** wire — divergence as "over the accuracy bound **and rising**"? D-038's
  case is flat at 0.20 from step 4000 on, so a trend wire would be silent there. Answer it by
  measurement in the window, or record explicitly that it stays open and why.
- **`022ac461c920`** (queued) — dead `smag_work` `(4, ny, nx)` allocation on warp, 32 MiB at 2M
  cells. T208 touches memory; take it or leave it explicitly.
- **`d5b27e51fcdc`** (queued) — two power probes disagreed once; not reproduced in sessions 27–30.
- **D-039** — file-writing sinks take `drop=False`; the live view takes `drop=True`. Save after a
  run therefore goes through `Result.save`, which writes from the frames the run kept, not from the
  window's dropped stream.
- **D-071** — `keep_frames=False` is the CLI's choice because it never saves afterwards; the window
  *does* save afterwards, so decide what it keeps and record the memory cost.
- **D-073** — `--live` is three-valued in the CLI; the window contradicts none of it.
- **D-097** — the widgets' event model. `Button.take_click()` is the poll-style read; an open
  `Dropdown` captures. The T207 loop is `wait(200)` then `get()`, drawing after events — **a running
  sim needs a frame cadence, so `run()` changes shape here**; keep `handle` / `draw` / `open` / `close`
  separable so `tests/test_app.py` keeps driving them headless.
- **D-098** — anything that costs more than a frame is not done per keystroke.
- **Q-204 (open — the user's)** — PyPI publication. Not this task's.

### Before you start

- **Nothing to install.**
- Check the mains (`Win32_Battery.BatteryStatus` must read 2) before the 30 fps measurement and
  the `Monitor`-on-warp cost — both are D-035 numbers.
- Run the long things detached under `outputs/ladder/`; a foreground command is hard-capped at ten
  minutes. `pytest` alone is ~6 min.

## Scope discipline

Work only what's in the contract. The drop rung is **T209**. A sixth widget, a layout engine, theming
or animation beyond the frame cadence the sim itself sets is the trap named in `DOCS/PLAN3.md`
§ Risks, and the valve is `/new-task`. If it is listed under `DOCS/IDEA4.md` § Deliberately deferred
(XLB, 3D, STL, KBC, MRT, curved boundaries, wall models, dynamic `Cs`, a web UI), the answer is no.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it — including watching a run in the
   real window and recording what was seen in `DOCS/STATE3.md`, as session 30 did.
2. Run `/validate` for every rung at or below this task — `pytest` and Rung I at minimum, Rung D if
   `flow/diagnose.py` moved; say which physics rungs were not re-run and why.
3. Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
   `PROMPTS/032-…` for the next session. Do not end the session without it.
