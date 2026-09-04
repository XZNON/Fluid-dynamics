# Session 30 — T207: `fengdong/app.py` — the window, the drop target, the setup panel, the plan preview

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer
with a fidelity band attached, shipped as a wheel named `fengdong`. **Phase 2 is live** and its spec is
`DOCS/IDEA4.md`. The solver is not the product: see `idea.md` § Risks — *"The trap"*, which names the
standing temptation to keep polishing it because that part is fun. **This task is the window itself**
— the first thing a person sees — and `DOCS/IDEA4.md` § The five things Phase 2 must get right (3)
names its rule: *"The app is a view, not a second brain."*

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
The closure (T201, T202), the bands (T204), the box (T205) and **the widgets (T206)** are done. What
remains: **T207** the window, T208 the live view, T209 the drop and Rung J.

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (**17**, **13**, **14**, **10** and **5** govern this
   session), the session protocol, the conventions, the module map (the three `fengdong/` rows), and
   § Current state (the T205 and T206 paragraphs say what the package and the widgets look like).
2. `DOCS/STATE3.md` — **in full.** § Snapshot, § Blockers, § Open questions (**Q-204** is the user's,
   **Q-205** is T208's), § Environment (nothing gets installed this session), § Decisions
   **D-080 … D-097** (**D-083** closed the widget set, **D-096** is the package, **D-097** is the
   widgets' event model — how `handle` reports change, how text arrives, what an open `Dropdown`
   captures), and the session-29 log entry.
3. `DOCS/TASKS3.md` § **T207** — the task contract, in full. Also § T206 (what you are consuming),
   § T208 (what will consume the window, so the state machine has the right seams) and the backlog
   index.
4. `DOCS/IDEA4.md` § **What Phase 2 is, concretely** (the diagram: `fengdong/app.py` sits between the
   widgets and `flow.Case`, and *"the app computes nothing"*), § **The five things Phase 2 must get
   right (3)** and (4), and § Scope (Windows-only, in writing).
5. `DOCS/PLAN3.md` § Why this order (point **5**, *"Widgets before the window"* — you are the window),
   § Session map, and § **Risks** — the rows *"The app becomes a second brain"* and *"Hand-rolled
   widgets swallow the phase"* are both this task's.
6. **The code you build on:** `fengdong/widgets.py` in full (every public class and its `handle`
   contract), `tests/test_widgets.py` (how the widgets are driven headless — the `App` tests use the
   same shape), `fengdong/__main__.py` (which must keep answering `--version` without pygame loaded —
   open the window only *inside* `main` after argument parsing), `flow/case.py::Case.from_image` and
   `::explain` (**D-067**, **D-068**), `flow/cli.py` (**D-073** — the flag semantics the window must
   not contradict; `_print_refusal` is the refusal surface you mirror), `flow/prepare.py` (the
   repaired mask and its verdict, **D-065** / **D-066**), and `lbm/render.py::LiveSink` (how the
   existing live window is opened lazily on one thread — the shape the app's window should share).

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 29 and it worked **T206 — the widgets. It is `done`: every acceptance
  criterion was run and passed.** What landed: `fengdong/widgets.py` — `Widget` (base), `Label`,
  `TextField` (with `.speed(rect)` / `.size(rect)` constructors that pass the exact `expect` /
  `default_unit` pair `flow/case.py` passes), `Dropdown` (with `.fluids(rect)` read from
  `flow.fluids.FLUIDS` at construction), `Button`, `DropTarget`, `Panel` (one column, Tab /
  Shift+Tab) — and `tests/test_widgets.py` (50 tests). **Nothing in `lbm/` or `flow/` changed.**
- **Rung status:** R1–R4 🟩 · A–E 🟩 · F 🟩 · G 🟩 · H 🟩 · **I 🟨** · J ⬜. Session 29 re-ran
  `pytest`, Rung I, and Rung 1 and Rung C as spot checks; the physics rungs were **not** re-run
  because no file under `lbm/`, `flow/` or `validate/` changed and nothing imports the new module —
  the session log says so explicitly rather than claiming a re-run. **Rung I is 🟨, not 🟥**: every
  content clause passed (the wheel ships `widgets.py`, constraints 15/17 hold installed, the smoke
  reaches `quantitative`) but the 60 s clock read **97.7 s on battery at 1882 of 3201 MHz** — venv
  and pip each ~1.8x their mains figures. **D-092: your first act this session, before anything else
  is timed, is `myenv/Scripts/python.exe -m validate.install` on mains, idled**, and both readings go
  in the log. Session 28's mains readings (52.6 s, 55.5 s) are the published ones until then.
- **Milestones reached: M9, M10, M11.** **M12 is T209's.**
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110; Phase 2 **T201 … T206**.
- `pytest`: **971 passed, 2 skipped** (334.3 s).
- **Numbers to keep still.** Rung I **52.6 s / 55.5 s** on mains (session 28; owed a session-30
  reading). Rung H: quantitative `Cd` **1.4030** on both backends; qualitative
  `max(nu_t)/nu` **0.6906** / **0.6886** with **no `Cd`**; illustrative **3.374e4** / **3.797e4**;
  Q-203's **0.1057** with `Cd` **1.4143**, `St` **0.1719**. Rung E `Cd` **1.4040**, `St` **0.1672**.
  Rung 3 St **0.1731** Cd **1.4031**; Rung 4 **1.5279** / **1.4276**; Rung 1 **0.3650%** /
  **0.3649%**; Rung 2 **0.75%** / **0.21 cells**; Rung A **5.960e-08** / **9.611e-06**; Rung F
  **0.000e+00**; Rung G **0.2303%** / **1.1547%** / **3.0178%** / **0.9972**.

## Your task this session

**T207 — `fengdong/app.py`: window, drop target, setup panel, plan preview.** One task, this session
only.

Run this first:

    /start-task T207

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** A window opens, a picture can be dropped on it, three physical numbers can be typed, and
the plan `flow` would run is shown before anything runs. **No simulation yet.**

**In:** a dropped file path, a fluid name, a speed string, a size string, a quality level.
**Out:** `fengdong/app.py::App`, `fengdong/__main__.py::main` (now opens the window; `--version`
still answers first and without pygame); the setup panel; the plan preview.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `fengdong` opens a window titled **FengDong** with a visible drop target, and dropping a PNG on it loads and previews the mask that `flow.prepare` produced — the repaired one, with its verdict shown (**D-065**, **D-066**).
- [ ] Fluid, speed, size and quality are entered through the T206 widgets, and a bad entry shows the parse error without crashing the window.
- [ ] **The plan preview is `Case.explain()`'s content**, obtained from `flow.Case`, not recomputed: grid, `tau`, timestep, run length, expected fidelity band, and why each. Constraint 17 asserted — the app computes nothing.
- [ ] A refused case shows the refusal and its suggestions, and the suggestion the app would **act on** is the one it shows. *(This is queued issue `2fd69b874c32` — `Case.explain()` prints a different list than `Case.nearest()` acts on. Fix it here or carry it explicitly; do not reproduce the mismatch in a second surface.)*
- [ ] The window is resizable and the panel survives it; nothing is positioned by hard-coded pixel counts that break at another size.
- [ ] Closing the window exits cleanly with no pygame resource warnings, asserted headless.
- [ ] Headless test coverage of `App`'s state machine — file dropped, fields edited, plan computed — with no window opened. The manual gate (a human opens it and drops a file) is **recorded in `DOCS/STATE3.md` with what was seen**, per this project's habit of running things rather than reading them.
- [ ] `pytest` green.

**On the fourth criterion:** queued issue `2fd69b874c32` was **closed in session 27 (D-093)** —
`flow.diagnose._present` deduplicates by `(change, value)` and
`tests/test_cli.py::test_the_suggestions_the_cli_prints_are_the_ones_nearest_would_run` iterates every
reachable refusal and asserts `explain()` and `nearest()` agree. So the app inherits a guard rather
than a defect: show the list `Case` gives you, act on its first entry, and write the test in the same
shape over the window. Do not re-open the issue.

### Constraints that bite on this task

- **Constraint 17** (**D-083**) — *the app is a view.* Every grid, `tau`, timestep, run length and
  band the preview shows comes out of `flow.Case.explain()` / `Case.plan`; if `fengdong/app.py`
  computes any of them, the task has failed regardless of what the window looks like. The scans in
  `tests/test_packaging.py` and `tests/test_widgets.py` already assert `flow/` never imports
  `fengdong/`; write the `app.py` half in the same shape — and keep `app.py` importing `flow`, never
  `lbm` (the `test_widgets_imports_flow_and_never_lbm` test already walks every file in `fengdong/`,
  so `app.py` is covered the moment it exists).
- **Constraint 13** — the setup panel shows a picture, a fluid, a speed and a size (plus quality, in
  words — **D-068**). Nothing else is an input. `tests/test_widgets.py`'s identifier scan runs over
  `fengdong/widgets.py` only; extend it to `fengdong/app.py` (or every file in `fengdong/`) rather
  than trusting the new file. The preview may *display* `tau` and the grid, because that is
  `explain()`'s printed output (D-060's exemption for output records), but no widget accepts one.
- **Constraint 14** — a refusal names a fix and the fix is machine-checked. The window shows the
  refusal `Case` returned and the suggestion `Case.nearest()` would act on, from the same object.
- **Constraint 10** — `fengdong/` colours nothing. The mask preview is a boolean array drawn as
  two flat colours (solid / fluid) — that is chrome, not a field. The moment a vorticity field
  reaches the screen it goes through `lbm.render.render`, and that moment is **T208**, not this
  session.
- **Constraint 20** — `fengdong --version` must still answer without pygame, numpy or `flow` loaded
  (`tests/test_packaging.py::test_fengdong_imports_without_numpy_flow_or_a_display` and
  `tests/test_widgets.py::test_fengdong_main_still_answers_without_pygame_or_widgets`). Import
  `fengdong.app` **inside** `main`, after `--version` has been handled. Rung I must still pass with
  `app.py` in the wheel.
- **Constraint 5** — the ladder is a gate. This task touches no solver or product code; `pytest` and
  Rung I are the minimum re-run, and say which physics rungs were not re-run and why.
- **`CLAUDE.md` § Commands** — nothing should need installing. If the window seems to need a
  package `myenv` does not have, that is a `D-0XX`, a row in § Environment **and** `pyproject.toml`
  in the same session — and it is almost certainly the trap.
- **Coding conventions** — type hints with intent, docstrings citing `DOCS/IDEA4.md` § What Phase 2
  is, concretely, and no allocation on the per-frame path (the widgets already cache their text; the
  mask preview surface should be built once per drop, not per frame).

### Blockers, open questions and decisions that affect you

**Blockers: none.**

- **D-097 (session 29)** — the widgets' event model, which the window must honour: characters arrive
  as `pygame.TEXTINPUT` (so the window must **not** call `pygame.key.stop_text_input()`), `KEYDOWN`
  is for editing keys only; `Widget.handle(events) -> bool` reports a change in *reported state*
  (value, selection, click, path) and never a focus change; `Panel` routes keys to the focused child
  only, gives an **open** `Dropdown` every click first, and walks Tab / Shift+Tab; `Dropdown.draw`
  takes `overlay=False` inside a `Panel` so the list is painted last; `draw` allocates nothing on an
  unchanged frame. The `App` loop is therefore: `events = pygame.event.get()`, `panel.handle(events)`,
  react to the widgets that changed, `panel.draw(screen)`, `flip`.
- **D-083** — pygame-only; `DROPFILE` carries no position on SDL 2.28, so the one `DropTarget` per
  window receives any file dropped anywhere on it. Session 29 confirmed `DROPBEGIN` / `DROPFILE` /
  `DROPCOMPLETE` all synthesise and dispatch headless.
- **D-096** — `fengdong/app.py` ships in the wheel automatically (`packages.find` takes `fengdong*`);
  touch nothing in `pyproject.toml`.
- **D-065 / D-066** — the repaired mask and its verdict come from `flow.prepare` through
  `Case.from_image`; the window shows what `Case` holds, not a second reading of the PNG.
- **D-067 / D-068** — `Case.explain()` is the plan preview's one source; `quality` is a word.
- **D-072 / D-073** — `python -m flow` and `python -m lbm.runner` survive underneath the window with
  the knobs it deliberately has not got; the window contradicts none of the CLI's flag semantics.
- **D-093** — the closure engages instead of refusing below `TAU_FLOOR`; the preview's *expected*
  band is `Plan.expected_fidelity`, and a closure-on plan is shown as such.
- **Q-204 (open — the user's)** — PyPI publication. Not this task's.
- **Q-205 (open — T208's)** — a trend wire on `Monitor`. Not this task's.
- **`022ac461c920`** — dead `smag_work` allocation on warp. T208's or a `/new-task`'s.
- **`d5b27e51fcdc`** — two power probes disagree; not reproduced in sessions 27–29. Left queued.

### Two environment facts sessions 24–29 paid for, so you do not have to

- **A foreground command is hard-capped at ten minutes.** Run the long rungs detached, redirected to
  a file under `outputs/ladder/`, with `-u`, and poll the file. `pytest` alone is ~5 min and fits;
  session 29 ran it detached anyway so the Rung I timing gate could follow on an idle machine.
- **`pytest` must not open a real window.** `tests/test_widgets.py` and `tests/test_render.py` set
  `SDL_VIDEODRIVER=dummy` before importing pygame; the `App` tests must do the same, and the window
  must be constructible headless (`pygame.display.set_mode` under the dummy driver is fine for a test
  that needs a display surface — but the widget tests assert the *widgets* never need one, and that
  must stay true).

### Before you start

- **Nothing to install.**
- **Rung I's clock is owed on mains** (above). Run `myenv/Scripts/python.exe -m validate.install` first,
  on mains and idled (`Win32_Battery.BatteryStatus` must read 2; check for stray `python` processes);
  it is also the cheap proof that `app.py` ships once it exists.
- **The manual gate is part of this task**: a human opens the window and drops a file, and what was
  seen is written into `DOCS/STATE3.md`. Plan for it before checkpointing, not after.

## Scope discipline

Work only what's in the contract. The live view, the numbers panel and save are **T208**; the drop
rung is **T209**. A sixth widget, a layout engine, theming or animation is the trap named in
`DOCS/PLAN3.md` § Risks, and the valve is `/new-task`. If it is listed under `DOCS/IDEA4.md`
§ Deliberately deferred (XLB, 3D, STL, KBC, MRT, curved boundaries, wall models, dynamic `Cs`, a web
UI), the answer is no.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it — including the manual gate.
2. Run `/validate` for every rung at or below this task — `pytest` and Rung I at minimum; say which
   physics rungs were not re-run and why.
3. Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
   `PROMPTS/031-…` for the next session. Do not end the session without it.
