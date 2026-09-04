# Session 29 — T206: `fengdong/widgets.py` — the closed widget set, unit-tested headless

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer
with a fidelity band attached, now **shipped as a wheel named `fengdong`**. **Phase 2 is live** and its
spec is `DOCS/IDEA4.md`. The solver is not the product: see `idea.md` § Risks — *"The trap"*, which
names the standing temptation to keep polishing it because that part is fun. **This task is the first
piece of the application itself**, and `DOCS/PLAN3.md` § Risks names *its* trap by name: *"Hand-rolled
widgets swallow the phase."*

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
The closure (T201, T202) and the bands (T204) are done; the box the app ships in (T205) is done and
**Rung I is green — M11**. What remains is the app: **T206** widgets, T207 window, T208 live view,
T209 the drop and Rung J.

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (**17**, **13**, **10** and **5** are the ones that govern
   this session), the session protocol, the conventions, the module map (the `fengdong/` rows), and
   § Current state (the T205 paragraph says what the package looks like now).
2. `DOCS/STATE3.md` — **in full.** § Snapshot, § Blockers, § Open questions (**Q-204** is the user's,
   **Q-205** is T208's), § Environment (pygame 2.6.1 / SDL 2.28.4 is what you build on — nothing gets
   installed this session), § Decisions **D-080 … D-096** (**D-083** is the one that closed the widget
   set; **D-096** is the package you are writing into), and the session-28 log entry.
3. `DOCS/TASKS3.md` § **T206** — the task contract, in full. Also § T207 (what will consume the
   widgets, so you know the shape they need) and the backlog index.
4. `DOCS/IDEA4.md` § **What Phase 2 is, concretely** (the diagram: `fengdong/widgets.py` is the top
   box, and *"the app computes nothing"*), § **The five things Phase 2 must get right (3)** — *"The app
   is a view, not a second brain"* — and (4).
5. `DOCS/PLAN3.md` § Why this order (point **5**, *"Widgets before the window"*), § Session map, and
   § **Risks** — the row *"Hand-rolled widgets swallow the phase"* is this task's own, and its valve
   is written down.
6. `lbm/render.py` (constraint 10: how frames already become pixels — the widgets draw **chrome**,
   never fields), `flow/quantity.py::parse` (the one parser a `TextField` validates through),
   `flow/fluids.py::FLUIDS` (what the `Dropdown` is populated from), `fengdong/__init__.py` and
   `fengdong/__main__.py` (the skeleton you are adding to; `__main__` must keep answering `--version`
   without importing pygame), and `tests/test_flow_package.py` + `tests/test_packaging.py` (the
   constraint-13 and constraint-17 scans the new module has to pass).

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 28 and it worked **T205 — packaging. It is `done`: every acceptance
  criterion was run and passed, and M11 was claimed on printed output.** What landed:
  `pyproject.toml` (setuptools, PEP 621, **D-096**), `MANIFEST.in`, `fengdong/__init__.py`
  (`__version__ = "0.2.0"`) and `fengdong/__main__.py` (`main` prints the version and exits 0),
  `validate/install.py` (Rung I), `tests/test_packaging.py` (27 tests), and the `.gitignore` fix that
  closed queued issue `495777c58269` — `lbm/backends/__init__.py` and `tools/issues.py` are tracked
  for the first time. **Nothing in `lbm/` or `flow/` changed.**
- **Rung I, measured:** PASS, **52.6 s** from venv creation to the installed smoke's finite state
  against a 60 s limit on a warm pip cache (venv 13.4 s, pip 31.7 s, `--version` 0.19 s, smoke 7.3 s),
  at 3201 of 3201 MHz on mains, RTX 3050, driver 592.82. The margin is **7.4 s** and most of the
  clock is pip's, not ours.
- **Rung status:** R1–R4 🟩 · A–E 🟩 · F 🟩 · G 🟩 · H 🟩 · **I 🟩** · J ⬜ — **all fourteen re-run in session 28 on
  both backends where both apply, nothing carried, every published digit unmoved** (the three
  timing gates after a seven-minute idle: D **−0.39%**, E **55.0 s**, I **55.5 s**; Rung B's numpy
  half included this time, accuracy **1.7%**).
- **Milestones reached: M9, M10, M11** (all 2026-09-03). **M12 is T209's.**
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110; Phase 2 **T201 … T205**.
- `pytest`: **921 passed, 2 skipped** (286.2 s).
- **Numbers to keep still.** Rung H: quantitative `Cd` **1.4030** on both backends; qualitative
  `max(nu_t)/nu` **0.6906** numpy / **0.6886** warp with **no `Cd` emitted**; illustrative
  `max(nu_t)/nu` **3.374e4** / **3.797e4**; Q-203's evidence `max(nu_t)/nu` **0.1057** with `Cd`
  **1.4143**, `St` **0.1719** on both. Rung E `Cd` **1.4040**, `St` **0.1672**. Rung 3 St **0.1731**
  Cd **1.4031**; Rung 4 **1.5279** / **1.4276**; Rung 1 **0.3650%** / **0.3649%**; Rung 2 **0.75%** /
  **0.21 cells**; Rung A **5.960e-08** / **9.611e-06**; Rung F bitwise **0.000e+00**; Rung G
  **0.2303%** / **1.1547%** / bare `nu` **3.0178%** / **0.9972**.

## Your task this session

**T206 — `fengdong/widgets.py`, the closed widget set.** One task, this session only.

Run this first:

    /start-task T206

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** Five widgets, tested without a screen. The set is closed at the start of the task and stays
closed.

**In:** pygame surfaces and events.
**Out:** `fengdong/widgets.py` — `Label`, `TextField`, `Dropdown`, `Button`, `DropTarget`, and a
`Panel` that lays them out in a column. Each takes a rect, draws to a surface, and consumes an event
list returning whether it changed.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] Exactly those five widgets plus `Panel`. **No layout engine, no theming, no animation, no focus chain beyond tab order.** A sixth widget is `/new-task`.
- [ ] Every widget is **testable headless**: constructed and driven with synthesised `pygame.event` objects against an off-screen `Surface`, with no window opened. `pytest` runs them under `SDL_VIDEODRIVER=dummy` and the test asserts no display is initialised.
- [ ] `TextField` validates through `flow.quantity.parse` and shows the parse error **in the user's words** when it fails — the same message the CLI prints, obtained from the same code path, not re-worded (**constraint 14**'s posture).
- [ ] `Dropdown` for fluids is populated from `flow.fluids.FLUIDS` at construction; adding a fluid to the library adds it to the widget with no edit here, asserted by test.
- [ ] `DropTarget` consumes `pygame.DROPFILE` and reports the path; a test synthesises the event rather than requiring a human to drag anything.
- [ ] **Constraint 13:** no widget accepts or displays a lattice quantity. A test scans the module for the vocabulary the Phase 1 scan already forbids in `flow/`.
- [ ] **Constraint 17:** `fengdong/` imports `flow/`; `flow/` never imports `fengdong/`. A test asserts it, in the same shape as the existing constraint-15 test.
- [ ] `pytest` green, new tests counted.

### Constraints that bite on this task

- **Constraint 17** (**D-083**) — `fengdong/` may import `flow/`; `flow/` may never import
  `fengdong/`. **The test already exists** — `tests/test_packaging.py` scans `flow/` and `lbm/` for
  `fengdong` imports and walks every `flow.*` module at runtime — so this task inherits a guard and
  must keep it green; if the contract's own copy of that test is wanted beside the widgets, write it
  in the same shape rather than a third one. Rung I re-asserts the direction inside the installed
  wheel every time it runs.
- **Constraint 13** — no lattice quantity in a `fengdong/` widget. `tests/test_flow_package.py`'s
  `LATTICE_NAMES` is the vocabulary (it gained `cs_smag` / `cs` in T204); the scan over
  `fengdong/widgets.py` uses **that** list, not a copy. `Cs` is not a knob; the fidelity band is what
  a person sees instead.
- **Constraint 10** — one `render()`, and it lives in `lbm/render.py`. The widgets draw chrome —
  borders, text, a highlight — and never a field. `tests/test_flow_package.py::test_flow_defines_no_renderer_of_its_own`
  is the shape of the test that says so; `fengdong/` needs the same one.
- **Constraint 5** — the ladder is ordered and non-negotiable, and **all thirteen green rungs stay a
  gate**. This task touches no solver and no product code, and a widget module cannot move a digit —
  but `pytest` is part of green, and the constraint-13/17 scans are the ones a widget module can fail.
- **Constraint 20** — the wheel must still build and Rung I must still pass with the new module in it.
  `fengdong/widgets.py` imports pygame at module scope, which is fine (pygame is a runtime
  dependency, **D-096**) — but `fengdong/__main__.py` must **not** import `widgets` at module scope,
  because `fengdong --version` is required to answer without pygame or numpy loaded and
  `tests/test_packaging.py::test_fengdong_imports_without_numpy_flow_or_a_display` asserts it.
- **`CLAUDE.md` § Commands** — nothing should need installing. pygame 2.6.1 / SDL 2.28.4 is in
  `myenv` and in the wheel's runtime set; `pygame.DROPFILE` and `pygame.DROPBEGIN` were confirmed
  present in session 23. If a widget genuinely needs a package `myenv` does not have, that is a
  `D-0XX` and a row in § Environment **and** `pyproject.toml` in the same session — and it is almost
  certainly the trap.
- **Coding conventions** — type hints with intent, docstrings citing `DOCS/IDEA4.md` § What Phase 2
  is, concretely, and no allocation inside any per-frame path a widget will sit on.

### Blockers, open questions and decisions that affect you

**Blockers: none.**

- **D-083** — the widget set is **closed at five plus `Panel`**: `Label`, `TextField`, `Dropdown`,
  `Button`, `DropTarget`. pygame-only, because it is already a dependency, already the sink
  `lbm/render.py` feeds, and `DROPFILE` costs nothing. *"The price accepted is a hand-rolled widget
  layer, and it is bounded by closing the set on day one."* The fall-back, named in
  `DOCS/PLAN3.md` § Risks and `DOCS/TASKS3.md` § T206 Notes: a native file dialog plus keyboard entry,
  which needs no widgets at all — so **`DropTarget` is the one widget that is not negotiable**, and
  it is the one to write first.
- **D-096** — the package you are writing into. `fengdong/` ships in the wheel; `fengdong/__main__.py`
  is the console entry point and stays import-light; `__version__` lives in `fengdong/__init__.py`.
  Add `widgets.py` beside them; touch nothing in `pyproject.toml`.
- **D-093 / D-094 / D-095** — why `flow/` looks the way it does: the closure engages below
  `TAU_FLOOR`, `Monitor` widens on a closure-on run, and every `Result` carries a
  `flow.fidelity.Band`. **None of it is this task's**, but the `TextField` criterion routes through
  `flow.quantity.parse` and the `Dropdown` through `flow.fluids.FLUIDS`, so those two modules are the
  whole of `flow/` this task reads.
- **Q-204 (open — the user's)** — PyPI publication. Not this task's; nothing here changes the answer.
- **Q-205 (open — T208's)** — a trend wire on `Monitor`. Not this task's.
- **`022ac461c920`** — dead `smag_work` allocation on warp. T208's or a `/new-task`'s.
- **`d5b27e51fcdc`** — two power probes disagree. Session 28 could not reproduce it either (mains,
  3201 MHz all session, `bench.machine_state` reporting correctly from three rungs). Left queued.

### Two environment facts sessions 24–28 paid for, so you do not have to

- **A foreground command is hard-capped at ten minutes.** This task should need no long rung — but
  if you re-run the ladder, run the long ones (Rung 4 ~35 min per backend, Rung H numpy ~63 min,
  Rung B numpy ~3 h) **detached, redirected to a file under `outputs/ladder/`, with `-u`, and poll
  the file**. Session 28's driver ran the fast rungs first, idled seven minutes, ran the three timing
  gates (D, E, I), then the long ones — that order banks the timing claims early.
- **`pytest` must not run with pygame opening a real window.** The T206 contract's second criterion
  is `SDL_VIDEODRIVER=dummy`; set it in the test module *before* `pygame` is imported, or in
  `tests/conftest.py` for the whole run, and assert `pygame.display.get_init()` is false.

### Before you start

- **Nothing to install.** pygame 2.6.1 is present; the wheel already lists it as runtime.
- **Rung I is green and the wheel builds.** `myenv/Scripts/python.exe -m validate.install` takes
  ~75 s (build + venv + install + smoke) and is the cheap way to prove the new module ships.
- The M11 gate was run in full in session 28; nothing is owed from it.

## Scope discipline

Work only what's in the contract. The window is **T207**, the live view is **T208**, the drop is
**T209** — none of them is this session. **A sixth widget, a layout engine, theming, animation, or a
focus chain beyond tab order is the trap**, and the valve is `/new-task` or the file-dialog fall-back.
If it is listed under `DOCS/IDEA4.md` § Deliberately deferred (XLB, 3D, STL, KBC, MRT, curved
boundaries, wall models, dynamic `Cs`, a web UI), the answer is no.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate` for every rung at or below this task — `pytest` and Rung I at minimum; the
   thirteen physics and product rungs cannot move from a widget module, but say so rather than
   assume it if you skip them, and record which were re-run.
3. Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
   `PROMPTS/030-…` for the next session. Do not end the session without it.
