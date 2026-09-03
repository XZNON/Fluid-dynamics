# Session 28 — T205: packaging — `pyproject.toml`, the `fengdong` distribution, Rung I, and M11

## What this project is

A validated 2D fluid simulator — D2Q9 lattice Boltzmann on NumPy and Warp backends — under a product
layer (`flow/`) that takes a picture and three physical numbers and returns a correct, moving answer,
now with a fidelity band attached to every result. **Phase 2 is live** and its spec is
`DOCS/IDEA4.md`. The solver is not the product: see `idea.md` § Risks — *"The trap"*, which names the
standing temptation to keep polishing it because that part is fun. **This task is not solver work and
not product work either** — it is the box the product ships in, and it is deliberately independent of
everything T201–T204 built (`DOCS/PLAN3.md` § Why this order, 4).

**Phase 2 is FengDong** (风洞, *wind tunnel*): a Smagorinsky turbulence closure, the fidelity bands
that make it safe to ship, and a pygame desktop application distributed as `pip install fengdong`.
**Two of the three are done.** T201 landed the closure on NumPy, T202 on Warp, T203 closed Rung G and
with it M9, and **T204 closed Rung H and with it M10. T205 is packaging, Rung I, and M11.**

## Read these first, in this order

1. `CLAUDE.md` — the **20 hard constraints** (**20**, **17**, **15** and **5** are the ones that
   govern this session), the session protocol, the conventions, the module map, § Commands (every
   command in it must still work from the tree afterwards), § Current state.
2. `DOCS/STATE3.md` — **in full.** § Snapshot, § Blockers (one queued issue is nearly a blocker for
   *this* task — `495777c58269`), § Open questions (**Q-204** is yours), § **Environment** (this is
   the list `pyproject.toml` has to reproduce **exactly**, and it is an acceptance criterion),
   § Performance baseline, § Decisions **D-080 … D-095**, the constraint fate table, and the
   session-log entries. The session-27 entry matters most.
3. `DOCS/TASKS3.md` § **T205** — the task contract, in full. Also skim § T206 (which depends on this
   one for the package layout, not for any behaviour) and the backlog index.
4. `DOCS/IDEA4.md` § **Scope** (the Windows-only claim, in writing), § **The five things Phase 2 must
   get right (5)** — *"One command installs it, on a machine that is not ours"* — and § **Validation
   ladder** (Rung I's row).
5. `DOCS/PLAN3.md` § Why this order (point **4** is this task's whole justification), § Session map,
   § **Milestone gates** (M11's gate command is literal), and § **Risks** — the row *"Packaging drags
   in a dependency the solver did not need"* is this task's own.
6. `.gitignore` (queued issue `495777c58269` lives there), `lbm/backends/__init__.py` (the optional
   Warp import is what `fengdong[gpu]` has to be shaped around), `lbm/record.py` (same, for
   `fengdong[video]`), and `validate/minute.py` for the shape of a rung that times a whole path from
   process start.

Decisions cited by number from earlier phases live in `old-Docs/STATE1.md` § Decisions (D-005 … D-040)
and `DOCS/STATE2.md` § Decisions (D-041 … D-079), **both frozen**. Read the entry a task names, not
the whole file.

Do not write code before finishing this reading.

## Where the project stands

- **Last session** was session 27 and it worked **T204 — the fidelity bands. It is `done`: every
  acceptance criterion was run and passed, and M10 was claimed on printed output.** What landed:
  `flow/fidelity.py` (`Band`, `band_for`, `sentence`, `Qualified`), `validate/fidelity.py` (Rung H),
  `tests/test_fidelity.py` (55 tests), plus the wiring through `flow/autoconfig.py`,
  `flow/report.py`, `flow/case.py`, `flow/diagnose.py` and `flow/cli.py`, and Rung D's D-038 section
  inverted. **Nothing in `lbm/` was touched** — the only edit outside `flow/`, `validate/` and
  `tests/` is a defaulted `sim` field on `CylinderResult`.
- **Rung status:** R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 · A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 · F 🟩 · G 🟩 ·
  **H 🟩** — **twelve green**, all re-run in session 27 on both backends where both apply, with no
  published digit moved. I ⬜ · J ⬜.
- **Milestones reached: M9 and M10** (both 2026-09-03). **M11 is yours.**
- **Completed tasks:** Phase 0 T001 … T011; Phase 1 T101 … T110; Phase 2 **T201**, **T202**, **T203**,
  **T204**.
- `pytest`: **894 passed, 2 skipped** (287.9 s).
- **Numbers to keep still.** Rung H: quantitative `Cd` **1.4030** on both backends; qualitative
  `max(nu_t)/nu` **0.6906** numpy / **0.6886** warp with **no `Cd` emitted**; illustrative
  `max(nu_t)/nu` **3.374e4** / **3.797e4**; Q-203's evidence `max(nu_t)/nu` **0.1057** with `Cd`
  **1.4143**, `St` **0.1719** on both. Rung E **48.2 s** (limit 60), `Cd` **1.4040**, `St` **0.1672**.
  Rung D `Monitor` cost **+1.96%** against a 2% limit. Rung 3 St **0.1731** Cd **1.4031**; Rung 4
  **1.5279** / **1.4276**; Rung 1 **0.3650%** / **0.3649%**; Rung 2 **0.75%** / **0.21 cells**;
  Rung A **5.960e-08** / **9.611e-06**; Rung F bitwise **0.000e+00**; Rung G **0.2303%** / **1.1547%**
  / bare `nu` **3.0178%** / **0.9972**.
- **The product command a user types now works on the case the project was built around**:
  `python -m flow --shape tests/data/shapes/disc.png --fluid air --speed "20 m/s" --size "1.5 m"`
  exits **0** and reports `illustrative` with no `Cd`. That command has to keep working from the tree
  after this task, and it has to work from an **installed wheel** too.

## Your task this session

**T205 — `pyproject.toml`, the `fengdong` distribution → Rung I → M11.** One task, this session only.

Run this first:

    /start-task T205

It will re-read the contract, restate goal and acceptance criteria, and wait for confirmation before
implementing. Confirm, then implement.

**Goal:** `pip install fengdong` works into a fresh virtual environment with no repository on the
path. The distribution is the deliverable; **nothing about the simulation changes.**

**In:** the working tree.
**Out:** `pyproject.toml` (PEP 621, setuptools or hatchling); packages `lbm`, `flow`, `fengdong`;
console entry point `fengdong = "fengdong.__main__:main"`; a built wheel and sdist;
`validate/install.py` (Rung I harness); a `fengdong/` package skeleton whose `main` prints a version
and exits, so the entry point is real before the app exists.

### Acceptance criteria (restated in full — this is what marks the task done)

- [ ] `pyproject.toml` declares name `fengdong`, and **every runtime dependency matches a row in `DOCS/STATE3.md` § Environment** — no dependency appears in the package that was not installed and recorded in a session. A test asserts the two lists agree.
- [ ] `python -m build` produces a wheel and an sdist; the wheel contains `lbm`, `flow` and `fengdong` and **no** `validate`, `tests`, `DOCS`, `myenv`, `outputs`, `Navier-Fluid-Equation` or `scripts`.
- [ ] `validate/install.py` builds the wheel, creates a **fresh venv**, installs the wheel into it, and runs `fengdong --version` plus a headless smoke of the app's model layer — **with no repository directory on `sys.path`**, asserted inside the child process, not assumed.
- [ ] The install-to-first-answer elapsed time is **printed**, and under **60 s** on a warm pip cache (**D-035** conditions quoted).
- [ ] Optional extras are declared for what is genuinely optional: the Warp backend (`fengdong[gpu]`) and recording (`fengdong[video]`). The base install runs the NumPy backend and the app.
- [ ] `myenv/Scripts/python.exe -m validate.install` prints **PASS**.
- [ ] The repository still works uninstalled: every existing command in `CLAUDE.md` § Commands runs unchanged from the tree.
- [ ] `pytest` green.

### Constraints that bite on this task

- **Constraint 20** (new in Phase 2, **D-083**) — *one `pip install`, one command.* **This task is
  where it stops being a design rule and becomes code.** The distribution is `fengdong`; Rung I
  installs a built wheel into a fresh venv with **no repository on the path**. A package that only
  installs from the developer's tree is not distributed.
- **Constraint 17** — `fengdong/` may import `flow/`; `flow/` may never import `fengdong/`, and a
  test asserts it. This task creates `fengdong/` for the first time, so it is where that test is
  written — and the contract asks for it to be asserted **inside the installed package**, not only in
  the tree.
- **Constraint 15** — `flow/` may import `lbm/`; `lbm/` may never import `flow/`. Same: the package
  layout must not create an import path that violates it, and the assertion must hold after install.
- **Constraint 5** — the ladder is ordered and non-negotiable, and **all twelve green rungs stay a
  gate**. T205 depends on none of T201–T204, but it can still break them: a `pyproject.toml` that
  makes a package importable by a different path, or a `.gitignore` fix that changes what is on disk,
  reaches every one of them.
- **`CLAUDE.md` § Commands** — adding a dependency means `myenv/Scripts/pip.exe install <pkg>` **and**
  a row in `DOCS/STATE3.md` § Environment, **in the same session**. `build` and a backend
  (`setuptools` or `hatchling`) are expected; anything else is a real decision.
- **Coding conventions** — type hints with intent, shapes in docstrings, docstrings cite
  `DOCS/IDEA4.md`, and `validate/install.py` prints pass/fail like every other rung.

### Blockers, open questions and decisions that affect you

**Blockers: none.** But one queued issue is effectively this task's:

- **`495777c58269`** — **`.gitignore` drops `*/__init__.py` and `tools/`.** Open since session 16.
  A wheel has to contain every `__init__.py` it ships, and a file that is not tracked is a file a
  clean checkout does not have. **Fix it here or explicitly carry it and say why** — do not discover
  it inside Rung I.
- **Q-204 (open — this task raises it, and the user decides)** — *"Does `fengdong` publish to PyPI
  inside this phase, or does Rung I's locally built wheel close it? Publishing needs an account, a
  `LICENCE` file and a considered first version number, and none of the three is a packaging
  detail."* Nothing is uploaded in this task either way; the question is whether Phase 2 ends with a
  published package.
- **D-083** — the distribution is named `fengdong` because **`flow` is taken on PyPI and `fengdong`
  is free** (both checked in session 23), so the distribution name and the import name differ *by
  necessity rather than by preference*, and `fengdong` is then also the command and the title bar.
  The app is **pygame-only**; `pygame.DROPFILE` and `DROPBEGIN` were confirmed present on the
  installed pygame 2.6.1 / SDL 2.28.4.
- **D-080** — **no XLB.** `xlb` 0.3.1 imports at warp-lang 1.11.0 and fails at 1.14.0 and our 1.16.0,
  so adopting it would pin warp five minors back for our own validated backend. It is not a
  dependency and must not become one.
- **D-035** — every absolute timing figure is quoted with `Win32_Processor.CurrentClockSpeed`, the
  power state and the GPU name beside it. Rung I has a 60 s timing clause, so this applies to it.
- **D-092** — a wall-clock A/B that fails under load is re-run on an idled machine and **both**
  readings are recorded. Rung I's install time is a wall-clock claim; expect to need this.
- **D-093 / D-094 / D-095 (session 27, and they are why `flow/` looks the way it does)** — the
  closure now engages instead of refusing below `TAU_FLOOR`; `Monitor`'s speed and mass wires move to
  the lattice sound speed and half the domain's mass on a closure-on run; and every `Result` carries
  a `flow.fidelity.Band` that gates its coefficients. **None of it should need touching here** — if
  packaging seems to require a change in `flow/`, that is a signal the layout is wrong, not the
  product.
- **`022ac461c920`** — `Sim` allocates `smag_work` `(4, ny, nx)` on the Warp backend, which never
  reads it (32 MiB of dead device memory at 2M cells with the closure on). Not a blocker, not this
  task's; T208's or a `/new-task`'s.
- **`d5b27e51fcdc`** — `validate/refusals.py` and `validate/minute.py` disagree about the power
  state. Session 27 could not reproduce it. Left queued; **`validate/install.py` will need a power
  probe of its own, so prefer sharing one implementation over writing a third.**

### Three environment facts sessions 24–27 paid for, so you do not have to

- **A foreground command is hard-capped at ten minutes and is killed on the dot.** Run long rungs
  **detached, in the background, redirected to a file under `outputs/ladder/`, and poll the file.**
  A `nohup … &` inside a tool call returns immediately and the wrapper reports success for the
  *shell*, not the work — poll the output file, not the exit code. Python block-buffers a redirected
  stdout unless you pass **`-u`**. Measured wall clocks: **Rung 4 ~27–43 min per backend**, **Rung H
  ~63 min on numpy and ~5 min on warp**, **Rung B numpy ~3 h**.
- **Idle the machine before any timing gate; do not merely read the clock.**
  `Win32_Processor.CurrentClockSpeed` is instantaneous and reports 3201 while sustained load is being
  clocked well below it. Session 27 killed the ladder, idled seven minutes, and only then ran Rungs D
  and E — which read **+1.96%** (limit 2%) and **48.2 s** (limit 60), the best Rung E figure on file.
  **Rung I's 60 s install claim is a timing gate and needs the same treatment.**
- **Before timing anything, check for stray processes**
  (`Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' }`) and kill them. Also
  **check the power state** — `Win32_Battery.BatteryStatus` is `2` on mains, `1` discharging. Session
  26 lost most of a night to the laptop running on battery at 1882 of 3201 MHz.

### Before you start

- **Two packages are expected and are the only ones that should be needed:** a build front-end
  (`build`) and a backend (`setuptools` **or** `hatchling` — pick one and record which and why).
  Install with `myenv/Scripts/pip.exe install <pkg>` and **add a row to `DOCS/STATE3.md`
  § Environment in the same session**. Anything beyond those two is a real decision and needs a
  `D-0XX`.
- **`DOCS/STATE3.md` § Environment is the authority for runtime dependencies** and one of this task's
  acceptance criteria is that `pyproject.toml` agrees with it exactly. As of session 27 that list is:
  numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, pytest 9.1.1, pygame 2.6.1, imageio 2.37.4,
  imageio-ffmpeg 0.6.0, psutil 7.2.2, warp-lang 1.16.0. Decide which of those are **runtime** and
  which are **development or optional** — `pytest` is not a runtime dependency, `warp-lang` is the
  `[gpu]` extra, `imageio` / `imageio-ffmpeg` are the `[video]` extra, and the rest need a defensible
  home. That decision is the substance of the first acceptance criterion.
- **Rung B's numpy half was still running when session 27 checkpointed.** Its warp half is green and
  its numpy figures in § Snapshot are session 26's, labelled as carried. If it matters to you,
  `outputs/ladder/B_numpy.txt` may have finished by now; otherwise re-run it detached.

## Scope discipline

Work only what's in the contract. The widget set is **T206**, the window is **T207**, the live view
is **T208** and the drop is **T209** — none of them is this session, and the `fengdong/` package this
task creates is a **skeleton whose `main` prints a version and exits**, nothing more. If something
else needs doing, `/new-task` it; do not expand this one. If it is listed under `DOCS/IDEA4.md`
§ Deliberately deferred (XLB, 3D, STL, KBC, MRT, curved boundaries, wall models, dynamic `Cs`, a web
UI), the answer is no.

## Verify, then close

1. Run every acceptance criterion. Running it, not reading it.
2. Run `/validate` for every rung at or below this task — **all twelve** — and confirm nothing
   regressed. T205 depends on none of them and can still break all of them.
3. Run **`/checkpoint`**. It updates `DOCS/STATE3.md`, syncs `DOCS/TASKS3.md`, and writes
   `PROMPTS/029-…` for the next session. Do not end the session without it.
