# TASKS3.md — Phase 2 task contracts

One task per session. Plan and ordering rationale: `DOCS/PLAN3.md`. Live status: `DOCS/STATE3.md`.
Phase 1's closed backlog is `DOCS/TASKS2.md` (T101 → T110) and Phase 0's is `old-Docs/TASKS1.md`
(T001 → T011) — read them, never edit them.

**Status vocabulary:** `not_started` · `in_progress` · `blocked` · `done`
A task is `done` only when **every** acceptance criterion is checked. Code written ≠ done.

**Numbering:** Phase 2 tasks are `T2xx` so they can never collide with `T0xx` or `T1xx`.

---

## Backlog index

| ID | Title | Status | Depends on | Gate |
|---|---|---|---|---|
| T201 | Smagorinsky closure: `lbm/core.py` + NumPy backend | `done` | — | **Rung F** (numpy) 🟩 |
| T202 | The closure on the Warp backend | `done` | T201 | **Rung F** (full) 🟩 + Rung A 🟩 |
| T203 | Taylor–Green harness | `done` | T202 | **Rung G** 🟩 → **M9** 🟩 |
| T204 | `flow/fidelity.py` — the bands, wired through | `done` | T203 | **Rung H** 🟩 → **M10** 🟩 |
| T205 | Packaging: `pyproject.toml`, the `fengdong` distribution | `not_started` | — | **Rung I** → **M11** |
| T206 | `fengdong/widgets.py` — the closed widget set | `not_started` | T205 | unit tests, headless |
| T207 | `fengdong/app.py` — window, drop target, setup panel | `not_started` | T206 | manual gate + tests |
| T208 | Live view, numbers panel, save, refusal UI | `not_started` | T207, T204 | manual gate + tests |
| T209 | The drop: end to end, timed | `not_started` | T208 | **Rung J** → **M12** |

---

## T201 — Smagorinsky closure: `lbm/core.py` + NumPy backend

**Status:** `done` (session 24, 2026-09-02)

### Goal

BGK gains an optional per-cell effective relaxation time computed from the second moment of
`f - feq`. With `Cs = 0` it is **bitwise** the collision Phase 1 shipped, on every path. The NumPy
backend implements it; Warp is T202. Nothing gets faster and nothing gets more accurate — the
switchability is the deliverable.

### Reads / depends on

- `DOCS/IDEA4.md` § The five things Phase 2 must get right (1) and (2), § Validation ladder Rung F
- `lbm/core.py` (`collide`, `collide_stream`, the D-011 / D-020 / D-033 order), `lbm/backends/__init__.py` (**D-054**)
- `old-Docs/STATE1.md` **D-011**, **D-020**, **D-033**; `DOCS/STATE2.md` **D-054**, **D-055**
- Tasks: none

### Inputs / outputs

**In:** `f`, `feq` `(9, ny, nx)` `float32`; `tau` float; `cs_smag` float, default `0.0`.
**Out:** `lbm/core.py::smagorinsky_omega(f, feq, tau, cs_smag, out=None) -> NDArray[np.float32]`
returning `(ny, nx)` `float32` per-cell inverse relaxation time; `lbm/core.py::collide` and
`::collide_stream` gain a keyword-only `cs_smag: float = 0.0`; `Backend.collide` and
`Backend.collide_stream` gain the same in the protocol; `lbm/backends/numpy_backend.py` implements
it; `lbm/probe.py::eddy_viscosity(f, feq, tau, cs_smag) -> NDArray[np.float32]` `(ny, nx)`, the
`nu_t` field Rung G and constraint 18 both need; `validate/les.py` (Rung F harness).

### Acceptance criteria

- [x] `smagorinsky_omega` cites its source in the docstring — Hou et al. (1996) or equivalent — and the **exact algebra is pinned by a test**, not just the shape. The filter width is one lattice unit and the docstring says so.
- [x] **Bitwise degeneracy, both paths, asserted first:** with `cs_smag=0.0`, `collide` and `collide_stream` produce `f` `numpy.array_equal` to the Phase 1 functions after 1000 steps of Rung 3's case. Fused and unfused agree bitwise with each other, as **D-055** already requires.
- [x] The closure **defaults off**: no call site in `lbm/`, `flow/` or `validate/` passes `cs_smag` unless it is testing the closure. `git grep cs_smag` in those trees returns only the closure's own definitions and tests.
- [x] `nu_t = cs2 * (tau_eff - tau)` is derived through `tau`, never as a viscosity assigned directly — **constraint 2** applies to the effective relaxation time exactly as it applies to the base one, and a test asserts `nu_t >= 0` everywhere and `nu_t == 0` when `cs_smag == 0`.
- [x] `tau_eff >= tau` for every cell, always (the closure adds viscosity and never removes it); asserted on a case with strong shear.
- [x] **No allocation inside the step loop.** `smagorinsky_omega` takes an `out=` buffer, `Sim` preallocates it only when the closure is on, and a test asserts the allocation count is unchanged when it is off.
- [x] `validate/les.py` exists and prints PASS/FAIL. On numpy it asserts (a) the bitwise clause above and (b) Rung 3's case with `cs_smag=0.17` still prints `Cd` in 1.25–1.45 and `St` in 0.155–0.175.
- [x] `myenv/Scripts/python.exe -m validate.les` prints **PASS**.
- [x] **All nine existing rungs re-run and print their published digits.** R1–R4, A–E. Any moved digit is a stop-work.
- [x] `pytest` green, with the new tests counted in `DOCS/STATE3.md`.

### Constraints that bite here

- **Constraint 1 (Phase 2 form)** — one closure, named, and nothing else. No KBC, no MRT, no dynamic `Cs`.
- **Constraint 2** — `nu` still comes from `tau`. The closure modifies `tau`; it never sets a viscosity.
- **Constraint 19** — `Cs = 0` is bitwise Phase 1. This is the criterion that protects the other nine rungs.
- **Constraint 4** — the nine constants still come from `lbm/core.py` only.
- **Coding conventions** — preallocate; `float32`; docstrings cite `DOCS/IDEA4.md`.

### Notes

The temptation here is to reach for XLB's form, which session 23 read at
`Autodesk/XLB:xlb/operator/collision/smagorinsky_les_bgk.py` (it has an explicit 2D branch and a Warp
implementation). Read it as a **cross-check**, not as a source: their normalisation
(`tau = 0.5 (tau0 + sqrt(tau0^2 + 36 Cs^2 sqrt(strain)))`) bakes in choices about the filter width and
the strain norm that our docstring has to state explicitly and our test has to pin. Two
implementations agreeing is evidence; one copied is not.

`Cs = 0.17` is the literature constant and this phase does not tune it. If a case wants a different
value, that is a decision with a measurement, recorded — not an edit.

---

## T202 — The closure on the Warp backend

**Status:** `done` (session 25, 2026-09-02)

### Goal

The same arithmetic on the GPU, agreeing with NumPy to the tolerance **D-053** and **D-056** already
established, and costing less than 25% of the BGK step rate.

### Reads / depends on

- `DOCS/IDEA4.md` § Performance budget, § Validation ladder Rung F
- `lbm/backends/warp_backend.py`; `DOCS/STATE2.md` **D-052**, **D-053**, **D-054**, **D-056**, **D-057**
- Tasks: T201

### Inputs / outputs

**In:** the T201 signatures, unchanged.
**Out:** `cs_smag` support in the Warp backend's `collide` and `collide_stream` kernels;
`validate/les.py` gains `--backend`; `bench.py` gains `--les`; the LES row of
`DOCS/STATE3.md` § Performance baseline.

### Acceptance criteria

- [x] **Bitwise degeneracy on warp too:** `cs_smag=0.0` gives `f` bitwise identical to the Phase 1 warp kernel after 1000 steps. Not "within tolerance" — identical. The closure must compile out or multiply by zero without touching the result.
- [x] Cross-backend agreement with the closure **on** meets the existing contract: per-kernel worst under **1e-6** in `f` units (**D-053**'s bar), whole step under **1e-4** in `max|Δu|/U` at 1000 steps (**D-056**'s bar). The measured numbers are printed and recorded, not just compared.
- [x] Any `float64`-then-rounded scalar the closure needs is computed **host-side in NumPy's own expression order** and uploaded (**D-057**). No per-thread `float32` recomputation of a constant.
- [x] `myenv/Scripts/python.exe -m validate.les --backend warp` prints **PASS**.
- [x] `myenv/Scripts/python.exe -m validate.parity --backend warp` re-run and prints **PASS** with the closure off; its published numbers unmoved.
- [x] `bench.py --backend warp --les` clears **≥3116 / ≥568 / ≥331** steps/s at 40k / 1M / 2M cells, quoted with the CPU clock, power state and GPU name (**D-035**), by alternating rounds. The NumPy column takes the same 25% rule against its own measured baseline.
- [x] **All nine existing rungs re-run on both backends** and print their published digits.
- [x] `pytest` green.

### Constraints that bite here

- **Constraint 6's replacement** — no backend optimisation before its parity rung passes. Rung F on warp is that rung for this feature.
- **Constraint 11** — restart bit-identical within a backend, a printed tolerance across; the closure's state is `tau_eff`, which is derived and therefore adds nothing to the checkpoint. Assert that: `f`, `mask`, `step_count` are still the entire state (**D-022**, **D-050**).
- **Constraint 19**, **constraint 4**, **D-035**.

### Notes

If the closure cannot be made bitwise-degenerate on warp — a fused multiply-add contracting
differently in the `cs_smag=0` branch is the plausible way that happens — **the branch is the fix, not
the tolerance**. Compile two kernels, or guard the whole term. **D-053** already documents that
`collide` and `equilibrium` differ from NumPy by an FMA contraction; that is accepted *between*
backends and is not acceptable *within* one against its own previous self.

**What session 25 did (D-088, D-089).** Two compiled kernels, as the paragraph above anticipated:
`cs_smag = 0` launches `_collide_kernel` / `_collide_bb_kernel` **unedited**, so bitwise degeneracy is
by construction and **Q-201** never gets asked. The closure's own kernels are separate —
`_smag_scale_kernel` (the reduction) plus `_collide_smag_kernel` on the unfused path, and a single
`_collide_bb_smag_kernel` that folds the reduction into the fused pass, because a separate scale
kernel is a second full pass over `f` and `feq` on a memory-bound step (27.7% of the BGK rate at 2M
cells as two kernels, 8.8% as one). `validate/les.py` gained `--backend`, a frozen Phase 1 **warp**
oracle under D-087's one-copy rule, and a fourth clause measuring cross-backend agreement with the
closure **on** against Rung A's own bars; `validate/parity.py::step_case` and `::whole_step` gained a
defaulted `cs_smag` so that clause runs Rung A's case rather than a copy of it.

---

## T203 — Taylor–Green harness → Rung G → M9

**Status:** `done` (session 26, 2026-09-03)

### Goal

Prove the closure adds the viscosity it claims to add and no more, against an exact analytic solution
rather than against a benchmark table.

### Reads / depends on

- `DOCS/IDEA4.md` § Validation ladder Rung G
- `validate/poiseuille.py` (Rung 1 — the existing analytic-solution harness, and the source of the 1% bar)
- Tasks: T202

### Inputs / outputs

**In:** a doubly periodic domain, no bodies.
**Out:** `validate/taylorgreen.py` printing PASS/FAIL, taking `--backend` and `--cs`.

### Acceptance criteria

- [x] The harness initialises the exact 2D Taylor–Green vortex, `u = u0 cos(kx) sin(ky)`, `v = -u0 sin(kx) cos(ky)`, on a periodic domain, and measures the decay rate of the kinetic energy against `exp(-2 nu k^2 t)`.
- [x] With `--cs 0`: the measured viscosity returns `nu = (tau - 0.5)/3` to **under 1%**, the bar Rung 1 already meets. This is a fourth independent check on the base solver and it must pass before any LES number is believed. — **0.2303%**, `ln E` fit `R^2` 1.000000
- [x] With `--cs 0.17`: the measured viscosity returns `nu + <nu_t>` to **under 2%**, where `<nu_t>` is the domain average of `lbm.probe.eddy_viscosity` **computed from the model during the run**, not fitted to the decay curve afterwards. A fitted `nu_t` proves nothing and the test says so. — **1.1547%**; the "not fitted" half is asserted by an **AST** test as well as by value
- [x] The peak lattice velocity stays under 0.1 throughout (constraint 3) and the harness prints it. — **0.08000**, sampled through the warm-up too, so "throughout" is measured
- [x] Both backends pass, and the printed digits agree to the **D-056** whole-step tolerance. — `max|du|/u0` **1.150e-05** against 1e-4; the measured `nu` agrees to **1.434e-06**
- [x] `myenv/Scripts/python.exe -m validate.taylorgreen` and `--backend warp` both print **PASS**.
- [x] **M9 gate run in full** — Rungs F and G on both backends, all nine existing rungs re-run, and `bench.py --backend warp --les` clearing its floors. Milestone claimed only on printed output. — 18 ladder runs + Rung E + bench; every published digit unmoved; bench **3504.0 / 661.6 / 403.7** against **3116 / 568 / 331**
- [x] `pytest` green. — **827 passed, 2 skipped**
- [x] *(added in session 26, **D-091**)* The `Cs = 0.17` clause cannot pass with the `<nu_t>` term deleted: bare `nu` must **fail** the same 2% bar, and the measured excess must equal the **dissipation-weighted** `<nu_t> = <nu_t^3>/<nu_t^2>` to 5%. — **3.0178%** and **0.9972**

### Constraints that bite here

- **Constraint 5** — this is a rung; it prints PASS/FAIL and Rung H does not start while it fails.
- **Constraint 3** — the ceiling is checked, not assumed.
- **Constraint 12** — no bodies here, so the geometry checks are vacuous; say so in the docstring rather than leaving a reader to wonder.

### Notes

Taylor–Green is chosen over decaying 2D turbulence because it has an **exact** solution and this
project's ladder is built on known answers, not on statistical scalings. The enstrophy-cascade check
is a better test of a turbulence model and a worse test of *this* claim, which is that the closure
adds a known, small, computable amount of viscosity to a flow that is fully resolved.

Expect `<nu_t>` at `Cs = 0.17` on a resolved Taylor–Green to be a small fraction of `nu`. If it is
not — if the model fires hard on a smooth flow — that is a finding about the implementation, and it
belongs in `DOCS/STATE3.md` § Decisions with its measurement.

**What session 26 did (D-091), and the trap it found.** `<nu_t>/nu` came out at **1.8418%** and the
model does not fire hard on a smooth flow — but that number is a **design output, not a fact about
Taylor–Green**. It scales as `0.147 u0 / (L nu)`, and at a comfortable resolved point (L = 64,
`tau = 0.55`, `u0 = 0.05`) it is **0.14%**, at which the 2% bar above passes *with the `<nu_t>` term
deleted*. A green rung that proves nothing is exactly the failure mode constraint 5 names, so the
case is **sized** (64x64, `tau = 0.52`, `u0 = 0.08`) and a **discriminator** clause was added: bare
`nu` must miss the same 2% bar, and it misses by **3.0178%**.

Two further facts worth carrying. Taylor–Green has `S_xy = 0` identically, so `nu_t` is non-uniform
and the energy decay responds to the **dissipation-weighted** mean, which is **1.7780x** the domain
average analytically and **1.69–1.79x** across every case measured — the contract's domain-average
comparison is therefore systematically high by `0.78 <nu_t>`, which is why its 2% bar and the
discriminator's 2% bar together admit only `<nu_t>/nu` in roughly **1.1%–2.6%**. That window is
narrow but *deterministic*: it is a property of the case, not of the run, and both backends land in
it to seven digits. And the bias vanishes entirely if the model's own field is re-weighted as
`<nu_t^3>/<nu_t^2>` — no analytic input, no fitting, since `S_ab S_ab` is proportional to `nu_t^2`
with every constant cancelling — against which the measured excess is **0.9972**.

---

## T204 — `flow/fidelity.py` — the bands, wired through → Rung H → M10

**Status:** `done`

### Goal

Every result carries a band, and the band's claim is machine-checked. This is the task that makes the
closure safe to ship, and it is the most product-defining module of the phase the way `autoconfig`
was of Phase 1.

### Reads / depends on

- `DOCS/IDEA4.md` § The five things Phase 2 must get right (1), § Validation ladder Rung H
- `flow/autoconfig.py` (**D-059**, **D-075**, **D-079**), `flow/diagnose.py` (**D-045**, **D-061**, **D-063**), `flow/report.py` (**D-069**, **D-070**, **D-071**), `flow/case.py` (**D-067**, **D-068**)
- `old-Docs/STATE1.md` **D-029**, **D-032**, **D-036**, **D-038**
- Tasks: T203

### Inputs / outputs

**In:** a planned case and, after a run, the `nu_t` field.
**Out:** `flow/fidelity.py::Band` (an enum or frozen dataclass: `quantitative` / `qualitative` /
`illustrative`), `::band_for(plan, nu_t_max=None) -> Band`, `::sentence(band) -> str`;
`flow.autoconfig.plan` gains the closure as a planned parameter; `Result` gains `fidelity`;
`validate/fidelity.py` (Rung H harness).

### Acceptance criteria

- [x] `band_for` implements `DOCS/IDEA4.md`'s table exactly: **quantitative** iff `Re <= 200` and `max(nu_t)/nu < 0.1`; **qualitative** iff `max(nu_t)/nu < 1`; **illustrative** otherwise. The `Re <= 200` gate cites Williamson (1996) in the docstring. — Rung H clause 1, **14 points straddling both boundaries from both sides**, plus the D-091 discriminator: deleting *either* gate changes a verdict.
- [x] Before a run, `band_for` returns the band the plan **expects**, from `Re` alone; after a run it returns the band the run **earned**, from the measured `nu_t`. When they differ, the earned band wins and `Result.warnings` says so. A plan that expected `quantitative` and earned `qualitative` is a finding, not a footnote. — measured on D-038's own case: expected `qualitative`, **earned `illustrative`** at `max(nu_t)/nu` 3.797e4, and the warning naming both is printed.
- [x] **Constraint 18, machine-checked:** for every case outside the quantitative band, `Result` emits **no unqualified `Cd`** — asserted by inspecting the object and the rendered summary, not by reading the prose. A test tries to obtain a bare `Cd` from an `illustrative` result and fails if it succeeds. — `flow.report.Result.__post_init__` is the single gate; `validate/fidelity.py::check_constraint_18` checks six routes out of the object and `tests/test_fidelity.py` holds the bare-`Cd` test.
- [x] `flow/autoconfig.py` turns the closure **on** only when the plan needs it to satisfy the `TAU_FLOOR` (**D-059**), and the plan records `cs_smag` and *why* — printed by `--explain`. A case that fits under BGK is still run under BGK, bitwise as Phase 1 ran it. — **D-093**; Rung F re-run green on both backends, worst |diff| **0.000e+00**.
- [x] **Constraint 13 holds:** `cs_smag` never appears in a public `flow/` signature. It is a planned, printed quantity like `tau`, not an input. — an AST scan in `tests/test_smagorinsky.py` asserts no function in `flow/` takes it, and `cs_smag` / `cs` are now in `tests/test_flow_package.py`'s `LATTICE_NAMES`.
- [x] **Constraint 16 holds:** a run that engaged the closure says so in the printed summary, the report and the video metadata, alongside its band. — `substituted=True` with its own sentence, a `closure ON` line, `as_dict()`, and `fidelity=…; closure=on` in the container.
- [x] **D-038's own case runs.** `--fluid air --speed "20 m/s" --size "1.5 m"` completes, reports `illustrative`, prints no `Cd`, and says in the user's units what it is and is not showing. The Phase 1 refusal is superseded for this case and the supersession is recorded as a decision. — Rung H clause 4 runs the literal command: **exit 0**, `illustrative`, no `Cd`. Supersession is **D-093**.
- [x] Refusals that remain refusals still name a working fix (**constraint 14**); `myenv/Scripts/python.exe -m validate.refusals` re-run and prints **PASS**. — Rung D **PASS**, all eleven physics checks `[ok]`, `Monitor` cost **1.96%** (limit 2%) at 3201/3201 MHz on mains.
- [x] `myenv/Scripts/python.exe -m validate.fidelity` prints **PASS** over a Re sweep spanning all three bands. — **PASS on both backends**: quantitative Re 99.6 `Cd` **1.4030**, qualitative Re 159.4 `max(nu_t)/nu` **0.6906**/**0.6886**, illustrative Re 1.979e6 `max(nu_t)/nu` **3.4e4**.
- [x] `myenv/Scripts/python.exe -m validate.autoconfig` and `-m validate.minute --backend warp` re-run and print **PASS** with their published digits. — B (warp) **24/24, 0 failures**, worst Re error **0.0000%**, accuracy **1.0%**; E **48.2 s** (limit 60), `Cd` **1.4040**, `St` **0.1672**.
- [x] `pytest` green. — **894 passed, 2 skipped**; +67 over session 26's 827.

### Constraints that bite here

- **Constraint 18** (new) — no unqualified quantitative claim outside the validated band. This task is where it becomes real.
- **Constraint 5** — *"A wrong sim that looks plausible is the main failure mode of this project."* This task exists because the closure makes that failure mode reachable on purpose.
- **Constraints 13, 14, 16** — all three already exist and all three bite here at once.

### Notes

The pressure valve from `DOCS/PLAN3.md` applies to this task specifically: **if the bands cannot be
made falsifiable, the closure ships stability-only** — it stabilises the run so the user sees a
picture, and the tool declines to report `Cd` at all outside the quantitative band. That is a worse
product and an honest one, and it is the correct fallback. Widening a band to make a number reportable
is the one thing this task must not do.

`sentence(band)` is prose and prose is not what the rung tests (**D-047**'s posture). What is tested is
the verdict and the absence of the number.

---

## T205 — Packaging: `pyproject.toml`, the `fengdong` distribution → Rung I → M11

**Status:** `not_started`

### Goal

`pip install fengdong` works into a fresh virtual environment with no repository on the path. The
distribution is the deliverable; nothing about the simulation changes.

### Reads / depends on

- `DOCS/IDEA4.md` § Scope, § Performance budget (the install), § Validation ladder Rung I
- `DOCS/STATE3.md` § Environment (the dependency list this must reproduce exactly)
- Tasks: none — deliberately independent of T201–T204 (`DOCS/PLAN3.md` § Why this order, 4)

### Inputs / outputs

**In:** the working tree.
**Out:** `pyproject.toml` (PEP 621, setuptools or hatchling); packages `lbm`, `flow`, `fengdong`;
console entry point `fengdong = "fengdong.__main__:main"`; a built wheel and sdist;
`validate/install.py` (Rung I harness); a `fengdong/` package skeleton whose `main` prints a version
and exits, so the entry point is real before the app exists.

### Acceptance criteria

- [ ] `pyproject.toml` declares name `fengdong`, and **every runtime dependency matches a row in `DOCS/STATE3.md` § Environment** — no dependency appears in the package that was not installed and recorded in a session. A test asserts the two lists agree.
- [ ] `python -m build` produces a wheel and an sdist; the wheel contains `lbm`, `flow` and `fengdong` and **no** `validate`, `tests`, `DOCS`, `myenv`, `outputs`, `Navier-Fluid-Equation` or `scripts`.
- [ ] `validate/install.py` builds the wheel, creates a **fresh venv**, installs the wheel into it, and runs `fengdong --version` plus a headless smoke of the app's model layer — **with no repository directory on `sys.path`**, asserted inside the child process, not assumed.
- [ ] The install-to-first-answer elapsed time is **printed**, and under **60 s** on a warm pip cache (**D-035** conditions quoted).
- [ ] Optional extras are declared for what is genuinely optional: the Warp backend (`fengdong[gpu]`) and recording (`fengdong[video]`). The base install runs the NumPy backend and the app.
- [ ] `myenv/Scripts/python.exe -m validate.install` prints **PASS**.
- [ ] The repository still works uninstalled: every existing command in `CLAUDE.md` § Commands runs unchanged from the tree.
- [ ] `pytest` green.

### Constraints that bite here

- **Constraint 15 and constraint 17** — the package layout must not create an import path that violates either. A test asserts the direction inside the *installed* package, not only in the tree.
- **`CLAUDE.md` § Commands** — adding a dependency means `pip install` **and** a row in `DOCS/STATE3.md` § Environment, in the same session.

### Notes

Name checked in session 23: `fengdong` is free on PyPI, `flow` is taken — which is why the
distribution name and the import name differ, and that difference is deliberate rather than a
workaround. Nothing is uploaded to PyPI in this task; Rung I installs from a locally built wheel.
Publishing is a separate decision and needs an account, a licence file and a considered first version
number.

`Navier-Fluid-Equation/` is prior work and is excluded from the distribution explicitly. `myenv/` is
gitignored and must also be excluded — a wheel that ships a virtual environment is a wheel nobody
should install.

---

## T206 — `fengdong/widgets.py` — the closed widget set

**Status:** `not_started`

### Goal

Five widgets, tested without a screen. The set is closed at the start of the task and stays closed.

### Reads / depends on

- `DOCS/IDEA4.md` § What Phase 2 is, concretely; `DOCS/PLAN3.md` § Risks (the widget row)
- `lbm/render.py` (how frames already become pixels — constraint 10)
- Tasks: T205

### Inputs / outputs

**In:** pygame surfaces and events.
**Out:** `fengdong/widgets.py` — `Label`, `TextField`, `Dropdown`, `Button`, `DropTarget`, and a
`Panel` that lays them out in a column. Each takes a rect, draws to a surface, and consumes an event
list returning whether it changed.

### Acceptance criteria

- [ ] Exactly those five widgets plus `Panel`. **No layout engine, no theming, no animation, no focus chain beyond tab order.** A sixth widget is `/new-task`.
- [ ] Every widget is **testable headless**: constructed and driven with synthesised `pygame.event` objects against an off-screen `Surface`, with no window opened. `pytest` runs them under `SDL_VIDEODRIVER=dummy` and the test asserts no display is initialised.
- [ ] `TextField` validates through `flow.quantity.parse` and shows the parse error **in the user's words** when it fails — the same message the CLI prints, obtained from the same code path, not re-worded (**constraint 14**'s posture).
- [ ] `Dropdown` for fluids is populated from `flow.fluids.FLUIDS` at construction; adding a fluid to the library adds it to the widget with no edit here, asserted by test.
- [ ] `DropTarget` consumes `pygame.DROPFILE` and reports the path; a test synthesises the event rather than requiring a human to drag anything.
- [ ] **Constraint 13:** no widget accepts or displays a lattice quantity. A test scans the module for the vocabulary the Phase 1 scan already forbids in `flow/`.
- [ ] **Constraint 17:** `fengdong/` imports `flow/`; `flow/` never imports `fengdong/`. A test asserts it, in the same shape as the existing constraint-15 test.
- [ ] `pytest` green, new tests counted.

### Constraints that bite here

- **Constraint 17** (new), **constraint 13**, **constraint 10** (`fengdong/` colours nothing — the widgets draw chrome; fields are `lbm.render`'s job).

### Notes

The fall-back if the widget layer starts eating the session is named in `DOCS/PLAN3.md`: a native file
dialog plus keyboard entry, which needs no widgets at all and still satisfies "drop a picture on it"
via `DropTarget` alone. `DropTarget` is therefore the one widget that is not negotiable.

---

## T207 — `fengdong/app.py` — window, drop target, setup panel

**Status:** `not_started`

### Goal

A window opens, a picture can be dropped on it, three physical numbers can be typed, and the plan
`flow` would run is shown before anything runs. No simulation yet.

### Reads / depends on

- `DOCS/IDEA4.md` § The five things Phase 2 must get right (3)
- `flow/case.py` (`Case.from_image`, `explain()` — **D-067**, **D-068**), `flow/cli.py` (**D-073**, the flag semantics this must not contradict)
- Tasks: T206

### Inputs / outputs

**In:** a dropped file path, a fluid name, a speed string, a size string, a quality level.
**Out:** `fengdong/app.py::App`, `fengdong/__main__.py::main`; the setup panel; the plan preview.

### Acceptance criteria

- [ ] `fengdong` opens a window titled **FengDong** with a visible drop target, and dropping a PNG on it loads and previews the mask that `flow.prepare` produced — the repaired one, with its verdict shown (**D-065**, **D-066**).
- [ ] Fluid, speed, size and quality are entered through the T206 widgets, and a bad entry shows the parse error without crashing the window.
- [ ] **The plan preview is `Case.explain()`'s content**, obtained from `flow.Case`, not recomputed: grid, `tau`, timestep, run length, expected fidelity band, and why each. Constraint 17 asserted — the app computes nothing.
- [ ] A refused case shows the refusal and its suggestions, and the suggestion the app would **act on** is the one it shows. *(This is queued issue `2fd69b874c32` — `Case.explain()` prints a different list than `Case.nearest()` acts on. Fix it here or carry it explicitly; do not reproduce the mismatch in a second surface.)*
- [ ] The window is resizable and the panel survives it; nothing is positioned by hard-coded pixel counts that break at another size.
- [ ] Closing the window exits cleanly with no pygame resource warnings, asserted headless.
- [ ] Headless test coverage of `App`'s state machine — file dropped, fields edited, plan computed — with no window opened. The manual gate (a human opens it and drops a file) is **recorded in `DOCS/STATE3.md` with what was seen**, per this project's habit of running things rather than reading them.
- [ ] `pytest` green.

### Constraints that bite here

- **Constraint 17** — the app is a view. If a solver parameter is computed here, the task has failed regardless of what the window looks like.
- **Constraint 13** — the setup panel shows a picture, a fluid, a speed and a size. Nothing else is an input.
- **Constraint 14** — refusals name a fix, and the app shows the fix it would run.

### Notes

`--explain` in the CLI exits 0 and runs nothing (**D-072**'s neighbourhood). The window's plan preview
is the same thing with a button next to it, and that symmetry is deliberate: one code path, two
surfaces.

---

## T208 — Live view, numbers panel, save, refusal UI

**Status:** `not_started`

### Goal

Press the button and watch it. The sim runs, the vorticity streams into the window, the numbers
update, and the result can be saved — without the display ever costing a simulation step.

### Reads / depends on

- `DOCS/IDEA4.md` § The five things Phase 2 must get right (4), § Performance budget (the app)
- `lbm/runner.py` (the ring buffer), `lbm/render.py`, `lbm/record.py` (**D-039**), `flow/report.py`, `flow/fidelity.py` (T204)
- `old-Docs/STATE1.md` **D-039**; `DOCS/STATE2.md` **D-071**, **D-073**
- Tasks: T207, T204

### Inputs / outputs

**In:** a runnable `Case`.
**Out:** the live view, the numbers panel, save actions, the in-window refusal path.

### Acceptance criteria

- [ ] The sim runs on a **worker thread**; the window consumes frames from the **existing ring buffer** as a fourth sink. `steps_per_frame` still comes from `flow.autoconfig.plan` (constraint 7).
- [ ] **Zero simulation steps are dropped, ever.** Display frames may drop; the app **counts them and shows the count** (constraint 8, **D-039**'s posture). A test drives a deliberately slow consumer and asserts the step count is unaffected.
- [ ] The view draws **vorticity, diverging colormap, fixed symmetric limits**, through `lbm.render` (constraints 9 and 10). `fengdong/` colours nothing — asserted by test.
- [ ] The numbers panel shows `Cd`, `Cl`, `St`, peak `|u|` against the 0.1 ceiling, convergence, elapsed — **and the fidelity band**, with `Cd` withheld or qualified exactly as `Result` does it (**constraint 18**). The panel cannot show a number `Result` declines to give.
- [ ] Pause, resume and restart work; restart from a checkpoint reproduces bit-identically within the backend (**constraint 11**), asserted by test rather than by eye.
- [ ] Save writes the MP4, the plot and the summary through `flow.report.Result.save` — no second writer. The video metadata carries `substituted` and the band (**constraint 16**).
- [ ] The **30 fps at `quality="balanced"` on warp** budget is measured and printed, with **D-035** conditions beside it.
- [ ] A run that diverges is caught by the existing `Monitor` and shown in the window with its cause and fix (**D-061**), not as a frozen picture. ***And `Monitor` is finally timed on warp*** — the oldest open thread in the product layer, carried since session 18.
- [ ] `pytest` green.

### Constraints that bite here

- **Constraints 7, 8, 9, 10** — all four are Phase 0 constraints that a GUI is the natural place to break. None of them bends.
- **Constraint 18** — the panel is a second surface for the same claim; it must not be a laxer one.
- **Constraint 11** — restart is bit-identical, and a button that restarts is a claim about that.

### Notes

`Monitor` on warp being unmeasured has been carried through five sessions (18 → 22) and is called out
in `DOCS/STATE2.md` § Session 22 as *"worth a `/new-task` in Phase 2 rather than a silent carry"*. It
is folded into this task's criteria instead, because this is the first task where the probe runs on
the GPU in front of a user.

---

## T209 — The drop: end to end, timed → Rung J → M12

**Status:** `not_started`

### Goal

Rung 3's published bands, reached by dropping a picture on a window. The phase's claim, measured.

### Reads / depends on

- `DOCS/IDEA4.md` § Validation ladder Rung J, § Goal
- `validate/minute.py` (Rung E — the model for a timed end-to-end rung, including `psutil` process-start timing and `machine_state()` collected *after* the run)
- `validate/cylinder.py` (the bands, **imported** and never copied)
- Tasks: T208

### Inputs / outputs

**In:** `tests/data/shapes/disc.png` — the committed corpus disc Rung E already uses, so no new binary.
**Out:** `validate/drop.py` printing PASS/FAIL.

### Acceptance criteria

- [ ] The rung synthesises a `pygame.DROPFILE` event carrying the committed PNG and drives the real `App` — not a bypass that calls `flow.Case` directly. If the app's own path is not exercised, the rung is not testing the phase.
- [ ] `Cd` lands in **1.25–1.45** and `St` in **0.155–0.175**, the bands **imported** from `validate/cylinder.py` so this rung cannot drift from Rung 3's numbers by a typo.
- [ ] The band reported is **quantitative**, and the closure is **off** for this case — the product's headline path is still the validated one, and the closure exists for the cases beyond it.
- [ ] Elapsed wall clock is printed, taken from process start via `psutil` so imports and the pygame context are inside the number.
- [ ] `machine_state()` is collected **after** the run and printed: CPU clock of max, power state, GPU name (**D-035**). No absolute timing without it.
- [ ] `myenv/Scripts/python.exe -m validate.drop` prints **PASS** on mains.
- [ ] **M12 gate run in full**, and **all fourteen rungs re-run** — R1–R4, A–E, F–J — with their published digits, on both backends where both apply.
- [ ] `README.md` § Quickstart leads with `pip install fengdong` and the window; § Current state and the ladder table carry all fourteen rungs.
- [ ] `CLAUDE.md` § Commands, § Module map and § Current state updated for a finished Phase 2.
- [ ] `pytest` green.

### Constraints that bite here

- **Constraint 5** — fourteen rungs, ordered, all of them a gate.
- **Constraint 3** — peak `|u|` printed against its ceiling, as every timed rung does.
- **D-035** — the machine's power state has already invalidated one gate run (session 22, 71.9 s on battery vs 49.5 s on mains, identical physics). Check the mains before quoting a number.

### Notes

Rung E's own structure is the model and should be reused rather than reinvented: bands imported from
the rung that owns them, run length derived from the report's constants rather than chosen, wall clock
from process start, machine state after. Rung J differs in exactly one way — the entry point is a
window event instead of a function call — and that difference is the whole point.
