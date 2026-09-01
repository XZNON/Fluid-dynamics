# STATE3.md — live project state, Phase 2

**Read this first, every session, in full.** Updated by `/checkpoint` at the end of every session.
Never rewrite or condense the session log — append only.

**Phase 0's state file is `old-Docs/STATE1.md` and Phase 1's is `DOCS/STATE2.md`; both are frozen**
(**D-041**, **D-084**). Their § Decisions (D-005 … D-040 and D-041 … D-079) remain in force and are
cited by number throughout this file; their session logs are history and are never edited. Decision
numbering continues here at **D-080**.

---

## Snapshot

| Field | Value |
|---|---|
| **Phase** | **Phase 2 — live.** FengDong: the Smagorinsky closure, the fidelity bands, and the desktop application (`DOCS/IDEA4.md`) |
| **Current task** | **T202** — the closure on the Warp backend |
| **Task status** | `not_started` |
| **Completed tasks** | Phase 0: T001 … T011, all eleven. Phase 1: T101 … T110, all ten. Phase 2: **T201** |
| **Milestone reached** | **M8** (2026-08-27, session 22) — the last of Phase 1. Phase 2's are **M9** … **M12** and none is reached |
| **Phase 0 rung status** | R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **re-run session 24 on numpy, every published digit unmoved**: R1 L2 **0.3650%**, R2 **0.75%** / **0.21 cells**, R3 St **0.1731** Cd **1.4031**, R4 square Cd **1.5279** and polygon Cd **1.4276**. The warp column is session 22's (R1 L2 0.3649%) and was not re-measured — **T202 re-runs all nine on both backends** and is where it is re-confirmed |
| **Phase 1 rung status** | A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 — **all re-run session 24**. A: worst kernel **5.960e-08**, whole step **9.611e-06**, checkpoint **8.196e-06**, restart bit-identical — every digit unmoved. B: sweep **24/24 on both**, accuracy warp **4.3%** (was 3.5%) and numpy **17.5%** (was 15.2%), both far inside the 25% limit. C: **15/15** in 30.8 s (was 16.2 s). D: caught before `nan` **3/3**, Monitor cost **+1.02%** (was −0.69%, limit 2%). E: warp **55.7 s** (was 49.5 s, limit 60), Cd **1.4040**, St **0.1672** — both physics digits exact. **Every figure that moved is wall-clock-derived**; see the session-24 log for the clock each was measured at |
| **Phase 2 rung status** | **F 🟨** · G ⬜ · H ⬜ · I ⬜ · J ⬜ — **F is green on numpy and not yet attempted on warp**, which is T202 and is why it is 🟨 rather than 🟩. Measured: `cs_smag=0` **bitwise** Phase 1 on both the fused and unfused paths after 1000 steps of Rung 3's case (`array_equal`, worst |diff| 0.000e+00); Rung 3's full case at `Cs = 0.17` printing Cd **1.4143** and St **0.1719**, inside the unwidened published bands; `max(nu_t)/nu` **0.191** on that wake |
| **Provenance of the rung rows above** | **Session 24's own measurements**, except the warp column of R1–R4, which is still session 22's and is labelled as such above. Rung B could not be run as a single process on this machine (see the session-24 log) and its numpy half was driven section by section through the rung's own `check_accuracy` and `run_case`; all 24 cases completed with 0 failures. **T202 re-runs all nine on both backends** |
| **Last updated** | 2026-09-02 — session 24 (**T201 done**: the Smagorinsky closure in `lbm/core.py` and the NumPy backend, `lbm/probe.py::eddy_viscosity`, `validate/les.py`. **Rung F green on numpy**; all nine existing rungs re-run with no physics digit moved. **D-085** … **D-087**. `pytest` **800 passed, 1 skipped** (317.1 s), 28 of them new). Previously: 2026-09-01 — session 23 (**Phase 2 planned**: `DOCS/IDEA4.md`, `DOCS/PLAN3.md`, `DOCS/TASKS3.md`, `DOCS/STATE3.md` written; `DOCS/STATE2.md` frozen; **D-080** chooses the phase against XLB and 3D on measured evidence; **D-081** … **D-084** rewrite constraint 1, add constraints 17–20, name the app package and price the document move. No code written. `pytest` **772 passed, 1 skipped**) |

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

**None.** Phase 1 closed with an empty § Blockers and Phase 2 opens the same way.

Two entries stay in the local issue queue and neither is a blocker:

- **`2fd69b874c32`** — `Case.explain()` prints a different suggestion list than `Case.nearest()` acts
  on. A real T108 defect; not user-facing today because `flow/cli.py` prints the list it will actually
  execute. **Folded into T207's acceptance criteria**, because the app would be a second surface for
  the same mismatch.
- **`495777c58269`** — `.gitignore` drops `*/__init__.py` and `tools/`. Open since session 16. It will
  bite T205, where a wheel has to contain every `__init__.py` it ships; fix it there or explicitly
  carry it.

A third entry was queued in session 24 and is likewise not a blocker:

- **`d5b27e51fcdc`** — `validate/refusals.py` printed `on battery` in its D-035 conditions line while
  `Win32_Battery.BatteryStatus` read **2 (mains)** for the whole session and `validate/minute.py`
  printed `power: mains` for the same machine minutes later. One of the two power probes is wrong,
  and **D-035** requires the power state beside every absolute timing figure, so a probe that
  misreports it makes those figures unquotable. The fix is to make the two rungs share one
  implementation. Rung D passes either way — the discrepancy is in the *label*, not in the check.

One thread carried since session 18 and now scheduled rather than carried: **`Monitor` on `warp` has
never been timed.** Rung D runs on `numpy` by design, so the divergence probe's device-side cost is
unmeasured. It is an acceptance criterion of **T208**.

**On this machine, a process that runs longer than roughly ten minutes under the agent's own tooling
is liable to be killed** (session 24 lost several Rung B attempts that way, one of them after 98
CPU-minutes, and an orphaned survivor then competed for CPU with its own replacement). This is an
environment property, not a project defect, and it is recorded here because **Rung B (~23 min) and
Rung 4 (~36 min) are both past that line** and the next session will hit it. The workaround session
24 used is in its log.

## Open questions

- **Q-201** — Does bitwise degeneracy on the Warp backend survive multiplying the closure term by
  zero, or does `cs_smag = 0` need a separately compiled kernel? **D-053** documents that the GPU
  contracts `x * a + b` into one rounding where NumPy does two, so a term that is algebraically zero
  is not automatically bitwise inert. **Still open — T202 answers it by measurement.** The note in
  `DOCS/TASKS3.md` § T202 is explicit that the branch is the fix and the tolerance is not.
  **What T201 changed about it (session 24):** the NumPy side did not wait to find out. **D-086**
  makes `cs_smag == 0.0` an explicit early-return branch in `lbm/core.py::collide` and a scalar-vs-field
  branch in `::collide_stream`, so nothing algebraically zero is ever multiplied in on *either*
  backend's reference path. T202 inherits a branch to port rather than a tolerance to argue about,
  and the question narrows to whether Warp needs *two compiled kernels* or one guarded branch
  suffices.
- **Q-202** — What is `<nu_t>` on a *resolved* 2D Taylor–Green at `Cs = 0.17`, as a fraction of `nu`?
  Expected small. If the model fires hard on a smooth flow, that is a finding about the
  implementation rather than about turbulence, and it belongs in § Decisions with its measurement.
  **T203 answers it.**
- **Q-203** — Can the fidelity bands be made falsifiable enough to report a qualified `Cd` outside the
  quantitative band, or does the closure ship **stability-only** (a picture, and no `Cd` at all
  outside the validated band)? This is the phase's central product question and its pressure valve is
  already written into `DOCS/PLAN3.md` § Risks. **T204 answers it.**
- **Q-204** — Does `fengdong` publish to PyPI inside this phase, or does Rung I's locally built wheel
  close it? Publishing needs an account, a `LICENCE` file and a considered first version number, and
  none of the three is a packaging detail. **T205 raises it; the user decides.**

## Environment

Project venv: `myenv/` (gitignored). Python 3.11.15. **Unchanged in session 24 as well —
nothing was installed into `myenv` by T201, which was the expectation `PROMPTS/024` set.**

| Package | Version | Added by |
|---|---|---|
| numpy | 2.4.6 | pre-existing |
| matplotlib | 3.11.1 | pre-existing |
| pillow | 12.3.0 | pre-existing |
| pytest | 9.1.1 | T001 (session 1) |
| pygame | 2.6.1 | T007 (session 7) |
| imageio | 2.37.4 | T011 (session 11) |
| imageio-ffmpeg | 0.6.0 | T011 (session 11) |
| psutil | 7.2.2 | T011 (session 11) |
| warp-lang | 1.16.0 | T102 (session 14) — CUDA Toolkit **12.9**, Driver **13.1**; device `cuda:0` = NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB, sm_86, mempool enabled); `nvidia-smi` driver **592.82** |

**Expected in Phase 2:** a build front-end (`build`, and `setuptools` or `hatchling`) in **T205**, and
nothing else. **No XLB** (**D-080**), no Taichi, no Qt, no web framework, no `tkinterdnd2` — the app
is pygame-only (**D-083**), and pygame is already here. `pygame.DROPFILE` and `pygame.DROPBEGIN` were
both confirmed present on pygame 2.6.1 / SDL 2.28.4 in session 23, so drag-and-drop costs no
dependency.

Install with `myenv/Scripts/pip.exe install <pkg>` and **add a row above in the same session.**
T205's `pyproject.toml` must declare exactly the runtime rows in this table and no others — that
agreement is one of its acceptance criteria.

**A scratch venv exists outside the repo** at the session-23 scratchpad, holding `xlb` 0.3.1 and
warp-lang 1.11.0, built solely for **D-080**'s measurement. It is not `myenv`, it is not on any
import path the project uses, and nothing depends on it. Delete it freely.

## Performance baseline

Phase 0's and Phase 1's measured tables stand and are not re-measured by this session.

Phase 0 (`old-Docs/STATE1.md` § Performance baseline): **696.7 / 161.7 / 16.8** steps/s at
40k / 160k / 1M cells against floors of 400 / 120 / 15.

Phase 1 (`DOCS/STATE2.md` § Performance baseline, filled in by T103, anchors added by T110). Conditions
per **D-035**: AMD Ryzen 7 5800H at `Win32_Processor.CurrentClockSpeed` **3201 of 3201 MHz**, on
**mains**; NVIDIA GeForce RTX 3050 Laptop GPU, driver **592.82**, CUDA Toolkit 12.9 / Driver 13.1.

| Grid | Cells | NumPy measured | **Warp measured** | Speedup |
|---|---|---|---|---|
| 400×100 | 40k | 775.1 | **4155.0** | 5× |
| 800×200 | 160k | 185.6 | **3560.4** | 19× |
| 800×500 | 400k | 76.5 | **1403.9** | 18× |
| 2000×500 | 1M | 23.0 | **757.3** | 33× |
| 2000×1000 | 2M | 8.3 | **441.0** | 53× |

Device memory at 2M cells: **391 MiB in 13 `Sim`-owned arrays**, 2882 MiB of 4096 free.

**Phase 2's budget** (`DOCS/IDEA4.md` § Performance budget), **not yet filled in**:

| Grid | Cells | BGK (warp) | LES floor — within 25% | Measured |
|---|---|---|---|---|
| 400×100 | 40k | 4155.0 | ≥ 3116 | — (T202) |
| 2000×500 | 1M | 757.3 | ≥ 568 | — (T202) |
| 2000×1000 | 2M | 441.0 | ≥ 331 | — (T202) |

Plus: the window sustains **30 fps of display** at `quality="balanced"` on warp with **zero dropped
simulation steps** (T208), and `pip install` to first window is under **60 s** on a warm cache (T205).

**One indicative external number, carried from session 23's XLB probe and explicitly not a D-035
measurement** (machine read `CurrentClockSpeed` 1990 of 3201 MHz; XLB's step carried one bounce-back
BC against our Zou–He inlet plus convective outlet): XLB's own Warp D2Q9 at 100k cells ran **4782.7**
steps/s with BGK and **3958.2** with `SmagorinskyLESBGK` — the closure costing **17%** there. Useful
only as a sanity check that our own 25% floor is generous, never as a target.

**D-035 governs every number here**: alternating-round A/B, best round per variant, and no absolute
steps/s figure without the CPU clock, the power state and the GPU name beside it.

## Decisions

Anything chosen that wasn't already specified in `DOCS/IDEA4.md`, `idea.md` or `CLAUDE.md`. Append;
never edit a past entry — supersede it with a new one that says so. Numbering continues from
`old-Docs/STATE1.md` (D-005 … D-040) and `DOCS/STATE2.md` (D-041 … D-079), both of which remain in
force.

| ID | Date | Decision | Why |
|---|---|---|---|
| D-080 | 2026-09-01 | **Phase 2 is the Smagorinsky closure, the fidelity bands it makes necessary, and the `fengdong` pygame desktop application, shipped as one `pip install`. `idea.md` § Roadmap's Phase 3 (swap in XLB) and Phase 4 (3D + STL) are both deferred past it.** This is a deliberate, recorded deviation from `idea.md`'s ordering, the second such (precedent: **D-043**), and it un-defers two items `DOCS/IDEA3.md` § Deliberately deferred listed by name — a UI (**D-044**) and packaging/distribution. | User's call with the alternatives measured rather than argued. **XLB, installed and run in this session, not read about**: `pip install "xlb[warp]"` succeeds on Windows/py3.11 (xlb 0.3.1) and its Warp D2Q9 path compiles and runs on this RTX 3050 — but (a) xlb 0.3.1 imports `ScopedTimer` from `warp.utils`, removed between warp-lang 1.11 and 1.14, so it imports at **1.11.0** and fails at **1.14.0** and our **1.16.0**, meaning adoption pins warp five minors back *for our own validated backend too* and re-opens Rung A; (b) `IncompressibleNavierStokesStepper` owns grid, boundaries and the whole timestep, where **D-054**'s seam is per-kernel precisely so **D-053** can bisect a parity failure; (c) its only 2D-relevant gift is the LES closure, which is ~20 lines in our own `collide` with no seam change, while its real gifts (D3Q19/D3Q27, KBC, trimesh voxeliser) are all 3D. **3D refused by arithmetic**: our own floor of 30 cells across the body at a 24 D span (**D-059**, **D-075**) is 720³ = **28.4 GB per buffer** in D3Q19 `float32`, four buffers needed; 12 D span is still 3.5 GB/buffer on a 4 GB card; what fits (~192³) gives **8–16 cells across the body**, a resolution the 2D product already refuses. Against that, the closure removes the wall every real user hits first (**D-038**, **D-074**) and the app finally makes `idea.md`'s success sentence literal. |
| D-081 | 2026-09-01 | **Constraint 1 is rewritten to permit exactly one turbulence closure: a Smagorinsky eddy-viscosity model that modifies the relaxation rate and nothing else.** Still forbidden by name: MRT, cumulant, KBC, curved or interpolated boundaries, wall models, dynamic (Germano) `Cs`. The implementation may still move to another backend; the *base* arithmetic — D2Q9, BGK, bounce-back — may still not change, and with the closure off it is required to be **bitwise** what Phase 1 shipped. Supersedes **D-046**'s rewrite of constraint 1. | The prompt for this session named it: constraint 1 says the arithmetic may not change and an added closure is exactly the thing that tests where that line is, so it gets decided in the spec rather than discovered in a task. The line drawn: the closure is *additive and switchable*, so Phase 1's arithmetic is not modified but extended, and constraint 19 makes "extended" a testable claim rather than a reassuring word. Everything still forbidden is forbidden because it would replace the collision or the boundary rather than add a term to it. |
| D-082 | 2026-09-01 | **The closure is a stability device, not a fidelity device, and every `Result` says which.** Three bands: **quantitative** (`Re <= 200` and `max(nu_t)/nu < 0.1`), **qualitative** (`max(nu_t)/nu < 1`), **illustrative** (otherwise). The upper boundaries are **measured per run from the eddy viscosity the run generated**, not read off a Reynolds-number table. Outside the quantitative band no unqualified `Cd` is emitted — **constraint 18**, machine-checked by Rung H. | Two physical facts and one project fact. (1) The cylinder wake becomes three-dimensional at **Re ≈ 190** (Williamson 1996, mode-A), which is why Rung 3 sits at Re 100; above it a 2D run is wrong about the flow, not the numerics. (2) Smagorinsky descends from Kolmogorov's forward cascade and 2D turbulence cascades energy the other way (Kraichnan 1967), so a 2D closure is a stabiliser and not a model of what is actually happening; the drag crisis near Re 3e5 cannot appear in 2D at any resolution. (3) Constraint 5 names *"a wrong sim that looks plausible"* as this project's main failure mode, and the closure makes it reachable **on purpose** — so the band is not documentation, it is the safety interlock. `max(nu_t)/nu < 1` is chosen as the outer boundary because it is the point where the model supplies more viscosity than the fluid does, which is a statement about this run that a test can evaluate. |
| D-083 | 2026-09-01 | **The application is a new top-level package `fengdong/`, built on pygame and nothing else, and the distribution is named `fengdong`.** Import layering: `fengdong/` may import `flow/`, `flow/` may import `lbm/`, and neither may import upward — **constraint 17**, asserted by a test in the shape of the existing constraint-15 test. The widget set is closed at five (`Label`, `TextField`, `Dropdown`, `Button`, `DropTarget`) plus `Panel`. | 风洞 (fēngdòng) is Chinese for *wind tunnel*; the user chose it. `flow` is **taken** on PyPI and `fengdong` is **free** (both checked this session), so the distribution and import names differ by necessity rather than by preference, and `fengdong` is then also the command and the title bar. pygame-only because it is **already a dependency** (2.6.1), already the sink `lbm/render.py` feeds, and `pygame.DROPFILE` was confirmed present this session — so *"drags in a picture"* costs no new dependency and the validated render path is reused rather than duplicated. PySide6 would have meant a ~150 MB dependency, a second event loop to reconcile with the ring buffer, and frames reaching the screen by a path no rung has validated; tkinter would have needed `tkinterdnd2` for the one feature that is not negotiable. The price accepted is a hand-rolled widget layer, and it is bounded by closing the set on day one. |
| D-084 | 2026-09-01 | **`DOCS/STATE2.md` is frozen in place and Phase 1's documents do not move to `old-Docs/`. Decision numbering continues unbroken at D-080.** The freeze is a header banner plus this rule, not a path change; `DOCS/IDEA3.md`, `DOCS/PLAN2.md` and `DOCS/TASKS2.md` are likewise marked closed where they sit. **Extends D-049 rather than repeating it.** | D-049 moved Phase 0's three session-management documents to `old-Docs/` but **deliberately left `DOCS/IDEA2.md` put**, because ~100 docstrings cited it by path. Priced this session before deciding: `DOCS/IDEA3.md` has **144 citations across 52 files, all 52 in `.py`**; `TASKS2` **120 / 34 files / 37 in `.py`**; `STATE2` **121 / 36 / 16**; `PLAN2` **83 / 30 / 15**. That is ~470 citations and ~120 docstring paths to rewrite for no reader benefit, against D-049's own stated threshold of ~100 for keeping a file put. A second migration would also make `old-Docs/` ambiguous — *which* old? — where a freeze banner is unambiguous and costs nothing. `CLAUDE.md` § Session protocol names the live documents, so there is no question about which file is current. |
| D-085 | 2026-09-02 | **The closure's normalisation is fixed and written down: filter width `Delta = 1` lattice unit, strain norm `|S| = sqrt(2 S_ab S_ab)`, and `Q_ab = sum_i e_ia e_ib (f_i - feq_i)` with no factor of two folded in — giving `tau_eff = 0.5 (tau + sqrt(tau^2 + 18 sqrt(2) Cs^2 |Q| / rho))`. `lbm.core.smagorinsky_tau_eff` is the primitive and `smagorinsky_omega` is its reciprocal, not the other way round.** `SMAG_Q_COEFF = 18 sqrt(2)` is defined once in `lbm/core.py`. | Two separate reasons. **(1) The coefficient is a convention, not a fact.** Session 23 read XLB's 2D closure as a cross-check and it carries **36** in that position — exactly `sqrt(2)` times ours. The difference is entirely the strain-norm convention, so a docstring that does not state which one it took cannot be checked against any paper, and `DOCS/TASKS3.md` § T201 required the algebra be *pinned by a test*. It is pinned twice: against the quadratic written out independently with an `einsum` `Q`, and against a velocity field built **backwards** from a chosen strain rate, which is the only check that pins the filter width and the strain norm *together*. Derivation and the XLB comparison are in the docstring. **(2) The direction matters in `float32`.** `1 / (1 / tau)` is not `tau`, so deriving `tau_eff` back from `omega` would leave `nu_t = cs2 (tau_eff - tau)` a few ulps from zero with the closure off, breaking T201's "`nu_t == 0` exactly" criterion and, with it, constraint 18's ability to say a run generated no eddy viscosity. Computing `tau_eff` first and reciprocating once makes the zero exact. |
| D-086 | 2026-09-02 | **Constraint 19 is implemented as an explicit `cs_smag == 0.0` branch, not as a term that multiplies to zero.** `lbm.core.collide` returns early through Phase 1's three operations verbatim; `collide_stream` selects a `float32` *scalar* factor when off and an `(ny, nx)` *field* when on. | The cheap, absolute version of the guarantee. Even on NumPy the two are not the same: a scalar `float32(1 - 1/tau)` is not required to equal `1 - float32(1/tau)` evaluated elementwise, so a "closure term that happens to be zero" would have been bitwise-*close* rather than bitwise-*equal*, and constraint 19 says equal. **Q-201** then makes the same point far more sharply for the GPU (**D-053**: the device contracts `x * a + b` into one rounding where NumPy does two). `DOCS/TASKS3.md` § T202 already ruled that *the branch is the fix, not the tolerance*; writing the branch on the reference backend first means T202 ports a shape that is known to work rather than discovering it under a failing rung. Measured: `array_equal` after 1000 steps of Rung 3's case, worst |diff| **0.000e+00**, on both the fused and the unfused path. |
| D-087 | 2026-09-02 | **Rung F runs Rung 3's own harness rather than a copy of it, and the frozen Phase 1 collision lives in exactly one place.** `validate/cylinder.py::make_config` and `::run_cylinder` gain a `cs_smag: float = 0.0` parameter that every caller inside that module leaves at zero; `validate/les.py` holds verbatim transcriptions of Phase 1's `collide` / `collide_stream` plus a `Phase1Backend` shim, and `tests/test_smagorinsky.py` **imports** them rather than transcribing them again. | "Bitwise what Phase 1 shipped" needs a Phase 1 to compare against, and after T201 edited `lbm/core.py` there is not one in the tree. A frozen transcription is the only oracle available — so it is marked *do not edit*, and there is **one** of it, because two copies that drift make the rung and the unit tests disagree about what Phase 1 was. Reusing Rung 3's own harness for clause 3 is the same argument in the other direction: a second copy of Rung 3's setup would be a case whose agreement with the real Rung 3 nobody checks, and `cs_smag` threaded through the existing one costs a defaulted parameter. The AST-based test `test_the_closure_is_off_everywhere_it_is_not_being_tested` is what keeps that parameter honest — it checks *syntax*, not text, so no docstring can pass or fail it, and it asserts `flow/` does not mention the closure at all. |

### Constraint fate table (D-081, D-083)

The fate of each of Phase 1's sixteen, decided in session 23 rather than left to rot — the same
exercise **D-046** did for Phase 0's twelve. `CLAUDE.md` § Hard constraints is the authority; this
table is the record of why each reads as it does.

| # | Phase 1 constraint | Phase 2 status |
|---|---|---|
| 1 | D2Q9, BGK, bounce-back; no turbulence model; implementation may move backend | **Rewritten (D-081).** Exactly one closure is permitted and named — Smagorinsky, additive and switchable. Still forbidden: MRT, cumulant, KBC, curved/interpolated boundaries, wall models, dynamic `Cs`. The base arithmetic still may not change, and with the closure **off** it must be bitwise Phase 1's. |
| 2 | `nu = (tau - 0.5)/3`; no `nu` setter that bypasses `tau` | **Permanent**, and it now governs `tau_eff` too: the closure modifies the relaxation time, never a viscosity directly. |
| 3 | Lattice velocity under 0.1, warned at setup | **Permanent.** |
| 4 | Backend owns its layout; `to_host` yields `(9, ny, nx)` `float32`; constants only from `lbm/core.py` | **Permanent.** Noted in passing: XLB returns `(q, x, y, z)`, which is why a swap would owe this contract a transpose (**D-080**). |
| 5 | The validation ladder is ordered and non-negotiable | **Permanent**, extended to Phase 2's five rungs **F–J**. All nine existing rungs stay a gate for every Phase 2 task. |
| 6 | ~~Do not optimise before Rung 3 passes~~ | **Stays retired.** Its replacement stands: no backend optimisation before its parity rung passes — which for the closure is Rung F on that backend. |
| 7 | Simulation and rendering decoupled; `steps_per_frame` computed | **Permanent**, and the app is the first surface that makes it visible. |
| 8 | Never block the sim on the display; drop display frames, never steps | **Permanent.** The app is a *fourth sink on the existing ring buffer*, and it counts the frames it drops. |
| 9 | Draw vorticity; diverging map, fixed symmetric limits | **Permanent.** |
| 10 | One `render()`, three sinks; `flow/` colours nothing | **Permanent**, and now also means `fengdong/` colours nothing. |
| 11 | Restart bit-identical within a backend, printed tolerance across | **Permanent.** The closure adds no state — `tau_eff` is derived — so `f`, `mask`, `step_count` remain the whole checkpoint (**D-022**, **D-050**). |
| 12 | Geometry is one bool `solid`; ≥3 cells thick, ≥8 D downstream, <10% blockage | **Permanent.** |
| 13 | No lattice quantity in any public `flow/` signature | **Permanent**, extended: none in a `fengdong/` widget either. `Cs` is planned and printed, never an input. |
| 14 | Every refusal names a fix, and the fix is machine-checked | **Permanent.** Fewer cases refuse now; the ones that still do are held to the same bar. |
| 15 | `flow/` may import `lbm/`; `lbm/` may never import `flow/` | **Permanent**, and the model for constraint 17. |
| 16 | No silent substitution — every artifact says so | **Permanent**, and strengthened: a run that engaged the closure is a substituted-fidelity run and says so alongside its band. |
| **17** | — | **New (D-083).** `fengdong/` may import `flow/`; `flow/` may never import `fengdong/`. |
| **18** | — | **New (D-082).** No unqualified quantitative claim outside the validated band. |
| **19** | — | **New (D-081).** The closure defaults off, and `Cs = 0` is bitwise Phase 1 on every backend. |
| **20** | — | **New (D-083).** One `pip install`, one command; Rung I proves it off this tree. |

## Session log

*(Appended by `/checkpoint` at the end of every session. Never rewritten, never condensed.)*

### 2026-09-01 — Session 23: Phase 2 planning

**Task:** none — a planning session, no `T2xx` live. **Status: done. Phase 2 is planned and Phase 1
is frozen.** No solver or product code was written, by design (`PROMPTS/023-phase2-planning.md`
§ Scope discipline).

**Read, in the protocol's order:** root `idea.md` and `README.md` in full; `CLAUDE.md`;
`DOCS/STATE2.md` § Snapshot, § Blockers, § Open questions, § Environment, § Performance baseline,
§ Decisions D-041 … D-079, the constraint fate table, and the session-22 entry; `DOCS/PLAN2.md` as a
model; `DOCS/IDEA3.md` § Deliberately deferred; `DOCS/TASKS2.md`'s contract shape; and
`lbm/backends/__init__.py` — the seam an XLB backend would have had to satisfy.

**The question the session existed to answer, answered by measurement rather than by inheriting the
roadmap.** `idea.md` says Phase 3 is the XLB swap. Three things were measured before the user chose:

- **XLB was installed and run on this machine**, in a scratch venv outside the repo.
  `pip install "xlb[warp]"` succeeds on Windows / Python 3.11 — xlb **0.3.1**, jax CPU wheels,
  warp-lang, 43 packages, no build step. Then `import xlb` **failed**: `ImportError: cannot import
  name 'ScopedTimer' from 'warp.utils'`. Bisected across warp versions — fails at **1.16.0** (ours)
  and **1.14.0**, imports at **1.11.0**, where warp itself deprecation-warns the symbol XLB uses.
  With warp pinned to 1.11.0 a D2Q9 case compiled and ran on `cuda:0`: **4782.7** steps/s at 100k
  cells with BGK, **3958.2** with `SmagorinskyLESBGK` — so the closure costs **17%** there, and their
  2D LES path works. `f` came back as `(9, 500, 200, 1)`, i.e. `(q, x, y, z)` against our
  `(9, ny, nx)`. **Not a D-035 measurement**: the machine read `CurrentClockSpeed` **1990 of 3201
  MHz**, and their step carried one bounce-back BC against our Zou–He inlet plus convective outlet.
- **The 3D memory arithmetic**: our own `QUALITY_CELLS["fast"] = 30` cells across the body at a 24 D
  span is 720³ in 3D = **28.4 GB per buffer** in D3Q19 `float32`, four needed; 12 D span is 3.5
  GB/buffer; ~192³ fits and gives 8–16 cells across the body — below what the 2D product refuses.
- **`pygame.DROPFILE` and `DROPBEGIN` confirmed present** on the installed pygame 2.6.1 / SDL 2.28.4,
  so a drag-and-drop desktop app costs no new dependency.
- **PyPI names checked**: `flow` **taken**, `fengdong` / `dropflow` / `windtunnel` all **free**.

**The user's call, on that evidence:** 2D plus a turbulence closure written in our own kernel, plus a
pip-installable desktop application, **pygame only, no website**, named **FengDong** (风洞, wind
tunnel). Recorded as **D-080** and **D-083**.

**Written**

- **`DOCS/IDEA4.md`** — the Phase 2 spec: goal, scope with an explicit out-list, the reasons XLB and
  3D are deferred with their measurements, the pipeline as modules, the five things the phase must get
  right, the five-rung ladder F–J with a known answer each, the performance budget with its
  arithmetic, and § Deliberately deferred.
- **`DOCS/PLAN3.md`** — nine tasks T201 → T209, the dependency graph, the one-task-per-session map
  (sessions 24–32), four milestone gates **M9**–**M12** with literal gate commands, and a risks table
  where every row has a signal and a pressure valve.
- **`DOCS/TASKS3.md`** — a full contract per task: goal, reads/depends-on, inputs and outputs with
  types and shapes, acceptance criteria as checklists, the constraints that bite, and notes.
- **`DOCS/STATE3.md`** — this file. Snapshot, blockers, four open questions Q-201 … Q-204,
  environment, performance baseline, decisions **D-080 … D-084**, and this log.
- **`DOCS/STATE2.md`**, **`DOCS/IDEA3.md`**, **`DOCS/PLAN2.md`**, **`DOCS/TASKS2.md`** — frozen in
  place with header banners (**D-084**).
- **`CLAUDE.md`** — § Session protocol repointed at the Phase 2 documents, § Current state rewritten,
  § Module map extended (`flow/fidelity.py`, `fengdong/`, the five new rungs), § Commands extended,
  `pyproject.toml` added to § Everything else at the root, and the hard-constraint list taken from 16
  to 20. Constraints **2, 5, 10, 13, 15** each gained a clause so the list stays self-contained
  against the fate table rather than deferring to it.
- **`.claude/commands/*.md` and `PROMPTS/templates/session-prompt-template.md`** — repointed at the
  Phase 2 documents, the same housekeeping **D-049** did at the last phase boundary. Without it
  `/checkpoint` would have written into a file this session had just frozen. `/validate` also gained
  the five Phase 2 rungs and `/new-task` now allocates `T2XX`. **Note: `.claude/` is gitignored**
  (`.gitignore:5`) and untracked, so those edits live on this machine only — consistent with how the
  repo already treats it, but worth knowing on a fresh clone.
- **`PROMPTS/024-t201-smagorinsky-closure.md`** — the next session's prompt.

**Decisions made**

- **D-080** — what Phase 2 is, with XLB and 3D rejected on measured evidence.
- **D-081** — constraint 1 rewritten to permit exactly one closure; supersedes D-046's rewrite.
- **D-082** — the closure is a stability device; three fidelity bands, measured per run.
- **D-083** — `fengdong/`, pygame-only, the distribution name, constraint 17's import direction.
- **D-084** — Phase 1's documents freeze in place; the move to `old-Docs/` was priced and rejected.

**Measured**

`myenv/Scripts/python.exe -m pytest` — **772 passed, 1 skipped**, run twice: **308.7 s** at session
start as the baseline confirmation, and **302.6 s** after all document edits. Unchanged, which is the
acceptance criterion for a session that writes no code.

Also verified rather than remembered: the spec's own arithmetic was recomputed —
LES floors `4155.0/757.3/441.0 × 0.75` = **3116 / 568 / 331**; the 3D refusal
`720³ × 19 × 4 B` = **28.37 GB**, `360³` = **3.55 GB**, `192³` = **0.54 GB** per buffer; and D-084's
citation counts **468 total, 120 in `.py`**.

**Not done / deferred**

- **The validation ladder was not re-run, and no rung status in § Snapshot was re-measured this
  session.** Every rung figure above is session 22's, carried forward and labelled as such. Re-running
  it costs ~40 minutes (Rung B alone is ~23) for a session that wrote no executable code, and the
  prompt's own acceptance criterion was `pytest`, which was run twice and matched. **T201 re-runs all
  nine**, and its contract says so.
- **`validate.minute --backend warp` was not re-run**, though the prompt asked for it as a
  starting-point confirmation. The machine read `CurrentClockSpeed` **1990 of 3201 MHz** during the
  session, which is not the clean mains state **D-035** requires for a 60 s gate, so a re-run would
  have produced a number that could not honestly be quoted. M8's gate output stands as session 22
  measured it.
- **The XLB scratch venv** is left in the session scratchpad. Outside the repo, on no import path,
  safe to delete.
- **`DOCS/STATE3.md` § Session log is not empty**, where `PROMPTS/023` asked for it to be. The
  session-23 entry was written directly rather than left for `/checkpoint` to fill, since `/checkpoint`
  re-reads the file first and this is the entry it would have written. Flagged rather than silently
  deviated from.

**Blockers:** none.

**Next:** **T201** — the Smagorinsky closure in `lbm/core.py` and the NumPy backend, with Rung F's
bitwise degeneracy clause written *before* the model does anything. Prompt written to
`PROMPTS/024-t201-smagorinsky-closure.md`.


### 2026-09-02 — Session 24: T201 — the Smagorinsky closure, `lbm/core.py` + the NumPy backend

**Task:** **T201**. **Status: done — every acceptance criterion in `DOCS/TASKS3.md` § T201 was run and
passed.** Phase 2 has its first code and **Rung F is green on numpy**.

**Read, in the protocol's order:** `PROMPTS/024-t201-smagorinsky-closure.md`; `CLAUDE.md`;
`DOCS/STATE3.md` in full; `DOCS/TASKS3.md` § T201 and § T202; `DOCS/IDEA4.md` § The five things
Phase 2 must get right (1) and (2) and § Validation ladder; `DOCS/PLAN3.md` § Why this order,
§ Session map, § Risks; `lbm/core.py`, `lbm/backends/__init__.py`, `lbm/backends/numpy_backend.py`,
`lbm/backends/warp_backend.py`, `lbm/runner.py`, `lbm/probe.py`, `validate/cylinder.py`.

**Done**

- **`lbm/core.py`** — `smagorinsky_tau_eff` and `smagorinsky_omega` (the T201 contract's named entry
  point), `CS_SMAG_LITERATURE = 0.17` and `SMAG_Q_COEFF = 18 sqrt(2)`. `collide` and `collide_stream`
  gained keyword-only `cs_smag: float = 0.0`, `smag_out=` and `smag_work=`. The full derivation, the
  three normalisation choices and the XLB comparison are in the docstring (**D-085**).
- **`lbm/probe.py`** — `eddy_viscosity(f, feq, tau, cs_smag) -> (ny, nx)`: `nu_t = cs2 (tau_eff - tau)`,
  derived through `tau` and never assigned (constraint 2). This is the field **D-082**'s bands and
  Rung G both read.
- **`lbm/runner.py`** — `SimConfig.cs_smag: float = 0.0`; `Sim` allocates `smag_out` `(ny, nx)` and
  `smag_work` `(4, ny, nx)` **only when the closure is on**, and refuses a negative `cs_smag`.
- **The seam** — `Backend.collide` / `collide_stream` carry the keyword; `numpy_backend` delegates to
  `lbm.core` as it always has; `warp_backend` accepts the keyword and **raises
  `NotImplementedError` naming T202** for a non-zero value rather than silently computing plain BGK.
- **`validate/les.py`** — Rung F, three clauses, printing PASS/FAIL, with the frozen Phase 1
  collision and a `Phase1Backend` shim as the oracle (**D-087**).
- **`tests/test_smagorinsky.py`** — 28 tests, one per acceptance criterion plus the invariants.
- **`validate/cylinder.py`** — `make_config` and `run_cylinder` gained `cs_smag: float = 0.0`, left at
  zero by every caller in that module, so Rung F runs Rung 3's own harness (**D-087**).

**Measured**

- **Rung F, full: PASS.** `cs_smag = 0` is `numpy.array_equal` to the frozen Phase 1 kernels after
  **1000 steps of Rung 3's case**, worst |diff| **0.000e+00**, on the fused *and* the unfused path;
  fused and unfused agree bitwise with each other (**D-055** survives). Rung 3's full case at
  `Cs = 0.17` printed **Cd 1.4143**, **St 0.1719** — inside the unwidened published bands — with
  `tau_eff` 0.5378 … 0.5450 and `max(nu_t)/nu` **0.191**.
- **`pytest`: 800 passed, 1 skipped** in 317.1 s (session 23: 772 passed, 1 skipped; +28 new).
- **All nine existing rungs re-run, and no physics digit moved.** R1 L2 **0.3650%** · R2 **0.75%** /
  **0.21 cells** · R3 St **0.1731** Cd **1.4031** · R4 **1.5279** / **1.4276** · A worst kernel
  **5.960e-08**, whole step **9.611e-06**, checkpoint **8.196e-06**, restart bit-identical ·
  B sweep **24/24 on both backends** · C **15/15** · D **3/3** caught before `nan` · E Cd **1.4040**,
  St **0.1672**.
- **Every figure that did move is wall-clock-derived, and the clock it was measured at explains it.**
  The machine sat at `CurrentClockSpeed` **1990 of 3201 MHz on mains** for most of the session and
  recovered to **3201 of 3201** before Rung E. Measured while throttled: Rung C **30.8 s** (was
  16.2 s), Rung D's Monitor cost **+1.02%** (was −0.69%, limit 2%), Rung B accuracy warp **4.3%**
  (was 3.5%) and numpy **17.5%** (was 15.2%), both against a 25% limit. Measured after recovery:
  **Rung E 55.7 s** against its 60 s limit (was 49.5 s), at **3201 of 3201 MHz, on mains**, NVIDIA
  RTX 3050 Laptop GPU, driver **592.82** (**D-035**).

**Decisions made**

- **D-085** — the closure's normalisation, fixed and written down; `tau_eff` is the primitive and
  `omega` its reciprocal, because the other direction cannot make `nu_t` exactly zero in `float32`.
- **D-086** — constraint 19 is an explicit `cs_smag == 0.0` branch, not a zero-valued term. Narrows
  **Q-201** for T202.
- **D-087** — Rung F reuses Rung 3's harness; one frozen Phase 1 transcription, imported by the tests.

**Not done / deferred**

- **`validate/les.py` has no `--backend` flag**, deliberately: `DOCS/TASKS3.md` § T202 names adding it
  as one of that task's acceptance criteria, and the Warp kernels do not exist yet. Rung F is
  therefore **🟨 — green on numpy, unattempted on warp** — and the script says so in its own output.
- **The warp column of R1–R4 was not re-measured.** Those four were re-run on numpy only. T202's
  contract already requires all nine on **both** backends, so it is scheduled rather than skipped.
- **Rung B could not be run as a single process on this machine.** Several attempts were killed
  mid-run by the environment; one survived as an orphan for 98 CPU-minutes and competed for CPU with
  its own replacement until it was found and killed, which is what made the earlier attempts look
  stalled. The warp half then completed normally in one 3m41s foreground run. The **numpy** half was
  driven section by section through the rung's *own* `check_accuracy` and `run_case`, over the same
  24-case list in the same order, accumulating into `outputs/ladder/B_numpy.json`: accuracy
  **17.5%**, then all **24 of 24** cases, **0 failures**, with every `tau`, `domain`, `peak|u|` and
  `Re err` identical to the warp table. Recorded plainly because it is a *deviation in how the rung
  was invoked*, not in what it checked. The driver is scratch and is not committed.

**Blockers:** none.

**Next:** **T202** — the closure on the Warp backend: bitwise degeneracy there too (**Q-201**),
cross-backend agreement with the closure on against **D-053**'s 1e-6 and **D-056**'s 1e-4, the LES
performance floors, and all nine rungs on both backends. Prompt written to
`PROMPTS/025-t202-closure-on-warp.md`.
