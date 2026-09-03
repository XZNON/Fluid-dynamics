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
| **Current task** | **T205** — packaging: `pyproject.toml`, the `fengdong` distribution (Rung I, and **M11**) |
| **Task status** | `not_started` — T204 closed in session 27 with every acceptance criterion run and passed |
| **Completed tasks** | Phase 0: T001 … T011, all eleven. Phase 1: T101 … T110, all ten. Phase 2: **T201**, **T202**, **T203**, **T204** |
| **Milestone reached** | **M10** (2026-09-03, session 27) — `DOCS/PLAN3.md`'s gate command run in full and on **both** backends: `validate.fidelity` prints **PASS**, every case in the sweep gets its band, **no run outside the quantitative band emits an unqualified `Cd`** (asserted by inspecting the object, six routes out of it), and **D-038's own case runs to completion and reports `illustrative`** — through the literal command the contract names, exiting 0 with no `Cd`. Previously **M9** (2026-09-03, session 26) — Phase 2's first — and **M8** (2026-08-27, session 22), the last of Phase 1. **M11** and **M12** remain |
| **Phase 0 rung status** | R1 🟩 · R2 🟩 · R3 🟩 · R4 🟩 — **re-run session 27 on BOTH backends, every published digit unmoved**. R1 L2 **0.3650%** (numpy) / **0.3649%** (warp). R2 **0.75%** deviation and **0.21 cells** vortex centre, *identical on both*. R3 St **0.1731** Cd **1.4031**, *identical on both*. R4 square Cd **1.5279** and polygon Cd **1.4276**, *identical on both* |
| **Phase 1 rung status** | A 🟩 · B 🟩 · C 🟩 · D 🟩 · E 🟩 — **all re-run session 27** (Rung B's numpy half was still running at checkpoint; its warp half is green — see § Provenance). A: worst kernel **5.960e-08**, whole step **9.611e-06**, checkpoint **8.196e-06**, restart within warp bit-identical — every digit unmoved. B: sweep **24/24 on warp, 0 failures, worst Re error 0.0000%**, worst peak |u| **0.0656**; accuracy warp **1.0%** (predicted 33.22 s, actual 32.88 s) against a 25% limit — the **numpy half was still running at checkpoint** and its session-26 figures (24/24, 0 failures, accuracy 3.2%) are what stand. C: **15/15** in **15 s**. D: caught before `nan` **3/3**, all eleven checks `[ok]`, Monitor cost **+1.96%** (limit 2%) on its first run after a seven-minute idle — see § Provenance. E: warp **48.2 s** (limit 60, and the fastest of the six readings on file) at **3201 of 3201 MHz on mains**, Cd **1.4040**, St **0.1672** — both physics digits exact, and the summary now carries `fidelity: quantitative` above them |
| **Phase 2 rung status** | **F 🟩 · G 🟩 · H 🟩** · I ⬜ · J ⬜. **H (T204, session 27) is green on BOTH backends and is M10.** Clause 1, the table: **14 points straddling both boundaries from both sides**, every verdict the spec's, and the **D-091 discriminator** — deleting *either* gate changes a verdict. Clause 2: a BGK plan expects its band from `Re` alone (`nu_t` is exactly 0), a closure-on plan may not expect the top band, and the measured `nu_t` overrides the expectation. Clause 3, the sweep, three full product-path runs per backend: quantitative Re **99.6**, `Cs` 0, `max(nu_t)/nu` **0**, `Cd` **1.4030** on *both* backends against Rung 3's unwidened 1.25–1.45, peak |u| **0.0972**; qualitative Re **159.4**, `tau` 0.5282, `Cs` 0.17, `max(nu_t)/nu` **0.6906** numpy / **0.6886** warp, **no `Cd` emitted** — and it is the D-091 discriminator case, at `Re <= 200` and therefore quantitative by Reynolds number alone; illustrative Re **1.979e6**, `tau` **0.500002**, `max(nu_t)/nu` **3.374e4** numpy / **3.797e4** warp, peak |u| **0.2188** / **0.2247**, **no coefficient anywhere**. Clause 4: the literal command `python -m flow --fluid air --speed "20 m/s" --size "1.5 m"` **exits 0**, reports `illustrative`, prints no `Cd`. Clause 5, **Q-203's evidence**: Rung 3's own case with the closure on reads `max(nu_t)/nu` **0.1057** — inside the qualitative band, whose boundary is 0.1 — and still prints `Cd` **1.4143**, `St` **0.1719**, *identically on both backends* — **both are green on both backends**, and together they are **M9**. F (T202, session 25; re-run session 26 unmoved): `cs_smag=0` **bitwise** the frozen Phase 1 collision after 1000 steps of Rung 3's case (`array_equal`, worst |diff| **0.000e+00**) on the fused and unfused path, against the frozen NumPy oracle on numpy and the frozen **warp** oracle on warp; Rung 3 at `Cs = 0.17` printing Cd **1.4143** and St **0.1719** on *both*; `max(nu_t)/nu` **0.1910** on that wake; cross-backend with the closure on, worst kernel **2.980e-08** against 1e-6 and whole step **9.611e-06** against 1e-4. G (T203, session 26): on a 64x64 doubly periodic Taylor–Green at `tau = 0.52`, `u0 = 0.08`, `Cs = 0` returns `(tau-0.5)/3` to **0.2303%** against Rung 1's own 1% bar; `Cs = 0.17` returns `nu + <nu_t>` to **1.1547%** against 2%, while bare `nu` misses by **3.0178%** — so deleting the `<nu_t>` term *breaks* the clause rather than passing it; the measured excess equals the dissipation-weighted `<nu_t>` to **0.9972** (**D-091**); peak |u| **0.08000** throughout against the 0.1 ceiling; cross-backend `max|du|/u0` **1.150e-05** against **D-056**'s 1e-4 and the measured `nu` agreeing to **1.434e-06** |
| **Provenance of the rung rows above** | **Session 27's own measurements.** Every one of the eleven existing rungs was re-run and **no published digit moved**: R1 L2 **0.3650%** numpy / **0.3649%** warp · R2 **0.75%** and **0.21 cells**, identical on both · R3 St **0.1731** Cd **1.4031**, identical on both · R4 square **1.5279** and polygon **1.4276**, identical on both · A worst kernel **5.960e-08**, whole step **9.611e-06**, restart within warp bit-identical · B (warp) **24/24, 0 failures**, worst Re error **0.0000%**, accuracy **1.0%** (predicted 33.22 s, actual 32.88 s) · C **15/15** in 15 s · D **3/3** caught before `nan`, all eleven checks `[ok]`, `Monitor` cost **1.96%** against its 2% limit · E warp **48.2 s** (limit 60), Cd **1.4040**, St **0.1672** · F bitwise **0.000e+00** on both backends, Rung 3 at `Cs = 0.17` printing Cd **1.4143** / St **0.1719** on both · G **0.2303%** / **1.1547%** / bare `nu` **3.0178%** / dissipation-weighted **0.9972**. **Two things to read honestly.** (1) **Rung B's numpy half was still running when the session checkpointed** — it is a ~3 h run and the warp half is green; it is the one row above carried from session 26 rather than re-measured, and it is flagged rather than quietly counted. (2) Rung D's `Monitor` cost read **1.96%** against a **2%** limit on its **first** run, taken after killing the ladder and idling the machine for seven minutes at 3201 of 3201 MHz on mains — inside the limit, and inside it by less than the machine's own run-to-run spread. **D-092** now has six readings: −0.69%, +1.02%, +0.17%, +2.11%, −0.55%, **+1.96%**. |
| **Last updated** | 2026-09-03 — session 27 (**T204 done, M10 reached**: `flow/fidelity.py` (`Band`, `band_for`, `sentence`, `Qualified`), `validate/fidelity.py` (Rung H) and `tests/test_fidelity.py` (55 tests); `flow/autoconfig.py` engages the closure instead of refusing and plans `cs_smag`; `flow/report.py` gates every reduced coefficient on the band in `Result.__post_init__`; `flow/case.py` samples `nu_t`, bands the run and marks it substituted; `flow/diagnose.py` moves two tripwires and counts what they stopped stopping; `flow/cli.py`'s `--help` rewritten; Rung D's D-038 section inverted; `validate/cylinder.py` returns its `Sim`. **Rung H green on both backends**, **Q-203 answered** (**D-095**), **D-093** and **D-094**; queued issue `2fd69b874c32` closed as a side effect. `pytest` **894 passed, 2 skipped** (287.9 s), 67 of them new). Previously: 2026-09-03 — session 26

Legend: ⬜ not attempted · 🟩 passing · 🟥 failing · 🟨 partial

## Blockers

**None.** Phase 1 closed with an empty § Blockers and Phase 2 opens the same way.

Two entries stay in the local issue queue and neither is a blocker:

- ~~**`2fd69b874c32`** — `Case.explain()` prints a different suggestion list than `Case.nearest()`
  acts on.~~ **Closed in session 27 and dropped from the queue (D-093).** The divergence was measured
  on D-038's own case, which no longer refuses at all; every refusal that remains is fixed by the same
  change from both code paths, and `flow.diagnose._present` now deduplicates by `(change, value)`.
  `tests/test_cli.py::test_the_suggestions_the_cli_prints_are_the_ones_nearest_would_run` iterates
  every reachable refusal and asserts the two lists agree, so **T207 inherits a guard rather than a
  defect**.
- **`495777c58269`** — `.gitignore` drops `*/__init__.py` and `tools/`. Open since session 16. It will
  bite T205, where a wheel has to contain every `__init__.py` it ships; fix it there or explicitly
  carry it. **This is the next task's problem and is the one queued entry that is nearly a blocker for
  it.**

A fourth was queued in session 25 and is likewise not a blocker:

- **`022ac461c920`** — *(T204 did **not** take it; it is now T208's or a `/new-task`'s.)* `Sim`
  allocates `smag_work` `(4, ny, nx)` on backends that never read it. The
  Warp backend (T202) ignores it — a GPU thread has registers, so there is nothing to stage — so with
  the closure on at 2M cells that is **32 MiB of device memory allocated and never read**, against a
  391 MiB `Sim` footprint on a 4 GiB card. Correct, bounded, and nothing today runs the closure at
  that size. The fix crosses the T101 seam (an optional backend attribute, or moving the scratch
  decision into the backend's own allocation), so it is a decision rather than a tidy-up; **T204** and
  **T208** both touch memory and either is a natural place.

A third entry was queued in session 24 and is likewise not a blocker. **Session 27 could not
reproduce it**: Rung D printed `on mains` and `Win32_Battery.BatteryStatus` read 2 for the whole
session, so it is either intermittent or already fixed by something since. It is left queued rather
than dropped, because "could not reproduce once" is not "does not happen":

- **`d5b27e51fcdc`** — `validate/refusals.py` printed `on battery` in its D-035 conditions line while
  `Win32_Battery.BatteryStatus` read **2 (mains)** for the whole session and `validate/minute.py`
  printed `power: mains` for the same machine minutes later. One of the two power probes is wrong,
  and **D-035** requires the power state beside every absolute timing figure, so a probe that
  misreports it makes those figures unquotable. The fix is to make the two rungs share one
  implementation. Rung D passes either way — the discrepancy is in the *label*, not in the check.

One thread carried since session 18 and now **measured for the first time, though not by a rung that
was asked to**: `Monitor` on `warp`. Rung D runs on `numpy` by design; session 26's ladder script ran
`validate.refusals --backend warp` as well, which was **beyond the published M9 gate** and is recorded
here rather than dropped, because the number is the one **T208** exists to find. Measured:
**bare 1274.3 steps/s, watched 1149.2, cost 9.82%** against Rung D's 2% limit, with all eleven physics
checks `[ok]`. That is not a regression and it does not fail any gate — Rung D's published invocation
is numpy, which read **−0.55%** on the same machine — it is the expected price of a device-side probe
that pulls state back across the bus every check. **It stays T208's acceptance criterion**, and T208
now starts from a number instead of from a question.

**On this machine, a process that runs longer than roughly ten minutes under the agent's own tooling
is liable to be killed** (session 24 lost several Rung B attempts that way, one of them after 98
CPU-minutes, and an orphaned survivor then competed for CPU with its own replacement). This is an
environment property, not a project defect, and it is recorded here because **Rung B and Rung 4 are
both past that line** and every session that re-runs the ladder will hit it.

**Session 25 found the clean workaround and it should be the default from now on: run the long rungs
detached, in the background, redirected to a file under `outputs/ladder/`, and poll the file.**
Foreground commands are hard-capped at ten minutes and are killed on the dot; a backgrounded one is
not, and it leaves no orphan because it is the same process the poller is watching. Every rung this
session ran to completion that way, including the numpy half of Rung B, which session 24 could not
run as a single process at all. Two things to expect: Python **block-buffers a redirected stdout**,
so a long rung shows **zero bytes** for most of its life and silence is not a stall (check
`UserModeTime` on the worker instead — note `myenv\Scripts\python.exe` is a *trampoline* and the real
worker is a separate `uv` `python.exe`); and the measured wall clocks were **Rung 4 ~35 min per
backend** and **Rung B numpy ~3 h 15 m**, the latter mostly at a throttled 1882 MHz.

## Open questions

- **Q-201 — CLOSED by T202 (session 25), answered by measurement. The answer is: two separately
  compiled kernels, and the question then never gets asked.** With `cs_smag = 0` the Warp backend
  launches `_collide_kernel` / `_collide_bb_kernel` *unedited* — the Phase 1 kernels, not a Phase 1
  path through a new kernel — so bitwise equality is a property of the dispatch and not of the
  floating-point arithmetic, and no algebraically-zero term is ever multiplied in on any backend.
  Measured against a frozen transcription of those same Phase 1 warp kernels in `validate/les.py`:
  `numpy.array_equal` after 1000 steps of Rung 3's case, worst |diff| **0.000e+00**, on the fused and
  the unfused path. Recorded as **D-088**. The original question is kept below for the record.

  ~~Does bitwise degeneracy on the Warp backend survive multiplying the closure term by
  zero, or does `cs_smag = 0` need a separately compiled kernel?~~ **D-053** documents that the GPU
  contracts `x * a + b` into one rounding where NumPy does two, so a term that is algebraically zero
  is not automatically bitwise inert. **Still open — T202 answers it by measurement.** The note in
  `DOCS/TASKS3.md` § T202 is explicit that the branch is the fix and the tolerance is not.
  **What T201 changed about it (session 24):** the NumPy side did not wait to find out. **D-086**
  makes `cs_smag == 0.0` an explicit early-return branch in `lbm/core.py::collide` and a scalar-vs-field
  branch in `::collide_stream`, so nothing algebraically zero is ever multiplied in on *either*
  backend's reference path. T202 inherits a branch to port rather than a tolerance to argue about,
  and the question narrows to whether Warp needs *two compiled kernels* or one guarded branch
  suffices.
- **Q-202 — CLOSED by T203 (session 26), answered by measurement. The answer is `<nu_t>/nu` =
  1.8418%, and the model does *not* fire hard on a smooth flow.** On the 64x64 doubly periodic
  Taylor–Green at `tau = 0.52`, `u0 = 0.08`, `Cs = 0.17`, the domain-averaged eddy viscosity is
  **1.227895e-04** against `nu = 0.00666667` — 1.8% — where the calibration points are `max(nu_t)/nu`
  **0.1910** on Rung 3's shedding wake and **9.011e-02** on Rung A's smooth channel, both from session
  25. So a resolved laminar flow sits an order of magnitude below either, which is the expected
  behaviour and not a finding about the implementation. **What *is* a finding, and is recorded as
  D-091:** the number is a *design output*, not a fact about Taylor–Green — `<nu_t>/nu` scales as
  `u0 / (L nu)` and the case was sized deliberately so the term would be large enough for the 2% bar
  to have teeth. A more resolved case makes the closure more inert, not less.

  ~~What is `<nu_t>` on a *resolved* 2D Taylor–Green at `Cs = 0.17`, as a fraction of `nu`?
  Expected small.~~ Kept for the record.
- **Q-203 — CLOSED by T204 (session 27), answered by measurement. The answer is: a *qualified* `Cd`
  in the qualitative band, and stability-only in the illustrative one.** Recorded as **D-095**. The
  evidence is Rung 3's own case run with the closure forced on, which sits at `max(nu_t)/nu` =
  **0.1057** — inside the qualitative band, whose boundary is 0.1 — and still prints `Cd` **1.4143**
  and `St` **0.1719** against Rung 3's published, **unwidened** 1.25–1.45 and 0.155–0.175. That is a
  falsifiable claim about the band rather than an argument about it, so the qualitative band emits a
  `flow.fidelity.Qualified`: the number with its band and its caveat welded on, and deliberately no
  `__float__`. The illustrative band takes the pressure valve, because there the evidence cannot
  exist: at `max(nu_t)/nu` = **3.797e4** the model supplies four orders of magnitude more viscosity
  than the fluid does, the wake is three-dimensional above Re ~190, and no 2D closure repairs that —
  so **no coefficient is emitted at all**, anywhere on the object. **The margin is thin and is
  recorded as thin**: 0.1057 against a 0.1 boundary. Nothing was widened to reach it.

  ~~Can the fidelity bands be made falsifiable enough to report a qualified `Cd` outside the
  quantitative band, or does the closure ship **stability-only**?~~ Kept for the record.
- **Q-204** — Does `fengdong` publish to PyPI inside this phase, or does Rung I's locally built wheel
  close it? Publishing needs an account, a `LICENCE` file and a considered first version number, and
  none of the three is a packaging detail. **T205 raises it; the user decides.**
- **Q-205 (new, session 27)** — **D-094** moved `Monitor`'s speed and mass wires to the meaning bound
  on a closure-on run, and what it honestly gave up is early warning for a closure-on run that runs
  away *between* the two bounds: peak `|u|` climbing from 0.1 towards 0.5774, or mass leaking from 1%
  towards 50%. Both counters are kept and printed, so the *information* is there; nothing acts on it.
  Does a live run need a **trend** wire — divergence as "over the accuracy bound **and rising**" —
  and if so, does that replace the fixed bounds or sit beside them? Measured input already on file:
  D-038's own case is flat at 0.20 from step 4000 to 48000 and linear in mass, so a trend wire would
  be silent on it, which is the behaviour wanted. **T208 is where a run is watched live and is the
  natural place to raise it**; it is not a blocker and Rung H covers the finished-run case.

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

**Phase 2's budget** (`DOCS/IDEA4.md` § Performance budget), **measured by T202** with
`bench.py --backend <b> --les`. Conditions per **D-035**: AMD Ryzen 7 5800H at
`Win32_Processor.CurrentClockSpeed` **3201 of 3201 MHz**, on **mains**, clock read immediately before
and immediately after; NVIDIA GeForce RTX 3050 Laptop GPU, driver **592.82**. Alternating rounds,
five rounds, best round per variant, one `Sim` resident.

| Grid | Cells | Published floor | BGK (warp, this run) | **LES (warp) measured** | LES/BGK | Result |
|---|---|---|---|---|---|---|
| 400×100 | 40k | ≥ 3116 | 4103.8 | **3914.2** | 95.4% | **PASS** |
| 2000×500 | 1M | ≥ 568 | 724.2 | **660.0** | 91.1% | **PASS** |
| 2000×1000 | 2M | ≥ 331 | 445.4 | **406.0** | 91.2% | **PASS** |

**Re-measured in session 26 as the M9 gate** (same conditions: 3201 of 3201 MHz on mains, RTX 3050,
driver 592.82, five alternating rounds, machine idled first): warp **3504.0 / 661.6 / 403.7** steps/s
at 40k / 1M / 2M against the same floors **3116 / 568 / 331** — **PASS on all three**, with the BGK
column reading 3830.9 / 731.8 / 446.5 in the same rounds, so the closure cost **8.5% / 9.6% / 9.6%**.
The 1M and 2M figures reproduce session 25's to within a percent; **the 40k figure moved most**
(3914.2 -> 3504.0, and its BGK arm 4103.8 -> 3830.9), which is the grid where launch overhead is the
largest share of the step and therefore the one most sensitive to machine state. Both are far above
the floor and no floor moved.

The BGK column is measured in the same alternating rounds rather than read from T103's table
(4155.0 / 757.3 / 441.0), which is why it differs by a few percent; the **floors** are T103's numbers
times 0.75 and do not move. The NumPy column has no published floor and is held to the ratio alone,
against its own measured BGK column in the same rounds:

| Grid | Cells | BGK (numpy) | **LES (numpy)** | LES/BGK | Result |
|---|---|---|---|---|---|
| 400×100 | 40k | 683.4 | **558.9** | 81.8% | **PASS** |
| 2000×500 | 1M | 17.9 | **14.4** | 80.5% | **PASS** |
| 2000×1000 | 2M | 6.8 | **5.3** | 78.1% | **PASS** |

**The single number worth carrying forward: the closure costs 4.6% / 8.9% / 8.8% of the Warp step
rate at 40k / 1M / 2M cells, against a 25% budget** — better than XLB's own indicative 17%, and it
got there by *folding the second-moment reduction into the fused kernel* (**D-089**). The two-kernel
form measured first cost **27.7%** at 2M, which is inside the published floors but outside the 25%
rule read as a ratio; both numbers are recorded because the difference between them is the decision.

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
| D-088 | 2026-09-02 | **Constraint 19 on the Warp backend is two separately compiled kernels, not one guarded branch: with `cs_smag = 0` the backend launches `_collide_kernel` and `_collide_bb_kernel` unedited, and the closure lives in its own kernels beside them. This is the measured answer to Q-201, which is now closed.** Extends **D-086** onto the GPU rather than superseding it: NumPy takes a branch back into Phase 1's own three operations, warp takes a branch back into Phase 1's own compiled kernel. | The question **Q-201** asked was whether an algebraically-zero closure term stays bitwise inert on a device that contracts `x * a + b` into one rounding where NumPy does two (**D-053**). Two compiled kernels make the question unanswerable-and-irrelevant rather than answered: the ``cs_smag = 0`` path is *the same instructions it was before the closure existed*, so equality is a property of the dispatch, not of floating point, and it cannot regress under a future compiler or driver. It costs one Python ``if`` per call — no per-step cost, since the branch is on the host — and it is why the closure could be optimised (**D-089**) without putting a single one of the nine green rungs at risk. Measured against a frozen transcription of the Phase 1 **warp** kernels (**D-090**): `numpy.array_equal` after 1000 steps of Rung 3's case, worst |diff| **0.000e+00**, fused and unfused. |
| D-089 | 2026-09-02 | **On the fused path the closure's second-moment reduction is folded into `_collide_bb_smag_kernel` itself — two loops over the nine directions in one thread — rather than run as a separate `_smag_scale_kernel` launch. The unfused `collide` keeps the two-kernel form.** The arithmetic is identical either way: same operations, same order, same `float32`, and `smag_out` still receives `1 - omega_eff` so the buffer holds what NumPy's does. | Measured, not assumed. The step is memory-bound, and a separate scale kernel is a **second full pass over `f` and `feq`** — 18 planes of read traffic added to a kernel that already reads 18 and writes 9. `bench.py --backend warp --les` read the closure at **27.7% of the BGK step rate at 2M cells** as two kernels and **8.8%** as one, at 3201 of 3201 MHz on mains; the folded form clears the 25% budget on every grid and on both backends, where the two-kernel form cleared the published floors but not the ratio. The fold is legitimate *only* because the reduction is per-cell and this kernel is one thread per cell: each thread reduces its own nine directions before it writes any of them, which is the per-thread form of the "there is exactly one moment at which the whole pre-collision state exists" that `lbm.core.collide_stream` relies on. `collide` is one thread per **(direction, cell)**, so folding there would make every thread redo the whole reduction, and it is left alone. Constraint 6's replacement permits this at all only because Rung F on warp was green first: correct, then fast. |
| D-090 | 2026-09-02 | **Rung F takes `--backend`, carries a second frozen oracle — the Phase 1 *warp* kernels, transcribed once in `validate/les.py` and marked do-not-edit — and gains a fourth clause measuring cross-backend agreement with the closure ON against Rung A's own bars. Rung A itself stays closure-off.** `validate/parity.py::step_case` and `::whole_step` gain a `cs_smag: float = 0.0` that every caller in that module leaves at zero. | Three separate arguments, all of them **D-087**'s. (1) *A backend compared with itself proves nothing*, so "bitwise what Phase 1 shipped" on warp needs a Phase 1 warp kernel to compare against, and after T202 there is not one in the tree — hence a transcription, and exactly **one** of it, in the same file and under the same rule as the NumPy oracle. It is a transcription rather than an import so that a later edit to `lbm/backends/warp_backend.py` cannot silently move both sides of the comparison. `_stream_kernel` **is** imported, for the reason `_shift_blocks` is on the host side: streaming moves values and does no arithmetic. (2) *A second copy of Rung A's case would be a case whose agreement with the real Rung A nobody checks*, so clause 4 threads `cs_smag` through Rung A's own `step_case` and `whole_step` instead. (3) *Rung A measures what constraint 19 says it should* — the closure off — so the closure-on version of the same question belongs to the rung that owns the closure. Measured: worst kernel **2.980e-08** against 1e-6, whole step **9.611e-06** against 1e-4. Recorded in the rung's own output: the whole-step figure lands on **Rung A's closure-off digits** even though `max(nu_t)/nu` is **9.011e-02** on that case, because **D-053**'s FMA contractions dominate the disagreement and the closure perturbs both backends coherently — the clause prints the eddy-viscosity ratio precisely so no reader has to guess whether it was inert. |

| D-091 | 2026-09-03 | **Rung G's case is *sized*, and the rung carries a third clause that no contract asked for: the closure's contribution must be large enough that deleting it fails the rung.** The operating point is 64x64 doubly periodic, one wavelength, `tau = 0.52`, `u0 = 0.08`, warm-up 0.3 `T_d` and a fit window of 1.0 `T_d`, where `T_d = 1 / (2 nu K^2)`. Two checks are added beside the contract's own: (a) *bare `nu` must miss the 2% bar* — the discriminator; and (b) *the measured excess equals the **dissipation-weighted** `<nu_t> = <nu_t^3>/<nu_t^2>`* to 5%, which is the bias-free form of the same claim. `EPS_TOL = 0.05`. | Three measured facts, none of them guessable from the contract. **(1) The 2% bar is vacuous on a well-resolved case.** `<nu_t>/nu` scales as `0.147 u0 / (L nu)`; at a comfortable resolved point (L = 64, `tau = 0.55`, `u0 = 0.05`) it is **0.14%**, so the clause would pass with the `<nu_t>` term deleted — precisely the *"wrong sim that looks plausible"* constraint 5 names, and a green rung that proves nothing. The case is therefore sized to `<nu_t>/nu` = **1.84%**, where the contract's check reads **1.1547%** (inside 2%) and bare `nu` reads **3.0178%** (outside it). **(2) The domain average carries a known geometric bias on this flow, and it is not an error.** Taylor–Green has `S_xy = 0` identically, so `|S| = 2 u0 k |sin kx x sin ky y|` and `nu_t` is strongly non-uniform; the energy decay responds to the dissipation-weighted mean, which is `(<\|s\|^3>/<s^2>)/<\|s\|> = ((4/3pi)^2/(1/2)^2)/(2/pi)^2 = ` **1.7780** times the domain mean. Measured across seven cases spanning L = 16..64, `tau` = 0.51..0.55 and `u0` = 0.05..0.07, the ratio of measured excess to domain mean sat at **1.69–1.79** — so the contract's domain-average comparison is systematically high by `0.78 <nu_t>`, which is why its 2% bar and the discriminator's 2% bar together admit only `<nu_t>/nu` in roughly 1.1%–2.6%. That window is *narrow but deterministic*: it is a property of the case, not of the run, and numpy and warp land on it to seven digits. **(3) Re-weighting the model's own field removes the bias entirely and costs nothing.** `nu_t = Cs^2 |S|`, so the dissipation weight `S_ab S_ab = |S|^2/2` is proportional to `nu_t^2` with every constant cancelling — `<nu_t^3>/<nu_t^2>` is therefore computable from exactly the :func:`lbm.probe.eddy_viscosity` field already being sampled, with **no analytic input and no fitting**. Against it the measured excess is **0.9972** — the closure adds what it claims to **0.3%** — and that number was 0.997 on every one of the seven cases, so the check has margin where the contract's has a window. Both are kept: the contract's because it is the contract's, and this one because it is the one that would catch a wrong coefficient. |
| D-092 | 2026-09-03 | **A wall-clock A/B check that fails on a loaded machine is re-run on an idled one, and *both* readings are recorded.** Applied to Rung D this session; extends session 25's Rung E lesson from an observation into the standing procedure for every rung with a timing clause. | Rung D's `Monitor` cost check has now read **−0.69%**, **+1.02%**, **+0.17%**, **+2.11%** and **−0.55%** across five sessions against a **2%** limit — a ~3-point spread straddling zero, on a machine whose `CurrentClockSpeed` reports 3201 while sustained load clocks it well below (session 25). This session's first reading, **+2.11%**, came after an hour of continuous load and had its *bare* arm running **faster** than session 25's passing run (78.5 vs 76.0 steps/s), so the machine was not slow — the two arms drifted apart. The second reading, after a seven-minute idle at 3201 of 3201 MHz on mains, was **−0.55%**. All eleven physics checks were `[ok]` in both. Recording both readings rather than the passing one is what stops the next session reading a clean history and concluding the check is tight when it is in fact measuring this machine's run-to-run spread — which Rung D's own output already says is 12–21% between two runs of the *identical* path. |
| D-093 | 2026-09-03 | **D-038's refusal is superseded. `flow.autoconfig.plan` engages the closure below `TAU_FLOOR` instead of refusing, and the only floor left is the one constraint 2 sets — `nu > 0`, i.e. `tau > 0.5` (`TAU_FLOOR_CLOSURE`).** `Plan` gains `cs_smag` and a `why` entry for it, printed by `--explain`; a case that clears `TAU_FLOOR` still plans `cs_smag = 0.0` and runs bitwise as Phase 1 ran it (constraint 19). The `relaxation` refusal class stays **reachable** — by an inviscid fluid, `nu <= 0` — so constraint 14's machinery is exercised by a real case rather than a synthetic, and Rung D's section 2 flips from *"it must refuse"* to *"it plans, the closure is what let it, and the band it expects is not a quantitative one"*. `flow/cli.py`'s `RE_LIMIT_NOTE` and `flow/diagnose.py`'s `_FIRST_PARAGRAPH["relaxation"]` are both rewritten, because both said the tool has no turbulence model. | This is the wall the whole phase exists to remove (`DOCS/IDEA4.md` § Goal; **D-038**, **D-074**), and removing it is a one-line change to a comparison — the honesty is in what replaces the refusal, not in the switch. Three things decided with it. **(1) The remaining floor is not a taste.** `nu = (tau - 0.5)/3`, so `tau > 0.5` is exactly `nu > 0`, and the closure raises the *effective* relaxation time where there is strain (**D-085**) and cannot raise the base one — there is nothing below 0.5 for any model to rescue. **(2) The class stays reachable, which was not free.** With D-038's case gone the only way to `tau <= 0.5` is a zero viscosity, and `reynolds = u l / nu` was a `ZeroDivisionError` on that path — an unreachable crash that became reachable the moment the closure carried cases down to the floor. It is now a refusal, and `_tau_suggestions` gained an inviscid branch because neither a slower speed nor a smaller body fixes a fluid with no viscosity: the fix is a **fluid**, named from the library and checked before it is offered. Rung D applies it and runs 2000 steps of what it produces. **(3) A side effect worth recording: queued issue `2fd69b874c32` is closed.** `Case.explain()` printed a different suggestion list from the one `Case.nearest()` acted on, measured on D-038's own case (*speed, size* vs *fluid -> honey*). That case no longer refuses, every refusal that remains is fixed by the same change from both code paths, and `flow.diagnose._present` now deduplicates by `(change, value)` — so the two lists agree on **every** reachable refusal, which `tests/test_cli.py` iterates rather than asserting about one case. |
| D-094 | 2026-09-03 | **On a closure-on run, `flow.diagnose.Monitor`'s speed and mass tripwires move from the accuracy bound to the meaning bound — `CS_SOUND = 1/sqrt(3)` and `MASS_DRIFT_MEANINGLESS = 0.5` — and every crossing of the narrow bound is still counted and printed.** The criterion is `closure`, not the fidelity band: the closure is engaged exactly when `tau <= TAU_FLOOR`, which is knowable before the first timestep, where a band is not. `Monitor` gains `closure=`, `over_accuracy_ceiling`, `over_accuracy_drift`, `peak_seen` and `drift_seen`; `Result.warnings` reports both counts; `validate/refusals.py::run_plan` applies the same bound to the plan it runs. Rung D's own invocation is unaffected — it builds a default `Monitor`. | **Measured, and it is why the acceptance criterion is meetable at all.** D-038's case at `quality="fast"` on warp, the plan's own 48000 steps: the state is **finite at every sample**, the peak `|u|` climbs to **0.20** by step 4000 and is **flat** to the last step, `rho` sits in 0.78..1.12 throughout, and the fluid mass leaks **linearly** at ~0.11% per 1000 steps to **5.24%** at the end. Against that, `Monitor` as Phase 1 wrote it fires on the speed wire at **step 75** and on the mass wire at **step 11800**, raising `Diverging(cause="relaxation")` whose text reads *"the flow is growing without bound"* — which is false: the flow is bounded and steady, and the leak is a convective outlet radiating faster than a fluid with no viscosity can damp, not a domain filling until it bursts. So the wires were measuring the wrong thing in this regime, and the fix is to move them to a bound that means something rather than to switch them off. **Both new bounds come from one argument**, which is why this is a single decision: D2Q9 is an expansion about `rho = rho0` with `|u| << cs`, and constraint 3's Mach-squared error *is* that expansion's error term. Above `U_LATTICE_MAX` and above `MASS_DRIFT_ACCURACY` the answer is **inaccurate** — which is precisely what the fidelity band exists to say, and `Result.peak_u` still prints `** OVER THE LIMIT **`; above `cs` and above half the domain's mass the expansion has **nothing left to say**, and a run that reaches either is running away. Measured margin on D-038's own case: **0.2247 against 0.5774 (2.6x)** and **5.24% against 50% (9.5x)**. The interlock against this being a quiet weakening is threefold: the narrow bounds are still evaluated and their crossings printed (1919 of 1920 samples over 0.1, 1370 of 1920 over 1%, both in `Result.warnings`); Rung H asserts the illustrative run is **finite at its last step**, so nothing was hidden; and the not-finite wire is untouched. What is honestly given up is early warning for a closure-on run that genuinely runs away between `0.1` and `cs` — recorded here rather than glossed. |
| D-095 | 2026-09-03 | **Q-203 answered: the qualitative band ships a *qualified* `Cd`, and the illustrative band ships stability-only.** `flow.fidelity.Qualified` is a frozen record carrying `cd`, `cd_std`, `cl`, `strouhal`, its `band` and its `caveat`, with **no `__float__`**, so the number cannot be slipped into arithmetic or a format string as if it were validated. Constraint 18 is implemented in **one place** — `Result.__post_init__` — which sets every entry of `flow.report.GATED_QUANTITIES` to `None` outside the quantitative band and drops `cd_qualified` too in the illustrative one; `summary`, `as_dict`, `plot` and the video metadata therefore have nothing to leak. The raw force histories stay on the result in every band, labelled as the run's data and not as a coefficient. | **The evidence existed before the module did, and Rung H re-measures it rather than citing it.** Rung 3's own case with `Cs = 0.17` sits at `max(nu_t)/nu` = **0.1057** at the end of its full 45500-step run — *inside* the qualitative band, since the boundary is 0.1 — and prints `Cd` **1.4143** and `St` **0.1719** against Rung 3's published, **unwidened** 1.25–1.45 and 0.155–0.175, on both backends. That is a falsifiable claim about the band and not an argument about it, so the pressure valve in `DOCS/PLAN3.md` § Risks is **not** taken for the qualitative band. It **is** taken for the illustrative one, where there is no evidence and by construction cannot be: at `max(nu_t)/nu` = 3.797e4 the model supplies four orders of magnitude more viscosity than the fluid, the wake is three-dimensional above Re ~190 (Williamson 1996) and no 2D closure repairs that — so no coefficient is emitted at all. Two implementation choices are load-bearing. **(1) The gate is on the record, not on the renderer**: a printer that formats a number the object withheld cannot exist, which is what lets Rung H assert constraint 18 *by inspecting the object* as the contract requires. **(2) `Result.fidelity` defaults to `quantitative`**, the band a plain-BGK run at `Re <= 200` earns by construction, so a `Result` built without one behaves exactly as Phase 1's did — and because that default is the permissive one, `tests/test_fidelity.py` reads `flow/case.py`'s **syntax** to assert the one production caller always passes `fidelity=`, `expected_fidelity=` and `closure_engaged=` explicitly. Note the margin the qualitative claim rests on: **0.1057 against a 0.1 boundary**. It is a measurement, it is close, and it is recorded as close rather than smoothed — if a future change moved it under 0.1 the case would band *quantitative* and this clause would have no case left to make, which is a thing to notice rather than to discover. |

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
| **18** | — | **New (D-082); landed by T204.** No unqualified quantitative claim outside the validated band. `flow/fidelity.py` decides the band from the eddy viscosity the run generated, `flow.report.Result.__post_init__` withholds every claim the band forbids, and Rung H asserts it by inspecting the object. **D-095** is which way Q-203 went. |
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


### 2026-09-02 — Session 25: T202 — the Smagorinsky closure on the Warp backend

**Task:** **T202**. **Status: done — every acceptance criterion in `DOCS/TASKS3.md` § T202 was run and
passed.** **Rung F is green on both backends**, which is half of **M9**; Rung G (T203) is the other
half and is not started.

**Read, in the protocol's order:** `PROMPTS/025-t202-closure-on-warp.md`; `CLAUDE.md`;
`DOCS/STATE3.md` in full; `DOCS/TASKS3.md` § T201 and § T202 and the backlog index;
`DOCS/IDEA4.md` § Performance budget and § Validation ladder; `DOCS/PLAN3.md` § Why this order,
§ Session map, § Milestone gates, § Risks; `lbm/core.py`'s closure, `lbm/backends/__init__.py`,
`lbm/backends/warp_backend.py`, `lbm/runner.py`'s `cs_smag` wiring, `validate/les.py`,
`validate/parity.py`, `validate/cylinder.py`, `bench.py`.

**Done**

- **`lbm/backends/warp_backend.py`** — three new kernels and one helper. `_smag_scale_kernel` is
  :func:`lbm.core.smagorinsky_tau_eff` transcribed term for term into one thread per cell, followed
  by the single reciprocal and the ``1 - omega`` the collision multiplies by;
  `_collide_smag_kernel` is `_collide_kernel` with the factor read from an ``(ny, nx)`` field;
  `_collide_bb_smag_kernel` is `_collide_bb_kernel` with the whole reduction folded in ahead of the
  direction loop (**D-089**). `WarpBackend._smag_scalars` folds the three `float64`-then-rounded
  scalars host-side in NumPy's own expression order (**D-057**). **Both `NotImplementedError` stubs
  are gone**, and `cs_smag = 0.0` now dispatches to the *unedited* Phase 1 kernels (**D-088**).
- **`validate/les.py`** — `--backend` and `--skip-cross`; a frozen Phase 1 **warp** oracle
  (`_phase1_collide_kernel`, `_phase1_collide_bb_kernel`, `Phase1WarpBackend`) under D-087's
  one-copy rule, guarded so the module still imports without warp-lang; `PHASE1_ORACLES`, one per
  backend, which `_run` selects from and which *refuses* rather than falling back if a backend has
  none; and **clause 4**, cross-backend agreement with the closure on (**D-090**).
- **`validate/parity.py`** — `step_case` and `whole_step` gained `cs_smag: float = 0.0`, left at zero
  by every caller in that module. Rung A is unchanged in what it measures.
- **`bench.py`** — `--les`, `LES_FLOORS`, `LES_MIN_RATIO`, `LES_STEPS`, `LES_FLOORS_BY_BACKEND` and
  `print_les_table`. Pass condition is the published absolute floor where there is one (warp) and the
  75% ratio against this run's own BGK column where there is not (numpy); both are printed either
  way.
- **`lbm/backends/__init__.py`** — the `Backend` protocol's two `cs_smag` docstrings had their
  `cs_smag:` line mis-indented out of the `Args:` block; fixed, and `smag_work` now says a backend
  whose threads have registers may ignore it, because the Warp one does.
- **`tests/test_smagorinsky.py`** — the T201 test asserting warp *refuses* the closure is replaced by
  five that assert it implements it: cross-backend agreement at D-053's bar on both entry points,
  `cs_smag = 0` bitwise against the backend's own BGK kernels, `_smag_scalars` against core's exact
  expressions, an AST/source check that `SMAG_Q_COEFF` is imported and never restated (constraint 4),
  and **constraint 11 with the closure on** — the checkpoint is still exactly
  `{f, solid, step_count, config, format}` and a warp restart is bit-identical.
- **`CLAUDE.md`** — § Commands gained `validate.les --backend warp` and `bench.py --les`, the
  `bench.py` root-file row describes `--les`, and § Current state records T202 and Rung F 🟩.

**Measured — every figure below is this session's own run**

- **Rung F, full, on BOTH backends: PASS.** `cs_smag = 0` is `numpy.array_equal` to the frozen Phase 1
  kernels after **1000 steps of Rung 3's case**, worst |diff| **0.000e+00**, fused and unfused, on
  numpy against the NumPy oracle and on warp against the frozen **warp** oracle; fused and unfused
  agree bitwise with each other (**D-055** survives on both). Rung 3's full case at `Cs = 0.17`
  printed **Cd 1.4143**, **St 0.1719** on *both backends* — the same four digits — with `tau_eff`
  0.5378 … 0.5450 and `max(nu_t)/nu` **0.1910**. Clause 4 (warp): worst kernel **2.980e-08** against
  **D-053**'s 1e-6, whole step **9.611e-06** at 1000 steps against **D-056**'s 1e-4, with
  `max(nu_t)/nu` **9.011e-02** on that case printed beside it so the number cannot be mistaken for an
  inert clause.
- **`bench.py --les`, at 3201 of 3201 MHz on mains, RTX 3050, driver 592.82, five alternating
  rounds.** Warp **3914.2 / 660.0 / 406.0** steps/s at 40k / 1M / 2M against floors
  **3116 / 568 / 331** — PASS on all three, the closure costing **4.6% / 8.9% / 8.8%**. NumPy
  **558.9 / 14.4 / 5.3**, i.e. **81.8% / 80.5% / 78.1%** of its own BGK column, all inside the 25%
  rule. The two-kernel form measured **27.7%** at 2M before the fold; both numbers are in
  § Performance baseline because the gap between them is **D-089**.
- **All nine existing rungs re-run on BOTH backends, and no physics digit moved.** R1 L2 **0.3650%**
  numpy / **0.3649%** warp · R2 **0.75%** and **0.21 cells**, identical on both · R3 St **0.1731**
  Cd **1.4031**, identical on both · R4 square **1.5279** and polygon **1.4276**, identical on both ·
  A worst kernel **5.960e-08**, whole step **9.611e-06**, checkpoint **8.196e-06**, restart within
  warp bit-identical · B **24/24 with 0 failures on both**, worst Re error **0.0000%**, accuracy warp
  **0.6%** and numpy **0.3%** · C **15/15** in 14.5 s · D **3/3** caught before `nan`, Monitor cost
  **+0.17%** · E warp **57.2 s** (limit 60), Cd **1.4040**, St **0.1672**.
- **`pytest`: 803 passed, 2 skipped** in 286.1 s (session 24: 800 passed, 1 skipped). +4 net: five new
  warp tests less the one T201 refusal test they replace. The extra skip is not a regression — it is
  `test_smagorinsky.py`'s git-history test, which skips now that T201 is committed and could only run
  while it was not.

**Two measurements that had to be taken twice, recorded rather than quietly re-run**

- **Rung E read 68.2 s on its first run of the session and 57.2 s on its second**, against a 60 s
  limit, with identical physics digits (Cd 1.4040, St 0.1672) both times and
  `Win32_Processor.CurrentClockSpeed` reading **3201 of 3201 on mains** both times. The difference is
  the machine's thermal state: the first run followed ~2.5 hours of continuous full-load compute, the
  second followed a seven-minute idle. **`CurrentClockSpeed` is an instantaneous reading and does not
  detect this** — it says 3201 while a sustained load is being clocked well below it, which is the
  same trap session 24 fell into from the other direction. The published figure is the second, and the
  lesson for any future timing gate is to idle the machine before it, not merely to read the clock.
- **Rung 3 on warp reported FAIL on its first invocation and PASS on its second**, with the physics
  digits (St 0.1731, Cd 1.4031) identical in both. Of that rung's seven checks exactly one is
  wall-clock-derived — *"window costs < 10% of steps/s (constraint 8)"*, which read **-0.84%** on the
  passing run — so that is what moved. Recorded because a rung that flakes on a timing check under
  load is worth knowing about before Rung J tries to time the whole product.

**Decisions made**

- **D-088** — constraint 19 on warp is two compiled kernels; `cs_smag = 0` launches the Phase 1 kernel
  unedited. **Closes Q-201 by measurement.**
- **D-089** — the reduction is folded into the fused kernel; 27.7% of the BGK step rate as two
  kernels at 2M cells, 8.8% as one.
- **D-090** — Rung F takes `--backend`, carries a frozen Phase 1 *warp* oracle, and owns the
  closure-on cross-backend clause; Rung A stays closure-off.

**Not done / deferred**

- **`Sim` still allocates `smag_work` `(4, ny, nx)` on the Warp backend, which never reads it** — 32
  MiB of dead device memory at 2M cells with the closure on. Queued as **`022ac461c920`** rather than
  fixed, because the fix crosses the T101 seam and is a decision, not a tidy-up. Not a blocker: it is
  correct, it is bounded, and nothing today runs the closure at that size.
- **`lbm.probe.eddy_viscosity` has no device-side implementation.** Rung F reads it host-side off
  `Sim.host_f()`, which is what T204's banding will also do at frame cadence. Nothing in T202 needed
  it on the GPU and nothing was written speculatively.
- **Rung G was not started**, per constraint 5 and scope discipline. M9 needs it.

**Blockers:** none.

**Next:** **T203** — the Taylor–Green harness: `validate/taylorgreen.py`, Rung G, and with it **M9**.
Prompt written to `PROMPTS/026-t203-taylor-green.md`.


### 2026-09-03 — Session 26: T203 — the Taylor–Green harness, Rung G, and **M9**

**Task:** **T203**. **Status: done — every acceptance criterion in `DOCS/TASKS3.md` § T203 was run and
passed.** **Rung G is green on both backends, and with Rung F that is milestone M9** — Phase 2's
first, and the gate was run in full rather than claimed.

**Read, in the protocol's order:** `PROMPTS/026-t203-taylor-green.md`; `CLAUDE.md`; `DOCS/STATE3.md`
in full; `DOCS/TASKS3.md` § T201, § T202, § T203 and the backlog index; `DOCS/IDEA4.md` § The five
things Phase 2 must get right (1) and (2) and § Validation ladder; `DOCS/PLAN3.md` § Why this order,
§ Session map, § Milestone gates, § Risks; `validate/poiseuille.py` (Rung 1, the model for an
analytic-solution harness), `lbm/core.py`'s closure, `lbm/probe.py::eddy_viscosity`,
`lbm/runner.py`'s `SimConfig` / `Sim`, `validate/les.py`, `validate/parity.py`.

**Done**

- **`validate/taylorgreen.py`** — Rung G. `taylor_green` builds the exact vortex on integer node
  positions with the Taylor–Green pressure folded into `rho`; `decay_time` is the analytic e-folding
  time; `run_decay` decays one vortex and fits `ln E` against `t`; `check_cross_backend` runs both
  backends from a bit-identical `f` in the shape of `validate.parity.whole_step`; `report` prints
  every check. Takes `--backend`, `--cs`, `--ny/--nx/--tau/--u0` and `--skip-cross`.
- **`tests/test_taylorgreen.py`** — 24 tests, one per acceptance criterion plus the invariants,
  including an **AST** check that the only curve fit in the module is the energy fit and that it
  never touches a `nu_t` series (a fitted `<nu_t>` is the answer copied into the question).
- **`CLAUDE.md`** — § Commands gained `validate.taylorgreen --backend warp`.

**Measured — every figure below is this session's own run**

- **Rung G, on BOTH backends: PASS.** 64x64 doubly periodic, no bodies, `tau = 0.52`
  (`nu = 0.00666667`), `u0 = 0.08`, warm-up 1167 steps then 3880 fitted in 41 samples.
  `Cs = 0`: measured `nu` **0.00665131** — **0.2303%** against Rung 1's own **1%** bar, with
  `ln E` fit `R^2` = **1.000000** and `<nu_t>` **exactly** zero at every sample (constraint 19).
  `Cs = 0.17`: measured `nu` **0.00686786** against `nu + <nu_t>` = **0.00678946** — **1.1547%**
  against the **2%** bar; bare `nu` misses by **3.0178%**, so **deleting the `<nu_t>` term breaks the
  clause rather than passing it** (D-091). The excess equals the dissipation-weighted `<nu_t>` to
  **0.9972**. Peak |u| **0.08000** throughout, warm-up included, against the 0.1 ceiling
  (constraint 3). Cross-backend: `max|du|/u0` **1.150e-05** against **D-056**'s 1e-4, worst |df|
  **1.013e-06**, and the measured `nu` agreeing to **1.434e-06** (0.00686786 vs 0.00686787).
- **The M9 gate, run in full.** Eighteen ladder runs plus Rung E and `bench.py --les`. Rungs F and G
  on both backends; **all nine existing rungs re-run and every published digit unmoved** — R1 L2
  **0.3650%** numpy / **0.3649%** warp · R2 **0.75%** and **0.21 cells**, identical on both · R3 St
  **0.1731** Cd **1.4031**, identical on both · R4 square **1.5279** and polygon **1.4276**, identical
  on both · A worst kernel **5.960e-08**, whole step **9.611e-06**, checkpoint **8.196e-06**, restart
  bit-identical · B **24/24 on both, 0 failures**, worst Re error **0.0000%** · C **15/15** in
  **15.9 s** · D **3/3** caught before `nan`, Monitor cost **−0.55%** · E warp **55.6 s** (limit 60),
  Cd **1.4040**, St **0.1672**.
- **`bench.py --backend warp --les`**, machine idled first, at 3201 of 3201 MHz on mains, RTX 3050,
  driver 592.82, five alternating rounds: **3504.0 / 661.6 / 403.7** steps/s at 40k / 1M / 2M against
  floors **3116 / 568 / 331** — **PASS on all three**, the closure costing **8.5% / 9.6% / 9.6%**.
- **`pytest`: 827 passed, 2 skipped** in 224.3 s (session 25: 803 passed, 2 skipped; +24 new, which
  is exactly the new file).

**Two measurements that had to be taken twice, recorded rather than quietly re-run (D-092)**

- **Rung D read `Monitor` cost +2.11% against its 2% limit on its first run and −0.55% on its
  second**, with all eleven physics checks `[ok]` both times. The first followed an hour of
  continuous load; the second followed a seven-minute idle at 3201 of 3201 MHz on mains. The *bare*
  arm of the failing run was **faster** than session 25's passing run (78.5 vs 76.0 steps/s), so the
  machine was not slow — the two arms drifted apart. The published figure is the second.
- **Rung B numpy took 11.9 hours of wall clock for 70 minutes of CPU.** The machine slept overnight
  mid-run and then ran on **battery at 1882 of 3201 MHz**. It cost wall clock and moved no digit:
  24/24, 0 failures, worst Re error **0.0000%**. Both timing gates (`bench --les`, Rung E) had
  already been banked on mains at 3201 of 3201 before that happened, which is why **D-035**'s
  requirement to quote the power state beside every absolute figure is what kept them quotable.

**Decisions made**

- **D-091** — Rung G's case is *sized* so the closure's contribution is large enough to matter, and
  the rung carries a discriminator plus a dissipation-weighted check. **Closes Q-202** with
  `<nu_t>/nu` = **1.8418%**.
- **D-092** — a wall-clock A/B that fails under load is re-run on an idled machine and **both**
  readings are recorded.

**Not done / deferred**

- **`Monitor` on `warp` is now measured, but not by a rung that was asked to.** The ladder script ran
  `validate.refusals --backend warp`, which is **beyond the published M9 gate** — Rung D's published
  invocation is numpy — and it read **9.82%** against the 2% limit with all eleven physics checks
  `[ok]`. That is the price of a device-side probe pulling state across the bus, not a regression,
  and it stays **T208**'s acceptance criterion; T208 now starts from a number. See § Blockers.
- **`Sim` still allocates `smag_work` `(4, ny, nx)` on the Warp backend, which never reads it** —
  queued as **`022ac461c920`**, unchanged from session 25. Not a blocker; the fix crosses the T101
  seam and is T204's or T208's to take.
- **Nothing was banded.** **D-082**'s three fidelity bands are **T204** and no part of this session
  started them, per scope discipline.
- **No enstrophy-cascade check was written**, per `DOCS/TASKS3.md` § T203 Notes: Taylor–Green is
  chosen *because it has an exact solution*, and a cascade check is a better test of a turbulence
  model and a worse test of this claim.

**Blockers:** none.

**Next:** **T204** — `flow/fidelity.py`: the three bands decided from the eddy viscosity a run
generated, wired through `autoconfig` / `diagnose` / `report`, and **Rung H** → **M10**. Prompt
written to `PROMPTS/027-t204-fidelity-bands.md`.


### 2026-09-03 — Session 27: T204 — `flow/fidelity.py`, the bands wired through, Rung H, and **M10**

**Task:** **T204**. **Status: done — every acceptance criterion in `DOCS/TASKS3.md` § T204 was run and
passed.** **Rung H is green on both backends, and that is M10.** With it, the wall this whole phase
exists to remove is gone: `--fluid air --speed "20 m/s" --size "1.5 m"` — **D-038**'s case, refused by
Phase 0 and re-refused by Phase 1 — now exits 0, reports `illustrative`, and prints no `Cd`.

**Read, in the protocol's order:** `PROMPTS/027-t204-fidelity-bands.md`; `CLAUDE.md`;
`DOCS/STATE3.md` in full; `DOCS/TASKS3.md` § T201–T204 and the backlog index; `DOCS/IDEA4.md`
§ The five things Phase 2 must get right (1) and § Validation ladder; `DOCS/PLAN3.md` § Why this
order, § Session map, § Milestone gates, § Risks; `flow/autoconfig.py`, `flow/case.py`,
`flow/report.py`, `flow/diagnose.py`, `flow/cli.py`, `lbm/probe.py::eddy_viscosity`,
`validate/les.py`, `validate/refusals.py`, and the constraint-13/15 tests in
`tests/test_flow_package.py`.

**Two measurements taken before any code was written, because both changed the design**

- **Rung 3's own case does not want the closure**, confirmed by arithmetic and then by running it:
  the closure engages iff `Re >= 3.75 N`, i.e. Re 112.5 / 150 / 187.5 at fast / balanced / accurate,
  and the product's Rung 3 case is Re **99.6**. So `nu_t` is identically zero there, the band is
  decided from `Re` alone, and Rungs 3, B and E cannot move. `PROMPTS/027` asked for this to be the
  first thing checked and it was.
- **D-038's case runs, and it sits outside constraint 3's ceiling.** On warp at `quality="fast"`
  (720x540, `tau` 0.5000023, `nu` 7.5e-7), the plan's own 48000 steps: **finite throughout**, peak
  `|u|` **0.20** and *flat* from step 4000 to the last one, `rho` in **0.78..1.12**, mass leaking
  **linearly** at ~0.11% per 1000 steps to **5.24%**, `max(nu_t)/nu` **3.7e4**. The peak sits at the
  **inlet** (column x = 2), where the flow is uniform and the closure therefore supplies nothing, and
  it is ~0.2 at `U` = 0.05, 0.03, 0.02 and 0.01 alike — an undamped acoustic mode, not a flow feature.
  Against that, `Monitor` as Phase 1 wrote it fires on the speed wire at **step 75** and on the mass
  wire at **step 11800**. That is what **D-094** exists to answer.

**Done**

- **`flow/fidelity.py`** (new) — `Band` (`quantitative` / `qualitative` / `illustrative`, with
  `rank`, `reports_bare_numbers`, `reports_qualified_numbers`, `worse_of`),
  `band_for(plan, nu_t_max=None)`, `ratio_for`, `sentence(band)`, `Qualified`, and the two boundaries
  as named constants — `RE_3D_ONSET = 200` **cited to Williamson (1996)**, `RATIO_QUANTITATIVE = 0.1`,
  `RATIO_QUALITATIVE = 1.0`.
- **`flow/autoconfig.py`** — `TAU_FLOOR_CLOSURE = 0.5`, `CS_SMAG_PLANNED`, `Plan.cs_smag` with
  `closure_engaged` and `expected_fidelity`, a `why["cs_smag"]` in both directions, and two setup
  warnings (constraint 3's "warn at setup", and an answer to `stability_note`'s "expect nan", which is
  a plain-BGK measurement). The `tau <= TAU_FLOOR` refusal becomes closure engagement (**D-093**); the
  refusal that remains is `tau <= 0.5`. `reynolds` no longer divides by zero on an inviscid fluid, and
  `_tau_suggestions` gained the branch that names a **fluid** for that case, because no speed and no
  size fixes it.
- **`flow/report.py`** — `Result.fidelity`, `expected_fidelity`, `cd_qualified`, `nu_t_ratio`,
  `closure_engaged`; `GATED_QUANTITIES`; and **`__post_init__` as the single implementation of
  constraint 18**. `summary` gained `_coefficient_lines` (three shapes, one per band), `as_dict`
  carries the band and the qualified record, `plot` draws the trace in every band but titles it with
  the band, and `metadata_entries` gained `fidelity=`, `closure=` and `provisional=`.
- **`flow/case.py`** — `FIDELITY_SAMPLES_PER_RUN = 12`; `measure_nu_t` at a coarse cadence **only
  when the closure is on** (with the closure off `nu_t` is exactly zero and nothing is sampled, which
  is what keeps Rung E costing what it costs); the earned band from a final sample of the end state;
  `_substitution`, which makes a closure-on run a substituted run (constraint 16) and joins the two
  sentences when both apply; `Monitor(closure=...)`; the two "what the widened wire stopped stopping"
  warnings; and `cs_smag` and the expected band printed by `explain()`.
- **`flow/diagnose.py`** — `CS_SOUND`, `MASS_DRIFT_ACCURACY`, `MASS_DRIFT_MEANINGLESS`;
  `Monitor(closure=)` with `speed_ceiling`, `over_accuracy_ceiling`, `over_accuracy_drift`,
  `peak_seen`, `drift_seen`; `_present` deduplicates suggestions by `(change, value)`; and
  `_FIRST_PARAGRAPH["relaxation"]` rewritten, because it told the user this tool has no turbulence
  model.
- **`flow/cli.py`** — `RE_LIMIT_NOTE` rewritten around the **fidelity band** rather than around a
  refusal that no longer happens.
- **`validate/fidelity.py`** (new) — Rung H, five clauses, printing PASS/FAIL, `--backend`,
  `--skip-sweep`, `--skip-cylinder`.
- **`validate/refusals.py`** — the `relaxation` row is now the inviscid case; `run_plan` runs the plan
  it was given, closure included, against the same bound `Monitor` uses; § 2 flipped from "the D-038
  refusal" to "the D-038 supersession".
- **`validate/cylinder.py`** — `CylinderResult.sim`, a defaulted handle so Rung H can measure
  `max(nu_t)/nu` and `Cd` **on the same run**. Nothing in Rung 3 reads it.
- **`tests/test_fidelity.py`** (new, 55 tests) plus edits to `test_autoconfig`, `test_diagnose`,
  `test_report`, `test_case`, `test_cli`, `test_flow_package` (`cs_smag`/`cs` added to
  `LATTICE_NAMES`) and `test_smagorinsky` (the T201 ban on `flow/` mentioning the closure becomes the
  sharper rule: no **function** in `flow/` may *take* `cs_smag`).
- **`CLAUDE.md`** — § Commands gained Rung H's three invocations and D-038's own command; the
  constraint list records that 18 and 19 have landed; § Current state rewritten.

**Measured — every figure below is this session's own run**

- **Rung H, full, on BOTH backends: PASS.** The table: **14 points** straddling both boundaries from
  both sides, every verdict the spec's, plus the **D-091 discriminator** — deleting either gate
  changes a verdict. The sweep, three full product-path runs per backend: quantitative Re **99.6**,
  `Cs` 0, `max(nu_t)/nu` **0**, **`Cd` 1.4030 on both backends** against Rung 3's unwidened 1.25–1.45,
  peak `|u|` 0.0972; qualitative Re **159.4**, `Cs` 0.17, `max(nu_t)/nu` **0.6906** numpy /
  **0.6886** warp, **no `Cd` emitted** — and this is the discriminator case, inside the `Re <= 200`
  gate and therefore quantitative by Reynolds number alone; illustrative Re **1.979e6**, `tau`
  **0.500002**, `max(nu_t)/nu` **3.374e4** / **3.797e4**, peak `|u|` **0.2188** / **0.2247**, no
  coefficient anywhere. The literal command exits **0**. And **Q-203's evidence**: Rung 3's own case
  with the closure on reads `max(nu_t)/nu` **0.1057** — inside the qualitative band — and still prints
  `Cd` **1.4143**, `St` **0.1719**, *identically on both backends*. numpy took **63 min**, warp
  **5 min**.
- **All eleven existing rungs re-run, and no published digit moved.** R1 **0.3650%** / **0.3649%** ·
  R2 **0.75%** and **0.21 cells** · R3 St **0.1731** Cd **1.4031** · R4 **1.5279** / **1.4276** ·
  A **5.960e-08** / **9.611e-06** · B (warp) **24/24, 0 failures**, worst Re error **0.0000%** ·
  C **15/15** · D **3/3** caught before `nan` · E **48.2 s**, Cd **1.4040**, St **0.1672** ·
  F **0.000e+00** bitwise on both · G **0.2303%** / **1.1547%** / **3.0178%** / **0.9972**.
- **`pytest`: 894 passed, 2 skipped** in 287.9 s (session 26: 827 passed, 2 skipped; **+67**).

**Decisions made**

- **D-093** — D-038's refusal is superseded; the closure engages below `TAU_FLOOR` and the only floor
  left is `nu > 0`. Closes queued issue `2fd69b874c32` as a side effect, and fixes a
  `ZeroDivisionError` that became reachable the moment the closure carried cases to the floor.
- **D-094** — on a closure-on run `Monitor`'s speed and mass wires move from the accuracy bound to the
  meaning bound (`1/sqrt(3)`, half the domain's mass), and every crossing of the narrow bounds is
  still counted and printed.
- **D-095** — **closes Q-203**: a *qualified* `Cd` in the qualitative band, stability-only in the
  illustrative one, with constraint 18 implemented in one place.

**Not done / deferred**

- **Rung B's numpy half was still running at checkpoint.** It is a ~3 h run; its warp half is green
  (24/24, 0 failures, worst Re error 0.0000%, accuracy 1.0%) and its numpy figures in § Snapshot are
  session 26's, carried and **labelled as carried**. Everything else in the ladder is this session's
  own.
- **`022ac461c920` (the dead `smag_work` allocation on warp) was not taken.** `PROMPTS/027` offered
  T204 as a natural place and said taking it was optional and a `/new-task`; it was not taken, so the
  32 MiB of unread device memory at 2M cells with the closure on is unchanged and stays T208's or a
  `/new-task`'s.
- **No enstrophy or cascade check, and no `Cs` tuning.** Constraint 1 forbids a dynamic procedure and
  `DOCS/PLAN3.md` § Risks names tuning `Cs` as the phase's trap; the literature 0.17 is planned and
  printed and was not touched.
- **`bench.py --les` was not re-run.** It is M9's gate, not M10's, and nothing this session touched
  `lbm/` at all — the only edit outside `flow/`, `validate/` and `tests/` is a defaulted field on
  `CylinderResult`.
- **The `d5b27e51fcdc` power-probe discrepancy could not be reproduced**: Rung D printed `on mains`
  and `Win32_Battery.BatteryStatus` read 2 all session. Left queued rather than dropped.

**One thing worth carrying forward, stated plainly**: the qualitative band's whole claim rests on
`max(nu_t)/nu` = **0.1057** against a boundary of **0.1**. That is a measurement and nothing was
widened to reach it — but the margin is 5.7%, and if a future change pushed that case under 0.1 it
would band `quantitative` and Rung H's clause 5 would have no case left to make. Notice it rather than
discover it.

**Blockers:** none.

**Next:** **T205** — packaging: `pyproject.toml`, the `fengdong` distribution, `validate/install.py`,
and **Rung I** → **M11**. It depends on nothing in T201–T204 (`DOCS/PLAN3.md` § Why this order, 4).
Prompt written to `PROMPTS/028-t205-packaging.md`.
