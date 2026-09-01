# IDEA4.md — Phase 2: FengDong — the closure and the window

**Phase 2 in one line:** the two things standing between a validated 2D solver and a person using it
— a turbulence closure that turns Phase 1's honest refusal into an honest *answer*, and a window you
drop a picture onto — shipped as one `pip install`.

**风洞 (fēngdòng)** is Chinese for *wind tunnel*. It is the distribution name, the command, and the
title bar. `flow` was taken on PyPI; `fengdong` is free (checked, session 23).

Phase 0's spec is `DOCS/IDEA2.md` and Phase 1's is `DOCS/IDEA3.md`; both are closed. **If this file
conflicts with `idea.md`, `idea.md` wins** — except where it says otherwise below and names the
decision. There are two such deviations, both recorded as **D-080**: `idea.md`'s Phase 3 (swap in
XLB) and its Phase 4 (3D + STL) are *both* deferred past this phase, on measured evidence.

---

## Goal

Someone who has never heard of a Reynolds number installs it with one command, drags a picture onto
a window, and watches a flow the tool is honest about — **including the cases Phase 1 had to
refuse.**

That is `idea.md` § Definition of success, finally taken literally. Phase 1's **D-044** replaced
*"opens the tool, drags in a picture"* with *"three lines of Python or one command"*, on the grounds
that a UI is the one layer addable later without invalidating anything beneath it. Everything beneath
it is now built and validated by nine rungs. This phase spends that.

The second half of the sentence — *"including the cases Phase 1 had to refuse"* — is the physics
half. **D-038** and **D-074** are the same event one layer apart: the very first thing a plausible
user asks for (air at 20 m/s past a 1.5 m body; air at 5 m/s past 10 cm) is refused, because BGK with
bounce-back and no closure cannot represent it at any resolution this project will run. The refusal
is correct and Phase 1 was right to ship it. It is also the wall every real user hits first.

## Scope

**In:** everything Phase 1 had — 2D · external flow · incompressible · single fluid · steady inlet ·
rigid stationary bodies · D2Q9 / BGK / bounce-back · NumPy and Warp backends · the `flow` API and the
`python -m flow` CLI — **plus**:

- a **Smagorinsky eddy-viscosity closure** on the existing BGK collision, both backends, defaulting
  **off**;
- a **fidelity band** printed on every result, so a stabilised answer is never mistaken for a
  validated one;
- a **desktop application** on pygame — a window, a drop target, a live view, a numbers panel;
- **distribution**: `pip install fengdong`, then `fengdong`.

**Out, by name:** 3D · STL · voxelisation · **XLB** · KBC · MRT · cumulant · curved or interpolated
boundaries · wall models · dynamic (Germano) Smagorinsky · moving or deforming bodies · multiphase ·
thermal coupling · adaptive refinement · multi-GPU · **a web UI, a browser, a hosted service** ·
a documentation site · parameter sweeps · drag polars · multi-body interaction studies.

**Platform:** Phase 2 claims **Windows**, in writing. The code is portable Python and nothing in it is
deliberately Windows-only, but only Windows is tested, and the README says so rather than implying a
support surface nobody has run.

### Why not XLB, and why not 3D

`idea.md` § Roadmap puts XLB at Phase 3 and 3D at Phase 4. Both are deferred past this phase and
**D-080** records the measurements, taken in session 23 on this machine.

**XLB was installed and run, not read about.** `pip install "xlb[warp]"` succeeds on Windows /
Python 3.11 (xlb 0.3.1, jax CPU wheels, no build step), and its Warp D2Q9 path compiles and runs on
this RTX 3050 — including `SmagorinskyLESBGK`, which has an explicit 2D branch. Three findings
decided against it:

1. **Its current release does not import against the Warp we run.** `xlb` 0.3.1 imports
   `ScopedTimer` from `warp.utils`, removed somewhere between warp-lang 1.11 and 1.14. It imports at
   **1.11.0** and fails at **1.14.0** and **1.16.0** (ours). Adopting it means pinning warp five
   minors back *for our own validated backend too*, and re-proving Rung A there. `idea.md` § Risks —
   *"XLB dependency. If it stagnates, we inherit that"* — arriving in the install.
2. **Its seam is the wrong shape.** `IncompressibleNavierStokesStepper` owns the grid, the boundary
   conditions and the whole timestep. Our `Backend` protocol (**D-054**) is per-kernel *because* that
   is what lets Rung A bisect a parity failure to `equilibrium` vs `collide` vs `stream`
   (**D-053**). Fitting XLB behind it means surrendering that, or reaching past their stepper into
   their operators and depending on internals.
3. **It buys nothing this phase needs.** Its 2D-relevant gift is the LES closure, which is ~20 lines
   in our own `collide` and needs **no seam change**. Its real gifts — D3Q19/D3Q27, KBC, a trimesh
   voxeliser — are all 3D, and 3D is out.

Measured throughput, **indicative only and not a D-035 measurement** (the machine read
`CurrentClockSpeed` 1990 of 3201 MHz, and XLB's step carried one bounce-back BC against our Zou–He
inlet plus convective outlet): XLB warp D2Q9 at 100k cells, **4782.7** steps/s BGK and **3958.2**
with Smagorinsky, against our measured 4155.0 at 40k and 3560.4 at 160k (**D-077**). Same class. The
**17%** the closure costs there is the one number worth carrying forward, as a sanity check on ours.

Recorded because it is load-bearing for constraint 4: XLB returns `f` as `(9, nx, ny, 1)` —
`(q, x, y, z)` — against our `(9, ny, nx)` `(direction, y, x)`. Satisfiable by a transpose at
`to_host` cadence, so the contract survives. It is not free.

**3D is refused by arithmetic, not by taste.** Our own quality floor is `QUALITY_CELLS["fast"] = 30`
cells across the body (**D-059**) at a 24 D span (**D-075**), which in 3D is a 720³ grid:
`720³ × 19 × 4 B` = **28.4 GB per buffer**, and a run needs four. Halving the span to 12 D still gives
360³ = **3.5 GB per buffer** on a 4 GB card. What fits — about 192³, 538 MB per buffer — delivers
**8 to 16 cells across the body**, a resolution *the 2D product already refuses*. `idea.md`'s "caps
around 192³–224³ — fine for development" is true for development and false for the validated answer
this project's ladder demands. 3D is a rent-a-GPU phase.

## What Phase 2 is, concretely

```
   fengdong                  <-- a window. a PNG dropped on it. air. 20 m/s. 1.5 m.
        |
        v
   [ fengdong/widgets.py ]   label, text field, dropdown, button, drop target. that list, closed.
   [ fengdong/app.py ]       the window, the event loop, the panels
        |
        v
   [ flow.Case ]             unchanged. the app computes no solver parameter of its own
        |
        v
   [ flow/autoconfig.py ]    + the closure: the tau floor becomes reachable at high Re
   [ flow/fidelity.py ]      NEW: a case -> a band, a verdict, a sentence a person can act on
        |
        v
   [ lbm/core.py ]           + Smagorinsky: per-cell tau_eff from the second moment of (f - feq)
   [ lbm/backends/ ]         + the same on both backends, bitwise identical to BGK at Cs = 0
        |
        v
   [ flow/report.py ]        Result carries `fidelity`. Outside the validated band there is no
                             unqualified Cd — the number is withheld or qualified, never bare.
```

Every box below `flow.Case` already exists and is validated. `lbm/` gains **one** physics option and
no new seam method beyond the one it rides on. `fengdong/` is entirely new and computes nothing.

## The five things Phase 2 must get right

### 1. The closure buys stability, not fidelity — and the tool says which

This is the whole risk of the phase, stated once, plainly.

Smagorinsky raises the effective relaxation time where strain is high, so a case that currently reads
`tau` 0.5000 and gets refused instead **runs**, at affordable resolution. That is real, and it is what
makes the refusal wall passable.

It does not make the answer right. **The cylinder wake becomes three-dimensional at Re ≈ 190**
(Williamson 1996, the mode-A instability) — which is exactly why Rung 3 sits at Re 100. Above that a
2D simulation is wrong about *the flow*, not about the numerics, and no two-dimensional closure
repairs it: Smagorinsky descends from Kolmogorov's forward energy cascade, and 2D turbulence cascades
energy the other way (Kraichnan 1967). The drag crisis near Re 3e5 is boundary-layer transition plus
spanwise instability and cannot appear in a 2D run at any resolution with any model.

So the closure extends **what runs**, not **what is trustworthy**, and shipping it without saying so
would walk this project straight into the failure mode constraint 5 names — *a wrong sim that looks
plausible*. Hence **constraint 18** and `flow/fidelity.py`: three bands, and the boundary between them
is **measured per case, not read off a Reynolds number**.

| Band | Condition | What the tool prints |
|---|---|---|
| **quantitative** | `Re <= 200` **and** `max(nu_t)/nu < 0.1` | the numbers, unqualified — the range the nine rungs validate |
| **qualitative** | `max(nu_t)/nu < 1` | the wake, the picture, the trends; `Cd` **qualified**, never bare |
| **illustrative** | otherwise — the closure supplies more viscosity than the fluid does | a moving picture and no quantitative claim at all |

`nu_t` is the Smagorinsky eddy viscosity the run actually generated. The upper boundary is therefore
*"the model is now doing more of the work than the physics"* — a measurable statement about **this
run**, rather than a magic Reynolds number argued into a table. The `Re <= 200` gate on the top band
is the physics one, and it is cited, not chosen.

### 2. Off must be off, bitwise

`Cs = 0` must reproduce plain BGK **bit for bit**, on both backends, and the closure defaults **off**
for every existing rung. A closure you cannot switch off is a closure you cannot validate against,
and nine green rungs are the thing this phase has to not break. That is **constraint 19**, and it is
Rung F's first assertion — checked before anything else in the phase runs.

### 3. The app is a view, not a second brain

`fengdong/` may import `flow/`; `flow/` may never import `fengdong/`, and a test asserts it
(**constraint 17**, the same shape as **constraint 15** and for the same reason). Every solver
parameter the app displays comes from `flow.autoconfig.plan` — the app has no opinion about grids,
`tau`, run lengths or colour limits, and if it ever computes one, the seam has already failed.

Constraint 13 extends with it: **no lattice quantity appears in a widget.** `Cs` is not a user knob.
The fidelity band is what surfaces instead, because that is the thing a person can act on.

### 4. Watching stays smooth, and never at the sim's expense

Constraints 7 and 8 are untouched and re-asserted. The app is a **fourth sink on the existing ring
buffer**, not a new path to the screen: `steps_per_frame` is still computed from target playback
speed, display frames are still dropped when the buffer fills, and simulation steps still never are.
`lbm/render.py` still produces the pixels; `fengdong/` still colours nothing (constraint 10).

### 5. One command installs it, on a machine that is not ours

`pip install fengdong`, then `fengdong`, in a fresh virtual environment, with no repository checkout
and no `myenv/Scripts/python.exe` in sight. Rung I is that, run as a test, because a package that only
installs from the developer's tree is not distributed.

## Validation ladder — five rungs, ordered, non-negotiable

Same rule as Phases 0 and 1 (constraint 5): each rung is a script in `validate/` printing PASS/FAIL,
**Rung N+1 is not started while Rung N fails**, and a rung's harness is built in the task that needs
it, *before* the code it validates. **All nine existing rungs stay a gate for every Phase 2 task** —
Phase 2's five are added to them, never instead of them.

| Rung | Script | What it checks | Known answer |
|---|---|---|---|
| **F** | `validate/les.py` | The closure is switchable and does not disturb what already works | `Cs = 0` reproduces plain BGK **bitwise** on both backends (`numpy.array_equal` on `f` after 1000 steps); and Rung 3 with the closure **on** still prints `Cd` 1.25–1.45, `St` 0.155–0.175 |
| **G** | `validate/taylorgreen.py` | The closure does not pollute a resolved laminar flow | 2D Taylor–Green: `u = u0 cos(kx) sin(ky) exp(-2 nu k^2 t)`, exact. At `Cs = 0` the measured decay rate returns `nu` to under 1% (Rung 1's own bar); at `Cs = 0.17` it returns `nu + <nu_t>` to under 2%, with `<nu_t>` computed from the model rather than fitted |
| **H** | `validate/fidelity.py` | Every band's claim is true, and no run overclaims | A Re sweep across all three bands. The quantitative band reproduces published data; **no run outside it emits an unqualified `Cd`**, asserted by inspecting `Result`; every band carries its sentence; **D-038**'s own case (air, 20 m/s, 1.5 m) runs and lands in `illustrative` |
| **I** | `validate/install.py` | One command installs it, off this tree | A **fresh venv**, `pip install <built wheel>`, then `fengdong --version` and a headless smoke of the app's model layer. No repo on the path |
| **J** | `validate/drop.py` | The whole app path, end to end | A PNG delivered as a `pygame.DROPFILE` event reaches Rung 3's published bands — `St` 0.155–0.175, `Cd` 1.25–1.45 — through the app, with the elapsed time printed |

**Rung J is the phase**, the way Rung E was Phase 1's: the only rung that touches every box at once,
and deliberately the Phase 0 cylinder again, so a regression anywhere shows up as a regression in
physics this project has already measured five times.

Rung F's bitwise clause is the one to run first and to keep running. It is cheap, it is absolute, and
it is the difference between "we added a turbulence model" and "we changed all nine rungs and did not
notice".

## Performance budget

Measured on the dev machine (RTX 3050 4 GB, Ryzen 7 5800H, 16 GB RAM). **Any absolute number is quoted
with the CPU clock, the power state and the GPU name beside it** (**D-035**), and A/B is by
alternating rounds.

**The closure.** One extra second-moment reduction over nine directions per cell, then a square root.
Bandwidth is unchanged — it reads `f` and `feq`, both already resident — so the cost is arithmetic on
a memory-bound kernel and should be small.

| Grid | Cells | BGK measured (warp, T103 / D-077) | LES floor |
|---|---|---|---|
| 400×100 | 40k | 4155.0 | ≥ 3116 (within 25%) |
| 2000×500 | 1M | 757.3 | ≥ 568 (within 25%) |
| 2000×1000 | 2M | 441.0 | ≥ 331 (within 25%) |

25% is a pass condition, not a prediction. XLB's own 2D Smagorinsky costs **17%** on this card
(4782.7 → 3958.2 steps/s at 100k, indicative), so the budget has headroom for a first implementation
to be inelegant in. NumPy takes the same 25% rule against its own measured column.

**The app.** The window must sustain **30 fps of display** while the sim runs at `quality="balanced"`
on the warp backend, with **zero dropped simulation steps** — display frames may drop, and the app
must report how many rather than hide it (constraint 8, **D-039**'s posture).

**The install.** `pip install fengdong` into a fresh venv, then first window on screen, under
**60 seconds** on a warm pip cache. Rung I prints it.

## Deliberately deferred (Phase 2)

Everything in `DOCS/IDEA2.md` and `DOCS/IDEA3.md` § Deliberately deferred stays deferred except the
two items this phase un-defers by name — **a UI** (D-044) and **packaging and distribution** — both
now in scope, and both recorded in **D-080**. Still deferred, and now with reasons that are measured
rather than assumed:

- **XLB** — measured in session 23 and deferred on the evidence above (**D-080**). Not forgotten: the
  D-054 seam still exists and still makes the swap a substitution the day 3D is worth it.
- **3D and STL** — deferred by the memory arithmetic above, not by preference. A rent-a-GPU phase.
- **KBC, MRT, cumulant, curved and interpolated boundaries, wall models.** Constraint 1 permits *one*
  closure and it is named. Interpolated bounce-back is the most tempting of these — XLB has it, and a
  digitised disc would measurably benefit — and it is still out.
- **Dynamic (Germano) Smagorinsky.** `Cs` is a literature constant this phase: fixed and cited. A
  procedure that computes it per cell is a second closure wearing the first one's name.
- **A web UI, a browser, a hosted service, a docs site.** The user's call in session 23: a desktop
  application, no website.
- **Parameter sweeps, drag polars, multi-body studies, moving bodies.**

Deferred is not forgotten. Un-deferring one is a decision to record in `DOCS/STATE3.md`.
