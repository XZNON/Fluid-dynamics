# IDEA3.md — Phase 1: the product layer — **CLOSED**

> **CLOSED 2026-08-27 at M8; frozen 2026-09-01, session 23 (D-084).** Read for history; never
> edited. Phase 2's spec is **`DOCS/IDEA4.md`**. This file is cited by path from ~52 modules and
> stays where it is.


**Phase 1 in one line:** the boxes in `idea.md`'s pipeline diagram that are not the solver —
geometry repair, physical-unit configuration, stability guardrails, results that render
themselves — shipped as a Python package (`flow/`) with a CLI, running on a GPU kernel.

Phase 0's spec is `DOCS/IDEA2.md` and it is closed. **If this file conflicts with `idea.md`,
`idea.md` wins**; log the conflict in `DOCS/STATE2.md` § Decisions rather than silently picking one.
Where this file deviates from `idea.md` deliberately, it says so and names the decision (there is
exactly one such deviation: **D-043**, the GPU port moves from Phase 2 into Phase 1).

---

## Goal

Someone who has never heard of a Reynolds number gets a correct, moving, believable answer from a
picture of a shape and three physical numbers — in under a minute, from a cold shell.

That sentence is `idea.md` § Definition of success with one word added: **correct**. Phase 0 exists
because a plausible wrong answer is worse than no answer, and Phase 1 inherits that whole posture.
The difference is what "validated" now means: Phase 0 validated *physics* against published data,
and Phase 1 must additionally validate *judgement* — the parameters the tool picks on the user's
behalf, the shapes it agrees to run, and the cases it refuses.

## Scope

**In:** 2D · external flow · incompressible · single fluid · steady inlet · one or more rigid,
stationary bodies · D2Q9 / BGK / bounce-back physics · CPU (NumPy) and GPU (Warp) backends ·
a Python API and a CLI.

**Out, by name:** 3D, STL, voxelisation, moving or deforming bodies, multiphase, thermal coupling,
turbulence models, adaptive refinement, MRT, curved boundaries, XLB, a browser or desktop UI,
a hosted service, and any packaging or distribution work (`pip install`, wheels, docs sites).

`idea.md` § Risks — Scope: *"'Fluid dynamics of anything' is unbounded. Phase 1 must be narrow: 2D,
external flow, incompressible, single fluid."* The list above is that sentence made enforceable.

**A UI is not in Phase 1** (**D-044**). `idea.md`'s success test says "drags in a picture", and a
CLI does not literally do that. Phase 1 builds every layer *behind* such a UI and stops there, so
that the UI — whenever it comes — is a view over a tested API rather than the place the logic lives.
Phase 1's own success test replaces "drags in a picture" with "three lines of Python or one command,
from a cold shell, under a minute" and is otherwise identical.

## What Phase 1 is, concretely

```
   flow.Case.from_image("wing.png", fluid="air", speed="20 m/s", size="1.5 m")
        |
        v
   [ flow/prepare.py ]     mask repair: components, holes, hairlines, anti-aliasing
        |                  refuses or repairs; never silently ships a 1-cell wall
        v
   [ flow/quantity.py ]    "20 m/s" -> 20.0 m/s;  fluid names -> nu, rho
   [ flow/fluids.py ]
        |
        v
   [ flow/autoconfig.py ]  Re -> resolution, tau, U, domain size, steps, dt
        |                  every guardrail from Phase 0 enforced here, at setup
        v
   [ flow/diagnose.py ]    refusal + explanation + the nearest runnable case
        |                  and live divergence detection during the run
        v
   [ lbm/ ]                the solver. numpy or warp backend, same API, same physics
        |
        v
   [ flow/report.py ]      Cd/Cl history, St, convergence, frames, video — self-rendering
```

Every box except `lbm/` is new. `lbm/` gains a backend seam and a Warp implementation and **no new
physics**.

## The five things Phase 1 must get right

### 1. The user never types a lattice quantity

No `tau`, no lattice `U`, no `steps_per_frame`, no cell counts in any public `flow/` signature
(**Phase 1 constraint 13**). The inputs are a picture, a fluid, a speed, a size, and optionally a
duration and a quality level. Everything else is derived and **printed**, because a derived number
the user cannot see is a number they cannot check.

`lbm/units.py` already does the conversion arithmetic (`LatticeUnits.from_physical`). What it does
not do is *choose*: given Re, it will tell you a resolution that satisfies `tau > 0.51`, but nothing
picks the domain size, the run length, the frame rate, the colour limits, or the trade between
accuracy and wall clock. `flow/autoconfig.py` is that chooser, and it is the single most
product-defining module in the phase.

### 2. Refusal is a feature, and it comes with a way forward

Phase 0 ends with the CLI refusing its own acceptance command (**D-038**): air at 20 m/s past a
1.5 m body is Re 2e6, `tau` reads 0.5000, and BGK with bounce-back and no turbulence model cannot
represent it at any resolution this project will run. That refusal is correct and it stays.

It is also, as written, a dead end — and the user it will meet first is exactly the user this
product is for. Phase 1's answer (**D-045**):

- **Refuse the case as asked.** Never run something other than what was asked and call it the
  answer.
- **Explain in the user's units**, not in `tau`: "air moving that fast past something that big is
  far beyond what this simulator can represent — it needs a turbulence model, which this tool does
  not have."
- **Offer the nearest runnable case, concretely and quantitatively**: the speed that would work at
  this size, the size that would work at this speed, or the same shape at a lower Reynolds number
  clearly labelled as *not your case*.
- **Never substitute silently** (**Phase 1 constraint 16**). If the tool ran something other than
  what was asked, every artifact it produces says so — the printed summary, the report, and the
  metadata of the recorded video.

And the offer is a **testable claim**: Rung E applies the tool's own suggestion and asserts the
resulting case actually runs. A suggestion that does not fix its case is a failing test, not a
wording problem.

### 3. Real shapes are not convex blobs

`from_png` already thresholds, resamples and runs `check_mask`; `min_thickness` already measures
component-wise Chebyshev depth (**D-017**) and `--resolution` already rescales until the *measured*
body matches the request (**D-040**). What none of them do is *fix* anything, and D-017's documented
limit is precisely the real-user case: a thin appendage fused to a thick body shares its component
and is never reported.

`flow/prepare.py` owns this and has to survive a committed corpus of deliberately awful inputs
(Rung C): hairlines, detached specks, interior holes, unclosed outlines, heavy anti-aliasing, huge
margins, extreme aspect ratios, near-empty and fully-solid images. Every one has a committed
expected **verdict** — repair, warn, or refuse — plus expected measured properties. The verdict is
the known answer.

### 4. Results render themselves

Phase 0 gives one `render()` and three sinks. Phase 1 adds the *numbers*: a `Result` that carries
the Cd and Cl history, the Strouhal estimate with its confidence, the convergence trace, the peak
`|u|` against the 0.1 ceiling, the final `tau`, and the wall clock — and can emit them as a printed
summary, a plot, or a dict. Constraint 10 survives untouched: `flow/` **colours nothing**; it
composes `lbm.render` output and matplotlib figures for the scalar histories.

### 5. Speed is part of correctness of experience

"Under a minute" is not a nicety. At the NumPy kernel's measured 16.8 steps/s at 1M cells, the
Phase 0 M4 gate took **335 seconds** for 5 physical seconds at 185k cells. No amount of product
polish makes that a minute. Hence **D-043**: the Warp port (`idea.md`'s M5, its Phase 2) moves
*into* Phase 1, before the product layer, because the product layer's headline acceptance criterion
is a wall clock.

The port changes the backend and **not the physics**: same D2Q9, same BGK, same bounce-back, same
guardrails, same rungs, same published numbers within their bands. NumPy is **kept as the reference
oracle**, not replaced — every GPU claim is checked against it (Rung A parity), and a GPU that
disagrees with NumPy is a broken GPU backend, never a new answer.

## Validation ladder — five rungs, ordered, non-negotiable

Same rule as Phase 0 (constraint 5): each rung is a script in `validate/` printing PASS/FAIL, and
**Rung N+1 is not started while Rung N fails**. The harness for a rung is built in the task that
needs it, *before* the code it validates.

| Rung | Script | What it checks | Known answer |
|---|---|---|---|
| **A** | `validate/parity.py` | The Warp backend reproduces the NumPy backend | NumPy itself, plus the four Phase 0 rungs re-run on GPU and still inside their published bands |
| **B** | `validate/autoconfig.py` | The parameters auto-config picks are sane and stable | The guardrails (`tau` floors, `U < 0.1`, blockage, downstream fetch) and the analytic `Re`, reproduced to 0.1% |
| **C** | `validate/shapes.py` | Awful user geometry is repaired, warned about, or refused | A committed expectations table: one verdict + measured properties per corpus image |
| **D** | `validate/refusals.py` | Every refusal names a fix that works | Apply the tool's own suggestion; the resulting case must run |
| **E** | `validate/minute.py` | The whole product path, end to end, against published physics | Rung 3's cylinder — `St` 0.155–0.175, `Cd` 1.25–1.45 — reached through `flow.Case` from a PNG, inside a stated wall clock |

**Rung E is the phase.** It is the only rung that touches every box in the diagram at once, and it
is deliberately the *Phase 0 cylinder*: the product path has to reproduce a number this project has
already measured four times, so a regression in judgement shows up as a regression in physics.

What "a known answer" means for a usability layer, stated plainly: **a verdict and a suggestion are
falsifiable claims.** "This mask is too thin" is checkable. "Use 133334 cells and it will run" is
checkable by running it. Only the wording is subjective, and the wording is not what the rung tests.

## Performance budget

Measured on the dev machine (RTX 3050 4 GB, 16 GB RAM). **Any absolute number is quoted with the CPU
clock and the GPU name beside it** (**D-035**, generalised), and A/B is by alternating rounds.

| Grid | Cells | NumPy today | Warp floor | Warp target |
|---|---|---|---|---|
| 400×100 | 40k | 696.7 | ≥2000 | ~5000 |
| 2000×500 | 1M | 16.8 | ≥250 | ~600 |
| 2000×1000 | 2M | ~8 (est.) | ≥150 | ~400 |

Reasoning for the floors, so a future session can argue with the arithmetic instead of the number:
D2Q9 is memory-bandwidth bound. One step at 2M cells moves `9 × 4 bytes × 2M × 2` ≈ 144 MB;
at a realistic 60% of the 3050's bandwidth that is ~800 steps/s of headroom. The floors sit well
under that on purpose — they are pass conditions, not predictions, and the gap is where a first
port's inefficiency is allowed to live.

`idea.md` § Roadmap Phase 2: *"real-time 2D at 2000x1000"*. At 150 steps/s that is 2.5 steps per
display frame — real-time in the sense of a smooth window, which is the sense the product needs.

## Deliberately deferred (Phase 1)

Everything in `DOCS/IDEA2.md` § Deliberately deferred stays deferred: MRT, Smagorinsky, curved
boundaries, moving objects, thermal coupling, adaptive refinement. Added to that list for Phase 1:

- **XLB** — `idea.md` puts it at Phase 3 and Phase 1 keeps our validated kernel as the fallback the
  roadmap describes. The backend seam built in T101 is what makes that swap cheap later.
- **A UI** (D-044), **3D and STL** (Phase 4), **distribution** (packaging, wheels, a docs site).
- **Multi-body interaction studies, drag polars, parameter sweeps.** The API should not make them
  impossible; Phase 1 does not ship them.

Deferred is not forgotten. Un-deferring one is a decision to record in `DOCS/STATE2.md`.
