# PLAN3.md — Phase 2 implementation plan

Implementation plan for `DOCS/IDEA4.md`: the Smagorinsky closure, the fidelity bands it makes
necessary, and the `fengdong` desktop application that ships on top of them.

**Plan shape:** 9 tasks, T201 → T209. **One task per session.** Task contracts and acceptance
criteria live in `DOCS/TASKS3.md`; live status lives in `DOCS/STATE3.md`. Phase 0's equivalents
(`old-Docs/PLAN1.md`, `old-Docs/TASKS1.md`, `old-Docs/STATE1.md`) and Phase 1's
(`DOCS/PLAN2.md`, `DOCS/TASKS2.md`, `DOCS/STATE2.md`) are closed and frozen — read them for history,
never edit them.

---

## Why this order

1. **The closure before anything that depends on it.** `flow/fidelity.py` (T204) cannot band a run
   whose eddy viscosity does not exist yet, and the app (T206–T208) cannot show a band that has not
   been defined. Physics first, judgement second, view third — the same spine Phase 1 used.
2. **NumPy before Warp, again.** T201 lands the closure on the reference oracle, where a wrong
   answer is debuggable; T202 ports it. This is **D-043**'s posture unchanged: NumPy is the oracle,
   and a GPU that disagrees with it is a broken GPU (**D-053**, **D-056**).
3. **Bitwise degeneracy before any physics claim.** Rung F's `Cs = 0` clause is the very first thing
   T201 writes, before the model itself does anything. Nine green rungs are what this phase risks,
   and the cheapest possible test protects all nine. Constraint 19 exists to make that non-optional.
4. **Packaging before the app, not after.** T205 comes before T206 deliberately. A package assembled
   *after* a GUI exists is a package shaped around whatever imports the GUI happened to grow; a
   package that exists first means every app task is written inside a working distribution and Rung I
   never has a bad day. It is also the natural fall-through if the closure work stalls, because it
   depends on nothing in T201–T204.
5. **Widgets before the window.** T206 builds and unit-tests the closed widget set headless; T207
   assembles a window out of tested parts. A widget layer written *inside* an app is a widget layer
   nobody can test without a screen.
6. **Every rung's harness lands in the task that needs it, before the code it validates**
   (constraint 5's spirit), and **every session ends on a green boundary** — a passing rung, a passing
   unit-test set, or an explicit "half-done, here is what is missing" in `DOCS/STATE3.md`. Never on
   ambiguity.

## Dependency graph

```
T201 Smagorinsky in lbm/core.py + NumPy backend ──► RUNG F (numpy: bitwise + Rung 3 on)
  └─ T202 the closure on the Warp backend ────────► RUNG F full + RUNG A re-run
       └─ T203 Taylor-Green harness ──────────────► RUNG G ──────────────► M9
            └─ T204 flow/fidelity.py + autoconfig/diagnose/report under LES
                                                  ─► RUNG H ──────────────► M10
T205 packaging: pyproject, `fengdong` dist, entry points  (independent of T201-T204)
                                                  ─► RUNG I ──────────────► M11
  └─ T206 fengdong/widgets.py — the closed widget set, headless-tested
       └─ T207 fengdong/app.py — window, drop target, setup panel, explain()
            └─ T208 live view + results panel + refusal UI  (needs T204)
                 └─ T209 the drop: end to end ────► RUNG J ──────────────► M12
```

`T205` is independent of all closure work and is the fall-through if T201 or T202 stalls.
`T206` depends on T205 only for the package layout, not for any behaviour.

## Session map

| Session | Task | Deliverable at end of session | Milestone |
|---|---|---|---|
| 23 | — | Phase 2 planned: `IDEA4`, `PLAN3`, `TASKS3`, `STATE3`; `STATE2` frozen | |
| 24 | T201 | Smagorinsky in `lbm/core.py` + NumPy backend; **Rung F green on numpy**; nine existing rungs unmoved | |
| 25 | T202 | The closure on the Warp backend; **Rung F full**, Rung A re-run, LES perf table | |
| 26 | T203 | `validate/taylorgreen.py`; **Rung G passes** | **M9** |
| 27 | T204 | `flow/fidelity.py`, bands wired through autoconfig/diagnose/report; **Rung H passes** | **M10** |
| 28 | T205 | `pyproject.toml`, `fengdong` distribution, entry points; **Rung I passes** | **M11** |
| 29 | T206 | `fengdong/widgets.py` — the closed widget set, unit-tested headless | |
| 30 | T207 | `fengdong/app.py` — window, drop target, setup panel, plan preview | |
| 31 | T208 | Live view, numbers panel, save, refusal UI | |
| 32 | T209 | **Rung J passes** — a dropped picture to a correct moving answer, timed | **M12** |

Phase 2 ends at M12. Phase 3 (3D + STL, with the XLB swap as its kernel question) gets its own plan.

## Milestone gates

A milestone is claimed only when its gate command is **run** and prints pass. Every timing claim is
quoted with `Win32_Processor.CurrentClockSpeed`, the power state, and the GPU name (**D-035**).

| Milestone | Gate |
|---|---|
| **M9** | `myenv/Scripts/python.exe -m validate.les` **and** `myenv/Scripts/python.exe -m validate.les --backend warp` **and** `myenv/Scripts/python.exe -m validate.taylorgreen` **and** `myenv/Scripts/python.exe -m validate.taylorgreen --backend warp` all print PASS, **and** all nine existing rungs re-run and print their published digits with the closure off, **and** `bench.py --backend warp --les` clears ≥3116 / ≥568 / ≥331 steps/s at 40k / 1M / 2M cells |
| **M10** | `myenv/Scripts/python.exe -m validate.fidelity` prints PASS: every case in the sweep gets its band, no run outside the quantitative band emits an unqualified `Cd`, and **D-038**'s own case (air, 20 m/s, 1.5 m) runs to completion and reports `illustrative` |
| **M11** | `myenv/Scripts/python.exe -m validate.install` prints PASS: a wheel is built, installed into a **fresh venv** with no repo on the path, `fengdong --version` answers, and the headless app-model smoke runs — with the elapsed install-to-answer time printed |
| **M12** | `myenv/Scripts/python.exe -m validate.drop` prints PASS: a PNG delivered as a `pygame.DROPFILE` event reaches `St` 0.155–0.175 and `Cd` 1.25–1.45 through the application, with the elapsed time printed |

Rung F and Rung G share M9 because neither is meaningful alone: F says the closure changes nothing it
should not, G says it changes the right amount when it does.

## Risks and their pressure valves

| Risk | Signal | Valve |
|---|---|---|
| **The trap, wearing a closure** (`idea.md` § Risks) — Phase 2 becomes a turbulence-modelling project | Sessions 24–25 overrun; talk of KBC, MRT, dynamic `Cs`, wall models, better boundaries | **Hard valve: if T201 and T202 together overrun by one session, `Cs` freezes at the literature 0.17, no dynamic procedure is attempted, and the phase moves to T205** — which depends on none of it. The closure is one `collide` variant or it is out of scope. This is **D-043**'s valve in the same shape and for the same reason |
| **A plausible wrong answer at high Re** — the project's own stated main failure mode, now reachable on purpose | A banded run prints a `Cd` a reader could mistake for a validated one | **Constraint 18, and Rung H is a gate rather than a report**: no run outside the quantitative band may emit an unqualified `Cd`, asserted by inspecting `Result` and not by reading the wording. If the bands cannot be made falsifiable, the closure ships **stability-only** — it stabilises, and the tool refuses to report `Cd` at all outside the validated band |
| The closure silently moves a Phase 0 or Phase 1 rung | Any of the nine changes a digit | Rung F's bitwise `Cs = 0` clause, run **first** in T201 and re-run in every later task. The closure defaults **off** everywhere. A digit that moves with the closure off is a stop-work, not a tolerance discussion |
| Hand-rolled widgets swallow the phase | T206 grows a layout engine, theming, animation, focus chains | **The widget set is closed at the start of T206**: label, text field, dropdown, button, drop target. Nothing else. Anything more is `/new-task`, and the fall-back is a file dialog plus keyboard entry, which needs no widgets at all |
| The app becomes a second brain | A solver parameter is computed anywhere in `fengdong/` | **Constraint 17 plus a test**, the same one constraint 15 already has: `fengdong/` may import `flow/`, never the reverse. Every parameter comes from `flow.autoconfig.plan` |
| The window blocks the sim | Display frame rate drives step rate; steps dropped to keep the window smooth | Constraints 7 and 8, unchanged and re-asserted in T208. The app is a **fourth sink on the existing ring buffer**, never a new path. Dropped *display* frames are counted and shown |
| pygame drag-and-drop behaves differently off Windows | Rung I or J fails on another OS | Phase 2 claims **Windows only, in writing** (`DOCS/IDEA4.md` § Scope). Other platforms are best-effort and untested, and the README says so |
| Packaging drags in a dependency the solver did not need | `pyproject.toml` grows entries `myenv` never had | Every dependency in `pyproject.toml` must already be a row in `DOCS/STATE3.md` § Environment, added in the session that installed it. A new one is a decision, not a packaging detail |
| Phase 0 or Phase 1 code needs a change to make Phase 2 possible | A task wants to edit `lbm/` or `flow/` beyond what its contract names | `/new-task` against `DOCS/TASKS3.md` naming the rungs it must re-prove; never fold a solver change into an app task |

## What "done" means for Phase 2

Five more rungs green, fourteen in total. A closure that is bitwise invisible when off and honestly
labelled when on. A tool that answers the question Phase 1 had to refuse, and says exactly how much
that answer is worth. And a person who has never heard of a Reynolds number typing
`pip install fengdong`, dropping a picture on a window, and watching the flow.

Then Phase 2 closes, and 3D — with the XLB swap as its kernel question rather than its own phase —
gets designed against a product that has already survived real users.
