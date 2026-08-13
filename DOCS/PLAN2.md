# PLAN2.md — Phase 1 implementation plan

Implementation plan for `DOCS/IDEA3.md`: the product layer above the Phase 0 solver, on a GPU
kernel.

**Plan shape:** 10 tasks, T101 → T110. **One task per session.** Task contracts and acceptance
criteria live in `DOCS/TASKS2.md`; live status lives in `DOCS/STATE2.md`. Phase 0's equivalents
(`old-Docs/PLAN1.md`, `old-Docs/TASKS1.md`, `old-Docs/STATE1.md`) are closed and frozen — read them for history,
never edit them.

---

## Why this order

1. **The backend seam before the backend.** T101 introduces the seam with only the NumPy backend
   behind it, so the day Warp arrives the only new thing is Warp. A seam invented *during* a port is
   a seam shaped by one implementation.
2. **The port before the product layer.** Phase 1's headline acceptance criterion is a wall clock
   (`DOCS/IDEA3.md` § Goal), and the NumPy kernel cannot meet it — the M4 gate took 335 s. Building
   the product on a backend that is about to be replaced would mean measuring every product-level
   timing twice. **This is the one deviation from `idea.md`'s roadmap ordering and it is recorded as
   D-043**, with a pressure valve below.
3. **Judgement before convenience.** `autoconfig` (T105) and `diagnose` (T106) come before the
   `Case` API (T108) that wraps them. The API is the easy part; what it decides on the user's behalf
   is the product.
4. **Every rung's harness lands in the task that needs it, before the code it validates**
   (constraint 5's spirit). No task both writes a capability and invents its pass condition
   afterwards.
5. **Every session ends on a green boundary** — a passing rung, a passing unit-test set, or an
   explicit "half-done, here is what is missing" in `DOCS/STATE2.md`. Never on ambiguity.

## Dependency graph

```
T101 backend seam, NumPy behind it  (all Phase 0 rungs still green)
  └─ T102 Warp kernel: equilibrium, collide, stream ──► RUNG A parity (kernels)
       └─ T103 Warp boundaries, checkpoint, perf ─────► RUNG A full ──► M5
            ├─ T104 flow/quantity.py + flow/fluids.py  (unit parsing, fluid library)
            │    └─ T105 flow/autoconfig.py ──────────► RUNG B ──► M6
            │         └─ T106 flow/diagnose.py ───────► RUNG D refusals
            ├─ T107 flow/prepare.py + shape corpus ───► RUNG C ──► M7
            └─ T108 flow/case.py + flow/report.py      (needs T105, T106, T107)
                 └─ T109 CLI on flow, live + record wiring
                      └─ T110 the minute: end to end ─► RUNG E ──► M8
```

`T104` is independent of the GPU work and is the natural place to go if T102/T103 stall.
`T107` is independent of `T104`–`T106` and can be reordered with them.

## Session map

| Session | Task | Deliverable at end of session | Milestone |
|---|---|---|---|
| 12 | — | Phase 1 planned: `IDEA3`, `PLAN2`, `TASKS2`, `STATE2` | |
| 13 | T101 | `Backend` protocol; `lbm.backends.numpy_backend`; all four Phase 0 rungs green through it | |
| 14 | T102 | Warp `equilibrium` / `collide` / `stream`; **Rung A (kernels) passes** | |
| 15 | T103 | Warp boundaries + checkpoint + perf table; **Rung A (full) passes** | **M5** |
| 16 | T104 | `flow/quantity.py`, `flow/fluids.py`, unit tests green | |
| 17 | T105 | `flow/autoconfig.py`; **Rung B passes** | **M6** |
| 18 | T106 | `flow/diagnose.py`; **Rung D passes** | |
| 19 | T107 | `flow/prepare.py` + committed corpus; **Rung C passes** | **M7** |
| 20 | T108 | `flow.Case` / `flow.Result`, `explain()`, `report()` | |
| 21 | T109 | `python -m flow` CLI, live + record through `flow` | |
| 22 | T110 | **Rung E passes** — cold shell to correct answer, timed | **M8** |

Phase 1 ends at M8. Phase 2 (XLB swap, per `idea.md` Phase 3) and a UI get their own plan.

## Milestone gates

A milestone is claimed only when its gate command is **run** and printed pass. Every timing claim is
quoted with `Win32_Processor.CurrentClockSpeed`, the power state, and the GPU name (**D-035**).

| Milestone | Gate |
|---|---|
| **M5** | `myenv/Scripts/python.exe -m validate.parity --backend warp` prints PASS, **and** all four Phase 0 rungs re-run with `--backend warp` print PASS inside their published bands, **and** `bench.py --backend warp` clears ≥2000 / ≥250 / ≥150 steps/s at 40k / 1M / 2M cells |
| **M6** | `myenv/Scripts/python.exe -m validate.autoconfig` prints PASS: every case in the sweep obeys every guardrail, runs 5000 steps without `nan`, and reproduces its requested `Re` to 0.1% |
| **M7** | `myenv/Scripts/python.exe -m validate.shapes` prints PASS: every corpus image gets its committed verdict and its measured properties, with no manual step |
| **M8** | `myenv/Scripts/python.exe -m validate.minute --backend warp` prints PASS: from a cold shell, a PNG of a disc plus physical numbers reaches `St` 0.155–0.175 and `Cd` 1.25–1.45 through `flow.Case`, in **under 60 s wall clock**, with the elapsed time printed |

Rung D has no milestone of its own; it is a gate inside T106 and is re-run by M8.

## Risks and their pressure valves

| Risk | Signal | Valve |
|---|---|---|
| **The trap** — Phase 1 becomes a solver-optimisation project, which is what pulling M5 in makes easiest (`idea.md` § Risks) | Sessions 14–15 overrun; talk of MRT, better boundaries, kernel micro-tuning | **Hard valve: if T102 or T103 overruns by one session, the port is demoted back to Phase 2 and Phase 1 continues on NumPy** with M8's wall clock restated against the NumPy backend and the shortfall recorded honestly. The seam from T101 makes this a config change, not a rewrite. |
| Warp is painful on Windows / RTX 3050 (driver, CUDA, install) | T102 cannot get a kernel to run in the first half of its session | Fall through to T104 (independent of the GPU work) the same session, log the blocker, and try Taichi in the next GPU session before spending a third one |
| GPU results differ from NumPy and the difference is *not* obviously float ordering | Rung A parity fails at a tolerance no reordering explains | The GPU backend is wrong until proven otherwise; never adjust the published band to fit it. Bisect by kernel — Rung A checks `equilibrium`, `collide`, `stream` and the boundaries separately for exactly this |
| Auto-config becomes a pile of tuned constants nobody can defend | T105 grows magic numbers with no measurement behind them | Every constant in `autoconfig` cites a Phase 0 decision (D-019, D-029, D-032, D-036, D-040) or gets measured in the session that adds it and recorded in `DOCS/STATE2.md` § Decisions |
| Shape repair silently changes the user's geometry | T107 "fixes" a shape into a different shape | Repair is opt-in per class of defect, always reported in `explain()`, and Rung C asserts the *measured* properties of the repaired mask — not just that it stopped warning |
| Scope creep toward a UI, 3D, STL, or XLB | Any of them named in a session's work | All four are in `DOCS/IDEA3.md` § Deliberately deferred by name. Phase 1 ends at M8 |
| Phase 0 code needs a change to make Phase 1 possible | A Phase 1 task wants to edit `lbm/` beyond the backend seam | `/new-task` against `DOCS/TASKS2.md` with the Phase 0 rung it must re-prove; never fold a solver change into a product task |

## What "done" means for Phase 1

Five rungs green. A GPU backend that agrees with NumPy and clears the budget. A user who types a
picture, a fluid, a speed and a size — and nothing else — gets either a correct moving answer with
its numbers, or a refusal that names a case that works and is tested to actually work. Then Phase 1
closes and the XLB swap gets designed, with a product layer that has already survived real shapes.
