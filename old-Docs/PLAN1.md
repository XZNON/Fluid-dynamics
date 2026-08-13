# PLAN1.md — Phase 0 implementation plan

Implementation plan for `DOCS/IDEA2.md`: a validated, continuously-running D2Q9 lattice-Boltzmann
simulator in pure NumPy.

**Plan shape:** 11 tasks, T001 → T011. **One task per session.** Task contracts and acceptance
criteria live in `old-Docs/TASKS1.md`; live status lives in `old-Docs/STATE1.md`.

---

## Why this order

The plan is the validation ladder with plumbing hung off it. Three rules set the order:

1. **Correctness before capability.** Rung 1 (Poiseuille) catches every sign error in `collide`.
   Rung 2 (cavity vs Ghia) catches boundary-condition errors. Neither needs rendering, geometry, or
   units — so neither gets built first. A pretty wrong sim is the failure mode we're guarding against.
2. **Nothing is optimised before Rung 3.** Performance work (T010) sits after the cylinder produces
   the right Strouhal number, because a fused kernel that's subtly wrong is very hard to unfuse.
3. **Every session ends on a green boundary.** A session closes with either a passing rung, a
   passing unit-test set, or an explicit "this is half-done and here is what's missing" in
   `old-Docs/STATE1.md`. Never on ambiguity.

## Dependency graph

```
T001 core constants + equilibrium  (unit tests)
  └─ T002 collide + stream + walls + body force ──► RUNG 1 Poiseuille ──► M1
       └─ T003 moving-lid BC ──────────────────────► RUNG 2 Cavity/Ghia ──► M2
            ├─ T004 geometry primitives + mask sanity
            │    └─ T005 inlet/outlet + probes (vorticity, Cd, Cl, Strouhal)
            │         └─ T006 runner: decoupled loop, ring buffer, restart
            │              └─ T007 render + live sink ──► RUNG 3 Cylinder ──► M3
            │                   ├─ T008 ──────────────► RUNG 4 Square cyl
            │                   ├─ T009 units + PNG/SVG mask
            │                   │    └─ T011 CLI + record sinks ──────────► M4
            │                   └─ T010 performance pass (gated on Rung 3)
```

`T008`, `T009`, `T010` are independent of each other and can be reordered if a session gets blocked.
`T011` needs `T009`.

## Session map

| Session | Task | Deliverable at end of session | Milestone |
|---|---|---|---|
| 1 | T001 | `lbm/core.py` constants + `macroscopic()` + `equilibrium()`, unit tests green | |
| 2 | T002 | `collide` + `stream` + bounce-back + body force; **Rung 1 passes** | **M1** |
| 3 | T003 | Moving-lid BC; **Rung 2 passes at Re 100 / 400 / 1000** | **M2** |
| 4 | T004 | `lbm/geometry.py` primitives + the three mask sanity checks | |
| 5 | T005 | Velocity inlet, zero-gradient outlet, `lbm/probe.py` (vorticity, Cd, Cl, Strouhal) | |
| 6 | T006 | `lbm/runner.py` — decoupled loop, derived `steps_per_frame`, ring buffer, bit-identical restart | |
| 7 | T007 | `lbm/render.py` + live pygame sink; **Rung 3 passes** (St ≈ 0.164, Cd ≈ 1.34) | **M3** |
| 8 | T008 | **Rung 4 passes** — square cylinder Re 100, Cd ≈ 1.5 | |
| 9 | T009 | `lbm/units.py` physical↔lattice + PNG/SVG mask loading | |
| 10 | T010 | Performance pass against the budget table, all rungs still green | |
| 11 | T011 | `lbm/record.py` MP4/GIF + headless sink + CLI entry point | **M4** |

**M5** (Warp/Taichi port of the kernel, same API) is explicitly out of Phase 0. It gets its own plan
after M4 ships.

## Milestone gates

A milestone is claimed only when its gate command is run and printed pass.

| Milestone | Gate |
|---|---|
| M1 | `python -m validate.poiseuille` prints PASS, L2 error < 1%, and the halving-`(tau-0.5)` doubling check passes |
| M2 | `python -m validate.cavity --re 100 --re 400 --re 1000` prints PASS against Ghia et al. (1982) |
| M3 | `python -m validate.cylinder` prints PASS with St in 0.155–0.175 and Cd in 1.25–1.45, and the live window runs without stuttering the physics |
| M4 | An arbitrary PNG becomes a mask, runs in physical units, and records an MP4 — end to end, one command |

## Risks and their pressure valves

| Risk | Signal | Valve |
|---|---|---|
| Rung 2 stalls on Ghia mismatch (BC subtlety, not a bug in collide) | Rung 1 green, cavity centreline off by >5% | Timebox to one session. If it fails twice, log it in STATE1 § Decisions, try Zou–He walls, and only then consider proceeding to T004 with Rung 2 flagged red — never silently. |
| Cylinder shows no shedding | Steady wake at Re 100 | Almost always insufficient upstream/downstream space or blockage ratio; T004's sanity checks exist to catch it before the run |
| Live display drags the physics down | steps/s drops when window opens | The ring buffer in T006 is the fix and it is built before rendering exists, deliberately |
| Performance pass breaks correctness | A rung goes red after T010 | T010's acceptance criteria include re-running **all** rungs; revert rather than debug a fused kernel |
| Scope creep toward the real product (`idea.md`) | Talk of STL, 3D, GPU, UI | All deferred by name in `DOCS/IDEA2.md` § Deliberately deferred. Phase 0 ends at M4. |

## What "done" means for Phase 0

All four rungs green, a live window showing a vortex street, an arbitrary PNG driving the geometry,
an MP4 on disk, and a restart that continues bit-identically. Then Phase 0 closes and the product
layer above it gets designed — with real understanding of the method, which was the entire point.
