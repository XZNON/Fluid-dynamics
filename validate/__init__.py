"""Validation ladder — the integration tests for the solver.

``DOCS/IDEA2.md`` § "Validation ladder — do not skip a rung". Each module is a
rung, runnable as ``myenv/Scripts/python.exe -m validate.<rung>`` from the repo
root, and each prints PASS/FAIL with the measured numbers rather than asserting
silently. A wrong sim that looks plausible is this project's main failure mode
(``CLAUDE.md`` constraint 5).

| Rung | Module | Gate |
|---|---|---|
| 1 | ``validate.poiseuille`` | analytic parabola, L2 < 1% (T002, M1) |
| 2 | ``validate.cavity``     | Ghia et al. 1982 centrelines (T003, M2) |
| 3 | ``validate.cylinder``   | St ~ 0.164, Cd ~ 1.34 (T007, M3) |
| 4 | ``validate.polygons``   | square cylinder, Cd ~ 1.5 (T008) |

Rungs are ordered: do not start rung N+1 while rung N fails.
"""
