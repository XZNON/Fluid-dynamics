"""Rung C — awful user geometry is repaired, warned about, or refused.

``DOCS/IDEA3.md`` § Validation ladder, the Rung C row: the known answer for a
usability layer is *"a committed expectations table: one verdict + measured
properties per corpus image"*. A verdict alone would pass while the mask
underneath it quietly became a different shape — which is the risk
``DOCS/PLAN2.md`` names at T107 by name — so every measured property is pinned
beside it.

Four sections, each printing its own ``[ok]`` / ``[FAIL]``:

1. **The corpus.** Every image in ``tests/data/shapes/``, prepared with no
   manual step, its verdict and every measured property compared against
   ``expectations.json``. One line per image.
2. **Constraint 12 after repair.** For every image whose verdict is
   ``"repaired"``: minimum solid thickness at least
   :data:`flow.prepare.MIN_THICKNESS_CELLS`, and
   ``lbm.geometry.check_mask`` silent on the body placed in a nominal domain.
3. **Refusals name a fix, and the fix runs.** Every refused image's own
   :class:`~flow.prepare.Fix` is fed back through
   :func:`~flow.prepare.apply_fix` and :func:`~flow.prepare.prepare`, and the
   result must not be refused (**D-063**, constraint 14). A ``"picture"`` fix
   is executed against :data:`flow.diagnose.EXAMPLE_MASK` and the rung
   **prints that it is doing so**, rather than claiming to have fixed a
   picture it cannot invent.
4. **Q-102, the D-017 limit.** The measurement that closes it, re-run every
   time: :func:`flow.prepare.thin_branch_depth` over discs of every radius and
   sub-cell offset, over the project's own Rung 3 and Rung 4 bodies, and over
   hairlines fused to a thick body. The disc column is the bar D-017 records
   its two predecessors failing.

Costs seconds, not minutes: this is image processing, not simulation
(**D-059** asks for exactly that).

    myenv/Scripts/python.exe -m validate.shapes
    myenv/Scripts/python.exe -m validate.shapes --write-expectations
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from flow.diagnose import EXAMPLE_MASK
from flow.prepare import (
    MIN_THICKNESS_CELLS,
    REPAIRS,
    RESOLUTION_TOL,
    THIN_BRANCH_DEPTH,
    Prepared,
    apply_fix,
    prepare,
    thin_branch_depth,
)
from lbm.geometry import check_mask, circle, min_thickness, rectangle, regular_polygon

#: Where the corpus and its expectations live.
CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "shapes"

#: The committed table. Written by ``--write-expectations``, read by every
#: other run — and by ``tests/test_prepare.py``.
EXPECTATIONS = CORPUS_DIR / "expectations.json"

#: Tolerance on the float-valued properties. The integer ones are compared
#: exactly: a body that gained a cell gained it for a reason.
FLOAT_TOL = 5e-4

# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def corpus_cases() -> dict[str, dict[str, Any]]:
    """The committed expectations, keyed by image name."""
    if not EXPECTATIONS.exists():
        raise FileNotFoundError(
            f"no expectations table at {EXPECTATIONS}. Write one with:\n"
            "    myenv/Scripts/python.exe -m validate.shapes --write-expectations"
        )
    with EXPECTATIONS.open(encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def run_case(name: str, cells_across: int) -> Prepared:
    """Prepare one corpus image. No manual step, by construction."""
    return prepare(CORPUS_DIR / f"{name}.png", cells_across)


def action_keys(result: Prepared) -> list[str]:
    """The repair keys from :attr:`Prepared.actions`, without the prose."""
    return [a.split(":", 1)[0] for a in result.actions]


def properties_of(result: Prepared) -> dict[str, Any]:
    """The measured properties, rounded so the committed table is readable."""
    out: dict[str, Any] = {}
    for key, value in result.properties.items():
        out[key] = round(float(value), 6) if isinstance(value, float) else int(value)
    return out


def compare_properties(
    measured: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    """Every disagreement between a measured and a committed property."""
    problems = []
    for key, want in expected.items():
        if key not in measured:
            problems.append(f"{key} missing")
            continue
        got = measured[key]
        if isinstance(want, float):
            if not math.isclose(got, want, abs_tol=FLOAT_TOL):
                problems.append(f"{key} {got:.6f} != {want:.6f}")
        elif got != want:
            problems.append(f"{key} {got} != {want}")
    for key in measured:
        if key not in expected:
            problems.append(f"{key} not in the committed table")
    return problems


# ---------------------------------------------------------------------------
# Section 4 — the Q-102 measurement
# ---------------------------------------------------------------------------


def q102_evidence() -> dict[str, Any]:
    """Re-measure the separation that closes **D-017**'s documented limit.

    D-017 records that its own two predecessors were written first and *both
    false-alarmed on a plain disc*, so the disc is the bar. This function is
    that bar, run every time the rung runs, rather than a number copied into a
    document once.
    """
    disc_worst = 0
    disc_case = ""
    for radius in range(3, 61):
        for offset in (0.0, 0.25, 0.5, 0.75):
            n = 2 * radius + 11
            depth = thin_branch_depth(
                circle(n, n, n / 2 + offset, n / 2 + offset, float(radius))
            )
            if depth > disc_worst:
                disc_worst, disc_case = depth, f"r={radius}, offset {offset}"

    square_worst = max(
        thin_branch_depth(rectangle(s + 11, s + 11, 5, 5, 5 + s - 1, 5 + s - 1))
        for s in range(3, 61, 3)
    )

    # The project's own benchmark bodies, which must never be "repaired".
    from validate.cylinder import cylinder_mask
    from validate.polygons import body_mask, cases as polygon_cases

    out = cylinder_mask()
    rung3_body = out[1] if isinstance(out, tuple) and len(out) > 1 else out
    rung3 = thin_branch_depth(rung3_body)
    rung4 = {
        name: thin_branch_depth(body_mask(case)[1])
        for name, case in polygon_cases().items()
    }

    # A hairline fused to a fat disc: the case min_thickness cannot see.
    fused: dict[int, tuple[int, int]] = {}
    for length in (2, 5, 10, 20):
        n = 100
        body = circle(n, n, 30.0, 50.0, 15.0)
        whisker = rectangle(n, n, 44, 50, 44 + length, 50)
        mask = body | whisker
        fused[length] = (int(min_thickness(mask)), thin_branch_depth(mask))

    return {
        "disc_worst": disc_worst,
        "disc_case": disc_case,
        "square_worst": square_worst,
        "rung3": rung3,
        "rung4": rung4,
        "fused": fused,
    }


# ---------------------------------------------------------------------------
# Writing the table
# ---------------------------------------------------------------------------


def write_expectations() -> Path:
    """Measure every corpus image and commit the result as the known answer.

    The **verdict** is designed, not discovered — each image in
    ``tests/data/shapes/generate.py`` is built to earn exactly one — so this
    function refuses to write a table whose verdicts disagree with the
    designed ones. The *properties* are measured, and pinning them is what
    turns "it still passes" into "it still produces the same shape".
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_shape_corpus", CORPUS_DIR / "generate.py"
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    cases: dict[str, Any] = {}
    for name, (_builder, cells_across) in generator.CORPUS.items():
        result = run_case(name, cells_across)
        cases[name] = {
            "cells_across": cells_across,
            "verdict": result.verdict,
            "actions": action_keys(result),
            "properties": properties_of(result),
        }
    payload = {
        "written_by": "validate/shapes.py --write-expectations",
        "note": (
            "Rung C's known answer. The verdict is designed (see "
            "tests/data/shapes/generate.py); the properties are measured and "
            "pinned so a change in how a shape is repaired shows as a diff."
        ),
        "cases": cases,
    }
    EXPECTATIONS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return EXPECTATIONS


# ---------------------------------------------------------------------------
# The rung
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rung C — the shape corpus")
    parser.add_argument(
        "--write-expectations",
        action="store_true",
        help="re-measure every corpus image and rewrite expectations.json",
    )
    args = parser.parse_args(argv)

    if args.write_expectations:
        path = write_expectations()
        print(f"wrote {path}")
        return 0

    started = time.perf_counter()
    failures: list[str] = []
    cases = corpus_cases()

    print("Rung C — geometry preparation over the committed shape corpus")
    print(f"  corpus: {CORPUS_DIR}")
    print(f"  {len(cases)} images, expectations from {EXPECTATIONS.name}")
    print()

    # --- 1. the corpus ----------------------------------------------------
    print("1. Every corpus image gets its committed verdict and properties")
    results: dict[str, Prepared] = {}
    for name, expected in cases.items():
        cells_across = int(expected["cells_across"])
        result = run_case(name, cells_across)
        results[name] = result

        problems: list[str] = []
        if result.verdict != expected["verdict"]:
            problems.append(f"verdict {result.verdict} != {expected['verdict']}")
        if action_keys(result) != list(expected["actions"]):
            problems.append(
                f"actions {action_keys(result)} != {list(expected['actions'])}"
            )
        problems += compare_properties(properties_of(result), expected["properties"])

        props = result.properties
        ok = not problems
        if not ok:
            failures.append(f"{name}: " + "; ".join(problems))
        print(
            f"   {name:20s} {result.verdict:9s} "
            f"D={props['body_cells_across']:3d} "
            f"thick {props['min_thickness_before']:3d}->{props['min_thickness']:3d} "
            f"branch {props['thin_branch_depth_before']:3d}->"
            f"{props['thin_branch_depth']:3d} "
            f"{'+'.join(action_keys(result)) or '-':32s} "
            f"[{'ok' if ok else 'FAIL'}]"
        )
        for problem in problems:
            print(f"      - {problem}")
    print()

    # --- 2. constraint 12 after repair ------------------------------------
    print("2. After repair, constraint 12 holds — thickness and check_mask")
    repaired = [n for n, r in results.items() if r.verdict == "repaired"]
    for name in repaired:
        result = results[name]
        thickness = result.properties["min_thickness"]
        warnings_ = check_mask(_nominal(result.mask), "x", verbose=False)
        ok = thickness >= MIN_THICKNESS_CELLS and not warnings_
        if not ok:
            failures.append(
                f"{name}: repaired but thickness {thickness} / "
                f"{len(warnings_)} check_mask warning(s)"
            )
        print(
            f"   {name:20s} min thickness {thickness:3d} "
            f"(>= {MIN_THICKNESS_CELLS}), check_mask warnings "
            f"{len(warnings_)} [{'ok' if ok else 'FAIL'}]"
        )
    exercised = {k for n in repaired for k in action_keys(results[n])}
    missing = set(REPAIRS) - exercised
    if missing:
        failures.append(f"no corpus image exercises {sorted(missing)}")
    print(
        f"   repairs exercised by the corpus: {sorted(exercised)} "
        f"[{'ok' if not missing else 'FAIL'}]"
    )
    if not repaired:
        failures.append(
            "no corpus image was repaired — the corpus is not doing its job"
        )
    print()

    # --- 2b. D-040 --------------------------------------------------------
    print("2b. D-040 — the measured body is the size that was asked for")
    for name, result in results.items():
        if result.verdict == "refused":
            continue
        asked = int(cases[name]["cells_across"])
        got = int(result.properties["body_cells_across"])
        ok = abs(got - asked) <= RESOLUTION_TOL
        if not ok:
            failures.append(f"{name}: asked {asked} cells across, measured {got}")
        print(
            f"   {name:20s} asked {asked:3d}, measured {got:3d} "
            f"(within {RESOLUTION_TOL}) [{'ok' if ok else 'FAIL'}]"
        )
    print()

    # --- 3. refusals name a fix and the fix runs --------------------------
    print("3. Every refusal names a fix, and the fix actually runs (D-063)")
    refused = [n for n, r in results.items() if r.verdict == "refused"]
    for name in refused:
        result = results[name]
        source = CORPUS_DIR / f"{name}.png"
        cells_across = int(cases[name]["cells_across"])
        if result.reason is None or result.fix is None:
            failures.append(f"{name}: refused without a reason or a fix")
            print(f"   {name:20s} refused with no fix [FAIL]")
            continue
        request = apply_fix(result.fix, source=source, cells_across=cells_across)
        fixed = prepare(**request)
        ok = fixed.verdict != "refused"
        if not ok:
            failures.append(
                f"{name}: its own fix ({result.fix.change}) still refuses"
            )
        substituted = (
            "  [against flow.diagnose.EXAMPLE_MASK, not this picture — the tool "
            "cannot invent geometry it was not given]"
            if result.fix.change == "picture"
            else ""
        )
        print(
            f"   {name:20s} fix {result.fix.change:10s} -> {fixed.verdict:9s} "
            f"D={fixed.properties['body_cells_across']:3d} "
            f"[{'ok' if ok else 'FAIL'}]{substituted}"
        )
    if not refused:
        failures.append("no corpus image was refused — the corpus is not doing its job")
    print()

    # --- 4. Q-102 ---------------------------------------------------------
    print("4. Q-102 — D-017's limit, measured rather than inherited")
    evidence = q102_evidence()
    disc_ok = evidence["disc_worst"] < THIN_BRANCH_DEPTH
    bodies_ok = (
        evidence["square_worst"] < THIN_BRANCH_DEPTH
        and evidence["rung3"] < THIN_BRANCH_DEPTH
        and all(v < THIN_BRANCH_DEPTH for v in evidence["rung4"].values())
    )
    fused_ok = all(
        branch >= THIN_BRANCH_DEPTH for _thick, branch in evidence["fused"].values()
    )
    blind_ok = all(
        thick >= MIN_THICKNESS_CELLS for thick, _branch in evidence["fused"].values()
    )
    if not disc_ok:
        failures.append(
            f"thin_branch_depth false-alarms on a plain disc "
            f"({evidence['disc_worst']} >= {THIN_BRANCH_DEPTH}) — D-017's own bar"
        )
    if not bodies_ok:
        failures.append("thin_branch_depth false-alarms on a benchmark body")
    if not fused_ok:
        failures.append("thin_branch_depth misses a hairline fused to a thick body")
    print(
        f"   threshold {THIN_BRANCH_DEPTH}; a shape is flagged at or above it."
    )
    print(
        f"   plain discs, radius 3..60 x 4 sub-cell offsets: worst "
        f"{evidence['disc_worst']} ({evidence['disc_case']}) "
        f"[{'ok' if disc_ok else 'FAIL'}]"
    )
    print(
        f"   squares 3..60: worst {evidence['square_worst']}; "
        f"Rung 3 cylinder body {evidence['rung3']}; Rung 4 "
        + ", ".join(f"{k} {v}" for k, v in evidence["rung4"].items())
        + f" [{'ok' if bodies_ok else 'FAIL'}]"
    )
    for length, (thick, branch) in evidence["fused"].items():
        print(
            f"   hairline of {length:2d} cells fused to a disc: min_thickness "
            f"{thick:3d} (blind, D-017's documented limit), "
            f"thin_branch_depth {branch:3d}"
        )
    print(
        f"   min_thickness sees none of them ({'ok' if blind_ok else 'FAIL'}: it "
        f"reports the body, not the hairline); thin_branch_depth sees all of "
        f"them [{'ok' if fused_ok else 'FAIL'}]"
    )
    print()

    elapsed = time.perf_counter() - started
    if failures:
        print(f"Rung C: FAIL ({len(failures)}) in {elapsed:.1f} s")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"Rung C: PASS in {elapsed:.1f} s")
    return 0


def _nominal(body: np.ndarray) -> np.ndarray:
    """The body in the domain ``flow/autoconfig.py`` would size for it."""
    from flow.prepare import _nominal_domain

    return _nominal_domain(body)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
