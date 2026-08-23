"""T107 — ``flow/prepare.py``: geometry preparation, repair, refusal.

``DOCS/TASKS2.md`` § T107. The corpus-wide assertions live in
``validate/shapes.py`` (Rung C, the integration test that prints PASS/FAIL);
what is here is the unit-level behaviour underneath it, plus the two tests with
teeth that the acceptance criteria turn on:

* :func:`test_min_thickness_is_blind_to_a_fused_hairline_and_thin_branch_depth_is_not`
  — **Q-102**, stated as a test rather than as a claim.
* :func:`test_thin_branch_depth_does_not_false_alarm_on_a_plain_disc` — the bar
  **D-017** records killing its own two predecessors.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import flow
import flow.prepare
from flow.diagnose import EXAMPLE_MASK
from flow.prepare import (
    MIN_BODY_CELLS,
    MIN_THICKNESS_CELLS,
    REPAIRS,
    THIN_BRANCH_DEPTH,
    VERDICTS,
    Fix,
    Prepared,
    apply_fix,
    measure,
    prepare,
    thin_branch_depth,
)
from lbm import geometry
from lbm.geometry import check_mask, circle, min_thickness, rectangle

CORPUS = Path(__file__).resolve().parent / "data" / "shapes"

#: The **module**, not the function. ``flow/__init__.py`` re-exports
#: ``prepare`` the callable, which shadows ``prepare`` the submodule on the
#: package — deliberately, because ``flow.prepare(picture, 40)`` is the API the
#: product wants. Reaching the module therefore goes through ``sys.modules``.
prep = sys.modules["flow.prepare"]


# ---------------------------------------------------------------------------
# Fixtures — shapes built here rather than loaded, so a test names its defect
# ---------------------------------------------------------------------------


def fused_hairline(length: int = 20, width: int = 1) -> np.ndarray:
    """A fat disc with a ``width``-cell whisker welded to it.

    The whisker shares the disc's 8-connected component. That is the entire
    content of D-017's documented limit.
    """
    body = circle(100, 100, 30.0, 50.0, 15.0)
    y0 = 50 - width // 2
    whisker = rectangle(100, 100, 44, y0, 44 + length, y0 + width - 1)
    return body | whisker


def donut(hole: int = 8) -> np.ndarray:
    return circle(80, 80, 40.0, 40.0, 25.0) & ~circle(80, 80, 40.0, 40.0, float(hole))


def disc_with_specks() -> np.ndarray:
    mask = circle(240, 240, 120.0, 120.0, 60.0)
    mask[10:16, 10:16] = True
    mask[224:229, 20:25] = True
    mask[30:35, 220:225] = True
    return mask


# ---------------------------------------------------------------------------
# Q-102 — the measurement the acceptance criterion asks for
# ---------------------------------------------------------------------------


def test_min_thickness_is_blind_to_a_fused_hairline_and_thin_branch_depth_is_not():
    """**Q-102**, and the reason this module has a new metric at all.

    D-017: *"a thin appendage fused to a thick body shares its component and is
    not reported"*. So ``min_thickness`` returns the **disc's** depth — it is
    not merely imprecise here, it is measuring something else entirely — while
    :func:`~flow.prepare.thin_branch_depth` grows with the hairline's length.
    """
    for length in (2, 5, 10, 20):
        mask = fused_hairline(length)
        assert min_thickness(mask) >= MIN_THICKNESS_CELLS, (
            "the premise of this test is that min_thickness sees nothing wrong"
        )
        assert thin_branch_depth(mask) >= THIN_BRANCH_DEPTH
    assert thin_branch_depth(fused_hairline(20)) > thin_branch_depth(
        fused_hairline(5)
    ), "a longer hairline must read deeper, or the metric is not measuring length"


def test_thin_branch_depth_does_not_false_alarm_on_a_plain_disc():
    """The bar D-017 records both of its predecessors failing.

    Run lengths and per-cell 3x3 opening each warn on a healthy cylinder, which
    would have made every Rung 3 run warn. Any replacement has to clear the
    same bar at every radius **and** every sub-cell offset — a half-cell offset
    is what ``validate/cylinder.py`` uses to break mirror symmetry, so it is
    not an exotic case.
    """
    worst = 0
    for radius in range(3, 41):
        for offset in (0.0, 0.25, 0.5, 0.75):
            n = 2 * radius + 11
            worst = max(
                worst,
                thin_branch_depth(
                    circle(n, n, n / 2 + offset, n / 2 + offset, float(radius))
                ),
            )
    assert worst < THIN_BRANCH_DEPTH, (
        f"thin_branch_depth reads {worst} on a plain disc against a threshold "
        f"of {THIN_BRANCH_DEPTH} — D-017's bar, and the one that kills candidates"
    )


def test_thin_branch_depth_does_not_false_alarm_on_the_projects_own_bodies():
    """Rung 3's cylinder and Rung 4's two bodies must never be "repaired"."""
    from validate.cylinder import cylinder_mask
    from validate.polygons import body_mask, cases

    out = cylinder_mask()
    body = out[1] if isinstance(out, tuple) and len(out) > 1 else out
    assert thin_branch_depth(body) < THIN_BRANCH_DEPTH
    for case in cases().values():
        assert thin_branch_depth(body_mask(case)[1]) < THIN_BRANCH_DEPTH


@pytest.mark.parametrize("shape", [(10, 10), (0, 0), (1, 1), (3, 0)])
def test_thin_branch_depth_is_zero_on_an_empty_mask(shape):
    """``(0, 0)`` is not a theoretical shape — it is what a refusal carries."""
    assert thin_branch_depth(np.zeros(shape, dtype=bool)) == 0


def test_a_fused_hairline_is_detected_through_prepare():
    """The metric is not merely available — ``prepare`` acts on it."""
    result = prepare(fused_hairline(20), 40)
    assert result.properties["thin_branch_depth_before"] >= THIN_BRANCH_DEPTH
    assert result.verdict == "repaired"
    assert any(a.startswith("thicken") for a in result.actions)
    assert result.properties["thin_branch_depth"] < THIN_BRANCH_DEPTH


# ---------------------------------------------------------------------------
# Repairs are individually switchable and individually reported
# ---------------------------------------------------------------------------


def test_repair_false_changes_nothing_and_reports_nothing():
    result = prepare(donut(), 40, repair=False)
    assert result.actions == []
    assert result.properties["holes"] == result.properties["holes_before"]


def test_each_repair_can_be_asked_for_on_its_own():
    result = prepare(donut(), 40, repair=["fill_holes"])
    assert [a.split(":")[0] for a in result.actions] == ["fill_holes"]
    assert result.properties["holes"] == 0


def test_a_repair_that_was_not_asked_for_does_not_happen():
    """Specks survive when only ``fill_holes`` is enabled."""
    result = prepare(disc_with_specks(), 40, repair=["fill_holes"])
    assert result.properties["components"] > 1
    assert not any(a.startswith("drop_specks") for a in result.actions)


def test_repairs_run_in_the_declared_order_whatever_order_they_arrive_in():
    a = prepare(disc_with_specks(), 40, repair=["thicken", "drop_specks"])
    b = prepare(disc_with_specks(), 40, repair=["drop_specks", "thicken"])
    assert [x.split(":")[0] for x in a.actions] == [
        x.split(":")[0] for x in b.actions
    ]


def test_an_unknown_repair_is_loud():
    with pytest.raises(ValueError, match="unknown repair"):
        prepare(donut(), 40, repair=["polish"])


def test_every_action_names_its_repair_and_every_repair_is_reachable():
    """Constraint 16: nothing is repaired silently, and the disclosure is keyed."""
    seen = set()
    for source in (donut(), disc_with_specks(), fused_hairline(20)):
        for action in prepare(source, 40).actions:
            key = action.split(":", 1)[0]
            assert key in REPAIRS, f"action {action!r} names no known repair"
            assert action[len(key) :].startswith(": "), (
                "an action must read 'key: what happened'"
            )
            seen.add(key)
    two = np.zeros((60, 120), dtype=bool)
    two[20:50, 10:40] = True
    two[25:35, 70:100] = True
    two[26, 50] = True  # a stray, so 'largest_component' has something to do
    for action in prepare(two, 30, repair=["largest_component"]).actions:
        seen.add(action.split(":", 1)[0])
    assert seen == set(REPAIRS), f"never exercised: {set(REPAIRS) - seen}"


def test_the_speck_repair_states_the_area_it_dropped():
    """"Drop specks below a stated area" — the area has to be *stated*."""
    result = prepare(disc_with_specks(), 40, repair=["drop_specks"])
    action = next(a for a in result.actions if a.startswith("drop_specks"))
    assert "cells or fewer" in action


def test_a_healthy_disc_is_never_touched():
    result = prepare(circle(200, 200, 100.0, 100.0, 70.0), 40)
    assert result.verdict == "ok"
    assert result.actions == []


# ---------------------------------------------------------------------------
# After repair, constraint 12 holds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [donut(), disc_with_specks(), fused_hairline(20)],
    ids=["donut", "specks", "hairline"],
)
def test_a_repaired_mask_satisfies_constraint_12(source):
    result = prepare(source, 40)
    assert result.verdict in ("ok", "repaired")
    assert result.properties["min_thickness"] >= MIN_THICKNESS_CELLS
    assert check_mask(prep._nominal_domain(result.mask), "x", verbose=False) == []


def test_repair_does_not_turn_a_shape_into_a_different_shape():
    """``DOCS/PLAN2.md`` § Risks, the row aimed at this task.

    Thickening a hairline may not quietly inflate the body it hangs off: the
    repaired disc keeps its area to within the couple of hundred cells the
    whisker itself accounts for.
    """
    plain = prepare(circle(100, 100, 30.0, 50.0, 15.0), 40)
    whiskered = prepare(fused_hairline(20), 40)
    grew = whiskered.properties["solid_cells"] - plain.properties["solid_cells"]
    assert 0 < grew < 0.5 * plain.properties["solid_cells"], (
        f"the body grew by {grew} cells, which is a different shape, not a repair"
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_all_white_picture_is_refused_with_a_reason_and_a_fix():
    result = prepare(np.zeros((80, 80), dtype=bool), 40)
    assert result.verdict == "refused"
    assert result.reason and "no shape" in result.reason
    assert result.fix is not None and result.fix.change == "picture"
    assert not result.runnable


def test_an_all_black_picture_is_refused_with_a_reason_and_a_fix():
    result = prepare(np.ones((80, 80), dtype=bool), 40)
    assert result.verdict == "refused"
    assert result.reason and "no fluid" in result.reason
    assert result.fix is not None and result.fix.change == "picture"


def test_a_body_under_eight_cells_across_is_refused():
    result = prepare(circle(400, 400, 200.0, 200.0, 5.0), MIN_BODY_CELLS - 2)
    assert result.verdict == "refused"
    assert result.reason and str(MIN_BODY_CELLS) in result.reason
    assert result.fix is not None and result.fix.change == "resolution"
    assert int(result.fix.value) >= MIN_BODY_CELLS


def test_a_refusal_still_reports_what_it_measured():
    """A refusal that cannot say what it measured is not diagnosable."""
    result = prepare(circle(400, 400, 200.0, 200.0, 5.0), 4)
    assert result.properties["body_cells_across"] > 0


def test_every_refusal_names_a_fix_that_actually_fixes_it():
    """Constraint 14 / **D-063**: a fix is a testable claim, so test it."""
    cases = [
        (np.zeros((80, 80), dtype=bool), 40),
        (np.ones((80, 80), dtype=bool), 40),
        (circle(400, 400, 200.0, 200.0, 5.0), 5),
    ]
    for source, cells_across in cases:
        refused = prepare(source, cells_across)
        assert refused.verdict == "refused" and refused.fix is not None
        fixed = prepare(
            **apply_fix(refused.fix, source=source, cells_across=cells_across)
        )
        assert fixed.verdict != "refused", (
            f"the fix for {refused.reason!r} does not fix its own case"
        )


def test_apply_fix_substitutes_the_worked_example_for_a_picture_fix():
    fix = Fix(change="picture", value="a picture with a shape in it", note="n/a")
    request = apply_fix(fix, source=np.zeros((4, 4), dtype=bool), cells_across=40)
    assert request["source"] is EXAMPLE_MASK


def test_apply_fix_is_loud_about_a_change_it_cannot_apply():
    with pytest.raises(ValueError, match="cannot apply a fix"):
        apply_fix(
            Fix(change="vibes", value=1, note="n/a"),
            source=np.zeros((4, 4), dtype=bool),
            cells_across=40,
        )


# ---------------------------------------------------------------------------
# D-040 — the measured body is the requested size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cells_across", [10, 16, 24, 40, 55])
def test_the_measured_body_is_the_requested_size(cells_across):
    result = prepare(circle(200, 200, 100.0, 100.0, 70.0), cells_across)
    assert abs(result.properties["body_cells_across"] - cells_across) <= 1


def test_a_huge_margin_does_not_shrink_the_body():
    """**D-040** in one line: the number is cells across the *body*."""
    result = prepare(circle(400, 400, 200.0, 200.0, 25.0), 40)
    assert abs(result.properties["body_cells_across"] - 40) <= 1


def test_a_repair_that_changes_the_extent_re_derives_rather_than_adjusts():
    """A plate broadside to the flow grows when thickened, and still lands.

    The repair adds a cell either side, so the raster it was measured in is no
    longer the raster that produces the requested body — which is why the
    resolution is re-derived around the *repaired* extent rather than patched
    afterwards.
    """
    plate = np.zeros((200, 200), dtype=bool)
    plate[20:180, 98:102] = True
    result = prepare(plate, 40)
    assert result.verdict == "repaired"
    assert any(a.startswith("thicken") for a in result.actions)
    assert result.properties["min_thickness"] >= MIN_THICKNESS_CELLS
    assert abs(result.properties["body_cells_across"] - 40) <= 1


def test_a_resolution_the_picture_cannot_deliver_is_refused_not_substituted():
    """Constraint 16, at the one place it would be easiest to break quietly.

    A plate two pixels wide in a two-hundred-pixel picture disappears at any
    raster fine enough to also make the body forty cells across. Returning a
    sixty-cell body instead would be a silent substitution — and ``tau`` is
    computed from the number that comes back (**D-040**), so it would be a
    silent substitution of the *physics*, not just of the picture.
    """
    plate = np.zeros((200, 200), dtype=bool)
    plate[20:180, 99:101] = True
    result = prepare(plate, 40)
    assert result.verdict == "refused"
    assert result.fix is not None and result.fix.change == "resolution"
    fixed = prepare(**apply_fix(result.fix, source=plate, cells_across=40))
    assert fixed.verdict != "refused"
    assert abs(fixed.properties["body_cells_across"] - int(result.fix.value)) <= 1


# ---------------------------------------------------------------------------
# Inputs, outputs and the package rules
# ---------------------------------------------------------------------------


def test_prepare_accepts_a_png_path():
    result = prepare(CORPUS / "disc.png", 32)
    assert result.verdict == "ok"
    assert result.mask.dtype == np.bool_ and result.mask.ndim == 2


def test_prepare_accepts_an_svg_path(tmp_path):
    svg = tmp_path / "square.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<path d="M20,20 L80,20 L80,80 L20,80 Z"/></svg>',
        encoding="utf-8",
    )
    result = prepare(svg, 24)
    assert result.verdict in ("ok", "repaired")
    assert abs(result.properties["body_cells_across"] - 24) <= 1


def test_a_missing_file_is_a_missing_file_not_a_refusal():
    with pytest.raises(FileNotFoundError):
        prepare(CORPUS / "no_such_shape.png", 40)


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4, 4), dtype=bool), np.zeros((4, 4), dtype=np.float32)]
)
def test_a_bad_source_array_raises_rather_than_refusing(bad):
    with pytest.raises(ValueError):
        prepare(bad, 40)


def test_cells_across_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        prepare(circle(50, 50, 25.0, 25.0, 10.0), 0)


def test_the_verdict_is_always_one_of_three():
    for source in (
        circle(200, 200, 100.0, 100.0, 70.0),
        donut(),
        np.zeros((40, 40), dtype=bool),
    ):
        assert prepare(source, 40).verdict in VERDICTS


def test_prepared_is_a_frozen_output_record():
    """**D-060**: the constraint-13 exemption only covers frozen dataclasses."""
    import dataclasses

    assert dataclasses.is_dataclass(Prepared)
    assert Prepared.__dataclass_params__.frozen
    assert dataclasses.is_dataclass(Fix) and Fix.__dataclass_params__.frozen


def test_measure_reports_zeros_for_an_empty_mask():
    props = measure(np.zeros((10, 10), dtype=bool))
    assert props["body_cells_across"] == 0 and props["solid_cells"] == 0


def test_prepare_builds_on_lbm_geometry_rather_than_forking_it():
    """Constraint 15 / the T107 contract: ``check_mask`` stays in ``lbm/``."""
    assert prep.min_thickness is geometry.min_thickness
    assert prep.check_mask is geometry.check_mask
    source = Path(prep.__file__).read_text(encoding="utf-8")
    assert "def check_mask" not in source and "def min_thickness" not in source


def test_the_package_exports_the_new_surface():
    for name in ("prepare", "Prepared", "Fix", "thin_branch_depth"):
        assert hasattr(flow, name), f"flow.{name} is not exported"
        assert name in flow.__all__


def test_verbose_prints_the_verdict_and_every_action(capsys):
    prepare(donut(), 40, verbose=True)
    printed = capsys.readouterr().out
    assert "verdict repaired" in printed and "fill_holes" in printed


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------


def _generator():
    spec = importlib.util.spec_from_file_location(
        "_shape_corpus_test", CORPUS / "generate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_corpus_covers_every_defect_the_contract_names():
    """The twelve defects of the T107 acceptance criterion, by name."""
    required = {
        "hairline_appendage",
        "specks",
        "donut",
        "unclosed_outline",
        "antialiased",
        "huge_margin",
        "tiny_body",
        "extreme_aspect",
        "diagonal_line",
        "self_touching",
        "two_bodies",
        "all_white",
        "all_black",
    }
    names = set(_generator().CORPUS)
    assert required <= names, f"the corpus is missing {required - names}"
    assert len(names) >= 12
    for name in names:
        assert (CORPUS / f"{name}.png").exists(), f"{name}.png is not committed"


def test_the_committed_images_are_the_ones_the_generator_produces():
    """A committed binary is only auditable if it matches its generator."""
    from PIL import Image

    generator = _generator()
    for name, (builder, _cells) in generator.CORPUS.items():
        with Image.open(CORPUS / f"{name}.png") as committed:
            on_disk = np.asarray(committed.convert("L"))
        assert np.array_equal(on_disk, np.asarray(builder().convert("L"))), (
            f"{name}.png differs from tests/data/shapes/generate.py — "
            "regenerate it rather than editing the PNG"
        )


def test_every_corpus_image_has_a_committed_verdict_and_properties():
    with (CORPUS / "expectations.json").open(encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    assert set(cases) == set(_generator().CORPUS)
    for name, case in cases.items():
        assert case["verdict"] in VERDICTS
        assert "cells_across" in case
        for key in ("body_cells_across", "min_thickness", "thin_branch_depth"):
            assert key in case["properties"], f"{name} pins no {key}"


def test_the_corpus_contains_at_least_one_of_each_verdict():
    with (CORPUS / "expectations.json").open(encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    verdicts = {case["verdict"] for case in cases.values()}
    assert verdicts == set(VERDICTS), (
        f"the corpus never produces {set(VERDICTS) - verdicts}"
    )
