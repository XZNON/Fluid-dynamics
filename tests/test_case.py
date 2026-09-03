"""T108 — ``flow.Case``: the front door, and what it is not allowed to do.

The T108 acceptance criteria this file covers:

* ``Case.from_image(...)`` builds **without running anything**, and
  :meth:`~flow.case.Case.explain` prints the plan, every ``why`` line, the
  geometry verdict and actions, and the estimated wall clock.
* No lattice quantity in any public ``flow/`` signature — here over every
  public callable's **annotations and defaults**, which is the half
  ``tests/test_flow_package.py``'s parameter-name scan does not do
  (constraint 13, **D-060**).
* ``run()`` composes ``live=`` / ``record=`` / ``headless=`` through
  :class:`lbm.record.TeeSink` and picks ``drop`` by **D-039** — asserted for
  every combination.
* Solid cells are seeded at rest (**D-030**) and the body interior still holds
  the rest state after 300 steps.

and the refusal path **D-065** carried from T107: a picture ``prepare``
refuses leaves the case built and not runnable, ``run()`` raises it, and
:meth:`~flow.case.Case.nearest` executes the fix it names (constraint 14).
"""

from __future__ import annotations

import inspect
import pkgutil

import numpy as np
import pytest

import flow
from flow.autoconfig import QUALITY_CELLS, Unrepresentable
from flow.quantity import Quantity
from flow.case import Case, _resolve_sinks, _seed_solid_at_rest
from flow.report import metadata_entries
from lbm.core import Q, W
from lbm.geometry import circle
from lbm.record import HeadlessSink, RecordSink, TeeSink, frame_count
from lbm.render import LiveSink
from lbm.runner import NullSink, Sim, SimConfig
from validate.polygons import interior_solid
from tests.test_flow_package import LATTICE_NAMES

DISC = "tests/data/shapes/disc.png"
ALL_BLACK = "tests/data/shapes/all_black.png"
TINY = "tests/data/shapes/tiny_body.png"

#: A case that plans: water at Re ~100 past a 2 cm body. Rung 3's Reynolds
#: number, reached through the product path rather than hand-configured — the
#: same case ``validate/autoconfig.py`` holds fixed for Rung B.
WATER_CASE = dict(fluid="water", speed="0.005 m/s", size="0.02 m", quality="fast")

#: ``old-Docs/STATE1.md`` **D-038**'s case, which the CLI refuses and must go on
#: refusing: air at 20 m/s past a 1.5 m body is ``Re = 2e6``.
AIR_CASE = dict(fluid="air", speed="20 m/s", size="1.5 m")


@pytest.fixture(scope="module")
def disc_case() -> Case:
    """One planned case, built once — building runs no timesteps."""
    return Case.from_image(DISC, **WATER_CASE)


# ---------------------------------------------------------------------------
# Building, and explaining before running
# ---------------------------------------------------------------------------


def test_from_image_builds_without_running_anything(disc_case: Case):
    """The first of ``DOCS/IDEA3.md``'s three lines. No timestep is executed."""
    assert disc_case.runnable
    assert disc_case.sim is None, "building a Case must not construct a Sim"
    assert disc_case.plan is not None
    assert disc_case.prepared.verdict in ("ok", "repaired")
    assert 90.0 < disc_case.plan.Re < 110.0


def test_from_array_takes_the_same_route_as_a_picture():
    """A bool array is prepared, measured and refused by the same rules."""
    case = Case.from_array(circle(200, 200, 100.0, 100.0, 60.0), **WATER_CASE)
    assert case.runnable
    assert case.prepared.properties["body_cells_across"] == QUALITY_CELLS["fast"]


def test_explain_prints_the_plan_every_why_line_the_geometry_and_the_clock(
    disc_case: Case, capsys
):
    """The T108 acceptance criterion, item by item, on the printed text."""
    text = disc_case.explain()
    printed = capsys.readouterr().out
    assert text in printed, "explain() must print, not only return"

    plan = disc_case.plan
    assert plan is not None

    # ...the plan: every field, by name and by value.
    for name in (
        "cells_per_length",
        "tau",
        "u_lattice",
        "domain",
        "steps",
        "steps_per_frame",
        "vorticity_limit",
        "dx",
        "dt",
        "Re",
    ):
        assert name in text, f"explain() does not print plan.{name}"

    # ...every `why` line, verbatim.
    for name, why in plan.why.items():
        assert why in text, f"explain() drops the why line for {name}"

    # ...the geometry verdict and what was repaired.
    assert f"geometry: {disc_case.prepared.verdict}" in text
    for action in disc_case.prepared.actions:
        assert action in text

    # ...and the estimated wall clock, as seconds.
    assert "estimated wall clock" in text
    assert f"{plan.estimated_seconds('numpy'):.1f} s" in text


def test_explain_says_the_gpu_estimate_currently_over_predicts():
    """Honesty about a red rung that is not this task's to fix.

    Rung B's accuracy check fails on ``--backend warp`` — the estimator is
    calibrated against the NumPy column of ``DOCS/STATE2.md`` § Performance
    baseline and over-predicts a GPU run at this size by ~1.8x. T110 owns it.
    Printing the estimate without that sentence would be publishing a number we
    know to be wrong.
    """
    case = Case.from_image(DISC, backend="warp", **WATER_CASE)
    text = case.explain(quiet=True)
    assert "over-predicts" in text and "T110" in text


def test_repr_says_whether_the_case_plans(disc_case: Case):
    assert "Re=" in repr(disc_case)
    assert "refused" in repr(Case.from_image(ALL_BLACK, **WATER_CASE))


# ---------------------------------------------------------------------------
# Constraint 13 — annotations and defaults, not only parameter names
# ---------------------------------------------------------------------------


def _public_callables():
    """Every public callable reachable through a ``flow`` module's ``__all__``."""
    import importlib
    import sys

    modules = [flow]
    for info in pkgutil.walk_packages(flow.__path__, prefix="flow."):
        importlib.import_module(info.name)
        modules.append(sys.modules[info.name])

    found = []
    for module in modules:
        names = getattr(module, "__all__", None) or [
            n for n in vars(module) if not n.startswith("_")
        ]
        for name in names:
            obj = getattr(module, name, None)
            if inspect.isfunction(obj):
                found.append((f"{module.__name__}.{name}", obj))
            elif inspect.isclass(obj):
                for member_name, member in vars(obj).items():
                    if member_name.startswith("_") or not callable(member):
                        continue
                    found.append((f"{module.__name__}.{name}.{member_name}", member))
    return found


def test_no_public_flow_signature_mentions_a_lattice_quantity_anywhere():
    """Constraint 13 over parameter names, **annotations** and **defaults**.

    ``tests/test_flow_package.py`` scans parameter names; the T108 criterion
    asks for the annotations and the defaults too, because
    ``def run(*, length: LatticeCells = NX)`` says the same forbidden thing
    three different ways and only one of them is the parameter's name.
    """
    offenders: list[str] = []
    for label, obj in _public_callables():
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        for parameter in signature.parameters.values():
            if parameter.name.lower() in LATTICE_NAMES:
                offenders.append(f"{label}: parameter {parameter.name}")
            annotation = parameter.annotation
            if annotation is not inspect.Parameter.empty:
                text = (
                    annotation
                    if isinstance(annotation, str)
                    else getattr(annotation, "__name__", str(annotation))
                )
                for word in _words(str(text)):
                    if word in LATTICE_NAMES:
                        offenders.append(f"{label}: {parameter.name}: {text}")
            default = parameter.default
            if isinstance(default, str) and default.lower() in LATTICE_NAMES:
                offenders.append(f"{label}: {parameter.name}={default!r}")
    assert not offenders, (
        "CLAUDE.md constraint 13: a lattice quantity reached a public flow/ "
        f"signature: {offenders}"
    )


def _words(text: str) -> set[str]:
    """Lower-case identifier-ish words in an annotation's text form."""
    out: set[str] = set()
    word = ""
    for ch in text:
        if ch.isalnum() or ch == "_":
            word += ch
        else:
            if word:
                out.add(word.lower())
            word = ""
    if word:
        out.add(word.lower())
    return out


def test_the_annotation_scan_would_actually_catch_a_violation():
    """A guard that never fires is not a guard."""

    def illegal(picture: str, spacing: "dx" = None):  # noqa: F821
        return picture, spacing

    signature = inspect.signature(illegal)
    caught = [
        p.name
        for p in signature.parameters.values()
        if p.annotation is not inspect.Parameter.empty
        and _words(str(p.annotation)) & LATTICE_NAMES
    ]
    assert caught == ["spacing"]


def test_case_speaks_only_physics():
    """The one number T107 crossed with, ``cells_across``, is not ``Case``'s.

    The T108 contract says so by name: it is ``prepare``'s argument. What
    ``Case`` takes is a picture, a fluid, a speed, a size and a quality level.
    """
    parameters = set(inspect.signature(Case).parameters)
    assert "cells_across" not in parameters
    assert parameters >= {"source", "fluid", "speed", "size", "quality"}
    for method in (Case.from_image, Case.from_array, Case.run):
        assert "cells_across" not in inspect.signature(method).parameters


# ---------------------------------------------------------------------------
# D-039 — the mode is not composable, the sinks are
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "live, record, headless, expect_types, expect_drop",
    [
        (False, None, None, (), True),
        (True, None, None, (LiveSink,), True),
        (False, "out.mp4", None, (RecordSink,), False),
        (False, None, "frames", (HeadlessSink,), False),
        (True, "out.mp4", None, (LiveSink, RecordSink), False),
        (True, None, "frames", (LiveSink, HeadlessSink), False),
        (False, "out.mp4", "frames", (RecordSink, HeadlessSink), False),
        (True, "out.mp4", "frames", (LiveSink, RecordSink, HeadlessSink), False),
    ],
)
def test_run_picks_the_mode_by_d039_for_every_combination(
    live, record, headless, expect_types, expect_drop, tmp_path, monkeypatch
):
    """**D-039**: any sink that writes a **file** takes ``drop=False``.

    Only a live-*only* run drops, which is exactly the case constraint 8
    describes. :class:`lbm.record.TeeSink` fans the frame out and is **not** a
    third mode (**D-024**).

    ``LiveSink`` is stubbed: opening a window in a test suite is not a test of
    the mode rule, and the rule is what this asserts.
    """

    class FakeLive(LiveSink):
        def __init__(self, *a, **k):  # noqa: D401 - a stub
            self.quit_requested = False

    monkeypatch.setattr("flow.case.LiveSink", FakeLive)

    sink, members, drop = _resolve_sinks(
        live=live,
        record=None if record is None else tmp_path / record,
        headless=None if headless is None else tmp_path / headless,
        metadata=metadata_entries(
            substituted=False, substitution=None, reynolds=100.0, backend="numpy"
        ),
    )
    assert drop is expect_drop
    assert tuple(type(m).__name__ for m in members) == tuple(
        t.__name__ if t is not LiveSink else "FakeLive" for t in expect_types
    )
    if len(members) > 1:
        assert isinstance(sink, TeeSink)
    elif members:
        assert sink is members[0]
    else:
        assert isinstance(sink, NullSink)


def test_a_recorded_video_carries_the_provenance_ffmpeg_arguments(tmp_path):
    """The recorder built by ``run()`` gets the same metadata ``save()`` writes."""
    sink, members, _ = _resolve_sinks(
        live=False,
        record=tmp_path / "out.mp4",
        headless=None,
        metadata=metadata_entries(
            substituted=True,
            substitution="fluid -> honey",
            reynolds=100.0,
            backend="numpy",
        ),
    )
    recorder = members[0]
    assert isinstance(recorder, RecordSink)
    params = recorder._extra["output_params"]
    assert "-metadata" in params
    assert any("substituted=True" in p for p in params)


def test_run_writes_a_video_and_a_frame_directory_through_one_tee(tmp_path):
    """The composition criterion through ``run()`` itself, not only the helper.

    ``--live --record`` and friends are composable (**D-039**); the tee is a
    sink, not a policy, and it hands **the same frame object** to every member
    (``lbm.record.TeeSink``), so what lands in the MP4 and what lands in the
    PNGs is the output of one :func:`lbm.render.render` call (constraint 10).
    """
    case = Case.from_image(DISC, **WATER_CASE)
    assert case.plan is not None
    video = tmp_path / "wake.mp4"
    directory = tmp_path / "frames"

    result = case.run(
        seconds=200 * case.plan.dt,
        record=video,
        headless=directory,
        keep_frames=False,
        quiet=True,
    )

    assert result.steps >= 200
    pngs = sorted(directory.glob("frame_*.png"))
    assert pngs, "the headless sink wrote nothing"
    assert video.exists()
    assert frame_count(video) == len(pngs), (
        "the two file sinks saw different numbers of frames, which is what "
        "drop=False exists to prevent (D-039)"
    )


# ---------------------------------------------------------------------------
# D-030 — the body interior starts, and stays, at rest
# ---------------------------------------------------------------------------


def test_seed_solid_at_rest_is_exactly_the_rest_equilibrium():
    """The seed itself, before any stepping — ``f[i] = w_i rho0`` on solid."""
    solid = np.zeros((32, 48), dtype=bool)
    solid[12:20, 16:24] = True
    sim = Sim(SimConfig(ny=32, nx=48, tau=0.6, inlet_U=0.05, use_inlet=True), solid)
    _seed_solid_at_rest(sim)
    f = sim.host_f()
    rest = np.float32(1.0) * W.astype(np.float32)
    for i in range(Q):
        assert np.array_equal(f[i][sim.solid], np.full(int(solid.sum()), rest[i]))


def test_the_body_interior_holds_the_rest_state_after_300_steps(disc_case: Case):
    """**D-030**, through the product path: 300 steps, still bit-identical.

    The rest state is a fixed point of both bounce-back (``W`` is symmetric
    under ``OPP``) and streaming (it is uniform), so "still at rest" is a
    *bitwise* claim and is asserted as one. Unseeded, D-030 measured the
    interior at ``|u| > 1e-3`` — the initial condition, not the boundary
    condition, is what T008's criterion would otherwise have been measuring.
    """
    plan = disc_case.plan
    assert plan is not None
    result = disc_case.run(seconds=300 * plan.dt, keep_frames=False, quiet=True)
    assert result.steps >= 300

    sim = disc_case.sim
    assert sim is not None
    f = sim.host_f()
    rest = np.float32(sim.config.rho0) * W.astype(np.float32)
    # The **interior**, as ``tests/test_polygons.py`` measures it: the surface
    # layer is where the reflection is held in flight, so it is not at rest and
    # was never claimed to be. The interior is a closed subsystem — every
    # population it exchanges with the surface came from the interior — which
    # is what makes "still at rest" a bitwise claim.
    interior = interior_solid(sim.solid)
    assert interior.any()
    for i in range(Q):
        assert np.array_equal(f[i][interior], np.full(int(interior.sum()), rest[i])), (
            f"direction {i} moved inside the body: D-030's seed did not hold"
        )


# ---------------------------------------------------------------------------
# Refusals — carried, never swallowed (D-045, D-065, constraint 16)
# ---------------------------------------------------------------------------


def test_a_refused_picture_is_built_not_raised_and_run_refuses_to_run_it():
    """**D-065**: the refusal is surfaced, and nothing is run in its place."""
    case = Case.from_image(ALL_BLACK, **WATER_CASE)
    assert case.runnable is False
    assert case.prepared.verdict == "refused"
    assert case.prepared.fix is not None

    text = case.explain(quiet=True)
    assert "refused" in text
    assert case.prepared.reason in text
    assert case.prepared.fix.note in text

    with pytest.raises(Unrepresentable) as caught:
        case.run(quiet=True)
    assert caught.value.suggestions, "constraint 14: a refusal names a fix"


def test_d038s_case_is_no_longer_refused_and_carries_a_band_instead():
    """**D-093**: air at 20 m/s past a 1.5 m body now plans, and is banded.

    Phase 1's version of this test asserted the refusal. The refusal is what
    Phase 2 exists to remove, so what is asserted now is the thing that replaced
    it: the plan, the closure that made it possible, the band it expects, and
    the fact that the band is not one that reports bare numbers.
    """
    case = Case.from_image(DISC, **AIR_CASE)
    assert case.runnable is True
    assert case.refusal is None
    assert case.plan is not None
    assert case.plan.closure_engaged
    assert not case.plan.expected_fidelity.reports_bare_numbers

    text = case.explain(quiet=True)
    assert "cs_smag" in text
    assert "fidelity" in text
    assert not case.suggestions, "nothing to fix: the case runs"


def test_an_inviscid_case_is_refused_and_explains_itself():
    """The refusal that survived T204, in plain language (constraint 14)."""
    case = Case.from_image(
        DISC,
        fluid=Quantity(0.0, default_unit="m^2/s"),
        speed="1 mm/s",
        size="1 cm",
        quality="fast",
    )
    assert case.runnable is False
    assert case.refusal is not None
    assert case.refusal.quantity == "tau"

    text = case.explain(quiet=True)
    assert "What would work" in text
    assert "no viscosity" in text
    assert case.suggestions, "constraint 14: a refusal names a fix"


def test_nearest_applies_the_tools_own_top_suggestion_and_marks_it():
    """**D-045** / **D-063**: the offer is executable, and labelled.

    ``Case.nearest()`` runs the first suggestion through
    :func:`flow.diagnose.apply_suggestion` — the same machinery Rung D uses —
    and the case it returns carries ``substituted=True``, because it is not
    the case that was asked for (constraint 16).
    """
    case = Case.from_image(
        DISC,
        fluid=Quantity(0.0, default_unit="m^2/s"),
        speed="1 mm/s",
        size="1 cm",
        quality="fast",
    )
    nearest = case.nearest()

    assert nearest.substituted is True
    assert nearest.substitution
    assert nearest.runnable, "the tool's own top suggestion must fix its case"
    assert case.substituted is False, "nearest() must not mutate the original"


def test_nearest_translates_a_resolution_fix_into_a_quality_level():
    """Constraint 13 reaches the fix, too.

    ``flow.prepare``'s fix for a picture that cannot make a body the requested
    size speaks ``cells_across``, which this layer may not take. The
    user-facing knob for body size is ``quality``, so the fix arrives as one —
    and when even the coarsest level is too much for the picture, the worked
    example is substituted and **says so**, exactly as
    :func:`flow.prepare.apply_fix` does for a ``"picture"`` fix.
    """
    asked = dict(WATER_CASE, quality="accurate")
    case = Case.from_image(TINY, **asked)
    assert case.runnable is False, "the corpus image refuses at 50 cells across"
    assert case.prepared.fix is not None
    assert case.prepared.fix.change == "resolution"

    nearest = case.nearest()
    assert nearest.substituted is True
    assert nearest.runnable
    # The picture reaches 41 cells; "balanced" is 40 and "accurate" is 50, so
    # the nearest runnable case is the finest level that fits, not the coarsest.
    assert nearest.quality == "balanced"
    assert "quality" in (nearest.substitution or "")


def test_nearest_on_a_case_that_runs_is_an_error(disc_case: Case):
    with pytest.raises(ValueError, match="not refused"):
        disc_case.nearest()


def test_an_unknown_quality_level_is_a_programmer_error():
    with pytest.raises(ValueError, match="quality="):
        Case.from_image(DISC, fluid="water", speed="1 m/s", size="1 m", quality="best")


# ---------------------------------------------------------------------------
# Assembly — the parts of the facade that are not delegation
# ---------------------------------------------------------------------------


def test_the_body_is_placed_where_the_plan_assumed_it_would_be(disc_case: Case):
    """The guardrails ``plan`` computed analytically are the ones the mask has.

    Leading edge :data:`flow.autoconfig.UPSTREAM_D` diameters from the inlet,
    centred across the flow with one cell of asymmetry — the same arithmetic as
    ``validate/autoconfig.py::build_solid`` (**D-019**, constraint 12).
    """
    from flow.autoconfig import UPSTREAM_D
    from lbm.geometry import bounding_box

    plan = disc_case.plan
    assert plan is not None
    solid = disc_case._domain()
    assert solid.shape == plan.domain

    box = bounding_box(solid)
    assert box is not None
    y0, y1, x0, x1 = box
    ny, nx = plan.domain
    assert x0 == int(round(UPSTREAM_D * plan.cells_per_length))
    assert (nx - 1 - x1) / (y1 - y0 + 1) >= 8.0, "constraint 12: 8 D downstream"
    assert (y1 - y0 + 1) / ny <= 0.10, "constraint 12: blockage under 10%"


def test_seconds_is_physical_time_and_goes_through_the_plans_dt(disc_case: Case):
    """Constraint 13: the duration is seconds, and ``dt`` converts it (**D-023**)."""
    plan = disc_case.plan
    assert plan is not None
    assert disc_case._steps("1 s") == max(1, int(round(1.0 / plan.dt)))
    assert disc_case._steps(None) == plan.steps
    with pytest.raises(ValueError, match="positive"):
        disc_case._steps("0 s")


def test_steps_per_frame_is_never_computed_here(disc_case: Case):
    """Constraint 7 / **D-023**: it comes from the plan, and only from there.

    A literal in this module is the thing the constraint exists to prevent, so
    the source is read rather than the behaviour inferred.
    """
    import pathlib

    source = pathlib.Path(flow.case.__file__).read_text(encoding="utf-8")
    assert "steps_per_frame=plan.steps_per_frame" in source.replace(" ", "")
    assert "steps_per_frame = " not in source
