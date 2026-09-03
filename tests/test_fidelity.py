"""T204 — ``flow/fidelity.py`` and the constraint-18 interlock.

Rung H (``validate/fidelity.py``) is the physics gate: it runs whole cases
through the product path and inspects what comes out. These are the fast checks
a rung cannot make cheaply — the table's exact boundaries, the gating on
:class:`flow.report.Result`, and the source-level guarantees that keep ``Cs`` a
planned quantity rather than a knob.

``DOCS/TASKS3.md`` § T204 · ``DOCS/IDEA4.md`` § The five things Phase 2 must get
right (1) · **D-082**, **D-093**, **D-094**.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from flow import case as case_module
from flow.autoconfig import TAU_FLOOR, Plan, plan
from flow.diagnose import (
    CS_SOUND,
    MASS_DRIFT_ACCURACY,
    MASS_DRIFT_MEANINGLESS,
    Monitor,
)
from flow.fidelity import (
    RATIO_QUALITATIVE,
    RATIO_QUANTITATIVE,
    RE_3D_ONSET,
    Band,
    Qualified,
    band_for,
    ratio_for,
    sentence,
)
from flow.fluids import fluid
from flow.report import GATED_QUANTITIES, Result, metadata_entries
from lbm.core import CS2, CS_SMAG_LITERATURE
from lbm.geometry import circle
from lbm.units import U_LATTICE_MAX

REPO = pathlib.Path(__file__).resolve().parents[1]
DISC = circle(80, 80, 40.0, 40.0, 20.0)

#: ``tau = 0.6`` gives ``nu = 1/30``, so a caller can ask for a ratio directly.
_NU = (0.6 - 0.5) / 3.0


def _plan(re: float, cs_smag: float = 0.0) -> Plan:
    """A minimal :class:`~flow.autoconfig.Plan` carrying a chosen ``Re``."""
    return Plan(
        cells_per_length=30,
        tau=0.6,
        u_lattice=0.05,
        domain=(720, 540),
        steps=1,
        steps_per_frame=1,
        vorticity_limit=1.0,
        dx=1.0,
        dt=1.0,
        Re=re,
        cs_smag=cs_smag,
        warnings=[],
        why={},
    )


def _result(band: Band, **kw) -> Result:
    """A :class:`~flow.report.Result` carrying real numbers and a chosen band."""
    fields = dict(
        cd=1.4030,
        cd_std=0.0123,
        cd_amplitude=0.0456,
        cl=0.3210,
        cl_mean=0.0001,
        strouhal=0.1676,
        strouhal_confidence=4.2,
        periods=9.5,
        convergence=1e-6,
        peak_u=0.0973,
        elapsed=1.0,
        substituted=False,
        backend="numpy",
        steps=100,
        stable=True,
        sample_steps=10,
        fps=60.0,
        fidelity=band,
        cd_history=np.full(40, 1.4030),
        cl_history=np.zeros(40),
    )
    fields.update(kw)
    return Result(**fields)


# ---------------------------------------------------------------------------
# The table — DOCS/IDEA4.md § 1, verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "re, ratio, expected",
    [
        (100.0, 0.0, Band.QUANTITATIVE),
        (100.0, 0.0999, Band.QUANTITATIVE),
        (100.0, 0.1, Band.QUALITATIVE),
        (200.0, 0.05, Band.QUANTITATIVE),
        (200.001, 0.05, Band.QUALITATIVE),
        (2.0e6, 0.05, Band.QUALITATIVE),
        (100.0, 0.9999, Band.QUALITATIVE),
        (100.0, 1.0, Band.ILLUSTRATIVE),
        (2.0e6, 1.0, Band.ILLUSTRATIVE),
        (2.0e6, 37000.0, Band.ILLUSTRATIVE),
    ],
)
def test_band_for_is_the_spec_table(re, ratio, expected):
    assert band_for(_plan(re), ratio * _NU) is expected


def test_the_boundaries_are_strict_inequalities():
    """``< 0.1`` and ``< 1``, not ``<=`` — the table says so and it matters."""
    assert band_for(_plan(100.0), RATIO_QUANTITATIVE * _NU) is Band.QUALITATIVE
    assert band_for(_plan(100.0), RATIO_QUALITATIVE * _NU) is Band.ILLUSTRATIVE
    assert RE_3D_ONSET == 200.0


def test_the_re_gate_cites_williamson():
    """The physics boundary is **cited, not chosen** (T204's own criterion)."""
    import flow.fidelity as fidelity

    doc = (fidelity.__doc__ or "") + (band_for.__doc__ or "")
    assert "Williamson" in doc and "1996" in doc


def test_ratio_refuses_a_tau_that_has_no_viscosity():
    with pytest.raises(ValueError, match="constraint 2"):
        ratio_for(_plan(100.0).__class__(**{**_plan(100.0).__dict__, "tau": 0.5}), 0.0)


def test_ratio_refuses_a_negative_eddy_viscosity():
    with pytest.raises(ValueError, match="never removes"):
        ratio_for(_plan(100.0), -1e-9)


# ---------------------------------------------------------------------------
# Expected before the run, earned after it
# ---------------------------------------------------------------------------


def test_a_bgk_plan_expects_its_band_from_re_alone():
    assert band_for(_plan(100.0)) is Band.QUANTITATIVE
    assert band_for(_plan(1000.0)) is Band.QUALITATIVE


def test_a_closure_on_plan_cannot_expect_the_quantitative_band():
    """It does not know its ``nu_t`` yet, so it may not claim the top band."""
    assert band_for(_plan(100.0, CS_SMAG_LITERATURE)) is Band.QUALITATIVE


def test_the_earned_band_wins_and_the_result_says_so():
    """A plan that expected quantitative and earned qualitative is a finding."""
    r = _result(
        Band.QUALITATIVE,
        expected_fidelity=Band.QUANTITATIVE,
        nu_t_ratio=0.34,
    )
    assert r.fidelity is Band.QUALITATIVE
    joined = " ".join(r.warnings)
    assert "quantitative" in joined and "qualitative" in joined
    assert "0.34" in joined


def test_agreement_between_expected_and_earned_adds_no_noise():
    r = _result(Band.QUANTITATIVE, expected_fidelity=Band.QUANTITATIVE)
    assert not [w for w in r.warnings if w.startswith("fidelity:")]


def test_worse_of_prefers_the_less_trustworthy_band():
    assert Band.worse_of(Band.QUANTITATIVE, Band.ILLUSTRATIVE) is Band.ILLUSTRATIVE
    assert Band.worse_of(Band.QUALITATIVE, Band.QUANTITATIVE) is Band.QUALITATIVE


# ---------------------------------------------------------------------------
# Constraint 18 — the interlock, on the object and on every rendering of it
# ---------------------------------------------------------------------------


def test_the_quantitative_band_reports_its_numbers_bare():
    r = _result(Band.QUANTITATIVE)
    assert r.cd == pytest.approx(1.4030)
    assert "1.4030" in r.summary(quiet=True)
    assert r.as_dict()["cd"] == pytest.approx(1.4030)


@pytest.mark.parametrize("band", [Band.QUALITATIVE, Band.ILLUSTRATIVE])
def test_no_gated_quantity_survives_outside_the_quantitative_band(band):
    r = _result(band)
    for name in GATED_QUANTITIES:
        assert getattr(r, name) is None, f"{band}: Result.{name} survived"
        assert r.as_dict()[name] is None, f"{band}: as_dict()[{name!r}] survived"


def test_an_illustrative_result_has_no_drag_coefficient_anywhere():
    """The test T204's contract names: try to get a bare ``Cd`` and fail."""
    r = _result(Band.ILLUSTRATIVE)
    assert r.cd is None
    assert r.cd_qualified is None
    text = r.summary(quiet=True)
    assert "1.4030" not in text
    assert "not reported" in text
    assert r.as_dict()["cd_qualified"] is None


def test_a_qualitative_result_reports_its_cd_only_through_qualified():
    r = _result(Band.QUALITATIVE)
    assert r.cd is None
    assert isinstance(r.cd_qualified, Qualified)
    assert r.cd_qualified.cd == pytest.approx(1.4030)
    assert r.cd_qualified.band is Band.QUALITATIVE
    assert str(Band.QUALITATIVE) in str(r.cd_qualified)
    # ...and every summary line carrying the number also carries the band.
    for line in r.summary(quiet=True).splitlines():
        if "1.4030" in line:
            assert str(Band.QUALITATIVE) in line, line


def test_qualified_is_not_usable_as_a_bare_number():
    """No ``__float__``: it cannot be slipped into arithmetic or a format spec."""
    q = _result(Band.QUALITATIVE).cd_qualified
    assert q is not None
    with pytest.raises(TypeError):
        float(q)  # type: ignore[arg-type]


def test_the_caveat_travels_with_the_number_in_as_dict():
    q = _result(Band.QUALITATIVE).as_dict()["cd_qualified"]
    assert q["band"] == "qualitative"
    assert q["caveat"] == sentence(Band.QUALITATIVE)


@pytest.mark.parametrize("band", list(Band))
def test_the_plot_never_draws_a_number_the_object_withheld(band):
    fig = _result(band).plot()
    titles = " ".join(ax.get_title() for ax in fig.axes)
    if band is Band.QUANTITATIVE:
        assert "1.4030" in titles
    else:
        assert str(band) in titles
        if band is Band.ILLUSTRATIVE:
            assert "1.4030" not in titles


@pytest.mark.parametrize("band", list(Band))
def test_the_band_reaches_the_video_metadata(band):
    """Constraint 16 and 18 together: the container says which band it is."""
    entries = metadata_entries(
        substituted=True,
        substitution="the closure is on",
        reynolds=2.0e6,
        backend="warp",
        fidelity=band,
        closure_engaged=True,
    )
    assert f"fidelity={band.value}" in entries["comment"]
    assert "closure=on" in entries["comment"]


def test_an_in_flight_video_labels_its_band_as_planned():
    """A verdict the writer cannot have yet is labelled rather than stamped."""
    entries = metadata_entries(
        substituted=True,
        substitution="x",
        reynolds=1.0,
        backend="numpy",
        fidelity=Band.QUALITATIVE,
        closure_engaged=True,
        provisional=True,
    )
    assert "fidelity=qualitative (planned)" in entries["comment"]


def test_a_result_defaults_to_the_band_phase_1_earned():
    """A ``Result`` built without a band behaves exactly as Phase 1's did."""
    r = _result(Band.QUANTITATIVE)
    assert Result.__dataclass_fields__["fidelity"].default is Band.QUANTITATIVE
    assert r.closure_engaged is False
    assert r.nu_t_ratio == 0.0


def test_case_run_always_passes_a_band_explicitly():
    """The default is permissive, so the one production caller must not use it.

    Read over the **syntax**: ``flow/case.py`` must construct its ``Result``
    with ``fidelity=`` present. Without this, the defaulted band would be a
    quiet way for a closure-on run to report bare numbers.
    """
    tree = ast.parse(pathlib.Path(case_module.__file__).read_text(encoding="utf-8"))
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Result"
    ]
    assert constructions, "flow/case.py no longer builds a Result"
    for node in constructions:
        keywords = {kw.arg for kw in node.keywords}
        for required in ("fidelity", "expected_fidelity", "closure_engaged"):
            assert required in keywords, (
                f"flow/case.py builds a Result without {required}="
            )


# ---------------------------------------------------------------------------
# The closure as a planned quantity (constraint 13) and D-093's switch
# ---------------------------------------------------------------------------


def test_the_closure_comes_on_only_below_the_bluff_body_floor():
    easy = plan(
        fluid=fluid("water"), speed="5 mm/s", size="2 cm", mask=DISC, quality="fast"
    )
    hard = plan(
        fluid=fluid("air"), speed="20 m/s", size="1.5 m", mask=DISC, quality="fast"
    )
    assert easy.tau > TAU_FLOOR and easy.cs_smag == 0.0
    assert hard.tau <= TAU_FLOOR and hard.cs_smag == CS_SMAG_LITERATURE


def test_the_d038_case_plans_and_is_not_quantitative():
    """**D-093**, the supersession, asserted rather than described."""
    p = plan(
        fluid=fluid("air"), speed="20 m/s", size="1.5 m", mask=DISC, quality="fast"
    )
    assert p.closure_engaged
    assert not p.expected_fidelity.reports_bare_numbers


# ---------------------------------------------------------------------------
# D-094 — the speed tripwire, moved and still counting
# ---------------------------------------------------------------------------


def test_the_default_monitor_still_watches_the_accuracy_ceiling():
    assert Monitor().speed_ceiling == U_LATTICE_MAX


def test_a_closure_on_monitor_watches_the_lattice_sound_speed():
    m = Monitor(closure=True)
    assert m.speed_ceiling == pytest.approx(CS_SOUND)
    assert CS_SOUND == pytest.approx(CS2**0.5)
    assert CS_SOUND > U_LATTICE_MAX


def test_the_moved_tripwire_still_counts_what_it_stopped_stopping():
    """Moving a wire is defensible only if the crossings are still reported."""

    class _FakeSim:
        step_count = 25
        solid = np.zeros((4, 4), dtype=bool)
        config = type("cfg", (), {"tau": 0.5001})()

        def host_u(self):
            u = np.zeros((2, 4, 4), dtype=np.float32)
            u[0, 2, 2] = 0.2  # over 0.1, well under the sound speed
            return u

        def host_rho(self):
            return np.ones((4, 4), dtype=np.float32)

    watched = Monitor(every=1, closure=True)
    sim = _FakeSim()
    for _ in range(5):
        watched(sim)  # does not raise: 0.2 < CS_SOUND
    assert watched.over_accuracy_ceiling == 5
    assert watched.peak_seen == pytest.approx(0.2, rel=1e-4)


def test_the_narrow_wire_would_have_fired_on_the_same_state():
    """The counter above is not vacuous: the default Monitor does raise here."""
    from flow.diagnose import Diverging

    class _FakeSim:
        step_count = 25
        solid = np.zeros((4, 4), dtype=bool)
        config = type("cfg", (), {"tau": 0.5001})()

        def host_u(self):
            u = np.zeros((2, 4, 4), dtype=np.float32)
            u[0, 2, 2] = 0.2
            return u

        def host_rho(self):
            return np.ones((4, 4), dtype=np.float32)

    narrow = Monitor(every=1)
    sim = _FakeSim()
    with pytest.raises(Diverging):
        for _ in range(5):
            narrow(sim)


def test_the_default_monitor_still_watches_the_one_percent_mass_bound():
    assert Monitor().mass_drift == pytest.approx(MASS_DRIFT_ACCURACY)


def test_a_closure_on_monitor_watches_the_meaningless_mass_bound():
    """The same argument as the speed wire, on the other variable (**D-094**)."""
    assert Monitor(closure=True).mass_drift == pytest.approx(MASS_DRIFT_MEANINGLESS)
    assert MASS_DRIFT_MEANINGLESS > MASS_DRIFT_ACCURACY


def test_an_explicit_mass_drift_still_wins_over_both_defaults():
    assert Monitor(closure=True, mass_drift=0.02).mass_drift == pytest.approx(0.02)


def test_the_moved_mass_wire_still_counts_what_it_stopped_stopping():
    """Same rule as the speed wire: a widened bound must still report crossings."""

    class _LeakingSim:
        step_count = 25
        solid = np.zeros((4, 4), dtype=bool)
        config = type("cfg", (), {"tau": 0.5001})()
        _mass = 16.0

        def host_u(self):
            return np.zeros((2, 4, 4), dtype=np.float32)

        def host_rho(self):
            self._mass -= 0.05  # a slow linear leak, ~0.3% a sample
            return np.full((4, 4), self._mass / 16.0, dtype=np.float32)

    watched = Monitor(every=1, closure=True)
    sim = _LeakingSim()
    for _ in range(8):
        watched(sim)  # does not raise: the drift is far under a half
    assert watched.over_accuracy_drift > 0
    assert watched.drift_seen > MASS_DRIFT_ACCURACY
    assert watched.drift_seen < MASS_DRIFT_MEANINGLESS


def test_the_wider_wire_still_fires_at_the_sound_speed():
    from flow.diagnose import Diverging

    class _FakeSim:
        step_count = 25
        solid = np.zeros((4, 4), dtype=bool)
        config = type("cfg", (), {"tau": 0.5001})()

        def host_u(self):
            u = np.zeros((2, 4, 4), dtype=np.float32)
            u[0, 2, 2] = 0.8  # supersonic on the lattice
            return u

        def host_rho(self):
            return np.ones((4, 4), dtype=np.float32)

    watched = Monitor(every=1, closure=True)
    sim = _FakeSim()
    with pytest.raises(Diverging):
        for _ in range(5):
            watched(sim)


# ---------------------------------------------------------------------------
# The prose contract (D-047's posture: it exists, it names its band, it leaks
# no lattice quantity — never its wording)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", list(Band))
def test_every_band_has_a_sentence_that_names_it(band):
    text = sentence(band)
    assert text.strip()
    assert str(band) in text


@pytest.mark.parametrize("band", list(Band))
def test_no_band_sentence_names_a_lattice_quantity(band):
    text = sentence(band).lower()
    leaks = [
        w
        for w in ("tau", "lattice", "timestep", "cs_smag", "reynolds", "mach")
        if w in text
    ]
    assert not leaks, f"{band}: {leaks}"


@pytest.mark.parametrize("band", list(Band))
def test_every_band_sentence_is_ascii(band):
    """A Windows console at its default codepage mojibakes an em dash (T104)."""
    sentence(band).encode("ascii")
