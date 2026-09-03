"""Rung H — every band's claim is true, and no run overclaims.

``DOCS/IDEA4.md`` § Validation ladder, Rung H::

    A Re sweep across all three bands. The quantitative band reproduces
    published data; **no run outside it emits an unqualified Cd**, asserted by
    inspecting Result; every band carries its sentence; D-038's own case
    (air, 20 m/s, 1.5 m) runs and lands in `illustrative`.

This is the gate for **T204** and, with it, **M10**. Run it from the repo root::

    myenv/Scripts/python.exe -m validate.fidelity
    myenv/Scripts/python.exe -m validate.fidelity --backend warp
    myenv/Scripts/python.exe -m validate.fidelity --skip-cylinder   # a faster smoke

**Wall clock.** Every case in the sweep is a full product-path run — the domain
:mod:`flow.autoconfig` sizes (24 D span, 8 D upstream, **D-075**) for the run
length it sizes (80 convective times, **D-079**) — because a shortened run would
be a case whose agreement with the product nobody checks. That is ~48000 steps
over ~389k cells each: about **3 minutes** for the whole rung on ``warp`` and
about **50 minutes** on ``numpy``. Run the long one detached.

Why this rung is written the way it is
---------------------------------------

**D-091 is the warning it was written against.** Rung G's ``Cs = 0.17`` clause
would have passed *with the closure term deleted* at any comfortably resolved
operating point, and the fix was to size the case and add a clause that fails if
the term is removed. Rung H carries the same risk in a worse form: **a band
assertion that no case in the sweep can violate is a green rung that proves
nothing.** Two things are done about it, and neither is optional:

* :func:`check_table` evaluates :func:`flow.fidelity.band_for` on points that
  straddle **both** boundaries from both sides, and then **inverts each
  comparison in turn** and asserts that at least one verdict changes. A
  boundary nothing can be on the wrong side of is not a boundary.
* The sweep's middle case is chosen so that it sits at ``Re <= 200`` **and**
  outside the quantitative ratio — so it is a case that would be called
  ``quantitative`` if the ``max(nu_t)/nu`` half of the top band's condition were
  dropped, and the rung says so by name and checks it.

**What is *not* tested is the prose** (**D-047**'s posture, and
``DOCS/TASKS3.md`` § T204 Notes say it in as many words): a sentence must exist,
must name its band, and must leak no lattice quantity. Its wording is not a
rung's business. What *is* tested is the verdict and the **absence of the
number**.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flow.autoconfig import TAU_FLOOR, Plan, plan as _plan
from flow.case import Case
from flow.fidelity import (
    RATIO_QUALITATIVE,
    RATIO_QUANTITATIVE,
    RE_3D_ONSET,
    Band,
    band_for,
    ratio_for,
    sentence,
)
from flow.fluids import fluid as _fluid
from flow.report import GATED_QUANTITIES, Result
from lbm.backends import BackendUnavailableError, get_backend
from lbm.core import CS_SMAG_LITERATURE

__all__ = [
    "SWEEP",
    "SweepCase",
    "check_table",
    "check_expectation",
    "run_sweep_case",
    "check_constraint_18",
    "main",
]


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

#: The picture every sweep case is run on — **Rung E's own**, the committed PNG
#: ``validate/minute.py`` uses, so the geometry is not a variable, the only
#: thing that changes across the sweep is the physics, and the quantitative
#: anchor is *literally* the case whose published digits are on file rather than
#: a similar one. (An 80x80 ``circle`` array rasterises to a slightly different
#: 30-cell body and reads ``Cd`` 1.4237 / peak ``|u|`` 0.1020 against this PNG's
#: 1.4030 / 0.0973 — small, and exactly the kind of difference a rung must not
#: introduce between itself and the number it is checking.)
DISC: str = "tests/data/shapes/disc.png"

#: Rung 3's published bands, imported in spirit and restated here only because
#: the module they live in is a rung and this is a different rung reading them.
#: They are **not widened** — ``DOCS/TASKS3.md`` § T204 Notes: widening a band to
#: make a number reportable is the one move this task must not make.
from validate.cylinder import CD_BAND, ST_BAND  # noqa: E402  (documented above)

#: Lattice words that must not appear in anything a user reads
#: (``flow/diagnose.py``'s own rule for ``_FIRST_PARAGRAPH``, applied to
#: :func:`flow.fidelity.sentence`).
LATTICE_WORDS: tuple[str, ...] = ("tau", "lattice", "timestep", "cs_smag")


@dataclass(frozen=True)
class SweepCase:
    """One case in the Re sweep, and the band it must land in.

    Attributes:
        name: what it is, for the printed table.
        request: the ``fluid`` / ``speed`` / ``size`` / ``quality`` a user types.
        expect: the :class:`flow.fidelity.Band` this case must **earn**.
        published: ``True`` when this case is Rung 3's own and must therefore
            reproduce ``CD_BAND`` and ``ST_BAND`` as well as its band.
        note: why this case is in the sweep — read by the printed report.
    """

    name: str
    request: dict[str, Any]
    expect: Band
    published: bool = False
    note: str = ""


SWEEP: tuple[SweepCase, ...] = (
    SweepCase(
        name="quantitative - Rung 3's own case through the product",
        request=dict(fluid="water", speed="5 mm/s", size="2 cm", quality="fast"),
        expect=Band.QUANTITATIVE,
        published=True,
        note=(
            "Re 99.6: tau clears the bluff-body floor, so the closure never "
            "comes on, nu_t is identically zero and Re <= 200. Both halves of "
            "the top band's condition are met and the numbers are reported "
            "bare -- against Rung 3's published, unwidened bands."
        ),
    ),
    SweepCase(
        name="qualitative - Re <= 200, but the model is doing 10%+ of the work",
        request=dict(fluid="water", speed="8 mm/s", size="2 cm", quality="fast"),
        expect=Band.QUALITATIVE,
        note=(
            "THE DISCRIMINATOR (D-091's lesson): Re is ~159, i.e. INSIDE the "
            "Re <= 200 gate, so this case is called quantitative by the "
            "Reynolds number alone. It is not quantitative, and the only thing "
            "that says so is the measured max(nu_t)/nu. Drop that half of the "
            "condition and this row lands in the wrong band."
        ),
    ),
    SweepCase(
        name="illustrative - D-038's own case (air, 20 m/s, 1.5 m)",
        request=dict(fluid="air", speed="20 m/s", size="1.5 m", quality="fast"),
        expect=Band.ILLUSTRATIVE,
        note=(
            "The case Phase 1 refused (D-038, D-074) and this whole phase "
            "exists for. It RUNS now (D-093), and the model supplies four "
            "orders of magnitude more viscosity than the fluid does, so it is "
            "illustrative: a moving picture and no coefficient anywhere."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Clause 1 — the table, and the proof that its boundaries have teeth
# ---------------------------------------------------------------------------


@dataclass
class TableResult:
    """What :func:`check_table` found.

    Attributes:
        rows: ``(Re, ratio, expected, got)`` for every straddling point.
        wrong: the rows whose verdict was not the table's.
        inversion_caught: for each inverted comparison, whether at least one
            row's verdict changed — the D-091 discriminator.
    """

    rows: list[tuple[float, float, Band, Band]] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    inversion_caught: dict[str, bool] = field(default_factory=dict)


def _fake_plan(re: float) -> Plan:
    """A :class:`~flow.autoconfig.Plan` carrying a chosen ``Re`` and a fixed ``nu``.

    :func:`flow.fidelity.band_for` reads three fields and nothing else, so the
    table can be exercised at any ``(Re, ratio)`` point without running a case —
    which is what lets clause 1 straddle both boundaries by *hundredths* rather
    than by whatever Reynolds numbers happen to be reachable.

    ``tau`` is fixed at 0.6 (``nu = 1/30``) so the ratio a caller asks for is
    the ratio :func:`flow.fidelity.ratio_for` computes.
    """
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
        cs_smag=0.0,
        warnings=[],
        why={},
    )


#: ``(Re, ratio)`` and the band ``DOCS/IDEA4.md``'s table gives it. Every
#: boundary is approached from **both** sides and by a margin small enough that
#: a ``<`` written as ``<=`` would be caught.
TABLE_POINTS: tuple[tuple[float, float, Band], ...] = (
    # Re inside the gate, ratio walking across the 0.1 boundary.
    (100.0, 0.0, Band.QUANTITATIVE),
    (100.0, 0.099, Band.QUANTITATIVE),
    (100.0, 0.1, Band.QUALITATIVE),
    (100.0, 0.101, Band.QUALITATIVE),
    # The Re gate itself, at a ratio the top band would otherwise allow.
    (199.0, 0.05, Band.QUANTITATIVE),
    (200.0, 0.05, Band.QUANTITATIVE),
    (201.0, 0.05, Band.QUALITATIVE),
    (2.0e6, 0.05, Band.QUALITATIVE),
    # The outer boundary, from both sides, at both kinds of Re.
    (100.0, 0.999, Band.QUALITATIVE),
    (100.0, 1.0, Band.ILLUSTRATIVE),
    (100.0, 1.001, Band.ILLUSTRATIVE),
    (2.0e6, 0.999, Band.QUALITATIVE),
    (2.0e6, 1.0, Band.ILLUSTRATIVE),
    (2.0e6, 37000.0, Band.ILLUSTRATIVE),
)


def _band_with(re: float, ratio: float, *, re_gate: bool, ratio_gate: bool) -> Band:
    """The table, re-implemented with either comparison optionally **removed**.

    This is the discriminator's other half: :func:`check_table` compares the real
    :func:`flow.fidelity.band_for` against this function with one gate dropped,
    and requires the two to disagree somewhere. A condition that can be deleted
    without changing any verdict is a condition the rung is not testing.
    """
    quantitative_re = (re <= RE_3D_ONSET) if re_gate else True
    small_ratio = (ratio < RATIO_QUANTITATIVE) if ratio_gate else True
    if quantitative_re and small_ratio:
        return Band.QUANTITATIVE
    if ratio < RATIO_QUALITATIVE:
        return Band.QUALITATIVE
    return Band.ILLUSTRATIVE


def check_table() -> TableResult:
    """``band_for`` **is** ``DOCS/IDEA4.md``'s table, and both gates have teeth.

    Returns:
        A :class:`TableResult`.
    """
    out = TableResult()
    nu = (0.6 - 0.5) / 3.0
    for re, ratio, expected in TABLE_POINTS:
        got = band_for(_fake_plan(re), ratio * nu)
        out.rows.append((re, ratio, expected, got))
        if got is not expected:
            out.wrong.append(
                f"Re {re:g}, max(nu_t)/nu {ratio:g}: expected {expected}, got {got}"
            )

    # D-091's discriminator: drop one gate at a time and require a disagreement.
    for label, kwargs in (
        ("the Re <= 200 gate", dict(re_gate=False, ratio_gate=True)),
        ("the max(nu_t)/nu < 0.1 gate", dict(re_gate=True, ratio_gate=False)),
    ):
        changed = any(
            _band_with(re, ratio, **kwargs) is not band_for(_fake_plan(re), ratio * nu)
            for re, ratio, _ in TABLE_POINTS
        )
        out.inversion_caught[label] = changed
    return out


# ---------------------------------------------------------------------------
# Clause 2 — expected before the run, earned after it
# ---------------------------------------------------------------------------


def check_expectation() -> list[tuple[str, bool, str]]:
    """``band_for(plan)`` is the plan's expectation; with ``nu_t`` it is the verdict.

    The T204 contract: *"Before a run, ``band_for`` returns the band the plan
    expects, from ``Re`` alone; after a run it returns the band the run earned,
    from the measured ``nu_t``."*

    Returns:
        ``(name, ok, detail)`` rows for the printed report.
    """
    nu = (0.6 - 0.5) / 3.0
    low = _fake_plan(100.0)
    high = _fake_plan(2.0e6)
    # A plan that engaged the closure cannot claim the top band before running.
    engaged = Plan(**{**low.__dict__, "cs_smag": CS_SMAG_LITERATURE})

    return [
        (
            "before a run, BGK at Re <= 200 expects quantitative",
            band_for(low) is Band.QUANTITATIVE,
            f"{band_for(low)} (nu_t is exactly 0 with the closure off)",
        ),
        (
            "before a run, BGK above Re 200 expects qualitative",
            band_for(high) is Band.QUALITATIVE,
            f"{band_for(high)}",
        ),
        (
            "before a run, a closure-on plan cannot expect quantitative",
            band_for(engaged) is not Band.QUANTITATIVE,
            f"{band_for(engaged)} at Re {engaged.Re:g} with Cs = "
            f"{engaged.cs_smag:g}",
        ),
        (
            "after a run, the measured nu_t overrides the expectation",
            band_for(engaged, 1.5 * nu) is Band.ILLUSTRATIVE
            and band_for(engaged) is Band.QUALITATIVE,
            "expected qualitative, earned illustrative at max(nu_t)/nu = 1.5",
        ),
        (
            "every band has a sentence, and it names its band",
            all(str(b) in sentence(b) for b in Band),
            ", ".join(f"{b}: {len(sentence(b))} chars" for b in Band),
        ),
        (
            "no sentence leaks a lattice quantity (D-047's posture)",
            not [
                f"{b}:{w}"
                for b in Band
                for w in LATTICE_WORDS
                if w in sentence(b).lower()
            ],
            "checked for " + ", ".join(LATTICE_WORDS),
        ),
    ]


# ---------------------------------------------------------------------------
# Clause 3 — constraint 18, asserted on the object and on the rendered summary
# ---------------------------------------------------------------------------


def check_constraint_18(result: Result) -> list[str]:
    """Every way a ``Result`` could leak an unqualified ``Cd``. Returns failures.

    ``CLAUDE.md`` constraint 18, machine-checked — *"asserted by inspecting the
    object and the rendered summary, not by reading the prose"*. Six checks, and
    the last two are the ones that would catch a renderer that formats a number
    the object withheld:

    1. the band is present at all;
    2. in the quantitative band the gated quantities **are** numbers (a rung
       that only ever checks for absence would pass on a tool that reports
       nothing);
    3. outside it, every one of :data:`flow.report.GATED_QUANTITIES` is ``None``;
    4. in the illustrative band :attr:`~flow.report.Result.cd_qualified` is
       ``None`` too, so there is no drag coefficient anywhere on the object;
    5. :meth:`~flow.report.Result.as_dict` carries the band and the same
       ``None``\\ s;
    6. the rendered summary contains the drag value **only** on a line that also
       names the band, and in the illustrative band does not contain it at all.

    Args:
        result: the run to inspect.

    Returns:
        A list of failure strings; empty means the run did not overclaim.
    """
    bad: list[str] = []
    band = result.fidelity
    text = result.summary(quiet=True)
    as_dict = result.as_dict()

    if not isinstance(band, Band):
        bad.append(f"no fidelity band on the result: {band!r}")
        return bad

    if band is Band.QUANTITATIVE:
        for name in GATED_QUANTITIES:
            if name in ("strouhal", "strouhal_confidence", "periods"):
                continue  # legitimately None when the wake is not shedding
            if getattr(result, name) is None:
                bad.append(f"quantitative band withheld {name}: nothing reported")
        if "Cd " not in text:
            bad.append("quantitative summary prints no Cd at all")
        return bad

    for name in GATED_QUANTITIES:
        if getattr(result, name) is not None:
            bad.append(
                f"{band}: Result.{name} is {getattr(result, name)!r}, not None "
                "-- an unqualified quantity outside the quantitative band "
                "(constraint 18)"
            )
        if as_dict.get(name) is not None:
            bad.append(f"{band}: as_dict()[{name!r}] is not None")
    if as_dict.get("fidelity") != band.value:
        bad.append(f"{band}: as_dict() does not carry the band")

    qualified = result.cd_qualified
    if band is Band.ILLUSTRATIVE:
        if qualified is not None:
            bad.append(
                "illustrative: a drag coefficient survived on the result "
                f"({qualified!r}) -- this band emits none at all"
            )
        # The number must not appear anywhere in the rendered text either.
        raw = float(np.mean(result.cd_history[result.cd_history.size // 2 :])) if (
            result.cd_history.size
        ) else float("nan")
        if np.isfinite(raw) and f"{raw:.4f}" in text:
            bad.append(
                f"illustrative summary contains the drag value {raw:.4f} "
                "-- the object withheld it and the renderer put it back"
            )
        if "not reported" not in text:
            bad.append("illustrative summary does not say the Cd is not reported")
        return bad

    # Qualitative: the number may appear, but never without its band.
    if qualified is None:
        bad.append("qualitative: no Qualified drag coefficient was emitted")
        return bad
    if hasattr(qualified, "__float__"):
        bad.append(
            "Qualified implements __float__, so it can be used as a bare number"
        )
    shown = f"{qualified.cd:.4f}"
    for line in text.splitlines():
        if shown in line and str(band) not in line:
            bad.append(
                f"qualitative: the drag value appears unqualified on a summary "
                f"line: {line.strip()!r}"
            )
    if str(band) not in text:
        bad.append("qualitative summary does not name its band")
    return bad


# ---------------------------------------------------------------------------
# Clause 4 — the sweep, run
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """One run of the sweep, reduced to what the report prints."""

    case: SweepCase
    plan: Plan
    result: Result
    failures: list[str]
    seconds: float


def run_sweep_case(case: SweepCase, backend: str, *, quiet: bool = True) -> SweepResult:
    """Plan it, run it, band it, and check every constraint-18 route out of it."""
    request = dict(case.request)
    quality = request.pop("quality", "fast")
    start = time.perf_counter()
    flow_case = Case.from_image(
        DISC,
        fluid=request["fluid"],
        speed=request["speed"],
        size=request["size"],
        quality=quality,
        backend=backend,
    )
    assert flow_case.plan is not None, f"{case.name}: the product refused to plan it"
    result = flow_case.run(live=False, keep_frames=False, quiet=quiet)
    seconds = time.perf_counter() - start

    failures = check_constraint_18(result)
    if result.fidelity is not case.expect:
        failures.append(
            f"earned band {result.fidelity}, expected {case.expect} "
            f"(max(nu_t)/nu = {result.nu_t_ratio:.4g})"
        )
    if not result.stable:
        failures.append("the state was not finite at the end of the run")
    if flow_case.plan.closure_engaged and not result.closure_engaged:
        failures.append("the plan engaged the closure and the result does not say so")
    if result.closure_engaged and not result.substituted:
        failures.append(
            "a closure-on run is a substituted run (constraint 16) and this one "
            "is not marked"
        )
    if result.closure_engaged:
        meta = result.metadata()["comment"]
        for token in (f"fidelity={result.fidelity.value}", "closure=on"):
            if token not in meta:
                failures.append(f"video metadata is missing {token!r}: {meta}")
    if case.published:
        cd_lo, cd_hi = CD_BAND
        st_lo, st_hi = ST_BAND
        if result.cd is None or not cd_lo <= result.cd <= cd_hi:
            failures.append(
                f"Cd {result.cd} outside Rung 3's published {cd_lo}-{cd_hi}"
            )
        if result.strouhal is None or not st_lo <= result.strouhal <= st_hi:
            failures.append(
                f"St {result.strouhal} outside Rung 3's published {st_lo}-{st_hi}"
            )
    return SweepResult(case, flow_case.plan, result, failures, seconds)


# ---------------------------------------------------------------------------
# Clause 5 — the qualitative band's own falsifiable claim (Q-203)
# ---------------------------------------------------------------------------


@dataclass
class QualitativeEvidence:
    """Rung 3's own case, run **inside** the qualitative band. Q-203's evidence.

    Attributes:
        ratio: ``max(nu_t)/nu`` the case generated at ``Cs = 0.17``.
        band: the band that ratio earns at Re 100.
        cd: the drag coefficient the same run measured.
        st: its Strouhal number.
        passed: whether ``validate.cylinder``'s own report passed.
        seconds: wall clock.
    """

    ratio: float
    band: Band
    cd: float
    st: float
    passed: bool
    seconds: float


def check_qualitative_claim(backend: str, cs: float = CS_SMAG_LITERATURE) -> QualitativeEvidence:
    """Does a **qualitative** run reproduce published data? Measure, don't assert.

    This is the whole of **Q-203**. The qualitative band ships a ``Cd`` at all
    only because a case that lands in it reproduces the benchmark this project
    has measured five times — so the rung runs that case rather than citing it.

    It is **Rung 3's own harness** with ``cs_smag`` threaded through
    (**D-087**'s rule: a second copy of Rung 3's setup would be a case whose
    agreement with the real Rung 3 nobody checks), and the bands it is held to
    are Rung 3's published ones, unwidened.

    Args:
        backend: the T101 backend to run on.
        cs: the Smagorinsky constant. The literature value; Phase 2 does not
            tune it.

    Returns:
        A :class:`QualitativeEvidence`.
    """
    from lbm.core import CS2, equilibrium, macroscopic
    from lbm.probe import eddy_viscosity
    from validate.cylinder import report as cylinder_report, run_cylinder

    start = time.perf_counter()
    res = run_cylinder(
        headless=True,
        bench_steps=0,
        verbose_mask=False,
        backend=backend,
        cs_smag=cs,
    )
    passed = cylinder_report(res)

    # The ratio and the Cd have to come from the **same run**, or the clause is
    # comparing a band measured on one case against a number measured on
    # another. `CylinderResult.sim` is the handle T204 added for exactly this.
    sim = res.sim
    assert sim is not None, "validate.cylinder stopped returning its Sim"
    f = sim.host_f()
    rho, u = macroscopic(f.copy())
    feq = equilibrium(rho, u)
    nu_t = eddy_viscosity(f, feq, res.tau, cs)
    nu = CS2 * (res.tau - 0.5)
    ratio = float(nu_t[~sim.solid].max()) / nu
    # Re 100 by construction, so the Re gate is open and the ratio is what
    # decides -- which is precisely the claim being tested.
    band = band_for(_fake_plan(100.0), ratio * ((0.6 - 0.5) / 3.0))
    return QualitativeEvidence(
        ratio=ratio,
        band=band,
        cd=res.cd_mean,
        st=res.st,
        passed=passed,
        seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Clause 6 — D-038's own case, through the command the contract names
# ---------------------------------------------------------------------------


def check_d038_cli(backend: str) -> tuple[bool, list[str], str]:
    """``python -m flow --fluid air --speed "20 m/s" --size "1.5 m"`` runs.

    The T204 contract names the **command**, so the rung runs the command:
    :func:`flow.cli.main` with the literal flags, capturing what a user would
    see. What is checked is the exit code, the band, the absence of a ``Cd`` and
    the presence of a sentence saying what is and is not being shown — in the
    user's own units.

    Returns:
        ``(ok, failures, captured_text)``.
    """
    import contextlib
    import io

    from flow.cli import main as cli_main

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli_main(
            [
                "--shape",
                "tests/data/shapes/disc.png",
                "--fluid",
                "air",
                "--speed",
                "20 m/s",
                "--size",
                "1.5 m",
                "--quality",
                "fast",
                "--no-live",
                "--backend",
                backend,
            ]
        )
    text = buf.getvalue()

    bad: list[str] = []
    if code != 0:
        bad.append(f"exit code {code}, expected 0 (the case must complete)")
    if "illustrative" not in text:
        bad.append("the output never says 'illustrative'")
    if "not reported" not in text:
        bad.append("the output does not say the drag coefficient is not reported")
    # In the user's own units, as they typed them (D-045).
    for token in ("20 m/s", "1.5 m"):
        if token not in text:
            bad.append(f"the output does not restate {token!r} in the user's units")
    return not bad, bad, text


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run Rung H and print PASS/FAIL. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Rung H — every fidelity band's claim is true, and no run "
        "outside the quantitative band emits an unqualified Cd"
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="the T101 backend every case runs on (default numpy, the "
        "reference oracle -- D-043). The whole rung is ~3 min on warp and "
        "~50 min on numpy; run the long one detached.",
    )
    parser.add_argument(
        "--skip-cylinder",
        action="store_true",
        help="skip clause 5, Rung 3's own case with the closure forced on. It "
        "is the expensive clause and it is Q-203's evidence, so a skipped one "
        "is reported and the rung is not a full pass.",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="skip clauses 4 and 6, the runs. Leaves the table and the "
        "expectation clauses, which need no timesteps at all -- a few "
        "milliseconds, and enough to catch a broken table.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    backend = args.backend
    try:
        get_backend(backend)
    except BackendUnavailableError as exc:
        print(f"SKIP - {exc}")
        return 2

    print("Rung H — the fidelity bands (DOCS/IDEA4.md § Validation ladder)")
    print(f"  backend {backend}   boundaries: Re <= {RE_3D_ONSET:g} "
          f"(Williamson 1996), max(nu_t)/nu < {RATIO_QUANTITATIVE:g} and "
          f"< {RATIO_QUALITATIVE:g} (D-082)")
    print(f"  the closure engages at tau <= {TAU_FLOOR} (D-029) and nowhere "
          f"else, so a case that fits under BGK is banded from Re alone")
    print()

    checks: list[tuple[str, bool, str]] = []

    # --- clause 1 ---------------------------------------------------------
    print("1. band_for IS the table, and both of its gates have teeth")
    table = check_table()
    for re, ratio, expected, got in table.rows:
        mark = "ok" if got is expected else "XX"
        print(f"   [{mark}] Re {re:>9.4g}   max(nu_t)/nu {ratio:>9.4g}   -> {got}")
    checks.append(
        (
            f"the table, {len(table.rows)} points straddling both boundaries",
            not table.wrong,
            "; ".join(table.wrong) if table.wrong else "every verdict is the table's",
        )
    )
    for label, caught in table.inversion_caught.items():
        print(
            f"   [{'ok' if caught else 'XX'}] deleting {label} changes a verdict "
            "(D-091: a condition nothing can violate is not tested)"
        )
        checks.append(
            (f"deleting {label} breaks the table", caught, "discriminator"),
        )
    print()

    # --- clause 2 ---------------------------------------------------------
    print("2. expected before the run, earned after it")
    for name, ok, detail in check_expectation():
        print(f"   [{'ok' if ok else 'XX'}] {name}   {detail}")
        checks.append((name, ok, detail))
    print()

    # --- clauses 4 and 6 --------------------------------------------------
    sweeps: list[SweepResult] = []
    if not args.skip_sweep:
        print(f"3. the sweep — {len(SWEEP)} product-path runs on {backend}")
        for case in SWEEP:
            print(f"   running: {case.name} ...", flush=True)
            sweep = run_sweep_case(case, backend)
            sweeps.append(sweep)
            r, p = sweep.result, sweep.plan
            print(
                f"     Re {p.Re:<11.4g} tau {p.tau:.6f}  Cs {p.cs_smag:g}  "
                f"max(nu_t)/nu {r.nu_t_ratio:.4g}  peak|u| {r.peak_u:.4f}"
            )
            print(
                f"     expected {p.expected_fidelity} -> earned {r.fidelity}   "
                f"Cd {r.cd if r.cd is not None else 'not reported'}   "
                f"({sweep.seconds:.0f} s)"
            )
            print(f"     why in the sweep: {case.note}")
            for line in sweep.failures:
                print(f"     [XX] {line}")
            checks.append(
                (
                    f"{case.name}",
                    not sweep.failures,
                    f"earned {r.fidelity}, {len(sweep.failures)} failure(s)",
                )
            )
        print()

        print("4. D-038's own case, through the command the contract names")
        ok, bad, text = check_d038_cli(backend)
        for line in text.splitlines():
            if line.strip():
                print(f"   | {line}")
        for line in bad:
            print(f"   [XX] {line}")
        checks.append(
            (
                "python -m flow --fluid air --speed '20 m/s' --size '1.5 m' "
                "runs, exits 0 and reports illustrative with no Cd",
                ok,
                "; ".join(bad) if bad else "exit 0, illustrative, no Cd",
            )
        )
        print()
    else:
        print("3-4. SKIPPED by --skip-sweep: no case was run.")
        print()

    # --- clause 5 ---------------------------------------------------------
    if not args.skip_cylinder:
        print("5. Q-203's evidence: a QUALITATIVE run reproduces published data")
        print("   (Rung 3's own harness with the closure forced on — D-087)")
        print()
        evidence = check_qualitative_claim(backend)
        print()
        print("  (back in Rung H)")
        cd_lo, cd_hi = CD_BAND
        st_lo, st_hi = ST_BAND
        print(
            f"   max(nu_t)/nu {evidence.ratio:.4f} -> band {evidence.band}; "
            f"Cd {evidence.cd:.4f}, St {evidence.st:.4f} "
            f"({evidence.seconds:.0f} s)"
        )
        checks.extend(
            [
                (
                    "Rung 3 at Cs = 0.17 lands in the QUALITATIVE band",
                    evidence.band is Band.QUALITATIVE,
                    f"max(nu_t)/nu {evidence.ratio:.4f} at Re 100",
                ),
                (
                    f"...and still prints Cd {cd_lo}-{cd_hi}, St {st_lo}-{st_hi} "
                    "(Rung 3's published bands, unwidened)",
                    evidence.passed
                    and cd_lo <= evidence.cd <= cd_hi
                    and st_lo <= evidence.st <= st_hi,
                    f"Cd {evidence.cd:.4f}, St {evidence.st:.4f}",
                ),
            ]
        )
        print()
    else:
        print("5. SKIPPED by --skip-cylinder: Q-203's evidence was not measured.")
        print()

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"    [{'ok' if ok else 'XX'}] {name.ljust(width)}   {detail}")

    passed = all(ok for _, ok, _ in checks)
    print()
    if args.skip_sweep or args.skip_cylinder:
        print("  NOTE: a clause was skipped, so this is not a full Rung H pass.")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
