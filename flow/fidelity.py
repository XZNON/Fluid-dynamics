"""``flow.fidelity`` — how much a run's answer is worth, decided per run (T204).

``DOCS/IDEA4.md`` § The five things Phase 2 must get right, item 1 — *"the
closure buys stability, not fidelity, and the tool says which"*. This module is
that sentence made into code, and ``CLAUDE.md`` constraint 18 is the rule it
enforces: **no unqualified quantitative claim outside the validated band.**

The three bands, verbatim from the spec's table
------------------------------------------------

=================  =====================================  ==========================================
Band               Condition                              What the tool prints
=================  =====================================  ==========================================
**quantitative**   ``Re <= 200`` **and** ``r < 0.1``       the numbers, unqualified
**qualitative**    ``r < 1``                              the wake, the picture, the trends;
                                                          ``Cd`` **qualified**, never bare
**illustrative**   otherwise                              a moving picture and no quantitative claim
=================  =====================================  ==========================================

where ``r = max(nu_t) / nu`` is measured **from the run**, not read off a
Reynolds-number table (**D-082**). ``nu_t`` is the Smagorinsky eddy viscosity
field :func:`lbm.probe.eddy_viscosity` derives through ``tau_eff``, and ``nu``
is the fluid's own ``(tau - 0.5) / 3`` (constraint 2). The outer boundary is
``r = 1`` because that is **the point where the model supplies more viscosity
than the fluid does** — a statement about *this run* that a test can evaluate,
rather than a magic Reynolds number argued into a table.

Why the top band also has a Reynolds gate, and why it is cited rather than chosen
---------------------------------------------------------------------------------
The cylinder wake becomes three-dimensional at **Re ~ 190** — Williamson (1996),
*"Vortex dynamics in the cylinder wake"*, Annu. Rev. Fluid Mech. 28, 477–539, the
mode-A instability — which is exactly why Rung 3 sits at Re 100. Above it a 2D
run is wrong about *the flow*, not about the numerics, and no two-dimensional
closure repairs it: Smagorinsky descends from Kolmogorov's forward energy
cascade and 2D turbulence cascades energy the other way (Kraichnan 1967). So
``Re <= 200`` is a physics gate with a citation, and ``r`` is a numerics gate
with a measurement; a run has to clear **both** to have its numbers reported
bare.

What this module does **not** do
---------------------------------
It computes no viscosity. ``nu_t`` is derived by :func:`lbm.probe.eddy_viscosity`
through ``tau_eff`` and nothing here assigns one (constraint 2). It is also not a
knob: ``Cs`` is planned by :func:`flow.autoconfig.plan` and printed, never typed
(constraint 13), and **the band is what surfaces to the user instead**.

Q-203, answered
----------------
*"Can the fidelity bands be made falsifiable enough to report a qualified ``Cd``
outside the quantitative band, or does the closure ship stability-only?"*
They can, and the evidence is a measurement that already existed before this
module did: Rung F runs **Rung 3's own case** with ``Cs = 0.17``, which sits at
``r = 0.1910`` — *inside* the qualitative band — and prints ``Cd`` 1.4143 and
``St`` 0.1719 against Rung 3's published, unwidened 1.25–1.45 and 0.155–0.175, on
**both** backends. So the qualitative band ships a :class:`Qualified` ``Cd``:
the number, with its band and its caveat welded to it, and never as a bare
attribute. ``validate/fidelity.py`` (Rung H) re-measures that clause rather than
citing it. The **illustrative** band ships stability-only, which is
``DOCS/PLAN3.md`` § Risks' pressure valve applied where it is actually true: a
moving picture, and no coefficient anywhere on the object.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flow.autoconfig import Plan

__all__ = [
    "Band",
    "Qualified",
    "RE_3D_ONSET",
    "RATIO_QUANTITATIVE",
    "RATIO_QUALITATIVE",
    "band_for",
    "sentence",
    "ratio_for",
]


# ---------------------------------------------------------------------------
# The two boundaries — one cited, one measured
# ---------------------------------------------------------------------------

#: The Reynolds number above which a 2D answer is wrong about the flow itself.
#: **Williamson (1996)**: the cylinder wake's mode-A instability sets in at
#: Re ~ 190, so the wake is three-dimensional above it and a two-dimensional
#: simulation of any quality is modelling something that does not happen. 200 is
#: that number rounded **up to the nearest round figure**, which is the only
#: freedom taken with it; it is not tuned and it does not move (**D-082**).
#: Rung 3, the benchmark this project reproduces, sits at Re 100.
RE_3D_ONSET: float = 200.0

#: ``max(nu_t) / nu`` below which the closure is a rounding error on the
#: fluid's own viscosity and the run may report its numbers bare. A tenth: the
#: closure is contributing under 10% of the momentum diffusion, which is inside
#: the spread Rung 3's own published band already carries.
RATIO_QUANTITATIVE: float = 0.1

#: ``max(nu_t) / nu`` at or above which nothing quantitative is claimed at all.
#: **One**, and the reason it is one is the whole of **D-082**: at ``r = 1`` the
#: model supplies exactly as much viscosity as the fluid does, so above it the
#: answer is more model than physics. It is a property of the run and a test can
#: evaluate it.
RATIO_QUALITATIVE: float = 1.0


class Band(Enum):
    """How much of a run's answer may be reported, and how.

    Ordered from most to least trustworthy; :meth:`worse_of` uses that order to
    settle a disagreement between the band a plan **expected** and the band a
    run **earned** (the earned one always wins, and it is never allowed to be
    the more optimistic of the two by accident).

    Attributes:
        QUANTITATIVE: ``Re <= 200`` and ``max(nu_t)/nu < 0.1`` — the range the
            fourteen rungs validate. Numbers reported unqualified.
        QUALITATIVE: ``max(nu_t)/nu < 1`` — the wake, the picture, the trends.
            ``Cd`` is reported through :class:`Qualified` and never bare.
        ILLUSTRATIVE: otherwise — the closure supplies more viscosity than the
            fluid does. A moving picture and no quantitative claim at all.
    """

    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    ILLUSTRATIVE = "illustrative"

    @property
    def rank(self) -> int:
        """0 for :attr:`QUANTITATIVE`, 2 for :attr:`ILLUSTRATIVE` — bigger is worse."""
        return _RANK[self]

    @property
    def reports_bare_numbers(self) -> bool:
        """Whether :class:`flow.report.Result` may expose ``cd`` as a float.

        Constraint 18 in one predicate: **only** the quantitative band. This is
        the property ``flow/report.py`` gates on, so there is exactly one place
        the rule is written down.
        """
        return self is Band.QUANTITATIVE

    @property
    def reports_qualified_numbers(self) -> bool:
        """Whether a :class:`Qualified` drag coefficient is emitted at all.

        True for the quantitative and qualitative bands; **False** for
        :attr:`ILLUSTRATIVE`, where there is no drag coefficient anywhere on the
        result — the stability-only half of Q-203's answer.
        """
        return self is not Band.ILLUSTRATIVE

    @staticmethod
    def worse_of(a: "Band", b: "Band") -> "Band":
        """The less trustworthy of two bands.

        Args:
            a: one band.
            b: the other.

        Returns:
            Whichever of the two has the higher :attr:`rank`.
        """
        return a if a.rank >= b.rank else b

    def __str__(self) -> str:
        return self.value


_RANK: dict[Band, int] = {
    Band.QUANTITATIVE: 0,
    Band.QUALITATIVE: 1,
    Band.ILLUSTRATIVE: 2,
}


# ---------------------------------------------------------------------------
# A number that cannot be read without its caveat
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qualified:
    """A drag coefficient with its band welded on — the qualitative band's ``Cd``.

    ``DOCS/IDEA4.md``'s table says the qualitative band prints ``Cd``
    **qualified, never bare**. This is what "qualified" is: a frozen record whose
    :attr:`cd` cannot be reached without naming the object that carries the
    caveat, whose ``str()`` *is* the caveat, and which deliberately implements no
    ``__float__`` — so it cannot be slipped into arithmetic or a format string as
    if it were a validated coefficient.

    :class:`flow.report.Result` exposes one of these as ``cd_qualified`` in the
    quantitative and qualitative bands and **``None``** in the illustrative one
    (constraint 18, and :attr:`Band.reports_qualified_numbers` is the predicate).

    Attributes:
        cd: mean drag coefficient over the measurement window.
        cd_std: its standard deviation over the same window.
        cl: lift-coefficient amplitude (half peak-to-peak), measured raw.
        strouhal: ``St = f D / U``, or ``None`` when shedding was not detected.
        band: the band this run earned.
        caveat: one sentence saying what the number is and is not worth.
    """

    cd: float
    cd_std: float
    cl: float
    strouhal: float | None
    band: Band
    caveat: str

    def __str__(self) -> str:
        st = "None" if self.strouhal is None else f"{self.strouhal:.4f}"
        return (
            f"Cd {self.cd:.4f} +- {self.cd_std:.4f}, St {st} "
            f"[{self.band}] {self.caveat}"
        )

    def as_dict(self) -> dict[str, object]:
        """The record as a plain dict, caveat and band included.

        The caveat travels with the number here for the same reason it does in
        :meth:`__str__`: a caller that logs or serialises this must not be able
        to keep the value and drop the qualification by accident.
        """
        return {
            "cd": self.cd,
            "cd_std": self.cd_std,
            "cl": self.cl,
            "strouhal": self.strouhal,
            "band": self.band.value,
            "caveat": self.caveat,
        }


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def ratio_for(plan: "Plan", nu_t_max: float) -> float:
    """``max(nu_t) / nu`` — the number the two upper boundaries are read off.

    The fluid's own lattice viscosity is ``nu = (tau - 0.5) / 3`` (constraint 2,
    and it is derived here rather than stored anywhere, for the same reason).

    Args:
        plan: the :class:`flow.autoconfig.Plan` the run was configured from.
        nu_t_max: the largest eddy viscosity the run generated, lattice units,
            from :func:`lbm.probe.eddy_viscosity`'s ``(ny, nx)`` field.

    Returns:
        The dimensionless ratio, ``>= 0``.

    Raises:
        ValueError: if ``plan.tau <= 0.5`` (``nu <= 0``, which constraint 2 says
            cannot happen) or ``nu_t_max < 0`` (the closure only ever adds).
    """
    nu = (float(plan.tau) - 0.5) / 3.0
    if nu <= 0.0:
        raise ValueError(
            f"tau = {plan.tau!r} gives nu = {nu:.6g} <= 0; CLAUDE.md "
            "constraint 2 says nu = (tau - 0.5) / 3 and tau -> 0.5 means the "
            "sim blows up, so there is no band to decide."
        )
    if nu_t_max < 0.0:
        raise ValueError(
            f"nu_t_max = {nu_t_max!r} is negative; the Smagorinsky closure adds "
            "viscosity and never removes it (lbm.probe.eddy_viscosity asserts "
            "nu_t >= 0)."
        )
    return float(nu_t_max) / nu


def band_for(plan: "Plan", nu_t_max: float | None = None) -> Band:
    """Which band a case is in — expected before a run, earned after one.

    ``DOCS/IDEA4.md`` § The five things Phase 2 must get right (1), the table
    verbatim, with ``r = max(nu_t) / nu``:

    * **quantitative** iff ``Re <= RE_3D_ONSET`` **and** ``r < 0.1``;
    * **qualitative** iff ``r < 1``;
    * **illustrative** otherwise.

    The ``Re <= 200`` gate is **Williamson (1996)**'s mode-A instability at
    Re ~ 190 (see the module docstring), not a tuned number.

    **Before a run** (``nu_t_max is None``) this returns the band the plan
    *expects*, from ``Re`` and the plan's own closure setting:

    * closure **off** — ``r`` is not unknown, it is **exactly zero**
      (:func:`lbm.probe.eddy_viscosity` returns an all-zero field at
      ``cs_smag = 0``, which T201 asserts and Rung G measured), so the verdict is
      decided by ``Re`` alone and it is a fact rather than a forecast;
    * closure **on** — ``r`` is unknown and non-zero, so the most the plan may
      claim is **qualitative**. A run that then earns ``illustrative`` is a
      finding, and :meth:`flow.report.Result.summary` says so.

    Args:
        plan: the :class:`flow.autoconfig.Plan`. Read for ``Re``, ``tau`` and
            ``cs_smag`` only.
        nu_t_max: the largest eddy viscosity the run generated, lattice units, or
            ``None`` for the pre-run expectation. It is a **measurement**, never
            something a user supplies — the caller is
            :meth:`flow.case.Case.run`, which reads it off
            :func:`lbm.probe.eddy_viscosity` at probe cadence.

    Returns:
        The :class:`Band`.

    Raises:
        ValueError: through :func:`ratio_for`, for a ``tau`` at or below 0.5 or a
            negative ``nu_t_max``.
    """
    quantitative_re = float(plan.Re) <= RE_3D_ONSET

    if nu_t_max is None:
        if float(getattr(plan, "cs_smag", 0.0)) == 0.0:
            # Not a forecast: with the closure off nu_t is identically zero, so
            # r = 0 < 0.1 is known, and Re is the only open question.
            return Band.QUANTITATIVE if quantitative_re else Band.QUALITATIVE
        # The closure is on, so r > 0 and unknown. Quantitative is not claimable
        # and illustrative is not yet earned.
        return Band.QUALITATIVE

    ratio = ratio_for(plan, nu_t_max)
    if quantitative_re and ratio < RATIO_QUANTITATIVE:
        return Band.QUANTITATIVE
    if ratio < RATIO_QUALITATIVE:
        return Band.QUALITATIVE
    return Band.ILLUSTRATIVE


# ---------------------------------------------------------------------------
# The prose — and D-047's posture about it
# ---------------------------------------------------------------------------

#: One sentence per band, for the person who has never heard of a Reynolds
#: number (the same audience ``flow/diagnose.py``'s ``_FIRST_PARAGRAPH`` is
#: written for, and the same rule: **no lattice quantity in it**).
#:
#: ``DOCS/TASKS3.md`` § T204 Notes: *"``sentence(band)`` is prose and prose is
#: not what the rung tests (D-047's posture). What is tested is the verdict and
#: the absence of the number."* So Rung H checks that a sentence exists, that it
#: names the band, and that it leaks no lattice quantity — never its wording.
_SENTENCES: dict[Band, str] = {
    Band.QUANTITATIVE: (
        "quantitative: the numbers below are this tool's validated output. The "
        "flow is smooth enough, and resolved finely enough, that the simulator "
        "is reproducing published measurements of the same flow -- the drag, "
        "the lift and the shedding frequency are all reported as measured."
    ),
    Band.QUALITATIVE: (
        "qualitative: the picture is right and the numbers are indicative. A "
        "turbulence model is carrying part of this flow, so the wake, the "
        "shedding and how they change when you change the shape are all worth "
        "watching -- but the drag coefficient is reported with that attached to "
        "it, and it is not a measurement you should quote."
    ),
    Band.ILLUSTRATIVE: (
        "illustrative: a moving picture, and no numbers at all. This flow is far "
        "more energetic than a two-dimensional simulation can represent -- a "
        "turbulence model is supplying more of the resistance to motion than the "
        "fluid itself is, and the real wake at this speed and size is "
        "three-dimensional, which nothing computed on a flat grid can show you. "
        "Watch it for the shape of the flow. No drag coefficient is reported, "
        "because any number here would be a convincing-looking guess."
    ),
}


def sentence(band: Band) -> str:
    """What a band means, in one sentence, in plain language.

    Args:
        band: the :class:`Band`.

    Returns:
        The sentence, beginning with the band's own name.

    Raises:
        KeyError: for something that is not a :class:`Band` — deliberately loud,
            because a band with no sentence is a verdict a user cannot act on.
    """
    return _SENTENCES[band]
