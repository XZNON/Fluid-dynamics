"""Diagnosis: a refusal in plain language, a fix that is tested to work.

``DOCS/IDEA3.md`` § The five things Phase 1 must get right (2) — *"Refusal is a
feature, and it comes with a way forward"* — and **D-045**, which turns that
paragraph into a policy: refuse the case as asked, explain it in the user's
units, offer the nearest runnable case, and never substitute silently.

``flow/autoconfig.py`` already raises the structured half of that: an
:class:`~flow.autoconfig.Unrepresentable` carries ``reason``, ``quantity``,
``value``, ``limit`` and a list of :class:`~flow.autoconfig.Suggestion`. This
module is the other half. It does three things and nothing else:

:func:`explain`
    Turns a refusal into prose. The **first paragraph carries no lattice
    quantity** — no ``tau``, no lattice velocity, no cell count — because the
    stated target user has never heard of a Reynolds number
    (``DOCS/TASKS2.md`` § T106 Notes). The numbers live under ``Details``, for
    the reader who has.

:func:`suggest`
    Takes the physical request, and returns the ranked list of ways to make it
    run. Each suggestion is a *testable claim* (**D-045**): Rung D
    (``validate/refusals.py``) feeds the top one back through
    :func:`~flow.autoconfig.plan` via :func:`apply_suggestion` and runs it, so
    a suggestion that does not fix its case is a failing test rather than a
    wording problem.

:class:`Monitor`
    A ``per_step``-compatible probe (**D-025**) that raises :class:`Diverging`
    with a named cause *before* the run reaches ``nan``. Its three tripwires
    are the failure modes of ``DOCS/IDEA2.md`` § Stability.

Why the suggestion vocabulary is ranked the way it is
------------------------------------------------------

``tau = 0.5 + 3 U_lattice N / Re``. Only one knob — resolution, which the user
sees as ``quality`` — raises ``tau`` **without changing the flow**. Changing
the speed, the size or the fluid changes the Reynolds number, which means
running a *different case*: dynamically similar to nothing the user asked for.
So :data:`SUGGESTION_ORDER` puts the case-preserving fix first, and every
case-changing suggestion returned by :func:`suggest` carries the words *"not
your case"* in its note — the label **D-045** asks for, attached to the object
rather than to one rendering of it.

What is deliberately not here
------------------------------

``flow.Result`` / ``flow.report`` and the video metadata are **T108/T109**.
The T106 contract's ``substituted=True`` criterion names all three, none of
which exist yet; per ``CLAUDE.md`` § Session protocol the conflict is logged in
``DOCS/STATE2.md`` § Decisions and the criterion is carried into T108 rather
than satisfied here against a stub.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from lbm.geometry import circle
from lbm.units import U_LATTICE_DEFAULT, U_LATTICE_MAX

from flow.autoconfig import (
    QUALITY_CELLS,
    TAU_FLOOR,
    Suggestion,
    Unrepresentable,
    plan,
)
from flow.fluids import FLUIDS, Fluid
from flow.fluids import fluid as _fluid_by_name
from flow.quantity import Quantity

__all__ = [
    "Diverging",
    "Monitor",
    "REFUSAL_CLASSES",
    "SUGGESTION_ORDER",
    "EXAMPLE_MASK",
    "apply_suggestion",
    "classify",
    "explain",
    "suggest",
]


# ---------------------------------------------------------------------------
# Refusal classes
# ---------------------------------------------------------------------------

#: Every class of refusal :func:`flow.autoconfig.plan` can raise, keyed by the
#: ``quantity`` field of the :class:`~flow.autoconfig.Unrepresentable` it
#: raises. Rung D iterates this tuple, so a new guardrail in ``autoconfig``
#: that forgets to register here fails :func:`classify` loudly rather than
#: quietly going unexplained.
REFUSAL_CLASSES: tuple[str, ...] = (
    "relaxation",
    "thickness",
    "empty_mask",
    "speed_ceiling",
    "blockage",
)

_QUANTITY_TO_CLASS: dict[str, str] = {
    "tau": "relaxation",
    "min solid thickness (cells, scaled to the chosen resolution)": "thickness",
    "solid cells (excluding domain border)": "empty_mask",
    "peak |u_lattice|": "speed_ceiling",
    "blockage": "blockage",
}

#: Which kind of change is offered first. The case-preserving fix
#: (``quality`` — the same flow, resolved more finely) outranks every fix that
#: changes the Reynolds number, because those answer a different question
#: (**D-045**: *"the same shape at a lower Reynolds number, clearly labelled as
#: not your case"*). ``mask`` sits second because a refusal about the picture
#: is not fixed by any physics knob at all.
SUGGESTION_ORDER: tuple[str, ...] = ("quality", "mask", "fluid", "speed", "size")

#: The changes that produce a **different flow** from the one asked for. Their
#: notes are labelled, and constraint 16 is why: a substituted answer that
#: looks like the requested one is the failure mode with a friendlier face.
_CHANGES_THE_CASE: frozenset[str] = frozenset({"fluid", "speed", "size"})

#: A worked example shape, used when the refusal is *"there is no body in this
#: picture"*. The tool cannot invent the user's geometry, so the suggestion it
#: makes is "a picture with a solid shape in it, like this one" — and Rung D
#: runs exactly this one, which is what keeps that suggestion a testable claim
#: rather than an instruction. Same disc ``validate/autoconfig.py`` holds fixed
#: for Rung B.
EXAMPLE_MASK: NDArray[np.bool_] = circle(80, 80, 40.0, 40.0, 20.0)

_NOT_YOUR_CASE = (
    "This is a different flow from the one you asked for -- not your case."
)


def classify(exc: Unrepresentable) -> str:
    """Which of :data:`REFUSAL_CLASSES` this refusal belongs to.

    Args:
        exc: the refusal raised by :func:`flow.autoconfig.plan`.

    Returns:
        One of :data:`REFUSAL_CLASSES`.

    Raises:
        ValueError: if ``exc.quantity`` is not one this module knows how to
            explain. That is deliberately loud: an unexplained refusal is a
            dead end, which is the exact thing **D-045** exists to remove.
    """
    try:
        return _QUANTITY_TO_CLASS[exc.quantity]
    except KeyError:  # pragma: no cover - guarded by a test with teeth
        raise ValueError(
            f"no explanation registered for a refusal about {exc.quantity!r}; "
            f"flow/diagnose.py knows {sorted(_QUANTITY_TO_CLASS)}. Add it "
            "there rather than letting a refusal reach a user unexplained "
            "(D-045)."
        ) from None


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------

#: First paragraph per refusal class. Written for someone who has never heard
#: of a Reynolds number (``DOCS/TASKS2.md`` § T106 Notes) and therefore
#: containing **no lattice quantity at all** — asserted by
#: ``tests/test_diagnose.py``. ``{case}`` is the user's own request, restated
#: in their own units, or an empty string when :func:`explain` was not given
#: one.
_FIRST_PARAGRAPH: dict[str, str] = {
    "relaxation": (
        "{case}is far more energetic than this simulator can represent. At "
        "that combination of speed, size and fluid the flow is turbulent, and "
        "this tool models smooth, orderly flow only -- it has no turbulence "
        "model. Running it anyway would produce a convincing-looking video of "
        "the wrong answer, so it refuses instead."
    ),
    "thickness": (
        "{case}has a feature too thin for this simulator to hold on to. Fluid "
        "would leak straight through it and the wake would come out wrong, in "
        "a way that still looks plausible, so the tool refuses rather than "
        "showing it to you."
    ),
    "empty_mask": (
        "{case}contains no shape for the fluid to flow around -- the picture "
        "is empty, or it is solid edge to edge. There is nothing here to "
        "simulate."
    ),
    "speed_ceiling": (
        "{case}would drive the fluid past the speed this simulator's model of "
        "a fluid stays accurate at. The flow speeds up as it squeezes around "
        "the body, and that peak, not the speed you typed, is what has to "
        "stay in range."
    ),
    "blockage": (
        "{case}puts a body so large in a channel so narrow that the walls "
        "would squeeze the flow and change the answer. What you would be "
        "watching is the walls, not the shape."
    ),
}


def _case_clause(request: Mapping[str, Any] | None) -> str:
    """``"Air at 20 m/s past a body 1.5 m across "`` — or ``""``.

    Physical units only, exactly as the user gave them (**D-045**: explain in
    the user's units). The trailing space is deliberate: the templates in
    :data:`_FIRST_PARAGRAPH` continue the sentence.
    """
    if not request:
        return "This case "
    fluid_name = request.get("fluid")
    if isinstance(fluid_name, Fluid):
        fluid_name = fluid_name.name
    speed = request.get("speed")
    size = request.get("size")
    parts: list[str] = []
    if fluid_name:
        parts.append(str(fluid_name).capitalize())
    if speed is not None:
        parts.append(f"at {_human(speed)}")
    if size is not None:
        parts.append(f"past a body {_human(size)} across")
    if not parts:
        return "This case "
    return " ".join(parts) + " "


def _human(value: Any) -> str:
    """What the user typed, if we still have it; otherwise the SI value."""
    if isinstance(value, Quantity):
        return value.given or str(value)
    return str(value)


def _render_suggestions(suggestions: Iterable[Suggestion]) -> list[str]:
    lines: list[str] = []
    for i, s in enumerate(suggestions, start=1):
        lines.append(f"  {i}. {s.note}")
    return lines


def explain(
    exc: Unrepresentable, *, request: Mapping[str, Any] | None = None
) -> str:
    """A refusal, in plain language first and numbers second.

    ``DOCS/IDEA3.md`` § 2. The layout is fixed and tested:

    1. **One paragraph** a non-specialist can act on. No ``tau``, no lattice
       velocity, no cell count — ``tests/test_diagnose.py`` greps for them.
    2. **What would work** — the suggestions, in :data:`SUGGESTION_ORDER`,
       each already carrying its own physical sentence and, where the fix
       changes the flow, the words *"not your case"*.
    3. **Details** — the guardrail that was violated, its number and its
       limit, plus the ``reason`` verbatim so the citing decision (**D-029**
       and friends) is one grep away.

    Args:
        exc: the refusal from :func:`flow.autoconfig.plan`.
        request: optionally what the user asked for — ``{"fluid": ...,
            "speed": ..., "size": ...}`` — so the first sentence can restate
            it in their own units. Values may be :class:`~flow.quantity.Quantity`,
            strings or a :class:`~flow.fluids.Fluid`.

    Returns:
        The whole message as one string, ready to print.
    """
    kind = classify(exc)
    first = _FIRST_PARAGRAPH[kind].format(case=_case_clause(request))

    ordered = _present(exc.suggestions)
    out: list[str] = [first, ""]
    if ordered:
        out.append("What would work")
        out.extend(_render_suggestions(ordered))
        out.append("")
    out.append("Details")
    out.append(f"  refused because: {exc.reason}")
    out.append(f"  {exc.quantity} = {exc.value:.6g} (limit {exc.limit:.6g})")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# suggest()
# ---------------------------------------------------------------------------


def _rank(suggestions: Iterable[Suggestion]) -> list[Suggestion]:
    """Stable sort by :data:`SUGGESTION_ORDER`; unknown kinds keep their place."""
    items = list(suggestions)
    return sorted(
        items,
        key=lambda s: (
            SUGGESTION_ORDER.index(s.change)
            if s.change in SUGGESTION_ORDER
            else len(SUGGESTION_ORDER)
        ),
    )


def _plain_note(suggestion: Suggestion) -> str:
    """One sentence on what this changes **physically**, in the user's words.

    ``flow/autoconfig.py`` writes its notes for the person maintaining the
    guardrail — they name ``tau`` and the checkerboard floor. This module
    writes the same fix for the person who has never heard of a Reynolds
    number (``DOCS/TASKS2.md`` § T106 Notes); the guardrail's own wording is
    still printed verbatim, one section down, under ``Details``.
    """
    change = suggestion.change
    value = suggestion.value
    if change == "speed":
        return (
            f"Run it slower -- about {_si(value)} m/s -- at the same size and "
            "in the same fluid"
        )
    if change == "size":
        return (
            f"Run a smaller body -- about {_si(value)} m across -- at the same "
            "speed and in the same fluid"
        )
    if change == "quality":
        return (
            f'Ask for quality="{value}": the same flow you asked for, resolved '
            "in more detail"
        )
    # "fluid" and "mask" suggestions are written in plain language where they
    # are built, so they pass through unchanged.
    return suggestion.note


def _si(value: Any) -> str:
    """The number of a suggestion's replacement, in SI, three digits."""
    return f"{value.si:.3g}" if isinstance(value, Quantity) else str(value)


def _present(suggestions: Iterable[Suggestion]) -> list[Suggestion]:
    """Rank, reword in plain language, and label what is *not your case*.

    The label goes on the **object**, not on one rendering of it: constraint 16
    is about every artifact saying so, and a note that only the pretty-printer
    knows about is a note the report and the video metadata will not carry.
    """
    out: list[Suggestion] = []
    for s in _rank(suggestions):
        note = _plain_note(s)
        if s.change in _CHANGES_THE_CASE and _NOT_YOUR_CASE not in note:
            note = f"{note}. {_NOT_YOUR_CASE}"
        elif not note.endswith("."):
            note = f"{note}."
        out.append(dataclasses.replace(s, note=note))
    return out


def _viscous_fluid_suggestions(
    *, u_phys: float, l_phys: float, cells_per_length: int, current: str
) -> list[Suggestion]:
    """Library fluids that clear :data:`~flow.autoconfig.TAU_FLOOR` unchanged.

    Never invents a viscosity: it walks :data:`flow.fluids.FLUIDS` and keeps
    only the entries whose cited ``nu`` actually makes the arithmetic work, so
    the suggestion is checkable before it is offered (**D-045**) and Rung D can
    run it.
    """
    margin = 0.01
    target = TAU_FLOOR + margin
    # tau = 0.5 + 3 U_lattice N nu / (u_phys l_phys) > target
    nu_needed = (target - 0.5) * u_phys * l_phys / (
        3.0 * U_LATTICE_DEFAULT * cells_per_length
    )
    out: list[Suggestion] = []
    for name, f in sorted(FLUIDS.items(), key=lambda kv: kv[1].nu.si):
        if name == current or f.nu.si < nu_needed:
            continue
        out.append(
            Suggestion(
                change="fluid",
                value=name,
                note=(
                    f"Run the same shape at the same speed and size in "
                    f"{name}, which is about "
                    f"{f.nu.si / max(nu_needed, 1e-300):.3g}x thicker than "
                    "this case needs -- thick enough that the flow stays "
                    "smooth and orderly"
                ),
            )
        )
        break  # the least viscous fluid that works: the smallest real change
    return out


def suggest(
    *,
    fluid: "Fluid | str",
    speed: "Quantity | str | float",
    size: "Quantity | str | float",
    mask: NDArray[np.bool_],
    quality: str = "balanced",
) -> list[Suggestion]:
    """Ranked, checkable ways to make a refused request run.

    Args:
        fluid: a :class:`~flow.fluids.Fluid` or a library name.
        speed: free-stream speed, as :func:`flow.autoconfig.plan` accepts it.
        size: the body's characteristic length, likewise.
        mask: ``(h, w)`` bool array — read only for the object's proportions.
        quality: one of :data:`flow.autoconfig.QUALITY_LEVELS`.

    Returns:
        ``[]`` if the case already plans — there is nothing to fix, and saying
        so with an empty list rather than a fabricated alternative is the
        point. Otherwise the suggestions in :data:`SUGGESTION_ORDER`, each
        with a physical sentence and, where it changes the flow, the words
        *"not your case"*.
    """
    f = fluid if isinstance(fluid, Fluid) else _fluid_by_name(fluid)
    try:
        plan(fluid=f, speed=speed, size=size, mask=mask, quality=quality)
    except Unrepresentable as exc:
        found = list(exc.suggestions)
        if classify(exc) == "relaxation":
            found.extend(
                _viscous_fluid_suggestions(
                    u_phys=Quantity(speed, default_unit="m/s").si,
                    l_phys=Quantity(size, default_unit="m").si,
                    cells_per_length=QUALITY_CELLS[quality],
                    current=f.name,
                )
            )
        return _present(found)
    return []


def apply_suggestion(
    suggestion: Suggestion,
    *,
    fluid: "Fluid | str",
    speed: "Quantity | str | float",
    size: "Quantity | str | float",
    mask: NDArray[np.bool_],
    quality: str = "balanced",
) -> dict[str, Any]:
    """The original request with this suggestion applied — ready for :func:`plan`.

    This is what makes a suggestion a *testable claim* rather than a sentence:
    Rung D calls ``plan(**apply_suggestion(s, **request))`` and then runs the
    result. Nothing here decides anything; it substitutes one field.

    Args:
        suggestion: one of the suggestions :func:`suggest` returned.
        fluid, speed, size, mask, quality: the request it was made about.

    Returns:
        Keyword arguments for :func:`flow.autoconfig.plan`.

    Raises:
        ValueError: if ``suggestion.change`` names a field this function does
            not know how to substitute — loud, for the same reason
            :func:`classify` is.
    """
    request: dict[str, Any] = {
        "fluid": fluid if isinstance(fluid, Fluid) else _fluid_by_name(fluid),
        "speed": speed,
        "size": size,
        "mask": mask,
        "quality": quality,
    }
    change = suggestion.change
    if change == "quality":
        request["quality"] = str(suggestion.value)
    elif change == "speed":
        request["speed"] = suggestion.value
    elif change == "size":
        request["size"] = suggestion.value
    elif change == "fluid":
        request["fluid"] = _fluid_by_name(str(suggestion.value))
    elif change == "mask":
        request["mask"] = EXAMPLE_MASK
    else:  # pragma: no cover - guarded by a test with teeth
        raise ValueError(
            f"flow.diagnose cannot apply a suggestion that changes "
            f"{change!r}; it knows {SUGGESTION_ORDER}."
        )
    return request


# ---------------------------------------------------------------------------
# Monitor — divergence caught before nan
# ---------------------------------------------------------------------------


class Diverging(Exception):
    """A run that is going unstable, caught before it reaches ``nan``.

    Structured for the same reason :class:`~flow.autoconfig.Unrepresentable`
    is (**D-045**): a caller that wants to print it gets prose, and a caller
    that wants to test it gets fields.

    Attributes:
        cause: which of ``DOCS/IDEA2.md`` § Stability's failure modes this is
            — ``"relaxation"``, ``"speed"`` or ``"mass"``.
        symptom: which tripwire fired, in one clause.
        fix: what to change, in physical language.
        step: the timestep it was caught on.
        details: the numbers, for the reader who wants them.
    """

    def __init__(
        self,
        *,
        cause: str,
        symptom: str,
        fix: str,
        step: int,
        details: dict[str, float],
    ) -> None:
        self.cause = cause
        self.symptom = symptom
        self.fix = fix
        self.step = step
        self.details = details
        super().__init__(f"{symptom} -- caught at step {step}. {fix}")


#: Cause -> (plain-language symptom, plain-language fix). The three failure
#: modes are ``DOCS/IDEA2.md`` § Stability's own: ``tau`` too close to 0.5,
#: lattice velocity over the 0.1 ceiling, and mass that will not stay put.
_CAUSE_TEXT: dict[str, tuple[str, str]] = {
    "relaxation": (
        "the flow is growing without bound and this case is running at the "
        "edge of what the simulator is stable at",
        "Raise quality so the body is resolved more finely, or slow the flow "
        "down -- either one moves the case back inside the stable range.",
    ),
    "speed": (
        "the fluid is moving faster than this simulator's model of a fluid "
        "stays accurate at",
        "Slow the flow down, or raise quality so the same flow is resolved "
        "more finely; the peak speed around the body, not the speed you "
        "typed, is what has to stay in range.",
    ),
    "mass": (
        "the amount of fluid in the domain is not staying put",
        "Check that the flow has somewhere to leave: a domain that is fed at "
        "the inlet and sealed downstream fills up until it bursts.",
    ),
}


class Monitor:
    """A ``per_step`` probe that raises :class:`Diverging` before ``nan``.

    Passed to :func:`lbm.runner.run` as ``per_step=Monitor()`` (**D-025**), or
    called directly with the sim after each :meth:`~lbm.runner.Sim.step`. It
    runs on the physics thread, so constraint 8 governs its cost: it is
    **sampled**, not per step, and every sample is three whole-array
    reductions into buffers it allocates once — never inside the loop
    (``CLAUDE.md`` § Coding conventions).

    Three tripwires, each a row of ``DOCS/IDEA2.md`` § Stability:

    ``sustained over the speed ceiling``
        Peak ``|u|`` at or above the constraint-3 ceiling on
        ``over_ceiling_samples`` **consecutive** samples. The *sustained* part
        is measured, not stylistic: a healthy start-up transient can cross the
        ceiling for one sample and recover, and a tripwire that fires on that
        is a false alarm — see ``DOCS/STATE2.md`` for the trace.
    ``mass drift``
        Total fluid mass away from its starting value by more than
        ``mass_drift``. The scheme is incompressible in the low-Mach limit, so
        anything above a fraction of a percent is a boundary condition losing
        or inventing fluid.
    ``already not finite``
        A last resort: if the state is already ``nan``, say so with the same
        structure rather than letting it surface as a bare array of ``nan``.

    The **cause** is attributed separately from the symptom. If the case is
    running at or below the bluff-body relaxation floor (**D-029**), that is
    the cause whatever tripwire fired first, because everything else is
    downstream of it — which is exactly the reading of ``DOCS/IDEA2.md``
    § Stability's first row.

    Args:
        every: sample cadence in timesteps.
        mass_drift: fractional mass change that counts as divergence.
        over_ceiling_samples: consecutive over-ceiling samples required.

    Attributes:
        samples: how many samples have been taken.
        peak_speed: the most recent sampled peak, ``nan`` before the first
            sample.
        drift: the most recent sampled fractional mass change.
    """

    def __init__(
        self,
        *,
        every: int = 25,
        mass_drift: float = 0.01,
        over_ceiling_samples: int = 3,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1 (got {every!r}).")
        if over_ceiling_samples < 1:
            raise ValueError(
                f"over_ceiling_samples must be at least 1 "
                f"(got {over_ceiling_samples!r})."
            )
        self.every = int(every)
        self.mass_drift = float(mass_drift)
        self.over_ceiling_samples = int(over_ceiling_samples)
        self.reset()

    def reset(self) -> None:
        """Forget the reference mass and the tripwire counters."""
        self.samples: int = 0
        self.peak_speed: float = float("nan")
        self.drift: float = 0.0
        self._calls: int = 0
        self._mass0: float | None = None
        self._over_run: int = 0
        self._sq: NDArray[np.float32] | None = None
        self._work: NDArray[np.float32] | None = None
        self._solid: NDArray[np.bool_] | None = None
        self._fluid: NDArray[np.bool_] | None = None

    # -- the probe ---------------------------------------------------------

    def __call__(self, sim: Any) -> None:
        """Sample the sim, and raise :class:`Diverging` if a tripwire fired.

        Args:
            sim: a :class:`lbm.runner.Sim`, after :meth:`~lbm.runner.Sim.step`.

        Raises:
            Diverging: naming the cause and the fix.
        """
        self._calls += 1
        if self._calls % self.every:
            return
        self.samples += 1

        u = sim.host_u()
        rho = sim.host_rho()
        self._ensure_buffers(sim, u.shape[1:])

        sq = self._sq
        work = self._work
        assert sq is not None and work is not None  # allocated just above
        np.square(u[0], out=sq)
        np.square(u[1], out=work)
        sq += work
        if self._solid is not None:
            sq[self._solid] = 0.0
        peak_sq = float(sq.max())
        peak = float(np.sqrt(peak_sq)) if np.isfinite(peak_sq) else float("inf")
        self.peak_speed = peak

        mass = (
            float(np.sum(rho, where=self._fluid, dtype=np.float64))
            if self._fluid is not None
            else float(np.sum(rho, dtype=np.float64))
        )
        if self._mass0 is None and np.isfinite(mass) and mass != 0.0:
            self._mass0 = mass
        if self._mass0 and np.isfinite(mass):
            self.drift = abs(mass / self._mass0 - 1.0)
        else:
            self.drift = float("inf")

        step = int(getattr(sim, "step_count", self._calls))

        if not (np.isfinite(peak_sq) and np.isfinite(mass)):
            self._raise(sim, "not finite", step, already_nan=True)

        if peak >= U_LATTICE_MAX:
            self._over_run += 1
            if self._over_run >= self.over_ceiling_samples:
                self._raise(sim, "speed", step)
        else:
            self._over_run = 0

        if self.drift > self.mass_drift:
            self._raise(sim, "mass", step)

    # -- internals ---------------------------------------------------------

    def _ensure_buffers(self, sim: Any, shape: tuple[int, ...]) -> None:
        """Allocate the two work arrays and the fluid mask **once**."""
        if self._sq is not None and self._sq.shape == shape:
            return
        self._sq = np.empty(shape, dtype=np.float32)
        self._work = np.empty(shape, dtype=np.float32)
        solid = getattr(sim, "solid", None)
        if solid is not None and np.asarray(solid).shape == shape:
            self._solid = np.asarray(solid, dtype=bool)
            self._fluid = ~self._solid
        else:
            self._solid = None
            self._fluid = None

    def _cause(self, sim: Any, tripwire: str) -> str:
        """Attribute the cause — the relaxation floor outranks every symptom."""
        tau = float(getattr(getattr(sim, "config", None), "tau", TAU_FLOOR + 1.0))
        if tau <= TAU_FLOOR:
            return "relaxation"
        return "mass" if tripwire == "mass" else "speed"

    def _raise(
        self, sim: Any, tripwire: str, step: int, *, already_nan: bool = False
    ) -> None:
        cause = self._cause(sim, tripwire)
        symptom, fix = _CAUSE_TEXT[cause]
        if already_nan:
            symptom = (
                "the simulation has already produced values that are not "
                "numbers, and " + symptom
            )
        raise Diverging(
            cause=cause,
            symptom=symptom,
            fix=fix,
            step=step,
            details={
                "peak_speed": self.peak_speed,
                "mass_drift": self.drift,
                "sample": float(self.samples),
                "tripwire_speed": 1.0 if tripwire == "speed" else 0.0,
                "tripwire_mass": 1.0 if tripwire == "mass" else 0.0,
            },
        )
