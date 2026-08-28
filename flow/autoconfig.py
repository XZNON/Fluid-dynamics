"""Auto-configuration: physics in, everything the solver needs out.

``DOCS/IDEA3.md`` § 1 calls this "the single most product-defining module in
the phase" — given a fluid, a speed, a size, a mask and a quality level,
:func:`plan` chooses the resolution, ``tau``, the lattice ``U``, the domain
shape, the run length, the frame cadence and the vorticity colour limit.
``lbm/units.py::LatticeUnits.from_physical`` already does the *conversion*
arithmetic; this module does the *choosing* — which is why it calls that
function rather than re-deriving anything (constraint 2).

Read before touching a constant here: ``validate/cylinder.py::tau_for`` /
``make_config`` / ``cylinder_mask`` and ``validate/polygons.py::tau_for_rung4``
are hand-tuned instances of exactly this function, for one case each. This one
has to work for any fluid, any speed, any size and any mask, so it cannot copy
their numbers verbatim — but every constant here is still either one of theirs,
generalised, or a fresh decision recorded in ``DOCS/STATE2.md`` (**D-059** on).

What the mask is for
---------------------

``Plan`` carries no ``solid`` array — rasterising the body into a grid is
``flow/prepare.py``'s job (T107), not this module's. What :func:`plan` needs
from ``mask`` is its **shape**: the cross-stream extent (``D`` of **D-019**)
and the streamwise extent of the immersed object, measured at whatever
resolution the caller's mask happens to be at. Scaling those two numbers by
``cells_per_length / D_measured`` predicts what the body's footprint will be
once it is rasterised at the resolution this module chooses — which is what
lets the domain be sized correctly *before* T107 exists.

Which ``tau`` floor
--------------------

``lbm/units.py`` documents three floors: 0.51 generic (**D-032**), 0.537
Rung 3's disc in a free stream (**D-036**), 0.54 the measured bluff-body floor
for a disc or a square (**D-029**). Every case this module plans is an
immersed object in a free stream — that is what "a mask" means here — so it
always applies the strictest of the three, 0.54, rather than trying to
classify the shape first. Using Rung 3's own looser 0.537 would need a reason
to believe the specific shape is *less* bluff than a disc, which nothing here
can know; being wrong in the safe direction is the point of a guardrail.

Domain sizing
--------------

Blockage and downstream fetch are **constant by construction**, not merely
checked after the fact: the fluid span is a fixed multiple of ``D``
(:data:`SPAN_D`) and so is the downstream fetch (:data:`DOWNSTREAM_D`), the
same shape of guarantee **D-026** used for Rung 3 (a fixed span in diameters
keeps blockage below the ceiling regardless of resolution). The margins **are**
Rung 3's tuned ones — 24 D of span and 8 D of upstream fetch — because a
product answer is a published force coefficient as far as its user is
concerned. **D-075** supersedes **D-059**'s smaller, cheaper domain, which was
measured to put ``Cd`` 14% above Rung 3's band on Rung 3's own case.
The one guardrail that is not automatic is minimum solid thickness, because
that depends on the mask's own proportions, so it is measured and checked
explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

from lbm.geometry import bounding_box, min_thickness, strip_solid_border
from lbm.runner import steps_per_frame as _steps_per_frame
from lbm.units import (
    BLUFF_BODY_SPEEDUP,
    U_LATTICE_DEFAULT,
    U_LATTICE_MAX,
    LatticeUnits,
)

from flow.fluids import Fluid
from flow.quantity import LENGTH, SPEED, Quantity, parse

__all__ = ["Plan", "Suggestion", "Unrepresentable", "plan", "QUALITY_LEVELS"]

# ---------------------------------------------------------------------------
# Constants — every one cites a decision or is measured and logged as one.
# ---------------------------------------------------------------------------

#: Resolution (cells across the characteristic length ``D``) per quality
#: level. **D-040**: "N cells across the body", the same meaning
#: ``--resolution`` gives it in ``lbm/runner.py``. Chosen so ``accurate`` is a
#: strict refinement of ``fast`` (higher resolution, longer run — the
#: acceptance criterion's monotonicity) and so that even ``fast`` clears
#: :data:`TAU_FLOOR` for Rung 3's own Re 100 benchmark at the fixed
#: :data:`~lbm.units.U_LATTICE_DEFAULT`: ``tau = 0.5 + 3 U N / Re`` needs
#: ``N >= Re (TAU_FLOOR - 0.5) / (3 U)`` ~ 26.7 cells at Re 100, so 30 is the
#: smallest round number with margin. Small enough that Rung B's 24-case x
#: 5000-step sweep runs in minutes on the numpy backend (measured this
#: session, **D-059** — see ``DOCS/STATE2.md``).
QUALITY_CELLS: dict[str, int] = {"fast": 30, "balanced": 40, "accurate": 50}

QUALITY_LEVELS: tuple[str, ...] = ("fast", "balanced", "accurate")

#: The floor this module always applies — see the module docstring for why the
#: strictest of the three project floors (**D-029**) is the only defensible
#: default for an arbitrary immersed mask.
TAU_FLOOR: float = 0.54

#: Fluid span, in diameters, wall to wall. **Rung 3's own 24 D** (4.17%
#: blockage, **D-026**) — **D-075**, superseding **D-059**'s 12 D.
#:
#: D-059 argued that this module need not reproduce a force coefficient, only
#: clear constraint 12's 10% ceiling, and picked 12 D (8.33%) to keep Rung B's
#: domains cheap. Measured through the finished product path on Rung 3's own
#: case (water, 5 mm/s, 2 cm, ``quality="fast"``, Re 99.6, 48000 steps, warp),
#: that costs the answer its correctness: ``Cd`` reads **1.5943** against a
#: published band of 1.25..1.45. The span is why — confinement accelerates the
#: flow past the body, which ``validate/cylinder.py``'s own ``SPAN_D``
#: docstring had already measured at 15 D. Measured here, same case, only the
#: domain changing:
#:
#: =================  ==========  ========  ========
#: span / upstream    blockage    ``Cd``    ``St``
#: =================  ==========  ========  ========
#: 12 D / 3 D         8.33%       1.5943    0.1835
#: 16 D / 6 D         6.25%       1.4523    0.1729
#: 20 D / 6 D         5.00%       1.4426    0.1713
#: 22 D / 6 D         4.55%       1.4394    0.1705
#: 24 D / 6 D         4.17%       1.4360    0.1696
#: **24 D / 8 D**     **4.17%**   1.4030    0.1676
#: =================  ==========  ========  ========
#:
#: Rung 3 itself prints ``Cd`` **1.4031** and ``St`` **0.1731** on this
#: machine, so the last row is the benchmark's own number to four decimals.
#: The cost is 389k cells against 140k, charged to Rung E's wall clock;
#: **D-076** is where that was paid back.
SPAN_D: float = 24.0

#: Upstream fetch, in diameters. Not a constraint-12 rule (only the downstream
#: fetch is, rule 2) — enough for the inlet's uniform profile to reach the body
#: without the body's own blockage reaching back to the inlet. **Rung 3's 8 D**
#: (**D-026**), restored by **D-075** from **D-059**'s 3 D. It is not a
#: second-order knob: at a fixed 24 D span, 6 D of fetch gives ``Cd`` 1.4360
#: and 8 D gives **1.4030** — the whole remaining distance to Rung 3's digit
#: (the table above).
UPSTREAM_D: float = 8.0

#: Downstream fetch, in diameters. Constraint 12 / **D-019**'s rule 2 sets an
#: 8 D floor; this uses 9 D for a small margin above it rather than sitting
#: exactly on the boundary (**D-059**). **Unchanged by D-075**, deliberately:
#: Rung 3 uses 12 D, and 9 D reproduces its ``Cd`` to four decimals, so the
#: three extra diameters are cells this module does not need to spend.
DOWNSTREAM_D: float = 9.0

#: Minimum solid thickness in cells, constraint 12 / **D-017**.
MIN_THICKNESS_CELLS: float = 3.0

#: Convective times (``D / U``, the same unit ``validate/cylinder.py`` uses)
#: the run length is sized for — **80** (**D-079**, superseding **D-059**'s 20).
#:
#: The default run has to be long enough that the report can report. Two
#: independent rules set the floor, and both are arithmetic rather than taste:
#:
#: * **D-069** opens the measurement window only after the startup kick has
#:   switched off (5 convective times) *and* washed out of the domain (one
#:   flow-through = ``UPSTREAM_D + 1 + DOWNSTREAM_D`` = **18** convective
#:   times), so nothing at all is measured before **23** — and since the window
#:   is the last 50%, a run under **46** measures nothing and says so.
#: * **D-070**'s gate 2 wants the window to hold two of the slowest plausible
#:   shedding periods, 20 convective times each, which is 40 of window and
#:   therefore **80** of run.
#:
#: 20 satisfied neither once **D-075** widened the domain, and the failure was
#: found by running the README's own command: `Cd` came back **`nan`** with
#: *"the run ended after 16000 steps, before the startup kick had ... washed
#: out (18400)"*. It had been passing before only because the narrower domain
#: made a flow-through shorter, which is the kind of coupling a constant chosen
#: for cheapness hides. 80 is the larger of the two floors, so the default run
#: reports every number :class:`flow.report.Result` has — and a longer,
#: converged run is still asked for through ``flow.case`` (T108) on top of it.
#: The cost is real and is recorded: Rung B's numpy accuracy case goes from
#: ~2 minutes to ~10.
RUN_CONVECTIVE_TIMES: float = 80.0

#: Vorticity colour-limit factor, ``vorticity_limit = FACTOR * U / D``
#: (constraint 9 / **D-028**: fixed, symmetric limits). ``validate/cylinder.py``
#: uses the same factor as the natural vorticity scale of a wake — see its
#: ``VMAX_FACTOR`` docstring.
VORTICITY_FACTOR: float = 4.0

#: Display cadence for :func:`lbm.runner.steps_per_frame` (**D-023**).
FPS: float = 60.0
PLAYBACK_SPEED: float = 1.0

#: Measured steps/s at five grid sizes, alternating-round A/B, one ``Sim``
#: resident (**D-035**), from ``DOCS/STATE2.md`` § Performance baseline. Used
#: by :meth:`Plan.estimated_seconds` as a log-log rate model. Quoted with their
#: conditions: AMD Ryzen 7 5800H at 3201 MHz on mains; NVIDIA RTX 3050 Laptop
#: GPU, driver 592.82, for the ``warp`` column.
#:
#: **The 160k and 400k rows are D-077**, and they are why Rung B passes on
#: ``warp``. With only the 40k / 1M / 2M anchors this model interpolated
#: log-log **between two bandwidth-bound points** and therefore straight
#: through the region where a GPU this size is *kernel-launch*-bound: at 160k
#: cells it predicted 1996 steps/s where the card does **3560.4**, which is
#: the 75.7% Rung B error session 19 recorded. The measured slope says the
#: same thing plainly — 40k to 160k is **-0.11** in log-log (nearly flat: the
#: launches, not the bytes, are the cost) and 160k to 400k is **-1.02**. Two
#: anchors inside the range the product actually runs in are the fix; the
#: model itself is unchanged.
_RATE_TABLE: dict[str, tuple[tuple[float, float], ...]] = {
    "numpy": (
        (40_000.0, 775.1),
        (160_000.0, 185.6),
        (400_000.0, 76.5),
        (1_000_000.0, 23.0),
        (2_000_000.0, 8.3),
    ),
    "warp": (
        (40_000.0, 4155.0),
        (160_000.0, 3560.4),
        (400_000.0, 1403.9),
        (1_000_000.0, 757.3),
        (2_000_000.0, 441.0),
    ),
}


def _rate_steps_per_second(cells: float, backend: str) -> float:
    """Log-log interpolation/extrapolation of steps/s from :data:`_RATE_TABLE`.

    D2Q9 is memory-bandwidth bound (``DOCS/IDEA3.md`` § Performance budget), so
    steps/s falls off roughly as a power of the cell count — a straight line in
    log-log space — which is what makes a two-point interpolation defensible
    between the table's anchors, and a one-sided extrapolation defensible just
    outside them.

    Args:
        cells: grid cell count, ``ny * nx``.
        backend: ``"numpy"`` or ``"warp"`` — a key of :data:`_RATE_TABLE`.

    Returns:
        Estimated steps per second.

    Raises:
        ValueError: if ``backend`` is not a key of :data:`_RATE_TABLE`.
    """
    if backend not in _RATE_TABLE:
        raise ValueError(
            f"no rate model for backend {backend!r}; have "
            f"{', '.join(_RATE_TABLE)}"
        )
    points = _RATE_TABLE[backend]
    log_c = math.log(max(cells, 1.0))
    xs = [math.log(c) for c, _ in points]
    ys = [math.log(r) for _, r in points]

    if log_c <= xs[0]:
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
    elif log_c >= xs[-1]:
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
    else:
        i = next(i for i in range(len(xs) - 1) if xs[i] <= log_c <= xs[i + 1])
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]

    slope = (y1 - y0) / (x1 - x0)
    log_rate = y0 + slope * (log_c - x0)
    return math.exp(log_rate)


# ---------------------------------------------------------------------------
# Refusal — structured, per D-045 / constraint 16
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suggestion:
    """One concrete, checkable way to make an :class:`Unrepresentable` case run.

    T106 (``flow/diagnose.py``) is what turns this into a sentence; Rung D
    feeds ``value`` back through :func:`plan` in place of the field named by
    ``change`` and runs it — so a suggestion that does not fix its case is a
    failing test there (**D-045**), not here.

    Attributes:
        change: which of ``plan``'s inputs to change — ``"speed"``, ``"size"``,
            ``"quality"`` or ``"mask"``. ``flow/diagnose.py`` adds ``"fluid"``
            and applies all five through
            :func:`flow.diagnose.apply_suggestion`; the whole vocabulary, in
            the order suggestions are offered, is
            :data:`flow.diagnose.SUGGESTION_ORDER`.
        value: the replacement — a :class:`~flow.quantity.Quantity` for
            ``"speed"``/``"size"``, one of :data:`QUALITY_LEVELS` for
            ``"quality"``, a fluid name for ``"fluid"``, and a description of
            the picture that is needed for ``"mask"`` (the tool cannot invent
            the user's geometry, so Rung D substitutes
            :data:`flow.diagnose.EXAMPLE_MASK` to run the claim).
        note: one sentence on what this changes physically.
    """

    change: str
    value: "Quantity | str"
    note: str


class Unrepresentable(Exception):
    """A case :func:`plan` refuses, with the structured fields **D-045** asks for.

    Never a formatted string alone: ``reason`` names the guardrail that would
    be violated, ``quantity`` / ``value`` / ``limit`` are the number and its
    ceiling or floor, and ``suggestions`` is a list of :class:`Suggestion`,
    each a testable claim.
    """

    def __init__(
        self,
        *,
        reason: str,
        quantity: str,
        value: float,
        limit: float,
        suggestions: list[Suggestion],
    ) -> None:
        self.reason = reason
        self.quantity = quantity
        self.value = value
        self.limit = limit
        self.suggestions = suggestions
        super().__init__(
            f"{reason}: {quantity} = {value:.4g}, limit {limit:.4g} "
            f"({len(suggestions)} suggestion(s))"
        )


# ---------------------------------------------------------------------------
# The output record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """Everything the solver needs, chosen from physics. Output only.

    Never constructed directly by a caller — :func:`plan` is the only thing
    that builds one. Every field has a matching entry in :attr:`why`
    (asserted by ``tests/test_autoconfig.py``), because a derived number the
    user cannot see is a number they cannot check (``DOCS/IDEA3.md`` § 1).

    Attributes:
        cells_per_length: resolution — cells across the characteristic length.
        tau: BGK relaxation time.
        u_lattice: lattice velocity for the free stream.
        domain: ``(ny, nx)`` grid shape, periodic sides (**D-026**-style).
        steps: total lattice timesteps to run.
        steps_per_frame: timesteps per rendered frame (**D-023**).
        vorticity_limit: symmetric colour limit for the vorticity field.
        dx: metres per cell.
        dt: seconds per lattice timestep.
        Re: Reynolds number of the case.
        warnings: non-fatal notes (e.g. :meth:`~lbm.units.LatticeUnits.stability_note`).
        why: one sentence per field above, keyed by field name.
    """

    cells_per_length: int
    tau: float
    u_lattice: float
    domain: tuple[int, int]
    steps: int
    steps_per_frame: int
    vorticity_limit: float
    dx: float
    dt: float
    Re: float
    warnings: list[str]
    why: dict[str, str]

    def estimated_seconds(self, backend: str = "numpy") -> float:
        """Predicted wall clock for :attr:`steps`, from the measured rate table.

        Args:
            backend: which measured rate curve to use — ``"numpy"`` or
                ``"warp"`` (:data:`_RATE_TABLE`).

        Returns:
            Estimated seconds, ``steps / rate(cells, backend)``.
        """
        ny, nx = self.domain
        rate = _rate_steps_per_second(float(ny * nx), backend)
        return self.steps / rate


# ---------------------------------------------------------------------------
# The chooser
# ---------------------------------------------------------------------------


def _measure_object(mask: NDArray[np.bool_]) -> tuple[float, float, float]:
    """``(D_measured, streamwise_measured, native_thickness)`` from ``mask``.

    ``D`` is the cross-stream (bounding-box height) extent, per **D-019**;
    ``streamwise`` is the bounding-box width. Both are in the mask's own
    cells, at whatever resolution the caller's mask happens to be — scaled by
    the caller once :func:`plan` has chosen a resolution.

    Raises:
        Unrepresentable: if ``mask`` has no solid cell once its (if any)
            fully-solid border is stripped — there is no body to plan around.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError(f"mask must be a 2-D bool array, got shape/dtype {mask.shape}/{mask.dtype}")

    obj = strip_solid_border(mask)
    box = bounding_box(obj)
    if box is None:
        raise Unrepresentable(
            reason="mask has no immersed object to plan around",
            quantity="solid cells (excluding domain border)",
            value=0.0,
            limit=1.0,
            suggestions=[
                Suggestion(
                    change="mask",
                    value="a picture with a solid shape in it, at least 3 cells thick",
                    note=(
                        "Give a picture that contains a body -- this one is "
                        "empty, or solid edge to edge, so there is nothing for "
                        "the fluid to flow around"
                    ),
                )
            ],
        )
    y0, y1, x0, x1 = box
    height = float(y1 - y0 + 1)
    width = float(x1 - x0 + 1)
    thickness = float(min_thickness(obj))
    return height, width, thickness


def _tau_suggestions(
    *, u_phys: float, l_phys: float, nu_phys: float, cells_per_length: int
) -> list[Suggestion]:
    """Concrete ways to raise ``tau`` back above :data:`TAU_FLOOR`.

    Every suggestion targets ``tau = TAU_FLOOR + margin`` rather than exactly
    the floor, so re-planning at the suggested value does not land back on the
    boundary it just failed (**D-045**: a suggestion is a testable claim, and a
    claim that barely fails a re-check is not one).
    """
    margin = 0.01
    target = TAU_FLOOR + margin
    # tau = 0.5 + 3 U_lattice N nu_phys / (u_phys l_phys)   (from LatticeUnits)
    # Solve for the physical knob that raises tau to `target`.
    k = 3.0 * U_LATTICE_DEFAULT * cells_per_length * nu_phys / (target - 0.5)
    u_needed = k / l_phys
    l_needed = k / u_phys

    suggestions = [
        Suggestion(
            change="speed",
            value=Quantity(u_needed, default_unit="m/s"),
            note=(
                f"slow the free stream to about {u_needed:.3g} m/s: a slower "
                "flow needs less resolution to keep tau clear of the "
                "checkerboard floor"
            ),
        ),
        Suggestion(
            change="size",
            value=Quantity(l_needed, default_unit="m"),
            note=(
                f"shrink the body to about {l_needed:.3g} m: a smaller body "
                "at the same speed and fluid is the same Reynolds number at a "
                "coarser, more stable resolution"
            ),
        ),
    ]
    for level in QUALITY_LEVELS:
        n = QUALITY_CELLS[level]
        if n > cells_per_length:
            tau_at_level = 0.5 + 3.0 * U_LATTICE_DEFAULT * n * nu_phys / (u_phys * l_phys)
            if tau_at_level > TAU_FLOOR:
                suggestions.insert(
                    0,
                    Suggestion(
                        change="quality",
                        value=level,
                        note=(
                            f'quality="{level}" resolves the body at {n} cells '
                            "instead -- enough resolution alone clears the floor "
                            "for this case"
                        ),
                    ),
                )
                break
    return suggestions


def plan(
    *,
    fluid: Fluid,
    speed: "Quantity | str | float",
    size: "Quantity | str | float",
    mask: NDArray[np.bool_],
    quality: str = "balanced",
) -> Plan:
    """Choose everything the solver needs from a physical description.

    ``DOCS/IDEA3.md`` § 1: the user never types a lattice quantity. Every
    argument here is physical; every guardrail that would make a lattice
    quantity nonsense is enforced before :class:`Plan` is built, and raises
    :class:`Unrepresentable` rather than silently choosing something else
    (constraint 16).

    Args:
        fluid: a :class:`~flow.fluids.Fluid` (from :func:`flow.fluids.fluid`).
        speed: the free-stream speed — a :class:`~flow.quantity.Quantity`, a
            string ``flow.quantity.parse`` can read as a speed, or a bare
            number of m/s.
        size: the body's characteristic length ``D`` — same acceptance as
            ``speed``, a length.
        mask: ``(h, w)`` bool array. Read only for the immersed object's
            proportions (see the module docstring) — never rasterised or
            returned.
        quality: one of :data:`QUALITY_LEVELS`.

    Returns:
        A frozen :class:`Plan`.

    Raises:
        ValueError: if ``quality`` is not one of :data:`QUALITY_LEVELS`, or if
            ``mask`` is not a 2-D bool array. Programmer-facing, not a physics
            refusal.
        Unrepresentable: if the case cannot be planned within every guardrail
            — an empty mask, a body too thin once scaled, or a Reynolds number
            that pushes ``tau`` at or below :data:`TAU_FLOOR`.
    """
    if quality not in QUALITY_LEVELS:
        raise ValueError(
            f"quality={quality!r} is not one of {QUALITY_LEVELS}"
        )

    speed_q = parse(speed, expect=SPEED, default_unit="m/s")
    size_q = parse(size, expect=LENGTH, default_unit="m")
    u_phys = speed_q.si
    l_phys = size_q.si
    nu_phys = fluid.nu.si

    cells_per_length = QUALITY_CELLS[quality]

    d_measured, streamwise_measured, thickness_native = _measure_object(mask)
    scale = cells_per_length / d_measured
    thickness_scaled = thickness_native * scale
    streamwise_cells = max(1, int(round(streamwise_measured * scale)))

    if thickness_scaled < MIN_THICKNESS_CELLS:
        needed_cells_per_length = math.ceil(
            MIN_THICKNESS_CELLS * d_measured / thickness_native
        )
        best_level = None
        for level in QUALITY_LEVELS:
            if QUALITY_CELLS[level] >= needed_cells_per_length:
                best_level = level
                break
        # A suggestion is a testable claim (**D-045**), so which suggestion is
        # offered depends on whether resolution can actually fix it. When no
        # quality level is enough, offering the highest one anyway would hand
        # back a "fix" that re-raises this very refusal -- a failing test in
        # Rung D rather than a wording preference (T106).
        if best_level:
            suggestions = [
                Suggestion(
                    change="quality",
                    value=best_level,
                    note=(
                        "A higher quality level thickens the thinnest feature "
                        f"proportionally; {needed_cells_per_length} cells "
                        "across the body would clear the 3-cell minimum"
                    ),
                )
            ]
        else:
            suggestions = [
                Suggestion(
                    change="mask",
                    value=(
                        "the same shape with its thinnest feature drawn "
                        "thicker relative to the body"
                    ),
                    note=(
                        "Redraw the shape with its thinnest feature thicker: "
                        "even the highest quality level is not enough, because "
                        "that feature is disproportionately thin relative to "
                        "the body, and more detail scales both together"
                    ),
                )
            ]
        raise Unrepresentable(
            reason=(
                "thinnest solid feature would be under the 3-cell minimum "
                "once rasterised (CLAUDE.md constraint 12, D-017)"
            ),
            quantity="min solid thickness (cells, scaled to the chosen resolution)",
            value=thickness_scaled,
            limit=MIN_THICKNESS_CELLS,
            suggestions=suggestions,
        )

    reynolds = u_phys * l_phys / nu_phys
    tau_value = 0.5 + 3.0 * U_LATTICE_DEFAULT * cells_per_length / reynolds
    if tau_value <= TAU_FLOOR:
        raise Unrepresentable(
            reason=(
                "tau would sit at or below the bluff-body stability floor of "
                "0.54 -- the strictest of the project's three tau floors "
                "(D-029; see D-032 for the generic 0.51 floor and D-036 for "
                "Rung 3's 0.537)"
            ),
            quantity="tau",
            value=tau_value,
            limit=TAU_FLOOR,
            suggestions=_tau_suggestions(
                u_phys=u_phys,
                l_phys=l_phys,
                nu_phys=nu_phys,
                cells_per_length=cells_per_length,
            ),
        )

    units = LatticeUnits.from_physical(
        u_phys=u_phys,
        l_phys=l_phys,
        nu_phys=nu_phys,
        cells_per_length=cells_per_length,
        u_lattice=U_LATTICE_DEFAULT,
        u_max=U_LATTICE_MAX,
        tau_floor=TAU_FLOOR,
    )

    peak_estimate = units.peak_velocity_estimate(BLUFF_BODY_SPEEDUP)
    if peak_estimate >= U_LATTICE_MAX:
        # Unreachable with the fixed U_LATTICE_DEFAULT below the 1.8x headroom
        # margin, but the check is real: it is what "enforced at plan time"
        # means for constraint 3, and it stays exercised on every call.
        raise Unrepresentable(
            reason=(
                "estimated peak lattice velocity (1.8x free stream, D-032's "
                "BLUFF_BODY_SPEEDUP) at or above the CLAUDE.md constraint 3 "
                "ceiling"
            ),
            quantity="peak |u_lattice|",
            value=peak_estimate,
            limit=U_LATTICE_MAX,
            suggestions=[
                Suggestion(
                    change="speed",
                    value=Quantity(u_phys * 0.5, default_unit="m/s"),
                    note="halve the free stream to buy back Mach-number headroom",
                )
            ],
        )

    ny = int(round(SPAN_D * cells_per_length))
    nx = (
        int(round(UPSTREAM_D * cells_per_length))
        + streamwise_cells
        + int(round(DOWNSTREAM_D * cells_per_length))
    )
    blockage = cells_per_length / ny
    if blockage > 0.10:  # pragma: no cover - guarded by SPAN_D's fixed margin
        raise Unrepresentable(
            reason=(
                "blockage ratio at or above CLAUDE.md constraint 12's 10% "
                "ceiling (D-019, D-026)"
            ),
            quantity="blockage",
            value=blockage,
            limit=0.10,
            suggestions=[
                Suggestion(
                    change="quality",
                    value="fast",
                    note="a lower quality level does not change blockage (span scales with resolution); widen the mask's own aspect ratio instead",
                )
            ],
        )

    dx = units.dx
    dt = units.dt
    spf = _steps_per_frame(dt, fps=FPS, speed=PLAYBACK_SPEED)
    steps = max(1, int(round(RUN_CONVECTIVE_TIMES * cells_per_length / U_LATTICE_DEFAULT)))
    vorticity_limit = VORTICITY_FACTOR * U_LATTICE_DEFAULT / cells_per_length

    warnings: list[str] = [units.stability_note()]

    why: dict[str, str] = {
        "cells_per_length": (
            f'quality="{quality}" resolves the body at {cells_per_length} cells '
            "across its characteristic length (D-040)."
        ),
        "tau": (
            f"tau = 0.5 + 3 U_lattice N / Re = {tau_value:.4f}, derived (never "
            "set directly, CLAUDE.md constraint 2) and checked above the "
            f"bluff-body floor of {TAU_FLOOR} (D-029)."
        ),
        "u_lattice": (
            f"fixed at {U_LATTICE_DEFAULT} regardless of the case: "
            f"{BLUFF_BODY_SPEEDUP}x it ({peak_estimate:.4f}) stays under the "
            f"{U_LATTICE_MAX} compressibility ceiling with headroom for the "
            "flow accelerating around the body (CLAUDE.md constraint 3, D-032)."
        ),
        "domain": (
            f"{ny}x{nx} cells: a {SPAN_D:g} D fluid span (periodic sides, "
            "D-026's approach) keeps blockage at "
            f"{blockage:.1%} (well under the 10% ceiling of constraint 12), "
            f"{UPSTREAM_D:g} D upstream and {DOWNSTREAM_D:g} D downstream of "
            "the body (constraint 12 / D-019 needs at least 8 D downstream)."
        ),
        "steps": (
            f"{RUN_CONVECTIVE_TIMES:g} convective times (D/U each) at this "
            "resolution and speed, the same unit validate/cylinder.py uses "
            "for its own run length. Not a round number: the measurement "
            "window is the last half of the run, it opens only after the "
            "startup kick has washed out "
            f"(5 + {UPSTREAM_D + 1.0 + DOWNSTREAM_D:g} = "
            f"{5.0 + UPSTREAM_D + 1.0 + DOWNSTREAM_D:g} convective times, "
            "D-069), and it has to hold two of the slowest shedding period we "
            "would believe (20 convective times each, D-070) or there is no "
            "Strouhal number to report."
        ),
        "steps_per_frame": (
            f"steps_per_frame(dt, fps={FPS:g}, speed={PLAYBACK_SPEED:g}) = "
            f"{spf}, computed from dt rather than hardcoded (CLAUDE.md "
            "constraint 7, D-023)."
        ),
        "vorticity_limit": (
            f"{VORTICITY_FACTOR:g} * U / D = {vorticity_limit:.4g}, the "
            "natural vorticity scale of the wake, used as a fixed symmetric "
            "colour limit (CLAUDE.md constraint 9, D-028)."
        ),
        "dx": f"L / cells_per_length = {size_q.si:g} m / {cells_per_length} = {dx:.6g} m/cell.",
        "dt": f"U_lattice * dx / u_phys = {dt:.6g} s/step (D-023's dt).",
        "Re": f"u_phys * L / nu = {reynolds:.4g}, from the fluid, speed and size as given.",
    }

    return Plan(
        cells_per_length=cells_per_length,
        tau=tau_value,
        u_lattice=U_LATTICE_DEFAULT,
        domain=(ny, nx),
        steps=steps,
        steps_per_frame=spf,
        vorticity_limit=vorticity_limit,
        dx=dx,
        dt=dt,
        Re=reynolds,
        warnings=warnings,
        why=why,
    )


def _plan_field_names() -> tuple[str, ...]:
    """Field names of :class:`Plan` other than ``warnings`` and ``why`` themselves."""
    return tuple(f.name for f in fields(Plan) if f.name not in ("warnings", "why"))
