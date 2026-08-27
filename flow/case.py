"""``flow.case`` — the front door (T108).

``DOCS/IDEA3.md`` § What Phase 1 is, concretely. These three lines are the
whole task::

    case = flow.Case.from_image("wing.png", fluid="air", speed="5 m/s", size="10 cm")
    case.explain()
    result = case.run(record="wake.mp4")

Everything below that line already exists and was tested without a run:
:mod:`flow.prepare` turns the picture into a body (T107), :mod:`flow.quantity`
and :mod:`flow.fluids` turn the strings into SI (T104), :mod:`flow.autoconfig`
turns the physics into every solver parameter (T105), :mod:`flow.diagnose`
turns a refusal into a way forward and watches the run for divergence (T106),
and :mod:`lbm` runs it. **This module is a facade over those five and nothing
else** — the T108 Notes say so in as many words, and any judgement that
accumulates here belongs in ``autoconfig``, ``diagnose`` or ``prepare`` where
it can be tested without a run.

What a facade still has to own, and does:

* **Assembly.** Placing the prepared body in the domain the :class:`~flow.autoconfig.Plan`
  sized, seeding the solid at rest (**D-030**), the startup kick, and the
  sampling cadences the report needs. Ported from ``lbm/runner.py``'s
  ``main`` — the behaviour that already works (M4).
* **The refusal, carried rather than swallowed.** A picture
  :func:`flow.prepare.prepare` refuses (**D-065**) and a case
  :func:`flow.autoconfig.plan` refuses (**D-045**) both leave the
  :class:`Case` *built* and *not runnable*: :attr:`Case.refusal` holds it,
  :meth:`Case.explain` prints it with its way forward, and :meth:`Case.run`
  raises it. Nothing runs a case that was refused, and nothing hides that it
  was.
* **Substitution, marked.** :meth:`Case.nearest` applies the tool's own top
  suggestion (:func:`flow.diagnose.apply_suggestion`) or geometry fix
  (:func:`flow.prepare.apply_fix`) and returns a new :class:`Case` whose runs
  carry ``substituted=True`` all the way into the video's metadata
  (constraint 16, **D-045**, **D-062**).

Constraint 13 governs every public signature here: a picture, a fluid, a
speed, a size, a quality level, a duration in seconds, and where the frames
should go. No ``tau``, no lattice ``U``, no cell count, no
``steps_per_frame`` — all of those are on the :class:`~flow.autoconfig.Plan`,
derived and printed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from flow.autoconfig import (
    QUALITY_CELLS,
    QUALITY_LEVELS,
    UPSTREAM_D,
    Plan,
    Suggestion,
    Unrepresentable,
    plan as _plan,
)
from flow.diagnose import Monitor, apply_suggestion, explain as _explain, suggest
from flow.fluids import Fluid, fluid as _fluid
from flow.prepare import Fix, Prepared, apply_fix, prepare as _prepare
from flow.quantity import LENGTH, SPEED, TIME, Quantity, parse
from flow.report import Result, _analyse, _ffmpeg_metadata_args, metadata_entries
from lbm.core import Q, W
from lbm.record import HeadlessSink, TeeSink
from lbm.render import LiveSink, render
from lbm.runner import NullSink, Sim, SimConfig, Sink, run as _run

__all__ = ["Case", "KICK_FACTOR", "KICK_TIMES", "FORCE_SAMPLES_PER_TIME"]


# ---------------------------------------------------------------------------
# Assembly constants — ported from lbm/runner.py's CLI, which is the behaviour
# that already works (M4). Each is cited rather than chosen here.
# ---------------------------------------------------------------------------

#: Startup kick, as ``lbm.runner.DEMO_KICK_FACTOR`` / ``DEMO_KICK_TC``: a
#: cross-stream inlet velocity of ``KICK_FACTOR`` times the free stream for the
#: first :data:`KICK_TIMES` convective times, then zero. A symmetric body on a
#: symmetric grid stays symmetric far longer than physics would, and a product
#: run is short. The runner's demo values are used rather than Rung 3's gentler
#: 0.10 / 3.0 because the product is making a picture on a short clip, which is
#: the case ``lbm/runner.py`` chose these for; the kick is off long before any
#: measurement window opens, and :attr:`flow.report.Result.cl_mean` is the
#: check that it left nothing behind.
KICK_FACTOR: float = 0.20
KICK_TIMES: float = 5.0

#: Force samples per convective time ``D / U``. The shedding period is ~6
#: convective times, so this is ~300 samples per period — far above what
#: :func:`lbm.probe.strouhal` needs — while the acoustic ringing
#: ``flow.report._lowpass`` exists to reject (period ~0.4 convective times at
#: Rung 3's scale) still gets ~20 samples and cannot alias into the wake's
#: band. Sampling forces costs two host reads, so it is done on a **probe
#: cadence and never per step** (constraint 8, the T103 rule for device
#: backends).
FORCE_SAMPLES_PER_TIME: float = 50.0

#: Settling allowance after the startup kick is switched off, in **domain
#: flow-through times** (``nx / U`` steps each). Nothing is measured until the
#: fluid the kick perturbed has left the domain: measured on a steady Re 10
#: disc, the window that starts the instant the kick stops reports a lift
#: "amplitude" of 0.55 against a Cd of 3.6 — which is the kick's shutdown
#: transient decaying, not an oscillation, and reads as a shedding wake to
#: anything downstream of it. Rung 3 discards 70 convective times where its
#: kick stops at 3, which is ~2.5 flow-through times; one is the floor this
#: sets, not a rival to that choice.
SETTLE_FLOW_THROUGHS: float = 1.0

#: Cross-stream displacement of the body from the centre line, in cells. One
#: cell breaks the grid's mirror symmetry, which is half of why shedding starts
#: (``lbm.runner.demo_domain``'s ``offset``); the kick is the other half.
_OFFSET_CELLS: int = 1

#: How much memory :meth:`Case.run` will spend keeping rendered frames for
#: :meth:`flow.report.Result.save`, in bytes. A frame is ``ny * nx * 3`` bytes
#: and a plan's ``steps_per_frame`` is computed for 60 fps playback, so a
#: 20-convective-time run at ``quality="fast"`` asks for ~6000 frames of
#: 421 KB — 2.5 GB, which is not a default anybody chose. When the budget runs
#: out the run **says so**: the count kept and the count seen go into
#: :attr:`flow.report.Result.warnings` and are printed by
#: :meth:`~flow.report.Result.summary`. A silent cap would read as "these are
#: your frames" when they are the first third of them.
FRAME_MEMORY_BUDGET: int = 512 * 1024 * 1024

#: What :meth:`Case.run` pushes when nothing is drawing. **Not** ``None``:
#: ``None`` is :class:`lbm.runner.RingBuffer`'s "the buffer is empty" sentinel,
#: so pushing it would be indistinguishable from pushing nothing and would
#: leave ``RunStats.delivered`` counting frames that were never delivered. A
#: 1x1 frame goes to :class:`lbm.runner.NullSink`, which discards it, and the
#: ring's bookkeeping stays true.
_NO_FRAME: NDArray[np.uint8] = np.zeros((1, 1, 3), dtype=np.uint8)

#: Playback rate of the frames a run produces. ``lbm.runner``'s CLI default and
#: :data:`flow.autoconfig.FPS`, which is what ``Plan.steps_per_frame`` was
#: computed against (**D-023**) — the two must agree or the video plays at the
#: wrong speed.
_FPS: float = 60.0


class Case:
    """One case: a picture, a fluid, a speed, a size — and what to do with them.

    Building a :class:`Case` runs :func:`flow.prepare.prepare` and
    :func:`flow.autoconfig.plan` and **no timesteps**. Both may refuse; a
    refused case is still built, with :attr:`refusal` set and :attr:`runnable`
    ``False``, so that :meth:`explain` can say what is wrong and what would fix
    it before anything is run — and :meth:`run` raises rather than
    substituting (constraint 16).

    Args:
        source: a ``.png`` (anything Pillow opens), a ``.svg`` (the **D-031**
            subset), or an ``(h, w)`` bool array with ``True`` on solid.
        fluid: a library name (:func:`flow.fluids.known_fluids`), a
            :class:`~flow.fluids.Fluid`, or a viscosity
            :class:`~flow.quantity.Quantity`.
        speed: free-stream speed — ``"5 m/s"``, ``"20 km/h"``, a
            :class:`~flow.quantity.Quantity`, or a bare number of m/s.
        size: the body's characteristic length across the flow — ``"10 cm"``,
            same acceptance as ``speed``.
        quality: one of :data:`flow.autoconfig.QUALITY_LEVELS`. Higher
            resolves more and costs more; it is the only knob between accuracy
            and wall clock, and it is spelled in words rather than cells
            (constraint 13).
        repair: what :func:`flow.prepare.prepare` may fix — ``True`` for
            every repair, ``False`` for none, or an iterable of repair keys.
            Whatever it does is disclosed in :attr:`Prepared.actions` and
            printed by :meth:`explain`.
        backend: which :class:`lbm.backends.Backend` runs the timesteps —
            ``"numpy"`` (the reference oracle, **D-043**) or ``"warp"``.
        substituted: **True** when this case is not the one the user asked
            for — set by :meth:`nearest`, never by a user, and carried into
            every artifact the run produces (constraint 16).
        substitution: one sentence naming what was changed.

    Attributes:
        prepared: the :class:`flow.prepare.Prepared` geometry.
        plan: the :class:`flow.autoconfig.Plan`, or ``None`` if refused.
        refusal: the :class:`flow.autoconfig.Unrepresentable` that refused
            this case, or ``None``. A refused *picture* is carried as
            :attr:`prepared` with ``verdict="refused"`` and its own
            :class:`~flow.prepare.Fix` (**D-065**).
        sim: the :class:`lbm.runner.Sim` of the most recent :meth:`run`, or
            ``None`` before the first one.
    """

    def __init__(
        self,
        source: str | Path | NDArray[np.bool_],
        *,
        fluid: "str | Fluid | Quantity",
        speed: "Quantity | str | float",
        size: "Quantity | str | float",
        quality: str = "balanced",
        repair: bool | Any = True,
        backend: str = "numpy",
        substituted: bool = False,
        substitution: str | None = None,
    ) -> None:
        if quality not in QUALITY_LEVELS:
            raise ValueError(f"quality={quality!r} is not one of {QUALITY_LEVELS}")

        self.source = source
        self.fluid: Fluid = fluid if isinstance(fluid, Fluid) else _fluid(fluid)
        self.speed: Quantity = parse(speed, expect=SPEED, default_unit="m/s")
        self.size: Quantity = parse(size, expect=LENGTH, default_unit="m")
        self.quality = quality
        self.repair = repair
        self.backend = backend
        self.substituted = bool(substituted)
        self.substitution = substitution
        self.sim: Sim | None = None

        # The resolution a quality level implies is autoconfig's choice, not
        # this module's: `plan` sets `cells_per_length` from QUALITY_CELLS
        # regardless of the mask, so preparing the picture at that same number
        # is what makes `plan`'s measured body and the rasterised one the same
        # body (D-040 — a body adjusted after measurement is a body whose `tau`
        # is a fiction).
        self.prepared: Prepared = _prepare(
            source, QUALITY_CELLS[quality], repair=repair
        )

        self.plan: Plan | None = None
        self.refusal: Unrepresentable | None = None
        if self.prepared.runnable:
            try:
                self.plan = _plan(
                    fluid=self.fluid,
                    speed=self.speed,
                    size=self.size,
                    mask=self.prepared.mask,
                    quality=quality,
                )
            except Unrepresentable as exc:
                self.refusal = exc

    # -- construction ------------------------------------------------------

    @classmethod
    def from_image(
        cls,
        path: str | Path,
        *,
        fluid: "str | Fluid | Quantity",
        speed: "Quantity | str | float",
        size: "Quantity | str | float",
        quality: str = "balanced",
        repair: bool | Any = True,
        backend: str = "numpy",
    ) -> "Case":
        """Build a case from a picture on disk. Runs nothing.

        ``DOCS/IDEA3.md`` § What Phase 1 is, concretely — the first of the
        three lines.

        Args:
            path: ``.png``, ``.svg``, or anything Pillow opens.
            fluid, speed, size, quality, repair, backend: see :class:`Case`.

        Returns:
            The :class:`Case`, refused or not.

        Raises:
            FileNotFoundError: if the picture is not there.
        """
        return cls(
            Path(path),
            fluid=fluid,
            speed=speed,
            size=size,
            quality=quality,
            repair=repair,
            backend=backend,
        )

    @classmethod
    def from_array(
        cls,
        mask: NDArray[np.bool_],
        *,
        fluid: "str | Fluid | Quantity",
        speed: "Quantity | str | float",
        size: "Quantity | str | float",
        quality: str = "balanced",
        repair: bool | Any = True,
        backend: str = "numpy",
    ) -> "Case":
        """Build a case from a boolean array — ``True`` on solid. Runs nothing.

        The array is treated exactly as a picture is: it goes through
        :func:`flow.prepare.prepare`, so it is repaired, measured and refused
        by the same rules (constraint 12, **D-065**).

        Args:
            mask: ``(h, w)`` bool array.
            fluid, speed, size, quality, repair, backend: see :class:`Case`.
        """
        return cls(
            np.asarray(mask, dtype=bool),
            fluid=fluid,
            speed=speed,
            size=size,
            quality=quality,
            repair=repair,
            backend=backend,
        )

    # -- state -------------------------------------------------------------

    @property
    def runnable(self) -> bool:
        """``True`` when neither the picture nor the physics was refused."""
        return self.prepared.runnable and self.refusal is None

    @property
    def suggestions(self) -> list[Suggestion]:
        """Ranked, checkable ways to make a refused case run — ``[]`` if it runs.

        Straight through to :func:`flow.diagnose.suggest`; a refused *picture*
        has a :class:`~flow.prepare.Fix` instead, reached as :attr:`fix`.
        """
        if self.plan is not None or not self.prepared.runnable:
            return []
        return suggest(
            fluid=self.fluid,
            speed=self.speed,
            size=self.size,
            mask=self.prepared.mask,
            quality=self.quality,
        )

    @property
    def fix(self) -> Fix | None:
        """The geometry fix for a refused picture, or ``None`` (**D-065**)."""
        return self.prepared.fix if not self.prepared.runnable else None

    def nearest(self) -> "Case":
        """The nearest runnable case, with the substitution marked.

        **D-045**: *"offer the nearest runnable case, concretely and
        quantitatively"* — and **D-063**: what is offered has to be executable
        by the same machinery the rungs use. So this applies the tool's own
        **top** suggestion through :func:`flow.diagnose.apply_suggestion`, or
        the picture's own :class:`~flow.prepare.Fix` through
        :func:`flow.prepare.apply_fix`, and returns the resulting
        :class:`Case` with ``substituted=True``.

        The returned case may itself be refused — the honest outcome when the
        first fix is not enough, and visible rather than hidden.

        Returns:
            A new :class:`Case`. ``self`` unchanged.

        Raises:
            ValueError: when this case runs already, or was refused with no
                fix at all — which constraint 14 says cannot happen and this
                is where it would be noticed.
        """
        if self.runnable:
            raise ValueError(
                "this case is not refused, so there is no nearest runnable "
                "case to offer: Case.nearest() is the answer to a refusal."
            )

        if not self.prepared.runnable:
            fix = self.prepared.fix
            if fix is None:  # pragma: no cover - constraint 14 forbids it
                raise ValueError(
                    "a refused picture with no fix: CLAUDE.md constraint 14 "
                    "says every refusal names one."
                )
            source, quality, note = self._picture_fix(fix)
            return Case(
                source,
                fluid=self.fluid,
                speed=self.speed,
                size=self.size,
                quality=quality,
                repair=self.repair,
                backend=self.backend,
                substituted=True,
                substitution=note,
            )

        offers = self.suggestions
        if not offers:  # pragma: no cover - constraint 14 forbids it
            raise ValueError(
                "a refused case with no suggestion: CLAUDE.md constraint 14 "
                "says every refusal names a fix."
            )
        top = offers[0]
        request = apply_suggestion(
            top,
            fluid=self.fluid,
            speed=self.speed,
            size=self.size,
            mask=self.prepared.mask,
            quality=self.quality,
        )
        source: Any = request["mask"] if top.change == "mask" else self.source
        return Case(
            source,
            fluid=request["fluid"],
            speed=request["speed"],
            size=request["size"],
            quality=request["quality"],
            repair=self.repair,
            backend=self.backend,
            substituted=True,
            substitution=f"{top.change} -> {top.value}: {top.note}",
        )

    def _picture_fix(self, fix: Fix) -> tuple[Any, str, str]:
        """A geometry :class:`~flow.prepare.Fix`, translated into this API's words.

        :func:`flow.prepare.apply_fix` speaks ``cells_across``, which is
        ``prepare``'s argument and **not** one this layer may take
        (constraint 13 — the T108 contract says so by name). So the two fixes
        are translated:

        ``"resolution"``
            The picture cannot make a body that big. The user-facing knob for
            body size is ``quality``, so the fix becomes the **highest quality
            level the picture can actually resolve** — the nearest runnable
            case is the one closest to what was asked, not the coarsest one
            available. When even ``"fast"`` is too much for it, no quality
            level fixes it and the
            honest answer is the same one T106 and T107 already give: the tool
            cannot invent geometry, so it runs the worked example instead and
            says so (:data:`flow.diagnose.EXAMPLE_MASK`).
        ``"picture"``
            There is nothing in the image to run. Same worked example, same
            disclosure.

        Args:
            fix: the fix from the refused :class:`~flow.prepare.Prepared`.

        Returns:
            ``(source, quality, note)`` for the substituted :class:`Case`.
        """
        if fix.change == "resolution":
            reachable = int(fix.value)
            for level in reversed(QUALITY_LEVELS):  # the finest that fits
                if QUALITY_CELLS[level] <= reachable:
                    return (
                        self.source,
                        level,
                        f'quality -> "{level}": {fix.note}',
                    )
        # Either a "picture" fix, or a picture too small for even the coarsest
        # quality level. flow.prepare.apply_fix's own answer, for its own
        # reason: run the worked example and print that it is what happened.
        request = apply_fix(
            Fix(change="picture", value=fix.value, note=fix.note),
            source=self.source,
            cells_across=QUALITY_CELLS[self.quality],
            repair=self.repair,
        )
        return (
            request["source"],
            self.quality,
            (
                f"{fix.note} This tool cannot invent your geometry, so this "
                "run is the worked example shape, not your picture."
            ),
        )

    # -- explanation -------------------------------------------------------

    def explain(self, *, quiet: bool = False) -> str:
        """Print everything decided on the user's behalf, before running anything.

        ``DOCS/IDEA3.md`` § 1: *"everything else is derived and printed,
        because a derived number the user cannot see is a number they cannot
        check."* What is printed: the request in the user's own units, the
        geometry verdict and every repair actually applied, every field of the
        :class:`~flow.autoconfig.Plan` with its ``why`` line, and the estimated
        wall clock. For a refused case, the refusal in plain language and the
        way forward instead — :func:`flow.diagnose.explain` for a physics
        refusal, the picture's own :class:`~flow.prepare.Fix` for a geometry
        one.

        Args:
            quiet: build the text without printing it.

        Returns:
            The whole explanation as one string.
        """
        lines: list[str] = ["", f"case: {self._source_name()}"]
        lines.append(
            f"  fluid     {self.fluid.name} — {self.fluid.summary()}"
        )
        lines.append(f"  speed     {self.speed}")
        lines.append(f"  size      {self.size} across the flow")
        lines.append(f"  quality   {self.quality!r}")
        if self.substituted:
            lines.append(
                "  ** SUBSTITUTED ** this is not the case originally asked "
                f"for: {self.substitution}"
            )

        lines.append("")
        lines.append(f"geometry: {self.prepared.verdict}")
        for action in self.prepared.actions:
            lines.append(f"  repaired  {action}")
        if not self.prepared.actions and self.prepared.runnable:
            lines.append("  repaired  nothing — the picture was already clean")
        for key in ("body_cells_across", "min_thickness", "thin_branch_depth",
                    "components", "holes", "solid_cells"):
            if key in self.prepared.properties:
                lines.append(f"  measured  {key} = {self.prepared.properties[key]}")
        for warning in self.prepared.warnings:
            lines.append(f"  warning   {warning}")

        if not self.prepared.runnable:
            lines.append("")
            lines.append("refused — the picture")
            lines.append(f"  {self.prepared.reason}")
            if self.prepared.fix is not None:
                lines.append("")
                lines.append("What would work")
                lines.append(f"  - {self.prepared.fix.note}")
                lines.append(
                    "    Case.nearest() applies this and returns the case it "
                    "makes, marked as a substitution."
                )
            return self._emit(lines, quiet)

        if self.refusal is not None:
            lines.append("")
            lines.append(
                _explain(
                    self.refusal,
                    request={
                        "fluid": self.fluid,
                        "speed": self.speed,
                        "size": self.size,
                    },
                )
            )
            lines.append(
                "  Case.nearest() applies the first of these and returns the "
                "case it makes, marked as a substitution."
            )
            return self._emit(lines, quiet)

        assert self.plan is not None  # runnable, so plan() returned
        plan = self.plan
        lines.append("")
        lines.append("plan — every number derived, none typed")
        for name in (
            "Re",
            "cells_per_length",
            "tau",
            "u_lattice",
            "domain",
            "steps",
            "steps_per_frame",
            "vorticity_limit",
            "dx",
            "dt",
        ):
            value = getattr(plan, name)
            shown = f"{value:.6g}" if isinstance(value, float) else f"{value}"
            lines.append(f"  {name:<17}{shown}")
            why = plan.why.get(name)
            if why:
                lines.append(f"      why: {why}")
        for warning in plan.warnings:
            lines.append(f"  warning   {warning}")

        seconds = plan.steps * plan.dt
        estimate = plan.estimated_seconds(self.backend)
        lines.append("")
        lines.append("cost")
        lines.append(
            f"  simulates {seconds:.4g} s of physical time in "
            f"{plan.steps} timesteps on backend {self.backend!r}"
        )
        lines.append(
            f"  estimated wall clock {estimate:.1f} s "
            f"({-(-plan.steps // plan.steps_per_frame)} frames at {_FPS:g} fps)"
        )
        if self.backend != "numpy":
            lines.append(
                "  note: the estimate is calibrated against the NumPy column "
                "of DOCS/STATE2.md § Performance baseline; on a GPU backend it "
                "currently over-predicts by up to ~1.8x at this size (Rung B's "
                "accuracy check, owned by T110)."
            )
        return self._emit(lines, quiet)

    # -- the run -----------------------------------------------------------

    def run(
        self,
        *,
        seconds: "Quantity | str | float | None" = None,
        live: bool = False,
        record: str | Path | None = None,
        headless: str | Path | None = None,
        keep_frames: bool = True,
        monitor: bool = True,
        quiet: bool = False,
    ) -> Result:
        """Run the case and return its :class:`~flow.report.Result`.

        The third of ``DOCS/IDEA3.md``'s three lines. What happens, in order:
        the prepared body is placed in the domain the plan sized, the solid is
        seeded at rest (**D-030**), the sinks the arguments ask for are
        composed through :class:`lbm.record.TeeSink`, and
        :func:`lbm.runner.run` drives the loop — with ``steps_per_frame`` taken
        from the plan (constraint 7, **D-023**), the one
        :func:`lbm.render.render` producing frames at the plan's fixed
        symmetric colour limits (constraint 9, **D-028**), and the run mode
        picked by **D-039**: any sink that writes a **file** takes
        ``drop=False``; only a live-*only* run may drop display frames.

        Args:
            seconds: how much **physical** time to simulate — ``"2 s"``, a
                :class:`~flow.quantity.Quantity`, or a bare number of seconds.
                ``None`` runs the plan's own length (20 convective times).
            live: open a window (:class:`lbm.render.LiveSink`).
            record: write a video here (``.mp4``, ``.gif``, …).
            headless: write numbered PNGs into this directory.
            keep_frames: keep every rendered frame on the
                :class:`~flow.report.Result` so :meth:`flow.report.Result.save`
                can write them later. Costs ``ny * nx * 3`` bytes per frame;
                pass ``False`` for a long run that is already being recorded.
            monitor: watch for divergence with :class:`flow.diagnose.Monitor`
                and raise :class:`flow.diagnose.Diverging` before ``nan``
                (**D-061**).
            quiet: do not print the running/summary lines.

        Returns:
            A :class:`~flow.report.Result`.

        Raises:
            flow.autoconfig.Unrepresentable: if this case was refused. The
                refusal is raised here rather than at construction so that
                :meth:`explain` can print it first — but nothing is ever run
                in its place (constraint 16).
            flow.diagnose.Diverging: if ``monitor`` is on and a tripwire fires.
        """
        self._raise_if_refused()
        assert self.plan is not None  # _raise_if_refused guarantees it
        plan = self.plan

        solid = self._domain()
        ny, nx = plan.domain
        u = plan.u_lattice
        d_cells = float(self.prepared.mask.shape[0])
        t_conv = d_cells / u
        total_steps = self._steps(seconds)
        kick_steps = int(round(KICK_TIMES * t_conv))
        flow_through = nx / u
        settle_steps = kick_steps + int(round(SETTLE_FLOW_THROUGHS * flow_through))
        sample_every = max(1, int(round(t_conv / FORCE_SAMPLES_PER_TIME)))

        cfg = SimConfig(
            ny=ny,
            nx=nx,
            tau=plan.tau,
            inlet_U=u,
            profile="uniform",
            inlet_uy=KICK_FACTOR * u,
            use_inlet=True,
            use_outlet=True,
            convective_outlet=True,
            inlet_axis="x",
            check_geometry=True,
            verbose_mask=not quiet,
            backend=self.backend,
        )
        sim = Sim(cfg, solid)
        self.sim = sim
        _seed_solid_at_rest(sim)

        sink, members, drop = _resolve_sinks(
            live=live,
            record=record,
            headless=headless,
            metadata=metadata_entries(
                substituted=self.substituted,
                substitution=self.substitution,
                reynolds=plan.Re,
                backend=self.backend,
            ),
        )
        frames: list[NDArray[np.uint8]] = []

        n_samples = total_steps // sample_every + 1
        cd_series = np.zeros(n_samples, dtype=np.float64)
        cl_series = np.zeros(n_samples, dtype=np.float64)
        residuals = np.zeros(n_samples, dtype=np.float64)
        taken = 0
        peak_u = 0.0
        watcher = Monitor() if monitor else None

        def probe(s: Sim) -> None:
            """Sample forces, residual and peak speed; switch the kick off.

            On the physics thread, so it runs on a **cadence**, never per step
            (constraint 8): two host reads per sample on a device backend.
            """
            nonlocal taken, peak_u
            if s.step_count == kick_steps:
                s.u_in[1].fill(0.0)
                s.refresh_inlet_profile()
            if s.step_count % sample_every == 0 and taken < n_samples:
                cd_series[taken], cl_series[taken] = s.forces()
                residuals[taken] = s.residual() / sample_every
                s.mark_residual()
                velocity = s.host_u()
                fluid_cells = ~s.solid
                speed_field = np.sqrt(
                    velocity[0][fluid_cells] ** 2 + velocity[1][fluid_cells] ** 2
                )
                peak_u = max(peak_u, float(speed_field.max()))
                taken += 1
            if watcher is not None:
                watcher(s)

        kept_bytes = 0
        seen_frames = 0
        # Nothing to show and nothing to keep: do not compute a vorticity field
        # and colour it for a sink that discards it. `run` still calls `field`
        # once per frame, so this is where that work is declined rather than in
        # a second copy of the loop.
        drawing = keep_frames or bool(members)

        def field(s: Sim) -> NDArray[np.uint8]:
            """One frame, through the one ``render()`` (constraint 10)."""
            nonlocal kept_bytes, seen_frames
            if not drawing:
                return _NO_FRAME
            frame = render(s.vorticity(), plan.vorticity_limit)
            seen_frames += 1
            if keep_frames and kept_bytes + frame.nbytes <= FRAME_MEMORY_BUDGET:
                frames.append(frame)
                kept_bytes += frame.nbytes
            return frame

        window = next((m for m in members if isinstance(m, LiveSink)), None)
        stop: Callable[[Sim], bool] | None = (
            (lambda _s: bool(getattr(window, "quit_requested", False)))
            if window is not None
            else None
        )

        if not quiet:
            kinds = ", ".join(type(s).__name__ for s in members) or "NullSink"
            mode = (
                "drop=True (live only, display frames may be dropped — "
                "constraint 8)"
                if drop
                else "drop=False (a file is being written, every frame in "
                "order — D-024, D-039)"
            )
            print(f"  sinks:    {kinds}   mode: {mode}")
            print(f"  running {total_steps} steps ...", flush=True)

        sim.mark_residual()
        start = time.perf_counter()
        stats = _run(
            sim,
            sink,
            steps=total_steps,
            steps_per_frame=plan.steps_per_frame,
            field=field,
            drop=drop,
            per_step=probe,
            stop=stop,
        )
        elapsed = time.perf_counter() - start
        sink.close()

        cd_series = cd_series[:taken]
        cl_series = cl_series[:taken]
        residuals = residuals[:taken]
        stable = bool(np.isfinite(sim.host_f()).all())

        notes = list(plan.warnings)
        if total_steps <= settle_steps:
            notes.append(
                f"the run ended after {total_steps} steps, before the startup "
                f"kick had switched off ({kick_steps}) and washed out of the "
                f"domain ({settle_steps}) — so nothing was measured, because "
                "what there would be to measure is the kick. Run for at least "
                f"{2 * settle_steps} steps "
                f"({2 * settle_steps / t_conv:.0f} convective times)."
            )
        if keep_frames and len(frames) < seen_frames:
            notes.append(
                f"kept {len(frames)} of the {seen_frames} frames rendered — the "
                f"{FRAME_MEMORY_BUDGET / 1e6:.0f} MB frame budget ran out. Pass "
                "record=<path> to write every frame as it runs, which keeps "
                "none of them in memory."
            )

        measured = _analyse(
            cd_series,
            cl_series,
            sample_every=sample_every,
            d_cells=d_cells,
            u_lattice=u,
            skip_steps=settle_steps,
        )
        result = Result(
            cd=measured["cd"],
            cd_std=measured["cd_std"],
            cd_amplitude=measured["cd_amplitude"],
            cl=measured["cl"],
            cl_mean=measured["cl_mean"],
            strouhal=measured["strouhal"],
            strouhal_confidence=measured["strouhal_confidence"],
            periods=measured["periods"],
            convergence=float(residuals[-1]) if residuals.size else float("nan"),
            peak_u=peak_u,
            elapsed=elapsed,
            substituted=self.substituted,
            substitution=self.substitution,
            backend=self.backend,
            steps=stats.steps,
            stable=stable,
            sample_steps=sample_every,
            fps=_FPS,
            frames=frames,
            convergence_history=residuals,
            cd_history=cd_series,
            cl_history=cl_series,
            plan=plan,
            prepared=self.prepared,
            warnings=notes,
        )
        if not quiet:
            result.summary()
        return result

    # -- internals ---------------------------------------------------------

    def _raise_if_refused(self) -> None:
        """Raise the refusal this case is carrying, if it is carrying one.

        **D-065**: a refused picture is surfaced through the same path as an
        :class:`~flow.autoconfig.Unrepresentable`, never swallowed, and it
        names the executable fix — :func:`flow.prepare.apply_fix`, reached as
        :meth:`nearest`.
        """
        if self.refusal is not None:
            raise self.refusal
        if not self.prepared.runnable:
            fix = self.prepared.fix
            raise Unrepresentable(
                reason=str(self.prepared.reason),
                quantity="body cells across (the picture)",
                value=float(self.prepared.properties.get("body_cells_across", 0)),
                limit=float(QUALITY_CELLS[self.quality]),
                suggestions=(
                    [Suggestion(change="mask", value=str(fix.note), note=fix.note)]
                    if fix is not None
                    else []
                ),
            )

    def _steps(self, seconds: "Quantity | str | float | None") -> int:
        """Physical seconds to timesteps, through the plan's ``dt`` (**D-023**)."""
        assert self.plan is not None
        if seconds is None:
            return self.plan.steps
        duration = parse(seconds, expect=TIME, default_unit="s")
        if duration.si <= 0.0:
            raise ValueError(f"seconds must be positive (got {duration}).")
        return max(1, int(round(duration.si / self.plan.dt)))

    def _domain(self) -> NDArray[np.bool_]:
        """Place the prepared body in the domain the plan sized.

        The placement :func:`flow.autoconfig.plan` assumed when it computed the
        blockage and the downstream fetch — leading edge
        :data:`flow.autoconfig.UPSTREAM_D` diameters from the inlet, centred
        across the flow with :data:`_OFFSET_CELLS` of asymmetry — so the
        guardrails it checked analytically are the ones this mask actually has
        (the same arithmetic as ``validate/autoconfig.py::build_solid``).

        Returns:
            ``(ny, nx)`` bool, ``True`` on solid.
        """
        assert self.plan is not None
        ny, nx = self.plan.domain
        body = self.prepared.mask
        bh, bw = body.shape
        if ny <= bh + 2 or nx <= bw + 2:  # pragma: no cover - plan sizes both
            raise ValueError(
                f"domain {ny}x{nx} is not larger than the body {bh}x{bw}."
            )
        solid = np.zeros((ny, nx), dtype=bool)
        y0 = (ny - bh) // 2 + _OFFSET_CELLS
        x0 = int(round(UPSTREAM_D * self.plan.cells_per_length))
        x0 = min(x0, nx - bw - 2)
        solid[y0 : y0 + bh, x0 : x0 + bw] = body
        return solid

    def _source_name(self) -> str:
        """A short label for the picture, for the printed header."""
        if isinstance(self.source, (str, Path)):
            return Path(self.source).name
        arr = np.asarray(self.source)
        return f"<array {arr.shape[0]}x{arr.shape[1]}>"

    @staticmethod
    def _emit(lines: list[str], quiet: bool) -> str:
        text = "\n".join(lines)
        if not quiet:
            print(text)
        return text

    def __repr__(self) -> str:
        state = (
            f"plan Re={self.plan.Re:.4g}" if self.plan is not None else "refused"
        )
        return (
            f"Case({self._source_name()!r}, fluid={self.fluid.name!r}, "
            f"speed={self.speed}, size={self.size}, quality={self.quality!r}, "
            f"{state})"
        )


# ---------------------------------------------------------------------------
# Assembly helpers — the two things ported from lbm/runner.py's CLI
# ---------------------------------------------------------------------------


def _seed_solid_at_rest(sim: Sim) -> None:
    """Seed every solid cell with the rest equilibrium ``w_i rho0`` (**D-030**).

    :meth:`lbm.runner.Sim._init_equilibrium` seeds the whole domain with the
    equilibrium of the inlet profile, solid included, so at step 0 there is
    fluid moving at ``U`` *inside* the body — and bounce-back **reverses** that
    momentum every step rather than clearing it, so the interior oscillates at
    ``±U`` forever. The rest state is a fixed point of both bounce-back (``W``
    is symmetric under ``OPP``) and streaming (it is uniform), so seeding it
    once at setup holds bit-identically. Applied here rather than inside
    ``lbm/`` for D-030's own reason: changing the initial condition in ``Sim``
    would perturb T006's bit-identical restart claim and Rung 3's published
    numbers for no physical gain.

    Args:
        sim: the simulation, before its first step.
    """
    rest = np.float32(sim.config.rho0) * W.astype(np.float32)
    seed = sim.host_f().copy()
    for i in range(Q):
        seed[i][sim.solid] = rest[i]
    sim.load_f(seed)


def _resolve_sinks(
    *,
    live: bool,
    record: str | Path | None,
    headless: str | Path | None,
    metadata: dict[str, str],
) -> tuple[Sink, list[Sink], bool]:
    """Compose the sinks asked for, and pick the run mode by **D-039**.

    ``--live``, ``--record`` and ``--headless`` are composable and the *mode*
    is not: **D-024** allows exactly two, and **D-039** decides between them by
    one rule — **any sink that writes a file takes ``drop=False``**. A video
    with a missing frame and a PNG series with a gap in the numbering are both
    wrong output rather than slow output, and the sim is allowed to wait for a
    writer. ``drop=True`` is therefore reached only by a live-only run, which
    is exactly the case constraint 8 describes.
    :class:`lbm.record.TeeSink` is a sink, not a third mode.

    Args:
        live: open a window.
        record: video path, or ``None``.
        headless: PNG directory, or ``None``.
        metadata: :func:`flow.report.metadata_entries` for this run — written
            into a recorded video's container, so a substituted run says so in
            the file as well as in the summary (constraint 16, **D-062**). A
            GIF goes through Pillow, which has no ffmpeg command line to put
            it on, so it carries none.

    Returns:
        ``(sink, members, drop)``.
    """
    from lbm.record import RecordSink

    members: list[Sink] = []
    if live:
        members.append(LiveSink(title="flow — vorticity"))
    if record is not None:
        extra: dict[str, Any] = {}
        if Path(record).suffix.lower() != ".gif":
            extra["output_params"] = _ffmpeg_metadata_args(metadata)
        members.append(RecordSink(record, fps=_FPS, **extra))
    if headless is not None:
        members.append(HeadlessSink(headless))

    drop = not (record is not None or headless is not None)
    if not members:
        return NullSink(), [], drop
    sink: Sink = members[0] if len(members) == 1 else TeeSink(*members)
    return sink, members, drop
