"""``flow.report`` — the numbers a run produced, rendering themselves (T108).

``DOCS/IDEA3.md`` § The five things Phase 1 must get right, item 4 — *"Results
render themselves"*: Phase 0 gives one ``render()`` and three sinks; Phase 1
adds the **numbers**. A :class:`Result` carries the ``Cd`` and ``Cl`` history,
the Strouhal estimate with its confidence, the convergence trace, the peak
``|u|`` against the 0.1 ceiling and the wall clock, and emits them as a printed
summary, a matplotlib figure, a dict, or a video/PNG series.

Three rules govern what is and is not allowed in here:

* **Constraint 10 — ``flow/`` colours nothing.** The frames a :class:`Result`
  holds were produced by the one ``render()`` in :mod:`lbm.render`; this module
  hands them to :mod:`lbm.record` and never touches a colormap.
  :meth:`Result.plot` is a matplotlib figure of *scalar histories*, which is a
  different kind of output from a frame — it draws lines, not fields, and says
  so rather than becoming a fourth renderer.
* **Constraint 16 / D-045 / D-062 — never a silent substitution.**
  :attr:`Result.substituted` lives on the object, not in one rendering of it,
  so :meth:`summary`, :meth:`as_dict` and the metadata of a saved video all
  carry it. That is the half of T106's criterion **D-062** deliberately
  carried here.
* **Constraint 13 — no lattice quantity in a public signature.**
  :class:`Result` is a frozen dataclass, so **D-060**'s narrow exemption
  applies to its generated constructor: nothing but :meth:`flow.case.Case.run`
  ever builds one, and its fields are exactly the *derived and printed*
  numbers ``DOCS/IDEA3.md`` § 1 asks for.
* **Constraint 18 — no unqualified quantitative claim outside the validated
  band** (new in Phase 2, **D-082**, landed by T204). Every :class:`Result`
  carries a :class:`flow.fidelity.Band`, and :meth:`Result.__post_init__` is the
  **one place** that rule is implemented: outside the quantitative band the
  reduced coefficients are withheld from the object itself, so ``result.cd`` is
  ``None`` and there is nothing for :meth:`summary`, :meth:`as_dict`,
  :meth:`plot` or a video's metadata to leak. The qualitative band keeps the
  number in a :class:`flow.fidelity.Qualified`, where it cannot be read without
  its caveat; the illustrative band keeps no coefficient at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from lbm.probe import strouhal as _strouhal

from flow.fidelity import Band, Qualified, sentence as _band_sentence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

    from flow.autoconfig import Plan
    from flow.prepare import Prepared

__all__ = [
    "Result",
    "GATED_QUANTITIES",
    "metadata_entries",
    "CL_AMPLITUDE_MIN",
    "TRANSIENT_FRACTION",
    "LOWPASS_SIGMA_TC",
    "MIN_PERIODS",
    "ST_PLAUSIBLE",
]


# ---------------------------------------------------------------------------
# Constants — each one measured elsewhere in this project and cited, never
# tuned here to make a number land (DOCS/PLAN2.md § Risks, the auto-config row).
# ---------------------------------------------------------------------------

#: Shedding is *confirmed*, never assumed: the peak-to-peak lift over the
#: measurement window must exceed this fraction of the mean drag before a
#: Strouhal number is reported at all. ``validate/cylinder.py``'s
#: ``CL_AMPLITUDE_MIN``, same value and same reason — a steady, symmetric wake
#: produces ``Cl ~ 1e-5`` and an FFT of it reports a meaningless "dominant
#: frequency" from numerical noise. Below it, :attr:`Result.strouhal` is
#: ``None`` rather than a number (a T108 acceptance criterion).
CL_AMPLITUDE_MIN: float = 0.01

#: Fraction of a run discarded before anything is measured. Rung 3 discards 70
#: convective times of 130 (``validate/cylinder.py``: ``TRANSIENT_TC`` /
#: ``TRANSIENT_TC + MEASURE_TC``) = 0.54; this is that ratio rounded, expressed
#: as a fraction so it scales with whatever run length the user asked for
#: rather than assuming Rung 3's.
TRANSIENT_FRACTION: float = 0.5

#: Shedding periods the measurement window must be long enough to contain
#: before a Strouhal number is reported at all. Two, because one is not a
#: frequency: a window shorter than a couple of periods always has a
#: "dominant" bin, and reporting it produces exactly the artefact the
#: validation ladder exists to prevent — a converged, plausible, wrong number
#: (``CLAUDE.md`` constraint 5). Rung 3 measures over ~10 periods; this is the
#: floor, not the target.
#:
#: The count is taken against the **longest plausible** period,
#: ``D / (U * min(ST_PLAUSIBLE))`` = 20 convective times, so the gate is
#: "could this window have shown two cycles of the slowest shedding we would
#: believe?" — a question with an answer that does not depend on the estimate.
#: Two reasons it is not the estimate's own period:
#:
#: * **Self-reference.** Measured, on a one-period synthetic sine: the
#:   estimator returned ``St = 0.459`` for a planted 0.17, and that wrong
#:   estimate implies a short period and therefore "2.7 periods observed" — a
#:   guard computed from the answer it guards passes exactly when it should
#:   fire.
#: * **Wrong-answer aversion.** ``CLAUDE.md`` constraint 5: a wrong sim that
#:   looks plausible is the main failure mode of this project. A window under
#:   40 convective times is refused a frequency rather than given a plausible
#:   one; Rung 3 measures over 60 and is unaffected.
MIN_PERIODS: float = 2.0

#: The band a Strouhal number has to fall in to be vortex shedding at all,
#: rather than something else the spectrum happens to contain. Bluff-body
#: shedding sits at 0.12–0.25 across a very wide Reynolds range; this is that
#: with room either side. Outside it the oscillation is real and is **not the
#: wake** — ``validate/cylinder.py::lowpass`` measured exactly that failure,
#: an unfiltered FFT reporting ``St = 1.49`` from the domain's acoustics, and
#: the low-pass alone is not a guarantee.
ST_PLAUSIBLE: tuple[float, float] = (0.05, 0.5)

#: Gaussian low-pass width for the *frequency* estimate only, in convective
#: times ``D / U``. ``validate/cylinder.py``'s ``LOWPASS_SIGMA_TC``, and its
#: docstring is the measurement: the impulsive start rings against the Zou-He
#: inlet at period ~305 steps with power comparable to the wake's at ~2500, and
#: an unfiltered FFT reported ``St = 1.49``. The amplitude that decides
#: *whether* shedding happened is measured on the **raw** series; only the
#: frequency sees the filtered one.
LOWPASS_SIGMA_TC: float = 0.5

#: The reduced coefficients constraint 18 gates. These are **claims** — one
#: number offered as the answer — as distinct from the raw force histories
#: (:attr:`Result.cd_history` and friends), which are the run's own data and stay
#: available in every band. Outside the quantitative band
#: :meth:`Result.__post_init__` sets every one of them to ``None`` on the object,
#: which is what makes "no unqualified ``Cd``" a property of the record rather
#: than of any one rendering of it.
GATED_QUANTITIES: tuple[str, ...] = (
    "cd",
    "cd_std",
    "cd_amplitude",
    "cl",
    "cl_mean",
    "strouhal",
    "strouhal_confidence",
    "periods",
)

#: Spectral peaks this many bins either side of the winner are treated as the
#: same peak when :func:`_peak_dominance` looks for a runner-up. A Hann window
#: spreads one tone over three bins, so a ±2 guard is the narrowest one that
#: cannot report a peak's own skirt as its competition.
_PEAK_GUARD_BINS: int = 2


# ---------------------------------------------------------------------------
# Analysis — private, because every argument below is a lattice quantity
# ---------------------------------------------------------------------------


def _lowpass(series: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Gaussian smoothing of a force history, before the frequency estimate.

    The same filter, for the same measured reason, as
    ``validate/cylinder.py::lowpass`` — see that docstring for the spectrum
    that justifies it. It is duplicated rather than imported because ``flow/``
    depending on ``validate/`` would make the product layer import its own test
    harness; it is six lines of convolution and no physics constant
    (``CLAUDE.md`` § Coding conventions bans a *physics constant* twice, and
    ``sigma`` is a property of the case, passed in).

    Args:
        series: the history, shape ``(n,)``.
        sigma: kernel standard deviation, in samples.

    Returns:
        The smoothed series, shape ``(n - 2 * ceil(3 sigma),)`` — a ``valid``
        convolution, so no edge artefact enters the spectrum.
    """
    half = int(np.ceil(3.0 * sigma))
    if sigma <= 0.0 or series.size <= 2 * half + 8:
        return np.asarray(series, dtype=np.float64)
    t = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.asarray(series, dtype=np.float64), kernel, mode="valid")


def _peak_dominance(series: NDArray[np.float64]) -> float:
    """How far the dominant frequency stands above the next distinct one.

    :func:`lbm.probe.strouhal` returns the frequency and not the spectrum it
    picked it from, so the confidence is measured here on the same series with
    the same window: the ratio of the winning bin's magnitude to the largest
    magnitude more than :data:`_PEAK_GUARD_BINS` away from it. **1.0 means a
    coin toss between two peaks**; Rung 3's cylinder reads several.

    Args:
        series: the (already low-passed, already transient-cut) history.

    Returns:
        The ratio, ``1.0`` when there is nothing to compare against.
    """
    tail = np.asarray(series, dtype=np.float64).ravel()
    if tail.size < 8:
        return 1.0
    tail = tail - tail.mean()
    spectrum = np.abs(np.fft.rfft(tail * np.hanning(tail.size)))
    if spectrum.size < 4:
        return 1.0
    k0 = int(np.argmax(spectrum[1:])) + 1
    peak = float(spectrum[k0])
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    lo = max(1, k0 - _PEAK_GUARD_BINS)
    hi = min(spectrum.size, k0 + _PEAK_GUARD_BINS + 1)
    others = np.concatenate([spectrum[1:lo], spectrum[hi:]])
    if others.size == 0:
        return 1.0
    runner_up = float(others.max())
    if runner_up <= 0.0:
        return float("inf")
    return peak / runner_up


def _analyse(
    cd_series: NDArray[np.float64],
    cl_series: NDArray[np.float64],
    *,
    sample_every: int,
    d_cells: float,
    u_lattice: float,
    skip_steps: int = 0,
) -> dict[str, Any]:
    """Reduce a force history to the numbers :class:`Result` reports.

    All lattice units, all private — this is the inside of the boundary
    ``DOCS/IDEA3.md`` § 1 draws, and nothing here appears in a public
    signature (constraint 13).

    The measurement window is the last ``1 - TRANSIENT_FRACTION`` of the
    series, **and never earlier than** ``skip_steps`` — the startup kick has to
    be off before anything is measured, or the "lift amplitude" is the kick.
    ``Cd`` is the window's mean and standard deviation; ``Cl`` is its half
    peak-to-peak amplitude, measured **raw**.

    A Strouhal number survives three gates, and is ``None`` if any of them
    closes (a T108 acceptance criterion — ``None``, never a number):

    1. the lift amplitude clears :data:`CL_AMPLITUDE_MIN` of ``|Cd|`` (there
       is an oscillation at all);
    2. the window is long enough to hold :data:`MIN_PERIODS` of the shortest
       plausible shedding period (there is room to see one);
    3. the estimate lands inside :data:`ST_PLAUSIBLE` (what was found is
       shedding rather than the domain's acoustics).

    Args:
        cd_series: drag-coefficient history, one sample per ``sample_every``
            timesteps, shape ``(n,)``.
        cl_series: lift-coefficient history, same shape and cadence.
        sample_every: timesteps between consecutive samples.
        d_cells: characteristic length in cells (**D-019**).
        u_lattice: free-stream lattice velocity.
        skip_steps: timesteps at the start of the run that must not be
            measured — the startup kick's length.

    Returns:
        ``{"cd", "cd_std", "cd_amplitude", "cl", "cl_mean", "strouhal",
        "strouhal_confidence", "periods"}``. ``strouhal`` and its two
        companions are ``None`` when shedding was not detected.
    """
    cd = np.asarray(cd_series, dtype=np.float64).ravel()
    cl = np.asarray(cl_series, dtype=np.float64).ravel()
    start = max(
        int(round(TRANSIENT_FRACTION * cd.size)),
        -(-int(skip_steps) // max(1, int(sample_every))),  # ceil, in samples
    )
    cd_w = cd[start:]
    cl_w = cl[start:]

    out: dict[str, Any] = {
        "cd": float("nan"),
        "cd_std": float("nan"),
        "cd_amplitude": float("nan"),
        "cl": float("nan"),
        "cl_mean": float("nan"),
        "strouhal": None,
        "strouhal_confidence": None,
        "periods": None,
    }
    if cd_w.size < 2 or not np.isfinite(cd_w).all() or not np.isfinite(cl_w).all():
        if cd_w.size:
            out["cd"] = float(np.mean(cd_w))
        return out

    out["cd"] = float(np.mean(cd_w))
    out["cd_std"] = float(np.std(cd_w))
    out["cd_amplitude"] = float((cd_w.max() - cd_w.min()) / 2.0)
    out["cl"] = float((cl_w.max() - cl_w.min()) / 2.0)
    out["cl_mean"] = float(np.mean(cl_w))

    if out["cl"] < CL_AMPLITUDE_MIN * abs(out["cd"]) or cl_w.size < 16:
        return out  # gate 1: not shedding — None, not a number

    # Gate 2: could this window have shown a couple of cycles of the *slowest*
    # shedding we would believe? Checked before the estimate, against a period
    # nothing here estimated (:data:`MIN_PERIODS`).
    longest_period = d_cells / (u_lattice * ST_PLAUSIBLE[0])
    if cl_w.size * sample_every < MIN_PERIODS * longest_period:
        return out

    t_conv = d_cells / u_lattice  # steps per convective time
    sigma_samples = LOWPASS_SIGMA_TC * t_conv / float(sample_every)
    filtered = _lowpass(cl_w, sigma_samples)
    try:
        st = _strouhal(
            filtered,
            float(sample_every),
            d_cells,
            u_lattice,
            transient=0.0,
        )
    except ValueError:
        return out  # too few samples to place a peak; still not a number
    if not math.isfinite(st) or st <= 0.0:
        return out

    # Gate 3: what was found has to be shedding rather than something else the
    # spectrum contains (:data:`ST_PLAUSIBLE`).
    if not ST_PLAUSIBLE[0] <= st <= ST_PLAUSIBLE[1]:
        return out

    period_steps = d_cells / (u_lattice * st)
    periods = float(cl_w.size * sample_every / period_steps)

    out["strouhal"] = float(st)
    out["strouhal_confidence"] = float(_peak_dominance(filtered))
    out["periods"] = periods
    return out


# ---------------------------------------------------------------------------
# The output record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    """What one run produced — numbers first, pixels second.

    An **output record**: only :meth:`flow.case.Case.run` builds one, which is
    what makes **D-060**'s frozen-dataclass exemption from the constraint-13
    scan apply to its constructor.

    Every one of the eight reduced coefficients below is gated by
    :attr:`fidelity` (:data:`GATED_QUANTITIES`, constraint 18): they hold a
    number in the **quantitative** band and ``None`` in the other two. In the
    qualitative band the four the summary prints survive inside
    :attr:`cd_qualified`, where they cannot be read without the caveat that
    belongs to them; in the illustrative band there is no coefficient anywhere on
    the object.

    Attributes:
        cd: mean drag coefficient over the measurement window — **quantitative
            band only**, ``None`` otherwise.
        cd_std: its standard deviation over the same window. Gated.
        cd_amplitude: half peak-to-peak drag over the window. Gated.
        cl: lift-coefficient **amplitude** (half peak-to-peak), measured raw.
            Gated.
        cl_mean: mean lift over the window — ~0 for a symmetric body, and the
            check that the startup kick left nothing behind. Gated.
        strouhal: ``St = f D / U``, or ``None`` when the lift amplitude is
            under :data:`CL_AMPLITUDE_MIN` of ``|Cd|`` — "this is not
            shedding" is an answer, and a number would be a fabrication. Gated,
            so ``None`` here means *either* "not shedding" *or* "not this
            band", and :meth:`summary` says which.
        strouhal_confidence: how far the winning spectral peak stands above
            the next distinct one (:func:`_peak_dominance`). ``None`` when
            :attr:`strouhal` is. Gated.
        periods: how many shedding periods the measurement window covered.
            ``None`` when :attr:`strouhal` is. Gated.
        fidelity: the :class:`flow.fidelity.Band` this run **earned**, decided
            from the eddy viscosity it actually generated (**D-082**). Defaults
            to :attr:`~flow.fidelity.Band.QUANTITATIVE`, which is the band a
            plain-BGK run at ``Re <= 200`` earns by construction — a
            :class:`Result` built without one is Phase 1's result and behaves
            exactly as it did. :meth:`flow.case.Case.run` always passes it
            explicitly and ``tests/test_fidelity.py`` asserts that by reading
            the source.
        expected_fidelity: the band the :class:`~flow.autoconfig.Plan` expected
            *before* the run, or ``None``. When it differs from
            :attr:`fidelity` the earned band wins and :attr:`warnings` says so —
            a plan that expected ``quantitative`` and earned ``qualitative`` is
            a finding, not a footnote (T204's contract).
        cd_qualified: the drag coefficient with its band welded on
            (:class:`flow.fidelity.Qualified`), or ``None`` in the illustrative
            band, where nothing quantitative is claimed at all.
        nu_t_ratio: ``max(nu_t) / nu`` measured over the run's own measurement
            window — the number :attr:`fidelity` was decided from. Exactly
            ``0.0`` when the closure was off, because
            :func:`lbm.probe.eddy_viscosity` returns an all-zero field there.
        closure_engaged: whether the run used the Smagorinsky closure. A run
            that did is a substituted run (constraint 16) and says so in every
            artifact.
        convergence: the final velocity residual per timestep, scaled by the
            free-stream speed (:meth:`lbm.runner.Sim.residual`).
        convergence_history: the residual trace, shape ``(k,)``.
        peak_u: the largest fluid speed seen, in lattice units, against
            ``CLAUDE.md`` constraint 3's ceiling of 0.1.
        elapsed: wall clock of the run, seconds.
        substituted: **True** when this run is not the case that was asked
            for — constraint 16. It is carried here, on the object, so the
            summary, the dict and the video metadata all get it (**D-045**,
            **D-062**).
        substitution: what was changed, in one sentence; ``None`` when
            :attr:`substituted` is ``False``.
        frames: the rendered frames, ``(ny, nx, 3)`` ``uint8`` each, from the
            one :func:`lbm.render.render`. Empty when the run was asked not to
            keep them.
        fps: playback rate the frames were computed for.
        backend: which :class:`lbm.backends.Backend` ran the timesteps.
        steps: timesteps actually executed.
        stable: whether the state was finite at the end.
        cd_history: the whole drag history, shape ``(n,)``.
        cl_history: the whole lift history, same shape.
        sample_steps: timesteps between consecutive history samples.
        plan: the :class:`flow.autoconfig.Plan` this run was configured from.
        prepared: the :class:`flow.prepare.Prepared` geometry it ran.
        warnings: anything non-fatal worth printing, plan's included.
    """

    cd: float | None
    cd_std: float | None
    cd_amplitude: float | None
    cl: float | None
    cl_mean: float | None
    strouhal: float | None
    strouhal_confidence: float | None
    periods: float | None
    convergence: float
    peak_u: float
    elapsed: float
    substituted: bool
    backend: str
    steps: int
    stable: bool
    sample_steps: int
    fps: float
    fidelity: Band = Band.QUANTITATIVE
    expected_fidelity: Band | None = None
    cd_qualified: Qualified | None = None
    nu_t_ratio: float = 0.0
    closure_engaged: bool = False
    substitution: str | None = None
    frames: list[NDArray[np.uint8]] = field(default_factory=list)
    convergence_history: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    cd_history: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    cl_history: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    plan: "Plan | None" = None
    prepared: "Prepared | None" = None
    warnings: list[str] = field(default_factory=list)

    # -- constraint 18, in one place --------------------------------------

    def __post_init__(self) -> None:
        """Withhold every claim the band does not permit (**constraint 18**).

        ``DOCS/IDEA4.md`` § 1's table, applied to the record itself rather than
        to any one rendering of it. In order:

        1. If the band permits a *qualified* coefficient and none was supplied,
           build one out of the four numbers :meth:`summary` prints, so the
           qualitative band still has its ``Cd`` — welded to its caveat.
        2. If the band does not permit a *bare* coefficient, set every entry of
           :data:`GATED_QUANTITIES` to ``None`` on this object. After this,
           ``result.cd`` is ``None`` for anything but the quantitative band and
           there is nothing left for a printer, a dict or an MP4 comment to
           leak — which is why Rung H can assert constraint 18 *by inspecting
           the object*.
        3. If the band permits nothing quantitative at all
           (:attr:`~flow.fidelity.Band.ILLUSTRATIVE`), drop
           :attr:`cd_qualified` too. That is the stability-only half of Q-203's
           answer: a moving picture, and no drag coefficient anywhere.
        4. If the earned band is worse than the one the plan expected, say so in
           :attr:`warnings` — the earned band always wins.

        The dataclass is frozen, so the withholding goes through
        ``object.__setattr__``: freezing is there to stop a *caller* rewriting a
        result, and this is the record enforcing its own invariant before any
        caller has it.
        """
        band = self.fidelity
        if band.reports_qualified_numbers and self.cd_qualified is None:
            if self.cd is not None:
                object.__setattr__(
                    self,
                    "cd_qualified",
                    Qualified(
                        cd=float(self.cd),
                        cd_std=float(self.cd_std if self.cd_std is not None else float("nan")),
                        cl=float(self.cl if self.cl is not None else float("nan")),
                        strouhal=self.strouhal,
                        band=band,
                        caveat=_band_sentence(band),
                    ),
                )
        if not band.reports_bare_numbers:
            for name in GATED_QUANTITIES:
                object.__setattr__(self, name, None)
        if not band.reports_qualified_numbers:
            object.__setattr__(self, "cd_qualified", None)

        expected = self.expected_fidelity
        if expected is not None and expected is not band:
            note = (
                f"fidelity: this run was planned expecting the "
                f"'{expected}' band and earned '{band}'. The band it earned is "
                "the one that counts -- it was decided from the eddy viscosity "
                "this run actually generated (max(nu_t)/nu = "
                f"{self.nu_t_ratio:.4g}), not from its Reynolds number."
            )
            if note not in self.warnings:
                self.warnings.append(note)

    # -- the banner -------------------------------------------------------

    @property
    def substitution_banner(self) -> str:
        """One line saying this is not the case that was asked for, or ``""``.

        Constraint 16's actual text, in the one place every rendering of a
        :class:`Result` reads it from.
        """
        if not self.substituted:
            return ""
        what = self.substitution or "the case was changed to make it runnable"
        return (
            "  ** SUBSTITUTED ** this is not the case you asked for: "
            f"{what}"
        )

    def metadata(self) -> dict[str, str]:
        """Provenance for a saved artifact — the video's metadata included.

        Constraint 16: *"a run that differs from what was asked says so in
        every artifact it produces"*. :meth:`save` writes the ``comment``
        entry below into the container, so a file that outlives this process
        still says what it is. :meth:`flow.case.Case.run` writes the identical
        entry into a video it records *as it runs*, through the same builder —
        so which way the file was written cannot change what it claims.

        Returns:
            String-valued entries; ``substituted`` is always present, ``True``
            or ``False``, never absent.
        """
        return metadata_entries(
            substituted=self.substituted,
            substitution=self.substitution,
            reynolds=None if self.plan is None else self.plan.Re,
            backend=self.backend,
            fidelity=self.fidelity,
            closure_engaged=self.closure_engaged,
        )

    # -- renderings -------------------------------------------------------

    def summary(self, *, quiet: bool = False) -> str:
        """Print the numbers and return the same text.

        ``DOCS/IDEA3.md`` § 4. Everything the T108 contract names is here:
        ``Cd`` as mean ± standard deviation, the ``Cl`` amplitude, ``St`` with
        its confidence (or the reason there is none), peak ``|u|`` against the
        0.1 ceiling, the convergence, the elapsed wall clock, the backend —
        and, when it applies, the substitution banner (constraint 16).

        Args:
            quiet: build the text without printing it.

        Returns:
            The whole summary as one string.
        """
        lines: list[str] = ["", "results"]
        if self.substituted:
            lines.append(self.substitution_banner)
        lines.append(f"  fidelity      {_band_sentence(self.fidelity)}")
        if self.closure_engaged:
            lines.append(
                f"  closure       ON — the turbulence model supplied up to "
                f"{self.nu_t_ratio:.4g}x the fluid's own resistance to motion "
                "in this run (constraint 16: this is not the case you asked "
                "for, it is that case with a model carrying part of it)"
            )
        lines.extend(self._coefficient_lines())
        over = "  ** OVER THE LIMIT **" if self.peak_u >= 0.1 else ""
        lines.append(
            f"  peak |u|      {self.peak_u:.5f} of the 0.1 ceiling "
            f"(CLAUDE.md constraint 3){over}"
        )
        lines.append(
            f"  convergence   {self.convergence:.3e} velocity residual per step "
            f"(scaled by the free stream)"
        )
        lines.append(
            f"  elapsed       {self.elapsed:.1f} s for {self.steps} steps on "
            f"backend {self.backend!r}"
            + ("" if self.stable else "   ** the state is not finite **")
        )
        if self.frames:
            lines.append(
                f"  frames        {len(self.frames)} kept in memory at "
                f"{self.fps:g} fps — Result.save(path) writes them"
            )
        for warning in self.warnings:
            lines.append(f"  note          {warning}")
        text = "\n".join(lines)
        if not quiet:
            print(text)
        return text

    def _coefficient_lines(self) -> list[str]:
        """The ``Cd`` / ``Cl`` / ``St`` block, in whichever form the band allows.

        Three shapes, one per band, and the difference between them is the whole
        of constraint 18:

        * **quantitative** — the numbers, exactly as Phase 1 printed them.
        * **qualitative** — the same numbers, each on a line that names the band
          it came from, so the value never appears without the qualification.
          Read off :attr:`cd_qualified`, because the gated attributes are
          already ``None`` by then.
        * **illustrative** — no numbers. One line saying so, and why.
        """
        if self.fidelity is Band.ILLUSTRATIVE:
            return [
                "  Cd            not reported — illustrative. No drag "
                "coefficient, no lift amplitude and no shedding frequency are "
                "emitted in this band, because a number here would be a "
                "convincing-looking guess (CLAUDE.md constraint 18).",
                "  history       the raw force traces are still on this result "
                "(cd_history, cl_history) as the run's own data — they are not "
                "a coefficient and are not offered as one.",
            ]

        qualified = self.cd_qualified
        if self.fidelity is Band.QUALITATIVE and qualified is not None:
            st = (
                "None (not shedding, or too short a window)"
                if qualified.strouhal is None
                else f"{qualified.strouhal:.4f}"
            )
            return [
                f"  Cd            [qualitative] {qualified.cd:.4f} +- "
                f"{qualified.cd_std:.4f} — indicative, not a measurement to "
                "quote",
                f"  Cl            [qualitative] {qualified.cl:.4f} amplitude",
                f"  St            [qualitative] {st}",
            ]

        cd = float(self.cd) if self.cd is not None else float("nan")
        cd_std = float(self.cd_std) if self.cd_std is not None else float("nan")
        cl = float(self.cl) if self.cl is not None else float("nan")
        cl_mean = float(self.cl_mean) if self.cl_mean is not None else float("nan")
        lines = [
            f"  Cd            {cd:.4f} +- {cd_std:.4f} "
            f"(mean +- std over the last {1.0 - TRANSIENT_FRACTION:.0%} of the run)"
        ]
        cl_pct = (
            abs(cl / cd) * 100.0 if cd != 0.0 and math.isfinite(cd) else float("nan")
        )
        lines.append(
            f"  Cl            {cl:.4f} amplitude ({cl_pct:.1f}% of Cd), "
            f"{cl_mean:+.4f} mean"
        )
        if self.strouhal is None:
            lines.append(
                "  St            None — the lift amplitude is under "
                f"{CL_AMPLITUDE_MIN:.0%} of Cd (not shedding), or the window "
                f"is too short to hold {MIN_PERIODS:g} of the slowest "
                f"plausible shedding periods, or "
                f"what was found is outside St {ST_PLAUSIBLE[0]:g}..."
                f"{ST_PLAUSIBLE[1]:g} and is therefore not the wake. Not a "
                "number, because a number here would be a guess"
            )
        else:
            confidence = self.strouhal_confidence or float("nan")
            periods = self.periods or float("nan")
            lines.append(
                f"  St            {self.strouhal:.4f} "
                f"(peak {confidence:.1f}x the next distinct one, "
                f"{periods:.1f} periods observed)"
            )
        return lines

    def as_dict(self) -> dict[str, Any]:
        """The scalars as a plain dict — the third rendering ``DOCS/IDEA3.md`` § 4 names.

        Histories and frames are deliberately **not** in here: this is what a
        caller logs, compares or serialises, and it must stay small enough to
        do that with. ``substituted`` is present unconditionally.
        """
        out: dict[str, Any] = {
            "cd": self.cd,
            "cd_std": self.cd_std,
            "cd_amplitude": self.cd_amplitude,
            "cl": self.cl,
            "cl_mean": self.cl_mean,
            "strouhal": self.strouhal,
            "strouhal_confidence": self.strouhal_confidence,
            "periods": self.periods,
            "convergence": self.convergence,
            "peak_u": self.peak_u,
            "elapsed": self.elapsed,
            "steps": self.steps,
            "backend": self.backend,
            "stable": self.stable,
            "substituted": self.substituted,
            "substitution": self.substitution,
            # Constraint 18: the band travels with the numbers, always, and the
            # gated entries above are already None outside the quantitative one.
            "fidelity": self.fidelity.value,
            "expected_fidelity": (
                None if self.expected_fidelity is None else self.expected_fidelity.value
            ),
            "closure_engaged": self.closure_engaged,
            "nu_t_ratio": self.nu_t_ratio,
            "cd_qualified": (
                None if self.cd_qualified is None else self.cd_qualified.as_dict()
            ),
        }
        if self.plan is not None:
            out["Re"] = self.plan.Re
        return out

    def save(self, path: str | Path, *, fps: float | None = None) -> Path:
        """Write the kept frames — an MP4/GIF, or a directory of numbered PNGs.

        Both go through :mod:`lbm.record` (constraint 10: ``flow/`` composes
        the sinks, it does not colour anything — the frames were produced by
        the one :func:`lbm.render.render` during the run). Which sink is
        chosen is the **suffix**: a known video suffix or ``.gif`` writes a
        file through :class:`lbm.record.RecordSink`; anything without a suffix
        is a directory and gets :class:`lbm.record.HeadlessSink`.

        For a video the container also carries :meth:`metadata`, so
        ``substituted`` survives the process that produced it (constraint 16,
        **D-062**).

        Args:
            path: ``"wake.mp4"``, ``"wake.gif"`` or ``"frames/"``.
            fps: override the playback rate; defaults to :attr:`fps`.

        Returns:
            The path written — the file, or the directory.

        Raises:
            ValueError: when there are no frames to write, naming what to do
                about it. A file with no frames in it is not a saved result.
            RuntimeError: from :class:`lbm.record.RecordSink` when a video
                target has no ffmpeg — with :data:`lbm.record.FFMPEG_HINT`.
        """
        from lbm.record import HeadlessSink, RecordSink

        if not self.frames:
            raise ValueError(
                "this Result kept no frames, so there is nothing to save: run "
                "the case again with keep_frames=True, or pass record=<path> "
                "to Case.run() to write the video as it runs."
            )
        target = Path(path)
        rate = self.fps if fps is None else float(fps)

        sink: Any
        if target.suffix:
            extra: dict[str, Any] = {}
            if target.suffix.lower() != ".gif":
                # GIF goes through Pillow, which has no ffmpeg command line to
                # put a comment on; the video path does, and constraint 16 is
                # what it is for.
                extra["output_params"] = _ffmpeg_metadata_args(self.metadata())
            sink = RecordSink(target, fps=rate, **extra)
        else:
            sink = HeadlessSink(target)
        written = target
        try:
            for frame in self.frames:
                sink.push(frame)
        finally:
            sink.close()
        return written

    def plot(self, path: str | Path | None = None) -> "Figure":
        """A matplotlib figure of the scalar histories. **Not a renderer.**

        Constraint 10 says there is one ``render()`` and it lives in
        ``lbm/render.py``. This is a *different kind of output*: it draws
        ``Cd(t)``, ``Cl(t)`` and the convergence trace as lines on axes, and
        never maps a field to RGB — the field-to-RGB path is
        :func:`lbm.render.render` and there is still exactly one of it.

        Built on :class:`matplotlib.figure.Figure` directly rather than
        through ``pyplot``, so nothing here opens a window, chooses a backend
        or leaks global state into a caller that already has a figure open.

        Args:
            path: optionally save the figure there as well.

        Returns:
            The :class:`~matplotlib.figure.Figure`.
        """
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8.0, 7.0), dpi=110)
        ax_cd, ax_cl, ax_res = fig.subplots(3, 1, sharex=False)

        steps = np.arange(self.cd_history.size, dtype=np.float64) * self.sample_steps
        ax_cd.plot(steps, self.cd_history, lw=0.9, color="#3b4cc0")
        # Constraint 18 reaches the figure as well: the *trace* is the run's own
        # data and is always drawn, but the mean line and the title are a claim
        # about it, so they carry the band or they do not appear.
        qualified = self.cd_qualified
        if self.cd is not None:
            ax_cd.axhline(self.cd, ls="--", lw=0.8, color="#888888")
            cd_title = f"Cd {self.cd:.4f} +- {float(self.cd_std or float('nan')):.4f}"
        elif qualified is not None:
            ax_cd.axhline(qualified.cd, ls="--", lw=0.8, color="#888888")
            cd_title = (
                f"[{self.fidelity}] Cd {qualified.cd:.4f} +- "
                f"{qualified.cd_std:.4f} — indicative"
            )
        else:
            cd_title = f"[{self.fidelity}] no Cd reported — drag trace only"
        ax_cd.set_ylabel("Cd")
        ax_cd.set_title(
            cd_title + ("   ** SUBSTITUTED **" if self.substituted else "")
        )

        ax_cl.plot(steps, self.cl_history, lw=0.9, color="#b40426")
        ax_cl.set_ylabel("Cl")
        ax_cl.set_xlabel("timestep")
        if self.cl is not None:
            cl_title = "Cl amplitude " + f"{self.cl:.4f}" + (
                f"   St {self.strouhal:.4f}"
                if self.strouhal is not None
                else "   not shedding (St is None)"
            )
        elif qualified is not None:
            st = "None" if qualified.strouhal is None else f"{qualified.strouhal:.4f}"
            cl_title = (
                f"[{self.fidelity}] Cl amplitude {qualified.cl:.4f}   St {st} — "
                "indicative"
            )
        else:
            cl_title = f"[{self.fidelity}] no Cl or St reported — lift trace only"
        ax_cl.set_title(cl_title)

        res_steps = (
            np.arange(1, self.convergence_history.size + 1, dtype=np.float64)
            * self.sample_steps
        )
        ax_res.semilogy(
            res_steps,
            np.maximum(self.convergence_history, 1e-12),
            lw=0.9,
            color="#444444",
        )
        ax_res.set_ylabel("residual / step")
        ax_res.set_xlabel("timestep")
        ax_res.set_title(f"convergence {self.convergence:.3e}")

        fig.tight_layout()
        if path is not None:
            fig.savefig(str(path))
        return fig


def metadata_entries(
    *,
    substituted: bool,
    substitution: str | None,
    reynolds: float | None,
    backend: str,
    fidelity: Band | None = None,
    closure_engaged: bool = False,
    provisional: bool = False,
) -> dict[str, str]:
    """The provenance a recorded file carries, built in exactly one place.

    Constraint 16 says every artifact of a substituted run says so, and there
    are two ways a video gets written — :meth:`Result.save` after the fact and
    :meth:`flow.case.Case.run` while it runs. One builder, so the two cannot
    drift apart.

    Args:
        substituted: whether this run is the case that was asked for.
        substitution: what was changed, if anything.
        reynolds: the case's Reynolds number, if it is known.
        backend: which backend ran it.
        fidelity: the :class:`flow.fidelity.Band`, if it is known. **Constraint
            18 reaches into the container too**: a file that outlives this
            process carries the band beside the numbers it was made from, so an
            illustrative video cannot be found later and read as a measurement.
        closure_engaged: whether the Smagorinsky closure was on. Constraint 16
            in ``CLAUDE.md``'s own words — *"a run that engaged the closure is
            such a run, and carries its band with it."*
        provisional: mark ``fidelity`` as the band the plan **expected** rather
            than the one the run earned. :meth:`flow.case.Case.run` sets it,
            because a video written *as the run happens* has its container
            metadata fixed before the last timestep and cannot know the earned
            band; :meth:`Result.save` never does. A verdict the writer cannot
            have is labelled rather than stamped.

    Returns:
        ``{"comment": ..., "title": ...}``; ``substituted=True`` or
        ``substituted=False`` is always the first thing in the comment, and
        ``fidelity=`` is present whenever a band was given.
    """
    parts = [f"substituted={substituted}"]
    if substituted and substitution:
        parts.append(substitution)
    if fidelity is not None:
        parts.append(
            f"fidelity={fidelity.value}" + (" (planned)" if provisional else "")
        )
    parts.append(f"closure={'on' if closure_engaged else 'off'}")
    if reynolds is not None:
        parts.append(f"Re={reynolds:.4g}")
    parts.append(f"backend={backend}")
    return {"comment": "; ".join(parts), "title": "flow — vorticity"}


def _ffmpeg_metadata_args(metadata: dict[str, str]) -> list[str]:
    """``{"comment": "..."} -> ["-metadata", "comment=..."]`` for ffmpeg.

    ``imageio`` passes ``output_params`` straight to the ffmpeg command line,
    which is how :meth:`Result.save` gets ``substituted`` into an MP4's
    ``©cmt`` atom.
    """
    args: list[str] = []
    for key, value in metadata.items():
        args += ["-metadata", f"{key}={value}"]
    return args
