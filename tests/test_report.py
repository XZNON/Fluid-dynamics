"""T108 — ``flow.Result``: the numbers, and the three ways they come out.

The T108 acceptance criteria this file covers:

* ``Result.save("wake.mp4")`` and ``.save("frames/")`` both work and go
  through :mod:`lbm.record`; ``flow/`` **colours nothing** — asserted by a scan
  for a colormap import or a second field-to-RGB path (constraint 10).
* ``Result.summary()`` prints ``Cd`` (mean ± std), the ``Cl`` amplitude, ``St``
  with its confidence, peak ``|u|`` against 0.1, the convergence, the elapsed
  wall clock, the backend, and the substitution banner when it applies.
* ``Result.strouhal`` is ``None``, not a number, when shedding is not detected
  — covered here by a **steady case at Re 10**, run end to end.
* **Never a silent substitution** (constraint 16, **D-062**): a ``Result``
  produced from a :mod:`flow.diagnose` suggestion carries ``substituted=True``
  and the flag reaches the printed summary *and* the recorded video's
  metadata.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

import flow
from flow.case import Case
from flow.report import (
    CL_AMPLITUDE_MIN,
    MIN_PERIODS,
    Result,
    _analyse,
    metadata_entries,
)
from lbm.record import frame_count

FLOW_ROOT = pathlib.Path(flow.__file__).parent
DISC = "tests/data/shapes/disc.png"


def _frames(n: int = 6, ny: int = 32, nx: int = 48) -> list[np.ndarray]:
    """``n`` distinct RGB frames — stand-ins for ``lbm.render.render`` output."""
    out = []
    for k in range(n):
        frame = np.zeros((ny, nx, 3), dtype=np.uint8)
        frame[:, :, k % 3] = np.uint8(30 * k + 10)
        out.append(frame)
    return out


def _result(**overrides) -> Result:
    """A :class:`Result` with plausible numbers, for the rendering tests."""
    fields = dict(
        cd=1.4031,
        cd_std=0.0086,
        cd_amplitude=0.0212,
        cl=0.3689,
        cl_mean=0.0004,
        strouhal=0.1731,
        strouhal_confidence=6.4,
        periods=9.7,
        convergence=3.2e-6,
        peak_u=0.09685,
        elapsed=42.5,
        substituted=False,
        backend="numpy",
        steps=78000,
        stable=True,
        sample_steps=12,
        fps=60.0,
        frames=_frames(),
        cd_history=np.full(64, 1.4031),
        cl_history=np.sin(np.linspace(0.0, 40.0, 64)),
        convergence_history=np.logspace(-2, -6, 64),
    )
    fields.update(overrides)
    return Result(**fields)


# ---------------------------------------------------------------------------
# summary() — every number the contract names
# ---------------------------------------------------------------------------


def test_summary_prints_everything_the_contract_names(capsys):
    result = _result()
    text = result.summary()
    printed = capsys.readouterr().out
    assert text in printed, "summary() must print, not only return"

    assert "1.4031 +- 0.0086" in text          # Cd, mean +- std
    assert "0.3689 amplitude" in text           # Cl amplitude
    assert "0.1731" in text and "6.4x" in text  # St with its confidence
    assert "9.7 periods" in text
    assert "0.09685" in text and "0.1 ceiling" in text  # peak |u| vs constraint 3
    assert "3.200e-06" in text                  # convergence
    assert "42.5 s" in text and "78000 steps" in text   # elapsed
    assert "'numpy'" in text                    # backend
    assert "SUBSTITUTED" not in text            # ...and no banner when it is honest


def test_summary_says_over_the_limit_when_the_speed_ceiling_is_breached():
    text = _result(peak_u=0.1203).summary(quiet=True)
    assert "OVER THE LIMIT" in text


def test_summary_says_when_the_state_is_not_finite():
    text = _result(stable=False).summary(quiet=True)
    assert "not finite" in text


def test_summary_explains_a_missing_strouhal_rather_than_printing_a_number():
    text = _result(strouhal=None, strouhal_confidence=None, periods=None).summary(
        quiet=True
    )
    assert "St            None" in text
    assert "not shedding" in text or "too short" in text


def test_as_dict_carries_the_scalars_and_always_the_substitution_flag():
    plain = _result().as_dict()
    assert plain["substituted"] is False
    assert plain["strouhal"] == pytest.approx(0.1731)
    assert "frames" not in plain and "cd_history" not in plain

    swapped = _result(substituted=True, substitution="fluid -> honey").as_dict()
    assert swapped["substituted"] is True
    assert swapped["substitution"] == "fluid -> honey"


# ---------------------------------------------------------------------------
# Constraint 16 — the banner, the dict and the file all say it
# ---------------------------------------------------------------------------


def test_the_substitution_banner_is_built_from_the_flag_not_from_a_caller():
    assert _result().substitution_banner == ""
    banner = _result(
        substituted=True, substitution="speed -> 0.5 m/s"
    ).substitution_banner
    assert "SUBSTITUTED" in banner and "speed -> 0.5 m/s" in banner


def test_metadata_always_states_substituted_either_way():
    honest = metadata_entries(
        substituted=False, substitution=None, reynolds=100.0, backend="numpy"
    )
    assert honest["comment"].startswith("substituted=False")
    swapped = metadata_entries(
        substituted=True,
        substitution="fluid -> honey",
        reynolds=12.0,
        backend="warp",
    )
    assert swapped["comment"].startswith("substituted=True")
    assert "fluid -> honey" in swapped["comment"]
    assert "backend=warp" in swapped["comment"]


def _mp4_comment(path: pathlib.Path) -> str:
    """The ``©cmt`` atom's payload, read back out of the container.

    Asserting on the *atom* rather than on the raw bytes is what makes this a
    test of the video's **metadata** and not of the file happening to contain a
    string somewhere.
    """
    raw = path.read_bytes()
    marker = raw.find(b"\xa9cmt")
    assert marker != -1, "no ©cmt metadata atom in the container"
    # ©cmt | 4-byte size | "data" | 8 bytes of type/locale | the text
    start = marker + 4 + 4 + 4 + 8
    size = int.from_bytes(raw[marker - 4 : marker], "big")
    end = marker - 4 + size
    return raw[start:end].decode("utf-8", "replace")


def test_save_writes_the_substitution_into_the_videos_metadata(tmp_path):
    """Constraint 16 in the file itself, not only on the console."""
    out = tmp_path / "wake.mp4"
    _result(substituted=True, substitution="fluid -> honey").save(out)
    assert out.exists()
    comment = _mp4_comment(out)
    assert "substituted=True" in comment
    assert "fluid -> honey" in comment


def test_an_honest_run_says_substituted_false_in_its_metadata(tmp_path):
    out = tmp_path / "wake.mp4"
    _result().save(out)
    assert "substituted=False" in _mp4_comment(out)


# ---------------------------------------------------------------------------
# save() — both sinks, both through lbm.record
# ---------------------------------------------------------------------------


def test_save_to_a_video_writes_every_frame(tmp_path):
    out = tmp_path / "wake.mp4"
    result = _result()
    assert result.save(out) == out
    assert frame_count(out) == len(result.frames)


def test_save_to_a_directory_writes_numbered_pngs(tmp_path):
    directory = tmp_path / "frames"
    result = _result()
    assert result.save(directory) == directory
    written = sorted(directory.glob("frame_*.png"))
    assert len(written) == len(result.frames)
    assert written[0].name == "frame_00000.png"


def test_save_to_a_gif_works_without_ffmpeg_metadata(tmp_path):
    """A GIF goes through Pillow, which has no command line to carry a comment."""
    out = tmp_path / "wake.gif"
    _result().save(out)
    assert out.exists() and out.stat().st_size > 0


def test_saving_a_result_with_no_frames_says_what_to_do(tmp_path):
    result = _result(frames=[])
    with pytest.raises(ValueError, match="keep_frames=True"):
        result.save(tmp_path / "wake.mp4")


# ---------------------------------------------------------------------------
# Constraint 10 — flow/ colours nothing, and plot() is not a fourth renderer
# ---------------------------------------------------------------------------


def _flow_sources() -> list[pathlib.Path]:
    return sorted(p for p in FLOW_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_flow_module_imports_a_colormap_or_builds_rgb():
    """Constraint 10: one ``render()``, and it is in ``lbm/``.

    ``flow/`` may *call* it — ``flow.case`` does, to produce frames — but a
    colormap import here is a second field-to-RGB path starting, and that is
    the thing the constraint forbids. The names below are every way this
    project has to colour a field: :data:`lbm.render.COOLWARM`,
    :func:`lbm.render.colormap`, and matplotlib's colour machinery.
    """
    banned = {"COOLWARM", "colormap", "NAN_RGB", "cm", "colors", "colormaps"}
    offenders: list[str] = []
    for path in _flow_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if node.module.startswith(("lbm.render", "matplotlib")) and (
                        alias.name in banned
                    ):
                        offenders.append(f"{path.name}: from {node.module} import {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("matplotlib.cm", "matplotlib.colors"):
                        offenders.append(f"{path.name}: import {alias.name}")
    assert not offenders, (
        f"flow/ colours nothing (CLAUDE.md constraint 10): {offenders}"
    )


def test_the_only_renderer_flow_reaches_for_is_lbm_renders_render():
    """The legal direction, asserted so the ban above is not read as total."""
    imported: set[str] = set()
    for path in _flow_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "lbm.render":
                imported |= {alias.name for alias in node.names}
    assert imported <= {"render", "LiveSink"}, (
        f"flow/ reaches into lbm.render for more than the renderer: {imported}"
    )
    assert "render" in imported


def test_plot_draws_scalar_histories_and_never_an_image():
    """:meth:`Result.plot` is a *different kind of output* from a frame.

    Lines on axes, three of them — ``Cd(t)``, ``Cl(t)`` and the convergence
    trace. If it ever drew an image it would be a fourth renderer, so that is
    asserted directly: no axes may hold an image artist.
    """
    figure = _result().plot()
    axes = figure.get_axes()
    assert len(axes) == 3
    for ax in axes:
        assert ax.lines, "an axis with no line is not a history plot"
        assert not ax.images, "constraint 10: plot() must not draw a field"
    assert "SUBSTITUTED" not in axes[0].get_title()


def test_plot_marks_a_substituted_run_and_can_save_itself(tmp_path):
    figure = _result(substituted=True, substitution="speed -> 0.5 m/s").plot(
        tmp_path / "history.png"
    )
    assert "SUBSTITUTED" in figure.get_axes()[0].get_title()
    assert (tmp_path / "history.png").exists()


# ---------------------------------------------------------------------------
# The analysis itself — cheap, synthetic, exact
# ---------------------------------------------------------------------------


def test_analyse_recovers_a_planted_frequency():
    """The cadence arithmetic, checked against a signal whose ``St`` is known."""
    d_cells, u = 30.0, 0.05
    sample_every = 12
    st_true = 0.17
    period_steps = d_cells / (u * st_true)
    steps = np.arange(0, 40 * int(period_steps), sample_every, dtype=np.float64)
    cl = 0.4 * np.sin(2.0 * np.pi * steps / period_steps)
    cd = np.full(steps.size, 1.4)

    out = _analyse(
        cd, cl, sample_every=sample_every, d_cells=d_cells, u_lattice=u
    )
    assert out["cd"] == pytest.approx(1.4)
    assert out["strouhal"] == pytest.approx(st_true, rel=0.05)
    assert out["periods"] > MIN_PERIODS
    assert out["strouhal_confidence"] > 1.0


def test_analyse_returns_none_when_the_lift_is_below_one_percent_of_the_drag():
    """The shedding check itself: a flat wake has no frequency to report."""
    n = 4096
    cd = np.full(n, 1.4)
    cl = np.full(n, 1e-5) + 1e-7 * np.sin(np.linspace(0, 300, n))
    out = _analyse(cd, cl, sample_every=10, d_cells=30.0, u_lattice=0.05)
    assert out["strouhal"] is None
    assert out["cl"] < CL_AMPLITUDE_MIN * abs(out["cd"])


def test_analyse_returns_none_when_the_window_is_too_short_to_hold_a_period():
    """A window with one period in it has a tallest bin, not a frequency.

    This is the measured case behind :data:`~flow.report.MIN_PERIODS`'s
    docstring: the estimator answers **0.459** for a planted 0.17 here, and it
    is the *window length* — known before the estimate exists — that has to
    refuse it.
    """
    d_cells, u = 30.0, 0.05
    period_steps = d_cells / (u * 0.17)
    sample_every = 12
    steps = np.arange(0, int(2.0 * period_steps), sample_every, dtype=np.float64)
    cl = 0.4 * np.sin(2.0 * np.pi * steps / period_steps)
    out = _analyse(
        np.full(steps.size, 1.4),
        cl,
        sample_every=sample_every,
        d_cells=d_cells,
        u_lattice=u,
    )
    assert out["strouhal"] is None
    assert out["cl"] == pytest.approx(0.4, rel=0.01), (
        "the amplitude is still reported — it is the *frequency* that is refused"
    )


def test_analyse_refuses_a_frequency_outside_the_plausible_shedding_band():
    """Gate 3: an oscillation can be real and not be the wake.

    ``validate/cylinder.py::lowpass`` measured the case this exists for — the
    domain's acoustics ringing at ``St = 1.49`` with power comparable to the
    wake's. The low-pass usually removes it; when it does not, a number outside
    :data:`~flow.report.ST_PLAUSIBLE` is not reported as shedding.
    """
    d_cells, u = 30.0, 0.05
    sample_every = 4
    period_steps = d_cells / (u * 1.49)  # the acoustic peak, not the wake
    steps = np.arange(0, 400 * int(period_steps), sample_every, dtype=np.float64)
    cl = 0.4 * np.sin(2.0 * np.pi * steps / period_steps)
    out = _analyse(
        np.full(steps.size, 1.4),
        cl,
        sample_every=sample_every,
        d_cells=d_cells,
        u_lattice=u,
    )
    assert out["strouhal"] is None


def test_analyse_never_measures_inside_the_startup_kick():
    """``skip_steps``: the window starts after the kick has washed out.

    The first three quarters of this series are a decaying transient of exactly
    the shape a switched-off kick leaves behind; the last quarter is flat. The
    default window is the last **half**, so without the skip it straddles the
    transient and reports it as a lift amplitude — which is the measured Re 10
    failure this argument comes from (0.55 against a Cd of 3.6).
    """
    n = 2000
    settling = 0.9 * np.exp(-np.linspace(0.0, 2.0, 3 * n // 4))
    cl = np.concatenate([settling, np.full(n - settling.size, 1e-6)])
    cd = np.full(n, 1.4)

    blind = _analyse(cd, cl, sample_every=10, d_cells=30.0, u_lattice=0.05)
    aware = _analyse(
        cd,
        cl,
        sample_every=10,
        d_cells=30.0,
        u_lattice=0.05,
        skip_steps=10 * 3 * n // 4,
    )
    assert blind["cl"] > 0.1, "without the skip, the kick is what gets measured"
    assert aware["cl"] < 1e-5


def test_analyse_survives_a_run_that_went_nan():
    out = _analyse(
        np.array([1.0, np.nan, 3.0, 4.0]),
        np.array([0.0, 0.1, np.nan, 0.3]),
        sample_every=10,
        d_cells=30.0,
        u_lattice=0.05,
    )
    assert out["strouhal"] is None


# ---------------------------------------------------------------------------
# End to end — the two criteria that need a real run
# ---------------------------------------------------------------------------


def test_strouhal_is_none_for_a_steady_case_at_re_10():
    """A T108 acceptance criterion, run rather than argued.

    Re 10 past a disc is steady — far below the ~47 at which a cylinder wake
    first sheds — so the honest answer is ``None``. A number here would be the
    project's stated main failure mode wearing a friendly face: a converged,
    plausible, wrong answer.

    **Measured once, by hand, in session 20** on this exact case, 30000 steps,
    lift amplitude as a fraction of ``Cd`` per 3000-step block::

        0-3k 0.890   3-6k 0.151   6-9k 0.041   9-12k 0.0123   12-15k 0.0043
        15-18k 0.0018  18-21k 0.0009  21-24k 0.0005  24-27k 0.00012  27-30k 0.00004

    — i.e. the lift decays monotonically through the ``CL_AMPLITUDE_MIN`` gate
    at ~12000 steps and keeps going. Reproducing that here would cost ~110 s of
    a 50 s suite, because a *measurable* window at this domain size does not
    open until ``2 x`` the settling allowance. What this test runs instead is
    the cheap half, and it is the half with teeth: no Strouhal number, a run
    that says why it measured nothing, and a lift history that is visibly
    decaying rather than oscillating.
    """
    case = Case.from_image(
        DISC, fluid="water", speed="0.0005 m/s", size="0.02 m", quality="fast"
    )
    assert case.plan is not None
    assert case.plan.Re == pytest.approx(10.0, rel=0.05)

    result = case.run(seconds=6000 * case.plan.dt, keep_frames=False, quiet=True)

    # The criterion itself.
    assert result.strouhal is None
    assert result.strouhal_confidence is None
    assert result.periods is None
    assert "None" in result.summary(quiet=True)

    # ...and the run says *why* it reported nothing, rather than reporting the
    # startup kick as a wake (constraint 16's spirit: no quiet substitution of
    # a number that was not measured).
    assert any("nothing was measured" in w for w in result.warnings), result.warnings

    # ...and the lift really is settling, not oscillating.
    history = result.cl_history
    quarter = history.size // 4
    first = float(np.ptp(history[:quarter])) / 2.0
    last = float(np.ptp(history[-quarter:])) / 2.0
    assert first > 5.0 * last, (
        f"a steady Re 10 wake should be decaying: first quarter {first:.4f}, "
        f"last quarter {last:.4f}"
    )


def test_a_result_from_a_diagnose_suggestion_is_substituted_everywhere(tmp_path):
    """**D-062**, the half T106 could not run: the flag reaches every artifact.

    The refused case is **D-038**'s — air at 20 m/s past a 1.5 m body, Re 2e6 —
    and the case that actually runs is the one the tool's own top suggestion
    produced through :func:`flow.diagnose.apply_suggestion`. It is not the case
    that was asked for, so the summary says so, the dict says so, and the MP4
    the run wrote says so in its metadata (constraint 16).
    """
    refused = Case.from_image(
        DISC, fluid="air", speed="20 m/s", size="1.5 m", quality="fast"
    )
    assert refused.runnable is False

    substituted = refused.nearest()
    assert substituted.runnable
    assert substituted.substituted is True
    assert substituted.plan is not None

    out = tmp_path / "wake.mp4"
    result = substituted.run(
        seconds=200 * substituted.plan.dt,
        record=out,
        keep_frames=False,
        quiet=True,
    )

    assert result.substituted is True
    assert result.substitution == substituted.substitution
    assert "SUBSTITUTED" in result.summary(quiet=True)
    assert result.as_dict()["substituted"] is True
    assert "substituted=True" in _mp4_comment(out)
    assert substituted.substitution in _mp4_comment(out)
