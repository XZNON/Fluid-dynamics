"""T109 — ``python -m flow``: the flags, the exit codes, and what is printed.

The T109 acceptance criteria this file covers:

* the one-command form writes a playable file end to end;
* ``--explain`` / ``--dry-run`` prints the plan and exits **0 without
  simulating**, and a refused case prints the explanation and its suggestions
  and exits **2** (**D-038**'s convention);
* ``--quality`` / ``--seconds`` / ``--backend`` / ``--live`` / ``--record`` /
  ``--headless`` are wired, and ``--live --record`` composes with
  ``drop=False`` (**D-039**);
* ``python -m lbm.runner`` still runs, and says where the Phase 1 command is
  (**D-072**, closing **Q-101**);
* ``--help`` states the Re limit in plain words (**D-038**);
* a missing ffmpeg produces :data:`lbm.record.FFMPEG_HINT` before the first
  timestep rather than a traceback.

Two properties are asserted here that no single criterion names but that the
CLI would be wrong without. **Nothing runs until it is meant to**: the
``--explain`` tests assert no :class:`lbm.runner.Sim` was ever constructed,
not merely that the word "running" is absent. And **the printed suggestion
list is the executed one** — ``Case.explain`` renders the refusal's own
suggestions while ``Case.nearest`` acts on :attr:`flow.case.Case.suggestions`,
which **D-063** prepends a ``"fluid"`` option to; the CLI prints the second,
and ``test_the_suggestions_the_cli_prints_are_the_ones_nearest_would_run``
pins that.

Constraint 13 needs no test of its own here: ``tests/test_flow_package.py``
and ``tests/test_case.py`` both auto-parametrise over every module in
``flow/``, so ``flow/cli.py`` was covered the moment it existed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flow.case import Case, _resolve_sinks
from flow.cli import RE_LIMIT_NOTE, build_parser, main
from flow.report import metadata_entries
from lbm.record import FFMPEG_HINT, frame_count
from lbm.runner import PHASE1_CLI_POINTER

DISC = "tests/data/shapes/disc.png"
TINY = "tests/data/shapes/tiny_body.png"
ALL_BLACK = "tests/data/shapes/all_black.png"

#: A case that plans, in the CLI's own vocabulary: water at Re ~100 past a 2 cm
#: body — Rung 3's Reynolds number reached through the product path, the same
#: case ``tests/test_case.py`` and ``validate/autoconfig.py`` hold fixed.
WATER = [
    "--shape", DISC,
    "--fluid", "water",
    "--speed", "5 mm/s",
    "--size", "2 cm",
    "--quality", "fast",
]

#: **D-038**'s case, which Phase 0's CLI refuses and this one must go on
#: refusing: air at 20 m/s past a 1.5 m body is ``Re = 2e6``.
AIR = [
    "--shape", DISC,
    "--fluid", "air",
    "--speed", "20 m/s",
    "--size", "1.5 m",
]

#: The acceptance criterion's own literal case — air at 5 m/s past a 10 cm
#: body. It is ``Re = 32982`` and it is **refused**, which is
#: ``test_the_contracts_own_example_command_is_refused_and_that_is_correct``.
CRITERION = [
    "--shape", DISC,
    "--fluid", "air",
    "--speed", "5 m/s",
    "--size", "10 cm",
]


@pytest.fixture
def no_sim(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every :class:`lbm.runner.Sim` construction, so "ran nothing" is provable.

    ``--explain`` exiting 0 with no "running" line in the output is not the
    same claim as ``--explain`` running nothing: the second is the criterion
    and this is how it is checked.
    """
    import flow.case

    built: list[object] = []
    real = flow.case.Sim

    def spy(*args: object, **kwargs: object) -> object:
        built.append(args)
        return real(*args, **kwargs)  # type: ignore[operator]

    monkeypatch.setattr(flow.case, "Sim", spy)
    return built


# ---------------------------------------------------------------------------
# --help, and the arithmetic met before the run rather than after (D-038)
# ---------------------------------------------------------------------------


def test_help_states_the_reynolds_limit_in_plain_words() -> None:
    """**D-038**: the next person meets the arithmetic before the run.

    Phase 0's ``--help`` said air at 20 m/s past a 1.5 m body is Re 2e6 and is
    refused. This one says the same thing, and names what to do instead —
    which is the half Phase 0 could not offer because ``flow.diagnose`` did
    not exist yet.
    """
    text = build_parser().format_help()
    assert "Re 2e6" in text
    assert "REFUSED" in text
    assert "turbulence model" in text
    assert "--nearest" in text
    assert RE_LIMIT_NOTE.splitlines()[0] in text


def test_help_has_no_lattice_knob_in_it() -> None:
    """Constraint 13, at the surface a user actually types.

    ``--resolution`` in cells is the violation the T109 contract names by
    name; the knob for resolution is ``--quality``, in words (**D-068**).
    """
    options = {
        action.option_strings[0]
        for action in build_parser()._actions
        if action.option_strings
    }
    for banned in ("--resolution", "--tau-floor", "--u-lattice", "--span-d"):
        assert banned not in options
    assert "--quality" in options
    assert build_parser().get_default("quality") == "balanced"


def test_help_points_at_the_runner_for_the_knobs_it_does_not_have() -> None:
    """**D-072**: the two commands each say what the other is for."""
    text = build_parser().format_help()
    assert "python -m lbm.runner" in text
    assert "D-072" in text


# ---------------------------------------------------------------------------
# --explain / --dry-run: the plan, exit 0, and no timestep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--explain", "--dry-run"])
def test_explain_prints_the_plan_and_exits_zero_without_simulating(
    flag: str, no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 2, both spellings. Ran nothing is asserted, not inferred."""
    assert main(WATER + [flag]) == 0
    printed = capsys.readouterr().out
    assert "plan" in printed
    for derived in ("Re", "tau", "cells_per_length", "steps_per_frame", "dx", "dt"):
        assert derived in printed
    assert "why:" in printed  # every number carries its reason
    assert "estimated wall clock" in printed
    assert no_sim == [], "--explain must not construct a Sim"


def test_explain_prints_the_cost_before_the_cost_is_paid(
    no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """``DOCS/IDEA3.md`` § 5 — the wall clock is part of the plan, not a surprise."""
    assert main(WATER + ["--explain"]) == 0
    printed = capsys.readouterr().out
    assert "simulates" in printed and "timesteps on backend 'numpy'" in printed
    assert no_sim == []


# ---------------------------------------------------------------------------
# Refusal: the explanation, the suggestions, exit 2 (D-038's convention)
# ---------------------------------------------------------------------------


def test_a_refused_case_explains_itself_and_exits_two(
    no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 2's second half, on **D-038**'s own case.

    Air at 20 m/s past a 1.5 m body is Re 2e6; BGK with bounce-back and no
    turbulence model (constraint 1) cannot represent it. Phase 0 refused it and
    so does this — the refusal is the correct output and it stays.
    """
    assert main(AIR) == 2
    err = capsys.readouterr().err
    assert "turbulent" in err
    assert "tau" in err
    assert "What --nearest would run" in err
    assert no_sim == [], "a refused case must not construct a Sim"


def test_a_refused_case_exits_two_under_explain_too(no_sim: list[object]) -> None:
    """``--explain`` does not turn a refusal into a success."""
    assert main(AIR + ["--explain"]) == 2
    assert no_sim == []


def test_the_contracts_own_example_command_is_refused_and_that_is_correct(
    no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """The T109 criterion's literal command is a refusal, exactly as T011's was.

    ``--shape wing.png --fluid air --speed "5 m/s" --size "10 cm"`` is
    ``Re = 5 * 0.1 / 1.516e-5 = 32982``, and ``tau`` reads **0.500182** against
    the 0.54 bluff-body floor (**D-029**). This is **D-038** repeating itself
    one layer up: two acceptance criteria of the same task cannot both be met
    by one literal command — "writes a playable file" and "refuses a case it
    cannot represent" — and the second wins, for D-038's reason. What T109 has
    that T011 did not is a way through: ``--nearest``, tested below.
    """
    assert main(CRITERION) == 2
    assert "turbulent" in capsys.readouterr().err
    assert no_sim == []


def test_a_refused_picture_exits_two_as_a_refused_case_does(
    no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """**D-065** / **D-067**: a picture refusal is the same exit code and shape."""
    assert main(["--shape", ALL_BLACK, "--fluid", "water",
                 "--speed", "5 mm/s", "--size", "2 cm"]) == 2
    err = capsys.readouterr().err
    assert "refused" in err
    assert "--nearest" in err
    assert no_sim == []


def test_the_suggestions_the_cli_prints_are_the_ones_nearest_would_run() -> None:
    """The printed list is the executed list, on the case where they differ.

    ``Case.explain()`` renders the suggestions carried on the
    :class:`~flow.autoconfig.Unrepresentable`; ``Case.nearest()`` takes the top
    of :attr:`flow.case.Case.suggestions`, which is
    :func:`flow.diagnose.suggest`'s ranked list — and **D-063** has that list
    prepend a ``"fluid"`` option ``autoconfig`` never creates. On the
    criterion's own case the two disagree: ``explain`` shows *speed, size*
    while ``nearest`` runs *fluid -> honey*. The CLI prints the second, so
    "re-run with --nearest to run the first one listed above" is true.
    """
    case = Case.from_image(DISC, fluid="air", speed="5 m/s", size="10 cm")
    assert not case.runnable
    assert case.refusal is not None

    rendered = [s.change for s in case.refusal.suggestions]
    executed = [s.change for s in case.suggestions]
    assert executed[0] == "fluid"
    assert rendered != executed, (
        "this test is pinning a real divergence — if the two lists have been "
        "made one, the CLI's printed list can go back to explain()'s"
    )
    assert case.nearest().fluid.name == "honey"


def test_nearest_runs_the_substitute_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Constraint 16: a substitution is offered loudly, never quietly.

    Short enough to stay a unit test — the point is the banner and the exit
    code, not the physics, and D-069's "nothing was measured" note is the
    honest thing for a run this brief to print.
    """
    out = tmp_path / "sub.gif"
    code = main(CRITERION + ["--out", str(out), "--seconds", "0.02 s", "--nearest"])
    assert code == 0
    assert out.exists() and frame_count(out) >= 1
    captured = capsys.readouterr()
    assert "** SUBSTITUTED **" in captured.out
    assert "NOT the one you asked for" in captured.err


def test_without_nearest_a_refused_case_writes_nothing(tmp_path: Path) -> None:
    """The refusal is not a warning: no file is produced."""
    out = tmp_path / "never.mp4"
    assert main(AIR + ["--out", str(out)]) == 2
    assert not out.exists()


# ---------------------------------------------------------------------------
# The flags, and the mode that is not one of them (constraint 8 / D-039)
# ---------------------------------------------------------------------------


def test_every_flag_the_contract_names_is_wired() -> None:
    """Criterion 3, as a parser fact rather than as prose."""
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    for flag in ("--quality", "--seconds", "--backend", "--live", "--record",
                 "--headless", "--out", "--frames-dir", "--explain",
                 "--dry-run", "--shape", "--fluid", "--speed", "--size"):
        assert flag in options


@pytest.mark.parametrize("quality", ["fast", "balanced", "accurate"])
def test_quality_reaches_the_plan_as_a_resolution(
    quality: str, no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """**D-068**: the user's knob is a word; the plan's is a cell count."""
    from flow.autoconfig import QUALITY_CELLS

    argv = [a for a in WATER if a not in ("--quality", "fast")]
    assert main(argv + ["--quality", quality, "--explain"]) == 0
    printed = capsys.readouterr().out
    assert f"cells_per_length {QUALITY_CELLS[quality]}" in printed
    assert no_sim == []


@pytest.mark.parametrize("backend", ["numpy", "warp"])
def test_backend_reaches_the_plan(
    backend: str, no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """``--backend`` selects the :class:`lbm.backends.Backend`, and is printed."""
    assert main(WATER + ["--backend", backend, "--explain"]) == 0
    assert f"backend {backend!r}" in capsys.readouterr().out
    assert no_sim == []


def test_the_warp_estimate_keeps_its_honest_caveat(
    no_sim: list[object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Rung B's warp accuracy failure is disclosed where the estimate is printed.

    § Blockers, queued ``e4874a146490``, owned by **T110**. It touches T109
    only here, and the answer is to keep the caveat rather than to hide the
    estimate.
    """
    assert main(WATER + ["--backend", "warp", "--explain"]) == 0
    assert "over-predicts" in capsys.readouterr().out
    assert no_sim == []


def test_seconds_is_physical_time_and_shortens_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--seconds`` is seconds of *physical* time, converted through ``dt``."""
    case = Case.from_image(DISC, fluid="water", speed="5 mm/s", size="2 cm",
                           quality="fast")
    assert case.plan is not None
    expected = max(1, int(round(0.1 / case.plan.dt)))
    assert expected < case.plan.steps  # the flag really did shorten it

    code = main(WATER + ["--frames-dir", str(tmp_path / "png"),
                         "--seconds", "0.1 s"])
    assert code == 0
    assert f"running {expected} steps" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flags, record, headless, live, drop",
    [
        ([], None, None, True, True),
        (["--live"], None, None, True, True),
        (["--out", "a.gif"], "a.gif", None, False, False),
        (["--frames-dir", "d"], None, "d", False, False),
        (["--live", "--out", "a.gif"], "a.gif", None, True, False),
        (["--live", "--frames-dir", "d"], None, "d", True, False),
        (["--out", "a.gif", "--frames-dir", "d"], "a.gif", "d", False, False),
        (["--live", "--out", "a.gif", "--frames-dir", "d"], "a.gif", "d", True, False),
        # --live is three-valued: left alone it is Phase 0's rule, --no-live
        # suppresses the window, and --no-live with no file sink is the
        # un-drawn run (D-071) Phase 0 had no way to ask for.
        (["--no-live"], None, None, False, True),
        (["--no-live", "--out", "a.gif"], "a.gif", None, False, False),
    ],
)
def test_the_flags_compose_and_the_mode_does_not(
    flags: list[str],
    record: str | None,
    headless: str | None,
    live: bool,
    drop: bool,
) -> None:
    """Criterion 3's last clause, and **D-039** — all eight combinations.

    ``--live``, ``--record`` and ``--headless`` compose through
    :class:`lbm.record.TeeSink`; the *mode* does not, and the CLI does not
    decide it. ``drop`` here is read back out of ``flow.case._resolve_sinks``,
    which is the one place the rule lives: any sink that writes a **file**
    takes ``drop=False``, so ``drop=True`` is reached only by a live-only run.
    A second copy of that rule in the CLI is what this asserts is absent.
    """
    from flow.cli import _run_mode

    args = build_parser().parse_args(WATER + flags)
    got_record, got_headless, got_live = _run_mode(args)
    assert (got_record, got_headless, got_live) == (record, headless, live)

    _sink, _members, got_drop = _resolve_sinks(
        live=got_live,
        record=got_record,
        headless=got_headless,
        metadata=metadata_entries(
            substituted=False, substitution=None, reynolds=100.0, backend="numpy"
        ),
    )
    assert got_drop is drop


def test_no_live_with_no_file_sink_draws_nothing_and_still_reports(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """**D-071**'s un-drawn run, reachable from the command line.

    With nothing to show and nothing to keep, :meth:`flow.case.Case.run`
    computes no vorticity field and colours no frame — it pushes a shared 1x1
    placeholder to :class:`lbm.runner.NullSink` so the ring buffer's ``None``
    sentinel keeps meaning "empty". The numbers still print, which is the
    whole point: this is the shape a script or a CI job wants, and Phase 0's
    CLI could not ask for it because it had no numbers to print.
    """
    assert main(WATER + ["--no-live", "--seconds", "0.02 s"]) == 0
    printed = capsys.readouterr().out
    assert "NullSink" in printed
    assert "results" in printed
    assert "peak |u|" in printed
    assert "frames" not in printed.split("results")[1]


def test_live_is_three_valued_not_a_switch() -> None:
    """``--live`` / ``--no-live`` / neither are three different answers."""
    from flow.cli import _run_mode

    parser = build_parser()
    assert parser.parse_args(WATER).live is None
    assert parser.parse_args(WATER + ["--live"]).live is True
    assert parser.parse_args(WATER + ["--no-live"]).live is False

    # Forced on alongside a file sink, and forced off without one.
    assert _run_mode(parser.parse_args(WATER + ["--live", "--out", "a.gif"]))[2]
    assert not _run_mode(parser.parse_args(WATER + ["--no-live"]))[2]


def test_record_without_a_destination_is_an_error_not_a_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--record`` with nowhere to write it exits 2 with the fix in the message."""
    assert main(WATER + ["--record"]) == 2
    assert "--out" in capsys.readouterr().err


def test_headless_defaults_its_directory_like_phase_zero_does() -> None:
    """``--headless`` alone writes into ``frames/`` — ``lbm.runner``'s rule."""
    from flow.cli import _run_mode

    args = build_parser().parse_args(WATER + ["--headless"])
    assert _run_mode(args)[1] == "frames"


# ---------------------------------------------------------------------------
# Missing ffmpeg: the hint, before the first timestep
# ---------------------------------------------------------------------------


def test_missing_ffmpeg_prints_the_hint_before_the_first_timestep(
    tmp_path: Path,
    no_sim: list[object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Criterion 6 — :data:`lbm.record.FFMPEG_HINT`, exit 2, no traceback.

    ``RecordSink`` already checks in its constructor, which is before the run
    loop. The CLI checks earlier still: before the picture is loaded and
    before a page of derived numbers is printed that is about to be discarded.
    ``no_sim`` is what makes "before the first timestep" a measurement rather
    than a reading of the source.
    """
    import lbm.record

    def absent() -> str:
        raise RuntimeError(FFMPEG_HINT)

    monkeypatch.setattr(lbm.record, "check_ffmpeg", absent)
    out = tmp_path / "never.mp4"
    assert main(WATER + ["--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "ffmpeg" in err
    assert 'install "imageio[ffmpeg]"' in err
    assert not out.exists()
    assert no_sim == [], "the hint must arrive before any Sim is built"


def test_a_gif_needs_no_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hint's own last line, asserted: GIF goes through Pillow.

    A check that fired on every output format would make the message false.
    """
    import lbm.record

    def absent() -> str:  # pragma: no cover - must not be reached
        raise RuntimeError(FFMPEG_HINT)

    monkeypatch.setattr(lbm.record, "check_ffmpeg", absent)
    out = tmp_path / "clip.gif"
    assert main(WATER + ["--out", str(out), "--seconds", "0.02 s"]) == 0
    assert out.exists()


# ---------------------------------------------------------------------------
# End to end, and the entry point itself
# ---------------------------------------------------------------------------


def test_one_command_writes_a_playable_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Criterion 1, in miniature: shape + fluid + speed + size in, file out.

    A hundredth of a second of physical time rather than the plan's own 80, so
    the suite stays a suite; the full-length run is in ``DOCS/STATE2.md`` with
    its output, which is where a gate is claimed. The fluid is water rather
    than the criterion's air because the criterion's own case is **refused** —
    see ``test_the_contracts_own_example_command_is_refused_and_that_is_correct``.
    """
    out = tmp_path / "wake.mp4"
    assert main(WATER + ["--out", str(out), "--seconds", "0.02 s"]) == 0
    assert out.exists() and out.stat().st_size > 0
    assert frame_count(out) >= 1
    printed = capsys.readouterr().out
    assert "steps_per_frame" in printed  # constraint 7: computed, and printed
    assert "results" in printed  # Result.summary ran


def test_headless_writes_numbered_pngs(tmp_path: Path) -> None:
    """``--headless`` / ``--frames-dir``, and **D-039**'s reason for it."""
    directory = tmp_path / "png"
    assert main(WATER + ["--frames-dir", str(directory), "--seconds", "0.02 s"]) == 0
    written = sorted(directory.glob("*.png"))
    assert written
    assert written[0].name == "frame_00000.png"


def test_quiet_prints_the_numbers_and_not_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--quiet`` suppresses the narration, **not** the result.

    Found by running it: passing ``quiet=True`` straight through to
    :meth:`flow.case.Case.run` suppresses :meth:`flow.report.Result.summary`
    as well, and a command that prints nothing at all is not a quieter answer.
    So the summary is printed back by the CLI, and this asserts both halves —
    which the first version of this test did not.
    """
    out = tmp_path / "q.gif"
    assert main(WATER + ["--out", str(out), "--seconds", "0.02 s", "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "why:" not in printed  # no plan
    assert "running" not in printed  # no narration
    assert "results" in printed  # but the numbers, always
    assert "Cd " in printed and "peak |u|" in printed


def test_a_missing_picture_is_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 2 with the path in the message."""
    assert main(["--shape", "no_such_file.png", "--fluid", "water",
                 "--speed", "5 mm/s", "--size", "2 cm", "--explain"]) == 2
    assert "no_such_file.png" in capsys.readouterr().err


def test_an_unknown_fluid_names_the_ones_there_are(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Constraint 14's habit, applied to a typo."""
    assert main(["--shape", DISC, "--fluid", "custard",
                 "--speed", "5 mm/s", "--size", "2 cm", "--explain"]) == 2
    err = capsys.readouterr().err
    assert "custard" in err
    assert "water" in err  # the library, offered


def test_python_dash_m_flow_is_the_entry_point(tmp_path: Path) -> None:
    """``python -m flow`` from a cold shell, which is the criterion's own form.

    A subprocess rather than a call to :func:`flow.cli.main`, because
    ``flow/__main__.py`` and the ``-m`` machinery are exactly what this is
    checking — an in-process call would pass with no ``__main__.py`` at all.
    """
    out = tmp_path / "cold.gif"
    done = subprocess.run(
        [sys.executable, "-m", "flow", "--shape", DISC, "--fluid", "water",
         "--speed", "5 mm/s", "--size", "2 cm", "--quality", "fast",
         "--seconds", "0.02 s", "--out", str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert out.exists() and frame_count(out) >= 1


def test_importing_flow_main_does_not_run_the_cli() -> None:
    """The ``__name__`` guard in ``flow/__main__.py``.

    ``tests/test_flow_package.py`` imports every module under ``flow.`` to
    scan it for constraint 13. Without the guard, collecting the suite would
    run the CLI with pytest's own argv.
    """
    import importlib

    module = importlib.import_module("flow.__main__")
    assert module.main is main


# ---------------------------------------------------------------------------
# Q-101 / D-072 — the Phase 0 entry point survives, and says where to go
# ---------------------------------------------------------------------------


def test_the_phase_zero_runner_still_produces_an_mp4(tmp_path: Path) -> None:
    """**D-072**, the half that had to be *chosen*: ``lbm.runner`` still runs.

    The M4 gate command in ``old-Docs/STATE1.md`` § Snapshot stays literally
    reproducible, in miniature here (0.05 s of physical time rather than 5)
    and at full length in ``DOCS/STATE2.md``'s session 21 entry. Delegating to
    ``flow.cli`` was never an option: ``flow/`` may import ``lbm/`` and
    ``lbm/`` may **never** import ``flow/`` (constraint 15), so the choice was
    only ever between keeping this working and reducing it to a pointer.
    """
    from lbm.runner import main as runner_main

    out = tmp_path / "legacy.mp4"
    code = runner_main(
        ["--geometry", "tests/data/test_body.png", "--re", "100",
         "--velocity", "20", "--length", "1.5", "--seconds", "0.05",
         "--resolution", "20", "--out", str(out)]
    )
    assert code == 0
    assert out.exists() and frame_count(out) >= 1


def test_the_phase_zero_runner_points_at_the_phase_one_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """**D-072**'s other half: it is kept *and* it says where to go."""
    from lbm.runner import _build_parser

    assert "python -m flow" in PHASE1_CLI_POINTER
    assert "D-072" in PHASE1_CLI_POINTER
    assert PHASE1_CLI_POINTER in _build_parser().format_help()

    from lbm.runner import main as runner_main

    with pytest.raises(SystemExit):
        runner_main(["--help"])
    assert "python -m flow" in capsys.readouterr().out


def test_the_runner_still_owns_the_knobs_flow_does_not_have() -> None:
    """**D-072**'s reason, asserted rather than asserted-in-prose.

    Keeping ``lbm.runner`` costs a ``--help`` line; deleting it would cost
    these. ``--span-d`` in particular is the knob **Q-104** is a question
    about, and T110 owns that question.
    """
    from lbm.runner import _build_parser

    options = {
        option
        for action in _build_parser()._actions
        for option in action.option_strings
    }
    for knob in ("--re", "--nu", "--resolution", "--span-d", "--upstream-d",
                 "--downstream-d", "--u-lattice", "--tau-floor", "--checkpoint"):
        assert knob in options

    # And the same knobs are absent from `flow`, which is constraint 13 read
    # from the other end: the two commands are not two spellings of one thing.
    flow_options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    lattice_knobs = {"--re", "--nu", "--resolution", "--span-d", "--upstream-d",
                     "--downstream-d", "--u-lattice", "--tau-floor", "--vmax",
                     "--fps"}
    assert not (lattice_knobs & flow_options)
