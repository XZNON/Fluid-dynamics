"""``flow.cli`` — ``python -m flow``, the thing a person runs (T109).

``DOCS/IDEA3.md`` § Scope: Phase 1's deliverable is *a library plus a CLI*
(**D-044**), and § What Phase 1 is, concretely names the one-command form of
the three Python lines::

    python -m flow --shape wing.png --fluid air --speed "5 m/s" --size "10 cm" \\
        --out wake.mp4

This module is the argument parser and the exit codes and nothing else. Every
decision it appears to make is made a layer down and is only *relayed* here:
:class:`flow.case.Case` prepares the picture, plans the physics, refuses what
it cannot represent and composes the sinks; :class:`flow.report.Result` prints
the numbers. That is deliberate — a CLI is the easiest place in a project for
judgement to accumulate untested, and none of it can be reached by
``pytest`` from in here.

**What replaces what.** ``python -m lbm.runner`` was Phase 0's entry point and
is kept, working and tested, for the solver-level knobs this command
deliberately does not have (**D-072**). The difference is constraint 13: there
is no ``--resolution`` in cells here, no ``--tau-floor``, no ``--u-lattice``
and no ``--span-d``. The knob between accuracy and wall clock is
``--quality``, spelled in words (**D-068**), and every lattice number is
derived by :mod:`flow.autoconfig` and **printed** before the run.

**Exit codes**, matching Phase 0's convention (**D-038**):

``0``
    The run finished — or ``--explain`` printed a plan and ran nothing.
``1``
    The run started and did not survive: the state is not finite, or
    :class:`flow.diagnose.Monitor` tripped a divergence wire (**D-061**).
``2``
    Nothing ran. A refused case, a bad argument, or a missing tool — with the
    message and the way forward, never a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from flow.autoconfig import QUALITY_LEVELS, Unrepresentable
from flow.case import Case
from flow.diagnose import Diverging
from flow.fluids import known_fluids
from flow.report import Result

__all__ = ["build_parser", "main"]


#: What ``--help`` says about the case ``old-Docs/STATE1.md`` **D-038** made
#: this project's worked example of a refusal. The point of putting the
#: arithmetic in the help text is that the next person meets it *before* the
#: run rather than after it: air at 20 m/s past a 1.5 m body is ``Re = 2e6``,
#: and ``tau = 0.5 + 3 U N / Re`` reads 0.5000 at any resolution this project
#: will run, so BGK with bounce-back and no turbulence model (constraint 1)
#: cannot represent it. Phase 0's CLI answered that with ``--re 100``; this one
#: has no ``--re`` (it is not a quantity a user measures) and answers it with
#: ``--nearest`` instead, which runs the tool's own top suggestion and labels
#: the result as a substitution (constraint 16, **D-045**).
RE_LIMIT_NOTE: str = (
    "the Reynolds number is not a knob here -- it is speed x size / viscosity,\n"
    "and it is what decides whether a case can be simulated at all. This\n"
    "solver is D2Q9 / BGK / bounce-back with no turbulence model, so it tops\n"
    "out somewhere in the low thousands: --fluid air --speed '20 m/s'\n"
    "--size '1.5 m' is Re 2e6, tau reads 0.5000 at every resolution this\n"
    "project will run, and the case is REFUSED rather than run badly\n"
    "(exit 2). That is correct and it stays. What to do about it is printed\n"
    "with the refusal: slow it down, shrink it, change the fluid, or pass\n"
    "--nearest to run the tool's own top suggestion, clearly labelled as a\n"
    "different flow from the one you asked for."
)


def build_parser() -> Any:
    """The ``python -m flow`` argument parser.

    Every option here is a *physical* or an *output* quantity. Constraint 13
    is what that sentence is: no ``tau``, no lattice ``U``, no
    ``steps_per_frame``, no cell count — ``tests/test_flow_package.py`` and
    ``tests/test_case.py`` scan this module for them the moment it exists.

    Returns:
        The :class:`argparse.ArgumentParser`.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Simulate 2D flow past a shape and watch it, record it, or both.\n\n"
            "You give it a picture, a fluid, a speed and a size. Everything "
            "else -- resolution,\nrelaxation time, lattice velocity, domain, "
            "timestep, frame cadence, colour limits --\nis derived, printed "
            "before the run, and never typed."
        ),
        epilog=(
            "examples:\n"
            "  python -m flow --shape wing.png --fluid air --speed '5 m/s' "
            "--size '10 cm' --out wake.mp4\n"
            "  python -m flow --shape disc.png --fluid water --speed '5 mm/s' "
            "--size '2 cm' --explain\n"
            "  python -m flow --shape disc.png --fluid water --speed '5 mm/s' "
            "--size '2 cm' --live --out wake.mp4\n"
            "\n"
            + RE_LIMIT_NOTE
            + "\n\n"
            "fluids: " + ", ".join(known_fluids()) + "\n"
            "\n"
            "exit codes: 0 ran (or explained), 1 started and did not survive, "
            "2 nothing ran.\n"
            "\n"
            "python -m lbm.runner is still there for the solver-level knobs "
            "this command does\n"
            "not have (--re / --nu, --resolution in cells, --span-d, "
            "--u-lattice, --tau-floor,\n--checkpoint) -- see D-072.\n"
        ),
    )

    g = p.add_argument_group("the case")
    g.add_argument(
        "--shape",
        metavar="PATH",
        required=True,
        help="PNG or SVG of the body: solid dark on a light background",
    )
    g.add_argument(
        "--fluid",
        metavar="NAME",
        required=True,
        help="one of: " + ", ".join(known_fluids()),
    )
    g.add_argument(
        "--speed",
        metavar="Q",
        required=True,
        help="free-stream speed with its unit, e.g. '5 m/s', '20 km/h'",
    )
    g.add_argument(
        "--size",
        metavar="Q",
        required=True,
        help="the body's size across the flow, e.g. '10 cm', '1.5 m'",
    )
    g.add_argument(
        "--quality",
        choices=QUALITY_LEVELS,
        default="balanced",
        help="how finely the body is resolved (default balanced). The only "
        "knob between accuracy and wall clock, and it is spelled in words "
        "rather than cells on purpose",
    )
    g.add_argument(
        "--seconds",
        metavar="Q",
        default=None,
        help="physical time to simulate, e.g. '2 s' (default: the plan's own "
        "length, 20 convective times)",
    )
    g.add_argument(
        "--no-repair",
        dest="repair",
        action="store_false",
        help="do not repair the picture -- refuse it instead. Every repair "
        "that does happen is printed",
    )

    o = p.add_argument_group("output")
    o.add_argument("--out", metavar="PATH", help=".mp4 / .gif -- implies --record")
    o.add_argument(
        "--frames-dir",
        metavar="DIR",
        help="directory of numbered PNGs -- implies --headless",
    )
    o.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="open a window. Default: a window opens only when no --out and "
        "no --frames-dir was given, which is lbm.runner's rule. --no-live "
        "with neither of those runs the numbers and draws nothing at all",
    )
    o.add_argument("--record", action="store_true", help="write --out")
    o.add_argument("--headless", action="store_true", help="write --frames-dir")

    r = p.add_argument_group("run")
    r.add_argument(
        "--backend",
        choices=("numpy", "warp"),
        default="numpy",
        help="which backend runs the timesteps (default numpy, the reference "
        "oracle -- D-043)",
    )
    r.add_argument(
        "--explain",
        "--dry-run",
        dest="explain",
        action="store_true",
        help="print the plan and stop. Runs no timesteps; exits 0, or 2 if "
        "the case is refused",
    )
    r.add_argument(
        "--nearest",
        action="store_true",
        help="if the case is refused, run the nearest one that works instead. "
        "It is a DIFFERENT flow from the one asked for and every artifact it "
        "produces says so (constraint 16)",
    )
    r.add_argument("--quiet", action="store_true", help="numbers only")
    return p


def _run_mode(args: Any) -> tuple[str | None, str | None, bool]:
    """Turn the six output flags into the three :meth:`flow.case.Case.run` ones.

    ``--out`` implies ``--record`` and ``--frames-dir`` implies ``--headless``,
    which is Phase 0's rule (``lbm.runner._resolve_sinks``) and the one people
    already type. A run that asks for **no** sink at all opens a window, also
    as Phase 0 does: the answer to "what does this shape do" is the picture.

    ``--live`` is therefore three-valued rather than a switch. Left alone it
    is that rule; ``--live`` forces a window alongside whatever else is being
    written; ``--no-live`` suppresses it, and ``--no-live`` with neither
    ``--out`` nor ``--frames-dir`` is the only way to reach
    :meth:`flow.case.Case.run`'s **un-drawn** run — no vorticity field
    computed and no frame coloured, just the numbers (**D-071**). That run is
    what a script or a CI job wants and Phase 0 had no way to ask for, because
    Phase 0's CLI had no numbers to print at the end of it.

    The *mode* — whether display frames may be dropped — is **not** decided
    here and must not be: **D-039** makes it a function of whether a file is
    being written, :meth:`flow.case.Case.run` derives it through
    ``flow.case._resolve_sinks``, and ``tests/test_case.py`` asserts all eight
    combinations. Deriving it a second time in a CLI is how two copies of one
    rule start to disagree.

    Args:
        args: the parsed arguments.

    Returns:
        ``(record, headless, live)`` — the first two as paths or ``None``.

    Raises:
        ValueError: ``--record`` with nowhere to write it.
    """
    record = args.out if (args.record or args.out) else None
    if (args.record or args.out) and not args.out:
        raise ValueError("--record needs --out PATH (.mp4 or .gif)")

    headless: str | None = None
    if args.headless or args.frames_dir:
        headless = args.frames_dir or "frames"

    if args.live is None:
        live = not (record or headless)
    else:
        live = bool(args.live)
    return record, headless, live


def _check_writable(record: str | None) -> None:
    """Fail on a missing ffmpeg **before** anything is built, not at frame 1.

    ``lbm.record.RecordSink`` already calls :func:`lbm.record.check_ffmpeg` in
    its constructor for exactly this reason, and
    :meth:`flow.case.Case.run` builds its sinks before the first timestep — so
    the criterion is met either way. It is checked here as well because *here*
    is earlier still: a missing binary is known before the picture is loaded,
    before the plan is made, and before the user has read a page of derived
    numbers that are about to be thrown away.

    Args:
        record: the video path, or ``None``.

    Raises:
        RuntimeError: with :data:`lbm.record.FFMPEG_HINT`.
    """
    if record is None:
        return
    from lbm.record import VIDEO_SUFFIXES, check_ffmpeg

    if Path(record).suffix.lower() in VIDEO_SUFFIXES:
        check_ffmpeg()


def _print_refusal(case: Case, *, nearest_offered: bool) -> None:
    """Print why a case will not run, and what would fix it (constraint 14).

    :meth:`flow.case.Case.explain` prints the refusal in the user's own units
    together with a way forward. This adds the list ``--nearest`` will
    actually act on, because **the two are not the same list** and the
    difference is not cosmetic: ``explain`` renders the suggestions carried on
    the :class:`~flow.autoconfig.Unrepresentable` itself, while
    :meth:`flow.case.Case.nearest` takes the top of
    :attr:`flow.case.Case.suggestions`, which is
    :func:`flow.diagnose.suggest`'s ranked list — and that list is the one
    **D-063** prepends a ``"fluid"`` option to, since ``"fluid"`` is created
    only by :mod:`flow.diagnose` and only after the library entry's cited
    ``nu`` is checked to clear the floor. Measured on **D-038**'s own case
    (air, 20 m/s, 1.5 m): ``explain`` shows *speed, size*; ``nearest`` runs
    *fluid -> honey*. A CLI that printed the first list and then said "--nearest
    runs the first of these" would be telling the user something untrue about
    what it is about to do, which is constraint 16's concern one step earlier
    than constraint 16 looks. Queued as an inconsistency in ``explain`` itself;
    what is fixed **here** is that the printed list is the executed one.

    Args:
        case: the refused case.
        nearest_offered: whether ``--nearest`` was already passed (in which
            case offering it again would be noise).
    """
    print(case.explain(quiet=True), file=sys.stderr)

    offers = case.suggestions
    if offers:
        print(
            "\nWhat --nearest would run, in the order it would try them",
            file=sys.stderr,
        )
        for rank, offer in enumerate(offers, 1):
            print(f"  {rank}. {offer.change} -> {offer.value}", file=sys.stderr)
            print(f"     {offer.note}", file=sys.stderr)
    elif case.fix is not None:
        print(
            f"\nWhat --nearest would do\n  {case.fix.change} -> "
            f"{case.fix.value}\n     {case.fix.note}",
            file=sys.stderr,
        )

    if not nearest_offered:
        print(
            "\n  Nothing above is a tolerance that can be loosened: "
            "nu = (tau - 0.5) / 3 (CLAUDE.md constraint 2) and the lattice "
            "velocity ceiling is compressibility error (constraint 3).\n"
            "  Re-run with --nearest to run the first one listed above "
            "instead. It is a different flow from the one you asked for, and "
            "the summary, the report and the video's metadata all say so.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """``python -m flow`` — a picture and four physical numbers in, a video out.

    ``DOCS/IDEA3.md`` § Scope: the one-command form of the three Python lines
    the phase exists to make run. What happens, in order: the flags become a
    :class:`flow.case.Case` (which prepares the picture and plans the physics
    and runs **no timesteps**); the plan or the refusal is printed;
    ``--explain`` stops there; otherwise :meth:`flow.case.Case.run` drives the
    loop and :meth:`flow.report.Result.summary` prints the numbers.

    Args:
        argv: arguments, or ``None`` for ``sys.argv[1:]``.

    Returns:
        The process exit status — ``0``, ``1`` or ``2``, as this module's
        docstring describes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        record, headless, live = _run_mode(args)
        _check_writable(record)
    except (ValueError, RuntimeError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    try:
        case = Case.from_image(
            args.shape,
            fluid=args.fluid,
            speed=args.speed,
            size=args.size,
            quality=args.quality,
            repair=args.repair,
            backend=args.backend,
        )
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    if not case.runnable:
        _print_refusal(case, nearest_offered=args.nearest)
        if not args.nearest:
            return 2
        # D-045 / constraint 16: the nearest runnable case is offered
        # loudly and never as a silent substitution. It may itself be
        # refused -- the honest outcome when one fix is not enough, and
        # visible rather than hidden.
        case = case.nearest()
        print(
            "\n--nearest: running the case below instead. It is NOT the one "
            "you asked for.",
            file=sys.stderr,
        )
        if not case.runnable:
            _print_refusal(case, nearest_offered=True)
            return 2

    if args.explain:
        case.explain()
        return 0

    if not args.quiet:
        case.explain()

    try:
        result: Result = case.run(
            seconds=args.seconds,
            live=live,
            record=record,
            headless=headless,
            # The frames are already going to a file, or to a window, or
            # nowhere: this command never calls Result.save() afterwards, so
            # keeping a copy of every frame in memory as well is the 2.5 GB
            # default nobody chose (D-071).
            keep_frames=False,
            quiet=args.quiet,
        )
    except Unrepresentable as exc:  # pragma: no cover - runnable was checked
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except Diverging as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    if args.quiet:
        # `--quiet` is "numbers only", not "nothing": Case.run(quiet=True)
        # suppresses its own narration (the sinks line, the step count) and
        # the summary with it, so the summary is printed back here. A flag
        # that hid the result would leave the command with no output at all,
        # which is not a quieter answer -- it is no answer.
        result.summary()

    if not result.stable:
        print(
            "  the simulation produced nan -- the case was unstable.",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        if record is not None:
            print(f"  wrote {record}")
        if headless is not None:
            print(f"  wrote numbered PNGs to {headless}")
    return 0
