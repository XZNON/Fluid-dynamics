"""T011 — recording sinks and the command line.

Covers ``lbm/record.py`` (:class:`RecordSink`, :class:`HeadlessSink`,
:class:`TeeSink`) and the ``python -m lbm.runner`` entry point in
``lbm/runner.py``, one or more tests per acceptance criterion in
``old-Docs/TASKS1.md`` § T011.

The load-bearing ones, in the order the contract lists them:

* a recording of 50 frames is a **file with 50 frames**, counted off the disk
  and not off a counter the writer kept;
* the three sinks receive **byte-identical** frames — the enforceable form of
  ``CLAUDE.md`` constraint 10, "one ``render()``, three sinks";
* a recorder **never drops**, even behind a one-frame ring buffer and a slow
  writer (constraint 8's other half, **D-024**);
* a missing ffmpeg is an install line, not a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lbm.geometry import circle
from lbm.record import (
    FFMPEG_HINT,
    HeadlessSink,
    RecordSink,
    TeeSink,
    check_ffmpeg,
    frame_count,
)
from lbm.render import LiveSink, render
from lbm.runner import (
    RingBuffer,
    Sim,
    SimConfig,
    Sink,
    demo_domain,
    main,
    run,
    steps_per_frame,
)

DATA = Path(__file__).parent / "data"
TEST_PNG = DATA / "test_body.png"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def frames(n: int, ny: int = 32, nx: int = 48) -> list[np.ndarray]:
    """``n`` distinct valid frames — different data, identical shape."""
    out = []
    for i in range(n):
        a = np.empty((ny, nx, 3), dtype=np.uint8)
        a[..., 0] = (i * 5) % 256
        a[..., 1] = np.arange(nx, dtype=np.uint8)[None, :]
        a[..., 2] = np.arange(ny, dtype=np.uint8)[:, None]
        out.append(a)
    return out


def small_sim(steps: int = 12) -> Sim:
    """A little cylinder wake, advanced far enough to have structure in it."""
    solid = circle(40, 60, 18.0, 20.5, 4.0)
    cfg = SimConfig(
        ny=40,
        nx=60,
        tau=0.6,
        inlet_U=0.05,
        use_inlet=True,
        use_outlet=True,
        check_geometry=False,
    )
    sim = Sim(cfg, solid)
    sim.run_steps(steps)
    return sim


class CaptureSink(Sink):
    """Records the frames it is handed, and their identity."""

    def __init__(self) -> None:
        self.received: list[np.ndarray] = []
        self.ids: list[int] = []
        self.closed = False

    def push(self, frame: np.ndarray) -> None:
        self.received.append(np.array(frame, copy=True))
        self.ids.append(id(frame))

    def close(self) -> None:
        self.closed = True


class SpyLiveSink(LiveSink):
    """A :class:`LiveSink` with the window taken out.

    The criterion is about what the three sinks *receive*, and opening a real
    pygame window in the test suite would make it about SDL. Everything above
    ``push`` — the ring buffer, ``run``, the frame source — is untouched.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.received: list[np.ndarray] = []

    def push(self, frame: np.ndarray) -> None:  # type: ignore[override]
        self.received.append(np.array(frame, copy=True))
        self.frames += 1


# ---------------------------------------------------------------------------
# RecordSink — fixed framerate, never drops
# ---------------------------------------------------------------------------


def test_fifty_frames_in_gives_a_file_with_exactly_fifty(tmp_path: Path) -> None:
    """The T011 acceptance criterion, counted off the disk."""
    path = tmp_path / "fifty.mp4"
    sink = RecordSink(path, fps=30.0)
    for frame in frames(50):
        sink.push(frame)
    sink.close()

    assert sink.frames == 50
    assert path.exists() and path.stat().st_size > 0
    assert frame_count(path) == 50


def test_the_framerate_is_the_one_asked_for_and_not_the_arrival_rate(
    tmp_path: Path,
) -> None:
    """"Fixed framerate": the container says 24 because 24 was requested.

    Frames are pushed as fast as the loop can go; nothing samples the wall
    clock. A recorder whose rate came from arrival times would produce a
    different video on a busy machine.
    """
    import imageio.v2 as iio

    path = tmp_path / "rate.mp4"
    with RecordSink(path, fps=24.0) as sink:
        for frame in frames(24):
            sink.push(frame)  # type: ignore[attr-defined]

    reader = iio.get_reader(str(path))
    try:
        assert reader.get_meta_data()["fps"] == pytest.approx(24.0)
    finally:
        reader.close()


def test_gif_output_works_for_a_short_clip(tmp_path: Path) -> None:
    path = tmp_path / "clip.gif"
    with RecordSink(path, fps=20.0) as sink:
        for frame in frames(12):
            sink.push(frame)  # type: ignore[attr-defined]
    assert path.stat().st_size > 0
    assert frame_count(path) == 12


def test_a_gif_needs_no_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GIF goes through Pillow, so a machine without ffmpeg can still record."""

    def boom() -> str:
        raise RuntimeError("no binary here")

    import imageio_ffmpeg

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", boom)
    with pytest.raises(RuntimeError):
        check_ffmpeg()

    path = tmp_path / "no-ffmpeg.gif"
    with RecordSink(path, fps=10.0) as sink:
        for frame in frames(4):
            sink.push(frame)  # type: ignore[attr-defined]
    assert frame_count(path) == 4


def test_a_missing_ffmpeg_is_an_install_line_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``old-Docs/TASKS1.md`` § T011: "a clear install message, not a traceback"."""
    import imageio_ffmpeg

    def boom() -> str:
        raise RuntimeError("ffmpeg exe not found and could not be downloaded")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", boom)

    with pytest.raises(RuntimeError) as exc:
        RecordSink(tmp_path / "x.mp4")

    message = str(exc.value)
    assert message == FFMPEG_HINT
    assert 'pip.exe install "imageio[ffmpeg]"' in message
    # It fails at construction — before a single timestep has been run.
    assert not (tmp_path / "x.mp4").exists()


def test_the_ffmpeg_message_survives_imageio_ffmpeg_being_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other way it can be missing: the package itself is not installed."""
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    with pytest.raises(RuntimeError) as exc:
        check_ffmpeg()
    assert 'install "imageio[ffmpeg]"' in str(exc.value)


def test_a_push_after_close_raises_rather_than_being_ignored(tmp_path: Path) -> None:
    """A silently ignored frame is a video with a hole in it."""
    sink = RecordSink(tmp_path / "closed.gif", fps=10.0)
    sink.push(frames(1)[0])
    sink.close()
    with pytest.raises(ValueError, match="closed"):
        sink.push(frames(1)[0])


def test_record_sink_refuses_anything_that_is_not_render_output(
    tmp_path: Path,
) -> None:
    sink = RecordSink(tmp_path / "bad.gif", fps=10.0)
    with pytest.raises(ValueError, match="uint8"):
        sink.push(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        sink.push(np.zeros((8, 8, 3), dtype=np.float32))


def test_a_second_frame_of_a_different_size_is_refused(tmp_path: Path) -> None:
    sink = RecordSink(tmp_path / "size.gif", fps=10.0)
    sink.push(frames(1)[0])
    with pytest.raises(ValueError, match="one frame size"):
        sink.push(frames(1, ny=16, nx=16)[0])
    sink.close()


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_framerate_is_refused(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        RecordSink(tmp_path / "x.gif", fps=bad)


def test_an_unknown_suffix_names_the_ones_that_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported output suffix"):
        RecordSink(tmp_path / "frames.png")


# ---------------------------------------------------------------------------
# HeadlessSink
# ---------------------------------------------------------------------------


def test_headless_sink_writes_numbered_pngs(tmp_path: Path) -> None:
    sink = HeadlessSink(tmp_path / "out")
    pushed = frames(5)
    for frame in pushed:
        sink.push(frame)
    sink.close()

    written = sorted((tmp_path / "out").glob("*.png"))
    assert [p.name for p in written] == [
        "frame_00000.png",
        "frame_00001.png",
        "frame_00002.png",
        "frame_00003.png",
        "frame_00004.png",
    ]
    assert sink.frames == 5


def test_the_pngs_hold_exactly_the_bytes_they_were_given(tmp_path: Path) -> None:
    """PNG is lossless, so the file is the frame — no colour management."""
    import imageio.v2 as iio

    sink = HeadlessSink(tmp_path)
    frame = frames(1)[0]
    sink.push(frame)
    sink.close()
    assert np.array_equal(iio.imread(str(sink.paths[0])), frame)


def test_headless_needs_no_display(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing pygame at all would be a bug: "no display required"."""
    monkeypatch.setitem(sys.modules, "pygame", None)
    sink = HeadlessSink(tmp_path / "nodisplay")
    for frame in frames(3):
        sink.push(frame)
    sink.close()
    assert sink.frames == 3


def test_the_numbering_is_configurable_and_still_sorts(tmp_path: Path) -> None:
    sink = HeadlessSink(tmp_path, prefix="wake", digits=3, start=98)
    for frame in frames(3):
        sink.push(frame)
    sink.close()
    assert [p.name for p in sink.paths] == [
        "wake_098.png",
        "wake_099.png",
        "wake_100.png",
    ]


# ---------------------------------------------------------------------------
# Constraint 10 — one render(), three sinks
# ---------------------------------------------------------------------------


def test_the_three_sinks_receive_byte_identical_frames(tmp_path: Path) -> None:
    """``CLAUDE.md`` constraint 10, in its enforceable form.

    One sim state, one ``render()`` call, three sinks: what each of them
    receives must be the same bytes. If a sink ever coloured, scaled or
    gamma-corrected anything of its own, this is the test that fails.
    """
    import imageio.v2 as iio

    sim = small_sim()
    frame = render(sim.vorticity(), 0.01)

    live = SpyLiveSink()
    record = RecordSink(tmp_path / "three.gif", fps=10.0)
    headless = HeadlessSink(tmp_path / "three")
    capture = CaptureSink()

    tee = TeeSink(live, record, headless, capture)
    tee.push(frame)
    tee.close()

    # What the live sink got, what the headless sink wrote, and what the
    # recorder was handed are one array.
    assert np.array_equal(live.received[0], frame)
    assert np.array_equal(iio.imread(str(headless.paths[0])), frame)
    assert np.array_equal(capture.received[0], frame)
    # Same object, not merely equal values: the tee copies nothing.
    assert capture.ids[0] == id(frame)
    assert record.frames == 1


def test_the_three_sinks_agree_frame_by_frame_through_run(tmp_path: Path) -> None:
    """The same, but driven by :func:`lbm.runner.run` for several frames."""
    sim = small_sim(0)
    live = SpyLiveSink()
    headless = HeadlessSink(tmp_path / "series")
    capture = CaptureSink()

    stats = run(
        sim,
        TeeSink(live, headless, capture),
        steps=20,
        steps_per_frame=5,
        field=lambda s: render(s.vorticity(), 0.01),
        drop=False,
    )

    assert stats.frames == 4 == live.frames == headless.frames
    import imageio.v2 as iio

    for i in range(4):
        assert np.array_equal(live.received[i], capture.received[i])
        assert np.array_equal(iio.imread(str(headless.paths[i])), capture.received[i])


def test_a_sink_gets_render_output_and_not_a_physics_field() -> None:
    """The frame source is ``render()``; sinks never see floats."""
    sim = small_sim(2)
    frame = render(sim.vorticity(), 0.01)
    assert frame.dtype == np.uint8
    assert frame.shape == (sim.config.ny, sim.config.nx, 3)


# ---------------------------------------------------------------------------
# Constraint 8 — record must not drop; live may
# ---------------------------------------------------------------------------


def test_the_recorder_keeps_every_frame_behind_a_one_frame_buffer(
    tmp_path: Path,
) -> None:
    """**D-024**: ``drop=False`` drains inline and the sim waits for the sink.

    The buffer is deliberately one frame deep and the sink deliberately slow —
    the configuration that costs a live window 51 of 60 frames in T006's test.
    A recorder must come out with all of them.
    """
    import time

    class SlowRecordSink(RecordSink):
        def push(self, frame: np.ndarray) -> None:  # type: ignore[override]
            time.sleep(0.002)
            super().push(frame)

    sim = small_sim(0)
    path = tmp_path / "nodrop.gif"
    sink = SlowRecordSink(path, fps=10.0)
    ring = RingBuffer(1)

    stats = run(
        sim,
        sink,
        steps=40,
        steps_per_frame=2,
        field=lambda s: render(s.vorticity(), 0.01),
        drop=False,
        buffer=ring,
    )
    sink.close()

    assert stats.frames == 20
    assert stats.delivered == 20
    assert stats.dropped == 0
    assert ring.dropped == 0
    assert sink.frames == 20
    assert frame_count(path) == 20
    assert stats.steps == 40  # and never fewer: constraint 8


def test_the_recorded_file_length_matches_the_step_count_and_the_framerate(
    tmp_path: Path,
) -> None:
    """steps / steps_per_frame frames, and the file agrees."""
    sim = small_sim(0)
    path = tmp_path / "length.mp4"
    sink = RecordSink(path, fps=30.0)
    stats = run(
        sim,
        sink,
        steps=30,
        steps_per_frame=3,
        field=lambda s: render(s.vorticity(), 0.01),
        drop=False,
    )
    sink.close()
    assert stats.frames == 10
    assert frame_count(path) == 10


# ---------------------------------------------------------------------------
# TeeSink
# ---------------------------------------------------------------------------


def test_tee_closes_every_sink_even_if_one_raises() -> None:
    class Exploding(CaptureSink):
        def close(self) -> None:
            raise RuntimeError("boom")

    a, b, c = CaptureSink(), Exploding(), CaptureSink()
    tee = TeeSink(a, b, c)
    with pytest.raises(RuntimeError, match="boom"):
        tee.close()
    assert a.closed and c.closed


def test_tee_needs_at_least_one_sink() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TeeSink()


def test_tee_pushes_in_order() -> None:
    order: list[str] = []

    class Named(Sink):
        def __init__(self, name: str) -> None:
            self.name = name

        def push(self, frame: np.ndarray) -> None:
            order.append(self.name)

        def close(self) -> None:
            pass

    TeeSink(Named("a"), Named("b"), Named("c")).push(frames(1)[0])
    assert order == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Geometry placement for the CLI
# ---------------------------------------------------------------------------


def test_the_demo_domain_puts_the_blockage_where_span_d_says() -> None:
    """Constraint 12 falls out of the arithmetic, not out of a check."""
    body = circle(21, 21, 10.0, 10.0, 10.0)[1:-1, 1:-1]
    solid, d = demo_domain(body, span_d=12.0, upstream_d=6.0, downstream_d=10.0)
    ny, nx = solid.shape
    assert d / ny == pytest.approx(1.0 / 12.0, abs=0.005)
    assert d / ny < 0.10


def test_the_demo_domain_leaves_more_than_eight_diameters_downstream() -> None:
    from lbm.geometry import bounding_box

    body = circle(21, 21, 10.0, 10.0, 10.0)[1:-1, 1:-1]
    solid, d = demo_domain(body, span_d=12.0, upstream_d=6.0, downstream_d=10.0)
    box = bounding_box(solid)
    assert box is not None
    _y0, _y1, _x0, x1 = box
    assert (solid.shape[1] - 1 - x1) / d >= 8.0


def test_a_png_body_loads_cropped_to_itself_and_passes_the_mask_checks() -> None:
    """The committed T009 image, placed in a domain rather than stretched over one."""
    from lbm.geometry import check_mask
    from lbm.runner import _body_mask

    body, bh, bw = _body_mask(TEST_PNG, 30, verbose=False)
    assert body.any()
    # Cropped to the bounding box: every edge row and column touches the body.
    assert body[0].any() and body[-1].any() and body[:, 0].any() and body[:, -1].any()

    solid, d = demo_domain(body)
    assert d == bh
    assert check_mask(solid, "x", verbose=False) == []


def test_the_requested_resolution_is_the_body_and_not_the_picture() -> None:
    """``--resolution N`` means N cells across the **body** (D-019).

    ``tests/data/test_body.png`` has a wide margin: rasterised into a 30-row box
    the body is only 18 rows, which would run the case at ``tau = 0.527`` while
    the summary claimed 30 cells of resolution — inside the band D-029 measured
    a disc dying in. The loader rescales until the measured extent is the
    requested one; below 40 cells the same image is also a **1-cell** hairline,
    which is the constraint-12 failure ``check_mask`` exists to catch.
    """
    from lbm.geometry import min_thickness
    from lbm.runner import _body_mask

    for requested in (20, 30, 40):
        body, bh, _bw = _body_mask(TEST_PNG, requested, verbose=False)
        assert bh >= requested
        assert bh <= requested + 2  # and not wildly over
        solid, _d = demo_domain(body)
        assert min_thickness(solid) >= 3


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_cli_writes_a_playable_mp4_from_a_png_in_one_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """M4's gate in miniature: PNG in, physical units, MP4 out, one call.

    A tenth of a second of physical time rather than five, so the suite stays
    fast; the five-second run is in ``old-Docs/STATE1.md`` § Snapshot with its
    output, which is where a milestone is claimed.
    """
    out = tmp_path / "wake.mp4"
    code = main(
        [
            "--geometry", str(TEST_PNG),
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.05",
            "--resolution", "20",
            "--out", str(out),
        ]
    )
    assert code == 0
    assert out.exists() and out.stat().st_size > 0
    printed = capsys.readouterr().out
    assert "steps_per_frame" in printed  # constraint 7: computed, and printed
    assert frame_count(out) >= 1


def test_the_cli_computes_steps_per_frame_from_dt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Constraint 7 / **D-023** — the number in the output is the derived one."""
    from lbm.units import LatticeUnits

    out = tmp_path / "spf.gif"
    code = main(
        [
            "--demo", "cylinder",
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.02",
            "--resolution", "12",
            "--out", str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out

    # The same arithmetic the CLI did, done here from the physical inputs.
    body = circle(13, 13, 6.0, 6.0, 6.0)
    d = int(body.any(axis=1).sum())
    units = LatticeUnits.from_physical(
        u_phys=20.0, l_phys=1.5, re=100.0, cells_per_length=float(d)
    )
    expected = steps_per_frame(units.dt, 60.0, 1.0)
    assert f"= {expected} (constraint 7" in printed
    assert expected != 20  # the hardcoded number constraint 7 exists to forbid


def test_the_cli_refuses_a_case_it_cannot_represent_and_says_what_to_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Constraint 3 / 2, **D-032**: the units module raises and the CLI relays it.

    Air at 20 m/s past a 1.5 m body is Re 2e6. BGK with bounce-back and no
    turbulence model (constraint 1) cannot represent that at any resolution this
    project will run, so the honest outcome is a refusal naming the resolution
    that would be needed — not a plausible-looking video.
    """
    code = main(
        [
            "--geometry", str(TEST_PNG),
            "--fluid", "air",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.05",
            "--out", str(tmp_path / "never.mp4"),
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "tau" in err
    assert "cells_per_length" in err or "resolution" in err
    assert not (tmp_path / "never.mp4").exists()


def test_the_cli_composes_record_and_headless(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--record`` and ``--headless`` together, same frames to both."""
    out = tmp_path / "both.gif"
    dirn = tmp_path / "pngs"
    code = main(
        [
            "--demo", "cylinder",
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.02",
            "--resolution", "12",
            "--out", str(out),
            "--frames-dir", str(dirn),
        ]
    )
    assert code == 0
    capsys.readouterr()
    n_png = len(list(dirn.glob("*.png")))
    assert n_png >= 1
    assert frame_count(out) == n_png


def test_live_and_record_together_pick_the_non_dropping_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--live --record`` works, and the recorder's policy is the one that wins.

    The window is replaced by :class:`SpyLiveSink` so no SDL runs in the suite;
    what is under test is the wiring — that both sinks are built, that they are
    teed, and that the mode is ``drop=False`` (**D-024**), which is what makes
    the *recording* complete when the display is the slow one.
    """
    # `lbm/__init__.py` re-exports the *function* `render`, which shadows the
    # module attribute `lbm.render`; sys.modules is the module itself.
    monkeypatch.setattr(sys.modules["lbm.render"], "LiveSink", SpyLiveSink)
    out = tmp_path / "live-and-record.gif"
    code = main(
        [
            "--demo", "cylinder",
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.02",
            "--resolution", "12",
            "--live",
            "--out", str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "LiveSink" in printed and "RecordSink" in printed
    assert "drop=False" in printed
    assert frame_count(out) >= 1


def test_a_live_only_run_is_the_dropping_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Constraint 8: with nothing being written, display frames are droppable."""
    monkeypatch.setattr(sys.modules["lbm.render"], "LiveSink", SpyLiveSink)
    code = main(
        [
            "--demo", "cylinder",
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.02",
            "--resolution", "12",
            "--live",
        ]
    )
    assert code == 0
    assert "drop=True" in capsys.readouterr().out


def test_a_png_series_does_not_drop_either(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gap in the numbering is wrong output, not slow output."""
    dirn = tmp_path / "series"
    code = main(
        [
            "--demo", "cylinder",
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.02",
            "--resolution", "12",
            "--frames-dir", str(dirn),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "drop=False" in printed
    names = sorted(p.name for p in dirn.glob("*.png"))
    assert names == [f"frame_{i:05d}.png" for i in range(len(names))]


def test_record_without_an_output_path_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "--demo", "cylinder",
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.01",
            "--resolution", "12",
            "--record",
        ]
    )
    assert code == 2
    assert "--out" in capsys.readouterr().err


def test_the_cli_needs_a_fluid_and_a_geometry() -> None:
    with pytest.raises(SystemExit):
        main(["--velocity", "20", "--length", "1.5", "--seconds", "1"])
    with pytest.raises(SystemExit):
        main(
            [
                "--demo", "cylinder",
                "--velocity", "20",
                "--length", "1.5",
                "--seconds", "1",
            ]
        )


def test_a_missing_geometry_file_is_a_message_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "--geometry", "does-not-exist.png",
            "--re", "100",
            "--velocity", "20",
            "--length", "1.5",
            "--seconds", "0.01",
        ]
    )
    assert code == 2
    assert "not found" in capsys.readouterr().err
