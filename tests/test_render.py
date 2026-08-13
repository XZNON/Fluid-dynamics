"""Tests for :mod:`lbm.render` — T007.

The acceptance criteria of ``old-Docs/TASKS1.md`` § T007 that are unit-testable:
a diverging colormap with **fixed symmetric** limits, a mapping that is
byte-identical across frames (the assertion form of "no flicker",
``CLAUDE.md`` constraint 9), one renderer feeding the sink rather than three
renderers (constraint 10), and a ``LiveSink`` whose pygame calls all happen
where the sink is driven and never in the physics loop (constraint 8).

The window tests run against SDL's ``dummy`` video driver, so they need no
display and pass in CI.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from lbm.render import COOLWARM, NAN_RGB, LiveSink, colormap, render
from lbm.runner import RingBuffer, Sim, SimConfig, Sink, run

# ---------------------------------------------------------------------------
# The colormap
# ---------------------------------------------------------------------------


def test_the_colormap_is_uint8_rgb_of_the_requested_length() -> None:
    lut = colormap(64)
    assert lut.shape == (64, 3)
    assert lut.dtype == np.uint8


def test_the_colormap_is_diverging_not_sequential() -> None:
    """Cold at one end, warm at the other, neutral in the middle.

    A sequential map (dark -> light) has monotone luminance; a diverging one
    peaks in the middle, which is what puts zero vorticity in the background
    colour and both signs of rotation in front of it
    (``DOCS/IDEA2.md`` § What to actually draw).
    """
    lut = COOLWARM.astype(np.float64)
    lum = lut @ np.array([0.299, 0.587, 0.114])
    mid = lut.shape[0] // 2

    assert lum[mid] > lum[0]
    assert lum[mid] > lum[-1]
    # Cold end is blue-dominant, warm end red-dominant.
    assert lut[0, 2] > lut[0, 0]
    assert lut[-1, 0] > lut[-1, 2]
    # The midpoint is near-neutral: R, G and B within a few counts.
    assert np.ptp(lut[mid]) < 8


def test_the_colormap_has_an_odd_length_so_zero_lands_on_one_entry() -> None:
    assert COOLWARM.shape[0] % 2 == 1
    zero = render(np.zeros((1, 1), np.float32), 1.0)[0, 0]
    assert np.array_equal(zero, COOLWARM[COOLWARM.shape[0] // 2])


def test_the_map_is_symmetric_about_zero() -> None:
    """``+v`` and ``-v`` are mirror entries — a diverging map's whole point."""
    n = COOLWARM.shape[0]
    for value in (0.25, 0.5, 1.0, 3.0):
        hi = render(np.array([[value]], np.float32), 1.0)[0, 0]
        lo = render(np.array([[-value]], np.float32), 1.0)[0, 0]
        i_hi = int(np.flatnonzero((COOLWARM == hi).all(axis=1))[0])
        i_lo = int(np.flatnonzero((COOLWARM == lo).all(axis=1))[0])
        assert i_hi + i_lo == n - 1


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_returns_uint8_rgb_of_the_field_shape() -> None:
    out = render(np.zeros((7, 11), np.float32), 0.1)
    assert out.shape == (7, 11, 3)
    assert out.dtype == np.uint8


def test_a_fixed_value_maps_to_identical_bytes_across_two_different_frames() -> None:
    """**The no-flicker criterion** (``old-Docs/TASKS1.md`` § T007, constraint 9).

    Two frames of entirely different data, the same limits: the cell holding
    the same value must come out byte-identical. This fails the moment anyone
    "improves" ``render`` by autoscaling to the data.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 0.01, (32, 48)).astype(np.float32)
    b = rng.normal(0.0, 0.50, (32, 48)).astype(np.float32)
    probe = np.float32(0.0123)
    a[5, 9] = probe
    b[5, 9] = probe

    ra = render(a, 0.05)
    rb = render(b, 0.05)

    assert not np.array_equal(ra, rb)  # the frames really do differ
    assert np.array_equal(ra[5, 9], rb[5, 9])


def test_the_whole_frame_is_reproducible_for_the_same_data() -> None:
    rng = np.random.default_rng(1)
    field = rng.normal(0.0, 0.02, (16, 16)).astype(np.float32)
    assert np.array_equal(render(field, 0.05), render(field, 0.05))


def test_values_outside_the_limits_clamp_to_the_end_colours() -> None:
    field = np.array([[-100.0, 100.0]], np.float32)
    out = render(field, 0.05)
    assert np.array_equal(out[0, 0], COOLWARM[0])
    assert np.array_equal(out[0, 1], COOLWARM[-1])


def test_nan_cells_get_the_solid_colour_and_do_not_join_the_scale() -> None:
    """``lbm.probe.vorticity`` writes ``nan`` on solid cells; the body is a hole."""
    field = np.array([[np.nan, 0.0]], np.float32)
    out = render(field, 0.05)
    assert tuple(out[0, 0]) == NAN_RGB
    assert np.array_equal(out[0, 1], COOLWARM[COOLWARM.shape[0] // 2])


def test_an_all_nan_field_renders_without_raising() -> None:
    out = render(np.full((4, 4), np.nan, np.float32), 0.05)
    assert (out == np.asarray(NAN_RGB, np.uint8)).all()


def test_render_accepts_a_symmetric_pair_and_refuses_a_lopsided_one() -> None:
    field = np.zeros((2, 2), np.float32)
    assert np.array_equal(render(field, (-0.05, 0.05)), render(field, 0.05))
    with pytest.raises(ValueError, match="symmetric"):
        render(field, (-0.05, 0.10))


def test_render_refuses_degenerate_limits_and_a_non_2d_field() -> None:
    with pytest.raises(ValueError):
        render(np.zeros((2, 2), np.float32), 0.0)
    with pytest.raises(ValueError):
        render(np.zeros((2, 2, 2), np.float32), 0.05)


def test_render_writes_into_a_preallocated_out_buffer() -> None:
    field = np.linspace(-1, 1, 24).reshape(4, 6).astype(np.float32)
    out = np.empty((4, 6, 3), np.uint8)
    got = render(field, 1.0, out=out)
    assert got is out
    assert np.array_equal(out, render(field, 1.0))


def test_render_never_looks_at_the_data_range() -> None:
    """Scaling the data must not change the colour of an unchanged cell."""
    field = np.zeros((8, 8), np.float32)
    field[0, 0] = 0.01
    quiet = render(field, 0.05)[0, 0].copy()
    field[4, 4] = 1000.0  # a huge outlier elsewhere
    assert np.array_equal(render(field, 0.05)[0, 0], quiet)


# ---------------------------------------------------------------------------
# LiveSink
# ---------------------------------------------------------------------------


def a_frame(ny: int = 12, nx: int = 20) -> np.ndarray:
    rng = np.random.default_rng(3)
    return render(rng.normal(0, 0.02, (ny, nx)).astype(np.float32), 0.05)


def test_live_sink_is_a_sink_and_opens_no_window_before_the_first_frame() -> None:
    sink = LiveSink()
    assert isinstance(sink, Sink)
    assert sink._screen is None
    sink.close()


def test_live_sink_opens_a_window_on_the_first_frame_and_counts_frames() -> None:
    sink = LiveSink()
    try:
        sink.push(a_frame())
        assert sink._screen is not None
        sink.push(a_frame())
        assert sink.frames == 2
    finally:
        sink.close()
    assert sink.closed


def test_the_window_is_sized_from_the_frame_and_the_scale() -> None:
    sink = LiveSink(scale=3)
    try:
        sink.push(a_frame(12, 20))
        assert sink._screen.get_size() == (60, 36)
    finally:
        sink.close()


def test_live_sink_rejects_anything_that_is_not_a_rendered_frame() -> None:
    """Constraint 10: it consumes ``render``'s output; it does not colour."""
    sink = LiveSink()
    try:
        with pytest.raises(ValueError):
            sink.push(np.zeros((4, 4), np.float32))
    finally:
        sink.close()


def test_close_is_idempotent_and_a_closed_sink_ignores_frames() -> None:
    sink = LiveSink()
    sink.push(a_frame())
    sink.close()
    sink.close()
    sink.push(a_frame())
    assert sink.frames == 1


def test_no_pygame_call_happens_on_the_physics_thread() -> None:
    """Constraint 8, asserted rather than claimed.

    ``run(..., drop=True)`` drains the ring buffer from a consumer thread
    (D-024). This records the thread every :meth:`LiveSink.push` runs on and
    asserts it is never the thread that called :meth:`Sim.step`.
    """
    physics_thread: list[int] = []
    push_threads: set[int] = set()

    class ThreadRecordingSink(LiveSink):
        def push(self, frame: np.ndarray) -> None:  # type: ignore[override]
            push_threads.add(threading.get_ident())
            super().push(frame)

    cfg = SimConfig(ny=16, nx=32, tau=0.6, inlet_U=0.05, use_inlet=True,
                    use_outlet=True, check_geometry=False)
    sim = Sim(cfg)
    sink = ThreadRecordingSink()

    def note(_s: Sim) -> None:
        if not physics_thread:
            physics_thread.append(threading.get_ident())

    try:
        stats = run(
            sim,
            sink,
            steps=20,
            steps_per_frame=2,
            field=lambda s: render(s.vorticity(), 0.05),
            drop=True,
            per_step=note,
        )
    finally:
        sink.close()

    assert stats.steps == 20
    assert push_threads, "the sink never received a frame"
    assert physics_thread[0] not in push_threads


def test_a_slow_window_costs_display_frames_and_never_a_step() -> None:
    """The live half of ``DOCS/IDEA2.md`` § Never block the sim on the display."""
    import time as _time

    class SlowSink(LiveSink):
        def push(self, frame: np.ndarray) -> None:  # type: ignore[override]
            _time.sleep(0.004)
            super().push(frame)

    cfg = SimConfig(ny=16, nx=32, tau=0.6, inlet_U=0.05, use_inlet=True,
                    check_geometry=False)
    sim = Sim(cfg)
    sink = SlowSink()
    ring = RingBuffer(2)
    try:
        stats = run(
            sim,
            sink,
            frames=40,
            steps_per_frame=1,
            field=lambda s: render(s.vorticity(), 0.05),
            drop=True,
            buffer=ring,
        )
    finally:
        sink.close()

    assert stats.steps == 40  # every simulation step ran
    assert ring.dropped > 0  # display frames are what gave way
    assert sink.frames < stats.frames


def test_the_sink_receives_exactly_what_render_produced() -> None:
    """Constraint 10: one renderer. The sink is handed bytes, unmodified."""
    received: list[np.ndarray] = []

    class CapturingSink(LiveSink):
        def push(self, frame: np.ndarray) -> None:  # type: ignore[override]
            received.append(frame.copy())
            super().push(frame)

    cfg = SimConfig(ny=16, nx=32, tau=0.6, inlet_U=0.05, use_inlet=True,
                    check_geometry=False)
    sim = Sim(cfg)
    sink = CapturingSink()
    try:
        run(
            sim,
            sink,
            frames=1,
            steps_per_frame=5,
            field=lambda s: render(s.vorticity(), 0.05),
            drop=False,
        )
    finally:
        sink.close()

    assert len(received) == 1
    assert np.array_equal(received[0], render(sim.vorticity(), 0.05))
