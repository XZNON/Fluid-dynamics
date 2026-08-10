"""Tests for :mod:`lbm.runner` — T006.

The acceptance criteria of ``DOCS/TASKS1.md`` § T006, one test apiece:
allocation-freeness, a computed ``steps_per_frame``, a ring buffer that drops
the oldest *display* frame and never a step, an abstract ``Sink``, a checkpoint
that holds exactly four things, and — the one that earns the session — a
bit-identical restart (``CLAUDE.md`` constraint 11).
"""

from __future__ import annotations

import pickle
import time
import tracemalloc

import numpy as np
import pytest

from lbm.boundary import (
    apply_body_force,
    bounce_back,
    force_velocity_shift,
)
from lbm.core import Q, W, collide, equilibrium, macroscopic, stream
from lbm.geometry import bounding_box, channel_walls, circle, strip_solid_border
from lbm.runner import (
    NullSink,
    RingBuffer,
    Sim,
    SimConfig,
    Sink,
    load_checkpoint,
    run,
    save_checkpoint,
    steps_per_frame,
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

NY, NX = 24, 80


def channel_with_cylinder() -> np.ndarray:
    """Walls top and bottom plus a small disc — inlet, outlet and links all live."""
    return channel_walls(NY, NX) | circle(NY, NX, cx=20.0, cy=11.5, radius=3.0)


def flow_config(**over) -> SimConfig:
    """A driven channel: Zou–He inlet, convective outlet, an obstacle."""
    cfg = SimConfig(
        ny=NY,
        nx=NX,
        tau=0.6,
        inlet_U=0.05,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        check_geometry=False,
    )
    return cfg.replace(**over) if over else cfg


def forced_config(**over) -> SimConfig:
    """Rung 1's setup: periodic channel driven by a body force, no open ends."""
    cfg = SimConfig(ny=22, nx=32, tau=0.6, g=(2.6667e-5, 0.0), check_geometry=False)
    return cfg.replace(**over) if over else cfg


# ---------------------------------------------------------------------------
# Criterion 1 — Sim.step() allocates nothing
# ---------------------------------------------------------------------------


def test_step_keeps_the_f_buffer_identity():
    """``f`` is written in place forever; nothing rebinds it.

    The restart test leans on this: if ``step`` swapped ``f`` for a fresh array
    the checkpoint would be saving a different object than the one the loop
    advances.
    """
    sim = Sim(flow_config(), channel_with_cylinder())
    before = sim.f.__array_interface__["data"]
    sim.run_steps(1000)
    assert sim.f.__array_interface__["data"] == before


def test_step_allocates_nothing_over_1000_steps():
    """``tracemalloc`` sees no growth across 1000 steps (D-006, conventions)."""
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(50)  # let any one-off caches settle

    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    sim.run_steps(1000)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grown = sum(
        stat.size_diff for stat in after.compare_to(base, "filename") if stat.size_diff > 0
    )
    # One full (9, ny, nx) float32 buffer at this size is ~69 kB; the loop must
    # stay far below a single one of those.
    assert grown < 20_000, f"step loop grew the heap by {grown} bytes"


def test_step_advances_the_counter_and_stays_finite():
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(200)
    assert sim.step_count == 200
    assert np.isfinite(sim.f).all()
    assert sim.f.dtype == np.float32
    assert sim.f.shape == (Q, NY, NX)


def test_two_identical_sims_stay_identical():
    """No RNG anywhere in the step path (constraint 11's other half)."""
    a = Sim(flow_config(), channel_with_cylinder())
    b = Sim(flow_config(), channel_with_cylinder())
    a.run_steps(300)
    b.run_steps(300)
    assert np.array_equal(a.f, b.f)


# ---------------------------------------------------------------------------
# Criterion 2 — steps_per_frame is computed
# ---------------------------------------------------------------------------


def test_steps_per_frame_is_the_documented_arithmetic():
    """``round(speed / (fps * dt))`` — the docstring's formula, checked."""
    assert steps_per_frame(5e-4, 60.0, 1.0) == round(1.0 / (60.0 * 5e-4))
    assert steps_per_frame(1e-3, 30.0, 2.0) == round(2.0 / (30.0 * 1e-3))
    assert steps_per_frame(1e-4, 60.0, 1.0) == round(1.0 / (60.0 * 1e-4))


def test_steps_per_frame_halves_with_dt():
    """A grid refined by 2 halves ``dt`` and doubles the steps per frame.

    This is the property constraint 7 is actually about: the number tracks the
    grid instead of being a literal 20.
    """
    coarse = steps_per_frame(1e-3, 60.0)
    fine = steps_per_frame(5e-4, 60.0)
    assert fine == pytest.approx(2 * coarse, abs=1)
    assert coarse != 20 or fine != 20


def test_steps_per_frame_never_returns_zero():
    """Slow motion on a coarse grid still advances the physics."""
    assert steps_per_frame(1.0, 60.0, 0.001) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": -1e-3},
        {"dt": 1e-3, "fps": 0.0},
        {"dt": 1e-3, "speed": -1.0},
    ],
)
def test_steps_per_frame_rejects_nonpositive_arguments(kwargs):
    with pytest.raises(ValueError):
        steps_per_frame(**kwargs)


# ---------------------------------------------------------------------------
# Criterion 3 — RingBuffer drops the oldest
# ---------------------------------------------------------------------------


def test_ring_buffer_drops_the_oldest_frame():
    ring = RingBuffer(3)
    for i in range(5):
        ring.push(i)
    assert ring.dropped == 2
    assert [ring.pop() for _ in range(3)] == [2, 3, 4]
    assert ring.pop() is None


def test_ring_buffer_does_not_drop_under_capacity():
    ring = RingBuffer(4)
    for i in range(4):
        assert ring.push(i) is True
    assert ring.dropped == 0
    assert len(ring) == 4
    assert ring.push(4) is False
    assert ring.dropped == 1


def test_ring_buffer_rejects_zero_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)


class SlowSink(Sink):
    """A sink that takes visible wall-clock time per frame."""

    def __init__(self, delay: float = 0.004) -> None:
        self.delay = delay
        self.count = 0
        self.closed = False

    def push(self, frame) -> None:
        time.sleep(self.delay)
        self.count += 1

    def close(self) -> None:
        self.closed = True


def test_a_slow_sink_drops_frames_and_never_steps():
    """Constraint 8, asserted rather than commented.

    The sink is ~100x slower than a frame's worth of physics here, so the ring
    buffer must fill and shed display frames — while ``step_count`` comes out at
    exactly ``frames * steps_per_frame``.
    """
    sim = Sim(flow_config(), channel_with_cylinder())
    sink = SlowSink(delay=0.004)
    ring = RingBuffer(2)

    stats = run(sim, sink, frames=60, steps_per_frame=2, buffer_size=2, buffer=ring)

    assert sim.step_count == 120
    assert stats.steps == 120
    assert stats.frames == 60
    assert stats.dropped > 0
    assert sink.count + stats.dropped == 60


def test_record_mode_delivers_every_frame():
    """``drop=False`` is the recording sink of T011: never drop, always in order."""
    sim = Sim(flow_config(), channel_with_cylinder())
    sink = NullSink()
    stats = run(sim, sink, frames=12, steps_per_frame=3, drop=False, buffer_size=2)

    assert stats.dropped == 0
    assert sink.count == 12
    assert stats.delivered == 12
    assert sim.step_count == 36


# ---------------------------------------------------------------------------
# Criterion 4 — Sink is abstract, NullSink exists
# ---------------------------------------------------------------------------


def test_sink_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Sink()


def test_a_sink_missing_push_cannot_be_instantiated():
    class Halfway(Sink):
        def close(self) -> None:
            pass

    with pytest.raises(TypeError):
        Halfway()


def test_null_sink_counts_and_closes():
    sink = NullSink()
    for _ in range(3):
        sink.push(object())
    assert sink.count == 3
    assert sink.closed is False
    sink.close()
    assert sink.closed is True


def test_null_sink_is_the_default_and_gets_closed():
    sim = Sim(flow_config(), channel_with_cylinder())
    stats = run(sim, frames=3, steps_per_frame=2)
    assert stats.frames == 3
    assert sim.step_count == 6


# ---------------------------------------------------------------------------
# Criterion 5 — the checkpoint holds exactly four things
# ---------------------------------------------------------------------------


def test_checkpoint_contains_exactly_f_solid_step_count_and_config(tmp_path):
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(20)
    path = save_checkpoint(sim, tmp_path / "ckpt.pkl")

    with open(path, "rb") as fh:
        state = pickle.load(fh)

    assert set(state) == {"f", "solid", "step_count", "config", "format"}
    assert np.array_equal(state["f"], sim.f)
    assert np.array_equal(state["solid"], sim.solid)
    assert state["step_count"] == 20
    assert state["config"] == sim.config


def test_load_checkpoint_restores_state_byte_for_byte(tmp_path):
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(30)
    save_checkpoint(sim, tmp_path / "c.pkl")

    back = load_checkpoint(tmp_path / "c.pkl")
    assert np.array_equal(back.f, sim.f)
    assert np.array_equal(back.solid, sim.solid)
    assert back.step_count == sim.step_count
    assert back.f.dtype == np.float32


def test_load_checkpoint_rejects_a_foreign_pickle(tmp_path):
    path = tmp_path / "junk.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"hello": 1}, fh)
    with pytest.raises(ValueError):
        load_checkpoint(path)


# ---------------------------------------------------------------------------
# Criterion 6 — bit-identical restart
# ---------------------------------------------------------------------------


def test_restart_is_bit_identical(tmp_path):
    """The criterion, verbatim: 500 steps, checkpoint, 500 more, reload, 500.

    Run on a config with the Zou–He inlet, the **convective** outlet and an
    obstacle, so the outlet's ``prev`` column — the one piece of step-to-step
    state that is not ``f`` — is genuinely under test. If it were not
    recoverable from ``f`` this would fail on the first resumed step.
    """
    solid = channel_with_cylinder()
    sim = Sim(flow_config(), solid)

    sim.run_steps(500)
    save_checkpoint(sim, tmp_path / "half.pkl")
    sim.run_steps(500)
    reference = sim.f.copy()

    resumed = load_checkpoint(tmp_path / "half.pkl")
    assert resumed.step_count == 500
    resumed.run_steps(500)

    assert resumed.step_count == 1000
    assert np.array_equal(resumed.f, reference)


def test_restart_is_bit_identical_with_a_body_force(tmp_path):
    """Same claim for the Guo-forced path (D-010), which Rung 1 uses."""
    solid = channel_walls(22, 32)
    sim = Sim(forced_config(), solid)

    sim.run_steps(500)
    save_checkpoint(sim, tmp_path / "g.pkl")
    sim.run_steps(500)
    reference = sim.f.copy()

    resumed = load_checkpoint(tmp_path / "g.pkl")
    resumed.run_steps(500)

    assert np.array_equal(resumed.f, reference)


def test_restart_from_the_plain_copy_outlet_is_bit_identical(tmp_path):
    """The non-convective outlet keeps no state at all; assert it anyway."""
    sim = Sim(flow_config(convective_outlet=False), channel_with_cylinder())
    sim.run_steps(200)
    save_checkpoint(sim, tmp_path / "c.pkl")
    sim.run_steps(200)
    reference = sim.f.copy()

    resumed = load_checkpoint(tmp_path / "c.pkl")
    resumed.run_steps(200)
    assert np.array_equal(resumed.f, reference)


def test_the_outlet_prev_column_equals_f_at_the_end_of_a_step():
    """Why the checkpoint is still only four things (module docstring).

    Nothing after ``outlet_zero_gradient`` writes the outlet column, so
    ``out_prev`` is byte-identical to ``f[:, :, outlet_col]`` at the end of every
    step and is rebuilt from ``f`` on load.
    """
    sim = Sim(flow_config(), channel_with_cylinder())
    for _ in range(25):
        sim.step()
        assert np.array_equal(sim.out_prev, sim.f[:, :, sim.config.outlet_col])


# ---------------------------------------------------------------------------
# Criterion 7 — auto-checkpoint, off by default
# ---------------------------------------------------------------------------


def test_auto_checkpoint_is_off_by_default(tmp_path):
    cfg = flow_config(checkpoint_path=str(tmp_path / "auto.pkl"))
    assert cfg.checkpoint_every == 0

    sim = Sim(cfg, channel_with_cylinder())
    stats = run(sim, frames=10, steps_per_frame=5)

    assert stats.checkpoints == 0
    assert not (tmp_path / "auto.pkl").exists()


def test_auto_checkpoint_fires_every_n_steps(tmp_path):
    path = tmp_path / "auto.pkl"
    cfg = flow_config(checkpoint_every=20, checkpoint_path=str(path))
    sim = Sim(cfg, channel_with_cylinder())

    stats = run(sim, frames=10, steps_per_frame=5)  # 50 steps -> 20, 40

    assert stats.checkpoints == 2
    assert path.exists()
    assert load_checkpoint(path).step_count == 40


def test_an_auto_checkpoint_resumes_bit_identically(tmp_path):
    path = tmp_path / "auto.pkl"
    cfg = flow_config(checkpoint_every=100, checkpoint_path=str(path))
    sim = Sim(cfg, channel_with_cylinder())
    sim.run_steps(300)  # run_steps does not auto-checkpoint; run() does
    run(sim, frames=1, steps_per_frame=100, buffer_size=2)  # -> step 400 saved

    sim.run_steps(100)
    reference = sim.f.copy()

    resumed = load_checkpoint(path)
    assert resumed.step_count == 400
    resumed.run_steps(100)
    assert np.array_equal(resumed.f, reference)


# ---------------------------------------------------------------------------
# Setup validation (constraints 2, 3, 12) and the derived D (D-019)
# ---------------------------------------------------------------------------


def test_tau_at_or_below_one_half_is_rejected_by_name():
    with pytest.raises(ValueError, match="tau"):
        Sim(flow_config(tau=0.5), channel_with_cylinder())


def test_a_mask_of_the_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        Sim(flow_config(), np.zeros((NY + 1, NX), dtype=bool))


def test_an_inlet_at_or_above_the_mach_ceiling_warns_at_setup():
    """Constraint 3 — warn at setup, not at ``nan`` time."""
    with pytest.warns(UserWarning, match="0.1"):
        Sim(flow_config(inlet_U=0.12), channel_with_cylinder())


def test_characteristic_length_comes_from_the_bounding_box():
    """D-019: ``D`` is the cross-stream extent of the object's bbox, not a guess."""
    solid = channel_with_cylinder()
    sim = Sim(flow_config(), solid)

    y0, y1, _, _ = bounding_box(strip_solid_border(solid))
    assert sim.D == pytest.approx(float(y1 - y0 + 1))
    assert sim.D == pytest.approx(6.0)  # this disc, centred on a half cell


def test_an_explicit_D_overrides_the_derivation():
    sim = Sim(flow_config(D=9.0), channel_with_cylinder())
    assert sim.D == 9.0


def test_geometry_checks_run_at_setup_when_asked():
    """Constraint 12 — the mask sanity checks belong before the run, not after."""
    thin = np.zeros((NY, NX), dtype=bool)
    thin[10, 20] = True  # a one-cell obstacle: too thin, bounce-back leaks
    with pytest.warns(UserWarning):
        Sim(flow_config(check_geometry=True), thin)


# ---------------------------------------------------------------------------
# run() plumbing
# ---------------------------------------------------------------------------


def test_run_needs_a_bound():
    sim = Sim(flow_config(), channel_with_cylinder())
    with pytest.raises(ValueError):
        run(sim)


def test_run_rejects_a_zero_steps_per_frame():
    sim = Sim(flow_config(), channel_with_cylinder())
    with pytest.raises(ValueError):
        run(sim, frames=1, steps_per_frame=0)


def test_run_accepts_a_step_budget_instead_of_frames():
    sim = Sim(flow_config(), channel_with_cylinder())
    stats = run(sim, frames=None, steps=50, steps_per_frame=10)
    assert stats.frames == 5
    assert sim.step_count == 50


def test_run_stops_on_the_predicate():
    sim = Sim(flow_config(), channel_with_cylinder())
    stats = run(sim, frames=1000, steps_per_frame=5, stop=lambda s: s.step_count >= 40)
    assert sim.step_count == 40
    assert stats.frames == 8


def test_the_default_frame_is_a_copy_of_the_vorticity_field():
    """Constraint 9 — vorticity, not speed; and a copy, since the buffer moves on."""
    sim = Sim(flow_config(), channel_with_cylinder())
    frames: list[np.ndarray] = []

    class Collect(Sink):
        def push(self, frame):
            frames.append(frame)

        def close(self):
            pass

    run(sim, Collect(), frames=3, steps_per_frame=5, drop=False)

    assert len(frames) == 3
    assert frames[0].shape == (NY, NX)
    assert all(fr.base is None for fr in frames)  # copies, not views on sim.omega
    assert not np.array_equal(frames[0], frames[-1])  # the field is evolving


def test_vorticity_uses_the_owned_buffer():
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.step()
    out = sim.vorticity()
    assert out is sim.omega
    assert np.isnan(out[sim.solid]).all()  # solid cells are nan (probe.vorticity)


def test_forces_uses_the_two_snapshots_d020_fixed():
    """``Sim.forces`` reads ``f_bb`` (pre-stream) and ``f`` (post-stream)."""
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(50)
    cd, cl = sim.forces()
    assert np.isfinite(cd) and np.isfinite(cl)
    assert cd > 0.0  # the flow pushes the cylinder downstream


def test_residual_falls_as_the_flow_settles():
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(200)
    sim.mark_residual()
    sim.run_steps(100)
    early = sim.residual()

    sim.run_steps(1000)
    sim.mark_residual()
    sim.run_steps(100)
    late = sim.residual()

    assert late < early


# ---------------------------------------------------------------------------
# The runner runs the same physics the passing rung runs
# ---------------------------------------------------------------------------


def test_the_runner_reproduces_rung_1s_hand_rolled_loop():
    """``Sim.step`` is the order ``validate/poiseuille.py`` already passes with.

    A test that pins the loop body to the one a green rung uses: if T006 ever
    reorders the timestep, this fails before Rung 1 does.
    """
    ny, nx, tau, gx, steps = 22, 32, 0.6, 2.6667e-5, 400
    solid = channel_walls(ny, nx)
    g = (gx, 0.0)

    # --- the hand-rolled loop, copied from validate/poiseuille.py ---
    f = np.empty((Q, ny, nx), dtype=np.float32)
    f_pre = np.empty_like(f)
    feq = np.empty_like(f)
    buf = np.empty_like(f)
    rho = np.empty((ny, nx), dtype=np.float32)
    u = np.empty((2, ny, nx), dtype=np.float32)
    work = np.empty((3, ny, nx), dtype=np.float32)
    f[:] = W[:, None, None]

    for _ in range(steps):
        np.copyto(f_pre, f)
        macroscopic(f, rho, u)
        force_velocity_shift(rho, u, g, work)
        equilibrium(rho, u, feq, work)
        collide(f, feq, tau)
        apply_body_force(f, rho, u, tau, g, work)
        bounce_back(f, f_pre, solid)
        stream(f, buf)

    # --- the runner ---
    sim = Sim(SimConfig(ny=ny, nx=nx, tau=tau, g=g, check_geometry=False), solid)
    sim.run_steps(steps)

    assert np.array_equal(sim.f, f)


def test_a_fresh_sim_starts_at_the_rest_equilibrium():
    """``f = w_i rho`` with ``u = 0`` — the same start Rung 1 uses."""
    sim = Sim(forced_config(), channel_walls(22, 32))
    expected = np.broadcast_to(W[:, None, None], (Q, 22, 32))
    assert np.array_equal(sim.f, expected.astype(np.float32))
