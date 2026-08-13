"""T010's guard rails: the optimised path must be the same physics, exactly.

``DOCS/TASKS1.md`` § T010 § Constraints that bite here:

* **Constraint 1** — "optimisation must not quietly change the physics (e.g.
  skipping ``feq`` on solid cells must not change fluid-cell results at all:
  assert bitwise equality against the unoptimised path on a small grid)".
* **Constraint 11** — "fusing must not change float ordering in a way that
  breaks bit-identical restart. If it does, the fusion is reverted, not the
  test."
* **Constraint 4** — ``float32`` end to end. A ``float64`` temporary anywhere in
  the step path loses the speed *and* the bit-identical restart, so this file
  records the dtype of every ufunc result in a real timestep and asserts they
  are all ``float32``.

The reference path is ``SimConfig(fused=False)`` — the T009 solver, kept
selectable for exactly this comparison.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from lbm.boundary import bounce_back, inlet_velocity
from lbm.core import Q, collide, collide_stream, equilibrium, macroscopic, stream
from lbm.geometry import circle
from lbm.runner import Sim, SimConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def small_case(**overrides) -> tuple[SimConfig, np.ndarray]:
    """A small open-channel case with an immersed disc and both open boundaries.

    Small on purpose: bitwise equality is a statement about arithmetic, and a
    40x24 grid exercises every branch of the step in a fraction of a second.
    """
    ny, nx = 24, 40
    solid = circle(ny, nx, nx / 3.0, ny / 2.0, 3.0)
    solid[0, :] = True
    solid[-1, :] = True

    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=0.6,
        inlet_U=0.05,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        check_geometry=False,
        **overrides,
    )
    return cfg, solid


def run(cfg: SimConfig, solid: np.ndarray, steps: int = 40) -> Sim:
    sim = Sim(cfg, solid)
    sim.run_steps(steps)
    return sim


# ---------------------------------------------------------------------------
# constraint 1 — the fused kernel is the same physics, bitwise
# ---------------------------------------------------------------------------


def test_the_fused_step_is_bitwise_equal_to_the_unfused_one():
    cfg, solid = small_case()
    ref = run(cfg.replace(fused=False), solid)
    fast = run(cfg.replace(fused=True), solid)

    assert np.array_equal(ref.f, fast.f)
    assert ref.f.dtype == fast.f.dtype == np.float32


def test_fluid_cells_specifically_are_bitwise_equal():
    """The criterion is worded about *fluid* cells; assert them on their own.

    Solid cells are overwritten wholesale by the reflection, so an equality that
    held only because both paths wrote the same reflected values there would
    prove nothing about the flow.
    """
    cfg, solid = small_case()
    ref = run(cfg.replace(fused=False), solid)
    fast = run(cfg.replace(fused=True), solid)

    fluid = ~solid
    assert np.array_equal(ref.f[:, fluid], fast.f[:, fluid])
    assert np.array_equal(ref.u[:, fluid], fast.u[:, fluid])
    assert np.array_equal(ref.rho[fluid], fast.rho[fluid])


def test_the_two_paths_agree_on_the_force_snapshots():
    """``probe.forces`` reads ``f_bb`` (pre-stream) and ``f`` (post-stream).

    D-020. If the fusion dropped or mistimed the ``f_bb`` snapshot, Rung 3 and
    Rung 4 would measure nothing — and would do it quietly.
    """
    cfg, solid = small_case()
    ref = run(cfg.replace(fused=False), solid)
    fast = run(cfg.replace(fused=True), solid)

    assert np.array_equal(ref.f_bb, fast.f_bb)
    assert ref.forces() == fast.forces()


def test_the_two_paths_agree_with_an_empty_domain():
    """No solid cells at all: the fused path skips the reflection entirely."""
    cfg = SimConfig(ny=16, nx=24, tau=0.7, inlet_U=0.04, use_inlet=True, use_outlet=True)
    empty = np.zeros((16, 24), dtype=bool)
    ref = run(cfg.replace(fused=False), empty, steps=25)
    fast = run(cfg.replace(fused=True), empty, steps=25)
    assert np.array_equal(ref.f, fast.f)


def test_a_forced_run_keeps_the_unfused_sequence():
    """Guo's source term sits between collision and bounce-back (D-010).

    It is deliberately not folded into the fused kernel, so a forced config runs
    the T009 sequence whatever ``fused`` says — and the two must agree because
    they are the same code.
    """
    cfg, solid = small_case(g=(1e-5, 0.0))
    ref = run(cfg.replace(fused=False), solid)
    fast = run(cfg.replace(fused=True), solid)

    assert fast._fused is False
    assert np.array_equal(ref.f, fast.f)


def test_collide_stream_equals_collide_then_bounce_back_then_stream():
    """The same claim one level down, on the function rather than the runner."""
    rng = np.random.default_rng(7)
    ny, nx = 12, 20
    f0 = (0.1 + rng.random((Q, ny, nx))).astype(np.float32)
    feq = (0.1 + rng.random((Q, ny, nx))).astype(np.float32)
    solid = circle(ny, nx, nx / 2.0, ny / 2.0, 2.5)
    tau = 0.62

    # reference: collide -> bounce_back -> snapshot -> stream
    ref = f0.copy()
    f_pre = f0.copy()
    collide(ref, feq, tau)
    bounce_back(ref, f_pre, solid)
    ref_bb = ref.copy()
    stream(ref, np.empty_like(ref))

    # fused
    fast = f0.copy()
    fast_bb = np.empty_like(fast)
    collide_stream(
        fast,
        feq,
        tau,
        np.empty_like(fast),
        f_pre=f_pre,
        solid=solid,
        f_bb=fast_bb,
    )

    assert np.array_equal(ref, fast)
    assert np.array_equal(ref_bb, fast_bb)


def test_collide_stream_without_f_bb_stages_in_f():
    """A caller that never measures forces need not own the extra buffer."""
    rng = np.random.default_rng(11)
    f0 = (0.1 + rng.random((Q, 10, 14))).astype(np.float32)
    feq = (0.1 + rng.random((Q, 10, 14))).astype(np.float32)

    with_bb = f0.copy()
    collide_stream(with_bb, feq, 0.6, np.empty_like(f0), f_bb=np.empty_like(f0))
    without_bb = f0.copy()
    collide_stream(without_bb, feq, 0.6, np.empty_like(f0))

    assert np.array_equal(with_bb, without_bb)


def test_collide_stream_rejects_tau_at_the_floor_and_a_missing_f_pre():
    f = np.ones((Q, 8, 8), dtype=np.float32)
    feq = np.ones_like(f)
    with pytest.raises(ValueError, match="tau"):
        collide_stream(f, feq, 0.5, np.empty_like(f))
    with pytest.raises(ValueError, match="f_pre"):
        collide_stream(f, feq, 0.6, np.empty_like(f), solid=np.zeros((8, 8), bool))


def test_f_keeps_its_buffer_identity_through_the_fused_kernel():
    """T006's allocation test asserts this for the step; pin it on the kernel."""
    f = np.ones((Q, 8, 8), dtype=np.float32)
    before = f.__array_interface__["data"]
    collide_stream(f, np.ones_like(f), 0.6, np.empty_like(f))
    assert f.__array_interface__["data"] == before


# ---------------------------------------------------------------------------
# constraint 11 — restart across the fused kernel
# ---------------------------------------------------------------------------


def test_restart_is_bit_identical_with_the_fused_kernel(tmp_path):
    """T006's claim, re-run against the optimised path (constraint 11)."""
    cfg, solid = small_case()

    a = Sim(cfg, solid)
    a.run_steps(60)
    path = a.save_checkpoint(tmp_path / "ckpt.pkl")
    a.run_steps(60)
    expected = a.f.copy()

    from lbm.runner import load_checkpoint

    b = load_checkpoint(path)
    b.run_steps(60)

    assert np.array_equal(expected, b.f)


def test_a_run_fused_and_a_run_unfused_resume_each_other(tmp_path):
    """The toggle is a speed switch, so a checkpoint must cross it unchanged."""
    cfg, solid = small_case()

    a = Sim(cfg.replace(fused=True), solid)
    a.run_steps(50)
    path = a.save_checkpoint(tmp_path / "ckpt.pkl")
    a.run_steps(50)

    from lbm.runner import load_checkpoint

    b = load_checkpoint(path)
    b.config.fused = False
    b._fused = False
    b.run_steps(50)

    assert np.array_equal(a.f, b.f)


# ---------------------------------------------------------------------------
# constraint 4 — float32 end to end
# ---------------------------------------------------------------------------


class DtypeSpy(np.ndarray):
    """An ndarray that records the dtype of every ufunc result it takes part in.

    The float32 audit ``DOCS/TASKS1.md`` § T010 asks for, done by measurement
    rather than by reading the source: a single ``float64`` temporary anywhere
    in the step path costs the bandwidth the dtype exists to save and, worse,
    rounds differently on a resumed run (constraint 11).
    """

    seen: list[tuple[str, str]]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raw_i = [np.asarray(x).view(np.ndarray) if isinstance(x, DtypeSpy) else x for x in inputs]
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(
                np.asarray(o).view(np.ndarray) if isinstance(o, DtypeSpy) else o for o in out
            )
        result = getattr(ufunc, method)(*raw_i, **kwargs)
        for r in result if isinstance(result, tuple) else (result,):
            if isinstance(r, np.ndarray):
                DtypeSpy.seen.append((ufunc.__name__, str(r.dtype)))
            elif isinstance(r, np.generic):
                DtypeSpy.seen.append((ufunc.__name__, str(np.dtype(type(r)))))
        return result


def _spy_all_buffers(sim: Sim) -> None:
    for name in (
        "f", "f_pre", "f_bb", "buf", "rho", "u", "feq", "work",
        "out_prev", "inlet_work", "u_in",
    ):
        arr = getattr(sim, name)
        setattr(sim, name, arr.view(DtypeSpy))


@pytest.mark.parametrize("fused", [True, False])
def test_every_ufunc_in_a_timestep_produces_float32(fused):
    cfg, solid = small_case(fused=fused)
    sim = Sim(cfg, solid)
    sim.run_steps(3)

    DtypeSpy.seen = []
    _spy_all_buffers(sim)
    sim.step()

    assert DtypeSpy.seen, "the spy recorded nothing — it is not wired to the step"
    upcast = {(name, dt) for name, dt in DtypeSpy.seen if dt != "float32"}
    assert not upcast, f"non-float32 intermediates in the step path: {sorted(upcast)}"


def test_every_state_and_buffer_array_is_float32():
    cfg, solid = small_case()
    sim = Sim(cfg, solid)
    sim.run_steps(5)

    for name in (
        "f", "f_pre", "f_bb", "buf", "rho", "u", "u_prev", "feq", "work",
        "omega", "vort_work", "res_work", "out_prev", "inlet_work", "u_in",
    ):
        assert getattr(sim, name).dtype == np.float32, name

    assert sim.vorticity().dtype == np.float32
    assert sim.solid.dtype == np.bool_


def test_the_hot_functions_return_float32_from_float32():
    rng = np.random.default_rng(3)
    f = (0.1 + rng.random((Q, 9, 11))).astype(np.float32)
    rho, u = macroscopic(f)
    assert rho.dtype == np.float32 and u.dtype == np.float32
    assert equilibrium(rho, u).dtype == np.float32


# ---------------------------------------------------------------------------
# the preallocation audit
# ---------------------------------------------------------------------------


def test_inlet_velocity_allocates_nothing_when_given_its_fluid_mask():
    """Session 6's note: ``~solid[:, col]`` was the last allocation in the loop.

    It is transient and freed every step, so a heap-*growth* test cannot see it;
    what this measures is the peak traffic of 200 calls with and without the
    precomputed mask.
    """
    ny, nx = 64, 96
    f = np.full((Q, ny, nx), 0.1, dtype=np.float32)
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    from lbm.boundary import inlet_profile

    u_in = inlet_profile(ny, 0.05, "uniform", solid=solid, col=0)
    work = np.empty((5, ny), dtype=np.float32)
    fluid = ~solid[:, 0]

    def traffic(**kw):
        inlet_velocity(f, solid=solid, col=0, u_in=u_in, work=work, **kw)  # warm
        tracemalloc.start()
        for _ in range(200):
            inlet_velocity(f, solid=solid, col=0, u_in=u_in, work=work, **kw)
        total = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        return total

    without = traffic()
    with_mask = traffic(fluid=fluid)
    assert with_mask <= without


def test_a_precomputed_fluid_mask_changes_nothing():
    ny, nx = 20, 30
    rng = np.random.default_rng(5)
    base = (0.1 + rng.random((Q, ny, nx))).astype(np.float32)
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    from lbm.boundary import inlet_profile

    u_in = inlet_profile(ny, 0.05, "uniform", solid=solid, col=0)
    work = np.empty((5, ny), dtype=np.float32)

    a = base.copy()
    inlet_velocity(a, solid=solid, col=0, u_in=u_in, work=work)
    b = base.copy()
    inlet_velocity(b, solid=solid, col=0, u_in=u_in, work=work, fluid=~solid[:, 0])

    assert np.array_equal(a, b)


def test_the_step_still_allocates_nothing_after_the_fusion():
    """T006's criterion, re-asserted on the optimised path."""
    cfg, solid = small_case()
    sim = Sim(cfg, solid)
    sim.run_steps(20)
    before = sim.f.__array_interface__["data"]

    tracemalloc.start()
    start = tracemalloc.get_traced_memory()[0]
    sim.run_steps(500)
    growth = tracemalloc.get_traced_memory()[0] - start
    tracemalloc.stop()

    assert sim.f.__array_interface__["data"] == before
    one_buffer = sim.f.nbytes
    assert growth < one_buffer, f"heap grew {growth} bytes over 500 steps"
