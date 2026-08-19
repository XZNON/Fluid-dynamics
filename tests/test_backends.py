"""Tests for :mod:`lbm.backends` — T101, the backend seam.

The acceptance criteria of ``DOCS/TASKS2.md`` § T101, one test apiece: a
protocol with documented shapes, a NumPy implementation that **delegates**
rather than reimplements, a ``Sim`` that reaches every kernel through
``self.backend``, an import-level assertion that no kernel is imported into
:mod:`lbm.runner` any more, a bit-exact host round trip, a useful error for a
name nothing answers to, and a restart that is still bit-identical through the
seam (``CLAUDE.md`` constraint 11 in its ``DOCS/STATE2.md`` **D-046** form).

The rungs are the other half of this task's evidence and they are not here:
"every Phase 0 number comes back identical" is checked by running
``validate/``, not by a unit test.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import typing

import numpy as np
import pytest

import lbm.boundary as boundary
import lbm.core as core
import lbm.runner as runner
from lbm.backends import (
    Backend,
    BackendUnavailableError,
    available_backends,
    get_backend,
)
from lbm.backends.numpy_backend import NumpyBackend
from lbm.backends.registry import known_backends
from lbm.geometry import channel_walls, circle
from lbm.runner import Q, Sim, SimConfig, load_checkpoint, save_checkpoint

NY, NX = 24, 48

#: Every method the T101 contract requires of a backend.
PROTOCOL_METHODS = (
    "equilibrium",
    "collide",
    "stream",
    "collide_stream",
    "macroscopic",
    "bounce_back",
    "to_host",
    "from_host",
)

#: What T103 added when whole-step parity forced the seam wider (**D-051**,
#: **D-052** superseded): allocation, the general transfers, the remaining
#: boundaries and both halves of the Guo body force.
T103_METHODS = (
    "empty",
    "zeros",
    "copy",
    "upload",
    "download",
    "moving_wall",
    "inlet_velocity",
    "outlet_zero_gradient",
    "force_velocity_shift",
    "apply_body_force",
)

#: The kernels :mod:`lbm.runner` used to import directly and must not any more.
KERNEL_NAMES = frozenset(
    {"collide", "collide_stream", "equilibrium", "macroscopic", "stream", "bounce_back"}
)


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


def random_state(seed: int = 0) -> np.ndarray:
    """A plausible ``(9, ny, nx)`` ``float32`` distribution."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 0.2, size=(Q, NY, NX)).astype(np.float32)


# ---------------------------------------------------------------------------
# Criterion 1 — the protocol, with documented shapes, and a delegating backend
# ---------------------------------------------------------------------------


def test_the_protocol_declares_every_method_the_contract_names():
    """``DOCS/TASKS2.md`` § T101: "the protocol covers, at minimum ..."."""
    members = set(typing.get_type_hints(Backend).keys()) | {
        name for name in vars(Backend) if not name.startswith("_")
    }
    missing = [m for m in PROTOCOL_METHODS if m not in members]
    assert not missing, f"Backend is missing {missing}"
    assert "name" in members


def test_backend_is_a_protocol_and_the_numpy_backend_satisfies_it():
    assert issubclass(Backend, typing.Protocol)  # type: ignore[arg-type]
    assert isinstance(NumpyBackend(), Backend)


def test_every_protocol_method_documents_its_array_shapes():
    """A seam whose shapes live in a comment is a seam nobody can port to."""
    for name in PROTOCOL_METHODS:
        doc = inspect.getdoc(getattr(Backend, name)) or ""
        assert doc, f"Backend.{name} has no docstring"
        assert "(9, ny, nx)" in doc, f"Backend.{name} does not document its shapes"


def test_the_numpy_backend_delegates_to_lbm_core_bit_for_bit():
    """Delegation, not translation: the seam must not move a single bit.

    Constraint 11 in its D-046 form is bit-identical restart *within* a
    backend, and every Phase 0 rung printing its session-11 digits depends on
    the arithmetic being the same operations in the same order. Calling the
    backend and calling :mod:`lbm.core` are compared directly rather than
    trusted.
    """
    backend = get_backend("numpy")
    f_a, f_b = random_state(1), random_state(1)
    solid = channel_with_cylinder()

    rho_a, u_a = backend.macroscopic(f_a)
    rho_b, u_b = core.macroscopic(f_b)
    assert np.array_equal(rho_a, rho_b) and np.array_equal(u_a, u_b)

    feq_a = backend.equilibrium(rho_a, u_a)
    feq_b = core.equilibrium(rho_b, u_b)
    assert np.array_equal(feq_a, feq_b)

    backend.collide(f_a, feq_a, 0.6)
    core.collide(f_b, feq_b, 0.6)
    assert np.array_equal(f_a, f_b)

    pre_a, pre_b = random_state(2), random_state(2)
    backend.bounce_back(f_a, pre_a, solid)
    boundary.bounce_back(f_b, pre_b, solid)
    assert np.array_equal(f_a, f_b)

    backend.stream(f_a, np.empty_like(f_a))
    core.stream(f_b, np.empty_like(f_b))
    assert np.array_equal(f_a, f_b)


def test_the_fused_path_through_the_seam_is_bitwise_equal_to_the_unfused_one():
    """**D-033** survives the seam: both paths stay selectable and agree."""
    backend = get_backend("numpy")
    solid = channel_with_cylinder()
    tau = 0.6

    f_fused, f_plain = random_state(3), random_state(3)
    pre_fused, pre_plain = f_fused.copy(), f_plain.copy()
    rho, u = backend.macroscopic(f_fused.copy())
    feq = backend.equilibrium(rho, u)

    bb_fused = np.empty_like(f_fused)
    backend.collide_stream(
        f_fused, feq, tau, np.empty_like(f_fused), f_pre=pre_fused, solid=solid,
        f_bb=bb_fused,
    )

    backend.collide(f_plain, feq, tau)
    backend.bounce_back(f_plain, pre_plain, solid)
    bb_plain = f_plain.copy()
    backend.stream(f_plain, np.empty_like(f_plain))

    assert np.array_equal(f_fused, f_plain)
    assert np.array_equal(bb_fused, bb_plain)


def test_no_backend_redefines_the_nine_constants():
    """Constraint 4: ``E``, ``W``, ``OPP``, ``CS2`` come from ``lbm.core`` only."""
    root = pathlib.Path(runner.__file__).parent / "backends"
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        clash = assigned & {"E", "W", "OPP", "CS2", "E_F32"}
        assert not clash, f"{path.name} redefines {clash} (CLAUDE.md constraint 4)"


# ---------------------------------------------------------------------------
# Criterion 2 — the default is numpy, and the runner imports no kernel
# ---------------------------------------------------------------------------


def test_numpy_is_the_default_backend_and_sim_holds_one():
    cfg = SimConfig(ny=NY, nx=NX, tau=0.6)
    assert cfg.backend == "numpy"

    sim = Sim(cfg)
    assert isinstance(sim.backend, Backend)
    assert sim.backend.name == "numpy"


def test_the_runner_module_imports_no_kernel_from_lbm_core_or_lbm_boundary():
    """The import-level assertion the contract asks for, not a comment.

    Read from the source, so that a kernel imported inside a function is caught
    as well as one imported at module level. ``lbm.core`` is still imported for
    its constants — that is required, not merely allowed (constraint 4).
    """
    tree = ast.parse(pathlib.Path(runner.__file__).read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "lbm.core",
            "lbm.boundary",
        }:
            for alias in node.names:
                if alias.name in KERNEL_NAMES:
                    offenders.append(f"{node.module}.{alias.name}")
    assert not offenders, f"lbm/runner.py still imports kernels directly: {offenders}"

    assert not KERNEL_NAMES & set(vars(runner)), "a kernel is bound in lbm.runner"
    assert runner.Q is core.Q and runner.W is core.W


def test_sim_reaches_every_kernel_through_the_backend():
    """Replace the backend's kernels with counters; a step must hit them all."""
    sim = Sim(flow_config(), channel_with_cylinder())
    calls: dict[str, int] = {}

    class CountingBackend:
        name = "counting"

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            attr = getattr(self._inner, item)
            if item not in PROTOCOL_METHODS:
                return attr

            def wrapper(*args, **kwargs):
                calls[item] = calls.get(item, 0) + 1
                return attr(*args, **kwargs)

            return wrapper

    sim.backend = CountingBackend(sim.backend)  # type: ignore[assignment]
    sim.step()
    assert calls.get("macroscopic") == 1
    assert calls.get("equilibrium") == 1
    assert calls.get("collide_stream") == 1

    unfused = Sim(flow_config(fused=False), channel_with_cylinder())
    calls.clear()
    unfused.backend = CountingBackend(unfused.backend)  # type: ignore[assignment]
    unfused.step()
    assert calls.get("collide") == 1
    assert calls.get("bounce_back") == 1
    assert calls.get("stream") == 1


def test_the_seam_does_not_reintroduce_an_allocation_into_the_step():
    """Constraint "preallocate, never allocate inside the step loop"."""
    import tracemalloc

    sim = Sim(flow_config(), channel_with_cylinder())
    for _ in range(3):
        sim.step()

    f_before = sim.f
    tracemalloc.start()
    start = tracemalloc.take_snapshot()
    for _ in range(10):
        sim.step()
    grew = sum(
        stat.size_diff
        for stat in tracemalloc.take_snapshot().compare_to(start, "filename")
    )
    tracemalloc.stop()

    assert sim.f is f_before
    assert grew < 64 * 1024, f"the step grew the heap by {grew} bytes"


# ---------------------------------------------------------------------------
# Criterion 3 — to_host / from_host round-trip bit-identically
# ---------------------------------------------------------------------------


def test_to_host_from_host_round_trips_bit_identically():
    """The portability contract, and the hook Rung A / **Q-103** hang off."""
    backend = get_backend("numpy")
    original = random_state(4)
    reference = original.copy()

    round_tripped = backend.to_host(backend.from_host(original))

    assert round_tripped.shape == (Q, NY, NX)
    assert round_tripped.dtype == np.float32
    assert np.array_equal(round_tripped, reference)


def test_to_host_of_a_running_sim_is_the_host_layout():
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(5)
    host = sim.backend.to_host(sim.f)
    assert host.shape == (Q, NY, NX)
    assert host.dtype == np.float32
    assert np.array_equal(host, sim.f)


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((8, NY, NX), dtype=np.float32),
        np.empty((NY, NX), dtype=np.float32),
        np.empty((Q, NY, NX), dtype=np.float64),
    ],
)
def test_the_host_contract_rejects_a_wrong_shape_or_dtype(bad):
    """A layout mistake should fail at the seam, not three rungs later."""
    with pytest.raises(ValueError):
        get_backend("numpy").from_host(bad)


# ---------------------------------------------------------------------------
# Criterion 4 — an unknown backend name is a useful error
# ---------------------------------------------------------------------------


def test_an_unknown_backend_name_raises_naming_it_and_the_available_ones():
    with pytest.raises(ValueError) as excinfo:
        get_backend("cuda-quantum")

    message = str(excinfo.value)
    assert "cuda-quantum" in message
    for name in available_backends():
        assert name in message


def test_a_known_but_uninstalled_backend_names_its_install_line():
    """``warp`` is known before T102 writes it; the error must say how to get it."""
    assert "warp" in known_backends()
    if "warp" in available_backends():  # pragma: no cover - true from T102 on
        pytest.skip("warp is installed; there is no unavailable backend to check")

    with pytest.raises(BackendUnavailableError) as excinfo:
        get_backend("warp")

    message = str(excinfo.value)
    assert "warp" in message
    assert "pip" in message and "warp-lang" in message
    assert isinstance(excinfo.value, ValueError)


def test_numpy_is_always_available():
    assert "numpy" in available_backends()


def test_sim_rejects_an_unknown_backend_at_construction():
    with pytest.raises(ValueError, match="nonesuch"):
        Sim(SimConfig(ny=NY, nx=NX, tau=0.6, backend="nonesuch"))


# ---------------------------------------------------------------------------
# Criterion 5 — restart is still bit-identical, through the seam
# ---------------------------------------------------------------------------


def test_restart_through_the_seam_is_bit_identical(tmp_path):
    """T006's criterion re-run against a config that names its backend.

    500 steps, checkpoint, 500 more; reload and run 500. The Zou–He inlet, the
    convective outlet and an obstacle are all live, so ``out_prev`` — the one
    piece of step-to-step state that is not ``f`` (**D-022**) — is genuinely
    under test through the new load path.
    """
    solid = channel_with_cylinder()
    sim = Sim(flow_config(backend="numpy"), solid)

    sim.run_steps(500)
    save_checkpoint(sim, tmp_path / "seam.pkl")
    sim.run_steps(500)
    reference = sim.f.copy()

    resumed = load_checkpoint(tmp_path / "seam.pkl")
    assert resumed.step_count == 500
    assert resumed.backend.name == "numpy"
    resumed.run_steps(500)

    assert resumed.step_count == 1000
    assert np.array_equal(resumed.f, reference)


def test_a_checkpoint_carries_its_backend_name_and_load_can_override_it(tmp_path):
    """**D-050**: the name rides inside the config, and ``f`` stays portable."""
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(10)
    path = save_checkpoint(sim, tmp_path / "named.pkl")

    import pickle

    state = pickle.loads(path.read_bytes())
    # D-022 unchanged: exactly f, solid, step_count, config and format.
    assert set(state) == {"format", "f", "solid", "step_count", "config"}
    assert state["config"].backend == "numpy"
    assert state["f"].shape == (Q, NY, NX)
    assert state["f"].dtype == np.float32

    resumed = load_checkpoint(path, backend="numpy")
    assert resumed.backend.name == "numpy"
    assert np.array_equal(resumed.f, sim.f)


def test_a_checkpoint_written_before_t101_still_loads(tmp_path):
    """An old config has no ``backend`` field; the dataclass default supplies it."""
    import pickle

    cfg = flow_config()
    payload = cfg.__dict__.copy()
    payload.pop("backend")

    old_cfg = SimConfig.__new__(SimConfig)
    old_cfg.__dict__.update(payload)
    assert "backend" not in old_cfg.__dict__
    assert old_cfg.backend == "numpy"

    sim = Sim(flow_config(), channel_with_cylinder())
    sim.run_steps(5)
    path = tmp_path / "old.pkl"
    path.write_bytes(
        pickle.dumps(
            {
                "format": 1,
                "f": sim.f,
                "solid": sim.solid,
                "step_count": sim.step_count,
                "config": old_cfg,
            }
        )
    )

    resumed = load_checkpoint(path)
    assert resumed.backend.name == "numpy"
    assert np.array_equal(resumed.f, sim.f)


# ---------------------------------------------------------------------------
# T103 — the widened seam: allocation, transfer, and the rest of the boundaries
# ---------------------------------------------------------------------------


def test_the_protocol_declares_everything_t103_added():
    """``DOCS/TASKS2.md`` § T103: the whole timestep goes through the seam."""
    members = set(typing.get_type_hints(Backend).keys()) | {
        name for name in vars(Backend) if not name.startswith("_")
    }
    missing = [m for m in T103_METHODS if m not in members]
    assert not missing, f"Backend is missing {missing}"


def test_every_t103_method_has_a_docstring_naming_its_shapes():
    """Same rule as T101's: a seam whose shapes live in a comment is unportable."""
    for name in T103_METHODS:
        doc = inspect.getdoc(getattr(Backend, name)) or ""
        assert doc, f"Backend.{name} has no docstring"
        assert "(9, ny, nx)" in doc or "(ny, nx)" in doc, (
            f"Backend.{name} does not document its shapes"
        )


def test_the_numpy_backend_delegates_the_boundaries_bit_for_bit():
    """Delegation, not translation — the same rule T101 set for the kernels.

    Every Phase 0 rung printing its session-11 digits depends on the arithmetic
    being the same operations in the same order, so the seam is compared against
    :mod:`lbm.boundary` directly rather than trusted.
    """
    be = NumpyBackend()
    rng = np.random.default_rng(7)
    f = rng.uniform(0.0, 0.2, size=(Q, NY, NX)).astype(np.float32)
    f_pre = rng.uniform(0.0, 0.2, size=(Q, NY, NX)).astype(np.float32)
    rho = rng.uniform(0.9, 1.1, size=(NY, NX)).astype(np.float32)
    u = rng.uniform(-0.05, 0.05, size=(2, NY, NX)).astype(np.float32)
    wall = np.zeros((NY, NX), dtype=bool)
    wall[-1, :] = True
    u_in = boundary.inlet_profile(NY, 0.05, "uniform")
    fluid = np.ones(NY, dtype=bool)

    a, b = f.copy(), f.copy()
    be.moving_wall(a, f_pre, wall, (0.05, 0.0))
    boundary.moving_wall(b, f_pre, wall, (0.05, 0.0))
    assert np.array_equal(a, b)

    a, b = f.copy(), f.copy()
    be.inlet_velocity(a, col=0, u_in=u_in, fluid=fluid)
    boundary.inlet_velocity(b, col=0, u_in=u_in, fluid=fluid)
    assert np.array_equal(a, b)

    a, b = f.copy(), f.copy()
    pa = np.ascontiguousarray(f[:, :, -1])
    pb = pa.copy()
    be.outlet_zero_gradient(a, prev=pa)
    boundary.outlet_zero_gradient(b, prev=pb)
    assert np.array_equal(a, b) and np.array_equal(pa, pb)

    a, b = u.copy(), u.copy()
    be.force_velocity_shift(rho, a, (1e-5, 0.0))
    boundary.force_velocity_shift(rho, b, (1e-5, 0.0))
    assert np.array_equal(a, b)

    a, b = f.copy(), f.copy()
    be.apply_body_force(a, rho, u, 0.6, (1e-5, 0.0))
    boundary.apply_body_force(b, rho, u, 0.6, (1e-5, 0.0))
    assert np.array_equal(a, b)


def test_numpy_allocation_and_transfer_are_the_identity_they_claim_to_be():
    """The NumPy backend's "device" is the host, so nothing may move or convert."""
    be = NumpyBackend()

    a = be.empty((Q, NY, NX))
    assert isinstance(a, np.ndarray) and a.shape == (Q, NY, NX)
    assert a.dtype == np.float32
    assert not be.zeros((3, 4)).any()

    host = random_state()
    dev = be.upload(host)
    assert dev is not host  # a backend array is always distinct from the caller's
    assert np.array_equal(dev, host)
    assert be.download(dev) is dev  # ... but reading it back moves no bits

    out = np.empty_like(host)
    assert be.download(dev, out) is out
    assert np.array_equal(out, host)

    mask = np.zeros((NY, NX), dtype=bool)
    mask[2, 3] = True
    assert be.download(be.upload(mask)).dtype == np.bool_
    assert np.array_equal(be.download(be.upload(mask)), mask)


def test_the_fused_path_needs_no_pre_collision_copy_and_says_so_in_bits():
    """The removal :meth:`lbm.runner.Sim.step` makes, checked rather than argued.

    With ``f_bb`` supplied, :func:`lbm.core.collide_stream` stages every
    direction there and never writes ``f`` until the stream lands — so ``f`` is
    still the pre-collision state when the reflection reads it, and passing it
    where **D-011**'s copy would go is bitwise identical. That is what lets the
    timestep skip a whole ``(9, ny, nx)`` copy per step on **both** backends.
    """
    be = NumpyBackend()
    rng = np.random.default_rng(11)
    f = rng.uniform(0.0, 0.2, size=(Q, NY, NX)).astype(np.float32)
    feq = rng.uniform(0.0, 0.2, size=(Q, NY, NX)).astype(np.float32)
    solid = channel_with_cylinder()

    with_copy = f.copy()
    bb_a = np.empty_like(f)
    be.collide_stream(
        with_copy, feq, 0.6, np.empty_like(f),
        f_pre=f.copy(), solid=solid, f_bb=bb_a,
    )

    aliased = f.copy()
    bb_b = np.empty_like(f)
    be.collide_stream(
        aliased, feq, 0.6, np.empty_like(f),
        f_pre=aliased, solid=solid, f_bb=bb_b,
    )

    assert np.array_equal(with_copy, aliased)
    assert np.array_equal(bb_a, bb_b)


def test_sim_reaches_the_boundaries_through_the_backend_too():
    """T101 proved it for the six kernels; T103 extends it to the boundaries.

    A counting backend that wraps the real one records what ``Sim.step`` asks
    for. The open boundaries used to be called as free functions in
    :mod:`lbm.runner`; if they ever are again, this stops counting them.
    """

    class Counting:
        name = "counting"

        def __init__(self):
            self.inner = NumpyBackend()
            self.calls: dict[str, int] = {}

        def __getattr__(self, item):
            attr = getattr(self.inner, item)
            if not callable(attr):
                return attr

            def wrapped(*a, **k):
                self.calls[item] = self.calls.get(item, 0) + 1
                return attr(*a, **k)

            return wrapped

    counting = Counting()
    sim = Sim(flow_config(), channel_with_cylinder())
    sim.backend = counting
    sim.run_steps(3)

    for name in ("macroscopic", "equilibrium", "collide_stream",
                 "outlet_zero_gradient", "inlet_velocity"):
        assert counting.calls.get(name, 0) == 3, (name, counting.calls)


def test_a_forced_sim_reaches_both_halves_of_the_guo_scheme_through_the_backend():
    """**D-010**: the two halves go together or not at all, and both are kernels."""

    class Counting:
        name = "counting"

        def __init__(self):
            self.inner = NumpyBackend()
            self.calls: dict[str, int] = {}

        def __getattr__(self, item):
            attr = getattr(self.inner, item)
            if not callable(attr):
                return attr

            def wrapped(*a, **k):
                self.calls[item] = self.calls.get(item, 0) + 1
                return attr(*a, **k)

            return wrapped

    counting = Counting()
    cfg = SimConfig(ny=NY, nx=NX, tau=0.6, g=(1e-5, 0.0), check_geometry=False)
    sim = Sim(cfg, channel_walls(NY, NX))
    sim.backend = counting
    sim.run_steps(3)

    assert counting.calls.get("force_velocity_shift", 0) == 3
    assert counting.calls.get("apply_body_force", 0) == 3
    assert counting.calls.get("bounce_back", 0) == 3
    # A body force takes the unfused path (D-033), so the fusion is never used.
    assert "collide_stream" not in counting.calls


def test_load_f_writes_through_the_seam_and_reseeds_the_outlet_column():
    """The write half of the host accessors, used by ``validate/polygons.py``."""
    cfg = flow_config()
    sim = Sim(cfg, channel_with_cylinder())
    sim.run_steps(2)

    seed = sim.host_f().copy()
    seed[:] = 0.25
    sim.load_f(seed)

    assert np.array_equal(sim.host_f(), seed)
    assert np.array_equal(
        sim.backend.download(sim.out_prev), seed[:, :, cfg.outlet_col]
    )

    with pytest.raises(ValueError, match="to match the config"):
        sim.load_f(np.zeros((Q, 3, 3), dtype=np.float32))
