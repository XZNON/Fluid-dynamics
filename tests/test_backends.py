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
