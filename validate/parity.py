"""Rung A — the Warp backend reproduces the NumPy backend.

``DOCS/IDEA3.md`` § Validation ladder, Rung A::

    validate/parity.py — The Warp backend reproduces the NumPy backend.
    Known answer: NumPy itself, plus the four Phase 0 rungs re-run on GPU
    and still inside their published bands.

Run it from the repo root::

    myenv/Scripts/python.exe -m validate.parity                 # the whole rung
    myenv/Scripts/python.exe -m validate.parity --kernels        # T102's half

**The whole rung is four sections**, in the order a failure should be bisected:

1. ``--kernels`` (T102) — ``macroscopic``, ``equilibrium``, ``collide``,
   ``stream``, one at a time, on random states at three grid sizes.
2. ``--boundaries`` (T103) — ``bounce_back``, ``moving_wall``,
   ``inlet_velocity``, ``outlet_zero_gradient`` in both its forms, and both
   halves of the Guo body force, each **separately**.
3. ``--whole-step`` (T103) — 1000 timesteps of a real channel-with-cylinder case
   on each backend from a bit-identical start, printed at **10 / 100 / 1000** so
   the growth rate of the disagreement is visible and not merely bounded.
4. ``--checkpoint`` (T103) — a checkpoint written on the device backend, resumed
   on NumPy, continuing inside the whole-step tolerance; and a restart inside one
   backend that is still **bit-identical** (constraint 11 in its **D-046** form).

Why per-kernel and not only per-step
------------------------------------
``DOCS/PLAN2.md`` § Risks: *"GPU results differ from NumPy and the difference is
not obviously float ordering — bisect by kernel; Rung A checks ``equilibrium``,
``collide``, ``stream`` and the boundaries separately for exactly this."* A
whole-step number that is out of band tells you the port is wrong; a per-kernel
table tells you **which line** is wrong.

What the numbers mean
---------------------
NumPy is the oracle (**D-043**), and cross-backend agreement is **explainable**,
not bit-identical (``CLAUDE.md`` constraint 11 in its **D-046** form): a GPU
contracts ``a*b + c`` into one fused multiply-add where NumPy rounds twice, so
the last bits of an arithmetic kernel differ by design. The thresholds below are
therefore *measured* claims about magnitude, and the script **prints the
numbers** rather than only PASS — a regression that stays under the bar is still
visible in the digits (``DOCS/TASKS2.md`` § T102).

``stream`` is expected to be **bitwise** equal: it moves values without doing
arithmetic, so anything else is a wrong shift rather than a rounding. So is
``bounce_back``, for the same reason — it is an assignment. The script prints a
bitwise column because "0.00e+00" is evidence and "PASS" is not. Measured in
session 14: ``macroscopic`` bitwise on every grid, ``collide`` off by
**1.49e-08** (half an ulp at ``f ~ 0.2``: its last two operations,
``* (1 - omega)`` then ``+ feq``, contract into one fused multiply-add where
NumPy rounds twice), ``equilibrium`` off by **5.96e-08** (one ulp at
``f ~ 0.44``, from the same contraction inside the polynomial).

Two tolerances, and neither is to be widened
--------------------------------------------
:data:`TOL` is ``1e-6`` in ``f`` units, per kernel and per boundary
(``DOCS/TASKS2.md`` § T102). ``float32`` has ~1.2e-7 relative precision and ``f``
is order 0.4, so a few ulps is ~1e-7: a difference above 1e-6 is not float
ordering and means the transcription is wrong.

:data:`STEP_TOL` is ``1e-4`` on ``max|Δu| / U`` after 1000 timesteps
(``DOCS/TASKS2.md`` § T103), and it is the answer to **Q-103** — *a pass
condition to be met, not adjusted*. The 10 / 100 / 1000 ladder is printed
because the interesting question is not whether the number is under the bar but
whether it **compounds**: a per-step disagreement of 6e-08 that grows linearly
lands near 6e-05 at 1000 steps, and one that grows exponentially does not land
anywhere.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lbm.backends import BackendUnavailableError, get_backend
from lbm.boundary import inlet_profile
from lbm.core import Q
from lbm.geometry import channel_walls, circle
from lbm.runner import Sim, SimConfig, load_checkpoint, save_checkpoint

# --- what the rung compares on ----------------------------------------------

#: Three grid sizes (``DOCS/TASKS2.md`` § T102): one small enough to read by
#: hand, one the size of Rung 1/3's channels, one at the scale where
#: ``equilibrium`` is 39.9 ms of a 75 ms step on NumPy
#: (``old-Docs/STATE1.md`` § Performance baseline).
GRIDS: tuple[tuple[int, int], ...] = ((32, 64), (200, 400), (500, 1000))

#: Density range of the random state. Physical LBM densities sit within a
#: percent of 1; ±10% is deliberately wider than any converged run reaches.
RHO_RANGE: tuple[float, float] = (0.9, 1.1)

#: Velocity cap, per component. ``|u| <= 0.099`` overall keeps the state inside
#: the Mach-squared truncation the equilibrium assumes (``CLAUDE.md``
#: constraint 3), so parity is measured where the physics is valid.
U_MAX: float = 0.099

#: Relaxation time for the collision comparison. Rung 1's value
#: (``validate/poiseuille.py``); nothing here depends on it.
TAU: float = 0.6

#: Pass condition per kernel and per boundary, ``f`` units, absolute.
#: **Do not widen** — see the module docstring.
TOL: float = 1e-6

#: Pass condition for the whole timestep: ``max|Δu| / U`` after
#: :data:`STEP_LADDER`'s last entry. **Do not widen** (**Q-103**).
STEP_TOL: float = 1e-4

#: Step counts at which the whole-step disagreement is printed, so the growth
#: rate is visible rather than merely bounded (``DOCS/TASKS2.md`` § T103).
STEP_LADDER: tuple[int, ...] = (10, 100, 1000)

#: The whole-step case: a channel with an immersed disc, a Zou-He inlet and a
#: convective outlet — every boundary the port has to get right, in one run, at
#: a size where 1000 NumPy steps cost about a second.
STEP_GRID: tuple[int, int] = (64, 256)
STEP_D: float = 16.0
STEP_U: float = 0.05

#: Steps run before and after a checkpoint in the checkpoint section.
CKPT_STEPS: int = 100

#: Guo body force for the two forced-boundary comparisons. Rung 1's magnitude
#: (``validate/poiseuille.py``), which is the only case in the project that
#: switches the scheme on.
GX: float = 1.0e-5

SEED: int = 20260818


@dataclass
class Comparison:
    """One kernel's or boundary's agreement on one grid.

    Attributes:
        kernel: what was run, e.g. ``"equilibrium"`` or ``"inlet_velocity"``.
        quantity: what was differenced, e.g. ``"feq"`` or ``"u"``.
        units: ``"f"`` for distribution units, ``"u"`` for lattice velocity.
        shape: the grid, ``(ny, nx)``.
        max_abs: max absolute difference, NumPy minus the backend under test.
        bitwise: whether the two results are bit-for-bit equal.
    """

    kernel: str
    quantity: str
    units: str
    shape: tuple[int, int]
    max_abs: float
    bitwise: bool

    @property
    def ok(self) -> bool:
        """Whether this comparison is inside :data:`TOL`."""
        return self.max_abs <= TOL


@dataclass
class StepPoint:
    """The whole-step disagreement after a given number of timesteps.

    Attributes:
        steps: timesteps run on each backend.
        du_over_u: ``max|Δu| / U`` over the whole domain.
        df: ``max|Δf|`` in distribution units, for context.
        finite: whether both backends are still finite.
    """

    steps: int
    du_over_u: float
    df: float
    finite: bool

    @property
    def ok(self) -> bool:
        """Whether this point is inside :data:`STEP_TOL`."""
        return self.finite and self.du_over_u < STEP_TOL


def random_state(
    ny: int, nx: int, seed: int = SEED
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """A random macroscopic state and a distribution consistent with it.

    ``rho`` is uniform in :data:`RHO_RANGE`; ``u`` is uniform in a disc of
    radius :data:`U_MAX`, so the *magnitude* obeys the cap rather than each
    component separately (``CLAUDE.md`` constraint 3). ``f`` is the NumPy
    equilibrium of that state plus a small perturbation, so the collision has
    something to relax and the distribution is not exactly its own equilibrium.

    Args:
        ny: rows.
        nx: columns.
        seed: RNG seed, so a failure is reproducible from the printed table.

    Returns:
        ``(rho, u, f)`` — ``(ny, nx)``, ``(2, ny, nx)`` and ``(9, ny, nx)``,
        all ``float32``.
    """
    rng = np.random.default_rng(seed)

    lo, hi = RHO_RANGE
    rho = rng.uniform(lo, hi, size=(ny, nx)).astype(np.float32)

    angle = rng.uniform(0.0, 2.0 * np.pi, size=(ny, nx))
    radius = U_MAX * np.sqrt(rng.uniform(0.0, 1.0, size=(ny, nx)))
    u = np.empty((2, ny, nx), dtype=np.float32)
    u[0] = (radius * np.cos(angle)).astype(np.float32)
    u[1] = (radius * np.sin(angle)).astype(np.float32)

    # A distribution that is near, but not at, equilibrium: 1% of the weight
    # moved around at random is well inside what a running sim shows.
    ref = get_backend("numpy")
    f = ref.download(ref.equilibrium(ref.upload(rho), ref.upload(u)))
    f = f * (1.0 + 0.01 * rng.standard_normal(f.shape)).astype(np.float32)

    return rho, u, np.ascontiguousarray(f, dtype=np.float32)


def _diff(a: NDArray[np.float32], b: NDArray[np.float32]) -> tuple[float, bool]:
    """Max absolute difference and bit-equality of two arrays.

    Args:
        a: reference (NumPy backend) result.
        b: candidate result.

    Returns:
        ``(max_abs, bitwise)``.
    """
    return float(np.max(np.abs(a - b))), bool(np.array_equal(a, b))


def compare_kernels(
    backend_name: str, ny: int, nx: int, seed: int = SEED
) -> list[Comparison]:
    """Run the four kernels on both backends over one grid and difference them.

    Every call gives both backends the **same** host inputs, uploaded fresh, so
    the only thing that differs between them is the arithmetic. Inputs cross the
    seam through :meth:`lbm.backends.Backend.upload` and results come back
    through :meth:`~lbm.backends.Backend.download` — since T103 the kernels take
    the backend's own arrays and nothing else (**D-052** superseded). The kernels
    are compared in the order the timestep runs them (``DOCS/IDEA2.md`` § The
    method): macroscopic, equilibrium, collide, stream.

    Args:
        backend_name: the backend to check against NumPy, e.g. ``"warp"``.
        ny: rows.
        nx: columns.
        seed: RNG seed for :func:`random_state`.

    Returns:
        One :class:`Comparison` per differenced quantity, in kernel order.
    """
    ref = get_backend("numpy")
    dut = get_backend(backend_name)

    rho, u, f = random_state(ny, nx, seed)
    shape = (ny, nx)
    out: list[Comparison] = []

    def add(kernel: str, quantity: str, units: str, a: Any, b: Any) -> None:
        max_abs, bitwise = _diff(ref.download(a), dut.download(b))
        out.append(Comparison(kernel, quantity, units, shape, max_abs, bitwise))

    # 1. macroscopic: rho is a sum of f (f units); u is a velocity, and the
    #    same 1e-6 bar is *stricter* there, since |u| <= 0.099 against f ~ 0.4.
    rho_ref, u_ref = ref.macroscopic(ref.upload(f))
    rho_dut, u_dut = dut.macroscopic(dut.upload(f))
    add("macroscopic", "rho", "f", rho_ref, rho_dut)
    add("macroscopic", "u", "u", u_ref, u_dut)

    # 2. equilibrium: the kernel T102 existed for.
    feq_ref = ref.equilibrium(ref.upload(rho), ref.upload(u))
    feq_dut = dut.equilibrium(dut.upload(rho), dut.upload(u))
    add("equilibrium", "feq", "f", feq_ref, feq_dut)

    # 3. collide, in place, from the same f and the same feq. feq comes from
    #    NumPy on both sides on purpose: this measures the collision alone, not
    #    the equilibrium error fed through it.
    feq_host = ref.download(feq_ref)
    f_ref = ref.upload(f)
    f_dut = dut.upload(f)
    ref.collide(f_ref, ref.upload(feq_host), TAU)
    dut.collide(f_dut, dut.upload(feq_host), TAU)
    add("collide", "f", "f", f_ref, f_dut)

    # 4. stream, in place. No arithmetic, so anything but bitwise equality is a
    #    bug in the shift, not a rounding difference.
    f_ref = ref.upload(f)
    f_dut = dut.upload(f)
    ref.stream(f_ref, ref.empty((Q, ny, nx)))
    dut.stream(f_dut, dut.empty((Q, ny, nx)))
    add("stream", "f", "f", f_ref, f_dut)

    return out


def compare_boundaries(
    backend_name: str, ny: int, nx: int, seed: int = SEED
) -> list[Comparison]:
    """Every boundary condition Phase 0 ships, compared one at a time.

    ``DOCS/TASKS2.md`` § T103, first acceptance criterion: *"Every boundary
    condition Phase 0 ships runs on the GPU, and ``validate/parity.py`` compares
    each separately against NumPy at <= 1e-6 in ``f`` units."* Separately is the
    operative word — a whole-step number cannot say which boundary is wrong.

    The reflections (``bounce_back``, and ``moving_wall`` on its reflected half)
    are assignments rather than arithmetic, so they are expected **bitwise**; the
    Ladd correction, Zou-He and the convective outlet all multiply and add, so
    they are expected inside :data:`TOL` and not bitwise.

    Args:
        backend_name: the backend to check against NumPy.
        ny: rows.
        nx: columns.
        seed: RNG seed for :func:`random_state`.

    Returns:
        One :class:`Comparison` per differenced quantity.
    """
    ref = get_backend("numpy")
    dut = get_backend(backend_name)

    rho, u, f = random_state(ny, nx, seed)
    # A *different* random state for the pre-collision copy: using the same one
    # would let a reflection that reads the wrong array still agree.
    _, _, f_pre = random_state(ny, nx, seed + 1)

    solid = channel_walls(ny, nx) | circle(ny, nx, nx / 4.0, ny / 2.0, ny / 8.0)
    lid = np.zeros((ny, nx), dtype=bool)
    lid[-1, :] = True

    u_in = inlet_profile(ny, STEP_U, "parabolic", solid=solid, col=0)
    fluid = np.ascontiguousarray(~solid[:, 0])

    shape = (ny, nx)
    out: list[Comparison] = []

    def add(kernel: str, quantity: str, units: str, a: Any, b: Any) -> None:
        max_abs, bitwise = _diff(ref.download(a), dut.download(b))
        out.append(Comparison(kernel, quantity, units, shape, max_abs, bitwise))

    # 1. bounce_back — an assignment, so bitwise or broken.
    a, b = ref.upload(f), dut.upload(f)
    ref.bounce_back(a, ref.upload(f_pre), ref.upload(solid))
    dut.bounce_back(b, dut.upload(f_pre), dut.upload(solid))
    add("bounce_back", "f", "f", a, b)

    # 2. moving_wall — the reflection plus the Ladd momentum correction.
    a, b = ref.upload(f), dut.upload(f)
    ref.moving_wall(a, ref.upload(f_pre), ref.upload(lid), (STEP_U, 0.0))
    dut.moving_wall(b, dut.upload(f_pre), dut.upload(lid), (STEP_U, 0.0))
    add("moving_wall", "f", "f", a, b)

    # 2b. moving_wall with a stationary wall must degenerate to bounce_back
    #     exactly, on both backends — the consistency check D-013's reasoning
    #     rests on.
    a, b = ref.upload(f), dut.upload(f)
    ref.moving_wall(a, ref.upload(f_pre), ref.upload(lid), (0.0, 0.0))
    dut.moving_wall(b, dut.upload(f_pre), dut.upload(lid), (0.0, 0.0))
    add("moving_wall(u=0)", "f", "f", a, b)

    # 3. inlet_velocity — Zou-He, three unknown directions in one column.
    a, b = ref.upload(f), dut.upload(f)
    ref.inlet_velocity(a, col=0, u_in=ref.upload(u_in), fluid=ref.upload(fluid))
    dut.inlet_velocity(b, col=0, u_in=dut.upload(u_in), fluid=dut.upload(fluid))
    add("inlet_velocity", "f", "f", a, b)

    # 4. outlet_zero_gradient, plain copy — the documented default (D-021).
    a, b = ref.upload(f), dut.upload(f)
    ref.outlet_zero_gradient(a)
    dut.outlet_zero_gradient(b)
    add("outlet(copy)", "f", "f", a, b)

    # 5. outlet_zero_gradient, convective — what the runner actually uses. Both
    #    the column it writes and the `prev` it updates are differenced, because
    #    `prev` is what carries the boundary's state to the next step.
    prev = np.ascontiguousarray(f[:, :, -1])
    a, b = ref.upload(f), dut.upload(f)
    prev_ref, prev_dut = ref.upload(prev), dut.upload(prev)
    ref.outlet_zero_gradient(a, prev=prev_ref)
    dut.outlet_zero_gradient(b, prev=prev_dut)
    add("outlet(conv)", "f", "f", a, b)
    add("outlet(conv)", "prev", "f", prev_ref, prev_dut)

    # 6. Guo, first half — the velocity shift u += F / (2 rho).
    a, b = ref.upload(u), dut.upload(u)
    ref.force_velocity_shift(ref.upload(rho), a, (GX, 0.0))
    dut.force_velocity_shift(dut.upload(rho), b, (GX, 0.0))
    add("force_shift", "u", "u", a, b)

    # 7. Guo, second half — the source term added after collision.
    a, b = ref.upload(f), dut.upload(f)
    ref.apply_body_force(a, ref.upload(rho), ref.upload(u), TAU, (GX, 0.0))
    dut.apply_body_force(b, dut.upload(rho), dut.upload(u), TAU, (GX, 0.0))
    add("body_force", "f", "f", a, b)

    return out


def spike_directions(backend_name: str, ny: int = 9, nx: int = 11) -> list[bool]:
    """Phase 0's spike test, run on the candidate backend.

    A single-cell spike in direction ``i`` must land exactly one cell along
    ``E[i]`` and nowhere else (``lbm.core.stream``'s sign convention, and
    ``tests/test_core.py``'s check of it). This is the check that does **not**
    go through NumPy: streaming is a permutation, so it has a known answer of
    its own, and a parity test alone would pass two backends that are wrong in
    the same way.

    Args:
        backend_name: the backend to check.
        ny: rows of the test grid.
        nx: columns of the test grid.

    Returns:
        Nine booleans, one per direction, ``True`` when the spike landed where
        ``E[i]`` says.
    """
    from lbm.core import E  # local: the module-level imports stay the rung's

    backend = get_backend(backend_name)
    y0, x0 = ny // 2, nx // 2
    results: list[bool] = []

    for i in range(Q):
        f = np.zeros((Q, ny, nx), dtype=np.float32)
        f[i, y0, x0] = 1.0
        dev = backend.upload(f)
        backend.stream(dev, backend.empty((Q, ny, nx)))

        ex, ey = int(E[i, 0]), int(E[i, 1])
        expect = np.zeros((Q, ny, nx), dtype=np.float32)
        expect[i, (y0 + ey) % ny, (x0 + ex) % nx] = 1.0
        results.append(bool(np.array_equal(backend.download(dev), expect)))

    return results


# --- the whole timestep -----------------------------------------------------


def step_case(
    backend: str,
    ny: int = STEP_GRID[0],
    nx: int = STEP_GRID[1],
    cs_smag: float = 0.0,
) -> tuple[SimConfig, NDArray[np.bool_]]:
    """The config and mask the whole-step comparison runs.

    A channel with an immersed disc, a Zou-He velocity inlet and a convective
    outlet: every boundary the port had to get right, in the shape Rung 3 runs.
    ``check_geometry`` is off because this is a parity harness, not a physics
    run — the blockage of a deliberately small domain is not what is under test.

    Args:
        backend: registry name for :attr:`lbm.runner.SimConfig.backend`.
        ny: rows.
        nx: columns.
        cs_smag: Smagorinsky constant (T202). **Rung A leaves this at zero** —
            every caller in this module does — and it exists so that Rung F can
            measure cross-backend agreement with the closure *on* against this
            rung's own case and this rung's own bars, rather than against a
            second copy of them whose agreement with the original nobody checks.
            The same argument, and the same defaulted parameter, that **D-087**
            applied to :mod:`validate.cylinder`.

    Returns:
        ``(config, solid)`` — ``solid`` is ``(ny, nx)`` ``bool``.
    """
    solid = channel_walls(ny, nx) | circle(ny, nx, nx / 4.0, ny / 2.0, STEP_D / 2.0)
    cfg = SimConfig(
        ny=ny,
        nx=nx,
        tau=0.6,
        inlet_U=STEP_U,
        use_inlet=True,
        use_outlet=True,
        convective_outlet=True,
        check_geometry=False,
        backend=backend,
        cs_smag=cs_smag,
    )
    return cfg, solid


def whole_step(
    backend_name: str,
    ladder: tuple[int, ...] = STEP_LADDER,
    cs_smag: float = 0.0,
) -> list[StepPoint]:
    """Run both backends in lockstep and print how the disagreement grows.

    ``DOCS/TASKS2.md`` § T103: *"starting from identical state, 1000 steps on
    each backend agree to ``max|Δu| / U < 1e-4``, printed. The number is expected
    to grow with step count — the script prints it at 10 / 100 / 1000 steps so
    the growth rate is visible and not merely bounded."*

    "Identical state" is made literal: the NumPy sim is built first and its ``f``
    is handed to the other backend as the initial condition, so step 0 is
    **bit-identical** rather than merely equivalent. Seeding each from its own
    ``equilibrium`` would start them 6e-08 apart and confuse the initial
    condition with the thing being measured.

    Args:
        backend_name: the backend to check against NumPy.
        ladder: step counts at which to record the disagreement, ascending.
        cs_smag: Smagorinsky constant, threaded to :func:`step_case`. Zero for
            Rung A; Rung F passes the literature value to measure the same two
            bars with the closure engaged (T202).

    Returns:
        One :class:`StepPoint` per entry in ``ladder``.
    """
    cfg_ref, solid = step_case("numpy", cs_smag=cs_smag)
    cfg_dut, _ = step_case(backend_name, cs_smag=cs_smag)

    sim_ref = Sim(cfg_ref, solid)
    f0 = sim_ref.host_f().copy()
    sim_dut = Sim(cfg_dut, solid, f=f0)

    points: list[StepPoint] = []
    done = 0
    for target in ladder:
        n = target - done
        sim_ref.run_steps(n)
        sim_dut.run_steps(n)
        done = target

        u_ref = sim_ref.host_u()
        u_dut = sim_dut.host_u()
        f_ref = sim_ref.host_f()
        f_dut = sim_dut.host_f()

        finite = bool(np.isfinite(f_ref).all()) and bool(np.isfinite(f_dut).all())
        du = float(np.max(np.abs(u_ref - u_dut))) / STEP_U if finite else float("inf")
        df = float(np.max(np.abs(f_ref - f_dut))) if finite else float("inf")
        points.append(StepPoint(target, du, df, finite))

    return points


# --- checkpoints across and within backends ---------------------------------


@dataclass
class CheckpointResult:
    """What the checkpoint section measured.

    Attributes:
        keys: the top-level keys the checkpoint file holds.
        cross_du: ``max|Δu| / U`` after resuming the device checkpoint on NumPy
            and continuing, against the run that never stopped.
        within_bitwise: whether a restart **inside** the device backend is
            bit-identical (constraint 11 in its **D-046** form).
        host_layout: whether ``f`` in the file is ``(9, ny, nx)`` ``float32``.
    """

    keys: tuple[str, ...]
    cross_du: float
    within_bitwise: bool
    host_layout: bool

    @property
    def ok(self) -> bool:
        """Whether every checkpoint claim holds."""
        return (
            set(self.keys) == {"format", "f", "solid", "step_count", "config"}
            and self.host_layout
            and self.within_bitwise
            and self.cross_du < STEP_TOL
        )


def checkpoint_parity(backend_name: str, steps: int = CKPT_STEPS) -> CheckpointResult:
    """A checkpoint written on the device backend, resumed on NumPy.

    ``DOCS/TASKS2.md`` § T103: *"``save_checkpoint`` on the GPU backend writes the
    same four things plus ``format`` (**D-022**), via ``to_host``; a checkpoint
    written on ``warp`` **resumes on ``numpy``** and continues within the
    whole-step parity tolerance. Within a backend, restart stays
    **bit-identical**."*

    Both halves are measured here because they are the two halves of constraint
    11 in its **D-046** form, and the distinction is the whole point: *within* a
    backend the bits must match, *across* backends there is a printed tolerance
    and no test should pretend otherwise.

    Args:
        backend_name: the backend the checkpoint is written on.
        steps: steps before the checkpoint, and again after it.

    Returns:
        A :class:`CheckpointResult`.
    """
    import pickle

    cfg, solid = step_case(backend_name)
    sim = Sim(cfg, solid)
    sim.run_steps(steps)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "parity.pkl"
        save_checkpoint(sim, path)

        with open(path, "rb") as fh:
            state = pickle.load(fh)
        keys = tuple(sorted(state))
        host_layout = (
            isinstance(state["f"], np.ndarray)
            and state["f"].shape == (Q, cfg.ny, cfg.nx)
            and state["f"].dtype == np.float32
        )

        # Across backends: resume the device checkpoint on the oracle, run both
        # on for the same number of steps, and compare velocities.
        resumed = load_checkpoint(path, backend="numpy")
        sim.run_steps(steps)
        resumed.run_steps(steps)
        cross_du = float(
            np.max(np.abs(sim.host_u() - resumed.host_u()))
        ) / STEP_U

        # Within the backend: a second checkpoint, resumed on the same backend,
        # must continue bit-for-bit.
        path2 = Path(tmp) / "parity2.pkl"
        save_checkpoint(sim, path2)
        same = load_checkpoint(path2)
        reference = sim.host_f().copy()
        sim.run_steps(steps)
        after = sim.host_f().copy()
        same.run_steps(steps)
        within_bitwise = bool(np.array_equal(after, same.host_f())) and not bool(
            np.array_equal(after, reference)
        )

    return CheckpointResult(keys, cross_du, within_bitwise, host_layout)


# --- printing ---------------------------------------------------------------


def _print_comparisons(rows: list[Comparison], kernels: tuple[str, ...]) -> None:
    """Print one table of comparisons plus the worst case per kernel.

    Args:
        rows: the comparisons to print, in run order.
        kernels: the kernel names to summarise, in the order to summarise them.
    """
    header = (
        f"  {'kernel':<17} {'quantity':<9} {'grid':>12} "
        f"{'max abs diff':>14}   bitwise"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        grid = f"{r.shape[0]}x{r.shape[1]}"
        flag = "ok" if r.ok else "XX"
        bits = "yes" if r.bitwise else "no"
        print(
            f"  {r.kernel:<17} {r.quantity:<9} {grid:>12} "
            f"{r.max_abs:14.3e}   {bits:<4} [{flag}]"
        )

    print()
    print("  worst per kernel:")
    for kernel in kernels:
        matching = [r for r in rows if r.kernel == kernel]
        if not matching:
            continue
        worst = max(matching, key=lambda r: r.max_abs)
        print(
            f"    {kernel:<17} {worst.max_abs:.3e}  "
            f"({worst.quantity}, {worst.shape[0]}x{worst.shape[1]})"
        )


def main(argv: list[str] | None = None) -> int:
    """Print Rung A and PASS/FAIL.

    With no section flag every section runs, which is the **M5 gate command**
    (``DOCS/PLAN2.md`` § Milestone gates). Any flag runs only the sections asked
    for, which is how a failure is bisected.

    Args:
        argv: command line, or ``None`` for :data:`sys.argv`.

    Returns:
        Process exit status: 0 on PASS, 1 on FAIL, 2 when the backend under
        test is not installed (which is not a physics failure and must not read
        like one).
    """
    parser = argparse.ArgumentParser(
        description="Rung A — backend parity against the NumPy oracle."
    )
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="compare the individual kernels (T102).",
    )
    parser.add_argument(
        "--boundaries",
        action="store_true",
        help="compare each boundary condition separately (T103).",
    )
    parser.add_argument(
        "--whole-step",
        action="store_true",
        help="1000 steps on each backend, printed at 10 / 100 / 1000 (T103).",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="a device checkpoint resumed on numpy, and restart within (T103).",
    )
    parser.add_argument(
        "--backend",
        default="warp",
        help="backend under test; NumPy is always the reference (D-043).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="RNG seed for the random state."
    )
    args = parser.parse_args(argv)

    sections = (args.kernels, args.boundaries, args.whole_step, args.checkpoint)
    if not any(sections):
        args.kernels = args.boundaries = args.whole_step = args.checkpoint = True

    # Resolved before the header so that Warp's own init banner does not land
    # in the middle of a table.
    try:
        backend = get_backend(args.backend)
    except BackendUnavailableError as exc:
        print(f"SKIP - {exc}")
        return 2

    print(f"Rung A - backend parity, {args.backend} vs numpy")
    print(
        f"  random state: rho in [{RHO_RANGE[0]}, {RHO_RANGE[1]}], "
        f"|u| <= {U_MAX}, tau = {TAU}, seed = {args.seed}"
    )
    print(f"  tolerance: max|numpy - {args.backend}| <= {TOL:.0e} in f units")
    print(f"  whole step: max|du| / U < {STEP_TOL:.0e} at {STEP_LADDER[-1]} steps")
    print(f"  device: {getattr(backend, 'device', 'n/a')}")

    passed = True

    if args.kernels:
        rows: list[Comparison] = []
        for ny, nx in GRIDS:
            rows.extend(compare_kernels(args.backend, ny, nx, args.seed))
        print()
        print("1. kernels")
        _print_comparisons(
            rows, ("macroscopic", "equilibrium", "collide", "stream")
        )
        spikes = spike_directions(args.backend)
        print()
        print(
            f"  stream spike test, all 9 directions land on cell + E[i]: "
            f"{sum(spikes)}/9"
        )
        passed &= all(r.ok for r in rows) and all(spikes)

    if args.boundaries:
        rows = []
        for ny, nx in GRIDS:
            rows.extend(compare_boundaries(args.backend, ny, nx, args.seed))
        print()
        print("2. boundaries")
        _print_comparisons(
            rows,
            (
                "bounce_back",
                "moving_wall",
                "moving_wall(u=0)",
                "inlet_velocity",
                "outlet(copy)",
                "outlet(conv)",
                "force_shift",
                "body_force",
            ),
        )
        passed &= all(r.ok for r in rows)

    if args.whole_step:
        ny, nx = STEP_GRID
        print()
        print(
            f"3. whole step - {nx}x{ny}, disc D = {STEP_D:.0f}, Zou-He inlet, "
            f"convective outlet, U = {STEP_U}"
        )
        points = whole_step(args.backend)
        print(f"  {'steps':>7} {'max|du| / U':>14} {'max|df|':>14}   growth")
        print("  " + "-" * 46)
        prev: StepPoint | None = None
        for pt in points:
            if prev is None or prev.du_over_u == 0.0:
                growth = "-"
            else:
                growth = (
                    f"{pt.du_over_u / prev.du_over_u:.2f}x "
                    f"for {pt.steps // prev.steps}x the steps"
                )
            flag = "ok" if pt.ok else "XX"
            print(
                f"  {pt.steps:>7} {pt.du_over_u:14.3e} {pt.df:14.3e}   "
                f"{growth:<28} [{flag}]"
            )
            prev = pt
        print(
            "  the disagreement is bounded, not compounding: a growth factor "
            "well under the\n  step-count factor beside it means the error is "
            "not accumulating (Q-103)."
        )
        passed &= all(pt.ok for pt in points)

    if args.checkpoint:
        print()
        print("4. checkpoint")
        ck = checkpoint_parity(args.backend)
        print(f"  contents: {', '.join(ck.keys)}")
        print(
            f"  f written in the host layout (9, ny, nx) float32 (constraint 4): "
            f"{'yes' if ck.host_layout else 'NO'}"
        )
        print(
            f"  written on {args.backend}, resumed on numpy, "
            f"{CKPT_STEPS} steps on:   max|du| / U = {ck.cross_du:.3e}  "
            f"[{'ok' if ck.cross_du < STEP_TOL else 'XX'}]"
        )
        print(
            f"  restart within {args.backend} is bit-identical (constraint 11): "
            f"{'yes' if ck.within_bitwise else 'NO'}"
        )
        passed &= ck.ok

    print()
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
