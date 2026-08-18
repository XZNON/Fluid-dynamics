"""Rung A — the Warp backend reproduces the NumPy backend.

``DOCS/IDEA3.md`` § Validation ladder, Rung A::

    validate/parity.py — The Warp backend reproduces the NumPy backend.
    Known answer: NumPy itself, plus the four Phase 0 rungs re-run on GPU
    and still inside their published bands.

Run it from the repo root::

    myenv/Scripts/python.exe -m validate.parity --kernels

**This file is T102's half of the rung: the four kernels, one at a time.** The
whole-step comparison, the boundaries and the four Phase 0 rungs on GPU are
T103's (``DOCS/TASKS2.md`` § T103), and ``--kernels`` is a mode rather than the
whole script because the second mode lands beside it there.

Why per-kernel and not per-step
-------------------------------
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
the last bits of an arithmetic kernel differ by design. The threshold below is
therefore a *measured* claim about magnitude, and the script **prints the
number** rather than only PASS — a regression that stays under the bar is still
visible in the digits (``DOCS/TASKS2.md`` § T102).

``stream`` is expected to be **bitwise** equal: it moves values without doing
arithmetic, so anything else is a wrong shift rather than a rounding. The script
prints a bitwise column because "0.00e+00" is evidence and "PASS" is not — and
that column is where the FMA story is legible. Measured in session 14:
``macroscopic`` bitwise on every grid, ``collide`` off by **1.49e-08** (half an
ulp at ``f ~ 0.2``: its last two operations, ``* (1 - omega)`` then ``+ feq``,
contract into one fused multiply-add where NumPy rounds twice), ``equilibrium``
off by **5.96e-08** (one ulp at ``f ~ 0.44``, from the same contraction inside
the polynomial).

The tolerance is ``1e-6`` in ``f`` units and is not to be widened
(``DOCS/TASKS2.md`` § T102 acceptance criteria). ``float32`` has ~1.2e-7
relative precision and ``f`` is order 0.4, so a few ulps is ~1e-7: a difference
above 1e-6 is not float ordering and means the transcription is wrong.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lbm.backends import BackendUnavailableError, get_backend
from lbm.core import Q

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

#: Pass condition, ``f`` units, absolute. **Do not widen** — see the module
#: docstring.
TOL: float = 1e-6

SEED: int = 20260818


@dataclass
class Comparison:
    """One kernel's agreement on one grid.

    Attributes:
        kernel: the kernel's name, e.g. ``"equilibrium"``.
        quantity: what was differenced, e.g. ``"feq"`` or ``"u"``.
        units: ``"f"`` for distribution units, ``"u"`` for lattice velocity.
        shape: the grid, ``(ny, nx)``.
        max_abs: max absolute difference, NumPy minus Warp.
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
    f = get_backend("numpy").equilibrium(rho, u)
    f = f * (1.0 + 0.01 * rng.standard_normal(f.shape)).astype(np.float32)

    return rho, u, f.astype(np.float32)


def _diff(a: NDArray[np.float32], b: NDArray[np.float32]) -> tuple[float, bool]:
    """Max absolute difference and bit-equality of two arrays.

    Args:
        a: reference (NumPy backend) result.
        b: candidate (Warp backend) result.

    Returns:
        ``(max_abs, bitwise)``.
    """
    return float(np.max(np.abs(a - b))), bool(np.array_equal(a, b))


def compare_kernels(
    backend_name: str, ny: int, nx: int, seed: int = SEED
) -> list[Comparison]:
    """Run the four kernels on both backends over one grid and difference them.

    Every call gives both backends the **same** inputs, freshly copied, so the
    only thing that differs between them is the arithmetic. The kernels are
    compared in the order the timestep runs them (``DOCS/IDEA2.md`` § The
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

    # 1. macroscopic: rho is a sum of f (f units); u is a velocity, and the
    #    same 1e-6 bar is *stricter* there, since |u| <= 0.099 against f ~ 0.4.
    rho_ref, u_ref = ref.macroscopic(f.copy())
    rho_dut, u_dut = dut.macroscopic(f.copy())
    max_abs, bitwise = _diff(rho_ref, rho_dut)
    out.append(Comparison("macroscopic", "rho", "f", shape, max_abs, bitwise))
    max_abs, bitwise = _diff(u_ref, u_dut)
    out.append(Comparison("macroscopic", "u", "u", shape, max_abs, bitwise))

    # 2. equilibrium: the kernel this task exists for.
    feq_ref = ref.equilibrium(rho.copy(), u.copy())
    feq_dut = dut.equilibrium(rho.copy(), u.copy())
    max_abs, bitwise = _diff(feq_ref, feq_dut)
    out.append(Comparison("equilibrium", "feq", "f", shape, max_abs, bitwise))

    # 3. collide, in place, from the same f and the same feq. feq comes from
    #    NumPy on both sides on purpose: this measures the collision alone, not
    #    the equilibrium error fed through it.
    f_ref = f.copy()
    f_dut = f.copy()
    ref.collide(f_ref, feq_ref, TAU)
    dut.collide(f_dut, feq_ref, TAU)
    max_abs, bitwise = _diff(f_ref, f_dut)
    out.append(Comparison("collide", "f", "f", shape, max_abs, bitwise))

    # 4. stream, in place. No arithmetic, so anything but bitwise equality is a
    #    bug in the shift, not a rounding difference.
    f_ref = f.copy()
    f_dut = f.copy()
    ref.stream(f_ref, np.empty_like(f_ref))
    dut.stream(f_dut, np.empty_like(f_dut))
    max_abs, bitwise = _diff(f_ref, f_dut)
    out.append(Comparison("stream", "f", "f", shape, max_abs, bitwise))

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
        backend.stream(f, np.empty_like(f))

        ex, ey = int(E[i, 0]), int(E[i, 1])
        expect = np.zeros((Q, ny, nx), dtype=np.float32)
        expect[i, (y0 + ey) % ny, (x0 + ex) % nx] = 1.0
        results.append(bool(np.array_equal(f, expect)))

    return results


def main(argv: list[str] | None = None) -> int:
    """Print Rung A's kernel table and PASS/FAIL.

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
        help="compare the individual kernels (T102). The whole-step mode is T103.",
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

    if not args.kernels:
        print("validate.parity: pass --kernels (the whole-step mode lands in T103).")
        return 2

    # Resolved before the header so that Warp's own init banner does not land
    # in the middle of the table.
    try:
        backend = get_backend(args.backend)
    except BackendUnavailableError as exc:
        print(f"SKIP - {exc}")
        return 2

    print(f"Rung A - kernel parity, {args.backend} vs numpy")
    print(
        f"  random state: rho in [{RHO_RANGE[0]}, {RHO_RANGE[1]}], "
        f"|u| <= {U_MAX}, tau = {TAU}, seed = {args.seed}"
    )
    print(f"  tolerance: max|numpy - {args.backend}| <= {TOL:.0e} in f units")
    print(f"  device: {getattr(backend, 'device', 'n/a')}")
    print()

    rows: list[Comparison] = []
    for ny, nx in GRIDS:
        rows.extend(compare_kernels(args.backend, ny, nx, args.seed))

    header = f"  {'kernel':<12} {'quantity':<9} {'grid':>12} {'max abs diff':>14}   bitwise"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        grid = f"{r.shape[0]}x{r.shape[1]}"
        flag = "ok" if r.ok else "XX"
        bits = "yes" if r.bitwise else "no"
        print(
            f"  {r.kernel:<12} {r.quantity:<9} {grid:>12} "
            f"{r.max_abs:14.3e}   {bits:<4} [{flag}]"
        )

    print()
    print("  worst per kernel:")
    for kernel in ("macroscopic", "equilibrium", "collide", "stream"):
        worst = max((r for r in rows if r.kernel == kernel), key=lambda r: r.max_abs)
        print(
            f"    {kernel:<12} {worst.max_abs:.3e}  "
            f"({worst.quantity}, {worst.shape[0]}x{worst.shape[1]})"
        )

    spikes = spike_directions(args.backend)
    print()
    print(
        f"  stream spike test, all 9 directions land on cell + E[i]: "
        f"{sum(spikes)}/9"
    )

    passed = all(r.ok for r in rows) and all(spikes)
    print()
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
