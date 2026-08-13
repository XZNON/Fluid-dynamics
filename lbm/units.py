"""Physical <-> lattice conversion. The only module in ``lbm/`` that holds metres.

``CLAUDE.md`` § Coding conventions: *physical units never reach the solver.*
Everything else inside ``lbm/`` is in lattice units — cells and timesteps — and
:class:`LatticeUnits` is the boundary where a user's "air, 20 m/s, a 1.5 m wing"
becomes ``tau``, a lattice ``U`` and a grid. It converts and it refuses; it does
not run anything.

The derivation, in full
-----------------------

Three physical numbers describe the case: a characteristic length ``L`` (m), a
characteristic speed ``u_phys`` (m/s), and the kinematic viscosity ``nu_phys``
(m^2/s). Only their combination matters::

    Re = u_phys * L / nu_phys

Two lattice numbers are then *chosen* rather than derived — the resolution
``N`` (cells across ``L``) and the lattice velocity ``U`` — and everything else
follows:

1. **Cell size.** ``dx = L / N`` metres per cell.
2. **Timestep.** The lattice velocity is a velocity in cells per timestep, so
   matching the physical speed gives ``U = u_phys * dt / dx``, i.e.::

       dt = U * dx / u_phys          seconds per timestep

   This is exactly the ``dt`` of **D-023** — seconds of physical time per
   lattice timestep — and is what :func:`lbm.runner.steps_per_frame` consumes.
3. **Lattice viscosity.** Viscosity has units of length^2/time, so it scales as
   ``dt / dx^2``::

       nu = nu_phys * dt / dx^2 = U * N / Re

   The two forms are algebraically identical; the second shows that ``nu``
   depends on the physics only through ``Re``.
4. **Relaxation time** (``CLAUDE.md`` constraint 2, ``nu = cs2 (tau - 0.5)``)::

       tau = 0.5 + 3 nu = 0.5 + 3 U N / Re

   ``tau`` is *derived*. There is no ``nu`` setter here and no way to set ``tau``
   alongside one: :meth:`LatticeUnits.from_physical` takes either the resolution
   or a target ``tau`` and computes the other from that single expression.

Which knob to turn is worth stating once, because the two guards below pull
against each other: raising ``N`` raises ``tau`` (more stable) at a cost that
grows with the area of the domain, while raising ``U`` raises ``tau`` for free
but walks into the Mach ceiling. Session 8 spent three runs learning that ``D``
is the only knob that buys ``tau`` without moving the peak velocity
(``old-Docs/STATE1.md`` § Session log, session 8).

What it refuses
---------------

* **Lattice velocity at or above 0.1** — ``CLAUDE.md`` constraint 3.
  Compressibility error scales as Mach squared, and this is the guard's home
  *for users*: it raises, it does not warn. Note that ``U`` here is the
  free-stream/characteristic velocity, and the flow **accelerates around a
  body** — measured at ``1.79 U`` past a square cylinder and ``1.61 U`` past a
  disc (session 8). :data:`BLUFF_BODY_SPEEDUP` is that headroom, and
  :meth:`LatticeUnits.peak_velocity_estimate` applies it.
* **``tau`` at or below 0.51** — the classic ``nan``, and the "checkerboard"
  row of ``DOCS/IDEA2.md`` § Stability.

  **0.51 is a floor on nonsense, not a promise of stability.** It is the third
  ``tau`` floor in this project and deliberately the loosest: Rung 2 enforces
  0.53 (**D-016**) and Rung 4 enforces a measured 0.54 (**D-029**), because
  **D-029** measured a disc at ``tau = 0.5330`` blowing up by step 1500 and a
  square at 0.5346 by step 3200. Those floors are properties of *those cases* —
  a bluff body in a free stream — and this module converts units for cases it
  has never seen, so it cannot honestly enforce them. A config accepted here at
  ``tau = 0.52`` can still produce ``nan``; that is why
  :meth:`LatticeUnits.stability_note` exists and why the error message names the
  measured numbers rather than implying safety.

Both errors name the offending quantity and print the resolution that would fix
it, because "your config is unstable" without a number is a message a user
cannot act on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "LatticeUnits",
    "U_LATTICE_MAX",
    "TAU_FLOOR",
    "U_LATTICE_DEFAULT",
    "BLUFF_BODY_SPEEDUP",
]

#: ``CLAUDE.md`` constraint 3. Hard ceiling on the lattice velocity.
U_LATTICE_MAX: float = 0.1

#: The loosest defensible relaxation-time floor — see the module docstring. Rung
#: 2 uses 0.53 (D-016) and Rung 4 a measured 0.54 (D-029); neither is imposed
#: here because neither was measured on an arbitrary user case.
TAU_FLOOR: float = 0.51

#: Default lattice velocity when the caller does not pick one. Half the ceiling:
#: it leaves the ~1.8x headroom a bluff body needs (session 8 measured peak
#: ``= 1.79 U`` for a square cylinder) and keeps the compressibility error at a
#: quarter of what the ceiling allows.
U_LATTICE_DEFAULT: float = 0.05

#: Measured speed-up of the flow past a bluff body, as a multiple of the
#: free-stream velocity: 1.79 for the square cylinder of session 8 (peak 0.09758
#: at ``U = 0.053``), 1.61 for the Rung 3 disc (0.09685 at ``U = 0.06``). The
#: larger is used, because the guard should be pessimistic.
BLUFF_BODY_SPEEDUP: float = 1.8


@dataclass(frozen=True)
class LatticeUnits:
    """A physical case converted to lattice scalars, and the way back.

    Frozen: the conversion is fixed once the case is described, and a mutable
    ``tau`` would be a back door around the derivation (constraint 2). Build one
    with :meth:`from_physical`.

    Attributes:
        dx: metres per cell.
        dt: seconds per lattice timestep — the ``dt`` of **D-023**, which
            :func:`lbm.runner.steps_per_frame` takes.
        tau: BGK relaxation time, derived from ``Re``, ``U`` and the resolution.
        U: lattice velocity corresponding to ``u_phys``. Dimensionless, and
            below :data:`U_LATTICE_MAX` by construction.
        Re: Reynolds number of the case.
        cells_per_length: resolution ``N`` — cells across the characteristic
            length ``L``.
        u_phys: characteristic speed, m/s.
        l_phys: characteristic length, m.
        nu_phys: kinematic viscosity, m^2/s.
    """

    dx: float
    dt: float
    tau: float
    U: float
    Re: float
    cells_per_length: float
    u_phys: float
    l_phys: float
    nu_phys: float

    # -- derived lattice quantities ---------------------------------------

    @property
    def nu(self) -> float:
        """Lattice viscosity, ``(tau - 0.5) / 3`` (``CLAUDE.md`` constraint 2).

        Computed from ``tau`` and nothing else, so it cannot drift from the
        value the solver actually relaxes with.
        """
        return (self.tau - 0.5) / 3.0

    @property
    def D_cells(self) -> float:
        """The characteristic length in cells — an alias for the resolution.

        Named for **D-019**, where ``D`` is the cross-stream extent of the
        object's bounding box. When the mask comes from
        :func:`lbm.geometry.from_png` the *measured* extent is what force
        coefficients are divided by, so pass that measured number in as
        ``cells_per_length`` rather than a nominal one.
        """
        return self.cells_per_length

    # -- conversions -------------------------------------------------------

    def to_lattice_velocity(self, u_phys: float) -> float:
        """m/s -> lattice velocity, via ``u * dt / dx``."""
        return u_phys * self.dt / self.dx

    def to_physical_velocity(self, u_lattice: float) -> float:
        """Lattice velocity -> m/s. Inverse of :meth:`to_lattice_velocity`."""
        return u_lattice * self.dx / self.dt

    def to_lattice_length(self, metres: float) -> float:
        """Metres -> cells."""
        return metres / self.dx

    def to_physical_length(self, cells: float) -> float:
        """Cells -> metres."""
        return cells * self.dx

    def to_lattice_time(self, seconds: float) -> float:
        """Seconds -> timesteps (fractional; round where you need an integer)."""
        return seconds / self.dt

    def to_physical_time(self, steps: float) -> float:
        """Timesteps -> seconds."""
        return steps * self.dt

    def to_lattice_viscosity(self, nu_phys: float) -> float:
        """m^2/s -> lattice units, via ``nu * dt / dx^2``.

        Applied to :attr:`nu_phys` this reproduces :attr:`nu` — the identity the
        round-trip test checks, and the reason the derivation in the module
        docstring is written both ways.
        """
        return nu_phys * self.dt / (self.dx * self.dx)

    def reynolds(self) -> float:
        """``Re`` recomputed from the **lattice** numbers, ``U * N / nu``.

        The round trip of the acceptance criteria: this goes through ``tau``,
        the resolution and the lattice velocity only — no metres — and must
        reproduce :attr:`Re` to within 0.1%.
        """
        return self.U * self.cells_per_length / self.nu

    def peak_velocity_estimate(self, speedup: float = BLUFF_BODY_SPEEDUP) -> float:
        """Expected peak lattice velocity, ``speedup * U``.

        The free stream is not what constraint 3 caps — the *peak* is, and the
        flow accelerates round a body (:data:`BLUFF_BODY_SPEEDUP`). This is an
        estimate for planning a case, not a measurement; the run itself must
        still print its measured peak.
        """
        return speedup * self.U

    def stability_note(self) -> str:
        """One line on how much margin this ``tau`` actually has (**D-029**).

        Written as a method rather than an exception because a config that sits
        between the 0.51 floor here and the 0.54 measured for a bluff body is
        *accepted* — the module cannot know what geometry is coming — and the
        user is owed the number rather than silence.
        """
        if self.tau >= 0.5512:
            return (
                f"tau = {self.tau:.4f}: comfortable. Measured stable for 60000 "
                "steps on a disc and a square cylinder (D-029)."
            )
        if self.tau >= 0.54:
            return (
                f"tau = {self.tau:.4f}: adequate. Above Rung 4's measured floor "
                "of 0.54 (D-029); Rung 3 runs at 0.5378 and is stable."
            )
        if self.tau >= 0.5378:
            return (
                f"tau = {self.tau:.4f}: marginal. 0.5378 survived 60000 steps "
                "and 0.5346 produced nan by step 3200 (D-029). Raise the "
                "resolution if the body is bluff."
            )
        return (
            f"tau = {self.tau:.4f}: BELOW the measured bluff-body floor. A disc "
            "at 0.5330 blew up by step 1500 and a square at 0.5346 by step 3200 "
            f"(D-029). Accepted only because the 0.51 floor is a floor on "
            "nonsense, not a stability guarantee — expect nan. Raise "
            f"cells_per_length to at least "
            f"{self.resolution_for_tau(0.54):.0f}."
        )

    def resolution_for_tau(self, tau: float) -> float:
        """Cells across ``L`` that would give this case that ``tau``.

        Inverts ``tau = 0.5 + 3 U N / Re`` for ``N``. This is what the two
        rejection messages quote, so a refused config comes back with a number
        to act on rather than an adjective.
        """
        return self.Re * (tau - 0.5) / (3.0 * self.U)

    def summary(self) -> str:
        """Multi-line human-readable derivation. What a CLI would print."""
        return "\n".join(
            [
                f"  physical: L = {self.l_phys:g} m, U = {self.u_phys:g} m/s, "
                f"nu = {self.nu_phys:g} m^2/s  ->  Re = {self.Re:.4g}",
                f"  grid:     {self.cells_per_length:g} cells across L, "
                f"dx = {self.dx:.6g} m/cell, dt = {self.dt:.6g} s/step",
                f"  lattice:  U = {self.U:.6g}, nu = {self.nu:.6g}, "
                f"tau = 0.5 + 3 U N / Re = {self.tau:.6f}",
                f"  checks:   peak |u| <= {self.peak_velocity_estimate():.4f} "
                f"(upper bound, {BLUFF_BODY_SPEEDUP} U for the bluffest body "
                f"measured; limit {U_LATTICE_MAX}, and the run's own measured "
                f"peak is what decides); "
                f"{self.stability_note()}",
            ]
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_physical(
        cls,
        *,
        u_phys: float,
        l_phys: float,
        nu_phys: float | None = None,
        re: float | None = None,
        cells_per_length: float | None = None,
        tau: float | None = None,
        u_lattice: float = U_LATTICE_DEFAULT,
        u_max: float = U_LATTICE_MAX,
        tau_floor: float = TAU_FLOOR,
    ) -> "LatticeUnits":
        """Describe a case in physics; get ``dx``, ``dt``, ``tau``, ``U``, ``Re``.

        The arithmetic is the module docstring's four steps, in order::

            Re  = u_phys * l_phys / nu_phys
            dx  = l_phys / N
            dt  = u_lattice * dx / u_phys
            nu  = nu_phys * dt / dx^2 = u_lattice * N / Re
            tau = 0.5 + 3 nu

        Exactly one of ``nu_phys`` / ``re`` describes the fluid, and exactly one
        of ``cells_per_length`` / ``tau`` describes the grid — pass the target
        ``tau`` and the **resolution is derived** from
        ``N = Re (tau - 0.5) / (3 U)``, which is the "the code derives
        resolution" half of ``old-Docs/TASKS1.md`` § T009.

        There is no ``nu`` (lattice) argument and no way to set one alongside
        ``tau``: ``CLAUDE.md`` constraint 2 says viscosity is not a free
        parameter, and one input path is how that is enforced rather than
        documented.

        Args:
            u_phys: characteristic speed, m/s. Must be positive.
            l_phys: characteristic length, m — the ``D`` of **D-019**, i.e. the
                cross-stream extent of the object. Must be positive.
            nu_phys: kinematic viscosity, m^2/s. Air at 20 C is 1.5e-5, water
                1.0e-6.
            re: Reynolds number, as an alternative to ``nu_phys``.
            cells_per_length: resolution ``N`` — cells across ``l_phys``.
            tau: target relaxation time, as an alternative to
                ``cells_per_length``.
            u_lattice: the lattice velocity to represent ``u_phys`` with. The
                one genuinely free choice; see :data:`U_LATTICE_DEFAULT`.
            u_max: constraint-3 ceiling, exposed only so a test can move it.
            tau_floor: the floor described in the module docstring.

        Returns:
            A frozen :class:`LatticeUnits`.

        Raises:
            ValueError: if the fluid or the grid is described twice or not at
                all; if any input is non-positive; if ``u_lattice`` is at or
                above ``u_max`` (constraint 3); or if the implied ``tau`` is at
                or below ``tau_floor``. The last two name the offending quantity
                and the resolution that would fix it.
        """
        if (nu_phys is None) == (re is None):
            raise ValueError(
                "describe the fluid exactly once: pass nu_phys (m^2/s) or re, "
                f"not both and not neither (got nu_phys={nu_phys}, re={re})"
            )
        if (cells_per_length is None) == (tau is None):
            raise ValueError(
                "describe the grid exactly once: pass cells_per_length or a "
                f"target tau, not both and not neither (got "
                f"cells_per_length={cells_per_length}, tau={tau})"
            )
        if u_phys <= 0.0:
            raise ValueError(f"u_phys must be positive, got {u_phys}")
        if l_phys <= 0.0:
            raise ValueError(f"l_phys must be positive, got {l_phys}")
        if u_lattice <= 0.0:
            raise ValueError(f"u_lattice must be positive, got {u_lattice}")

        if nu_phys is not None:
            if nu_phys <= 0.0:
                raise ValueError(f"nu_phys must be positive, got {nu_phys}")
            reynolds = u_phys * l_phys / nu_phys
        else:
            assert re is not None
            if re <= 0.0:
                raise ValueError(f"re must be positive, got {re}")
            reynolds = float(re)
            nu_phys = u_phys * l_phys / reynolds

        # Constraint 3 first: the ceiling is on a quantity the caller passed,
        # so it can be reported before anything is derived from it.
        if u_lattice >= u_max:
            raise ValueError(
                f"lattice velocity U = {u_lattice} is at or above the ceiling "
                f"of {u_max} (CLAUDE.md constraint 3: compressibility error "
                f"scales as Mach squared). The flow also accelerates around a "
                f"body — measured peak = {BLUFF_BODY_SPEEDUP} U on a square "
                f"cylinder — so the free stream needs headroom. Use "
                f"u_lattice <= {u_max / BLUFF_BODY_SPEEDUP:.3f} and buy the "
                f"tau back with resolution instead."
            )

        if cells_per_length is not None:
            if cells_per_length <= 0.0:
                raise ValueError(
                    f"cells_per_length must be positive, got {cells_per_length}"
                )
            n_cells = float(cells_per_length)
            tau_value = 0.5 + 3.0 * u_lattice * n_cells / reynolds
        else:
            assert tau is not None
            if tau <= 0.5:
                raise ValueError(
                    f"target tau = {tau} is at or below 0.5, where the lattice "
                    "viscosity is zero or negative (CLAUDE.md constraint 2: "
                    "nu = (tau - 0.5) / 3)."
                )
            tau_value = float(tau)
            n_cells = reynolds * (tau_value - 0.5) / (3.0 * u_lattice)

        if tau_value <= tau_floor:
            need = reynolds * (tau_floor - 0.5) / (3.0 * u_lattice)
            raise ValueError(
                f"tau = {tau_value:.4f} is at or below the floor of "
                f"{tau_floor} for Re = {reynolds:.4g}, U = {u_lattice}, "
                f"N = {n_cells:.4g} cells across L. tau -> 0.5 means nu -> 0 "
                f"and the sim blows up (CLAUDE.md constraint 2, DOCS/IDEA2.md "
                f"§ Stability). Use cells_per_length >= {math.ceil(need):d}, or "
                f"raise u_lattice (subject to the {u_max} ceiling). Note the "
                f"floor is a floor on nonsense, not a stability guarantee: a "
                f"disc at tau = 0.5330 still produced nan by step 1500 "
                f"(D-029), so a bluff body wants >= 0.54."
            )

        dx = l_phys / n_cells
        dt = u_lattice * dx / u_phys

        return cls(
            dx=dx,
            dt=dt,
            tau=tau_value,
            U=float(u_lattice),
            Re=reynolds,
            cells_per_length=n_cells,
            u_phys=float(u_phys),
            l_phys=float(l_phys),
            nu_phys=float(nu_phys),
        )
