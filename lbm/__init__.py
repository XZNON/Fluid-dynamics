"""lbm — D2Q9 lattice Boltzmann solver in pure NumPy (Phase 0).

Everything inside this package works in **lattice units**. Physical units are
converted at the boundary by ``lbm.units`` (T009) and never reach the solver.

See ``DOCS/IDEA2.md`` for the full Phase 0 specification and ``CLAUDE.md`` for
the hard constraints.
"""

from lbm.boundary import (
    apply_body_force,
    bounce_back,
    force_velocity_shift,
    moving_wall,
)
from lbm.core import (
    CS2,
    E,
    OPP,
    W,
    collide,
    equilibrium,
    macroscopic,
    nu_from_tau,
    stream,
)
from lbm.geometry import (
    MaskWarning,
    bounding_box,
    channel_walls,
    check_mask,
    circle,
    min_thickness,
    polygon,
    rectangle,
    regular_polygon,
    strip_solid_border,
)

__all__ = [
    "CS2",
    "E",
    "OPP",
    "W",
    "MaskWarning",
    "apply_body_force",
    "bounce_back",
    "bounding_box",
    "channel_walls",
    "check_mask",
    "circle",
    "collide",
    "equilibrium",
    "force_velocity_shift",
    "macroscopic",
    "min_thickness",
    "moving_wall",
    "nu_from_tau",
    "polygon",
    "rectangle",
    "regular_polygon",
    "stream",
    "strip_solid_border",
]
