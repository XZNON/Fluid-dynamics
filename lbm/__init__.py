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
    inlet_profile,
    inlet_velocity,
    moving_wall,
    outlet_zero_gradient,
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
    from_png,
    from_svg,
    min_thickness,
    polygon,
    rectangle,
    regular_polygon,
    strip_solid_border,
)
from lbm.probe import (
    BoundaryLinks,
    boundary_links,
    forces,
    residual,
    strouhal,
    vorticity,
)
from lbm.render import (
    COOLWARM,
    NAN_RGB,
    LiveSink,
    colormap,
    render,
)
from lbm.runner import (
    NullSink,
    RingBuffer,
    RunStats,
    Sim,
    SimConfig,
    Sink,
    load_checkpoint,
    run,
    save_checkpoint,
    steps_per_frame,
)
from lbm.units import (
    BLUFF_BODY_SPEEDUP,
    TAU_FLOOR,
    U_LATTICE_DEFAULT,
    U_LATTICE_MAX,
    LatticeUnits,
)

__all__ = [
    "BLUFF_BODY_SPEEDUP",
    "COOLWARM",
    "CS2",
    "E",
    "NAN_RGB",
    "OPP",
    "TAU_FLOOR",
    "U_LATTICE_DEFAULT",
    "U_LATTICE_MAX",
    "W",
    "BoundaryLinks",
    "LatticeUnits",
    "LiveSink",
    "MaskWarning",
    "NullSink",
    "RingBuffer",
    "RunStats",
    "Sim",
    "SimConfig",
    "Sink",
    "apply_body_force",
    "bounce_back",
    "boundary_links",
    "bounding_box",
    "channel_walls",
    "check_mask",
    "circle",
    "collide",
    "colormap",
    "equilibrium",
    "force_velocity_shift",
    "forces",
    "from_png",
    "from_svg",
    "inlet_profile",
    "inlet_velocity",
    "load_checkpoint",
    "macroscopic",
    "min_thickness",
    "moving_wall",
    "nu_from_tau",
    "outlet_zero_gradient",
    "polygon",
    "rectangle",
    "regular_polygon",
    "render",
    "residual",
    "run",
    "save_checkpoint",
    "steps_per_frame",
    "stream",
    "strip_solid_border",
    "strouhal",
    "vorticity",
]
