"""lbm — D2Q9 lattice Boltzmann solver in pure NumPy (Phase 0).

Everything inside this package works in **lattice units**. Physical units are
converted at the boundary by ``lbm.units`` (T009) and never reach the solver.

See ``DOCS/IDEA2.md`` for the full Phase 0 specification and ``CLAUDE.md`` for
the hard constraints.

Phase 1 adds a **backend seam** (``DOCS/IDEA3.md`` § What Phase 1 is,
concretely, T101) and no new physics: :class:`lbm.backends.Backend` is the set
of kernels :class:`lbm.runner.Sim` calls, ``lbm.backends.numpy_backend`` is the
reference implementation over the functions below, and ``SimConfig.backend``
picks one by name.
"""

from lbm.backends import (
    Backend,
    BackendUnavailableError,
    available_backends,
    get_backend,
)
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
    CS_SMAG_LITERATURE,
    E,
    OPP,
    SMAG_Q_COEFF,
    W,
    collide,
    collide_stream,
    equilibrium,
    macroscopic,
    nu_from_tau,
    smagorinsky_omega,
    smagorinsky_tau_eff,
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
    eddy_viscosity,
    forces,
    residual,
    strouhal,
    vorticity,
)
from lbm.record import (
    HeadlessSink,
    RecordSink,
    TeeSink,
    check_ffmpeg,
    frame_count,
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
    "CS_SMAG_LITERATURE",
    "E",
    "NAN_RGB",
    "OPP",
    "SMAG_Q_COEFF",
    "TAU_FLOOR",
    "U_LATTICE_DEFAULT",
    "U_LATTICE_MAX",
    "W",
    "Backend",
    "BackendUnavailableError",
    "BoundaryLinks",
    "HeadlessSink",
    "LatticeUnits",
    "LiveSink",
    "MaskWarning",
    "NullSink",
    "RecordSink",
    "RingBuffer",
    "RunStats",
    "Sim",
    "SimConfig",
    "Sink",
    "TeeSink",
    "apply_body_force",
    "available_backends",
    "bounce_back",
    "boundary_links",
    "bounding_box",
    "channel_walls",
    "check_ffmpeg",
    "check_mask",
    "circle",
    "collide",
    "collide_stream",
    "colormap",
    "eddy_viscosity",
    "equilibrium",
    "force_velocity_shift",
    "forces",
    "frame_count",
    "from_png",
    "from_svg",
    "get_backend",
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
    "smagorinsky_omega",
    "smagorinsky_tau_eff",
    "steps_per_frame",
    "stream",
    "strip_solid_border",
    "strouhal",
    "vorticity",
]
