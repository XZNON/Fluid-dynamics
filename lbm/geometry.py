"""Geometry: primitives to one boolean mask, plus the mask sanity checks.

Implements ``DOCS/IDEA2.md`` § "Geometry from a mask". The entire geometry
interface of the solver is one boolean array, ``solid``, shape ``(ny, nx)``,
index order ``(y, x)`` matching the trailing axes of ``f`` (``CLAUDE.md``
constraint 4). Nothing here is hot — it runs once at setup — so functions
return fresh arrays rather than taking preallocated outputs (``DOCS/STATE1.md``
§ Decisions, **D-006**).

Coordinate convention
---------------------
Cell ``solid[i, j]`` is the cell whose **centre** sits at ``x = j``, ``y = i``,
both in lattice cells. Primitives are evaluated at those centres: a cell is
solid when its centre is inside the shape. ``x`` therefore indexes the second
axis and ``y`` the first, everywhere in this module.

The three rules of ``DOCS/IDEA2.md`` § Geometry from a mask are enforced by
:func:`check_mask`, not by the primitives:

1. solid at least 3 cells thick (thinner leaks through bounce-back —
   the "flow through the object" row of § Stability),
2. at least 8 characteristic lengths of clear domain downstream,
3. blockage ratio under ~10%.

Prior work (``DOCS/STATE1.md`` § Decisions, **D-004**): the vertex handling and
the even-odd point-in-polygon test of ``Navier-Fluid-Equation/polygonsDemo.py``
and ``panels.py`` are reimplemented here rather than imported. That directory is
potential flow, is read-only, and never becomes a dependency of ``lbm``.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "MaskWarning",
    "circle",
    "rectangle",
    "polygon",
    "regular_polygon",
    "channel_walls",
    "bounding_box",
    "min_thickness",
    "check_mask",
]


class MaskWarning(UserWarning):
    """Raised as a warning by :func:`check_mask` when a mask will mislead.

    A distinct category so callers (and tests) can filter on it rather than on
    every ``UserWarning`` in the process.
    """


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def _grid(ny: int, nx: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Cell-centre coordinates, broadcastable to ``(ny, nx)``.

    Returns:
        ``(X, Y)`` of shapes ``(1, nx)`` and ``(ny, 1)``, ``float32``. ``X[0, j]
        == j`` and ``Y[i, 0] == i`` — the centre convention of the module
        docstring.
    """
    if ny < 1 or nx < 1:
        raise ValueError(f"grid must be at least 1x1, got ny={ny}, nx={nx}")
    x = np.arange(nx, dtype=np.float32)[None, :]
    y = np.arange(ny, dtype=np.float32)[:, None]
    return x, y


def circle(ny: int, nx: int, cx: float, cy: float, radius: float) -> NDArray[np.bool_]:
    """Filled disc, shape ``(ny, nx)``, ``bool``.

    ``DOCS/IDEA2.md`` § Geometry from a mask, source 1 (primitives).

    Args:
        ny: rows (``y`` extent).
        nx: columns (``x`` extent).
        cx: centre ``x`` in cells; may be fractional (a half-cell offset is the
            documented cure for a cylinder that refuses to shed, ``T007``).
        cy: centre ``y`` in cells.
        radius: radius in cells. Cells whose centre is within ``radius`` are
            solid, so the disc's diameter in cells is about ``2*radius + 1``.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on solid.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    x, y = _grid(ny, nx)
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2


def rectangle(
    ny: int, nx: int, x0: float, y0: float, x1: float, y1: float
) -> NDArray[np.bool_]:
    """Axis-aligned filled rectangle, shape ``(ny, nx)``, ``bool``.

    ``DOCS/IDEA2.md`` § Geometry from a mask, source 1 (primitives).

    Bounds are **inclusive** and in cell-centre coordinates: a cell is solid
    when ``x0 <= x <= x1`` and ``y0 <= y <= y1``. Integer bounds therefore give
    ``x1 - x0 + 1`` solid columns. The bounds are sorted, so a caller that
    passes them the other way round gets the same rectangle rather than an
    empty mask.

    Args:
        ny: rows.
        nx: columns.
        x0: one ``x`` bound in cells.
        y0: one ``y`` bound in cells.
        x1: the other ``x`` bound in cells.
        y1: the other ``y`` bound in cells.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on solid.
    """
    x, y = _grid(ny, nx)
    xlo, xhi = (x0, x1) if x0 <= x1 else (x1, x0)
    ylo, yhi = (y0, y1) if y0 <= y1 else (y1, y0)
    return (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)


def polygon(ny: int, nx: int, vertices: ArrayLike) -> NDArray[np.bool_]:
    """Filled polygon from vertices, shape ``(ny, nx)``, ``bool``.

    ``DOCS/IDEA2.md`` § Geometry from a mask, source 1 (primitives). Reuses the
    even-odd (crossing-number) test that ``Navier-Fluid-Equation/`` reaches for
    via ``matplotlib.path.Path.contains_points``, reimplemented here so ``lbm``
    has no dependency on that directory (**D-004**) and no dependency on
    matplotlib.

    The even-odd rule handles **concave** polygons for free: a horizontal ray
    from each cell centre towards ``-x`` crosses the boundary an odd number of
    times exactly when the centre is inside, however the outline wanders. The
    loop is over *edges* (a handful) with a whole-grid NumPy op inside; there is
    no vectorisation over edges, per ``CLAUDE.md`` constraint 6 — geometry is
    setup code and is not what needs to be fast.

    Args:
        ny: rows.
        nx: columns.
        vertices: ``(n, 2)`` array-like of ``(x, y)`` in cells, in order, at
            least 3 of them. The outline is closed automatically; a repeated
            final vertex is harmless.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on solid.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f"vertices must be (n, 2) of (x, y), got shape {verts.shape}")
    if verts.shape[0] < 3:
        raise ValueError(f"a polygon needs at least 3 vertices, got {verts.shape[0]}")

    x, y = _grid(ny, nx)
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    inside = np.zeros((ny, nx), dtype=bool)
    n = verts.shape[0]
    for k in range(n):
        x1, y1 = verts[k]
        x2, y2 = verts[(k + 1) % n]
        if y1 == y2:
            # A horizontal edge is never crossed by a horizontal ray; skipping
            # it also keeps the division below finite.
            continue
        # Rows the edge spans, half-open in y so a shared vertex is counted once.
        spans = (y1 > y) != (y2 > y)
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= spans & (x < x_cross)
    return inside


def regular_polygon(
    ny: int,
    nx: int,
    nsides: int,
    cx: float,
    cy: float,
    radius: float,
    rotate: float = 0.0,
) -> NDArray[np.bool_]:
    """Regular ``nsides``-gon, shape ``(ny, nx)``, ``bool``.

    Convenience wrapper over :func:`polygon`. The vertex generator is the one
    from ``Navier-Fluid-Equation/panels.py::polygon``, reimplemented (**D-004**):
    ``nsides`` points on a circle of ``radius``, starting at angle ``rotate``.
    Rung 4's square cylinder (``T008``) is ``nsides=4, rotate=pi/4``.

    Args:
        ny: rows.
        nx: columns.
        nsides: number of sides, at least 3.
        cx: centre ``x`` in cells.
        cy: centre ``y`` in cells.
        radius: circumradius in cells.
        rotate: rotation in radians.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on solid.
    """
    if nsides < 3:
        raise ValueError(f"a polygon needs at least 3 sides, got {nsides}")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    a = np.linspace(0.0, 2.0 * np.pi, nsides, endpoint=False) + rotate
    verts = np.stack([cx + radius * np.cos(a), cy + radius * np.sin(a)], axis=1)
    return polygon(ny, nx, verts)


def channel_walls(ny: int, nx: int, thickness: int = 1) -> NDArray[np.bool_]:
    """No-slip top and bottom rows, shape ``(ny, nx)``, ``bool``.

    Generalises the inline mask that ``validate/poiseuille.py`` builds. Compose
    with an object using ``|``::

        solid = channel_walls(ny, nx) | circle(ny, nx, cx, cy, r)

    Wall-offset convention (``DOCS/STATE1.md`` § Decisions, **D-009**)
    ------------------------------------------------------------------
    Bounce-back puts the no-slip plane **halfway between the last fluid node and
    the first solid node**, not on a node. With ``thickness = 1`` the solid rows
    are ``y = 0`` and ``y = ny - 1``, the fluid rows are ``y = 1 .. ny - 2``, the
    wall planes are ``y = 0.5`` and ``y = ny - 1.5``, and the channel height is
    therefore::

        H = ny - 2 * thickness

    Measured, not argued: Rung 1 prints the L2 error under all three rival
    conventions on every run — halfway 0.365%, the two rivals 14.8% and 12.7%.

    One cell is enough for a **domain border**: constraint 12's 3-cell rule is
    about immersed objects, which have an interior for fluid to leak into. A
    border has none, and :func:`check_mask` exempts fully-solid border rows and
    columns from the thickness check for that reason.

    Args:
        ny: rows, including the wall rows.
        nx: columns.
        thickness: wall rows at each of top and bottom.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on the wall rows.
    """
    if thickness < 1:
        raise ValueError(f"thickness must be at least 1, got {thickness}")
    if 2 * thickness >= ny:
        raise ValueError(
            f"walls of thickness {thickness} leave no fluid rows in ny={ny}; "
            f"need ny > {2 * thickness}"
        )
    solid = np.zeros((ny, nx), dtype=bool)
    solid[:thickness, :] = True
    solid[ny - thickness :, :] = True
    return solid


# --------------------------------------------------------------------------
# Mask measurement
# --------------------------------------------------------------------------


def bounding_box(mask: NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    """Tight bounding box of the ``True`` cells.

    Args:
        mask: ``(ny, nx)`` bool array.

    Returns:
        ``(y0, y1, x0, x1)``, inclusive on both ends, or ``None`` when the mask
        is empty. Heights and widths are therefore ``y1 - y0 + 1`` and
        ``x1 - x0 + 1``.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _shift(mask: NDArray, dy: int, dx: int) -> NDArray:
    """``mask`` translated by ``(dy, dx)``, with zeros shifted in from outside.

    For a bool mask that means the outside of the domain counts as **fluid**,
    which is the conservative choice for the thickness check: a structure
    clipped by the domain edge is reported as thin rather than silently assumed
    to continue.
    """
    out = np.zeros_like(mask)
    ys_dst = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
    ys_src = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
    xs_dst = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
    xs_src = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def _erode(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """3x3 binary erosion: keep a cell only if all 8 neighbours are solid too."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out &= _shift(mask, dy, dx)
    return out


def _wall_distance(mask: NDArray[np.bool_], cap: int = 64) -> NDArray[np.int32]:
    """Chebyshev distance from each solid cell to the nearest fluid cell.

    ``1`` on a solid cell that touches fluid (including diagonally), ``2`` on a
    cell whose whole 3x3 neighbourhood is solid, and so on: repeated 3x3
    erosion is exactly a Chebyshev distance transform. ``0`` on fluid.

    Args:
        mask: ``(ny, nx)`` bool array.
        cap: stop at this distance. Nothing above 2 changes a decision.

    Returns:
        ``(ny, nx)`` ``int32`` distance map.
    """
    dist = np.zeros(mask.shape, dtype=np.int32)
    layer = mask
    r = 1
    while layer.any() and r <= cap:
        dist[layer] = r
        layer = _erode(layer)
        r += 1
    return dist


def _label(mask: NDArray[np.bool_], max_passes: int = 4096) -> NDArray[np.int32]:
    """Label 8-connected components of ``mask``, ``0`` on fluid.

    Plain max-propagation: each solid cell starts with a unique id and takes the
    largest id in its 3x3 neighbourhood until nothing changes. That is one pass
    per cell of component diameter, which is fine for setup-time geometry and
    keeps the module dependency-free (no ``scipy``).

    Args:
        mask: ``(ny, nx)`` bool array.
        max_passes: safety stop. On a component longer than this the labels are
            still valid as a *partition refinement* — several ids for one
            structure — which only makes :func:`min_thickness` more cautious.

    Returns:
        ``(ny, nx)`` ``int32`` label array.
    """
    ids = np.where(mask, np.arange(1, mask.size + 1).reshape(mask.shape), 0)
    ids = ids.astype(np.int32)
    for _ in range(max_passes):
        grown = ids
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    grown = np.maximum(grown, _shift(ids, dy, dx))
        grown = np.where(mask, grown, 0).astype(np.int32)
        if np.array_equal(grown, ids):
            break
        ids = grown
    return ids


def strip_solid_border(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """``mask`` with any fully-solid border rows and columns peeled off.

    Domain walls — the channel's top and bottom rows, the cavity's border — are
    one cell thick on purpose (**D-009**, and ``validate/cavity.py``). They are
    not what constraint 12's 3-cell rule is about: bounce-back leaks *through* a
    thin obstacle into fluid on the far side, and a border has no far side. So
    the checks in :func:`check_mask` are applied to what remains after the solid
    frame is peeled away — the immersed object.

    Only *entirely* solid edge rows/columns are peeled, one layer at a time, so
    an object that merely touches the edge stays in and stays checked.

    Args:
        mask: ``(ny, nx)`` bool array.

    Returns:
        A new ``(ny, nx)`` bool array, ``False`` on the peeled frame.
    """
    ny, nx = mask.shape
    y0, y1, x0, x1 = 0, ny - 1, 0, nx - 1
    changed = True
    while changed and y0 <= y1 and x0 <= x1:
        changed = False
        if mask[y0, x0 : x1 + 1].all():
            y0 += 1
            changed = True
        if y0 <= y1 and mask[y1, x0 : x1 + 1].all():
            y1 -= 1
            changed = True
        if y0 <= y1 and mask[y0 : y1 + 1, x0].all():
            x0 += 1
            changed = True
        if x0 <= x1 and y0 <= y1 and mask[y0 : y1 + 1, x1].all():
            x1 -= 1
            changed = True

    out = np.zeros_like(mask)
    if y0 <= y1 and x0 <= x1:
        out[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return out


def min_thickness(mask: NDArray[np.bool_]) -> int:
    """Thickness of the **thinnest connected solid structure**, in cells.

    ``DOCS/IDEA2.md`` § Geometry from a mask, rule 1: "solid must be at least 3
    cells thick anywhere. Thinner and fluid leaks through bounce-back."

    Method. Each 8-connected component is measured by the deepest point it
    contains: with ``d`` the Chebyshev distance from a solid cell to the nearest
    fluid cell (:func:`_wall_distance`), the component's thickness is
    ``2 * max(d) - 1`` — the side of the largest fully-solid square that fits
    inside it. The result is the smallest such value over all components, so one
    thin plate beside a fat cylinder still warns.

    Two alternatives were tried first and are wrong here, which is worth
    recording:

    * **Run lengths.** The topmost cell of a disc has a vertical run of 1, so a
      perfectly healthy cylinder would report thickness 1 and warn on every
      Rung 3 run.
    * **Per-cell opening** (does a fully-solid 3x3 square cover every solid
      cell?). The pole of a digital disc fails it too — its neighbours one row
      down and one column across are already outside the circle — so again the
      cylinder warns. Digitised curvature always produces locally thin boundary
      cells; they are not what leaks.

    Known limit: a thin appendage fused to a thick body shares its component and
    is not reported separately. Every primitive here produces convex or nearly
    convex blobs, and a composed mask is a union of them, so this has not bitten
    yet — but a future PNG mask (T009) with hairline features could hide behind
    it.

    The estimate is odd-valued and rounds down: a 4-cell-thick block reports 3.
    That is the right direction to round for a warning.

    Args:
        mask: ``(ny, nx)`` bool array.

    Returns:
        Thickness in cells of the thinnest structure, or ``0`` for an empty mask.
    """
    if not mask.any():
        return 0
    dist = _wall_distance(mask)
    labels = _label(mask)
    thinnest = None
    for label in np.unique(labels[labels > 0]):
        deepest = int(dist[labels == label].max())
        thickness = 2 * deepest - 1
        if thinnest is None or thickness < thinnest:
            thinnest = thickness
    assert thinnest is not None  # mask.any() above guarantees one component
    return thinnest


# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------


def check_mask(
    solid: NDArray[np.bool_],
    inlet_axis: str = "x",
    *,
    min_thickness_cells: int = 3,
    min_downstream_lengths: float = 8.0,
    max_blockage: float = 0.10,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Check a mask against the three rules of ``DOCS/IDEA2.md`` § Geometry.

    The three failures this exists to prevent, from § Geometry from a mask and
    the "flow through the object" row of § Stability (``CLAUDE.md`` constraint
    12):

    1. **Thickness** — solid thinner than ``min_thickness_cells`` leaks through
       bounce-back and the flow passes straight through the object.
    2. **Downstream space** — with less than ``min_downstream_lengths``
       characteristic lengths of clear domain behind the object, the outlet
       reflects into the wake. ``DOCS/PLAN1.md`` § Risks names this and blockage
       as the usual reason a cylinder shows no shedding at all.
    3. **Blockage** — a cross-stream extent above ``max_blockage`` of the
       domain lets the walls distort the answer.

    Domain walls are exempt from all three: fully-solid border rows and columns
    are peeled off first by :func:`strip_solid_border`, so what is measured is
    the immersed object. A mask with no immersed object passes silently.

    Characteristic length
    ---------------------
    Derived from the object's bounding box, never passed in: it is the
    **cross-stream** extent, i.e. the bounding-box height for flow along ``x``.
    That is the ``D`` of "8 diameters downstream" and of the blockage ratio, and
    it is printed with the rest of the geometry when ``verbose``.

    Args:
        solid: ``(ny, nx)`` bool array, index order ``(y, x)`` (constraint 4).
        inlet_axis: ``"x"`` for flow along ``+x`` — inlet at ``x = 0``, outlet
            at ``x = nx - 1``. ``"y"`` for flow along ``+y``.
        min_thickness_cells: rule 1's threshold, 3 cells per the spec.
        min_downstream_lengths: rule 2's threshold in characteristic lengths.
        max_blockage: rule 3's threshold as a fraction.
        strict: raise ``ValueError`` with every message instead of warning.
        verbose: print the measured geometry — characteristic length, bounding
            box, downstream distance, blockage, thickness.

    Returns:
        The warning messages, in rule order; empty when the mask is clean.

    Raises:
        ValueError: if ``solid`` is not a 2-D bool array, if ``inlet_axis`` is
            not ``"x"`` or ``"y"``, or — when ``strict`` — if any rule fails.
    """
    solid = np.asarray(solid)
    if solid.ndim != 2:
        raise ValueError(f"solid must be (ny, nx), got shape {solid.shape}")
    if solid.dtype != np.bool_:
        raise ValueError(f"solid must be bool, got dtype {solid.dtype}")
    if inlet_axis not in ("x", "y"):
        raise ValueError(f"inlet_axis must be 'x' or 'y', got {inlet_axis!r}")

    ny, nx = solid.shape
    obj = strip_solid_border(solid)
    box = bounding_box(obj)

    if box is None:
        if verbose:
            print(
                f"check_mask: {ny}x{nx} grid, no immersed object "
                f"(solid cells: {int(solid.sum())}, all of them domain border) "
                "— nothing to check."
            )
        return []

    y0, y1, x0, x1 = box
    height = y1 - y0 + 1  # cells spanned in y
    width = x1 - x0 + 1  # cells spanned in x

    if inlet_axis == "x":
        length = height  # cross-stream extent = characteristic length D
        streamwise = width
        downstream = nx - 1 - x1
        domain_cross = ny
    else:
        length = width
        streamwise = height
        downstream = ny - 1 - y1
        domain_cross = nx

    # The cross-stream domain that matters is the fluid gap, so discount solid
    # border layers: they are wall, not flow.
    border_cross = int(
        (solid.all(axis=1)).sum() if inlet_axis == "x" else (solid.all(axis=0)).sum()
    )
    domain_cross_fluid = max(domain_cross - border_cross, 1)

    thickness = min_thickness(obj)
    blockage = length / domain_cross_fluid
    downstream_lengths = downstream / length if length else float("inf")

    if verbose:
        print(
            f"check_mask: {ny}x{nx} grid, flow along {inlet_axis}. "
            f"Object bbox y {y0}..{y1}, x {x0}..{x1} "
            f"({height} x {width} cells, {int(obj.sum())} solid). "
            f"Characteristic length D = {length} cells (cross-stream extent), "
            f"streamwise extent {streamwise}. "
            f"Downstream {downstream} cells = {downstream_lengths:.2f} D. "
            f"Blockage {blockage:.1%} of a {domain_cross_fluid}-cell fluid span. "
            f"Min solid thickness {thickness} cells."
        )

    messages: list[str] = []

    if thickness < min_thickness_cells:
        messages.append(
            f"mask is only {thickness} cells thick at its thinnest "
            f"(minimum {min_thickness_cells}): fluid leaks through bounce-back "
            "and the flow will pass through the object "
            "(DOCS/IDEA2.md § Stability, 'flow through the object'). "
            "Thicken the solid or raise the resolution."
        )

    if downstream_lengths < min_downstream_lengths:
        trailing_edge = x1 if inlet_axis == "x" else y1
        needed = int(np.ceil(trailing_edge + 1 + min_downstream_lengths * length))
        messages.append(
            f"object is {downstream} cells = {downstream_lengths:.2f} "
            f"characteristic lengths from the outlet "
            f"(minimum {min_downstream_lengths:g} D, D = {length} cells): "
            "the outlet will reflect into the wake. "
            f"Extend the domain to at least {needed} cells along {inlet_axis}."
        )

    if blockage > max_blockage:
        messages.append(
            f"blockage ratio is {blockage:.1%} "
            f"(D = {length} cells in a {domain_cross_fluid}-cell fluid span, "
            f"maximum {max_blockage:.0%}): the walls will distort the answer. "
            f"Widen the domain to at least "
            f"{int(np.ceil(length / max_blockage)) + border_cross} cells across."
        )

    if messages:
        if strict:
            raise ValueError(
                "check_mask failed with strict=True:\n  - " + "\n  - ".join(messages)
            )
        for msg in messages:
            warnings.warn(msg, MaskWarning, stacklevel=2)

    return messages
