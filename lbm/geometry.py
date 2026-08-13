"""Geometry: primitives to one boolean mask, plus the mask sanity checks.

Implements ``DOCS/IDEA2.md`` § "Geometry from a mask". The entire geometry
interface of the solver is one boolean array, ``solid``, shape ``(ny, nx)``,
index order ``(y, x)`` matching the trailing axes of ``f`` (``CLAUDE.md``
constraint 4). Nothing here is hot — it runs once at setup — so functions
return fresh arrays rather than taking preallocated outputs (``old-Docs/STATE1.md``
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

Prior work (``old-Docs/STATE1.md`` § Decisions, **D-004**): the vertex handling and
the even-odd point-in-polygon test of ``Navier-Fluid-Equation/polygonsDemo.py``
and ``panels.py`` are reimplemented here rather than imported. That directory is
potential flow, is read-only, and never becomes a dependency of ``lbm``.
"""

from __future__ import annotations

import re as _re
import warnings
from pathlib import Path

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
    "from_png",
    "from_svg",
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

    Wall-offset convention (``old-Docs/STATE1.md`` § Decisions, **D-009**)
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
       reflects into the wake. ``old-Docs/PLAN1.md`` § Risks names this and blockage
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


# --------------------------------------------------------------------------
# Image sources — DOCS/IDEA2.md § Geometry from a mask, sources 2 and 3
# --------------------------------------------------------------------------


def _fit_to_grid(
    coverage: NDArray[np.float32], ny: int, nx: int, fit: str
) -> NDArray[np.float32]:
    """Place an ``(h, w)`` coverage field on an ``(ny, nx)`` grid.

    Args:
        coverage: solid fraction per source pixel, already resampled to the size
            implied by ``fit``.
        ny: rows of the target grid.
        nx: columns of the target grid.
        fit: ``"stretch"`` (already the grid size) or ``"contain"`` (centred,
            padded with fluid).

    Returns:
        ``(ny, nx)`` ``float32`` coverage.
    """
    if fit == "stretch":
        return coverage
    out = np.zeros((ny, nx), dtype=np.float32)
    h, w = coverage.shape
    y0 = (ny - h) // 2
    x0 = (nx - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = coverage
    return out


def from_png(
    path: str | Path,
    shape: tuple[int, int],
    *,
    threshold: float = 0.5,
    invert: bool = False,
    fit: str = "stretch",
    flip_y: bool = True,
    check: bool = True,
    inlet_axis: str = "x",
    strict: bool = False,
    verbose: bool = True,
) -> NDArray[np.bool_]:
    """Mask from an image file, shape ``(ny, nx)``, ``bool``.

    ``DOCS/IDEA2.md`` § Geometry from a mask, **source 2**: "PNG — load,
    threshold alpha or luminance, resize to grid".

    Which channel decides
    ---------------------
    **Alpha, when the image has one and it is not uniformly opaque** — a cut-out
    PNG of a wing is the case this exists for, and there the shape is exactly
    the opaque region whatever colour it was drawn in. Otherwise **luminance**,
    with *dark* meaning solid (ink on white paper). ``invert`` swaps the sense
    of whichever channel was chosen; the channel actually used is printed when
    ``verbose``.

    Resampling
    ----------
    The chosen channel is resampled with a **box filter** (area average) and
    thresholded afterwards, never the other way round. Area-averaging then
    thresholding at 0.5 preserves the solid *area* of a downscaled shape to
    within a boundary cell, which is what makes the committed test image's
    solid-cell count predictable to 2%; nearest-neighbour would lose or gain
    whole features depending on where the grid landed.

    Orientation
    -----------
    Image row 0 is the **top** of the picture, while the solver's ``y``
    increases upward and :class:`lbm.render.LiveSink` draws row 0 at the bottom
    (``flip_y`` there). ``flip_y=True`` here — the default — mirrors the image
    vertically on load so a wing loaded from a PNG appears the right way up in
    the live window.

    Constraint 12
    -------------
    :func:`check_mask` runs **automatically** (``check=True``). A downscaled PNG
    is the most likely source of a one-cell-thin wall in this whole project:
    the trailing edge of an aerofoil at 200 cells of chord is a couple of cells
    thick, it leaks through bounce-back, and the resulting flow looks entirely
    plausible. ``min_thickness`` (**D-017**) is what catches it, and its known
    limit applies here more than anywhere — a hairline *fused* to a thick body
    shares the body's connected component and is not reported.

    Args:
        path: image file. Anything Pillow can open; PNG is what is tested.
        shape: ``(ny, nx)`` target grid.
        threshold: solid when the resampled channel exceeds this, in ``[0, 1]``.
        invert: flip the solid/fluid sense of the chosen channel.
        fit: ``"stretch"`` resizes to exactly ``(ny, nx)``; ``"contain"``
            preserves the aspect ratio and pads the rest with fluid.
        flip_y: mirror vertically on load — see Orientation above.
        check: run :func:`check_mask` on the result.
        inlet_axis: passed to :func:`check_mask`.
        strict: passed to :func:`check_mask` — raise instead of warn.
        verbose: print the channel, the resize and the solid-cell count, and
            pass through to :func:`check_mask`.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on solid.

    Raises:
        ImportError: if Pillow is not installed, with the install command.
        FileNotFoundError: if ``path`` does not exist.
        ValueError: on a bad ``shape``, ``threshold`` or ``fit``.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is in myenv
        raise ImportError(
            "from_png needs Pillow. Install it into the project venv with:\n"
            "    myenv/Scripts/pip.exe install pillow\n"
            "and add a row to old-Docs/STATE1.md § Environment in the same session."
        ) from exc

    ny, nx = int(shape[0]), int(shape[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"shape must be at least 1x1, got {shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if fit not in ("stretch", "contain"):
        raise ValueError(f"fit must be 'stretch' or 'contain', got {fit!r}")

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"no such image: {src}")

    img = Image.open(src)
    img.load()

    alpha = None
    if "A" in img.getbands():
        alpha = np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0
    elif img.mode == "P" and "transparency" in img.info:
        alpha = np.asarray(img.convert("RGBA").getchannel("A"), dtype=np.float32) / 255.0

    if alpha is not None and float(alpha.min()) < 1.0:
        channel = alpha
        channel_name = "alpha (opaque = solid)"
    else:
        lum = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        channel = 1.0 - lum  # dark = solid
        channel_name = "luminance (dark = solid)"

    if invert:
        channel = 1.0 - channel
        channel_name += ", inverted"

    if flip_y:
        channel = channel[::-1, :]

    h, w = channel.shape
    if fit == "stretch":
        target = (ny, nx)
    else:
        scale = min(ny / h, nx / w)
        target = (max(int(round(h * scale)), 1), max(int(round(w * scale)), 1))

    # Box-filter resample through Pillow: float image, area average, no clipping.
    resized = np.asarray(
        Image.fromarray(channel.astype(np.float32), mode="F").resize(
            (target[1], target[0]), Image.Resampling.BOX
        ),
        dtype=np.float32,
    )
    coverage = _fit_to_grid(resized, ny, nx, fit)
    solid = coverage > threshold

    if verbose:
        print(
            f"from_png: {src.name} {w}x{h} px -> {nx}x{ny} cells ({fit}), "
            f"channel {channel_name}, threshold {threshold:g}"
            f"{', flipped in y' if flip_y else ''}. "
            f"{int(solid.sum())} solid cells "
            f"({solid.mean() * 100:.2f}% of the grid)."
        )

    if check:
        check_mask(
            solid, inlet_axis, strict=strict, verbose=verbose
        )
    return solid


# SVG parsing. Q-002 is answered here: no new dependency. The parser below
# handles the subset old-Docs/TASKS1.md § T009 asks for ("at least simple closed
# paths") and refuses everything else by name, pointing at cairosvg — which is
# what "fails obscurely" would otherwise look like.

_SVG_NUM = _re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_SVG_CMD = _re.compile(r"([MmLlHhVvCcQqZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")
_SVG_UNSUPPORTED = _re.compile(r"[AaSsTt]")

#: Segments per cubic/quadratic Bezier when flattening. Twelve is well past the
#: point where a curve's digitised outline stops changing at typical grid
#: resolutions, and flattening happens once at setup (constraint 6 is about the
#: step loop).
_SVG_BEZIER_SEGMENTS: int = 12


def _svg_dependency_error(what: str) -> ImportError:
    """The clear install message the acceptance criterion asks for."""
    return ImportError(
        f"lbm.geometry.from_svg cannot handle {what} with its built-in parser. "
        "Install a full rasteriser into the project venv:\n"
        "    myenv/Scripts/pip.exe install cairosvg\n"
        "and add a row to old-Docs/STATE1.md § Environment in the same session. "
        "Alternatively export the artwork to PNG and use from_png, which is the "
        "supported path (old-Docs/TASKS1.md § T009 Notes)."
    )


def _flatten_bezier(
    p0: tuple[float, float],
    ctrl: list[tuple[float, float]],
    p3: tuple[float, float],
    segments: int = _SVG_BEZIER_SEGMENTS,
) -> list[tuple[float, float]]:
    """Sample a quadratic or cubic Bezier, excluding its start point."""
    pts: list[tuple[float, float]] = []
    for k in range(1, segments + 1):
        t = k / segments
        s = 1.0 - t
        if len(ctrl) == 2:  # cubic
            c1, c2 = ctrl
            x = s**3 * p0[0] + 3 * s * s * t * c1[0] + 3 * s * t * t * c2[0] + t**3 * p3[0]
            y = s**3 * p0[1] + 3 * s * s * t * c1[1] + 3 * s * t * t * c2[1] + t**3 * p3[1]
        else:  # quadratic
            (c1,) = ctrl
            x = s * s * p0[0] + 2 * s * t * c1[0] + t * t * p3[0]
            y = s * s * p0[1] + 2 * s * t * c1[1] + t * t * p3[1]
        pts.append((x, y))
    return pts


def _parse_path_d(d: str) -> list[list[tuple[float, float]]]:
    """Flatten an SVG ``d`` attribute into subpaths of ``(x, y)`` points.

    Supports ``M/m L/l H/h V/v C/c Q/q Z/z`` — the "simple closed paths" of the
    acceptance criterion, with cubic and quadratic curves flattened to line
    segments. Arcs (``A``) and the smooth shorthands (``S``, ``T``) raise.

    Args:
        d: the attribute text.

    Returns:
        One list of points per subpath, in user coordinates. Subpaths are closed
        implicitly — the fill of an open path is the fill of the closed one.

    Raises:
        ImportError: on an unsupported command, naming it.
        ValueError: on a malformed ``d``.
    """
    bad = _SVG_UNSUPPORTED.search(d)
    if bad:
        raise _svg_dependency_error(
            f"path command {bad.group(0)!r} (arcs and smooth-curve shorthands)"
        )

    tokens: list[str | float] = []
    for m in _SVG_CMD.finditer(d):
        tokens.append(m.group(1) if m.group(1) else float(m.group(2)))

    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    pos = (0.0, 0.0)
    start = (0.0, 0.0)
    cmd: str | None = None
    i = 0

    def take(n: int) -> list[float]:
        nonlocal i
        if i + n > len(tokens) or any(isinstance(t, str) for t in tokens[i : i + n]):
            raise ValueError(f"malformed SVG path: {cmd!r} wants {n} numbers")
        vals = [float(t) for t in tokens[i : i + n]]  # type: ignore[arg-type]
        i += n
        return vals

    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            cmd = tok
            i += 1
            if cmd in ("Z", "z"):
                if current:
                    subpaths.append(current)
                    current = []
                pos = start
                continue
        elif cmd is None:
            raise ValueError("malformed SVG path: numbers before any command")
        elif cmd in ("M", "m"):
            # A repeated coordinate pair after M is an implicit lineto.
            cmd = "L" if cmd == "M" else "l"

        rel = cmd.islower()
        base = pos if rel else (0.0, 0.0)

        if cmd in ("M", "m"):
            x, y = take(2)
            if current:
                subpaths.append(current)
            pos = (base[0] + x, base[1] + y)
            start = pos
            current = [pos]
        elif cmd in ("L", "l"):
            x, y = take(2)
            pos = (base[0] + x, base[1] + y)
            current.append(pos)
        elif cmd in ("H", "h"):
            (x,) = take(1)
            pos = (base[0] + x, pos[1])
            current.append(pos)
        elif cmd in ("V", "v"):
            (y,) = take(1)
            pos = (pos[0], base[1] + y)
            current.append(pos)
        elif cmd in ("C", "c"):
            x1, y1, x2, y2, x, y = take(6)
            c1 = (base[0] + x1, base[1] + y1)
            c2 = (base[0] + x2, base[1] + y2)
            end = (base[0] + x, base[1] + y)
            current.extend(_flatten_bezier(pos, [c1, c2], end))
            pos = end
        elif cmd in ("Q", "q"):
            x1, y1, x, y = take(4)
            c1 = (base[0] + x1, base[1] + y1)
            end = (base[0] + x, base[1] + y)
            current.extend(_flatten_bezier(pos, [c1], end))
            pos = end
        else:  # pragma: no cover - the regex above admits nothing else
            raise ValueError(f"malformed SVG path: unknown command {cmd!r}")

    if current:
        subpaths.append(current)
    return [sp for sp in subpaths if len(sp) >= 3]


def _parse_svg(text: str) -> tuple[list[list[tuple[float, float]]], tuple[float, float, float, float] | None]:
    """Subpaths and ``viewBox`` from SVG source text.

    Reads ``<path d=...>`` and ``<polygon points=...>`` elements. A ``transform``
    attribute on either raises rather than being silently ignored, because an
    ignored transform produces a mask that is the wrong shape *and* looks
    deliberate.
    """
    for tag in ("path", "polygon"):
        for m in _re.finditer(rf"<{tag}\b[^>]*>", text):
            if "transform" in m.group(0):
                raise _svg_dependency_error("elements with a 'transform' attribute")

    subpaths: list[list[tuple[float, float]]] = []
    for m in _re.finditer(r"<path\b[^>]*\bd\s*=\s*([\"'])(.*?)\1", text, _re.S):
        subpaths.extend(_parse_path_d(m.group(2)))
    for m in _re.finditer(r"<polygon\b[^>]*\bpoints\s*=\s*([\"'])(.*?)\1", text, _re.S):
        nums = [float(v) for v in _SVG_NUM.findall(m.group(2))]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(pts) >= 3:
            subpaths.append(pts)

    view = None
    vb = _re.search(r"\bviewBox\s*=\s*([\"'])(.*?)\1", text, _re.S)
    if vb:
        nums = [float(v) for v in _SVG_NUM.findall(vb.group(2))]
        if len(nums) == 4 and nums[2] > 0 and nums[3] > 0:
            view = (nums[0], nums[1], nums[2], nums[3])
    return subpaths, view


def from_svg(
    path: str | Path,
    shape: tuple[int, int],
    *,
    margin: float = 0.0,
    fit: bool = True,
    flip_y: bool = True,
    check: bool = True,
    inlet_axis: str = "x",
    strict: bool = False,
    verbose: bool = True,
) -> NDArray[np.bool_]:
    """Mask from an SVG file, shape ``(ny, nx)``, ``bool``.

    ``DOCS/IDEA2.md`` § Geometry from a mask, **source 3**: "SVG path —
    rasterise".

    Q-002, answered
    ---------------
    No new dependency. Simple closed paths are parsed here — ``M/L/H/V/C/Q/Z``,
    absolute and relative, curves flattened to :data:`_SVG_BEZIER_SEGMENTS`
    segments — and filled with :func:`polygon`, which is the even-odd test T004
    already had. Anything outside that subset (arcs, the smooth shorthands,
    ``transform`` attributes, strokes, text, embedded images) raises an
    :class:`ImportError` naming the feature and giving the ``cairosvg`` install
    line. That is the acceptance criterion's "clear install message rather than
    failing obscurely", and it is deliberately an error and not a silent partial
    render: a silently dropped ``transform`` gives a plausible mask of the wrong
    shape, which is this project's stated main failure mode.

    Multiple subpaths are combined with the **even-odd** rule (``^=``), so a
    donut drawn as two rings has a hole. SVG's default fill rule is nonzero and
    the two differ only for self-intersecting or nested-same-direction outlines;
    that is a documented limit, not an oversight.

    Args:
        path: ``.svg`` file.
        shape: ``(ny, nx)`` target grid.
        margin: clear cells to leave around the artwork when fitting.
        fit: scale the artwork (aspect preserved) to fill the grid inside
            ``margin``. ``False`` uses the SVG's own coordinates as cells.
        flip_y: SVG's ``y`` axis points **down** and the solver's points up;
            ``True`` mirrors so the artwork appears upright.
        check: run :func:`check_mask` on the result.
        inlet_axis: passed to :func:`check_mask`.
        strict: passed to :func:`check_mask`.
        verbose: print what was parsed and the solid-cell count.

    Returns:
        ``(ny, nx)`` bool array, ``True`` on solid.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ImportError: on an SVG feature the built-in parser refuses, with the
            install command for a full rasteriser.
        ValueError: on a bad ``shape`` or an SVG with no fillable path.
    """
    ny, nx = int(shape[0]), int(shape[1])
    if ny < 1 or nx < 1:
        raise ValueError(f"shape must be at least 1x1, got {shape}")

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"no such SVG: {src}")

    subpaths, view = _parse_svg(src.read_text(encoding="utf-8", errors="replace"))
    if not subpaths:
        raise ValueError(
            f"{src.name} has no fillable <path d=...> or <polygon points=...>: "
            "nothing to rasterise. Strokes, text and embedded images are not "
            "filled shapes — export to PNG and use from_png instead."
        )

    pts = np.concatenate([np.asarray(sp, dtype=np.float64) for sp in subpaths], axis=0)
    if view is not None:
        x0, y0, vw, vh = view
    else:
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        vw = max(float(pts[:, 0].max()) - x0, 1e-12)
        vh = max(float(pts[:, 1].max()) - y0, 1e-12)

    if fit:
        scale = min((nx - 1 - 2 * margin) / vw, (ny - 1 - 2 * margin) / vh)
        if scale <= 0.0:
            raise ValueError(
                f"margin {margin} leaves no room in a {ny}x{nx} grid for the "
                f"artwork's {vw:g} x {vh:g} viewBox"
            )
        ox = (nx - 1 - vw * scale) / 2.0
        oy = (ny - 1 - vh * scale) / 2.0
    else:
        scale, ox, oy = 1.0, 0.0, 0.0

    solid = np.zeros((ny, nx), dtype=bool)
    for sp in subpaths:
        v = np.asarray(sp, dtype=np.float64)
        gx = (v[:, 0] - x0) * scale + ox
        gy = (v[:, 1] - y0) * scale + oy
        if flip_y:
            gy = (ny - 1) - gy
        solid ^= polygon(ny, nx, np.stack([gx, gy], axis=1))

    if verbose:
        print(
            f"from_svg: {src.name} -> {nx}x{ny} cells, {len(subpaths)} subpath(s), "
            f"{'viewBox' if view is not None else 'point bbox'} "
            f"{vw:g} x {vh:g}, scale {scale:.4g} cells/unit"
            f"{', flipped in y' if flip_y else ''}. "
            f"{int(solid.sum())} solid cells "
            f"({solid.mean() * 100:.2f}% of the grid)."
        )

    if check:
        check_mask(solid, inlet_axis, strict=strict, verbose=verbose)
    return solid
