"""Geometry preparation — a real picture becomes a mask the solver can run.

``DOCS/IDEA3.md`` § The five things Phase 1 must get right, **item 3**: *"Real
shapes are not convex blobs."* ``lbm/geometry.py`` already thresholds, resamples
and *measures* (``check_mask``, ``min_thickness``); what it does not do is
**fix** anything. This module is the judgement and the repair on top of it, and
it does not fork either function (Phase 1 constraint 15 — ``flow/`` may import
``lbm/``, never the reverse).

Three verdicts, and nothing in between:

``"ok"``
    The picture rasterised into a mask that satisfies ``CLAUDE.md``
    constraint 12 as drawn. Nothing was changed.
``"repaired"``
    Something was changed, and **every change is named** in
    :attr:`Prepared.actions`. A repair is a substitution of the user's geometry
    (constraint 16, **D-045**), so a silent one is a defect.
``"refused"``
    No repair in this module makes the picture runnable.
    :attr:`Prepared.reason` says what is wrong and :attr:`Prepared.fix` says
    what would fix it — and that fix is *executable*, via :func:`apply_fix`,
    which is how Rung C keeps it a testable claim rather than a sentence
    (**D-063**).

Resolution (**D-040**)
----------------------
``cells_across`` is cells across the **body**, not across the picture. A repair
changes the body's extent — thickening a one-cell plate makes it three — so the
resolution is **re-derived** rather than adjusted: every candidate raster size
is rasterised, repaired and *then* measured, and the one whose measured body
matches the request is the one returned. Nothing is cropped or padded after the
fact.

The Q-102 metric
----------------
:func:`thin_branch_depth` closes the limit **D-017** documented — a thin
appendage *fused* to a thick body shares the body's connected component and is
therefore invisible to ``min_thickness``. See that function's docstring for the
measurement that justifies the threshold, including the disc false-alarm test
that killed D-017's own two predecessors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from flow.diagnose import EXAMPLE_MASK
from lbm.geometry import (
    MaskWarning,
    _label,
    _shift,
    _wall_distance,
    bounding_box,
    check_mask,
    from_png,
    from_svg,
    min_thickness,
    strip_solid_border,
)

__all__ = [
    "Prepared",
    "Fix",
    "REPAIRS",
    "VERDICTS",
    "MIN_BODY_CELLS",
    "RESOLUTION_TOL",
    "MIN_THICKNESS_CELLS",
    "THIN_BRANCH_DEPTH",
    "SPECK_AREA_FRACTION",
    "apply_fix",
    "measure",
    "prepare",
    "thin_branch_depth",
]


# ---------------------------------------------------------------------------
# Constants — every one cited or measured (DOCS/PLAN2.md § Risks: "a pile of
# tuned constants nobody can defend" is the named failure mode of this layer)
# ---------------------------------------------------------------------------

#: The repairs, in the order they are applied. Each is individually switchable
#: through ``prepare(..., repair=...)`` and individually reported in
#: :attr:`Prepared.actions`.
REPAIRS: tuple[str, ...] = (
    "fill_holes",
    "drop_specks",
    "largest_component",
    "thicken",
)

#: The three verdicts. Nothing else is ever returned.
VERDICTS: tuple[str, ...] = ("ok", "repaired", "refused")

#: Smallest body this module will hand on, measured across the flow
#: (the ``D`` of **D-019**). Below this there is no shape left to resolve: a
#: 7-cell disc cannot be 3 cells thick *and* have a boundary, and every
#: coefficient derived from ``D`` becomes a rounding artefact.
MIN_BODY_CELLS: int = 8

#: ``CLAUDE.md`` constraint 12 / ``DOCS/IDEA2.md`` § Geometry, rule 1.
MIN_THICKNESS_CELLS: int = 3

#: A component smaller than this fraction of the largest one is a speck — a
#: stray pixel from a JPEG artefact or a signature in the corner — not a body.
#: One percent of the body's *area* is two orders of magnitude below anything
#: that changes a force coefficient, and comfortably above the few-cell debris
#: a threshold leaves behind.
SPECK_AREA_FRACTION: float = 0.01

#: A speck is never larger than this many cells regardless of the fraction
#: above, so a small body cannot have a *proportionally* small neighbour
#: dropped out from under it.
SPECK_MAX_CELLS: int = 12

#: The Q-102 threshold: a thin branch is reported when
#: :func:`thin_branch_depth` reaches this. **Measured this session** — see that
#: function's docstring for the table. A plain disc never exceeds 2.
THIN_BRANCH_DEPTH: int = 3

#: Dilation passes the ``thicken`` repair is allowed. One pass turns a
#: one-cell feature into three; four is well past what any single feature
#: needs, and a shape still failing after four is refused rather than fattened
#: indefinitely into a different shape (``DOCS/PLAN2.md`` § Risks).
MAX_THICKEN_PASSES: int = 4

#: **D-040**, as a gate rather than an aspiration: the measured body must be
#: the requested size to within this many cells. A picture that cannot deliver
#: it — a plate two pixels wide cannot be rasterised at any box that also makes
#: the body the requested size — is **refused** with the size it *can* deliver,
#: because quietly returning a different resolution is precisely the silent
#: substitution constraint 16 forbids, and ``tau`` is computed from the number
#: that comes back.
RESOLUTION_TOL: int = 1

#: How far either side of the bisection's answer the resolution search looks
#: (**D-040**). The measured body grows with the raster box, so a bisection
#: finds the smallest box that reaches the requested size; the scan either side
#: of it is what catches the case where a repair pushed the body a cell past
#: the request and one box smaller lands exactly.
RESOLUTION_SCAN: int = 3

#: Ceiling on the raster box the search will grow to, in rows. A source that
#: cannot produce the requested body inside this is a source with almost
#: nothing solid in it, and is refused as such rather than rasterised for ever.
MAX_BOX_ROWS: int = 4000


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fix:
    """What would make a refused picture runnable — as a runnable thing.

    Constraint 14: *every refusal names a fix, and the fix is machine-checked*.
    :func:`apply_fix` turns one of these back into keyword arguments for
    :func:`prepare`, and Rung C runs the result — so a fix that does not fix
    its own case is a failing test, exactly as **D-063** established for
    ``flow/autoconfig.py``'s suggestions.

    Attributes:
        change: ``"resolution"`` — ask for a bigger body — or ``"picture"``,
            which the tool cannot supply because it cannot invent the user's
            geometry. A ``"picture"`` fix is executed against
            :data:`flow.diagnose.EXAMPLE_MASK`, the same worked example Rung D
            substitutes.
        value: the replacement — an ``int`` body size for ``"resolution"``, a
            description of the needed picture for ``"picture"``.
        note: one plain sentence a non-specialist can act on.
    """

    change: str
    value: Any
    note: str


@dataclass(frozen=True)
class Prepared:
    """The outcome of preparing one picture — an output record, never built by a caller.

    A frozen output record, so ``tests/test_flow_package.py``'s constraint-13
    scan exempts its generated constructor and nothing else (**D-060**).

    Attributes:
        mask: ``(ny, nx)`` bool array, ``True`` on solid — the **body**,
            cropped to its own bounding box, ready to hand to
            :func:`flow.autoconfig.plan` and to be placed in a domain. Empty
            ``(0, 0)`` when the verdict is ``"refused"``.
        verdict: one of :data:`VERDICTS`.
        actions: one line per repair actually applied, each prefixed with its
            key from :data:`REPAIRS` (``"fill_holes: ..."``), so the disclosure
            is machine-readable as well as human-readable. Empty when nothing
            was changed.
        properties: the measured geometry — see :func:`measure`. Present for
            every verdict, because a refusal that cannot say what it measured
            is not diagnosable.
        reason: what is wrong, for ``"refused"``; ``None`` otherwise.
        fix: what would fix it, for ``"refused"``; ``None`` otherwise.
        warnings: anything ``lbm.geometry.check_mask`` still says about the
            body placed in a nominal domain. Empty for ``"ok"`` and
            ``"repaired"`` — that is an acceptance criterion, asserted by
            Rung C.
    """

    mask: NDArray[np.bool_]
    verdict: str
    actions: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    fix: Fix | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        """``True`` when the mask can be handed to the solver."""
        return self.verdict != "refused"


# ---------------------------------------------------------------------------
# Morphology — 3x3 (Chebyshev) neighbourhoods, matching D-017's metric
# ---------------------------------------------------------------------------
#
# `lbm.geometry` already holds the labelling, the Chebyshev distance transform
# and the shift these are built from, and the T107 contract is explicit that
# `flow/` "adds judgement and repair on top and does not fork them" — so they
# are imported, private names and all, rather than reimplemented here. What is
# new below is only what `lbm/` has no reason to own: repair.


def _dilate8(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """One 3x3 binary dilation — the dual of ``lbm.geometry._erode``."""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                out |= _shift(mask, dy, dx)
    return out


def _dilate4(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """One 4-neighbour dilation.

    Fluid connectivity is the complement of solid connectivity: solid is
    8-connected (``lbm.geometry._label``, **D-017**), so fluid must be
    4-connected or a diagonal touch would count as solid *and* let fluid
    through it.
    """
    out = mask.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        out |= _shift(mask, dy, dx)
    return out


def _component_sizes(labels: NDArray[np.int32]) -> dict[int, int]:
    """``{label: cell count}`` for every non-zero label."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts)}


def _enclosed_fluid(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Fluid cells with no 4-connected path to the outside of the picture.

    The grid is padded by one fluid cell first, so a shape that touches the
    picture's edge still has an outside to escape to — without the pad, a body
    drawn to the border would have its entire surroundings counted as an
    interior hole and filled.
    """
    ny, nx = mask.shape
    padded = np.zeros((ny + 2, nx + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    fluid = ~padded

    outside = np.zeros_like(fluid)
    outside[0, :] = fluid[0, :]
    outside[-1, :] = fluid[-1, :]
    outside[:, 0] = fluid[:, 0]
    outside[:, -1] = fluid[:, -1]
    while True:
        grown = _dilate4(outside) & fluid
        if np.array_equal(grown, outside):
            break
        outside = grown
    return (fluid & ~outside)[1:-1, 1:-1]


# ---------------------------------------------------------------------------
# The Q-102 metric
# ---------------------------------------------------------------------------


def _thin_shell(
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """Geodesic depth of the one-cell shell, measured outward from the core.

    Returns:
        ``(depth, coreless)`` — ``depth`` is ``0`` everywhere except on shell
        cells reached from a core cell, where it counts steps; ``coreless`` is
        the shell cells no core cell reaches at all, i.e. structures with no
        fully-solid 3x3 anywhere in them (``min_thickness`` already sees those).
    """
    depth = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return depth, np.zeros_like(mask)

    dist = _wall_distance(mask)
    core = dist >= 2  # a cell whose entire 3x3 neighbourhood is solid
    shell = mask & ~core
    if not shell.any():
        return depth, np.zeros_like(mask)

    front = _dilate8(core) & shell
    step = 1
    while front.any() and step < mask.size:
        depth[front] = step
        step += 1
        front = _dilate8(front) & shell & (depth == 0)
    return depth, shell & (depth == 0)


def thin_branch_depth(mask: NDArray[np.bool_]) -> int:
    """How far a sub-3-cell feature reaches out of the solid it hangs off.

    **This is the answer to Q-102**, and it closes the limit **D-017**
    documented in as many words: *"a thin appendage fused to a thick body
    shares its component and is not reported"*. ``min_thickness`` measures a
    whole 8-connected component by its deepest point, so a hairline welded to a
    disc is invisible to it — the disc's depth is what gets reported.

    Method. With ``d`` the Chebyshev distance from a solid cell to the nearest
    fluid cell (``lbm.geometry._wall_distance``, the same transform D-017 uses):

    * a **core** cell has ``d >= 2`` — its whole 3x3 neighbourhood is solid,
      so locally the shape is at least three cells thick;
    * the **shell** is everything else, ``d == 1``;
    * the depth is the 8-connected geodesic distance, *within the shell*, from
      the nearest core cell.

    A healthy convex body has a shell exactly one cell deep: every boundary
    cell touches a core cell. A hairline has no core anywhere along it, so its
    cells are 2, 3, 4 ... steps out, one per cell of length. That is the
    separation, and it is a **measurement**, not an argument — D-017 records
    that its own two predecessors (run lengths, per-cell 3x3 opening) were
    written first and both false-alarmed on a plain disc, so the disc is the
    bar any replacement has to clear:

    ======================================================  =====
    shape family                                            depth
    ======================================================  =====
    discs, radius 3 ... 60, four sub-cell offsets each      **2**
    squares, side 3 ... 60                                      1
    straight plates 3+ cells wide                               1
    Rung 3's cylinder body                                      2
    Rung 4's square body / arbitrary convex polygon body      1 / 1
    1- or 2-cell hairline of length ``L`` fused to a disc  ``L+1``
    ======================================================  =====

    So :data:`THIN_BRANCH_DEPTH` is 3: no disc reaches it at any radius or
    sub-cell offset, and a fused hairline reaches it at a length of two cells.

    The cost, measured in the same sweep and recorded rather than hidden: the
    metric also fires on shapes whose thinness is *intended* — a rotated
    triangle's apex (4), a 40x3 ellipse's tip (5), a three-cell-stroke ring of
    radius 60 (6). Every one of those genuinely has a one-cell-thick region and
    therefore genuinely leaks through bounce-back, so they are true positives
    by constraint 12's own rule — but a user will experience them as their
    shape being blunted, which is exactly why the ``thicken`` repair reports
    itself in :attr:`Prepared.actions` instead of happening quietly.

    Args:
        mask: ``(ny, nx)`` bool array.

    Returns:
        The deepest shell cell's distance from the core, in cells. ``0`` for an
        empty mask, for a mask with no shell at all, and for a structure with
        no core at all — the last of those is ``min_thickness``'s business, not
        this function's.
    """
    depth, _coreless = _thin_shell(np.asarray(mask, dtype=bool))
    # A zero-size mask has no maximum to take, and `prepare` produces exactly
    # that shape for a refused picture — so the empty case is reachable, not
    # theoretical.
    return int(depth.max()) if depth.size else 0


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure(mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Every number Rung C compares against ``expectations.json``.

    The verdict is the headline known answer, but a verdict alone would pass
    while the mask underneath it quietly became a different shape
    (``DOCS/PLAN2.md`` § Risks names that at this task by name), so the rung
    asserts the *measured properties* too.

    Args:
        mask: ``(ny, nx)`` bool array, ``True`` on solid.

    Returns:
        ``body_cells_across`` (the cross-stream extent, the ``D`` of
        **D-019**), ``body_cells_along``, ``solid_cells``, ``min_thickness``
        (**D-017**), ``thin_branch_depth`` (**Q-102**), ``components``
        (8-connected), ``holes`` (enclosed fluid regions), ``fill_fraction``
        (solid cells over bounding-box area) and ``aspect_ratio``
        (streamwise over cross-stream). An empty mask reports zeros.
    """
    mask = np.asarray(mask, dtype=bool)
    box = bounding_box(mask)
    if box is None:
        return {
            "body_cells_across": 0,
            "body_cells_along": 0,
            "solid_cells": 0,
            "min_thickness": 0,
            "thin_branch_depth": 0,
            "components": 0,
            "holes": 0,
            "fill_fraction": 0.0,
            "aspect_ratio": 0.0,
        }

    y0, y1, x0, x1 = box
    across = y1 - y0 + 1
    along = x1 - x0 + 1
    labels = _label(mask)
    holes = _enclosed_fluid(mask)
    n_holes = len(_component_sizes(_label(holes))) if holes.any() else 0

    return {
        "body_cells_across": int(across),
        "body_cells_along": int(along),
        "solid_cells": int(mask.sum()),
        "min_thickness": int(min_thickness(mask)),
        "thin_branch_depth": int(thin_branch_depth(mask)),
        "components": len(_component_sizes(labels)),
        "holes": int(n_holes),
        "fill_fraction": float(mask.sum()) / float(across * along),
        "aspect_ratio": float(along) / float(across),
    }


# ---------------------------------------------------------------------------
# Repairs — each one switchable, each one reported
# ---------------------------------------------------------------------------


def _repair_fill_holes(
    mask: NDArray[np.bool_], actions: list[str]
) -> NDArray[np.bool_]:
    """Fill fluid pockets with no path to the outside of the picture.

    A donut drawn as an outline is the case: the enclosed fluid is not
    reachable by the flow, contributes nothing but a trapped vortex, and its
    inner wall is a second bounce-back surface the user did not ask for.
    """
    holes = _enclosed_fluid(mask)
    count = int(holes.sum())
    if count == 0:
        return mask
    n = len(_component_sizes(_label(holes)))
    actions.append(
        f"fill_holes: filled {n} interior hole{'s' if n != 1 else ''} "
        f"({count} cells) that the flow could never reach"
    )
    return mask | holes


def _repair_drop_specks(
    mask: NDArray[np.bool_], actions: list[str]
) -> NDArray[np.bool_]:
    """Drop 8-connected components below the stated speck area."""
    labels = _label(mask)
    sizes = _component_sizes(labels)
    if len(sizes) <= 1:
        return mask
    largest = max(sizes.values())
    limit = min(int(round(SPECK_AREA_FRACTION * largest)), SPECK_MAX_CELLS)
    if limit < 1:
        return mask
    doomed = [i for i, n in sizes.items() if n <= limit]
    if not doomed:
        return mask
    out = mask.copy()
    dropped = 0
    for i in doomed:
        sel = labels == i
        dropped += int(sel.sum())
        out &= ~sel
    actions.append(
        f"drop_specks: dropped {len(doomed)} detached piece"
        f"{'s' if len(doomed) != 1 else ''} of {limit} cells or fewer "
        f"({dropped} cells in total, against a body of {largest})"
    )
    return out


def _repair_largest_component(
    mask: NDArray[np.bool_], actions: list[str]
) -> NDArray[np.bool_]:
    """Keep only the biggest 8-connected component."""
    labels = _label(mask)
    sizes = _component_sizes(labels)
    if len(sizes) <= 1:
        return mask
    keep = max(sizes, key=lambda i: sizes[i])
    out = labels == keep
    actions.append(
        f"largest_component: kept the largest of {len(sizes)} separate pieces "
        f"({sizes[keep]} cells) and discarded "
        f"{int(mask.sum()) - sizes[keep]} cells of the rest"
    )
    return out


def _thin_targets(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The cells the ``thicken`` repair may grow, and only those.

    A component is a target only if it actually fails — ``min_thickness``
    below :data:`MIN_THICKNESS_CELLS`, or a shell reaching
    :data:`THIN_BRANCH_DEPTH`. Within a failing component the targets are the
    coreless cells (a structure with no fully-solid 3x3 anywhere) plus the
    shell cells two or more steps out, which is the branch itself rather than
    the ordinary one-cell rim every convex body has. A healthy disc has no
    failing component, so it is never touched — which is the whole reason this
    is gated per component instead of globally.
    """
    depth, coreless = _thin_shell(mask)
    labels = _label(mask)
    targets = np.zeros_like(mask)
    for i in _component_sizes(labels):
        sel = labels == i
        thin = int(min_thickness(sel)) < MIN_THICKNESS_CELLS
        branchy = int(depth[sel].max()) >= THIN_BRANCH_DEPTH
        if thin or branchy:
            targets |= sel & (coreless | (depth >= 2))
    return targets


def _repair_thicken(
    mask: NDArray[np.bool_], actions: list[str]
) -> NDArray[np.bool_]:
    """Grow sub-3-cell features until they stop leaking, and nothing else.

    ``CLAUDE.md`` constraint 12 / ``DOCS/IDEA2.md`` § Stability, *"flow through
    the object"*: solid thinner than three cells lets fluid through
    bounce-back, and the resulting flow is plausible and wrong. Growing is the
    only repair available — the picture cannot supply detail it does not have —
    so it is bounded (:data:`MAX_THICKEN_PASSES`), applied only to the cells
    :func:`_thin_targets` identifies, and reported.
    """
    out = mask
    passes = 0
    grown = 0
    while passes < MAX_THICKEN_PASSES:
        targets = _thin_targets(out)
        if not targets.any():
            break
        before = int(out.sum())
        out = out | _dilate8(targets)
        grown += int(out.sum()) - before
        passes += 1
    if passes == 0:
        return mask
    actions.append(
        f"thicken: grew the thinnest features over {passes} pass"
        f"{'es' if passes != 1 else ''} ({grown} cells added) so no solid is "
        f"under {MIN_THICKNESS_CELLS} cells thick — thinner than that, fluid "
        "leaks straight through the object"
    )
    return out


_REPAIR_FUNCS = {
    "fill_holes": _repair_fill_holes,
    "drop_specks": _repair_drop_specks,
    "largest_component": _repair_largest_component,
    "thicken": _repair_thicken,
}


def _normalise_repairs(repair: bool | Iterable[str]) -> tuple[str, ...]:
    """``repair=`` in any of its three forms, as an ordered tuple of keys."""
    if repair is True:
        return REPAIRS
    if repair is False:
        return ()
    keys = tuple(str(r) for r in repair)
    unknown = [k for k in keys if k not in REPAIRS]
    if unknown:
        raise ValueError(
            f"unknown repair{'s' if len(unknown) != 1 else ''} {unknown}; "
            f"flow.prepare knows {REPAIRS}"
        )
    return tuple(r for r in REPAIRS if r in keys)


# ---------------------------------------------------------------------------
# Rasterising a source at a chosen box size
# ---------------------------------------------------------------------------


def _source_aspect(source: str | Path | NDArray[np.bool_]) -> float:
    """Width over height of the source, so a box preserves its proportions."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".svg":
            return 1.0
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
        return float(w) / float(h)
    arr = np.asarray(source)
    return float(arr.shape[1]) / float(arr.shape[0])


def _source_rows(source: str | Path | NDArray[np.bool_]) -> int:
    """The source's own height, in pixels or cells — the natural search ceiling."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".svg":
            return 1000  # the parser is analytic; there is no native raster
        from PIL import Image

        with Image.open(path) as img:
            return int(img.size[1])
    return int(np.asarray(source).shape[0])


def _resample_bool(
    mask: NDArray[np.bool_], rows: int, cols: int
) -> NDArray[np.bool_]:
    """Box-filter a bool array to ``(rows, cols)`` and threshold at 0.5.

    The same rule ``lbm.geometry.from_png`` uses and for the same reason:
    area-average first, threshold second, so the solid *area* of a downscaled
    shape survives to within a boundary cell.
    """
    from PIL import Image

    src = np.ascontiguousarray(mask.astype(np.float32))
    resized = np.asarray(
        Image.fromarray(src, mode="F").resize((cols, rows), Image.Resampling.BOX),
        dtype=np.float32,
    )
    return resized > 0.5


def _rasterise(
    source: str | Path | NDArray[np.bool_], box_rows: int, aspect: float
) -> NDArray[np.bool_]:
    """The source at a given box height, as a full-grid bool mask.

    ``check=False`` deliberately: constraint 12 is checked once, at the end, on
    the body placed in a nominal domain, where blockage and downstream distance
    mean something (the reasoning ``lbm.runner._body_mask`` records).
    """
    rows = max(3, int(box_rows))
    cols = max(3, int(round(rows * aspect)))
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".svg":
            return from_svg(path, (rows, cols), check=False, verbose=False)
        return from_png(path, (rows, cols), check=False, verbose=False)
    return _resample_bool(np.asarray(source, dtype=bool), rows, cols)


def _crop(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The mask cropped to its own bounding box, or ``(0, 0)`` when empty."""
    box = bounding_box(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    y0, y1, x0, x1 = box
    return np.ascontiguousarray(mask[y0 : y1 + 1, x0 : x1 + 1])


# ---------------------------------------------------------------------------
# Constraint 12, checked where it means something
# ---------------------------------------------------------------------------


def _nominal_domain(body: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Place a body in a domain the way ``flow/autoconfig.py`` sizes one.

    ``check_mask``'s three rules are not all properties of the *picture*: the
    blockage ratio and the distance to the outlet are properties of the domain
    the body is placed in, and on a body cropped to its own bounding box the
    blockage is 100% by construction. So the check runs here, on the body
    inside the margins :data:`flow.autoconfig.SPAN_D` /
    ``UPSTREAM_D`` / ``DOWNSTREAM_D`` will actually give it (**D-059**) — which
    is the domain T108 will build. The thickness rule, the one that is a
    property of the picture, is unaffected by placement either way.
    """
    from flow import autoconfig as ac

    across, along = body.shape
    ny = max(int(round(ac.SPAN_D * across)), across + 2)
    nx = int(round((ac.UPSTREAM_D + ac.DOWNSTREAM_D) * across)) + along
    solid = np.zeros((ny, nx), dtype=bool)
    y0 = (ny - across) // 2
    x0 = int(round(ac.UPSTREAM_D * across))
    solid[y0 : y0 + across, x0 : x0 + along] = body
    return solid


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


#: Fluid cells kept around the solid before repairing. The repairs need room —
#: ``thicken`` grows outward, and ``_enclosed_fluid`` needs an outside to escape
#: to — but they do not need the picture's whitespace, and a 400x400 picture of
#: a 50-pixel body costs sixty times as much to label as the body does. Cropping
#: is what keeps Rung C seconds rather than minutes (**D-059** asks for that
#: explicitly).
_REPAIR_PAD: int = MAX_THICKEN_PASSES + 2


def _prepare_at(
    source: str | Path | NDArray[np.bool_],
    box_rows: int,
    aspect: float,
    repairs: Sequence[str],
) -> tuple[NDArray[np.bool_], list[str]]:
    """Rasterise at one box size, repair, crop. One candidate of the D-040 search.

    The repairs run on the solid's bounding box plus :data:`_REPAIR_PAD` cells
    of fluid, not on the whole picture. That is not an approximation: every
    repair here is local to the solid, the pad leaves room for the only one
    that grows outward, and the result is cropped to the body's own bounding
    box either way.
    """
    actions: list[str] = []
    grid = _rasterise(source, box_rows, aspect)
    box = bounding_box(grid)
    if box is None:
        return np.zeros((0, 0), dtype=bool), actions
    y0, y1, x0, x1 = box
    pad = _REPAIR_PAD
    work = np.zeros(
        (y1 - y0 + 1 + 2 * pad, x1 - x0 + 1 + 2 * pad), dtype=bool
    )
    work[pad : pad + y1 - y0 + 1, pad : pad + x1 - x0 + 1] = grid[
        y0 : y1 + 1, x0 : x1 + 1
    ]
    for key in repairs:
        work = _REPAIR_FUNCS[key](work, actions)
    return _crop(work), actions


def _search(
    source: str | Path | NDArray[np.bool_],
    aspect: float,
    repairs: Sequence[str],
    cells_across: int,
) -> tuple[NDArray[np.bool_], list[str], int]:
    """Find the raster box whose **repaired** body is ``cells_across`` across.

    **D-040**, done the only way that is honest: a repair changes the body's
    extent, so the box size that produces a ``cells_across``-cell body cannot
    be computed from the *unrepaired* raster. Every candidate is therefore a
    full rasterise-repair-measure cycle. The measured extent grows with the
    box, so this is a bisection for the smallest box that reaches the request,
    plus a short scan either side of it — a thickening pass can push the body
    one cell past the request, and then the box one smaller is the one that
    lands.

    Returns:
        ``(body, actions, box_rows)`` for the best candidate found. The caller
        decides whether "best" is close enough; this function never adjusts a
        body to fit.
    """
    seen: dict[int, tuple[NDArray[np.bool_], list[str]]] = {}

    def candidate(rows: int) -> tuple[NDArray[np.bool_], list[str]]:
        rows = max(3, int(rows))
        if rows not in seen:
            seen[rows] = _prepare_at(source, rows, aspect, repairs)
        return seen[rows]

    def extent(rows: int) -> int:
        body_, _ = candidate(rows)
        return int(body_.shape[0]) if body_.size else 0

    # The ceiling on the doubling is set by the source, not by a round number:
    # four times its own height is well past the point where rasterising finer
    # reveals anything it does not already contain, and eight times the request
    # covers a body that occupies a small fraction of a large picture. Without
    # a source-derived ceiling, proving an empty picture empty costs a
    # 4000-row raster.
    ceiling = min(MAX_BOX_ROWS, max(4 * _source_rows(source), 8 * cells_across))
    hi = max(cells_across, 8)
    while extent(hi) < cells_across and hi < ceiling:
        hi = min(hi * 2, ceiling)
    lo = 3
    if extent(hi) >= cells_across:
        while lo < hi:
            mid = (lo + hi) // 2
            if extent(mid) >= cells_across:
                hi = mid
            else:
                lo = mid + 1
    else:
        lo = hi  # nothing this side of the ceiling reaches the request

    best_rows = lo
    best_err = abs(extent(lo) - cells_across)
    for rows in range(lo - RESOLUTION_SCAN, lo + RESOLUTION_SCAN + 1):
        if rows < 3:
            continue
        err = abs(extent(rows) - cells_across)
        if err < best_err:
            best_rows, best_err = rows, err
    body, actions = candidate(best_rows)
    return body, actions, best_rows


def _reachable_size(
    source: str | Path | NDArray[np.bool_],
    aspect: float,
    repairs: Sequence[str],
    start: int,
) -> int:
    """A body size this picture can actually deliver, nearest ``start``.

    When the request is unreachable the refusal has to name a size that *is*
    reachable, or its fix would fail its own test (**D-063**). The size the
    search happened to land on is not automatically that: asking for it can
    land somewhere else again, because thickening a feature that only appears
    at a finer raster moves the answer. So this iterates to a fixed point —
    a size that, asked for, comes back.
    """
    target = max(int(start), MIN_BODY_CELLS)
    for _ in range(4):
        body, _actions, _rows = _search(source, aspect, repairs, target)
        got = int(body.shape[0]) if body.size else 0
        if got == 0:
            return target
        if abs(got - target) <= RESOLUTION_TOL:
            return target
        target = got
    return target


def prepare(
    source: str | Path | NDArray[np.bool_],
    cells_across: int,
    *,
    repair: bool | Iterable[str] = True,
    verbose: bool = False,
) -> Prepared:
    """Turn a picture into a body mask the solver can run — or refuse it.

    ``DOCS/IDEA3.md`` § The five things Phase 1 must get right, item 3.

    What happens, in order: the source is rasterised into a box, every enabled
    repair runs on the full grid, the result is cropped to the body's bounding
    box, and the body's cross-stream extent is compared with ``cells_across``.
    If it does not match, the **whole thing is done again** at a corrected box
    size — never patched afterwards, because ``tau`` is computed from the
    measured body and a body that was adjusted after measurement is a body
    whose ``tau`` is a fiction (**D-040**).

    Args:
        source: a ``.png`` (anything Pillow opens), a ``.svg`` (the subset
            **D-031** parses), or an ``(ny, nx)`` bool array.
        cells_across: how many cells the **body** should span across the flow —
            the ``D`` of **D-019**, not the size of the picture. This is the
            one number from the solver's side of the fence that may cross into
            a ``flow/`` signature, and it crosses in the D-040 sense (the T107
            contract).
        repair: ``True`` for every repair in :data:`REPAIRS`, ``False`` for
            none, or an iterable of keys for some. Order is always
            :data:`REPAIRS`, whatever order the keys arrive in.
        verbose: print the verdict, the actions and the measured properties.

    Returns:
        A :class:`Prepared`. Never raises for a bad *picture* — that is what
        ``verdict="refused"`` is for; it raises only for a bad *call*.

    Raises:
        ValueError: on a ``cells_across`` below 1, an unknown repair key, or a
            source array that is not 2-D bool.
        FileNotFoundError: if a path source does not exist.
        ImportError: from ``lbm.geometry.from_svg`` for an SVG feature its
            built-in parser refuses (**D-031**).
    """
    cells_across = int(cells_across)
    if cells_across < 1:
        raise ValueError(f"cells_across must be at least 1, got {cells_across}")
    repairs = _normalise_repairs(repair)

    if not isinstance(source, (str, Path)):
        arr = np.asarray(source)
        if arr.ndim != 2:
            raise ValueError(f"a source array must be (ny, nx), got {arr.shape}")
        if arr.dtype != np.bool_:
            raise ValueError(f"a source array must be bool, got {arr.dtype}")
    elif not Path(source).exists():
        raise FileNotFoundError(f"no such geometry file: {source}")

    aspect = _source_aspect(source)

    # --- D-040: re-derive the resolution, never adjust the result ----------
    body, actions, best_rows = _search(source, aspect, repairs, cells_across)

    props = measure(body)
    # What the picture looked like *before* any repair, at the same raster —
    # the evidence half of constraint 16. Without it the corpus could only
    # record that a shape came out clean, never that it went in broken, and
    # Q-102's whole claim is about what is *detected*.
    raw_body, _ = _prepare_at(source, best_rows, aspect, ())
    raw = measure(raw_body)
    for key in ("min_thickness", "thin_branch_depth", "components", "holes"):
        props[f"{key}_before"] = raw[key]

    # --- refusals ---------------------------------------------------------
    refusal = _refuse(source, body, props, cells_across)
    missed = abs(props["body_cells_across"] - cells_across) > RESOLUTION_TOL
    if refusal is None and missed:
        got = _reachable_size(source, aspect, repairs, props["body_cells_across"])
        refusal = (
            f"this picture cannot make a body {cells_across} cells across: its "
            f"thinnest feature disappears below that scale, and the nearest it "
            f"reaches is {got}",
            Fix(
                change="resolution",
                value=got,
                note=(
                    f"Ask for a body {got} cells across, which is what this "
                    "picture can actually resolve — or supply a larger picture."
                ),
            ),
        )
    if refusal is not None:
        reason, fix = refusal
        result = Prepared(
            mask=np.zeros((0, 0), dtype=bool),
            verdict="refused",
            actions=actions,
            properties=props,
            reason=reason,
            fix=fix,
        )
        if verbose:
            _print(result, cells_across)
        return result

    with warnings.catch_warnings():
        # This module *acts* on what check_mask finds — a warning here becomes
        # a refusal with a fix a line later — so re-raising it as a Python
        # warning would report the same defect twice, once uselessly.
        warnings.simplefilter("ignore", MaskWarning)
        warnings_ = check_mask(_nominal_domain(body), "x", verbose=False)
    if warnings_:
        reason, fix = _refuse_on_warnings(warnings_, props, cells_across)
        result = Prepared(
            mask=np.zeros((0, 0), dtype=bool),
            verdict="refused",
            actions=actions,
            properties=props,
            reason=reason,
            fix=fix,
            warnings=warnings_,
        )
        if verbose:
            _print(result, cells_across)
        return result

    result = Prepared(
        mask=body,
        verdict="repaired" if actions else "ok",
        actions=actions,
        properties=props,
    )
    if verbose:
        _print(result, cells_across)
    return result


def _refuse(
    source: str | Path | NDArray[np.bool_],
    body: NDArray[np.bool_],
    props: dict[str, Any],
    cells_across: int,
) -> tuple[str, Fix] | None:
    """The refusals that are decided from the picture, before ``check_mask``."""
    # Nothing solid at all.
    if body.size == 0 or props["solid_cells"] == 0:
        return (
            "there is no shape in this picture: after thresholding, every cell "
            "came out as fluid",
            Fix(
                change="picture",
                value="a picture with a solid shape in it",
                note=(
                    "Draw or export the body as a dark shape on a light "
                    "background — or, if the shape is the light part, load it "
                    "inverted."
                ),
            ),
        )

    # Everything solid: no fluid means no flow, whatever the resolution. The
    # body being a filled rectangle is a necessary condition, and checking it
    # first is what keeps the extra raster off every ordinary call.
    if props["fill_fraction"] < 1.0:
        return _refuse_small(props, cells_across)
    if not isinstance(source, (str, Path)):
        whole = np.asarray(source, dtype=bool)
    else:
        whole = _rasterise(source, max(cells_across, 8), _source_aspect(source))
    if strip_solid_border(whole).sum() == 0 and whole.all():
        return (
            "this picture is solid everywhere: there is no fluid for anything "
            "to flow through",
            Fix(
                change="picture",
                value="a picture whose shape is smaller than the picture",
                note=(
                    "Leave a margin of background around the body — the flow "
                    "needs somewhere to go."
                ),
            ),
        )

    return _refuse_small(props, cells_across)


def _refuse_small(
    props: dict[str, Any], cells_across: int
) -> tuple[str, Fix] | None:
    """The one refusal that is about size rather than about content."""
    if props["body_cells_across"] < MIN_BODY_CELLS:
        return (
            f"the body would be only {props['body_cells_across']} cells across, "
            f"and nothing under {MIN_BODY_CELLS} is a shape: it cannot be "
            f"{MIN_THICKNESS_CELLS} cells thick and still have an outline",
            Fix(
                change="resolution",
                value=MIN_BODY_CELLS,
                note=(
                    f"Ask for a body at least {MIN_BODY_CELLS} cells across "
                    f"(this run asked for {cells_across})."
                ),
            ),
        )
    return None


def _refuse_on_warnings(
    messages: list[str], props: dict[str, Any], cells_across: int
) -> tuple[str, Fix]:
    """A body that still fails ``check_mask`` after every enabled repair."""
    if props["min_thickness"] < MIN_THICKNESS_CELLS:
        return (
            f"the thinnest part of this shape is {props['min_thickness']} "
            f"cell(s) of solid, and under {MIN_THICKNESS_CELLS} the fluid "
            "leaks straight through it — the flow would pass through the body "
            "instead of round it",
            Fix(
                change="resolution",
                value=max(cells_across * 2, MIN_BODY_CELLS),
                note=(
                    "Ask for a body about twice as many cells across, so the "
                    "thin part is resolved — or redraw it thicker."
                ),
            ),
        )
    return (
        "this shape still fails a geometry check after every repair: "
        + "; ".join(messages),
        Fix(
            change="picture",
            value="a simpler, more solid shape",
            note="Redraw the body with fewer hairline features.",
        ),
    )


def apply_fix(
    fix: Fix,
    *,
    source: str | Path | NDArray[np.bool_],
    cells_across: int,
    repair: bool | Iterable[str] = True,
) -> dict[str, Any]:
    """The original request with this fix applied — ready for :func:`prepare`.

    What makes a refusal a *testable claim* rather than a sentence: Rung C
    calls ``prepare(**apply_fix(p.fix, ...))`` and asserts the result is no
    longer refused. A fix that does not fix its own case is a failing test —
    the standard **D-063** set for ``flow/autoconfig.py``'s suggestions, and
    ``CLAUDE.md`` constraint 14.

    A ``"picture"`` fix cannot be executed against the user's own picture,
    because the tool cannot invent geometry it was not given. It is executed
    against :data:`flow.diagnose.EXAMPLE_MASK` instead — the same worked
    example Rung D substitutes — and the rung prints that it is doing so rather
    than quietly claiming to have run the user's case.

    Args:
        fix: the :class:`Fix` from a refused :class:`Prepared`.
        source: the source that was refused.
        cells_across: the body size that was asked for.
        repair: the repair setting that was used.

    Returns:
        Keyword arguments for :func:`prepare`.

    Raises:
        ValueError: if ``fix.change`` is not one this function knows how to
            apply — loud, for the same reason ``flow.diagnose.classify`` is.
    """
    request: dict[str, Any] = {
        "source": source,
        "cells_across": cells_across,
        "repair": repair,
    }
    if fix.change == "resolution":
        request["cells_across"] = int(fix.value)
    elif fix.change == "picture":
        request["source"] = EXAMPLE_MASK
    else:  # pragma: no cover - guarded by a test with teeth
        raise ValueError(
            f"flow.prepare cannot apply a fix that changes {fix.change!r}; "
            "it knows 'resolution' and 'picture'."
        )
    return request


def _print(result: Prepared, cells_across: int) -> None:
    """The verbose summary: verdict, every action, every measured number."""
    print(f"prepare: verdict {result.verdict} (asked for {cells_across} cells across)")
    for action in result.actions:
        print(f"  - {action}")
    if result.reason:
        print(f"  refused: {result.reason}")
    if result.fix:
        print(f"  fix: {result.fix.note}")
    props = ", ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in result.properties.items()
    )
    print(f"  measured: {props}")
