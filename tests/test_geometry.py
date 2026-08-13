"""T004 — geometry primitives and mask sanity checks.

Covers the acceptance criteria of ``old-Docs/TASKS1.md`` § T004 and the three rules
of ``DOCS/IDEA2.md`` § Geometry from a mask.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

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

NY, NX = 61, 121


# --------------------------------------------------------------------------
# Primitives: shape, dtype, index order
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask",
    [
        circle(NY, NX, 40.0, 30.0, 8.0),
        rectangle(NY, NX, 20.0, 25.0, 35.0, 40.0),
        polygon(NY, NX, [(20.0, 20.0), (50.0, 25.0), (35.0, 45.0)]),
        regular_polygon(NY, NX, 4, 60.0, 30.0, 10.0, np.pi / 4),
        channel_walls(NY, NX),
    ],
)
def test_primitives_return_ny_nx_bool(mask: np.ndarray) -> None:
    """Constraint 4: every primitive is ``(ny, nx)`` and ``bool``, not ``(nx, ny)``."""
    assert mask.shape == (NY, NX)
    assert mask.dtype == np.bool_


def test_index_order_is_y_then_x() -> None:
    """A rectangle wide in ``x`` and short in ``y`` must come back that way."""
    mask = rectangle(NY, NX, x0=10.0, y0=5.0, x1=99.0, y1=8.0)
    y0, y1, x0, x1 = bounding_box(mask)
    assert (y0, y1) == (5, 8)
    assert (x0, x1) == (10, 99)
    assert mask[6, 50] and not mask[20, 50]


def test_circle_centre_and_radius() -> None:
    mask = circle(NY, NX, cx=40.0, cy=30.0, radius=8.0)
    assert mask[30, 40]
    assert mask[30, 48] and mask[38, 40]  # exactly on the radius: inclusive
    assert not mask[30, 49]
    assert not mask[21, 40]


def test_circle_area_within_2_percent() -> None:
    """Cell-centre sampling of a disc converges on ``pi r^2``."""
    r = 20.0
    mask = circle(101, 101, 50.0, 50.0, r)
    assert abs(mask.sum() / (np.pi * r**2) - 1.0) < 0.02


def test_rectangle_bounds_inclusive_and_sorted() -> None:
    mask = rectangle(20, 20, 4.0, 6.0, 8.0, 9.0)
    assert mask.sum() == (8 - 4 + 1) * (9 - 6 + 1)
    flipped = rectangle(20, 20, 8.0, 9.0, 4.0, 6.0)
    assert np.array_equal(mask, flipped)


def test_polygon_known_area_within_2_percent() -> None:
    """Convex case with a known area: a 40x30 axis-aligned rectangle as a polygon.

    Acceptance criterion: "tested against a known-area convex case to within 2%".
    """
    verts = [(20.0, 15.0), (60.0, 15.0), (60.0, 45.0), (20.0, 45.0)]
    mask = polygon(NY, NX, verts)
    area = 40.0 * 30.0
    assert abs(mask.sum() / area - 1.0) < 0.02


def test_polygon_regular_hexagon_area_within_2_percent() -> None:
    """Second known-area convex case, this one not axis-aligned."""
    r = 25.0
    mask = regular_polygon(101, 101, 6, 50.0, 50.0, r)
    area = 1.5 * np.sqrt(3.0) * r**2  # regular hexagon of circumradius r
    assert abs(mask.sum() / area - 1.0) < 0.02


def test_polygon_handles_concave_shapes() -> None:
    """An L: the notch must be fluid, the two legs solid.

    The even-odd rule is what buys this; a convex-hull test would fill the notch.
    """
    verts = [
        (10.0, 10.0),
        (50.0, 10.0),
        (50.0, 20.0),
        (20.0, 20.0),
        (20.0, 50.0),
        (10.0, 50.0),
    ]
    mask = polygon(NY, NX, verts)
    assert mask[15, 40]  # bottom leg
    assert mask[40, 15]  # upright leg
    assert not mask[40, 40]  # the notch — inside the hull, outside the L
    # Area of the L (40x10 + 10x30), not of its hull (40x40 = 1600).
    assert abs(mask.sum() / 700.0 - 1.0) < 0.05


def test_polygon_vertex_order_does_not_matter() -> None:
    verts = [(20.0, 15.0), (60.0, 15.0), (55.0, 45.0), (25.0, 40.0)]
    assert np.array_equal(polygon(NY, NX, verts), polygon(NY, NX, verts[::-1]))


def test_polygon_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least 3 vertices"):
        polygon(NY, NX, [(1.0, 1.0), (2.0, 2.0)])
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        polygon(NY, NX, [1.0, 2.0, 3.0])


def test_circle_and_polygon_reject_bad_radius() -> None:
    with pytest.raises(ValueError, match="radius must be positive"):
        circle(NY, NX, 10.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="at least 3 sides"):
        regular_polygon(NY, NX, 2, 10.0, 10.0, 5.0)


# --------------------------------------------------------------------------
# channel_walls
# --------------------------------------------------------------------------


def test_channel_walls_matches_the_inline_rung_1_mask() -> None:
    """Generalises ``validate/poiseuille.py::channel_mask`` exactly."""
    expected = np.zeros((NY, NX), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    assert np.array_equal(channel_walls(NY, NX), expected)


def test_channel_walls_thickness() -> None:
    mask = channel_walls(20, 10, thickness=3)
    assert mask[:3, :].all() and mask[-3:, :].all()
    assert not mask[3:-3, :].any()


def test_channel_walls_composable_with_or() -> None:
    """Acceptance criterion: composable with ``|``."""
    solid = channel_walls(NY, NX) | circle(NY, NX, 40.0, 30.0, 8.0)
    assert solid.dtype == np.bool_ and solid.shape == (NY, NX)
    assert solid[0, :].all() and solid[-1, :].all()
    assert solid[30, 40]


def test_channel_walls_rejects_a_grid_with_no_fluid() -> None:
    with pytest.raises(ValueError, match="no fluid rows"):
        channel_walls(6, 10, thickness=3)


# --------------------------------------------------------------------------
# Thickness measurement
# --------------------------------------------------------------------------


def _diagonal_line(n: int = 40) -> np.ndarray:
    """A deliberately 1-cell-thick diagonal, the classic leaky mask."""
    mask = np.zeros((n, n), dtype=bool)
    idx = np.arange(5, n - 5)
    mask[idx, idx] = True
    return mask


def test_thickness_warns_on_a_1_cell_diagonal_not_on_a_4_cell_block() -> None:
    """The acceptance criterion, verbatim: warns for the first, not the second."""
    diagonal = _diagonal_line()
    block = np.zeros((40, 40), dtype=bool)
    block[18:22, 18:22] = True  # 4 cells thick

    assert min_thickness(diagonal) == 1
    assert min_thickness(block) >= 3

    with pytest.warns(MaskWarning, match="cells thick at its thinnest"):
        msgs = check_mask(diagonal, "x", verbose=False)
    assert any("thick at its thinnest" in m for m in msgs)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MaskWarning)
        msgs = check_mask(
            block,
            "x",
            min_downstream_lengths=0.0,
            max_blockage=1.0,
            verbose=False,
        )
    assert msgs == []


def test_thickness_of_a_disc_is_its_diameter_not_a_run_length() -> None:
    """A cylinder must not warn: the top cell of a disc has a vertical run of 1."""
    disc = circle(61, 61, 30.0, 30.0, 10.0)
    t = min_thickness(disc)
    assert 15 <= t <= 21  # diameter is 21 cells; the estimate is conservative


def test_thickness_of_a_2_cell_bar_warns() -> None:
    """A 2-cell bar has no interior cell, so its deepest point is 1 away."""
    bar = np.zeros((30, 30), dtype=bool)
    bar[10:12, 5:25] = True
    assert min_thickness(bar) == 1


def test_a_thin_plate_beside_a_thick_body_is_still_reported() -> None:
    """The min is over components: the fat blob must not mask the plate."""
    both = np.zeros((60, 60), dtype=bool)
    both[10:30, 10:30] = True  # thick
    both[45, 10:50] = True  # 1 cell thick, separate component
    assert min_thickness(both) == 1


def test_thickness_is_odd_and_conservative() -> None:
    block = np.zeros((40, 40), dtype=bool)
    block[10:30, 10:30] = True  # 20 cells thick
    t = min_thickness(block)
    assert t % 2 == 1 and 15 <= t <= 21


def test_empty_mask_has_zero_thickness() -> None:
    assert min_thickness(np.zeros((10, 10), dtype=bool)) == 0


# --------------------------------------------------------------------------
# strip_solid_border — domain walls are exempt
# --------------------------------------------------------------------------


def test_strip_removes_channel_walls_but_keeps_the_object() -> None:
    obj = circle(NY, NX, 40.0, 30.0, 8.0)
    solid = channel_walls(NY, NX) | obj
    assert np.array_equal(strip_solid_border(solid), obj)


def test_strip_removes_a_full_cavity_border() -> None:
    n = 20
    solid = np.zeros((n, n), dtype=bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    assert not strip_solid_border(solid).any()


def test_strip_keeps_a_partial_edge_blob() -> None:
    """Only *entirely* solid edge rows/columns are peeled."""
    solid = np.zeros((20, 20), dtype=bool)
    solid[0, 5:10] = True
    assert np.array_equal(strip_solid_border(solid), solid)


def test_channel_walls_alone_pass_every_check_silently() -> None:
    """Rung 1's own mask must not warn — a border has no interior to leak into."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", MaskWarning)
        assert check_mask(channel_walls(22, 16), "x", verbose=False) == []


# --------------------------------------------------------------------------
# check_mask — the three rules
# --------------------------------------------------------------------------


def _cylinder_case(
    ny: int = 121, nx: int = 900, d: float = 10.0, cx: float = 150.0
) -> np.ndarray:
    """A well-proportioned Rung 3-style case: thick, unblocked, far from the outlet."""
    return channel_walls(ny, nx) | circle(ny, nx, cx, (ny - 1) / 2.0, d / 2.0)


def test_a_good_cylinder_case_produces_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", MaskWarning)
        assert check_mask(_cylinder_case(), "x", verbose=False) == []


def test_warns_when_too_close_to_the_outlet() -> None:
    """Rule 2: fewer than 8 characteristic lengths downstream."""
    solid = _cylinder_case(nx=200, cx=170.0)
    with pytest.warns(MaskWarning, match="characteristic lengths from the outlet"):
        msgs = check_mask(solid, "x", verbose=False)
    assert any("outlet" in m for m in msgs)


def test_warns_on_blockage_above_10_percent() -> None:
    """Rule 3: object cross-stream extent over 10% of the fluid span."""
    solid = channel_walls(61, 900) | circle(61, 900, 150.0, 30.0, 15.0)
    with pytest.warns(MaskWarning, match="blockage ratio"):
        msgs = check_mask(solid, "x", verbose=False)
    assert any("blockage" in m for m in msgs)


def test_all_three_rules_can_fire_at_once() -> None:
    """A thin, fat, badly-placed plate: three messages, in rule order."""
    solid = np.zeros((40, 60), dtype=bool)
    solid[10:31, 55] = True  # 1 cell thick, 21 cells tall, 4 cells from the outlet
    with pytest.warns(MaskWarning):
        msgs = check_mask(solid, "x", verbose=False)
    assert len(msgs) == 3
    assert "thick at its thinnest" in msgs[0]
    assert "outlet" in msgs[1]
    assert "blockage" in msgs[2]


def test_strict_raises_instead_of_warning() -> None:
    """Acceptance criterion: ``strict=True`` raises."""
    diagonal = _diagonal_line()
    with pytest.raises(ValueError, match="check_mask failed with strict=True"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", MaskWarning)  # no warning may escape
            check_mask(diagonal, "x", strict=True, verbose=False)


def test_strict_is_silent_on_a_clean_mask() -> None:
    assert check_mask(_cylinder_case(), "x", strict=True, verbose=False) == []


def test_warnings_go_through_warnings_warn_with_our_category() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_mask(_diagonal_line(), "x", verbose=False)
    assert caught and all(issubclass(w.category, MaskWarning) for w in caught)
    assert all(issubclass(w.category, UserWarning) for w in caught)


def test_characteristic_length_is_the_bbox_cross_stream_extent_and_is_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Acceptance criterion: derived from the bounding box, and printed."""
    solid = channel_walls(121, 900) | rectangle(121, 900, 150.0, 50.0, 200.0, 70.0)
    check_mask(solid, "x", verbose=True)
    out = capsys.readouterr().out
    assert "Characteristic length D = 21 cells" in out  # y 50..70 inclusive
    assert "bbox y 50..70, x 150..200" in out
    assert "Downstream" in out and "Blockage" in out and "Min solid thickness" in out


def test_inlet_axis_y_swaps_the_two_extents() -> None:
    """Flow along ``y``: the characteristic length becomes the bbox width."""
    solid = rectangle(900, 121, 50.0, 150.0, 60.0, 200.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MaskWarning)
        assert check_mask(solid, "y", verbose=False) == []
    # The same mask read as flow along x is far too close to that outlet.
    msgs = check_mask(solid, "x", verbose=False)
    assert any("outlet" in m for m in msgs)


def test_check_mask_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="must be bool"):
        check_mask(np.zeros((10, 10), dtype=np.float32), "x", verbose=False)
    with pytest.raises(ValueError, match=r"must be \(ny, nx\)"):
        check_mask(np.zeros((3, 10, 10), dtype=bool), "x", verbose=False)
    with pytest.raises(ValueError, match="inlet_axis"):
        check_mask(np.zeros((10, 10), dtype=bool), "z", verbose=False)


def test_bounding_box_of_an_empty_mask_is_none() -> None:
    assert bounding_box(np.zeros((10, 10), dtype=bool)) is None
