"""T104 — physical quantities: parsing, dimensions, and honest refusals.

Covers ``flow/quantity.py``, one or more tests per acceptance criterion in
``DOCS/TASKS2.md`` § T104. The spec section is ``DOCS/IDEA3.md`` § The five
things Phase 1 must get right (1) — *the user never types a lattice quantity*.

The conversion table below is the criterion's "table-driven test": every unit
the contract names (m, cm, mm, in, ft, m/s, km/h, mph, knots, and both ``°C``
and ``K``) appears with an **independently computed** SI value, so the test is a
known answer rather than a re-run of the parser's own arithmetic.
"""

from __future__ import annotations

import pytest

from flow.quantity import (
    DENSITY,
    DIMENSIONS,
    EXAMPLE,
    LENGTH,
    SI_UNIT,
    SPEED,
    TEMPERATURE,
    TIME,
    VISCOSITY,
    Quantity,
    parse,
    to_si,
    units_for,
)

# ---------------------------------------------------------------------------
# Criterion 1 — the same SI value however it is written
# ---------------------------------------------------------------------------

#: ``(input, expected dimension, expected SI value)``. Every SI value here is
#: worked out from the definition of the unit, not from running the parser:
#: 1 in = 25.4 mm exactly, 1 ft = 12 in, 1 mph = 1609.344/3600 m/s, 1 kn =
#: 1852/3600 m/s, 0 °C = 273.15 K.
CONVERSIONS: list[tuple[str, str, float]] = [
    # length
    ("1 m", LENGTH, 1.0),
    ("2.5 m", LENGTH, 2.5),
    ("150 cm", LENGTH, 1.5),
    ("30 mm", LENGTH, 0.03),
    ("12 in", LENGTH, 12 * 0.0254),
    ("1 ft", LENGTH, 0.3048),
    ("3 ft", LENGTH, 0.9144),
    ("1 km", LENGTH, 1000.0),
    ("1 inch", LENGTH, 0.0254),
    ("2 feet", LENGTH, 0.6096),
    # speed
    ("20 m/s", SPEED, 20.0),
    ("72 km/h", SPEED, 20.0),
    ("1 mph", SPEED, 1609.344 / 3600.0),
    ("10 mph", SPEED, 4.4704),
    ("1 knot", SPEED, 1852.0 / 3600.0),
    ("10 knots", SPEED, 5.144444444444444),
    ("100 cm/s", SPEED, 1.0),
    ("1 ft/s", SPEED, 0.3048),
    # time
    ("5 s", TIME, 5.0),
    ("250 ms", TIME, 0.25),
    ("2 min", TIME, 120.0),
    ("1 h", TIME, 3600.0),
    # kinematic viscosity
    ("1.5e-5 m^2/s", VISCOSITY, 1.5e-5),
    ("1.5e-5 m2/s", VISCOSITY, 1.5e-5),
    ("15 cSt", VISCOSITY, 1.5e-5),
    ("1 St", VISCOSITY, 1e-4),
    ("1 mm^2/s", VISCOSITY, 1e-6),
    # temperature — both spellings the criterion names, plus the affine check
    ("0 °C", TEMPERATURE, 273.15),
    ("20 °C", TEMPERATURE, 293.15),
    ("20 degC", TEMPERATURE, 293.15),
    ("-40 °C", TEMPERATURE, 233.14999999999998),
    ("273.15 K", TEMPERATURE, 273.15),
    ("300 K", TEMPERATURE, 300.0),
    ("-40 °F", TEMPERATURE, 233.14999999999998),
    # density
    ("998.2 kg/m^3", DENSITY, 998.2),
    ("1 g/cm^3", DENSITY, 1000.0),
]


@pytest.mark.parametrize("text, dimension, si", CONVERSIONS)
def test_every_unit_converts_to_its_si_value(text: str, dimension: str, si: float):
    quantity = parse(text)
    assert quantity.dimension == dimension
    assert quantity.si == pytest.approx(si, rel=1e-12, abs=1e-15)
    assert quantity.unit == SI_UNIT[dimension]


def test_the_four_ways_of_writing_twenty_metres_per_second_agree():
    """The criterion, verbatim: four spellings, one SI value."""
    from_unit = parse("20 m/s")
    from_other_unit = parse("72 km/h")
    from_bare_string = parse("20", expect=SPEED, default_unit="m/s")
    from_bare_float = parse(20.0, expect=SPEED, default_unit="m/s")

    for quantity in (from_unit, from_other_unit, from_bare_string, from_bare_float):
        assert quantity.dimension == SPEED
        assert quantity.si == pytest.approx(20.0, rel=1e-12)

    assert from_unit == from_other_unit == from_bare_string == from_bare_float


def test_to_si_is_parse_then_the_number():
    assert to_si("72 km/h", expect=SPEED) == pytest.approx(20.0)
    assert to_si(20.0, expect=SPEED, default_unit="m/s") == pytest.approx(20.0)


def test_typography_does_not_change_the_answer():
    """Superscripts, missing spaces, padding, ``**`` and ``per`` all fold."""
    reference = parse("1.5e-5 m^2/s")
    for spelling in (
        "1.5e-5 m²/s",
        "1.5e-5m^2/s",
        "  1.5e-5   m^2 / s  ",
        "1.5e-5 m**2/s",
        "1.5e-5 m^2 per s",
    ):
        assert parse(spelling) == reference, spelling


def test_case_is_tolerated_where_it_is_unambiguous():
    assert parse("72 KM/H") == parse("72 km/h")
    assert parse("10 MPH") == parse("10 mph")
    assert parse("300 k") == parse("300 K")


def test_a_quantity_round_trips_through_any_unit_of_its_dimension():
    speed = parse("20 m/s")
    assert speed.as_unit("km/h") == pytest.approx(72.0)
    assert speed.as_unit("m/s") == pytest.approx(20.0)
    assert parse("20 °C").as_unit("°C") == pytest.approx(20.0)
    assert parse("20 °C").as_unit("°F") == pytest.approx(68.0)


def test_a_quantity_passes_through_parse_unchanged():
    """T105 can accept ``str | float | Quantity`` without a type switch."""
    speed = parse("20 m/s")
    assert parse(speed) == speed
    assert parse(speed, expect=SPEED) == speed


def test_float_and_str_of_a_quantity_are_the_si_value_and_unit():
    speed = parse("72 km/h")
    assert float(speed) == pytest.approx(20.0)
    assert str(speed) == "20 m/s"
    assert speed.given == "72 km/h"


def test_a_quantity_is_immutable():
    speed = parse("20 m/s")
    with pytest.raises(AttributeError):
        speed.si = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Criterion 2 — refusals name what was given, what was expected, and an example
# ---------------------------------------------------------------------------


def _message_is_a_proper_refusal(message: str, *, given: str, expect: str) -> None:
    """**D-045**'s three parts, one layer down: given, expected, example.

    ``flow/diagnose.py`` (T106) builds the structured version on the same
    vocabulary, so this shape is asserted here rather than left to prose.
    """
    assert given in message, f"the message does not quote what was given: {message}"
    assert expect in message, f"the message does not name the dimension: {message}"
    # "one valid example" — a quoted unit-carrying value for that dimension.
    assert "for example" in message or "e.g." in message, message
    assert any(unit in message for unit in units_for(expect)), message


def test_an_unparseable_string_is_refused_with_all_three_parts():
    with pytest.raises(ValueError) as excinfo:
        parse("fast", expect=SPEED)
    _message_is_a_proper_refusal(str(excinfo.value), given="fast", expect=SPEED)


def test_an_unknown_unit_is_refused_and_never_guessed_at():
    with pytest.raises(ValueError) as excinfo:
        parse("20 furlongs/fortnight", expect=SPEED)
    message = str(excinfo.value)
    _message_is_a_proper_refusal(message, given="furlongs/fortnight", expect=SPEED)


def test_a_dimensionally_wrong_string_is_refused_naming_both_dimensions():
    with pytest.raises(ValueError) as excinfo:
        parse("998 kg/m^3", expect=SPEED)
    message = str(excinfo.value)
    _message_is_a_proper_refusal(message, given="998 kg/m^3", expect=SPEED)
    assert DENSITY in message, message


def test_a_wrong_unit_is_never_reinterpreted_as_the_default_unit():
    """The "no silent default-unit assumption" half of the criterion.

    ``"20 kg/m^3"`` asked for as a speed must **not** become twenty metres per
    second just because ``default_unit="m/s"`` was declared — that is the silent
    substitution constraint 16 forbids, in miniature.
    """
    with pytest.raises(ValueError):
        parse("20 kg/m^3", expect=SPEED, default_unit="m/s")


def test_a_bare_number_without_a_declared_default_unit_is_refused():
    with pytest.raises(ValueError) as excinfo:
        parse("20", expect=SPEED)
    assert "no unit" in str(excinfo.value)
    _message_is_a_proper_refusal(str(excinfo.value), given="20", expect=SPEED)

    with pytest.raises(ValueError):
        parse(20.0, expect=SPEED)


def test_a_wrong_type_is_refused_rather_than_coerced():
    for bad in (None, [1, 2], True, {"m/s": 20}):
        with pytest.raises(ValueError):
            parse(bad, expect=SPEED)  # type: ignore[arg-type]


def test_an_unknown_expected_dimension_is_refused():
    with pytest.raises(ValueError) as excinfo:
        parse("20 m/s", expect="mass")
    assert "mass" in str(excinfo.value)
    for dimension in DIMENSIONS:
        assert dimension in str(excinfo.value)


def test_a_bad_default_unit_is_refused_rather_than_ignored():
    with pytest.raises(ValueError) as excinfo:
        parse("20", expect=SPEED, default_unit="furlongs")
    assert "furlongs" in str(excinfo.value)


def test_as_unit_refuses_a_unit_of_another_dimension():
    with pytest.raises(ValueError) as excinfo:
        parse("20 m/s").as_unit("kg/m^3")
    message = str(excinfo.value)
    assert DENSITY in message and SPEED in message


# ---------------------------------------------------------------------------
# Criterion 6 — no new dependency
# ---------------------------------------------------------------------------


def test_flow_quantity_imports_nothing_outside_the_standard_library():
    """**D-031**: a bounded parsing problem is worth ~200 lines, not a dependency.

    Read from the source, so an import inside a function is caught as well as
    one at module level.
    """
    import ast
    import pathlib
    import sys

    import flow.quantity as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    imported.discard("__future__")
    assert imported <= set(sys.stdlib_module_names), (
        f"flow/quantity.py imports non-stdlib modules: "
        f"{sorted(imported - set(sys.stdlib_module_names))}"
    )
    assert "pint" not in imported


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------


def test_every_dimension_carries_an_si_unit_headline_units_and_a_live_example():
    """The example in every refusal must itself parse — or the fix is a lie."""
    for dimension in DIMENSIONS:
        assert SI_UNIT[dimension]
        assert units_for(dimension)
        example = EXAMPLE[dimension].strip('"')
        assert parse(example).dimension == dimension, dimension
        for unit in units_for(dimension):
            assert parse(f"1 {unit}").dimension == dimension, (dimension, unit)


def test_the_si_unit_of_each_dimension_has_unit_scale():
    """``1 <SI unit>`` is 1 — the identity that makes the table self-checking.

    Temperature is affine rather than scaled, so 1 K is 1 K and nothing more is
    claimed for it.
    """
    for dimension, unit in SI_UNIT.items():
        quantity = parse(f"1 {unit}")
        assert quantity.dimension == dimension
        assert quantity.si == pytest.approx(1.0)


def test_a_quantity_object_is_the_documented_public_class():
    assert isinstance(parse("1 m"), Quantity)
    assert Quantity("72 km/h") == parse("72 km/h")
