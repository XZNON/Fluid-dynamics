"""Physical quantities: what the user types, in SI, with the dimension attached.

``DOCS/IDEA3.md`` § The five things Phase 1 must get right (1) — *the user never
types a lattice quantity*. This module is the **outer face** of the units
boundary: it speaks metres, seconds, m^2/s, kelvin and kg/m^3, and nothing else.
``lbm/units.py`` is the inner face, where those become ``dx``, ``dt``, ``tau``
and a lattice ``U`` (Phase 1 constraint 13). The two never meet in this file —
there is no ``tau`` here, no lattice velocity, no cell count, and nothing from
``lbm/`` is imported.

Why a parser and not ``pint``
-----------------------------

**D-031**: ``lbm.geometry.from_svg`` parses a subset of SVG itself rather than
taking a dependency, and says honestly what it does not support. The same trade
applies here — the unit set this product needs is bounded (six dimensions' worth
of everyday units), a dependency is not, and the honest failure message is half
the value. :data:`_UNITS` is the whole supported set; anything outside it raises
rather than guessing.

The shape of a refusal
----------------------

**D-045**: refuse, explain in the user's units, and name a way forward. Every
:class:`ValueError` raised here carries the three things one layer down from
that: **what was given**, **what dimension was expected**, and **one valid
example**. ``flow/diagnose.py`` (T106) builds the structured version on top of
the same vocabulary — do not invent a second one.

Bare numbers
------------

A number with no unit is only accepted when the caller **declares** the unit it
means (``default_unit=``). A number carrying a unit of the wrong dimension is
never quietly reinterpreted as the default: that is the "no silent default-unit
assumption" half of the contract, and it is what stops ``"20 kg/m^3"`` from
becoming twenty metres per second.
"""

from __future__ import annotations

import re

__all__ = [
    "Quantity",
    "parse",
    "to_si",
    "LENGTH",
    "SPEED",
    "TIME",
    "VISCOSITY",
    "TEMPERATURE",
    "DENSITY",
    "DIMENSIONS",
    "SI_UNIT",
    "EXAMPLE",
    "units_for",
]

# -- dimensions ------------------------------------------------------------
#
# Named as the user would name them, because they appear verbatim in the error
# messages ("a speed was expected"). Kinematic viscosity is the only one whose
# name is longer than a word, and it is spelled out for the same reason.

LENGTH: str = "length"
SPEED: str = "speed"
TIME: str = "time"
VISCOSITY: str = "kinematic viscosity"
TEMPERATURE: str = "temperature"
DENSITY: str = "density"

DIMENSIONS: tuple[str, ...] = (LENGTH, SPEED, TIME, VISCOSITY, TEMPERATURE, DENSITY)

#: The unit every :class:`Quantity` of that dimension is stored in.
SI_UNIT: dict[str, str] = {
    LENGTH: "m",
    SPEED: "m/s",
    TIME: "s",
    VISCOSITY: "m^2/s",
    TEMPERATURE: "K",
    DENSITY: "kg/m^3",
}

#: One valid example per dimension — the third thing every refusal must carry.
EXAMPLE: dict[str, str] = {
    LENGTH: '"1.5 m"',
    SPEED: '"20 m/s"',
    TIME: '"5 s"',
    VISCOSITY: '"1.5e-5 m^2/s"',
    TEMPERATURE: '"20 °C"',
    DENSITY: '"998 kg/m^3"',
}

#: The units named in error messages — the common spellings, not all ninety.
_HEADLINE_UNITS: dict[str, tuple[str, ...]] = {
    LENGTH: ("m", "cm", "mm", "km", "in", "ft"),
    SPEED: ("m/s", "km/h", "mph", "knots", "ft/s", "cm/s"),
    TIME: ("s", "ms", "min", "h"),
    VISCOSITY: ("m^2/s", "mm^2/s", "cSt", "St"),
    TEMPERATURE: ("°C", "K", "°F"),
    DENSITY: ("kg/m^3", "g/cm^3"),
}


def units_for(dimension: str) -> tuple[str, ...]:
    """The common unit spellings for ``dimension``, for messages and tests."""
    return _HEADLINE_UNITS[dimension]


# -- the unit table --------------------------------------------------------
#
# unit spelling -> (dimension, factor, offset), with si = value * factor +
# offset. The offset exists for temperature and nowhere else; it is carried by
# every unit so the conversion has one code path rather than a special case.

_UNITS: dict[str, tuple[str, float, float]] = {}


def _add(dimension: str, factor: float, offset: float, *spellings: str) -> None:
    for spelling in spellings:
        _UNITS[spelling] = (dimension, factor, offset)


# length
_add(LENGTH, 1.0, 0.0, "m", "metre", "metres", "meter", "meters")
_add(LENGTH, 1e-2, 0.0, "cm", "centimetre", "centimetres", "centimeter", "centimeters")
_add(LENGTH, 1e-3, 0.0, "mm", "millimetre", "millimetres", "millimeter", "millimeters")
_add(LENGTH, 1e-6, 0.0, "um", "micrometre", "micrometres", "micron", "microns")
_add(LENGTH, 1e3, 0.0, "km", "kilometre", "kilometres", "kilometer", "kilometers")
_add(LENGTH, 0.0254, 0.0, "in", "inch", "inches", '"')
_add(LENGTH, 0.3048, 0.0, "ft", "foot", "feet", "'")
_add(LENGTH, 0.9144, 0.0, "yd", "yard", "yards")
_add(LENGTH, 1609.344, 0.0, "mi", "mile", "miles")

# speed
_add(SPEED, 1.0, 0.0, "m/s", "m/sec", "mps", "metres/s", "meters/s")
_add(SPEED, 1e-2, 0.0, "cm/s", "cm/sec")
_add(SPEED, 1e-3, 0.0, "mm/s")
_add(SPEED, 1.0 / 3.6, 0.0, "km/h", "km/hr", "kph", "kmh")
_add(SPEED, 0.44704, 0.0, "mph", "mi/h", "miles/h")
_add(SPEED, 0.3048, 0.0, "ft/s", "fps", "feet/s")
_add(SPEED, 1852.0 / 3600.0, 0.0, "kn", "kt", "kts", "knot", "knots")

# time
_add(TIME, 1.0, 0.0, "s", "sec", "secs", "second", "seconds")
_add(TIME, 1e-3, 0.0, "ms", "millisecond", "milliseconds")
_add(TIME, 1e-6, 0.0, "us", "microsecond", "microseconds")
_add(TIME, 60.0, 0.0, "min", "mins", "minute", "minutes")
_add(TIME, 3600.0, 0.0, "h", "hr", "hrs", "hour", "hours")

# kinematic viscosity
_add(VISCOSITY, 1.0, 0.0, "m^2/s", "m2/s")
_add(VISCOSITY, 1e-4, 0.0, "cm^2/s", "cm2/s", "St", "stokes")
_add(VISCOSITY, 1e-6, 0.0, "mm^2/s", "mm2/s", "cSt", "centistokes")

# temperature — the only affine conversions in the table
_add(TEMPERATURE, 1.0, 0.0, "K", "kelvin", "kelvins")
_add(TEMPERATURE, 1.0, 273.15, "°C", "degC", "celsius", "centigrade")
_add(TEMPERATURE, 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0, "°F", "degF", "fahrenheit")

# density
_add(DENSITY, 1.0, 0.0, "kg/m^3", "kg/m3")
_add(DENSITY, 1e-3, 0.0, "g/m^3", "g/m3")
_add(DENSITY, 1e3, 0.0, "g/cm^3", "g/cm3", "g/cc", "kg/L", "kg/l")
_add(DENSITY, 1.0, 0.0, "g/L", "g/l")

# A case-insensitive index, built once. A spelling whose lowercase form is
# claimed by two *different* units is dropped from this index rather than
# guessed at, so a future "K"/"k" pair could never silently merge — the
# case-sensitive table above still resolves both.
_UNITS_LOWER: dict[str, tuple[str, float, float]] = {}
_ambiguous: set[str] = set()
for _spelling, _entry in _UNITS.items():
    _key = _spelling.lower()
    if _key in _UNITS_LOWER and _UNITS_LOWER[_key] != _entry:
        _ambiguous.add(_key)
    _UNITS_LOWER[_key] = _entry
for _key in _ambiguous:
    del _UNITS_LOWER[_key]
del _spelling, _entry, _key, _ambiguous


# -- parsing ---------------------------------------------------------------

#: A leading signed decimal or scientific-notation number; the rest is the unit.
_NUMBER = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$", re.DOTALL)


def _normalise_unit(text: str) -> str:
    """Fold the spellings that differ only in typography, and nothing else.

    Superscripts, the two micro signs, ``**`` for ``^``, spaces around the
    solidus, the word ``per`` and the ``deg C`` spelling all collapse here. Case
    does **not** — that is a separate, lower-priority lookup, because ``m`` and
    ``M`` are different units in SI even though nothing in this table exploits
    it.
    """
    unit = text.strip()
    unit = unit.replace("²", "^2").replace("³", "^3")
    unit = unit.replace("**", "^")
    unit = unit.replace("µ", "u").replace("μ", "u")
    unit = re.sub(r"\s*/\s*", "/", unit)
    unit = re.sub(r"\s+per\s+", "/", unit, flags=re.IGNORECASE)
    unit = re.sub(r"^deg\s+", "deg", unit, flags=re.IGNORECASE)
    unit = re.sub(r"^°\s+", "°", unit)
    unit = re.sub(r"\s+", " ", unit)
    return unit


def _lookup(unit: str) -> tuple[str, float, float] | None:
    """``unit`` -> ``(dimension, factor, offset)``, or ``None`` if unknown."""
    normalised = _normalise_unit(unit)
    entry = _UNITS.get(normalised)
    if entry is not None:
        return entry
    return _UNITS_LOWER.get(normalised.lower())


def _expected_clause(expect: str | None) -> str:
    """The "what was expected" half of every message in this module."""
    if expect is None:
        return "a number and a unit, e.g. " + " or ".join(
            EXAMPLE[d] for d in (LENGTH, SPEED, VISCOSITY)
        )
    return (
        f"a {expect} -- its units are "
        + ", ".join(units_for(expect))
        + f", for example {EXAMPLE[expect]}"
    )


class Quantity:
    """A number the user gave, converted to SI, with its dimension attached.

    ``DOCS/IDEA3.md`` § The five things Phase 1 must get right (1). Construct it
    from whatever the user typed::

        Quantity("20 m/s")             # -> 20.0 m/s
        Quantity("72 km/h")            # -> 20.0 m/s
        Quantity("1.5e-5 m^2/s")       # -> a kinematic viscosity
        Quantity(20.0, default_unit="m/s")

    Immutable, by ``__slots__`` plus a blocked ``__setattr__``: a quantity is the
    record of *what was asked for*, and a mutable one would let a later layer
    edit the user's request without saying so (constraint 16).

    Attributes:
        si: the value in :data:`SI_UNIT` for its dimension. Always a ``float``.
        dimension: one of :data:`DIMENSIONS`.
        given: the input verbatim, for error messages and reports.
    """

    __slots__ = ("si", "dimension", "given")

    si: float
    dimension: str
    given: str

    def __init__(
        self,
        spec: "str | float | int | Quantity",
        *,
        expect: str | None = None,
        default_unit: str | None = None,
    ) -> None:
        si, dimension, given = _to_fields(spec, expect, default_unit)
        object.__setattr__(self, "si", si)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "given", given)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Quantity is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Quantity is immutable")

    # -- reading it back ---------------------------------------------------

    @property
    def unit(self) -> str:
        """The SI unit this quantity is stored in."""
        return SI_UNIT[self.dimension]

    def as_unit(self, unit: str) -> float:
        """The value expressed in ``unit``. Inverse of the parse conversion.

        Raises:
            ValueError: if ``unit`` is unknown, or is a unit of some other
                dimension — naming what was given, what was expected and an
                example, like every other refusal here.
        """
        entry = _lookup(unit)
        if entry is None:
            raise ValueError(
                f"unit {unit!r} is not known. "
                f"Expected {_expected_clause(self.dimension)}."
            )
        dimension, factor, offset = entry
        if dimension != self.dimension:
            raise ValueError(
                f"unit {unit!r} is a unit of {dimension}, but this quantity is a "
                f"{self.dimension}. Expected {_expected_clause(self.dimension)}."
            )
        return (self.si - offset) / factor

    def __float__(self) -> float:
        return self.si

    def __str__(self) -> str:
        return f"{self.si:g} {self.unit}"

    def __repr__(self) -> str:
        return f"Quantity({self.si!r}, {self.dimension!r}, given={self.given!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        return self.si == other.si and self.dimension == other.dimension

    def __hash__(self) -> int:
        return hash((self.si, self.dimension))


def _to_fields(
    spec: "str | float | int | Quantity",
    expect: str | None,
    default_unit: str | None,
) -> tuple[float, str, str]:
    """The whole conversion, shared by :class:`Quantity` and :func:`parse`."""
    if expect is not None and expect not in SI_UNIT:
        raise ValueError(
            f"expect={expect!r} is not a dimension this module knows. "
            f"Expected one of: {', '.join(DIMENSIONS)}."
        )

    if isinstance(spec, Quantity):
        if expect is not None and spec.dimension != expect:
            raise ValueError(
                f"{spec.given!r} is a {spec.dimension}, but a {expect} was "
                f"expected. Give {_expected_clause(expect)}."
            )
        return spec.si, spec.dimension, spec.given

    if isinstance(spec, bool):
        # bool is an int, and "True metres" is a mistake, not a quantity.
        raise ValueError(
            f"cannot read {spec!r} as a quantity. Expected {_expected_clause(expect)}."
        )

    if isinstance(spec, (int, float)):
        number, unit_text, given = float(spec), "", repr(spec)
    elif isinstance(spec, str):
        given = spec
        match = _NUMBER.match(spec.strip())
        if match is None:
            raise ValueError(
                f"cannot read {spec!r} as a number with a unit. "
                f"Expected {_expected_clause(expect)}."
            )
        number = float(match.group(1))
        unit_text = match.group(2).strip()
    else:
        raise ValueError(
            f"cannot read {spec!r} (a {type(spec).__name__}) as a quantity. "
            f"Expected a string or a number -- {_expected_clause(expect)}."
        )

    if not unit_text:
        if default_unit is None:
            raise ValueError(
                f"{spec!r} has no unit and no default unit was declared. "
                f"Expected {_expected_clause(expect)}."
            )
        entry = _lookup(default_unit)
        if entry is None:
            raise ValueError(
                f"default_unit={default_unit!r} is not a unit this module knows. "
                f"Expected {_expected_clause(expect)}."
            )
    else:
        entry = _lookup(unit_text)
        if entry is None:
            raise ValueError(
                f"{spec!r}: the unit {unit_text!r} is not known. "
                f"Expected {_expected_clause(expect)}."
            )

    dimension, factor, offset = entry
    if expect is not None and dimension != expect:
        # The unit is present and it is the wrong dimension. It is *not*
        # reinterpreted with default_unit: that would be exactly the silent
        # substitution constraint 16 exists to forbid.
        raise ValueError(
            f"{spec!r} is a {dimension}, but a {expect} was expected. "
            f"Give {_expected_clause(expect)}."
        )

    return number * factor + offset, dimension, given


def parse(
    spec: "str | float | int | Quantity",
    *,
    expect: str | None = None,
    default_unit: str | None = None,
) -> Quantity:
    """``"20 m/s"``, ``"72 km/h"``, ``"20"`` or ``20.0`` -> a :class:`Quantity`.

    Args:
        spec: what the user typed. A string with a unit, a string without one, a
            bare number, or an already-parsed :class:`Quantity` (returned as an
            equal quantity, so callers can accept either without a type switch).
        expect: the dimension required, one of :data:`DIMENSIONS`. When given, a
            quantity of any other dimension is refused rather than converted.
        default_unit: the unit a **unitless** input means. Required for bare
            numbers; never applied to an input that carries a unit of its own.

    Returns:
        A :class:`Quantity` holding the SI value and its dimension.

    Raises:
        ValueError: naming what was given, what dimension was expected, and one
            valid example (**D-045**).
    """
    return Quantity(spec, expect=expect, default_unit=default_unit)


def to_si(
    spec: "str | float | int | Quantity",
    *,
    expect: str | None = None,
    default_unit: str | None = None,
) -> float:
    """:func:`parse`, then the bare SI number. For callers that want a float."""
    return parse(spec, expect=expect, default_unit=default_unit).si
