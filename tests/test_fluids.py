"""T104 — the fluid library: cited numbers, name resolution, custom viscosities.

Covers ``flow/fluids.py``, one or more tests per acceptance criterion in
``DOCS/TASKS2.md`` § T104. Spec section: ``DOCS/IDEA3.md`` § The five things
Phase 1 must get right (1).

A note on the ordering criterion
--------------------------------

The contract's parenthetical asks for ``nu`` "physically ordered (helium < air <
water < oil < glycerine)". **That ordering is not physical for kinematic
viscosity and the library does not reproduce it** — see ``DOCS/STATE2.md``
**D-058**. ``nu = mu / rho``, and helium is ~7x less dense than air while its
``mu`` is slightly *larger*, so helium's ``nu`` is ~8x air's; water's ``mu`` is
55x air's but its density is 830x, so water's ``nu`` is the *smallest* of the
six. The intent behind the criterion — that the table be ordered by physics and
not by typing — is kept and strengthened here: the order is asserted against the
measured values, **and** every entry's ``nu`` is checked against its
independently cited ``mu`` and ``rho``, which is the check that actually catches
a transcription error.
"""

from __future__ import annotations

import pytest

from flow.fluids import FLUIDS, Fluid, fluid, known_fluids
from flow.quantity import DENSITY, TEMPERATURE, VISCOSITY, Quantity, parse

#: The six the contract names, plus honey. Every one must be present.
REQUIRED = ("air", "water", "honey", "olive oil", "glycerine", "helium")

#: Ascending kinematic viscosity at 20 °C — the *measured* order, derived from
#: the cited ``mu`` and ``rho`` of each entry rather than asserted from memory.
#: See the module docstring and **D-058**.
EXPECTED_ORDER = ("water", "air", "olive oil", "helium", "glycerine", "honey")


# ---------------------------------------------------------------------------
# Criterion 3 — every entry has nu in m^2/s at a stated T, and a cited source
# ---------------------------------------------------------------------------


def test_every_required_fluid_is_present():
    for name in REQUIRED:
        assert name in FLUIDS, f"{name} missing from FLUIDS"


@pytest.mark.parametrize("name", sorted(FLUIDS))
def test_every_entry_is_a_viscosity_a_density_and_a_temperature(name: str):
    entry = FLUIDS[name]
    assert isinstance(entry, Fluid)
    assert entry.name == name

    assert entry.nu.dimension == VISCOSITY
    assert entry.nu.unit == "m^2/s"
    assert entry.nu.si > 0.0

    assert entry.rho is not None and entry.rho.dimension == DENSITY
    assert entry.rho.si > 0.0

    assert entry.T is not None and entry.T.dimension == TEMPERATURE
    assert entry.T.si > 0.0


@pytest.mark.parametrize("name", sorted(FLUIDS))
def test_every_entry_cites_a_non_empty_source_naming_its_temperature(name: str):
    entry = FLUIDS[name]
    assert entry.source.strip(), f"{name} has no source"
    assert len(entry.source) > 20, f"{name}'s source is not a citation: {entry.source}"
    # The temperature the numbers are for is stated in the citation, not only
    # in the field — DOCS/TASKS2.md § T104 Notes.
    assert "°C" in entry.source or "K" in entry.source


@pytest.mark.parametrize("name", sorted(FLUIDS))
def test_kinematic_viscosity_agrees_with_the_cited_dynamic_viscosity(name: str):
    """``nu = mu / rho``, from three numbers the source quotes independently.

    This is the check that catches a mistyped exponent: ``nu`` is not derived in
    the table, it is transcribed, so an error in it does not show up anywhere
    else until a Reynolds number is wrong by orders of magnitude.
    """
    entry = FLUIDS[name]
    assert entry.mu_pa_s is not None and entry.rho is not None
    derived = entry.mu_pa_s / entry.rho.si
    assert entry.nu.si == pytest.approx(derived, rel=2e-3), (
        f"{name}: nu = {entry.nu.si:.6g} but mu/rho = {derived:.6g}"
    )


def test_kinematic_viscosity_is_physically_ordered():
    """Ascending ``nu``, against the order the cited numbers actually produce.

    Deviates from the contract's parenthetical on purpose — see the module
    docstring and **D-058**. Ordering ``mu`` instead would not rescue it either:
    helium's ``mu`` (1.96e-5 Pa s) exceeds air's (1.825e-5 Pa s), so
    "helium < air" is false in both viscosities.
    """
    measured = tuple(sorted(FLUIDS, key=lambda n: FLUIDS[n].nu.si))
    assert measured == EXPECTED_ORDER

    values = [FLUIDS[name].nu.si for name in EXPECTED_ORDER]
    assert values == sorted(values)
    # Strictly ordered — no two entries share a value by accident.
    assert len(set(values)) == len(values)


def test_the_ordering_the_contract_asked_for_is_not_physical():
    """The conflict, asserted rather than described (**D-058**).

    If a future session "fixes" the data to satisfy the contract's
    parenthetical, this test fails and says why — which is the point of writing
    the disagreement down as a test instead of a comment.
    """
    assert FLUIDS["helium"].nu.si > FLUIDS["air"].nu.si
    assert FLUIDS["air"].nu.si > FLUIDS["water"].nu.si
    # The two halves of the parenthetical that *are* right, kept as claims.
    assert FLUIDS["water"].nu.si < FLUIDS["olive oil"].nu.si
    assert FLUIDS["olive oil"].nu.si < FLUIDS["glycerine"].nu.si


def test_air_and_water_match_the_values_lbm_units_documents():
    """``lbm/units.py`` tells users "air at 20 C is 1.5e-5, water 1.0e-6".

    Same numbers, one significant figure apart — the two faces of the units
    boundary must not disagree about what air is (constraint 13's whole point).
    """
    assert FLUIDS["air"].nu.si == pytest.approx(1.5e-5, rel=0.02)
    assert FLUIDS["water"].nu.si == pytest.approx(1.0e-6, rel=0.02)


def test_every_entry_is_quoted_at_twenty_celsius():
    """One value at one stated temperature (``DOCS/TASKS2.md`` § T104 Notes)."""
    for name, entry in FLUIDS.items():
        assert entry.T is not None
        assert entry.T.as_unit("°C") == pytest.approx(20.0), name


def test_summary_names_the_fluid_its_viscosity_and_its_source():
    line = FLUIDS["air"].summary()
    assert "air" in line
    assert "m^2/s" in line
    assert "20 °C" in line
    assert "White" in line


# ---------------------------------------------------------------------------
# Criterion 4 — name resolution, and an unknown name that lists the known ones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["air", "Air", "AIR", " air ", "\tair\n"])
def test_a_name_resolves_whatever_its_case_and_padding(spelling: str):
    assert fluid(spelling) is FLUIDS["air"]


@pytest.mark.parametrize(
    "spelling", ["olive oil", "Olive Oil", "olive_oil", "olive-oil", "  OLIVE   OIL  "]
)
def test_a_two_word_name_resolves_however_it_is_separated(spelling: str):
    assert fluid(spelling) is FLUIDS["olive oil"]


@pytest.mark.parametrize(
    "spelling, canonical",
    [("glycerol", "glycerine"), ("glycerin", "glycerine"), ("H2O", "water")],
)
def test_the_common_aliases_resolve(spelling: str, canonical: str):
    assert fluid(spelling) is FLUIDS[canonical]


def test_an_unknown_name_is_refused_listing_every_known_one():
    with pytest.raises(ValueError) as excinfo:
        fluid("mercury")
    message = str(excinfo.value)
    assert "mercury" in message
    for name in known_fluids():
        assert name in message, f"{name} missing from the refusal: {message}"
    # D-045: the refusal names the way forward, not only the problem.
    assert "m^2/s" in message


def test_known_fluids_lists_the_library():
    assert set(known_fluids()) == set(FLUIDS)
    assert len(known_fluids()) >= len(REQUIRED)


def test_a_wrong_type_is_refused_rather_than_coerced():
    for bad in (None, 20.0, ["air"]):
        with pytest.raises(ValueError):
            fluid(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Criterion 5 — a custom fluid, given as a viscosity, without touching FLUIDS
# ---------------------------------------------------------------------------


def test_a_custom_viscosity_becomes_a_fluid_without_touching_the_library():
    before = dict(FLUIDS)
    custom = fluid(Quantity("1.5e-5 m^2/s"))

    assert custom.nu.si == pytest.approx(1.5e-5)
    assert custom.nu.dimension == VISCOSITY
    assert custom.rho is None and custom.T is None
    assert custom.source
    assert FLUIDS == before, "a custom fluid mutated FLUIDS"
    assert "custom" not in FLUIDS


def test_a_custom_viscosity_may_be_given_as_a_plain_string():
    assert fluid("1.5e-5 m^2/s").nu.si == pytest.approx(1.5e-5)
    assert fluid("15 cSt").nu.si == pytest.approx(1.5e-5)


def test_a_quantity_of_the_wrong_dimension_is_refused_as_a_fluid():
    with pytest.raises(ValueError) as excinfo:
        fluid(parse("20 m/s"))
    message = str(excinfo.value)
    assert VISCOSITY in message
    assert "m^2/s" in message
    for name in known_fluids():
        assert name in message


def test_a_fluid_passes_through_fluid_unchanged():
    assert fluid(FLUIDS["water"]) is FLUIDS["water"]


# ---------------------------------------------------------------------------
# DOCS/TASKS2.md § T104 Notes — one temperature, and it says so
# ---------------------------------------------------------------------------


def test_asking_for_water_at_eighty_celsius_is_told_the_value_is_for_twenty():
    note = FLUIDS["water"].temperature_note("80 °C")
    assert note is not None
    assert "80" in note and "20" in note
    assert "does not correct" in note


def test_no_note_when_the_case_is_at_the_tabulated_temperature():
    assert FLUIDS["water"].temperature_note("20 °C") is None
    assert FLUIDS["water"].temperature_note("293.15 K") is None
    assert FLUIDS["water"].temperature_note("22 °C") is None


def test_a_custom_fluid_has_no_temperature_and_so_says_nothing():
    assert fluid("1.5e-5 m^2/s").temperature_note("80 °C") is None


# ---------------------------------------------------------------------------
# The library is reference data
# ---------------------------------------------------------------------------


def test_a_fluid_is_immutable():
    with pytest.raises(Exception):
        FLUIDS["air"].nu = parse("1 m^2/s")  # type: ignore[misc]


def test_fluids_imports_nothing_from_lbm():
    """Constraint 2's half of the boundary: a fluid's nu never meets ``tau``."""
    import ast
    import pathlib

    import flow.fluids as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "lbm" for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "lbm"

    # No identifier named ``tau`` anywhere: a fluid's nu is a physical
    # viscosity in m^2/s and never meets the lattice nu = (tau - 0.5)/3.
    # Prose may name the constraint; code may not enact it.
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "tau" not in identifiers
