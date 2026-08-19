"""The fluid library: a name the user knows -> a viscosity with a citation.

``DOCS/IDEA3.md`` § The five things Phase 1 must get right (1). ``"air"`` has to
become a number before anything can be simulated, and the number has to be
*checkable* — hence :attr:`Fluid.source` on every entry. A property with no
citation is a magic constant, and `DOCS/PLAN2.md` § Risks names "a pile of tuned
constants nobody can defend" as a Phase 1 failure mode.

Every value is at **one stated temperature**, and Phase 1 does not model the
temperature dependence (``DOCS/TASKS2.md`` § T104 Notes). That is a real
limitation, so each :class:`Fluid` carries its ``T`` and
:meth:`Fluid.temperature_note` says so when asked about a different one — a user
who wants water at 80 °C gets told, rather than silently handed the 20 °C
number (**D-045**, constraint 16).

Kinematic, not dynamic
----------------------

``nu`` (m^2/s) is what the solver needs, because ``Re = U L / nu``. ``mu``
(Pa s) is carried beside it because that is what the source tables actually
print, and ``nu = mu / rho`` is then a **checkable identity** rather than a
retyped number — ``tests/test_fluids.py`` asserts it for every entry, which is
what catches a transcription error in a table nobody re-derives.

Nothing here imports ``lbm/`` (constraint 15) and nothing here mentions ``tau``
(constraint 2): a fluid's ``nu`` is a physical viscosity in m^2/s and has no
relationship to the lattice ``nu = (tau - 0.5)/3`` beyond sharing a letter.
"""

from __future__ import annotations

from dataclasses import dataclass

from flow.quantity import (
    DENSITY,
    TEMPERATURE,
    VISCOSITY,
    Quantity,
    parse,
)

__all__ = ["Fluid", "FLUIDS", "fluid", "known_fluids"]


@dataclass(frozen=True)
class Fluid:
    """One fluid, at one temperature, with the source the numbers came from.

    Frozen: the library is reference data, and a mutable entry is a way for one
    caller to change what every later caller reads.

    Attributes:
        name: the canonical name, lowercase (``"olive oil"``).
        nu: kinematic viscosity, m^2/s. The one the solver needs.
        rho: density, kg/m^3. ``None`` for a user-supplied viscosity.
        T: the temperature the values are quoted at, K. ``None`` for a
            user-supplied viscosity.
        source: where the numbers come from. Never empty — asserted by test.
        mu_pa_s: dynamic viscosity in Pa s, when the source quotes it.
            ``nu = mu / rho`` is asserted for every library entry. A plain float
            rather than a :class:`~flow.quantity.Quantity` on purpose: Pa s is
            not a unit anyone describes a case in, so it stays out of the
            ``flow.quantity`` vocabulary.
    """

    name: str
    nu: Quantity
    rho: Quantity | None
    T: Quantity | None
    source: str
    mu_pa_s: float | None = None

    def summary(self) -> str:
        """One line: what this fluid is, and at what temperature. For reports."""
        where = f" at {self.T.as_unit('°C'):.0f} °C" if self.T is not None else ""
        density = f", rho = {self.rho.si:g} kg/m^3" if self.rho is not None else ""
        return (
            f"{self.name}: nu = {self.nu.si:.4g} m^2/s{where}{density}"
            f"  [{self.source}]"
        )

    def temperature_note(
        self, temperature: "str | float | Quantity", tolerance_k: float = 5.0
    ) -> str | None:
        """Say so when the case is not at the temperature these numbers are for.

        Phase 1 carries one value per fluid and does **not** interpolate
        (``DOCS/TASKS2.md`` § T104 Notes). Returning a sentence rather than
        raising keeps the judgement in ``flow/diagnose.py`` (T106) where it
        belongs; this is the *fact* that judgement will need.

        Args:
            temperature: the case temperature, as a :class:`Quantity` or
                anything :func:`flow.quantity.parse` accepts as one.
            tolerance_k: how far off is close enough to say nothing, in kelvin.

        Returns:
            ``None`` when the fluid has no stated temperature or the request is
            within ``tolerance_k`` of it; otherwise one sentence naming both
            temperatures and the fact that no correction was applied.
        """
        if self.T is None:
            return None
        asked = parse(temperature, expect=TEMPERATURE, default_unit="K")
        if abs(asked.si - self.T.si) <= tolerance_k:
            return None
        return (
            f"{self.name} is tabulated here at {self.T.as_unit('°C'):.0f} °C, not "
            f"{asked.as_unit('°C'):.0f} °C, and this tool does not correct "
            f"viscosity for temperature. nu = {self.nu.si:.4g} m^2/s is the "
            f"{self.T.as_unit('°C'):.0f} °C value; pass the viscosity you want "
            f'directly (fluid="{self.nu.si:.4g} m^2/s") if that matters.'
        )


def _fluid(
    name: str, *, nu: str, rho: str, T: str, mu_pa_s: float, source: str
) -> Fluid:
    """Build one library entry, parsing every number through its dimension.

    Parsing rather than assigning is the point: a typo like ``"1.004e-6 m/s"``
    is caught here, at import, by the dimension check — not three modules later
    as a Reynolds number that is wrong by six orders of magnitude.
    """
    return Fluid(
        name=name,
        nu=parse(nu, expect=VISCOSITY),
        rho=parse(rho, expect=DENSITY),
        T=parse(T, expect=TEMPERATURE),
        source=source,
        mu_pa_s=mu_pa_s,
    )


#: The library. Keys are canonical lowercase names; see :func:`fluid` for the
#: aliases and the normalisation that reach them.
#:
#: Ordered by kinematic viscosity, which is **not** the order intuition
#: suggests: helium is eight times more viscous than air *kinematically*
#: because it is seven times less dense, and air is fifteen times more viscous
#: than water for the same reason. That is the whole point of quoting ``nu``
#: rather than ``mu`` — ``Re`` is what the solver sees.
FLUIDS: dict[str, Fluid] = {
    f.name: f
    for f in (
        _fluid(
            "water",
            nu="1.004e-6 m^2/s",
            rho="998.2 kg/m^3",
            T="20 °C",
            mu_pa_s=1.002e-3,
            source=(
                "White, Fluid Mechanics 7th ed., Table A.1 — water at 20 °C, "
                "1 atm: mu = 1.002e-3 Pa s, rho = 998.2 kg/m^3"
            ),
        ),
        _fluid(
            "air",
            nu="1.516e-5 m^2/s",
            rho="1.204 kg/m^3",
            T="20 °C",
            mu_pa_s=1.825e-5,
            source=(
                "White, Fluid Mechanics 7th ed., Table A.3 — air at 20 °C, "
                "1 atm: mu = 1.825e-5 Pa s, rho = 1.204 kg/m^3"
            ),
        ),
        _fluid(
            "olive oil",
            nu="8.4e-5 m^2/s",
            rho="911 kg/m^3",
            T="20 °C",
            mu_pa_s=7.652e-2,
            source=(
                "Engineering ToolBox, 'Liquids — Kinematic Viscosities' — olive "
                "oil at 20 °C, ~84 cSt; rho = 911 kg/m^3. Kinematic is the "
                "quoted quantity here; mu is derived as nu * rho"
            ),
        ),
        _fluid(
            "helium",
            nu="1.178e-4 m^2/s",
            rho="0.1664 kg/m^3",
            T="20 °C",
            mu_pa_s=1.96e-5,
            source=(
                "Engineering ToolBox, 'Gases — Dynamic Viscosities' and 'Helium "
                "— Density' — helium at 20 °C, 1 atm: mu = 1.96e-5 Pa s, "
                "rho = 0.1664 kg/m^3"
            ),
        ),
        _fluid(
            "glycerine",
            nu="1.120e-3 m^2/s",
            rho="1261 kg/m^3",
            T="20 °C",
            mu_pa_s=1.412,
            source=(
                "CRC Handbook of Chemistry and Physics, 97th ed. — glycerol at "
                "20 °C: mu = 1.412 Pa s, rho = 1261 kg/m^3"
            ),
        ),
        _fluid(
            "honey",
            nu="7.042e-3 m^2/s",
            rho="1420 kg/m^3",
            T="20 °C",
            mu_pa_s=10.0,
            source=(
                "Engineering ToolBox, 'Absolute Viscosity of Common Liquids' — "
                "honey at 20 °C, ~10 Pa s (representative: honey varies by an "
                "order of magnitude with water content and crystallisation); "
                "rho = 1420 kg/m^3"
            ),
        ),
    )
}

#: Spellings that reach a canonical name. Normalised the same way the input is.
_ALIASES: dict[str, str] = {
    "glycerol": "glycerine",
    "glycerin": "glycerine",
    "oliveoil": "olive oil",
    "olive": "olive oil",
    "h2o": "water",
    "he": "helium",
}


def known_fluids() -> tuple[str, ...]:
    """The canonical names, in the order :data:`FLUIDS` declares them."""
    return tuple(FLUIDS)


def _normalise(name: str) -> str:
    """``" Olive_Oil "`` -> ``"olive oil"``. Case, padding, and separators."""
    cleaned = name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def fluid(name: "str | Fluid | Quantity") -> Fluid:
    """``"Air"``, ``" air "``, or a viscosity of your own -> a :class:`Fluid`.

    Args:
        name: a library name (case-insensitive, padding and ``_``/``-``
            separators tolerated), a :class:`~flow.quantity.Quantity` that is a
            kinematic viscosity, a string that parses as one
            (``"1.5e-5 m^2/s"``), or an already-built :class:`Fluid`.

    Returns:
        The library entry, or a :class:`Fluid` named ``"custom"`` carrying the
        given viscosity and no density or temperature — a custom fluid does not
        touch :data:`FLUIDS`.

    Raises:
        ValueError: for an unknown name, listing every known one. It does not
            guess at near-misses: a suggestion that is wrong is worse than a
            list that is complete (**D-045** — the message names the way
            forward).
    """
    if isinstance(name, Fluid):
        return name

    if isinstance(name, Quantity):
        return _custom(name)

    if not isinstance(name, str):
        raise ValueError(
            f"cannot read {name!r} (a {type(name).__name__}) as a fluid. Give a "
            f"name -- one of: {', '.join(known_fluids())} -- or a kinematic "
            f'viscosity such as "1.5e-5 m^2/s".'
        )

    key = _normalise(name)
    key = _ALIASES.get(key.replace(" ", ""), _ALIASES.get(key, key))
    if key in FLUIDS:
        return FLUIDS[key]

    # Not a name. It may still be a viscosity the user typed directly, which is
    # a different thing from an unknown name and is accepted as such.
    try:
        quantity = parse(name, expect=VISCOSITY)
    except ValueError:
        raise ValueError(
            f"unknown fluid {name!r}. Known fluids: "
            f"{', '.join(known_fluids())}. Or give a kinematic viscosity "
            f'directly, e.g. fluid("1.5e-5 m^2/s").'
        ) from None
    return _custom(quantity)


def _custom(viscosity: Quantity) -> Fluid:
    """Wrap a user-supplied kinematic viscosity as a :class:`Fluid`."""
    if viscosity.dimension != VISCOSITY:
        raise ValueError(
            f"{viscosity.given!r} is a {viscosity.dimension}, but a fluid needs "
            f'a {VISCOSITY} — for example "1.5e-5 m^2/s". Or name a fluid: '
            f"{', '.join(known_fluids())}."
        )
    return Fluid(
        name="custom",
        nu=viscosity,
        rho=None,
        T=None,
        source=f"user-supplied kinematic viscosity {viscosity.given!r}",
    )
