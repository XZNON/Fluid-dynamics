"""T104 — the two constraints the ``flow/`` package exists to keep.

**Constraint 15** — ``flow/`` may import ``lbm/``; ``lbm/`` may **never** import
``flow/``, and a test asserts it (**D-042**). That one-directional import is what
makes the Phase 3 XLB swap a substitution rather than a rewrite, and it is
written in the session that creates ``flow/`` rather than the session that first
needs it — an import added in T105 and caught in T108 is three sessions of
untangling.

**Constraint 13** — no lattice quantity in any public ``flow/`` signature. No
``tau``, no lattice ``U``, no ``steps_per_frame``, no cell counts. The inputs
are a picture, a fluid, a speed, a size; everything else is derived and printed
(``DOCS/IDEA3.md`` § The five things Phase 1 must get right (1)).

Both are read from the **source**, in the shape
``tests/test_backends.py::test_the_runner_module_imports_no_kernel_from_lbm_core_or_lbm_boundary``
already uses, so an import or a parameter hidden inside a function is caught as
well as one at module level.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import pathlib
import pkgutil

import pytest

import flow
import lbm

FLOW_ROOT = pathlib.Path(flow.__file__).parent
LBM_ROOT = pathlib.Path(lbm.__file__).parent

#: Names that mean "this signature speaks lattice", per constraint 13. ``dx``
#: and ``dt`` are on the list because they are the grid's units, not the user's:
#: a metres-per-cell in a ``flow/`` signature is a lattice quantity wearing an
#: SI hat.
LATTICE_NAMES = frozenset(
    {
        "tau",
        "tau_floor",
        "nu_lattice",
        "u_lattice",
        "lattice_u",
        "u_max",
        "steps_per_frame",
        "cells_per_length",
        "cells",
        "n_cells",
        "ncells",
        "nx",
        "ny",
        "dx",
        "dt",
        "timesteps",
        "n_steps",
        "nsteps",
        # T204: the Smagorinsky constant. It is planned by
        # flow.autoconfig.plan and printed by --explain; it is never typed, and
        # the *fidelity band* is what surfaces to the user instead
        # (constraint 13, extended by D-082/D-083). Plan.cs_smag is a field on a
        # frozen output record, which is D-060's own exemption -- exactly where
        # tau, dx and dt already live.
        "cs_smag",
        "cs",
        "smagorinsky",
    }
)


def _modules(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _public_functions(module) -> list[tuple[str, object]]:
    """Public callables a user could reach through ``module``'s ``__all__``."""
    names = getattr(module, "__all__", None) or [
        n for n in vars(module) if not n.startswith("_")
    ]
    found: list[tuple[str, object]] = []
    for name in names:
        obj = getattr(module, name, None)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            found.append((name, obj))
    return found


# ---------------------------------------------------------------------------
# Constraint 15 — the import goes one way, and only one way
# ---------------------------------------------------------------------------


def _flow_imports_in(source: str, label: str) -> list[str]:
    """Every ``flow`` import in ``source``, at module level or inside a function."""
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "flow":
                    offenders.append(f"{label}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # A relative import cannot reach a sibling top-level package, so
            # only absolute ones (level 0) can be a constraint-15 violation.
            if node.level == 0 and node.module and node.module.split(".")[0] == "flow":
                offenders.append(f"{label}: from {node.module} import ...")
    return offenders


def test_no_module_under_lbm_imports_flow():
    """The assertion **D-042** promised, over every file in ``lbm/``."""
    offenders: list[str] = []
    for path in _modules(LBM_ROOT):
        offenders += _flow_imports_in(path.read_text(encoding="utf-8"), path.name)
    assert not offenders, (
        "lbm/ imports flow/, which breaks CLAUDE.md constraint 15 and D-042: "
        f"{offenders}"
    )


def test_the_constraint_15_scan_would_actually_catch_a_violation():
    """A guard that never fires is not a guard. Prove the scan has teeth.

    All three shapes a real violation would take, including the one hidden
    inside a function body — which is why this reads the source rather than the
    imported module's namespace.
    """
    assert _flow_imports_in("import flow", "x") == ["x: import flow"]
    assert _flow_imports_in("from flow.quantity import parse", "x") == [
        "x: from flow.quantity import ..."
    ]
    assert _flow_imports_in(
        "def f():\n    import flow.fluids\n    return flow.fluids", "x"
    ) == ["x: import flow.fluids"]
    # ...and does not fire on the legal direction.
    assert _flow_imports_in("from lbm.units import LatticeUnits", "x") == []


def test_importing_every_lbm_module_does_not_pull_in_flow():
    """The runtime half: not merely absent from the source, absent from ``lbm``.

    A module reached only through a lazy import inside a function would slip
    past the AST scan on a path nobody exercises; this catches the ones that
    are actually imported.
    """
    import sys

    for module_info in pkgutil.walk_packages(lbm.__path__, prefix="lbm."):
        if "warp" in module_info.name:
            continue  # importing the Warp backend initialises a GPU; not here.
        __import__(module_info.name)
        module = sys.modules[module_info.name]
        for name, value in vars(module).items():
            assert getattr(value, "__module__", "").split(".")[0] != "flow", (
                f"{module_info.name}.{name} came from flow/"
            )


def test_flow_is_allowed_to_import_lbm():
    """The other direction is legal — asserted so the ban is not read as mutual.

    T104 happens to need nothing from ``lbm/``; T105 onward will import
    ``lbm.units`` and ``lbm.geometry``, and that must stay legal.
    """
    from lbm.units import LatticeUnits  # noqa: F401  (the point is that it works)

    assert LatticeUnits is not None


# ---------------------------------------------------------------------------
# Constraint 13 — no lattice quantity in any public flow/ signature
# ---------------------------------------------------------------------------


def _flow_modules():
    import importlib
    import sys

    modules = [flow]
    for module_info in pkgutil.walk_packages(flow.__path__, prefix="flow."):
        importlib.import_module(module_info.name)
        modules.append(sys.modules[module_info.name])
    return modules


def _is_frozen_output_record(obj: object) -> bool:
    """A frozen dataclass is a *result*, not a command the user types (T105 Notes).

    Constraint 13 bans a lattice quantity from a signature the user **fills
    in** — what they type. ``inspect.signature`` on a class resolves to its
    ``__init__``, which for a frozen dataclass like :class:`flow.autoconfig.Plan`
    is not that: nothing in the product ever calls ``Plan(tau=..., ...)``
    directly, only :func:`flow.autoconfig.plan` builds one, and its return value
    is exactly where ``tau`` / ``dx`` / ``dt`` / ``cells_per_length`` are
    *supposed* to live (``DOCS/IDEA3.md`` § 1: "everything else is derived and
    printed"). Scanning the constructor here would flag the contract's own
    acceptance criterion. A **mutable** class, or one that defines its own
    ``__init__`` by hand, is not exempted — see
    ``test_the_constraint_13_scan_still_catches_a_hand_written_constructor``.
    """
    return dataclasses.is_dataclass(obj) and dataclasses.is_dataclass and (
        getattr(obj, "__dataclass_params__", None) is not None
        and obj.__dataclass_params__.frozen
    )


@pytest.mark.parametrize("module", _flow_modules(), ids=lambda m: m.__name__)
def test_no_public_signature_in_flow_takes_a_lattice_quantity(module):
    offenders: list[str] = []
    for name, obj in _public_functions(module):
        if inspect.isclass(obj) and _is_frozen_output_record(obj):
            pass  # constructor not scanned — see _is_frozen_output_record
        else:
            try:
                signature = inspect.signature(obj)
            except (TypeError, ValueError):  # pragma: no cover - builtins
                continue
            for parameter in signature.parameters:
                if parameter.lower() in LATTICE_NAMES:
                    offenders.append(f"{module.__name__}.{name}({parameter})")
        for member_name, member in vars(obj).items() if inspect.isclass(obj) else []:
            if member_name.startswith("_") or not callable(member):
                continue
            try:
                member_signature = inspect.signature(member)
            except (TypeError, ValueError):  # pragma: no cover
                continue
            for parameter in member_signature.parameters:
                if parameter.lower() in LATTICE_NAMES:
                    offenders.append(
                        f"{module.__name__}.{name}.{member_name}({parameter})"
                    )
    assert not offenders, (
        "CLAUDE.md constraint 13: a lattice quantity reached a public flow/ "
        f"signature: {offenders}"
    )


@pytest.mark.parametrize("module", _flow_modules(), ids=lambda m: m.__name__)
def test_no_public_attribute_in_flow_is_named_for_a_lattice_quantity(module):
    exported = getattr(module, "__all__", None) or [
        n for n in vars(module) if not n.startswith("_")
    ]
    clash = {n for n in exported if n.lower() in LATTICE_NAMES}
    assert not clash, f"{module.__name__} exports lattice names: {clash}"


def test_the_constraint_13_scan_would_actually_catch_a_violation():
    """A guard that never fires is not a guard. Prove the scan has teeth."""

    def plausible_but_illegal(speed: str, tau: float = 0.6) -> float:
        return tau

    signature = inspect.signature(plausible_but_illegal)
    caught = [p for p in signature.parameters if p.lower() in LATTICE_NAMES]
    assert caught == ["tau"]


def test_the_frozen_dataclass_exemption_is_narrow():
    """D-059: the exemption is for *output records*, not classes in general.

    A frozen dataclass — :class:`flow.autoconfig.Plan`'s shape — is exempt,
    because nothing calls its constructor with the user's input. A class that
    is **not** a frozen dataclass, even one that merely carries a lattice-named
    field, still has its hand-written ``__init__`` scanned: that is a
    constructor someone could call directly, which is exactly what constraint
    13 is about.
    """

    @dataclasses.dataclass(frozen=True)
    class FrozenResult:
        tau: float
        dx: float

    class HandWritten:
        def __init__(self, tau: float) -> None:
            self.tau = tau

    assert _is_frozen_output_record(FrozenResult) is True
    assert _is_frozen_output_record(HandWritten) is False

    hand_written_signature = inspect.signature(HandWritten)
    caught = [p for p in hand_written_signature.parameters if p.lower() in LATTICE_NAMES]
    assert caught == ["tau"], (
        "a hand-written constructor must still be caught: exempting classes in "
        "general, not just frozen output records, would be the constraint-13 "
        "scan losing its teeth"
    )


# ---------------------------------------------------------------------------
# The package itself
# ---------------------------------------------------------------------------


def test_flow_exports_what_t104_promised():
    for name in ("Quantity", "parse", "to_si", "Fluid", "FLUIDS", "fluid"):
        assert hasattr(flow, name), f"flow.{name} missing"
        assert name in flow.__all__


def test_flow_defines_no_renderer_of_its_own():
    """Constraint 10 (**D-042**): one ``render()``, and it lives in ``lbm/``.

    ``flow/`` may *call* ``lbm.render`` and, from T108, compose matplotlib
    figures for the scalar histories — what it may not do is grow a second
    field-to-RGB path. A ``def render`` here is that second path starting.
    """
    offenders: list[str] = []
    for path in _modules(FLOW_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "render",
                "to_rgb",
                "colormap",
            }:
                offenders.append(f"{path.name}: def {node.name}")
    assert not offenders, (
        f"flow/ colours nothing (CLAUDE.md constraint 10): {offenders}"
    )
