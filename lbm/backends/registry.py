"""Name -> backend, with an honest error for one that is not installed.

Implements ``DOCS/TASKS2.md`` § T101: "``lbm/backends/registry.py`` mapping a
name to an implementation and raising a message naming the install line for an
unavailable one".

Two kinds of name, two kinds of failure, and the difference matters to whoever
reads the traceback:

* **Unknown** — nothing in this project answers to it. A typo, or a backend from
  a later phase. :func:`get_backend` raises :class:`ValueError` naming what was
  asked for and listing what exists.
* **Known but unavailable** — the implementation exists (or will) but its
  dependency is not installed. The error names the install line, because "no
  backend named warp" would be a lie: there is one, it needs ``warp-lang``.

:class:`BackendUnavailableError` subclasses :class:`ValueError` so a caller that
only wants "that name did not work" catches one exception type, while a caller
that wants to distinguish "install it" from "you typed it wrong" can.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from lbm.backends import Backend

__all__ = [
    "BackendUnavailableError",
    "available_backends",
    "get_backend",
    "known_backends",
]

#: Backend name -> ``(module, class, install line)``. Every backend this project
#: knows about lives here, installed or not. **T102 adds nothing to this table**
#: — the ``warp`` row is already correct — it only creates the module the row
#: points at (``DOCS/PLAN2.md`` § Session map, session 14).
_BACKENDS: dict[str, tuple[str, str, str | None]] = {
    "numpy": ("lbm.backends.numpy_backend", "NumpyBackend", None),
    "warp": (
        "lbm.backends.warp_backend",
        "WarpBackend",
        "myenv/Scripts/pip.exe install warp-lang",
    ),
}

#: The default. ``SimConfig.backend`` carries the same string, and NumPy stays
#: the reference oracle rather than becoming a legacy path (**D-043**).
DEFAULT_BACKEND: str = "numpy"


class BackendUnavailableError(ValueError):
    """A known backend whose dependency is not installed.

    A :class:`ValueError`, so ``pytest.raises(ValueError)`` and any caller that
    treats a bad backend name as one failure mode both still work.
    """


def known_backends() -> list[str]:
    """Every backend name this project recognises, installed or not.

    Returns:
        Sorted names, e.g. ``["numpy", "warp"]``.
    """
    return sorted(_BACKENDS)


def available_backends() -> list[str]:
    """The backend names that can actually be constructed right now.

    Import is the test — a name is available when its module imports and its
    class is there. That keeps the list honest on a machine where ``warp-lang``
    is installed but broken, which ``DOCS/PLAN2.md`` § Risks expects to happen
    at least once.

    Returns:
        Sorted names of importable backends.
    """
    out: list[str] = []
    for name in sorted(_BACKENDS):
        try:
            _construct(name)
        except Exception:  # noqa: BLE001 - unavailability is not an error here
            continue
        out.append(name)
    return out


def _construct(name: str) -> "Backend":
    """Import and instantiate one backend, letting failures propagate.

    Args:
        name: a key of :data:`_BACKENDS`.

    Returns:
        A fresh backend instance.
    """
    module_name, class_name, _ = _BACKENDS[name]
    module = importlib.import_module(module_name)
    factory: Callable[[], "Backend"] = getattr(module, class_name)
    return factory()


def get_backend(name: str = DEFAULT_BACKEND) -> "Backend":
    """The backend registered under ``name``.

    A fresh instance per call. :class:`lbm.runner.Sim` holds exactly one for its
    lifetime, so a device backend may allocate a context in its constructor
    without that becoming a per-step cost.

    Args:
        name: registry key, e.g. ``"numpy"``. Case- and whitespace-sensitive on
            purpose: silently accepting ``" NumPy "`` would hide a config that
            does not say what the user thinks it says.

    Returns:
        A backend satisfying :class:`lbm.backends.Backend`.

    Raises:
        ValueError: if ``name`` is not a known backend. The message names the
            request and lists what is available.
        BackendUnavailableError: if ``name`` is known but its dependency is not
            installed. The message names the install line.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown backend {name!r}; available backends are "
            f"{available_backends()} (known: {known_backends()})."
        )

    try:
        return _construct(name)
    except Exception as exc:  # noqa: BLE001 - re-raised with the install line
        install = _BACKENDS[name][2]
        hint = (
            f" Install it with: {install}"
            if install is not None
            else " It has no optional dependency, so this is a bug, not a"
            " missing install."
        )
        raise BackendUnavailableError(
            f"backend {name!r} is known but not available ({exc}).{hint} "
            f"Available backends: {available_backends()}."
        ) from exc
