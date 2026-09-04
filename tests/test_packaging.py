"""T205 — the ``fengdong`` distribution, checked against the documents that govern it.

``DOCS/TASKS3.md`` § T205, acceptance criteria 1 and 5, and the constraints
that bite there:

* **Every runtime dependency matches a row in ``DOCS/STATE3.md`` § Environment**
  — "no dependency appears in the package that was not installed and recorded
  in a session. A test asserts the two lists agree." That test is here, and it
  reads *both* files rather than a copy of either: ``pyproject.toml`` through
  :mod:`tomllib` and the Environment table by parsing its rows.
* **Optional extras are declared for what is genuinely optional** — the Warp
  backend as ``[gpu]``, recording as ``[video]``; the base install runs the
  NumPy backend and the app.
* **Constraint 17** — ``fengdong/`` may import ``flow/``; ``flow/`` may never
  import ``fengdong/``. Asserted here in the shape of ``tests/test_flow_package.py``'s
  constraint-15 scan, and again by Rung I inside the installed wheel.
* **Constraint 20** — the pieces a wheel needs are *tracked*: queued issue
  ``495777c58269`` (``.gitignore`` dropped every ``__init__.py`` and ``tools/``)
  is closed by this task, and a test keeps it closed.

Nothing here builds a wheel: that is Rung I (``validate/install.py``), and it
costs a venv. These tests are the cheap, always-on half.
"""

from __future__ import annotations

import ast
import importlib
import io
import pkgutil
import re
import subprocess
import sys
import tomllib
from contextlib import redirect_stdout
from fnmatch import fnmatch
from pathlib import Path

import pytest

import fengdong
import flow
import lbm
from validate import install as rung_i

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
STATE3 = REPO_ROOT / "DOCS" / "STATE3.md"


# ---------------------------------------------------------------------------
# Reading the two documents
# ---------------------------------------------------------------------------


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _normalise(name: str) -> str:
    """PEP 503 normalisation, so ``Pillow`` / ``pillow`` / ``imageio_ffmpeg`` agree."""
    return re.sub(r"[-_.]+", "-", name).lower()


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


def _split_requirement(req: str) -> tuple[str, str]:
    """``"numpy>=2.4.6"`` -> ``("numpy", ">=2.4.6")``. Extras markers are not used here."""
    m = _REQ.match(req)
    assert m, req
    return _normalise(m.group(1)), m.group(3).strip()


def _environment_table() -> dict[str, str]:
    """``DOCS/STATE3.md`` § Environment as ``{normalised name: version}``.

    Parses the markdown table between the ``## Environment`` heading and the
    next ``## `` heading: rows shaped ``| package | version | added by |``.
    """
    text = STATE3.read_text(encoding="utf-8")
    start = text.index("\n## Environment")
    end = text.index("\n## ", start + 1)
    rows: dict[str, str] = {}
    for line in text[start:end].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in ("Package", "") or set(cells[0]) <= {"-"}:
            continue
        if re.fullmatch(r"\d+(\.\d+)*", cells[1]):
            rows[_normalise(cells[0])] = cells[1]
    assert rows, "the Environment table did not parse"
    return rows


def _declared() -> dict[str, dict[str, str]]:
    """Every requirement ``pyproject.toml`` declares, grouped: ``{group: {name: spec}}``."""
    project = _pyproject()["project"]
    groups: dict[str, dict[str, str]] = {"runtime": {}}
    for req in project["dependencies"]:
        name, spec = _split_requirement(req)
        groups["runtime"][name] = spec
    for extra, reqs in project.get("optional-dependencies", {}).items():
        groups[extra] = dict(_split_requirement(r) for r in reqs)
    return groups


# ---------------------------------------------------------------------------
# Criterion 1 — the dependency lists agree
# ---------------------------------------------------------------------------


def test_the_distribution_is_named_fengdong():
    """**D-083**: ``flow`` is taken on PyPI; ``fengdong`` is the name and the command."""
    project = _pyproject()["project"]
    assert project["name"] == "fengdong"
    assert project["scripts"] == {"fengdong": "fengdong.__main__:main"}


def test_every_declared_dependency_is_a_row_in_the_environment_table():
    """No package in the distribution that ``myenv`` never had (PLAN3 § Risks)."""
    table = _environment_table()
    for group, reqs in _declared().items():
        for name in reqs:
            assert name in table, (
                f"pyproject.toml [{group}] declares {name!r}, which is not a row in "
                "DOCS/STATE3.md § Environment - install it and add the row in the same session"
            )


def test_every_declared_lower_bound_is_the_recorded_version():
    """The version a session installed and recorded is the only one this project has run on.

    ``>=`` at exactly that version: nothing older is claimed, and nothing newer
    is forbidden, because a pin would make the *wheel* fail on a machine with a
    newer numpy for no measured reason.
    """
    table = _environment_table()
    for group, reqs in _declared().items():
        for name, spec in reqs.items():
            assert spec == f">={table[name]}", (
                f"[{group}] {name}: pyproject says {spec!r}, the Environment table records "
                f"{table[name]!r}"
            )


def test_the_build_backend_is_a_row_too():
    """``[build-system] requires`` is a dependency of the *build*; same rule, same table."""
    table = _environment_table()
    for req in _pyproject()["build-system"]["requires"]:
        name, spec = _split_requirement(req)
        assert name in table, f"build-system requires {name!r}, not in the Environment table"
        assert spec == f">={table[name]}"


def test_every_environment_row_has_a_home_in_pyproject():
    """The other direction: a recorded package that the distribution silently dropped.

    Every row is either runtime, an extra, a dev tool, or the build backend.
    A row in none of them is a dependency the tree has and the wheel does not,
    and Rung I would be the first to find out.
    """
    table = _environment_table()
    declared = {n for reqs in _declared().values() for n in reqs}
    declared |= {_split_requirement(r)[0] for r in _pyproject()["build-system"]["requires"]}
    missing = sorted(set(table) - declared)
    assert not missing, f"Environment rows with no home in pyproject.toml: {missing}"


def test_the_environment_table_parser_would_actually_catch_a_drift():
    """A guard that never fires is not a guard."""
    table = _environment_table()
    assert table["numpy"].startswith("2."), table
    assert "warp-lang" in table and "imageio-ffmpeg" in table
    assert _normalise("imageio_ffmpeg") == "imageio-ffmpeg"
    assert _split_requirement("Pillow>=12.3.0") == ("pillow", ">=12.3.0")
    assert _split_requirement("warp-lang>=1.16.0") == ("warp-lang", ">=1.16.0")


# ---------------------------------------------------------------------------
# Criterion 5 — the extras are the optional things and nothing else
# ---------------------------------------------------------------------------


def test_the_runtime_set_is_what_the_base_install_needs_and_no_more():
    """Read from the imports, not the habits: numpy everywhere, pillow for the
    picture in, matplotlib for ``Result.plot``, pygame for the app and the live
    window. Nothing that only ``validate/``, ``tests/`` or ``bench.py`` imports."""
    assert set(_declared()["runtime"]) == {"numpy", "pillow", "matplotlib", "pygame"}


def test_gpu_and_video_extras_are_exactly_the_optional_backends():
    groups = _declared()
    assert set(groups["gpu"]) == {"warp-lang"}, "the Warp backend is fengdong[gpu]"
    assert set(groups["video"]) == {"imageio", "imageio-ffmpeg"}, "recording is fengdong[video]"


def test_dev_tools_are_not_runtime_dependencies():
    """``pytest``, ``psutil`` (Rung E's process clock) and ``build`` are the tree's, not the wheel's."""
    groups = _declared()
    assert set(groups["dev"]) == {"pytest", "psutil", "build"}
    for tool in ("pytest", "psutil", "build"):
        assert tool not in groups["runtime"]


def test_nothing_shipped_imports_a_dev_or_extra_package_at_module_scope():
    """The runtime split holds in the source, not only in the manifest.

    ``warp`` / ``imageio`` / ``imageio_ffmpeg`` may appear only inside a function
    (lazy, with an install hint) or in ``lbm/backends/warp_backend.py`` itself;
    ``pytest`` and ``psutil`` may not appear at all in the three packages.
    """
    banned_anywhere = {"pytest", "psutil", "build"}
    lazy_only = {"warp", "imageio", "imageio_ffmpeg"}
    for package in (lbm, flow, fengdong):
        for path in Path(package.__file__).parent.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for n in names:
                    assert n not in banned_anywhere, f"{path.relative_to(REPO_ROOT)} imports {n}"
                    if n in lazy_only and path.name != "warp_backend.py":
                        assert node.col_offset > 0, (
                            f"{path.relative_to(REPO_ROOT)} imports {n} at module scope; "
                            "it is an extra and must be lazy"
                        )


# ---------------------------------------------------------------------------
# Criterion 2's static half — what the package finder will and will not take
# ---------------------------------------------------------------------------


def test_only_the_three_packages_are_found():
    cfg = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    include = cfg["include"]
    for name in rung_i.REQUIRED_PACKAGES:
        assert any(fnmatch(name, pat) for pat in include), name
    for name in rung_i.EXCLUDED_TOP_LEVEL:
        assert not any(fnmatch(name, pat) for pat in include), (
            f"{name!r} would be packaged by {include}"
        )
    assert cfg.get("namespaces") is False, "an implicit namespace package would sweep in stray dirs"
    assert _pyproject()["tool"]["setuptools"].get("include-package-data") is False


def test_the_version_is_single_sourced_and_pep_440():
    project = _pyproject()
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "fengdong.__version__"}
    assert re.fullmatch(r"\d+(\.\d+)+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", fengdong.__version__)


def test_python_floor_matches_the_environment():
    assert _pyproject()["project"]["requires-python"] == ">=3.11"
    assert sys.version_info >= (3, 11)


# ---------------------------------------------------------------------------
# The entry point is real
# ---------------------------------------------------------------------------


def test_the_console_entry_point_resolves_and_prints_the_version():
    module_name, attr = _pyproject()["project"]["scripts"]["fengdong"].split(":")
    main = getattr(importlib.import_module(module_name), attr)
    out = io.StringIO()
    with redirect_stdout(out):
        code = main(["--version"])
    assert code == 0
    assert out.getvalue().strip() == f"fengdong {fengdong.__version__}"


def test_the_bare_command_opens_the_window_and_returns_its_exit_code(monkeypatch):
    """From T207 the bare command *is* the window. ``App.run`` is stubbed so
    this test opens nothing; ``tests/test_app.py`` drives the real loop under
    the dummy driver."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from fengdong.__main__ import main
    from fengdong.app import App

    monkeypatch.setattr(App, "run", lambda self: 0)
    out = io.StringIO()
    with redirect_stdout(out):
        code = main([])
    assert code == 0
    assert out.getvalue() == "", "the window prints nothing; --version is the printing path"


def test_fengdong_imports_without_numpy_flow_or_a_display():
    """``fengdong --version`` on a machine where numpy is broken should still answer.

    Run in a subprocess so this process's already-imported modules cannot mask it.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import fengdong.__main__ as m; "
         "assert 'numpy' not in sys.modules and 'flow' not in sys.modules and "
         "'pygame' not in sys.modules, sorted(k for k in sys.modules if k in ('numpy','flow','pygame')); "
         "print(m.version_line())"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"fengdong {fengdong.__version__}"


# ---------------------------------------------------------------------------
# Constraint 17 — the import goes one way, and only one way
# ---------------------------------------------------------------------------


def _imports_of(source: str, label: str, package: str) -> list[str]:
    """Every absolute import of ``package`` in ``source``, at any depth."""
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == package:
                    offenders.append(f"{label}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] == package:
                offenders.append(f"{label}: from {node.module} import ...")
    return offenders


def _modules(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_under_flow_imports_fengdong():
    """The assertion **D-083** promised, over every file in ``flow/``."""
    offenders: list[str] = []
    for path in _modules(Path(flow.__file__).parent):
        offenders += _imports_of(path.read_text(encoding="utf-8"), path.name, "fengdong")
    assert not offenders, f"flow/ imports fengdong/, which breaks constraint 17 and D-083: {offenders}"


def test_no_module_under_lbm_imports_fengdong_either():
    """Constraint 15's sibling: the solver knows nothing about the app."""
    offenders: list[str] = []
    for path in _modules(Path(lbm.__file__).parent):
        offenders += _imports_of(path.read_text(encoding="utf-8"), path.name, "fengdong")
    assert not offenders, offenders


def test_the_constraint_17_scan_would_actually_catch_a_violation():
    assert _imports_of("import fengdong", "x", "fengdong") == ["x: import fengdong"]
    assert _imports_of("from fengdong.widgets import Label", "x", "fengdong") == [
        "x: from fengdong.widgets import ..."
    ]
    assert _imports_of("def f():\n    import fengdong.app\n", "x", "fengdong") == [
        "x: import fengdong.app"
    ]
    assert _imports_of("from flow.case import Case", "x", "fengdong") == []


def test_importing_every_flow_module_does_not_pull_in_fengdong():
    """The runtime half of constraint 17."""
    for module_info in pkgutil.walk_packages(flow.__path__, prefix="flow."):
        __import__(module_info.name)
        module = sys.modules[module_info.name]
        for name, value in vars(module).items():
            assert getattr(value, "__module__", "").split(".")[0] != "fengdong", (
                f"{module_info.name}.{name} came from fengdong/"
            )


def test_fengdong_is_allowed_to_import_flow():
    """The legal direction, so the scan is known to be one-way and not a ban on both."""
    assert _imports_of("from flow.autoconfig import plan", "x", "flow") == [
        "x: from flow.autoconfig import ..."
    ]
    # No fengdong module may reach lbm's render path except through flow (constraint 10 posture):
    # today the skeleton imports neither; this pins that T206+ add `flow`, not `lbm.render`.
    for path in _modules(Path(fengdong.__file__).parent):
        source = path.read_text(encoding="utf-8")
        assert not _imports_of(source, path.name, "lbm") or "render" not in source, path


# ---------------------------------------------------------------------------
# Constraint 20's precondition — the files a wheel needs are tracked
# ---------------------------------------------------------------------------


def _git_ls_files() -> set[str] | None:
    try:
        proc = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    return {line.strip().replace("\\", "/") for line in proc.stdout.splitlines()}


def test_every_init_and_the_issue_tool_are_tracked():
    """Closes queued issue ``495777c58269``: ``.gitignore`` no longer drops ``__init__.py``."""
    tracked = _git_ls_files()
    if tracked is None:
        pytest.skip("git not available")
    for package in (lbm, flow, fengdong):
        for init in Path(package.__file__).parent.rglob("__init__.py"):
            rel = init.relative_to(REPO_ROOT).as_posix()
            assert rel in tracked, f"{rel} is not tracked - a clean clone cannot import it"
    assert "tools/issues.py" in tracked
    assert "pyproject.toml" in tracked


def test_gitignore_no_longer_ignores_inits_or_tools():
    lines = [ln.strip() for ln in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    for bad in ("*/__init__.py", "__init__.py", "tools"):
        assert bad not in lines, f".gitignore still has {bad!r}"
    for artefact in ("dist/", "build/", "*.egg-info/"):
        assert artefact in lines


# ---------------------------------------------------------------------------
# Rung I's own invariants (the cheap ones; the venv is the rung's)
# ---------------------------------------------------------------------------


def test_rung_i_uses_the_shared_machine_probe_and_defines_no_wmi_query():
    """``d5b27e51fcdc``: two power probes already disagree; this rung does not add a third."""
    source = (REPO_ROOT / "validate" / "install.py").read_text(encoding="utf-8")
    assert "from bench import machine_state" in source
    assert "Win32_" not in source and "Get-CimInstance" not in source


def test_rung_i_excludes_what_the_contract_names():
    for name in ("validate", "tests", "DOCS", "myenv", "outputs", "Navier-Fluid-Equation", "scripts"):
        assert name in rung_i.EXCLUDED_TOP_LEVEL
    assert rung_i.REQUIRED_PACKAGES == ("lbm", "flow", "fengdong")
    assert "lbm/backends/__init__.py" in rung_i.REQUIRED_FILES
    assert rung_i.TIME_LIMIT_SECONDS == 60.0


def test_rung_i_smoke_names_no_lattice_quantity():
    """Constraint 13 through the wheel: the child's case is a fluid, a speed, a size."""
    for value in (rung_i.SMOKE_FLUID, rung_i.SMOKE_SPEED, rung_i.SMOKE_SIZE, rung_i.SMOKE_QUALITY):
        assert isinstance(value, str)
    assert "tau" not in rung_i.SMOKE_SCRIPT.split("plan =")[0]
    assert "from_image(" in rung_i.SMOKE_SCRIPT and 'backend="numpy"' in rung_i.SMOKE_SCRIPT


def test_rung_i_inspect_wheel_has_teeth(tmp_path: Path):
    """A wheel with a stray ``tests/`` or a missing ``__init__`` is judged, not trusted."""
    import zipfile

    good = tmp_path / "good.whl"
    with zipfile.ZipFile(good, "w") as zf:
        for f in rung_i.REQUIRED_FILES:
            zf.writestr(f, "")
        zf.writestr("fengdong-0.0.0.dist-info/METADATA", "")
    assert rung_i.inspect_wheel(good)["ok"]

    bad = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad, "w") as zf:
        for f in rung_i.REQUIRED_FILES:
            if f != "lbm/backends/__init__.py":
                zf.writestr(f, "")
        zf.writestr("tests/test_x.py", "")
        zf.writestr("myenv/Scripts/python.exe", "")
    verdict = rung_i.inspect_wheel(bad)
    assert not verdict["ok"]
    assert verdict["missing_files"] == ["lbm/backends/__init__.py"]
    assert verdict["excluded_present"] == ["tests", "myenv"]
