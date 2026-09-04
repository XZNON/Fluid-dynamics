"""Rung I — one command installs it, off this tree.

``DOCS/IDEA4.md`` § Validation ladder, Rung I: *"A **fresh venv**, ``pip install
<built wheel>``, then ``fengdong --version`` and a headless smoke of the app's
model layer. No repo on the path."* And § The five things Phase 2 must get
right (5): *"a package that only installs from the developer's tree is not
distributed."* Constraint 20 stops being a design rule here and becomes a
measurement (**D-083**).

What this script asserts, in order:

1. **The build.** ``python -m build`` on this tree produces one wheel and one
   sdist under ``dist/``.
2. **The wheel's contents.** It ships ``lbm``, ``flow`` and ``fengdong`` — every
   ``__init__.py`` included, which is what queued issue ``495777c58269`` was
   about — and **none** of :data:`EXCLUDED_TOP_LEVEL`. Read from the zip, not
   from the build log.
3. **The install.** A venv is created **fresh** in a temporary directory and
   the wheel is installed into it with that venv's own pip. No ``-e``, no
   ``PYTHONPATH``, no ``.pth`` file pointing back here.
4. **The command.** ``<venv>/Scripts/fengdong --version`` answers with the
   version the tree declares — the console entry point survived the wheel.
5. **The model layer, headless.** A child process run **from outside the
   repository** imports ``lbm``, ``flow`` and ``fengdong``, and asserts —
   *inside the child, not from here* — that no entry of its ``sys.path``
   resolves into this repository and that every one of the three packages
   was loaded from the venv's own prefix. It then re-runs the constraint-15
   and constraint-17 import scans over the **installed** files, writes a
   PNG of a disc with Pillow, and drives it through :class:`flow.case.Case`
   on the NumPy backend for :data:`SMOKE_CONVECTIVE_TIMES` — a picture and
   three physical numbers (constraint 13), no repo asset, to a finite state
   inside constraint 3's ceiling. That is the "first answer" the clock stops
   on.
6. **The clock.** Venv creation + install + ``--version`` + the smoke, against
   :data:`TIME_LIMIT_SECONDS` (``DOCS/IDEA4.md`` § Performance budget, "The
   install"). The build is timed and printed but **not** gated: it is the
   developer's cost, not the user's. Every absolute figure is printed with the
   CPU clock, the power state and the GPU name beside it (**D-035**), via
   :func:`bench.machine_state` — the *shared* probe, deliberately not a third
   implementation (queued issue ``d5b27e51fcdc``).

Timing on this machine is a **warm pip cache** claim, and the machine should be
idled first (**D-092**); a reading taken under load is re-run and both are
recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from bench import machine_state, print_machine_state
from fengdong import __version__ as TREE_VERSION

#: The tree this rung packages — the directory that must **not** be on the
#: installed child's ``sys.path``.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Where ``python -m build`` writes; gitignored, cleared before every build.
DIST_DIR: Path = REPO_ROOT / "dist"

#: The three packages the wheel must contain (``DOCS/TASKS3.md`` § T205).
REQUIRED_PACKAGES: tuple[str, ...] = ("lbm", "flow", "fengdong")

#: Files that prove the packaging is complete rather than merely present:
#: ``lbm/backends/__init__.py`` is the Backend protocol that ``.gitignore``
#: untracked for twelve sessions (``495777c58269``); the other two are the
#: front doors of ``flow`` and the console entry point.
REQUIRED_FILES: tuple[str, ...] = (
    "lbm/__init__.py",
    "lbm/backends/__init__.py",
    "lbm/backends/registry.py",
    "lbm/backends/numpy_backend.py",
    "lbm/backends/warp_backend.py",
    "flow/__init__.py",
    "flow/__main__.py",
    "fengdong/__init__.py",
    "fengdong/__main__.py",
)

#: Top-level names that must not appear in the wheel. The contract's seven,
#: plus the other root directories that are not modules (``CLAUDE.md``
#: § Everything else at the root).
EXCLUDED_TOP_LEVEL: tuple[str, ...] = (
    "validate",
    "tests",
    "DOCS",
    "myenv",
    "outputs",
    "Navier-Fluid-Equation",
    "scripts",
    "tools",
    "examples",
    "PROMPTS",
    "old-Docs",
)

#: ``DOCS/IDEA4.md`` § Performance budget: install to first answer, warm cache.
TIME_LIMIT_SECONDS: float = 60.0

#: Run length of the smoke in convective times ``D / U``. Half a convective
#: time is 300 steps at ``quality="fast"`` — about five seconds on NumPy —
#: which is enough to prove the installed solver steps a real case to a
#: finite state and small enough to leave the minute mostly to pip. It is not
#: a physics claim: Rungs 3, E and H make those, and this rung's job is the
#: box, not the contents.
SMOKE_CONVECTIVE_TIMES: float = 0.5

#: The case the smoke drives: Rung E's own three numbers (**D-074**), water at
#: 5 mm/s past a 2 cm body, Re 99.6. No lattice quantity (constraint 13).
SMOKE_FLUID: str = "water"
SMOKE_SPEED: str = "5 mm/s"
SMOKE_SIZE: str = "2 cm"
SMOKE_QUALITY: str = "fast"

#: The child. Run as ``<venv python> - <repo root>`` with its cwd **outside**
#: the repository, so ``sys.path[0]`` (the empty string, the cwd) cannot reach
#: it either. Prints one JSON object on its last line; everything it asserts
#: is asserted *here*, in the installed interpreter, and reported as data.
SMOKE_SCRIPT: str = r'''
import ast, json, os, sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
report = {"version_in_child": None}


def under(path, root):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


# 1. No repository directory on sys.path -- asserted here, not assumed.
offenders = [p for p in sys.path if p and under(p, repo)]
if "" in sys.path and under(os.getcwd(), repo):
    offenders.append("<cwd>")
report["sys_path_offenders"] = offenders
report["sys_path_clean"] = not offenders

import numpy as np
import lbm
import lbm.backends
import flow
import fengdong

prefix = Path(sys.prefix).resolve()
locations = {m.__name__: str(Path(m.__file__).resolve()) for m in (lbm, flow, fengdong)}
report["locations"] = locations
report["installed_under_prefix"] = all(
    under(f, prefix) and not under(f, repo) for f in locations.values()
)
report["version_in_child"] = fengdong.__version__
report["numpy_backend_available"] = "numpy" in lbm.available_backends()


# 2. Constraints 15 and 17 over the *installed* files.
def top_level_imports(root):
    names = set()
    for path in Path(root).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


lbm_imports = top_level_imports(Path(lbm.__file__).parent)
flow_imports = top_level_imports(Path(flow.__file__).parent)
report["constraint_15"] = "flow" not in lbm_imports and "fengdong" not in lbm_imports
report["constraint_17"] = "fengdong" not in flow_imports

# 3. The model layer: a picture and three numbers, from a PNG this child
#    draws itself (dark = solid), through flow.Case on the NumPy backend.
from PIL import Image

h = w = 120
yy, xx = np.ogrid[:h, :w]
disc = (yy - 60) ** 2 + (xx - 60) ** 2 <= 40 ** 2
png = Path.cwd() / "disc.png"
Image.fromarray(np.where(disc, 0, 255).astype(np.uint8), mode="L").save(png)

case = flow.Case.from_image(
    str(png),
    fluid=%(fluid)r,
    speed=%(speed)r,
    size=%(size)r,
    quality=%(quality)r,
    backend="numpy",
)
report["runnable"] = bool(case.runnable)
plan = case.plan
report["plan"] = None if plan is None else {
    "Re": float(plan.Re),
    "tau": float(plan.tau),
    "domain": list(plan.domain),
    "cs_smag": float(plan.cs_smag),
}
report["explain_chars"] = len(case.explain(quiet=True))
seconds = %(convective)r * (case.size.si / case.speed.si)
result = case.run(seconds=seconds, keep_frames=False, quiet=True)
report["steps"] = int(result.steps)
report["peak_u"] = float(result.peak_u)
report["finite"] = bool(np.isfinite(result.peak_u))
report["fidelity"] = str(result.fidelity.value)
report["summary_chars"] = len(result.summary())
report["ok"] = all([
    report["sys_path_clean"],
    report["installed_under_prefix"],
    report["numpy_backend_available"],
    report["constraint_15"],
    report["constraint_17"],
    report["runnable"],
    report["steps"] > 0,
    report["finite"],
    report["peak_u"] < 0.1,
])
print(json.dumps(report))
'''


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None,
         stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` to completion, capturing text; never raises on exit code."""
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _child_env() -> dict[str, str]:
    """An environment with nothing that could smuggle the repo onto the path.

    ``PYTHONPATH`` is dropped, ``PYTHONSAFEPATH`` is set so the child's
    ``sys.path[0]`` is not even the cwd, and SDL is told there is no display
    (the smoke never opens one; this is belt and braces for pygame's import).
    """
    env = {k: v for k, v in os.environ.items() if k.upper() != "PYTHONPATH"}
    env["PYTHONSAFEPATH"] = "1"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def build(out_dir: Path = DIST_DIR, *, isolation: bool = True) -> tuple[Path, Path, float]:
    """``python -m build`` the tree into ``out_dir``.

    Args:
        out_dir: cleared first, so a stale wheel cannot pass for a fresh one.
        isolation: ``True`` runs the build in PEP 517's isolated environment —
            what a user's ``pip install`` does from an sdist; ``False`` passes
            ``--no-isolation`` and uses this interpreter's setuptools, which is
            faster and is what ``--fast-build`` selects.

    Returns:
        ``(wheel, sdist, seconds)``.

    Raises:
        RuntimeError: if the build fails or does not produce exactly one of each.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cmd = [sys.executable, "-m", "build", "--outdir", str(out_dir)]
    if not isolation:
        cmd.append("--no-isolation")
    t0 = time.perf_counter()
    proc = _run(cmd, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"python -m build failed ({proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    wheels = sorted(out_dir.glob("*.whl"))
    sdists = sorted(out_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            f"expected one wheel and one sdist in {out_dir}, found {wheels} and {sdists}"
        )
    return wheels[0], sdists[0], elapsed


def inspect_wheel(wheel: Path) -> dict[str, object]:
    """Read the wheel's file list and judge it against the contract.

    Returns:
        ``{"top_level", "missing_packages", "missing_files", "excluded_present",
        "ok"}``. ``top_level`` is every top-level name in the archive with the
        ``.dist-info`` directory removed.
    """
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    top_level = sorted(
        {n.split("/")[0] for n in names if not n.split("/")[0].endswith(".dist-info")}
    )
    missing_packages = [p for p in REQUIRED_PACKAGES if p not in top_level]
    missing_files = [f for f in REQUIRED_FILES if f not in names]
    excluded_present = [
        e for e in EXCLUDED_TOP_LEVEL
        if any(n == e or n.startswith(e + "/") for n in names)
    ]
    return {
        "top_level": top_level,
        "missing_packages": missing_packages,
        "missing_files": missing_files,
        "excluded_present": excluded_present,
        "ok": not (missing_packages or missing_files or excluded_present),
    }


def venv_python(venv: Path) -> Path:
    """The interpreter inside ``venv``, on this platform."""
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def venv_script(venv: Path, name: str) -> Path:
    """The console script ``name`` inside ``venv``, on this platform."""
    if sys.platform == "win32":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def create_venv(venv: Path) -> float:
    """``python -m venv`` a fresh environment, with pip, and return the seconds."""
    t0 = time.perf_counter()
    proc = _run([sys.executable, "-m", "venv", str(venv)], cwd=REPO_ROOT.parent)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0 or not venv_python(venv).exists():
        raise RuntimeError(f"venv creation failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return elapsed


def pip_install(venv: Path, wheel: Path) -> tuple[float, str]:
    """Install ``wheel`` into ``venv`` with that venv's own pip; warm cache assumed.

    Returns:
        ``(seconds, pip's last output line)``.
    """
    t0 = time.perf_counter()
    proc = _run(
        [str(venv_python(venv)), "-m", "pip", "install",
         "--no-warn-script-location", "--disable-pip-version-check", str(wheel)],
        cwd=venv,
        env=_child_env(),
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"pip install failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return elapsed, (lines[-1] if lines else "")


def run_version(venv: Path, cwd: Path) -> tuple[str, float]:
    """``fengdong --version`` through the venv's console script, from ``cwd``."""
    t0 = time.perf_counter()
    proc = _run([str(venv_script(venv, "fengdong")), "--version"], cwd=cwd, env=_child_env())
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"fengdong --version failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout.strip(), elapsed


def run_smoke(venv: Path, cwd: Path) -> tuple[dict[str, object], float]:
    """Run :data:`SMOKE_SCRIPT` in the venv's interpreter from ``cwd``.

    Returns:
        ``(the child's report, seconds)``.
    """
    script = SMOKE_SCRIPT % {
        "fluid": SMOKE_FLUID,
        "speed": SMOKE_SPEED,
        "size": SMOKE_SIZE,
        "quality": SMOKE_QUALITY,
        "convective": SMOKE_CONVECTIVE_TIMES,
    }
    t0 = time.perf_counter()
    proc = _run(
        [str(venv_python(venv)), "-", str(REPO_ROOT)],
        cwd=cwd, env=_child_env(), stdin=script,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"smoke failed ({proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    last = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return json.loads(last), elapsed


# ---------------------------------------------------------------------------
# The rung
# ---------------------------------------------------------------------------


def _flag(ok: bool) -> str:
    return "ok" if ok else "FAIL"


def main(argv: list[str] | None = None) -> int:
    """Run Rung I and print PASS/FAIL.

    Returns:
        Process exit code — ``0`` PASS, ``1`` FAIL.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-build", action="store_true",
        help="build with --no-isolation (this interpreter's setuptools) rather "
             "than in PEP 517's isolated environment; the build is not gated "
             "either way",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="leave the temporary venv on disk and print its path",
    )
    args = parser.parse_args(argv)

    print("Rung I - one command installs it, off this tree (constraint 20, D-083)")
    print(f"  tree:    {REPO_ROOT}")
    print(f"  version: {TREE_VERSION} (fengdong.__version__)")
    print()

    # 1. build
    wheel, sdist, t_build = build(isolation=not args.fast_build)
    print("1. the build - python -m build")
    print(f"   wheel {wheel.name}")
    print(f"   sdist {sdist.name}")
    print(f"   {t_build:.1f} s ({'--no-isolation' if args.fast_build else 'isolated'}; "
          "the developer's cost, printed and not gated)")

    # 2. contents
    contents = inspect_wheel(wheel)
    print()
    print("2. the wheel's contents - read from the zip")
    print(f"   top level: {', '.join(contents['top_level'])}")
    print(f"   ships {', '.join(REQUIRED_PACKAGES)}   "
          f"[{_flag(not contents['missing_packages'])}]"
          + (f"  missing {contents['missing_packages']}" if contents["missing_packages"] else ""))
    print(f"   every required file present ({len(REQUIRED_FILES)}, incl. lbm/backends/__init__.py)   "
          f"[{_flag(not contents['missing_files'])}]"
          + (f"  missing {contents['missing_files']}" if contents["missing_files"] else ""))
    print(f"   none of {', '.join(EXCLUDED_TOP_LEVEL)}   "
          f"[{_flag(not contents['excluded_present'])}]"
          + (f"  found {contents['excluded_present']}" if contents["excluded_present"] else ""))

    tmp_root = Path(tempfile.mkdtemp(prefix="fengdong-rung-i-"))
    venv = tmp_root / "venv"
    work = tmp_root / "work"
    work.mkdir()
    version_ok = smoke_ok = time_ok = False
    elapsed_total = float("nan")
    try:
        # 3. fresh venv + install -- the clock starts here.
        t_clock = time.perf_counter()
        t_venv = create_venv(venv)
        t_pip, pip_last = pip_install(venv, wheel)
        print()
        print("3. the install - a fresh venv, that venv's own pip, the wheel")
        print(f"   venv {venv}")
        print(f"   python -m venv          {t_venv:.1f} s")
        print(f"   pip install <wheel>     {t_pip:.1f} s   ({pip_last})")

        # 4. the command
        version_out, t_version = run_version(venv, work)
        version_ok = version_out == f"fengdong {TREE_VERSION}"
        print()
        print("4. the command - <venv>/Scripts/fengdong --version, cwd outside the repo")
        print(f"   printed {version_out!r} in {t_version:.2f} s, expected 'fengdong {TREE_VERSION}'   "
              f"[{_flag(version_ok)}]")

        # 5. the model layer, headless, from outside the repo
        report, t_smoke = run_smoke(venv, work)
        elapsed_total = time.perf_counter() - t_clock
        smoke_ok = bool(report.get("ok"))
        print()
        print("5. the model layer - a child process, cwd outside the repo, PYTHONSAFEPATH=1")
        print(f"   no repository directory on sys.path (asserted in the child)   "
              f"[{_flag(bool(report['sys_path_clean']))}]"
              + (f"  offenders {report['sys_path_offenders']}" if report["sys_path_offenders"] else ""))
        print(f"   lbm, flow, fengdong loaded from the venv's prefix   "
              f"[{_flag(bool(report['installed_under_prefix']))}]")
        for name, loc in report["locations"].items():  # type: ignore[union-attr]
            print(f"     {name:<9}{loc}")
        print(f"   lbm.backends imports and 'numpy' is available   "
              f"[{_flag(bool(report['numpy_backend_available']))}]")
        print(f"   constraint 15 in the installed lbm/ (no flow, no fengdong)   "
              f"[{_flag(bool(report['constraint_15']))}]")
        print(f"   constraint 17 in the installed flow/ (no fengdong)   "
              f"[{_flag(bool(report['constraint_17']))}]")
        plan = report.get("plan") or {}
        print(f"   {SMOKE_FLUID}, {SMOKE_SPEED}, {SMOKE_SIZE}, quality {SMOKE_QUALITY!r} from a PNG the "
              f"child drew: runnable   [{_flag(bool(report['runnable']))}]")
        if plan:
            ny, nx = plan["domain"]
            print(f"     Re {plan['Re']:.1f}, tau {plan['tau']:.4f}, grid {ny}x{nx}, "
                  f"Cs {plan['cs_smag']:g}; explain() {report['explain_chars']} chars")
        print(f"   {report['steps']} steps on numpy in {t_smoke:.1f} s (child total), "
              f"peak |u| {report['peak_u']:.5f} of the 0.1 ceiling, finite, "
              f"fidelity {report['fidelity']}   "
              f"[{_flag(bool(report['finite']) and report['peak_u'] < 0.1 and report['steps'] > 0)}]")

        # 6. the clock
        time_ok = elapsed_total < TIME_LIMIT_SECONDS
        print()
        print("6. the clock - venv + install + --version + smoke (the build excluded)")
        print(f"   {elapsed_total:.1f} s wall clock, limit {TIME_LIMIT_SECONDS:.0f} s "
              f"(warm pip cache)   [{_flag(time_ok)}]")
        print()
        print_machine_state(machine_state(), "numpy")
    except RuntimeError as exc:
        print()
        print(f"   FAIL: {exc}")
    finally:
        if args.keep:
            print(f"\n   kept: {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    ok = bool(contents["ok"]) and version_ok and smoke_ok and time_ok
    print()
    print(f"Rung I: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
