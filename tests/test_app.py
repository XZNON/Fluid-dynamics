"""T207 — ``fengdong/app.py``: the window, driven headless.

``DOCS/TASKS3.md`` § T207, one test per acceptance criterion plus the
invariants:

* the window is titled **FengDong**, has a drop target, and a dropped PNG
  previews the body :func:`flow.prepare.prepare` made — the repaired one,
  with its verdict (**D-065**, **D-066**);
* fluid, speed, size and quality go through the T206 widgets, and a bad entry
  shows the parser's message without the window crashing;
* the plan preview **is** :meth:`flow.case.Case.explain`'s text — asserted by
  string equality against the same ``Case`` — and ``app.py`` computes no
  solver parameter (constraint 17, read from the syntax);
* a refused case shows the refusal and the list :meth:`flow.case.Case.nearest`
  acts on, and the button applies that top entry (constraint 14);
* resizable — the layout is re-derived from the size and every rect stays
  inside the window at three sizes;
* closing exits cleanly, with no display left initialised and no
  ``ResourceWarning``;
* the state machine — drop, edit, plan — runs with no window opened;
* ``fengdong --version`` still answers without importing the window.

``SDL_VIDEODRIVER=dummy`` is set before pygame is imported, as
``tests/test_widgets.py`` does. The two tests that need a display surface
open one under the dummy driver and close it in a ``finally``; every other
test asserts the display was never initialised.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import warnings

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
import pytest  # noqa: E402

import fengdong.__main__ as entry  # noqa: E402
from fengdong import app as app_module  # noqa: E402
from fengdong.app import (  # noqa: E402
    MIN_SIZE,
    PROMPT_CURRENT,
    PROMPT_DROP,
    PROMPT_REFUSED,
    PROMPT_STALE,
    TITLE,
    App,
)
from fengdong.widgets import Button, Dropdown, DropTarget, Label, Panel, TextField  # noqa: E402
from flow.autoconfig import QUALITY_LEVELS  # noqa: E402
from flow.case import Case  # noqa: E402
from flow.fluids import FLUIDS, Fluid  # noqa: E402
from flow.quantity import VISCOSITY, Quantity, parse  # noqa: E402
from test_flow_package import LATTICE_NAMES  # noqa: E402
from test_widgets import _identifiers, _imports_of  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_PY = pathlib.Path(app_module.__file__)
SHAPES = REPO_ROOT / "tests" / "data" / "shapes"
DISC = SHAPES / "disc.png"
SPECKS = SHAPES / "specks.png"
ALL_WHITE = SHAPES / "all_white.png"
EXTREME_ASPECT = SHAPES / "extreme_aspect.png"

E = pygame.event.Event


def click(pos: tuple[int, int]) -> list[pygame.event.Event]:
    return [E(pygame.MOUSEBUTTONDOWN, pos=pos, button=1), E(pygame.MOUSEBUTTONUP, pos=pos, button=1)]


def key(code: int, mod: int = 0) -> pygame.event.Event:
    return E(pygame.KEYDOWN, key=code, mod=mod, unicode="")


def typed(text: str) -> list[pygame.event.Event]:
    return [E(pygame.TEXTINPUT, text=ch) for ch in text]


def drop(path: pathlib.Path | str) -> list[pygame.event.Event]:
    """What SDL delivers for one file dragged onto the window."""
    return [E(pygame.DROPBEGIN), E(pygame.DROPFILE, file=str(path)), E(pygame.DROPCOMPLETE)]


def fill(app: App, *, speed: str = "5 mm/s", size: str = "2 cm") -> None:
    """Type a speed and a size the way a person does: click, type, Enter."""
    app.handle(click(app.speed.rect.center))
    app.handle(typed(speed))
    app.handle(click(app.size.rect.center))
    app.handle(typed(size))
    app.handle([key(pygame.K_RETURN)])


@pytest.fixture
def app() -> App:
    return App()


@pytest.fixture
def surface(app: App) -> pygame.Surface:
    return pygame.Surface(app.window_size)


# ---------------------------------------------------------------------------
# Criterion 1 — a window titled FengDong, a drop target, the prepared body
# ---------------------------------------------------------------------------


def test_the_title_is_fengdong_and_the_window_has_one_drop_target(app):
    assert TITLE == "FengDong"
    targets = [w for w in app.panel.widgets if isinstance(w, DropTarget)]
    assert len(targets) == 1 and targets[0] is app.target
    assert app.target.rect.width > 0 and app.target.rect.height > 0


def test_opening_under_the_dummy_driver_sets_the_caption_and_closing_quits_the_display():
    app = App()
    try:
        screen = app.open()
        assert pygame.display.get_init()
        assert pygame.display.get_caption()[0] == TITLE
        assert screen.get_size() == app.window_size
        app.draw(screen)
    finally:
        app.close()
    assert pygame.display.get_init() is False
    app.close()  # idempotent


def test_a_dropped_png_previews_the_body_flow_prepare_made_with_its_verdict(app, surface):
    """**D-065 / D-066**: what is shown is ``Case.prepared`` — the repaired body."""
    app.handle(drop(SPECKS))
    assert app.path == str(SPECKS) and app.case is None, "no speed or size yet: nothing built"
    fill(app)
    case = app.case
    assert case is not None and case.runnable
    assert case.prepared.verdict == "repaired"
    assert case.prepared.actions and case.prepared.actions[0].startswith("drop_specks")
    assert app.verdict.text.startswith("geometry: repaired")
    assert case.prepared.actions[0] in app.verdict.text
    # the preview surface is the prepared mask, cell for cell, in two colours
    mask = case.prepared.mask
    assert app._mask_surface is not None
    assert app._mask_surface.get_size() == (mask.shape[1], mask.shape[0])
    solid = app._mask_surface.get_at((int(mask.shape[1] // 2), int(mask.shape[0] // 2)))[:3]
    fluid = app._mask_surface.get_at((0, 0))[:3]
    assert solid == app_module.SOLID_RGB and fluid == app_module.FLUID_RGB
    assert bool(mask[mask.shape[0] // 2, mask.shape[1] // 2]) and not bool(mask[0, 0])
    app.draw(surface)
    assert app.status.text == PROMPT_CURRENT


def test_the_preview_is_a_stencil_not_a_field():
    """Constraint 10: two flat colours from a bool array, nothing else."""
    source = APP_PY.read_text(encoding="utf-8")
    assert "colormap" not in source and "vorticity" not in source and "to_rgb" not in source
    assert "surfarray.make_surface" in source, "the stencil is blitted from a two-colour array"
    assert _imports_of(source, "app", "lbm") == []


def test_a_second_drop_replaces_the_first(app):
    app.handle(drop(SPECKS))
    fill(app)
    first = app.case
    app.handle(drop(DISC))
    assert app.path == str(DISC) and app.case is not first
    assert app.case is not None and app.case.prepared.verdict == "ok"
    assert app.verdict.text == "geometry: ok"


# ---------------------------------------------------------------------------
# Criterion 2 — the T206 widgets, and a bad entry does not crash the window
# ---------------------------------------------------------------------------


def test_the_inputs_are_exactly_a_fluid_a_speed_a_size_and_a_quality(app):
    """Constraint 13: nothing else in the column accepts input."""
    fields = [w for w in app.panel.widgets if isinstance(w, TextField)]
    pickers = [w for w in app.panel.widgets if isinstance(w, Dropdown)]
    buttons = [w for w in app.panel.widgets if isinstance(w, Button)]
    assert fields == [app.speed, app.size]
    assert pickers == [app.fluid, app.quality]
    assert app.fluid.options == tuple(FLUIDS)
    assert app.quality.options == QUALITY_LEVELS and app.quality.selected == "balanced"
    assert buttons == [app.preview, app.nearest_button]
    assert all(isinstance(w, (Label, TextField, Dropdown, Button, DropTarget)) for w in app.panel.widgets)
    assert isinstance(app.panel, Panel)


def test_a_bad_entry_shows_the_parsers_message_and_the_window_survives(app, surface):
    app.handle(drop(DISC))
    fill(app, speed="5 furlongs", size="2 cm")
    with pytest.raises(ValueError) as caught:
        parse("5 furlongs", expect=app.speed.expect, default_unit=app.speed.default_unit)
    assert app.speed.error == str(caught.value)
    assert app.case is None, "nothing was built from a speed that does not parse"
    assert app.status.text == "The speed does not parse — see the field."
    app.draw(surface)
    # the Preview button does not crash on it either
    app.handle(click(app.preview.rect.center))
    assert app.case is None
    app.draw(surface)
    # fix the field: the plan appears
    app.handle(click(app.speed.rect.center))
    app.handle([key(pygame.K_BACKSPACE)] * len("5 furlongs"))
    app.handle(typed("5 mm/s"))
    app.handle([key(pygame.K_RETURN)])
    assert app.case is not None and app.case.runnable
    app.draw(surface)


def test_a_file_that_is_not_a_picture_is_reported_not_raised(app, surface):
    app.handle(drop(REPO_ROOT / "pyproject.toml"))
    fill(app)
    assert app.case is None
    assert app.error and "pyproject.toml" in app.error
    assert app.status.text == app.error
    assert app.plan_text() == app.error
    app.draw(surface)
    app.handle(drop(REPO_ROOT / "does-not-exist.png"))
    assert app.case is None and app.error and "does-not-exist.png" in app.error
    app.draw(surface)


def test_the_plan_is_rebuilt_on_a_choice_a_drop_enter_or_the_button_and_not_per_keystroke(app):
    """**D-098**: a Case costs 0.4–1.1 s, so typing does not rebuild — Enter does."""
    app.handle(drop(DISC))
    fill(app)
    built = app.case
    assert built is not None
    app.handle(click(app.speed.rect.center))
    app.handle(typed("0"))  # "5 mm/s0" -- does not parse yet, and no rebuild
    assert app.case is built and app.stale and app.status.text.startswith("The speed does not parse")
    app.handle([key(pygame.K_BACKSPACE)])
    assert app.case is built and app.stale and app.status.text == PROMPT_STALE
    app.handle([key(pygame.K_RETURN)])
    assert app.case is not built and not app.stale
    rebuilt = app.case
    # a dropdown choice rebuilds
    app.panel.focus(app.quality)
    app.handle([key(pygame.K_DOWN)])
    assert app.case is not rebuilt and app.case is not None and app.case.quality == "accurate"
    # the button rebuilds
    latest = app.case
    app.handle(click(app.preview.rect.center))
    assert app.case is not latest


# ---------------------------------------------------------------------------
# Criterion 3 — the plan preview is Case.explain(), and the app computes nothing
# ---------------------------------------------------------------------------


def test_the_plan_preview_is_case_explain_verbatim(app):
    app.handle(drop(DISC))
    fill(app)
    case = app.case
    assert case is not None and case.runnable
    assert app.plan_text() == case.explain(quiet=True)
    reference = Case.from_image(DISC, fluid="water", speed="5 mm/s", size="2 cm", quality="balanced")
    assert app.plan_text() == reference.explain(quiet=True)
    for token in ("plan — every number derived, none typed", "fidelity", "(expected)", "domain", "cost"):
        assert token in app.plan_text()
    assert app.plan_pane.text == app.plan_text()


def test_the_backend_flag_reaches_the_estimate_as_the_cli_would(app):
    """**D-073**: the window contradicts none of ``python -m flow``'s flags."""
    warp_app = App(backend="warp")
    for a in (app, warp_app):
        a.handle(drop(DISC))
        fill(a)
    assert "backend 'numpy'" in app.plan_text()
    assert "backend 'warp'" in warp_app.plan_text()
    assert warp_app.case is not None and warp_app.case.backend == "warp"


def test_app_computes_no_solver_parameter():
    """Constraint 17, from the syntax: no lattice identifier, no arithmetic on a plan.

    ``app.py`` may *display* ``explain()``'s lines (D-060's exemption for output
    records) but may not name a lattice quantity — not as a variable, not as an
    attribute it reaches for, not in a string it could show. It also never
    reaches ``case.plan`` at all: the plan is read as text.
    """
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    clash = {n for n in _identifiers(tree) if n.lower() in LATTICE_NAMES}
    assert not clash, f"constraint 13/17: lattice vocabulary in fengdong/app.py: {sorted(clash)}"
    reached = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "plan" not in reached, "the app reads the plan as explain()'s text, never as fields"
    # the one thing taken from flow.autoconfig is the quality vocabulary -- words, not numbers
    from_autoconfig = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "flow.autoconfig"
        for alias in node.names
    }
    assert from_autoconfig == {"QUALITY_LEVELS"}
    assert _imports_of(APP_PY.read_text(encoding="utf-8"), "app", "lbm") == []
    assert _imports_of(APP_PY.read_text(encoding="utf-8"), "app", "flow") == [
        "app: from flow.autoconfig import ...",
        "app: from flow.case import ...",
    ]


def test_no_string_in_app_names_a_lattice_quantity():
    import re

    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    clash = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in re.findall(r"[A-Za-z_]+", node.value):
                if word.lower() in LATTICE_NAMES:
                    clash.add(word)
    assert not clash, f"constraint 13: the window could show {sorted(clash)}"


# ---------------------------------------------------------------------------
# Criterion 4 — a refusal shows its suggestions, and the one acted on is the one shown
# ---------------------------------------------------------------------------


def test_a_refused_picture_shows_the_fix_and_the_button_applies_it(app, surface):
    """**D-065**: the picture's own ``Fix``; the button is ``Case.nearest()``."""
    app.handle(drop(ALL_WHITE))
    fill(app)
    case = app.case
    assert case is not None and not case.runnable and case.fix is not None
    assert app.nearest_button.enabled
    assert app.status.text == PROMPT_REFUSED
    text = app.plan_text()
    assert text.startswith(case.explain(quiet=True))
    assert "refused — the picture" in text
    assert app.shown_suggestion() == f"{case.fix.change} -> {case.fix.value}"
    assert app.shown_suggestion() in text and case.fix.note in text
    app.draw(surface)
    expected = case.nearest()
    app.handle(click(app.nearest_button.rect.center))
    assert app.case is not case and app.case is not None
    assert app.case.substituted and app.case.substitution == expected.substitution
    assert app.case.runnable == expected.runnable
    assert "** SUBSTITUTED **" in app.plan_text()
    assert not app.nearest_button.enabled
    app.draw(surface)


def test_a_fix_that_is_not_enough_leaves_the_next_refusal_visible(app):
    """The honest outcome D-067 names: ``nearest()`` may itself be refused."""
    app.handle(drop(EXTREME_ASPECT))
    app.quality.select(QUALITY_LEVELS.index("fast"))
    fill(app)
    case = app.case
    assert case is not None and not case.runnable
    first = app.shown_suggestion()
    assert first is not None and first.startswith("resolution ->")
    app.handle(click(app.nearest_button.rect.center))
    assert app.case is not None and app.case.substituted
    assert app.case.quality == "balanced" and app.quality.selected == "balanced"
    assert app.case.runnable is False, "this picture cannot be fixed by one step"
    assert app.nearest_button.enabled
    assert app.shown_suggestion() is not None


def test_a_physics_refusal_shows_the_list_nearest_acts_on_in_its_order(monkeypatch, surface):
    """Constraint 14 over the window, in the shape of
    ``tests/test_cli.py::test_the_suggestions_the_cli_prints_are_the_ones_nearest_would_run``.

    After **D-093** the only physics refusal a library fluid can reach is an
    inviscid one, so a sixth fluid with no viscosity is added to the library
    for the test. The pane's list is ``case.suggestions`` in order, the
    button applies its first entry, and the substituted case says which.
    """
    water = FLUIDS["water"]
    vacuum = Fluid(
        name="vacuum",
        nu=Quantity("0 m^2/s", expect=VISCOSITY),
        rho=water.rho,
        T=water.T,
        source="a test fluid with no viscosity",
    )
    monkeypatch.setitem(FLUIDS, "vacuum", vacuum)
    app = App()
    assert "vacuum" in app.fluid.options
    app.fluid.select(app.fluid.options.index("vacuum"))
    app.handle(drop(DISC))
    fill(app)
    case = app.case
    assert case is not None and case.prepared.runnable and case.refusal is not None
    offers = case.suggestions
    assert offers, "constraint 14: a refusal names a fix"
    text = app.plan_text()
    assert text.startswith(case.explain(quiet=True))
    lines = text.splitlines()
    header = lines.index("What the button above would run, in the order it would try them")
    shown = [line.strip() for line in lines[header + 1:] if line.strip()[:2] in {f"{i}." for i in range(1, 10)}]
    assert shown == [f"{i}. {o.change} -> {o.value}" for i, o in enumerate(offers, 1)]
    assert all(o.note in text for o in offers)
    assert app.shown_suggestion() == f"{offers[0].change} -> {offers[0].value}"
    app.draw(surface)
    app.handle(click(app.nearest_button.rect.center))
    assert app.case is not None and app.case.substituted
    assert app.case.substitution.startswith(f"{offers[0].change} -> {offers[0].value}")
    # the fields now say what the plan is for
    assert app.fluid.selected == app.case.fluid.name
    assert app.speed.value == app.case.speed and app.size.value == app.case.size
    app.draw(surface)


# ---------------------------------------------------------------------------
# Criterion 5 — resizable, and nothing positioned by a pixel count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(640, 480), (1100, 720), (2560, 1440), (300, 200)])
def test_the_layout_is_derived_from_the_window_size(app, size):
    app.handle(drop(DISC))
    fill(app)
    app.handle([E(pygame.VIDEORESIZE, w=size[0], h=size[1], size=size)])
    w, h = app.window_size
    assert (w, h) == (
        max(size[0], MIN_SIZE[0]),
        max(size[1], MIN_SIZE[1], app.panel.content_height),
    )
    window = pygame.Rect(0, 0, w, h)
    assert window.contains(app.panel.rect)
    assert window.contains(app.preview_rect) and window.contains(app.plan_rect)
    assert not app.panel.rect.colliderect(app.preview_rect)
    assert not app.panel.rect.colliderect(app.plan_rect)
    assert not app.preview_rect.colliderect(app.plan_rect)
    assert app.panel.content_height <= app.panel.rect.height
    for widget in app.panel.widgets:
        assert app.panel.rect.contains(widget.rect)
    assert app.preview_rect.width > 0 and app.plan_rect.height > 0
    surface = pygame.Surface((w, h))
    app.draw(surface)
    scaled = app._scaled_mask()
    assert scaled.get_width() <= app.preview_rect.width and scaled.get_height() <= app.preview_rect.height


def test_the_column_scales_with_the_window_between_its_bounds():
    small, large = App(size=(640, 480)), App(size=(2560, 1440))
    assert small.panel.rect.width == app_module.COLUMN_MIN
    assert large.panel.rect.width == app_module.COLUMN_MAX
    mid = App(size=(1000, 700))
    assert app_module.COLUMN_MIN < mid.panel.rect.width < app_module.COLUMN_MAX
    assert mid.panel.rect.width == int(1000 * app_module.COLUMN_FRACTION)


def test_no_pixel_position_is_hard_coded_in_layout():
    """Every rect in ``layout`` comes from the size, a fraction, or a named constant."""
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    layout = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "layout")
    literals = {
        node.value for node in ast.walk(layout)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    }
    assert literals <= {0, 1, 2, 3}, f"a pixel count is hard-coded in layout(): {sorted(literals)}"


def test_the_wheel_scrolls_the_plan_pane_and_a_resize_keeps_it_readable(app, surface):
    app.handle(drop(DISC))
    fill(app)
    app.handle([E(pygame.VIDEORESIZE, w=700, h=480, size=(700, 480))])
    small = pygame.Surface((700, 480))
    app.draw(small)
    assert app.plan_pane.offset == 0
    app.handle([E(pygame.MOUSEWHEEL, x=0, y=-2)])
    assert app.plan_pane.offset == 2 * app_module.SCROLL_LINES
    app.handle([E(pygame.MOUSEWHEEL, x=0, y=50)])
    assert app.plan_pane.offset == 0
    app.handle([E(pygame.MOUSEWHEEL, x=0, y=-10_000)])
    app.draw(small)
    assert app.plan_pane.offset == max(len(app.plan_pane._wrapped) - app.plan_rect.height // app_module.LINE_HEIGHT, 0)


# ---------------------------------------------------------------------------
# Criterion 6 — closing exits cleanly, asserted headless
# ---------------------------------------------------------------------------


def test_run_returns_zero_on_quit_leaves_no_display_and_warns_about_no_resource(monkeypatch):
    """The close button, delivered as ``QUIT`` once the window is up — posting it
    before ``open()`` is not the same, because initialising the display
    starts a fresh event queue."""
    app = App()
    real_open = App.open

    def open_then_close(self: App) -> pygame.Surface:
        screen = real_open(self)
        pygame.event.post(E(pygame.QUIT))
        return screen

    monkeypatch.setattr(App, "open", open_then_close)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        try:
            code = app.run()
        finally:
            app.close()
    assert code == 0
    assert app.quit_requested
    assert pygame.display.get_init() is False
    assert not pygame.display.get_surface()
    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


def test_the_whole_command_exits_cleanly_in_a_subprocess_with_resource_warnings_as_errors():
    """``fengdong`` end to end: open, one drop, close, ``-W error::ResourceWarning``."""
    script = (
        "import os, sys; os.environ['SDL_VIDEODRIVER'] = 'dummy'; "
        "import pygame, fengdong.__main__ as m, fengdong.app as a; "
        "orig = a.App.open\n"
        "def opened(self):\n"
        "    s = orig(self)\n"
        f"    pygame.event.post(pygame.event.Event(pygame.DROPFILE, file={str(DISC)!r}))\n"
        "    pygame.event.post(pygame.event.Event(pygame.QUIT))\n"
        "    return s\n"
        "a.App.open = opened\n"
        "code = m.main([])\n"
        "assert code == 0, code\n"
        "assert not pygame.display.get_init()\n"
        "print('closed cleanly')"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "error::ResourceWarning", "-c", script],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("closed cleanly")


# ---------------------------------------------------------------------------
# Criterion 7 — the state machine, headless, with no window
# ---------------------------------------------------------------------------


def test_the_whole_state_machine_runs_with_the_display_never_initialised(surface):
    app = App()
    assert pygame.display.get_init() is False
    assert app.status.text == PROMPT_DROP and app.plan_text() == PROMPT_DROP
    app.draw(surface)
    assert app.handle(drop(DISC)) is False, "a drop with no numbers builds nothing yet"
    assert app.status.text == "Enter a speed."
    app.handle(click(app.speed.rect.center))
    app.handle(typed("5 mm/s"))
    assert app.status.text == "Enter a size."
    app.handle(click(app.size.rect.center))
    app.handle(typed("2 cm"))
    assert app.case is None and app.status.text == PROMPT_STALE
    assert app.handle([key(pygame.K_RETURN)]) is True
    assert app.case is not None and app.case.runnable and app.status.text == PROMPT_CURRENT
    assert app.handle([E(pygame.MOUSEMOTION, pos=(5, 5), rel=(0, 0), buttons=(0, 0, 0))]) is False
    app.draw(surface)
    assert pygame.display.get_init() is False
    assert not pygame.display.get_surface()


def test_drawing_an_unchanged_window_renders_no_new_text(app, surface, monkeypatch):
    """**D-097** (7) read across to the window: no allocation on an unchanged frame."""
    from fengdong import widgets

    app.handle(drop(DISC))
    fill(app)
    app.draw(surface)
    calls: list[object] = []
    real_font = widgets._font()

    class CountingFont:
        def render(self, *a, **k):
            calls.append(a)
            return real_font.render(*a, **k)

        def size(self, text):
            return real_font.size(text)

    monkeypatch.setattr(widgets, "_font", lambda: CountingFont())
    app.draw(surface)
    app.draw(surface)
    assert calls == [], "an unchanged frame rendered new text"


def test_the_widget_set_is_used_and_not_extended():
    """No Widget subclass outside ``fengdong/widgets.py`` (**D-083**: the set is closed)."""
    from fengdong.widgets import Widget

    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
            assert "Widget" not in bases and not (bases & {c.__name__ for c in Widget.__subclasses__()}), node.name


# ---------------------------------------------------------------------------
# The entry point: --version first, without the window; no argument opens it
# ---------------------------------------------------------------------------


def test_version_still_answers_without_importing_the_window():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, fengdong.__main__ as m; m.main(['--version']); "
         "assert 'fengdong.app' not in sys.modules and 'pygame' not in sys.modules "
         "and 'numpy' not in sys.modules and 'flow' not in sys.modules, "
         "sorted(k for k in sys.modules if k.split('.')[0] in ('fengdong','pygame','numpy','flow'))"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == entry.version_line()


def test_no_argument_opens_the_window_and_returns_its_code(monkeypatch):
    seen: list[str] = []

    def fake_run(self):
        seen.append(self.backend)
        return 7

    monkeypatch.setattr(App, "run", fake_run)
    assert entry.main([]) == 7
    assert entry.main(["--backend", "warp"]) == 7
    assert seen == ["numpy", "warp"]


def test_main_module_imports_no_window_at_module_scope():
    source = pathlib.Path(entry.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = {
        alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    assert not any(name.startswith(("fengdong.app", "pygame", "flow", "lbm", "numpy")) for name in top_level), top_level


# ---------------------------------------------------------------------------
# Last: still no display, after everything above
# ---------------------------------------------------------------------------


def test_no_display_is_left_initialised_by_any_test_in_this_file():
    assert pygame.display.get_init() is False
