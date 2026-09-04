"""T206 — ``fengdong/widgets.py``, the closed widget set, driven headless.

``DOCS/TASKS3.md`` § T206, one test per acceptance criterion plus the
invariants:

* the set is **closed** at five widgets plus ``Panel`` (**D-083**);
* every widget is built, driven with synthesised ``pygame.event`` objects
  and drawn onto an off-screen ``Surface`` with **no display initialised**;
* ``TextField`` validates through :func:`flow.quantity.parse` and shows the
  parser's own message — the one ``python -m flow`` prints — verbatim;
* ``Dropdown.fluids`` is populated from :data:`flow.fluids.FLUIDS` at
  construction, so a fluid added to the library appears with no edit;
* ``DropTarget`` consumes a synthesised :data:`pygame.DROPFILE`;
* **constraint 13** — the module is scanned against the lattice vocabulary
  ``tests/test_flow_package.py`` already forbids in ``flow/``;
* **constraint 17** — ``fengdong/`` imports ``flow/``, never ``lbm/``, and
  ``flow/`` never imports ``fengdong/`` (the scan in the constraint-15 shape);
* **constraint 10** — no ``render`` / ``to_rgb`` / ``colormap`` in ``fengdong/``.

``SDL_VIDEODRIVER=dummy`` is set before pygame is imported, as
``tests/test_render.py`` does, and the last test in the file asserts
``pygame.display.get_init()`` is still ``False`` after everything above ran.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
import pytest  # noqa: E402

import fengdong  # noqa: E402
import flow  # noqa: E402
import flow.cli  # noqa: E402
from fengdong import widgets  # noqa: E402
from fengdong.widgets import (  # noqa: E402
    ROW_HEIGHT,
    Button,
    Dropdown,
    DropTarget,
    Label,
    Panel,
    TextField,
    Widget,
)
from flow.fluids import FLUIDS, Fluid  # noqa: E402
from flow.quantity import LENGTH, SPEED, parse  # noqa: E402
from test_flow_package import LATTICE_NAMES  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WIDGETS_PY = pathlib.Path(widgets.__file__)
FENGDONG_ROOT = pathlib.Path(fengdong.__file__).parent
DISC = REPO_ROOT / "tests" / "data" / "shapes" / "disc.png"

E = pygame.event.Event


def click(pos: tuple[int, int]) -> list[pygame.event.Event]:
    """A left button down and up at ``pos``."""
    return [
        E(pygame.MOUSEBUTTONDOWN, pos=pos, button=1),
        E(pygame.MOUSEBUTTONUP, pos=pos, button=1),
    ]


def key(code: int, mod: int = 0) -> pygame.event.Event:
    return E(pygame.KEYDOWN, key=code, mod=mod, unicode="")


def typed(text: str) -> list[pygame.event.Event]:
    """What SDL2 delivers for ``text`` typed on a keyboard: one TEXTINPUT per character."""
    return [E(pygame.TEXTINPUT, text=ch) for ch in text]


@pytest.fixture
def surface() -> pygame.Surface:
    return pygame.Surface((400, 700))


# ---------------------------------------------------------------------------
# Criterion 1 — the set is closed
# ---------------------------------------------------------------------------


def test_the_widget_set_is_exactly_five_plus_panel():
    """**D-083**: ``Label``, ``TextField``, ``Dropdown``, ``Button``, ``DropTarget``, ``Panel``."""
    concrete = {
        name
        for name, obj in vars(widgets).items()
        if inspect.isclass(obj)
        and issubclass(obj, Widget)
        and obj is not Widget
        and obj.__module__ == widgets.__name__
    }
    assert concrete == {"Label", "TextField", "Dropdown", "Button", "DropTarget", "Panel"}
    assert set(widgets.__all__) >= concrete


def _identifiers(tree: ast.AST) -> set[str]:
    """Every name the module defines, binds, calls or reaches for — not its prose."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def test_no_layout_engine_theming_or_animation():
    """The three things the contract forbids, checked in the identifiers rather than argued.

    No clock, ticks or timers (animation); no theme, palette or style object
    (theming); no grid, row-span, anchor or weight vocabulary beyond the
    single column ``Panel`` stacks (layout engine). Identifiers, not prose:
    a docstring may say "no theming"; a name may not be one.
    """
    names = {n.lower() for n in _identifiers(ast.parse(WIDGETS_PY.read_text(encoding="utf-8")))}
    banned = {
        "get_ticks", "time", "clock", "set_timer", "userevent", "tick", "animate", "tween",
        "theme", "palette", "style", "skin",
        "grid", "anchor", "weight", "stretch", "flex", "span", "layout_engine",
    }
    assert not (names & banned), sorted(names & banned)


def test_the_only_focus_chain_is_tab_order():
    """No focus stack, no focus groups, no default/cancel buttons — Tab and Shift+Tab."""
    source = WIDGETS_PY.read_text(encoding="utf-8")
    assert "K_TAB" in source
    assert "KMOD_SHIFT" in source
    for banned in ("focus_next", "focus_stack", "FocusGroup", "default_button", "cancel_button"):
        assert banned not in source


# ---------------------------------------------------------------------------
# Criterion 2 — headless
# ---------------------------------------------------------------------------


def all_widgets() -> tuple[Panel, dict[str, Widget]]:
    speed = TextField.speed((0, 0, 300, TextField.DEFAULT_HEIGHT), placeholder="e.g. 5 mm/s")
    size = TextField.size((0, 0, 300, TextField.DEFAULT_HEIGHT), placeholder="e.g. 2 cm")
    fluids = Dropdown.fluids((0, 0, 300, ROW_HEIGHT))
    run = Button((0, 0, 300, ROW_HEIGHT), "Run")
    target = DropTarget((0, 0, 300, 120))
    parts = {
        "speed_label": Label((0, 0, 0, ROW_HEIGHT), "Speed"),
        "speed": speed,
        "size_label": Label((0, 0, 0, ROW_HEIGHT), "Size"),
        "size": size,
        "fluids": fluids,
        "run": run,
        "target": target,
    }
    panel = Panel((10, 10, 380, 680), list(parts.values()))
    return panel, parts


def test_every_widget_builds_draws_and_handles_with_no_window(surface):
    panel, parts = all_widgets()
    panel.draw(surface)
    events = [
        *click(parts["speed"].rect.center),
        *typed("5 mm/s"),
        key(pygame.K_TAB),
        *typed("2 cm"),
        *click(parts["fluids"].rect.center),
        *click(parts["run"].rect.center),
        E(pygame.DROPFILE, file=str(DISC)),
    ]
    panel.handle(events)
    panel.draw(surface)
    assert pygame.display.get_init() is False
    assert not pygame.display.get_surface()


def test_pytest_runs_under_the_dummy_video_driver():
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"


def test_the_module_never_touches_the_display_or_pygame_init():
    """Read from the AST: no ``pygame.init``, no ``pygame.display``, no ``set_mode``."""
    tree = ast.parse(WIDGETS_PY.read_text(encoding="utf-8"))
    reached = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "pygame":
                reached.add(f"pygame.{node.attr}")
            reached.add(node.attr)
    assert "pygame.init" not in reached
    assert "pygame.display" not in reached
    assert "set_mode" not in reached
    assert "pygame.font" in reached, "fonts are the one subsystem the widgets need"


# ---------------------------------------------------------------------------
# Criterion 3 — TextField validates through flow.quantity.parse, in the user's words
# ---------------------------------------------------------------------------


def test_a_valid_speed_parses_to_the_same_quantity_case_would_get(surface):
    field = TextField.speed((0, 0, 300, TextField.DEFAULT_HEIGHT))
    field.focused = True
    assert field.handle(typed("72 km/h")) is True
    assert field.error is None
    assert field.value == parse("72 km/h", expect=SPEED, default_unit="m/s")
    assert field.value is not None and abs(field.value.si - 20.0) < 1e-9
    field.draw(surface)


def test_a_bare_number_means_the_unit_case_declares():
    """``flow/case.py``: speed defaults to m/s, size to m. The field says the same."""
    speed = TextField.speed((0, 0, 300, 80), text="20")
    size = TextField.size((0, 0, 300, 80), text="1.5")
    assert speed.value is not None and speed.value.si == 20.0 and speed.value.dimension == SPEED
    assert size.value is not None and size.value.si == 1.5 and size.value.dimension == LENGTH


@pytest.mark.parametrize(
    "bad",
    ["5 furlongs", "fast", "20 kg", "2 cm", ""],
    ids=["unknown-unit", "not-a-number", "wrong-dimension", "a-length-not-a-speed", "empty"],
)
def test_the_error_is_the_parsers_own_message_verbatim(bad):
    field = TextField.speed((0, 0, 300, 80), text=bad)
    if not bad:
        assert field.error is None and field.value is None
        return
    try:
        parse(bad, expect=SPEED, default_unit="m/s")
    except ValueError as exc:
        expected = str(exc)
    else:  # pragma: no cover
        pytest.fail(f"{bad!r} should not parse as a speed")
    assert field.value is None
    assert field.error == expected


def test_the_error_is_what_the_cli_prints_from_the_same_code_path(capsys):
    """Constraint 14's posture: one message, two surfaces, no re-wording.

    ``python -m flow --speed "5 furlongs"`` refuses with the parser's message
    on stderr; the widget's ``error`` is that same string. Asserted by running
    the CLI's ``main`` in-process against the committed disc.
    """
    field = TextField.speed((0, 0, 300, 80), text="5 furlongs")
    assert field.error
    code = flow.cli.main(
        ["--shape", str(DISC), "--fluid", "water", "--speed", "5 furlongs",
         "--size", "2 cm", "--explain"]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert field.error in err
    assert err.strip() == field.error


def test_typing_and_backspace_revalidate_as_they_go():
    field = TextField.speed((0, 0, 300, 80))
    field.focused = True
    field.handle(typed("5 furlongs"))
    assert field.error and "furlongs" in field.error
    assert field.handle([key(pygame.K_BACKSPACE)] * 8) is True
    assert field.text == "5 "
    assert field.error is None and field.value is not None and field.value.si == 5.0, (
        "a bare number means the default unit, m/s, exactly as Case reads it"
    )
    field.handle(typed("mm/s"))
    assert field.error is None and field.value is not None and abs(field.value.si - 0.005) < 1e-12


def test_backspace_on_an_empty_field_changes_nothing():
    field = TextField.speed((0, 0, 300, 80))
    field.focused = True
    assert field.handle([key(pygame.K_BACKSPACE)]) is False


def test_enter_commits_a_valid_value_and_not_an_invalid_one():
    seen = []
    field = TextField.size((0, 0, 300, 80), on_commit=seen.append)
    field.focused = True
    field.handle(typed("2 cm"))
    field.handle([key(pygame.K_RETURN)])
    assert len(seen) == 1 and abs(seen[0].si - 0.02) < 1e-12
    field.set_text("2 lightyears")
    field.handle([key(pygame.K_RETURN)])
    assert len(seen) == 1


def test_a_field_ignores_keys_unless_focused_and_a_click_focuses_it():
    field = TextField.speed((10, 10, 300, 80))
    assert field.handle(typed("5")) is False and field.text == ""
    field.handle([E(pygame.MOUSEBUTTONDOWN, pos=(20, 20), button=1)])
    assert field.focused
    field.handle(typed("5"))
    assert field.text == "5"
    field.handle([E(pygame.MOUSEBUTTONDOWN, pos=(500, 500), button=1)])
    assert not field.focused
    field.handle([E(pygame.MOUSEBUTTONDOWN, pos=(20, 20), button=1), key(pygame.K_ESCAPE)])
    assert not field.focused


def test_the_error_is_drawn_and_the_placeholder_shows_when_empty(surface):
    """Drawing an error must not crash on a message wider than the field, and
    the error rows must be painted (some pixel in the error area is not the background)."""
    field = TextField.speed((0, 0, 200, TextField.DEFAULT_HEIGHT), placeholder="e.g. 5 mm/s")
    surface.fill(widgets.PAPER)
    field.draw(surface)
    field.set_text("5 furlongs")
    field.draw(surface)
    error_area = pygame.Rect(0, ROW_HEIGHT, 200, TextField.DEFAULT_HEIGHT - ROW_HEIGHT)
    sub = surface.subsurface(error_area)
    assert any(
        sub.get_at((x, y))[:3] != widgets.PAPER
        for y in range(0, error_area.height, 2)
        for x in range(0, error_area.width, 2)
    )


# ---------------------------------------------------------------------------
# Criterion 4 — Dropdown for fluids is populated from flow.fluids.FLUIDS
# ---------------------------------------------------------------------------


def test_the_fluid_dropdown_lists_the_library_in_its_own_order():
    picker = Dropdown.fluids((0, 0, 300, ROW_HEIGHT))
    assert picker.options == tuple(FLUIDS)
    assert picker.selected == next(iter(FLUIDS))


def test_adding_a_fluid_to_the_library_adds_it_to_the_widget_with_no_edit_here(monkeypatch):
    water = FLUIDS["water"]
    mercury = Fluid(name="mercury", nu=water.nu, rho=water.rho, T=water.T, source="a test")
    monkeypatch.setitem(FLUIDS, "mercury", mercury)
    assert "mercury" not in Dropdown.fluids.__code__.co_consts  # not hard-coded
    picker = Dropdown.fluids((0, 0, 300, ROW_HEIGHT))
    assert "mercury" in picker.options
    assert picker.options == tuple(FLUIDS)


def test_the_fluid_names_are_the_ones_flow_fluids_accepts():
    """Every option round-trips through ``flow.fluids.fluid`` — the app hands the string to ``Case``."""
    for name in Dropdown.fluids((0, 0, 300, ROW_HEIGHT)).options:
        assert flow.fluid(name).name == name


def test_click_opens_click_on_a_row_selects(surface):
    picker = Dropdown.fluids((10, 10, 300, ROW_HEIGHT))
    assert picker.handle([E(pygame.MOUSEBUTTONDOWN, pos=picker.rect.center, button=1)]) is False
    assert picker.open and picker.focused
    picker.draw(surface)
    assert picker.handle([E(pygame.MOUSEBUTTONDOWN, pos=picker.row_rect(2).center, button=1)]) is True
    assert picker.selected == tuple(FLUIDS)[2]
    assert not picker.open
    # clicking outside an open list closes it and selects nothing
    picker.handle([E(pygame.MOUSEBUTTONDOWN, pos=picker.rect.center, button=1)])
    assert picker.open
    assert picker.handle([E(pygame.MOUSEBUTTONDOWN, pos=(399, 699), button=1)]) is False
    assert not picker.open and picker.selected == tuple(FLUIDS)[2]


def test_arrow_keys_move_the_selection_and_call_on_change():
    seen = []
    picker = Dropdown((0, 0, 300, ROW_HEIGHT), ("fast", "balanced", "accurate"), on_change=seen.append)
    picker.focused = True
    assert picker.handle([key(pygame.K_DOWN)]) is True and picker.selected == "balanced"
    assert picker.handle([key(pygame.K_DOWN), key(pygame.K_DOWN)]) is True and picker.selected == "accurate"
    assert picker.handle([key(pygame.K_UP)]) is True and picker.selected == "balanced"
    assert seen == ["balanced", "accurate", "balanced"]
    picker.select(picker.index)
    assert seen == ["balanced", "accurate", "balanced"], "re-selecting the same row is not a change"


def test_a_dropdown_refuses_an_empty_list_or_an_unknown_initial_choice():
    with pytest.raises(ValueError):
        Dropdown((0, 0, 100, ROW_HEIGHT), ())
    with pytest.raises(ValueError):
        Dropdown((0, 0, 100, ROW_HEIGHT), ("a", "b"), selected="c")
    with pytest.raises(ValueError):
        Dropdown((0, 0, 100, ROW_HEIGHT), ("a", "b"), selected=7)
    assert Dropdown((0, 0, 100, ROW_HEIGHT), ("a", "b"), selected="b").selected == "b"


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------


def test_a_click_is_down_and_up_inside_the_rect():
    calls = []
    button = Button((10, 10, 100, ROW_HEIGHT), "Run", on_click=lambda: calls.append(1))
    assert button.handle(click(button.rect.center)) is True
    assert calls == [1] and button.take_click() is True and button.take_click() is False
    # down inside, up outside: not a click
    assert button.handle([
        E(pygame.MOUSEBUTTONDOWN, pos=button.rect.center, button=1),
        E(pygame.MOUSEBUTTONUP, pos=(300, 300), button=1),
    ]) is False
    # right button: nothing
    assert button.handle([
        E(pygame.MOUSEBUTTONDOWN, pos=button.rect.center, button=3),
        E(pygame.MOUSEBUTTONUP, pos=button.rect.center, button=3),
    ]) is False
    assert calls == [1]


def test_enter_and_space_click_a_focused_button_and_a_disabled_one_ignores_everything(surface):
    button = Button((0, 0, 100, ROW_HEIGHT), "Run")
    button.focused = True
    assert button.handle([key(pygame.K_RETURN), key(pygame.K_SPACE)]) is True
    assert button.clicks == 2
    button.enabled = False
    assert button.handle([key(pygame.K_RETURN), *click(button.rect.center)]) is False
    assert button.clicks == 2
    button.draw(surface)


# ---------------------------------------------------------------------------
# Criterion 5 — DropTarget consumes DROPFILE and reports the path
# ---------------------------------------------------------------------------


def test_a_synthesised_dropfile_reports_the_path(surface):
    seen = []
    target = DropTarget((0, 0, 300, 120), on_drop=seen.append)
    assert target.path is None
    events = [E(pygame.DROPBEGIN), E(pygame.DROPFILE, file=str(DISC)), E(pygame.DROPCOMPLETE)]
    assert target.handle(events[:1]) is False and target.hovering
    target.draw(surface)
    assert target.handle(events[1:2]) is True
    assert target.path == str(DISC) and target.drops == 1 and seen == [str(DISC)]
    assert target.handle(events[2:]) is False and not target.hovering
    target.draw(surface)


def test_a_second_drop_replaces_the_first_and_clicks_do_nothing():
    target = DropTarget((0, 0, 300, 120))
    target.handle([E(pygame.DROPFILE, file="a.png")])
    target.handle([E(pygame.DROPFILE, file="b.png")])
    assert target.path == "b.png" and target.drops == 2
    assert target.handle(click(target.rect.center)) is False
    assert not target.focusable


def test_the_drop_target_judges_nothing_about_the_file():
    """What the file *is* is ``flow.prepare``'s verdict (D-065/D-066), not the widget's."""
    target = DropTarget((0, 0, 300, 120))
    assert target.handle([E(pygame.DROPFILE, file="not-a-picture.txt")]) is True
    assert target.path == "not-a-picture.txt"
    source = inspect.getsource(DropTarget)
    assert ".png" not in source and "suffix" not in source and "PIL" not in source


# ---------------------------------------------------------------------------
# Panel — one column, tab order, an open list captures
# ---------------------------------------------------------------------------


def test_the_panel_stacks_its_children_in_one_column_without_overlap():
    panel, parts = all_widgets()
    rects = [w.rect for w in panel.widgets]
    for above, below in zip(rects, rects[1:]):
        assert above.bottom <= below.top
        assert above.x == below.x == panel.rect.x + panel.padding
        assert above.width == below.width == panel.rect.width - 2 * panel.padding
    assert all(panel.rect.contains(r) for r in rects), "the column fits the panel"
    assert panel.content_height <= panel.rect.height


def test_resize_restacks_and_keeps_heights():
    panel, parts = all_widgets()
    heights = [w.rect.height for w in panel.widgets]
    panel.resize((0, 0, 200, 900))
    assert [w.rect.height for w in panel.widgets] == heights
    assert all(w.rect.width == 200 - 2 * panel.padding for w in panel.widgets)


def test_tab_and_shift_tab_walk_the_focusable_children_in_order():
    panel, parts = all_widgets()
    order = panel.focusable_widgets
    assert order == (parts["speed"], parts["size"], parts["fluids"], parts["run"])
    assert panel.focused is None
    panel.handle([key(pygame.K_TAB)])
    assert panel.focused is parts["speed"]
    panel.handle([key(pygame.K_TAB), key(pygame.K_TAB)])
    assert panel.focused is parts["fluids"]
    panel.handle([key(pygame.K_TAB, mod=pygame.KMOD_SHIFT)])
    assert panel.focused is parts["size"]
    panel.handle([key(pygame.K_TAB)] * 3)
    assert panel.focused is parts["speed"], "tab wraps"
    assert sum(w.focused for w in panel.widgets) == 1


def test_keys_go_to_the_focused_child_only():
    panel, parts = all_widgets()
    panel.focus(parts["size"])
    panel.handle(typed("2 cm"))
    assert parts["size"].text == "2 cm" and parts["speed"].text == ""


def test_an_open_dropdown_swallows_the_click_meant_for_the_button_under_it(surface):
    panel, parts = all_widgets()
    fluids, run = parts["fluids"], parts["run"]
    panel.handle([E(pygame.MOUSEBUTTONDOWN, pos=fluids.rect.center, button=1)])
    assert fluids.open
    assert fluids.row_rect(0).colliderect(run.rect), "the list really does cover the button"
    panel.draw(surface)
    before = fluids.selected
    hit = next(i for i in range(len(fluids.options)) if fluids.row_rect(i).collidepoint(run.rect.center))
    changed = panel.handle([*click(run.rect.center)])
    assert changed is (hit != fluids.options.index(before)), "a change iff a different row was hit"
    assert fluids.selected == fluids.options[hit] and not fluids.open
    assert run.clicks == 0 and not run.pressed, "the button under the list did not fire"


def test_the_panel_reports_a_change_when_any_child_does():
    panel, parts = all_widgets()
    assert panel.handle([E(pygame.MOUSEMOTION, pos=(5, 5), rel=(0, 0), buttons=(0, 0, 0))]) is False
    assert panel.handle([E(pygame.DROPFILE, file="x.png")]) is True


def test_drawing_an_unchanged_panel_renders_no_new_text(surface, monkeypatch):
    """No allocation on the per-frame path (``CLAUDE.md`` § Coding conventions)."""
    panel, parts = all_widgets()
    parts["speed"].set_text("5 furlongs")
    parts["fluids"].open = True
    panel.draw(surface)
    calls = []
    real_font = widgets._font()

    class CountingFont:
        """``pygame.font.Font.render`` is read-only, so the module's ``_font`` is wrapped instead."""

        def render(self, *a, **k):
            calls.append(a)
            return real_font.render(*a, **k)

        def size(self, text):
            return real_font.size(text)

    monkeypatch.setattr(widgets, "_font", lambda: CountingFont())
    panel.draw(surface)
    panel.draw(surface)
    assert calls == [], "an unchanged frame rendered new text"
    parts["speed"].set_text("5 mm/s")
    panel.draw(surface)
    assert len(calls) == 1, "a changed field re-renders exactly its own line"


# ---------------------------------------------------------------------------
# Criterion 6 — constraint 13: no lattice quantity in a widget
# ---------------------------------------------------------------------------


def _fengdong_modules() -> list[pathlib.Path]:
    """Every module in ``fengdong/`` — the scans below run over all of them, so a
    file added to the package (T207's ``app.py``) is covered the moment it exists."""
    return sorted(p for p in FENGDONG_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _fengdong_modules(), ids=lambda p: p.name)
def test_no_identifier_in_fengdong_is_a_lattice_quantity(path):
    """The vocabulary is ``tests/test_flow_package.py::LATTICE_NAMES`` — imported, not copied."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    clash = {n for n in _identifiers(tree) if n.lower() in LATTICE_NAMES}
    assert not clash, f"constraint 13: lattice vocabulary in fengdong/{path.name}: {sorted(clash)}"


def _docstrings(tree: ast.AST) -> set[int]:
    """The ids of the docstring constants — prose that is never shown to a user."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found


@pytest.mark.parametrize("path", _fengdong_modules(), ids=lambda p: p.name)
def test_no_string_in_fengdong_names_a_lattice_quantity(path):
    """Nor may a widget *display* one: every string literal, word by word.

    Docstrings are exempt — they are the one kind of string a window never
    shows, and ``fengdong/__init__.py``'s names the forbidden vocabulary in
    order to forbid it. Every other literal (a caption, a prompt, a status
    line, an argparse ``help``) is scanned.
    """
    import re

    tree = ast.parse(path.read_text(encoding="utf-8"))
    prose = _docstrings(tree)
    clash = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            for word in re.findall(r"[A-Za-z_]+", node.value):
                if word.lower() in LATTICE_NAMES:
                    clash.add(word)
    assert not clash, f"constraint 13: fengdong/{path.name} could show {sorted(clash)}"


def _fengdong_imported_modules() -> list[object]:
    import importlib

    return [importlib.import_module(f"fengdong.{p.stem}") for p in _fengdong_modules()
            if p.stem != "__init__"] + [fengdong]


@pytest.mark.parametrize("module", _fengdong_imported_modules(), ids=lambda m: m.__name__)
def test_no_public_signature_in_fengdong_takes_a_lattice_quantity(module):
    """The same scan ``tests/test_flow_package.py`` runs over ``flow/``, over every module here."""
    offenders = []
    for name, obj in vars(module).items():
        if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        candidates = [(name, obj)]
        if inspect.isclass(obj):
            candidates += [
                (f"{name}.{m}", v) for m, v in vars(obj).items()
                if not m.startswith("_") and callable(v)
            ]
        for label, fn in candidates:
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):  # pragma: no cover
                continue
            offenders += [f"{label}({p})" for p in params if p.lower() in LATTICE_NAMES]
    assert not offenders, offenders


def test_the_constraint_13_scan_over_widgets_has_teeth():
    tree = ast.parse("def f(speed, tau=0.6):\n    return tau\nlabel = 'the tau is'\n")
    assert {n for n in _identifiers(tree) if n.lower() in LATTICE_NAMES} == {"tau"}
    tree = ast.parse('"""a docstring about tau"""\nlabel = "the tau is"\n')
    prose = _docstrings(tree)
    shown = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and id(n) not in prose]
    assert [n.value for n in shown] == ["the tau is"], "docstrings exempt, literals scanned"


def test_the_fields_the_app_has_are_a_speed_and_a_size_and_nothing_lattice():
    """The two constructors: ``expect`` is a *physical* dimension, in ``flow.quantity``'s words."""
    speed = TextField.speed((0, 0, 100, 80))
    size = TextField.size((0, 0, 100, 80))
    assert (speed.expect, speed.default_unit) == (SPEED, "m/s")
    assert (size.expect, size.default_unit) == (LENGTH, "m")
    # and they are the exact arguments Case passes (flow/case.py):
    case_source = inspect.getsource(flow.case.Case.__init__)
    assert 'parse(speed, expect=SPEED, default_unit="m/s")' in case_source
    assert 'parse(size, expect=LENGTH, default_unit="m")' in case_source


# ---------------------------------------------------------------------------
# Criterion 7 — constraint 17, in the shape of the constraint-15 scan
# ---------------------------------------------------------------------------


def _imports_of(source: str, label: str, package: str) -> list[str]:
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


def _modules(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_under_flow_or_lbm_imports_fengdong():
    for package in (flow, __import__("lbm")):
        for path in _modules(pathlib.Path(package.__file__).parent):
            assert not _imports_of(path.read_text(encoding="utf-8"), path.name, "fengdong")


def test_widgets_imports_flow_and_never_lbm():
    """The legal direction is used (the field validates through ``flow``), and
    the solver is never reached directly — not even ``lbm.render`` (constraint 10).
    The second half walks every file in ``fengdong/``, so ``app.py`` is covered too."""
    source = WIDGETS_PY.read_text(encoding="utf-8")
    assert _imports_of(source, "widgets", "flow") == [
        "widgets: from flow.fluids import ...",
        "widgets: from flow.quantity import ...",
    ]
    assert _imports_of(source, "widgets", "lbm") == []
    for path in _modules(FENGDONG_ROOT):
        assert _imports_of(path.read_text(encoding="utf-8"), path.name, "lbm") == []


def test_the_constraint_17_scan_here_has_teeth():
    assert _imports_of("import fengdong.widgets", "x", "fengdong") == ["x: import fengdong.widgets"]
    assert _imports_of("def f():\n    from fengdong import widgets\n", "x", "fengdong") == [
        "x: from fengdong import ..."
    ]


def test_importing_flow_does_not_load_fengdong_widgets():
    """Runtime half, in a subprocess so this process's imports cannot mask it."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, flow, flow.case, flow.cli; "
         "assert not [m for m in sys.modules if m.startswith('fengdong')], "
         "sorted(m for m in sys.modules if m.startswith('fengdong'))"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_fengdong_main_still_answers_without_pygame_or_widgets():
    """Constraint 20's guard from T205: ``fengdong --version`` must not import the widgets."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, fengdong.__main__ as m; m.main(['--version']); "
         "assert 'pygame' not in sys.modules and 'fengdong.widgets' not in sys.modules"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Constraint 10 — the widgets draw chrome, never a field
# ---------------------------------------------------------------------------


def test_fengdong_defines_no_renderer_of_its_own():
    """The ``tests/test_flow_package.py::test_flow_defines_no_renderer_of_its_own`` shape, over ``fengdong/``."""
    offenders = []
    for path in _modules(FENGDONG_ROOT):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "render", "to_rgb", "colormap",
            }:
                offenders.append(f"{path.name}: def {node.name}")
    assert not offenders, f"fengdong/ colours nothing (constraint 10): {offenders}"


def test_widgets_never_touch_numpy_or_a_field():
    source = WIDGETS_PY.read_text(encoding="utf-8")
    assert "numpy" not in source and "surfarray" not in source and "vorticity" not in source


# ---------------------------------------------------------------------------
# Last: still no display, after everything above
# ---------------------------------------------------------------------------


def test_no_display_was_initialised_by_any_test_in_this_file():
    assert pygame.display.get_init() is False
