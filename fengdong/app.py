"""``fengdong/app.py`` — the window (T207).

Implements the second box of ``DOCS/IDEA4.md`` § What Phase 2 is, concretely:
*"the window, the event loop, the panels"* — sitting between
:mod:`fengdong.widgets` and :class:`flow.case.Case`, and § The five things
Phase 2 must get right (3): *"The app is a view, not a second brain."*

What this session ships (``DOCS/TASKS3.md`` § T207): a resizable window titled
**FengDong** with one :class:`~fengdong.widgets.DropTarget`, the setup panel —
a fluid, a speed, a size and a quality, entered through the T206 widgets and
nothing else (constraint 13) — a preview of the body :func:`flow.prepare.prepare`
made of the dropped picture with its verdict (**D-065**, **D-066**), and the
plan preview, which is :meth:`flow.case.Case.explain`'s own text and nothing
recomputed. **No simulation runs here.** The live view, the numbers panel and
save are T208; the drop rung is T209.

The three rules
---------------

* **Constraint 17** — this module imports ``flow`` and never ``lbm``. Every
  quantity the window shows about a plan — the grid, the relaxation time, the
  timestep, the run length, the expected fidelity band and the reason for each
  — is a line of :meth:`flow.case.Case.explain`, obtained from the
  :class:`~flow.case.Case` the window built and displayed verbatim. If a solver
  parameter were computed here the task would have failed regardless of what
  the window looked like, and ``tests/test_app.py`` reads this file's syntax
  to make sure none is.
* **Constraint 14** — a refused case shows the refusal :class:`~flow.case.Case`
  returned and the suggestions **in the order** :meth:`~flow.case.Case.nearest`
  would try them (:attr:`flow.case.Case.suggestions`, or the picture's own
  :class:`~flow.prepare.Fix`), and the *Use the nearest case* button applies
  exactly that method. One list, two surfaces — the same posture
  ``flow/cli.py::_print_refusal`` takes.
* **Constraint 10** — the body preview is a boolean array painted as two flat
  colours: solid and fluid. That is chrome. No field reaches this window
  until T208, and then only through :func:`lbm.render.render`.

Event model
-----------

**D-097** is honoured rather than re-invented: the loop is
``events = pygame.event.get()``, :meth:`App.handle`, :meth:`App.draw`, flip.
Characters arrive as ``TEXTINPUT`` (text input is never stopped), keys go to
the focused child, an open dropdown captures, and ``handle`` returns whether
the *case* changed. Building a :class:`~flow.case.Case` costs 0.4–1.1 s on this
machine (measured, session 30 — it prepares the picture and plans the physics),
so the plan is **not** rebuilt on every keystroke: a drop, a dropdown choice,
Enter in a field, or the *Preview the plan* button rebuilds it, and the status
line says when the fields have moved past the plan on screen (**D-098**).

Headless
--------

:class:`App` is built, driven and drawn without a display — the tests do all
three under ``SDL_VIDEODRIVER=dummy`` on an off-screen surface, and
:meth:`App.open` is the only method that touches :mod:`pygame.display`.
:meth:`App.close` quits it, and a test asserts the process ends with no
resource warning.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pygame
from numpy.typing import NDArray

from fengdong.widgets import (
    BORDER,
    ERROR,
    FIELD,
    FONT_SIZE,
    INK,
    MUTED,
    PAD,
    PAPER,
    ROW_HEIGHT,
    Button,
    Dropdown,
    DropTarget,
    Label,
    Panel,
    TextField,
    _font,
    _Text,
)
from flow.autoconfig import QUALITY_LEVELS
from flow.case import Case

__all__ = ["App", "TITLE", "DEFAULT_SIZE", "MIN_SIZE"]

#: The window caption — also the distribution and the command (**D-083**).
TITLE: str = "FengDong"
#: The window's opening size. Everything inside is laid out from the size the
#: window *has*, so this is a starting point and not a coordinate system.
DEFAULT_SIZE: tuple[int, int] = (1100, 720)
#: The window lays out for at least this size (and at least the height its
#: own column needs); a smaller window shows the top-left of that layout
#: rather than a squashed one.
MIN_SIZE: tuple[int, int] = (640, 480)

#: The two flat colours the body preview is painted in. Chrome, not a field.
SOLID_RGB: tuple[int, int, int] = (40, 40, 40)
FLUID_RGB: tuple[int, int, int] = (235, 240, 250)

#: Height of the drop target inside the column.
DROP_HEIGHT: int = 110
#: Vertical pitch of one line in the plan pane.
LINE_HEIGHT: int = FONT_SIZE + 2
#: Lines scrolled per wheel notch in the plan pane.
SCROLL_LINES: int = 3

#: The fraction of the window width the setup column takes, and its bounds.
COLUMN_FRACTION: float = 0.34
COLUMN_MIN: int = 300
COLUMN_MAX: int = 440
#: The fraction of the right-hand height the body preview takes.
PREVIEW_FRACTION: float = 0.38

PROMPT_DROP: str = "Drop a picture on the window to begin."
PROMPT_STALE: str = "The fields have changed: press Enter, or Preview the plan."
PROMPT_CURRENT: str = "The plan below is what `flow` would run."
PROMPT_REFUSED: str = "Refused. The way forward is listed below."


def _wrap_indented(text: str, width: int) -> list[str]:
    """Wrap one pre-formatted line to ``width`` pixels, keeping its indent.

    :meth:`flow.case.Case.explain` indents its lines to mean something — a
    ``why:`` under the number it explains — so a continuation line keeps the
    indent of the line it came from rather than snapping to the margin.
    """
    font = _font()
    indent = text[: len(text) - len(text.lstrip(" "))]
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = indent + words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] > width:
            lines.append(current)
            current = indent + "  " + word
        else:
            current = candidate
    lines.append(current)
    return lines


class _TextPane:
    """A block of lines with a scroll offset, re-wrapped only when its text or
    width changes and rendered through the widgets' own per-line cache.

    Not a widget: it reports nothing and consumes nothing but a wheel notch.
    It exists so the plan preview draws without allocating on an unchanged
    frame, which is the same rule the widgets keep (**D-097** (7)).
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.colour: tuple[int, int, int] = INK
        self.offset: int = 0
        self._wrapped: list[str] = []
        self._wrapped_for: tuple[str, int] | None = None
        self._lines: list[_Text] = []

    def set(self, text: str, *, colour: tuple[int, int, int] = INK) -> None:
        if text != self.text:
            self.offset = 0
        self.text = text
        self.colour = colour

    def scroll(self, notches: int, rect: pygame.Rect) -> None:
        visible = max(rect.height // LINE_HEIGHT, 1)
        self._rewrap(rect.width - 2 * PAD)
        top = max(len(self._wrapped) - visible, 0)
        self.offset = max(0, min(top, self.offset - notches * SCROLL_LINES))

    def _rewrap(self, width: int) -> None:
        key = (self.text, width)
        if key == self._wrapped_for:
            return
        self._wrapped = [
            piece for line in self.text.splitlines() for piece in _wrap_indented(line, width)
        ]
        self._wrapped_for = key
        if len(self._lines) < len(self._wrapped):
            self._lines.extend(_Text() for _ in range(len(self._wrapped) - len(self._lines)))

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, FIELD, rect)
        pygame.draw.rect(surface, BORDER, rect, width=1)
        self._rewrap(rect.width - 2 * PAD)
        visible = max(rect.height // LINE_HEIGHT, 1)
        self.offset = max(0, min(max(len(self._wrapped) - visible, 0), self.offset))
        y = rect.y + PAD
        for cache, text in zip(
            self._lines[self.offset : self.offset + visible],
            self._wrapped[self.offset : self.offset + visible],
        ):
            line = cache.surface(text, self.colour)
            surface.blit(line, (rect.x + PAD, y))
            y += LINE_HEIGHT


class App:
    """The window: a drop target, four inputs, a body preview and the plan.

    Build it with no display. :meth:`handle` is the whole state machine and
    :meth:`draw` paints onto any surface, so the tests drive it with
    synthesised events on an off-screen surface; :meth:`run` is the only
    caller of :meth:`open`.

    Args:
        size: the opening window size. Laid out again on every resize.
        backend: which backend :meth:`flow.case.Case.explain` estimates the
            wall clock for — ``python -m flow --backend``'s own choice, passed
            straight through to :class:`~flow.case.Case` (**D-073**: the window
            contradicts none of the CLI's flags).

    Attributes:
        path: the last dropped file, or ``None``.
        case: the :class:`~flow.case.Case` built from the inputs, refused or
            not, or ``None`` while there is no picture or the picture could
            not be read.
        error: what went wrong reading the picture, in the library's own
            words, or ``None``.
        stale: whether a field has changed since :attr:`case` was built.
        quit_requested: the close button was pressed.
    """

    def __init__(
        self,
        *,
        size: tuple[int, int] = DEFAULT_SIZE,
        backend: str = "numpy",
    ) -> None:
        self.backend: str = backend
        self.path: str | None = None
        self.case: Case | None = None
        self.error: str | None = None
        self.stale: bool = False
        self.quit_requested: bool = False
        self._rebuild_pending: bool = False
        self._nearest_pending: bool = False

        # -- the setup column: one Panel, the T206 widgets, nothing else ------
        self.title = Label((0, 0, 0, ROW_HEIGHT), f"{TITLE} — drop a picture, type three numbers")
        self.target = DropTarget((0, 0, 0, DROP_HEIGHT), on_drop=self._on_drop)
        self.fluid = Dropdown.fluids((0, 0, 0, ROW_HEIGHT), on_change=self._on_choice)
        self.speed = TextField.speed(
            (0, 0, 0, TextField.DEFAULT_HEIGHT),
            placeholder="e.g. 5 mm/s, 20 km/h",
            on_commit=self._on_commit,
        )
        self.size = TextField.size(
            (0, 0, 0, TextField.DEFAULT_HEIGHT),
            placeholder="e.g. 2 cm, 1.5 m",
            on_commit=self._on_commit,
        )
        self.quality = Dropdown(
            (0, 0, 0, ROW_HEIGHT), QUALITY_LEVELS, selected="balanced", on_change=self._on_choice
        )
        self.preview = Button((0, 0, 0, ROW_HEIGHT), "Preview the plan", on_click=self._on_preview)
        self.nearest_button = Button(
            (0, 0, 0, ROW_HEIGHT), "Use the nearest case that runs",
            on_click=self._on_nearest, enabled=False,
        )
        self.status = Label((0, 0, 0, 2 * ROW_HEIGHT), PROMPT_DROP, colour=MUTED)
        self.panel = Panel(
            (0, 0, COLUMN_MIN, 100),
            [
                self.title,
                self.target,
                Label((0, 0, 0, ROW_HEIGHT), "Fluid"),
                self.fluid,
                Label((0, 0, 0, ROW_HEIGHT), "Speed"),
                self.speed,
                Label((0, 0, 0, ROW_HEIGHT), "Size across the flow"),
                self.size,
                Label((0, 0, 0, ROW_HEIGHT), "Quality"),
                self.quality,
                self.preview,
                self.nearest_button,
                self.status,
            ],
        )

        # -- the right-hand side: the body preview and the plan pane ----------
        self.verdict = Label((0, 0, 0, ROW_HEIGHT), "", colour=MUTED)
        self.plan_pane = _TextPane()
        self.plan_pane.set(PROMPT_DROP, colour=MUTED)
        self.preview_rect = pygame.Rect(0, 0, 0, 0)
        self.plan_rect = pygame.Rect(0, 0, 0, 0)
        self._mask_surface: pygame.Surface | None = None
        self._mask_scaled: pygame.Surface | None = None
        self._mask_scaled_for: tuple[int, int] | None = None

        self._screen: pygame.Surface | None = None
        self.window_size: tuple[int, int] = size
        self.layout(size)

    # -- layout ----------------------------------------------------------------

    def layout(self, size: tuple[int, int]) -> None:
        """Place the column, the preview and the plan pane for ``size``.

        Everything is a fraction of the window with a clamp, so the same code
        lays out 640x480 and 2560x1440; nothing is a pixel count that only
        works at one size. Runs on construction and on every resize — never
        per frame.
        """
        w = max(int(size[0]), MIN_SIZE[0])
        # The column's own height is the floor the widgets set (their heights
        # are theirs, D-097 (6)); a window shorter than that shows the top of
        # a layout made for the height it needs rather than a squashed one.
        h = max(int(size[1]), MIN_SIZE[1], self.panel.content_height)
        self.window_size = (w, h)
        column = max(COLUMN_MIN, min(COLUMN_MAX, int(w * COLUMN_FRACTION)))
        self.panel.resize((0, 0, column, h))
        right = pygame.Rect(column, 0, w - column, h)
        preview_h = max(int(right.height * PREVIEW_FRACTION), 3 * ROW_HEIGHT)
        self.verdict.rect.update(right.x + PAD, right.y + PAD, right.width - 2 * PAD, ROW_HEIGHT)
        self.preview_rect.update(
            right.x + PAD,
            right.y + PAD + ROW_HEIGHT,
            right.width - 2 * PAD,
            preview_h - ROW_HEIGHT - PAD,
        )
        self.plan_rect.update(
            right.x + PAD,
            right.y + preview_h + PAD,
            right.width - 2 * PAD,
            right.height - preview_h - 2 * PAD,
        )
        self._mask_scaled = None
        self._mask_scaled_for = None

    # -- the state machine -------------------------------------------------------

    def _on_drop(self, path: str) -> None:
        self.path = path
        self._rebuild_pending = True

    def _on_choice(self, _value: str) -> None:
        self._rebuild_pending = True

    def _on_commit(self, _value: Any) -> None:
        self._rebuild_pending = True

    def _on_preview(self) -> None:
        self._rebuild_pending = True

    def _on_nearest(self) -> None:
        self._nearest_pending = True

    def handle(self, events: Iterable[pygame.event.Event]) -> bool:
        """Consume one frame's events. Returns whether :attr:`case` changed.

        The window-level events — quit, resize, the wheel over the plan —
        are taken here; everything else goes to the :class:`Panel`, whose
        widgets report through their callbacks. A pending rebuild or
        substitution is then applied once, after every event of the frame.
        """
        panel_events: list[pygame.event.Event] = []
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.VIDEORESIZE:
                self.layout((event.w, event.h))
            elif event.type == pygame.MOUSEWHEEL:
                self.plan_pane.scroll(event.y, self.plan_rect)
            else:
                panel_events.append(event)
        fields_before = (self.speed.text, self.size.text)
        self.panel.handle(panel_events)
        if (self.speed.text, self.size.text) != fields_before and self.case is not None:
            self.stale = True

        changed = False
        if self._nearest_pending:
            self._nearest_pending = False
            changed = self._apply_nearest()
        if self._rebuild_pending:
            self._rebuild_pending = False
            changed = self.rebuild() or changed
        self._refresh_status()
        return changed

    def rebuild(self) -> bool:
        """Build the :class:`~flow.case.Case` from the inputs as they stand.

        Returns whether a case was built or dropped. A field that does not
        parse leaves the previous case in place and says so in the status
        line — the widget already shows the parser's own message under the
        field (constraint 14's posture, **D-097** (3)).
        """
        if self.path is None:
            return False
        if self.speed.value is None or self.size.value is None:
            self.error = None
            self.stale = True
            return False
        try:
            case = Case.from_image(
                self.path,
                fluid=self.fluid.selected,
                speed=self.speed.value,
                size=self.size.value,
                quality=self.quality.selected,
                backend=self.backend,
            )
        except (FileNotFoundError, OSError, ValueError, KeyError, ImportError) as exc:
            # The library's own message, verbatim: a file that is not a
            # picture, a picture Pillow cannot open, a path that has gone.
            self.error = str(exc)
            self.case = None
            self._show_case()
            return True
        self.error = None
        self.case = case
        self.stale = False
        self._show_case()
        return True

    def _apply_nearest(self) -> bool:
        """Apply :meth:`flow.case.Case.nearest` — the tool's own top suggestion.

        The case it returns replaces the one on screen, marked as a
        substitution in its own ``explain()`` (constraint 16), and the fields
        are moved to say what is now being previewed rather than what was
        typed — a window whose fields disagree with its plan would be lying
        about one of them.
        """
        if self.case is None or self.case.runnable:
            return False
        try:
            case = self.case.nearest()
        except ValueError as exc:  # pragma: no cover - constraint 14 forbids it
            self.error = str(exc)
            self._show_case()
            return True
        self.case = case
        self.error = None
        self.stale = False
        # The widgets are moved to the substituted case's values *without*
        # their callbacks: a callback here would queue a rebuild from the
        # fields, which would replace the substituted case with a fresh,
        # unsubstituted one and lose the disclosure constraint 16 requires.
        if case.fluid.name in self.fluid.options:
            self.fluid.index = self.fluid.options.index(case.fluid.name)
        self.speed.set_text(str(case.speed))
        self.size.set_text(str(case.size))
        self.quality.index = self.quality.options.index(case.quality)
        self._rebuild_pending = False
        self._show_case()
        return True

    # -- what is shown -------------------------------------------------------------

    def _show_case(self) -> None:
        """Move the case, whatever it is, onto the screen: verdict, body, plan."""
        self._mask_surface = None
        self._mask_scaled = None
        self._mask_scaled_for = None
        case = self.case
        if case is None:
            self.verdict.text = "could not read the picture" if self.error else ""
            self.verdict.colour = ERROR
            self.plan_pane.set(self.error or PROMPT_DROP, colour=ERROR if self.error else MUTED)
            self.nearest_button.enabled = False
            return
        prepared = case.prepared
        self.verdict.text = f"geometry: {prepared.verdict}" + (
            f" — {prepared.actions[0]}" if prepared.actions else ""
        )
        self.verdict.colour = ERROR if not prepared.runnable else INK
        if prepared.mask.size:
            self._mask_surface = _mask_to_surface(prepared.mask)
        self.plan_pane.set(self.plan_text(), colour=INK)
        self.nearest_button.enabled = not case.runnable

    def plan_text(self) -> str:
        """The plan preview: :meth:`flow.case.Case.explain` verbatim, plus — for
        a refusal — the list :meth:`flow.case.Case.nearest` would act on, in
        its order, exactly as ``flow/cli.py::_print_refusal`` prints it.

        Nothing here is computed. The lines are the library's.
        """
        case = self.case
        if case is None:
            return self.error or PROMPT_DROP
        text = case.explain(quiet=True)
        if case.runnable:
            return text
        offers = case.suggestions
        lines = [text, ""]
        if offers:
            lines.append("What the button above would run, in the order it would try them")
            for rank, offer in enumerate(offers, 1):
                lines.append(f"  {rank}. {offer.change} -> {offer.value}")
                lines.append(f"     {offer.note}")
        elif case.fix is not None:
            lines.append("What the button above would do")
            lines.append(f"  {case.fix.change} -> {case.fix.value}")
            lines.append(f"     {case.fix.note}")
        return "\n".join(lines)

    def shown_suggestion(self) -> str | None:
        """The one change the *nearest* button acts on, as the pane shows it.

        ``"<change> -> <value>"`` for the top of :attr:`flow.case.Case.suggestions`,
        or the picture's :class:`~flow.prepare.Fix`; ``None`` when the case
        runs. Constraint 14: the tests compare this with what
        :meth:`flow.case.Case.nearest` actually did.
        """
        case = self.case
        if case is None or case.runnable:
            return None
        offers = case.suggestions
        if offers:
            return f"{offers[0].change} -> {offers[0].value}"
        if case.fix is not None:
            return f"{case.fix.change} -> {case.fix.value}"
        return None  # pragma: no cover - constraint 14 forbids it

    def _refresh_status(self) -> None:
        if self.error:
            self.status.text, self.status.colour = self.error, ERROR
        elif self.path is None:
            self.status.text, self.status.colour = PROMPT_DROP, MUTED
        elif self.speed.value is None or self.size.value is None:
            missing = "speed" if self.speed.value is None else "size"
            self.status.text = (
                f"Enter a {missing}." if not getattr(self, missing).text
                else f"The {missing} does not parse — see the field."
            )
            self.status.colour = ERROR if getattr(self, missing).error else MUTED
        elif self.case is None or self.stale:
            self.status.text, self.status.colour = PROMPT_STALE, MUTED
        elif not self.case.runnable:
            self.status.text, self.status.colour = PROMPT_REFUSED, ERROR
        else:
            self.status.text, self.status.colour = PROMPT_CURRENT, INK

    # -- drawing -----------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        """Paint the whole window onto ``surface``. Allocates only when the
        body preview's target size moved (a resize or a new picture)."""
        surface.fill(PAPER)
        self.panel.draw(surface)
        self.verdict.draw(surface)
        pygame.draw.rect(surface, FIELD, self.preview_rect)
        pygame.draw.rect(surface, BORDER, self.preview_rect, width=1)
        if self._mask_surface is not None:
            scaled = self._scaled_mask()
            surface.blit(
                scaled,
                (
                    self.preview_rect.centerx - scaled.get_width() // 2,
                    self.preview_rect.centery - scaled.get_height() // 2,
                ),
            )
        self.plan_pane.draw(surface, self.plan_rect)

    def _scaled_mask(self) -> pygame.Surface:
        """The body preview at the size that fits the preview box, aspect kept.
        Built once per (picture, box size); nearest-neighbour so a cell is a cell."""
        assert self._mask_surface is not None
        box = (max(self.preview_rect.width - 2 * PAD, 1), max(self.preview_rect.height - 2 * PAD, 1))
        if self._mask_scaled is not None and self._mask_scaled_for == box:
            return self._mask_scaled
        w, h = self._mask_surface.get_size()
        factor = min(box[0] / w, box[1] / h)
        target = (max(int(w * factor), 1), max(int(h * factor), 1))
        self._mask_scaled = pygame.transform.scale(self._mask_surface, target)
        self._mask_scaled_for = box
        return self._mask_scaled

    # -- the window ----------------------------------------------------------------------

    def open(self) -> pygame.Surface:
        """Open the display. The only method that touches :mod:`pygame.display`."""
        if not pygame.display.get_init():
            pygame.display.init()
        self._screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        pygame.display.set_caption(TITLE)
        return self._screen

    def close(self) -> None:
        """Shut the display down. Idempotent; fonts stay up for the widgets."""
        self._screen = None
        if pygame.display.get_init():
            pygame.display.quit()

    def run(self) -> int:
        """Open the window and loop until it is closed. Returns the exit code.

        Blocks in :func:`pygame.event.wait` between events rather than
        spinning: nothing in this window moves on its own, so a frame is
        drawn after events and not on a timer.
        """
        screen = self.open()
        try:
            while not self.quit_requested:
                first = pygame.event.wait(200)
                events = [first] if first.type != pygame.NOEVENT else []
                events.extend(pygame.event.get())
                self.handle(events)
                if self._screen is not None and self._screen.get_size() != self.window_size:
                    screen = self._screen = pygame.display.get_surface() or screen
                self.draw(screen)
                pygame.display.flip()
        finally:
            self.close()
        return 0


def _mask_to_surface(mask: NDArray[np.bool_]) -> pygame.Surface:
    """A ``(h, w)`` bool body, ``True`` on solid, as a surface of two flat
    colours. This is the whole of what ``fengdong/`` does to an array
    (constraint 10): it does not colour a field, it paints a stencil."""
    h, w = mask.shape
    rgb = np.empty((w, h, 3), dtype=np.uint8)  # pygame's surfarray is (x, y, 3)
    solid = np.transpose(mask)
    rgb[...] = FLUID_RGB
    rgb[solid] = SOLID_RGB
    return pygame.surfarray.make_surface(rgb)
