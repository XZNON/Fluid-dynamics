"""``fengdong/widgets.py`` — the closed widget set (T206).

Implements the top box of ``DOCS/IDEA4.md`` § What Phase 2 is, concretely:
*"label, text field, dropdown, button, drop target. that list, closed."*
Five widgets and a :class:`Panel` that stacks them in one column. Nothing
else — **D-083** closed the set on day one, and ``DOCS/PLAN3.md`` § Risks
names the trap this module must not fall into: *"Hand-rolled widgets swallow
the phase."* No layout engine, no theming, no animation, no focus chain
beyond tab order. A sixth widget is ``/new-task``.

What a widget is here
---------------------

Each widget owns a :class:`pygame.Rect`, draws itself onto whatever
:class:`pygame.Surface` it is handed, and consumes a list of
:class:`pygame.event.Event` objects, returning whether its *reported state*
changed — the text field's value, the dropdown's selection, the button's
click, the drop target's path. Focus and hover are drawn but are not
"changes": the window redraws every frame regardless.

Every widget is driven headless. Fonts come from :mod:`pygame.font`, which
initialises without a display; surfaces are off-screen; events are plain
:class:`pygame.event.Event` objects a test can synthesise. Nothing in this
module calls :func:`pygame.init` or touches :mod:`pygame.display`, and
``tests/test_widgets.py`` asserts no display is initialised after every
widget has been built, driven and drawn.

The two rules of ``fengdong/`` (``DOCS/IDEA4.md`` § The five things Phase 2
must get right (3))
--------------------------------------------------------------------------

* **Constraint 17** — this module imports ``flow/`` (:func:`flow.quantity.parse`
  and :data:`flow.fluids.FLUIDS`) and never ``lbm/``. It computes no solver
  parameter: the text field *validates* what the user typed through the same
  parser :class:`flow.case.Case` uses, and hands back the
  :class:`~flow.quantity.Quantity` it got; it never converts, rounds or
  derives.
* **Constraint 13** — no lattice quantity is accepted or displayed. The
  inputs are a picture, a fluid, a speed and a size, and the identifiers in
  this file are scanned against the vocabulary ``tests/test_flow_package.py``
  already forbids in ``flow/``.

And **constraint 10**: the widgets draw *chrome* — borders, text, a caret, a
highlight — and never a field. Frames become pixels in :func:`lbm.render.render`
and nowhere else; there is no ``render`` here and no colour map.

Text entry
----------

Characters arrive as :data:`pygame.TEXTINPUT` events, which SDL2 delivers
by default and which carry the composed character rather than the key —
so a shifted ``2`` on a German keyboard is ``"``, not ``@``.
:data:`pygame.KEYDOWN` is used only for the editing keys (backspace,
enter, escape, arrows). A window that stops text input
(``pygame.key.stop_text_input()``) stops the field receiving characters;
T207 must not.

Per-frame cost
--------------

``draw`` allocates nothing when nothing changed: rendered text surfaces are
cached per widget and re-rendered only when the text or its colour
changes. ``tests/test_widgets.py`` asserts a second draw of an unchanged
widget calls :meth:`pygame.font.Font.render` zero times.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import pygame

from flow.fluids import FLUIDS
from flow.quantity import LENGTH, SPEED, Quantity, parse

__all__ = [
    "Widget",
    "Label",
    "TextField",
    "Dropdown",
    "Button",
    "DropTarget",
    "Panel",
    "FONT_SIZE",
    "ROW_HEIGHT",
    "PAD",
]

# ---------------------------------------------------------------------------
# The one palette, the one font. Constants, not a theme: there is no second
# set and no way to install one (D-083 — "no theming").
# ---------------------------------------------------------------------------

#: Point size of the single UI font.
FONT_SIZE: int = 18
#: Height of one row of chrome — a label, a field line, a dropdown row.
ROW_HEIGHT: int = 28
#: Inner padding between a border and its text, and between stacked widgets.
PAD: int = 6

RGB = tuple[int, int, int]

INK: RGB = (30, 30, 30)
MUTED: RGB = (120, 120, 120)
PAPER: RGB = (245, 245, 245)
FIELD: RGB = (255, 255, 255)
BORDER: RGB = (160, 160, 160)
FOCUS: RGB = (60, 120, 200)
HOVER: RGB = (225, 235, 250)
PRESSED: RGB = (200, 215, 240)
ERROR: RGB = (180, 40, 40)
DROP_HOVER: RGB = (225, 245, 225)

_FONT: pygame.font.Font | None = None


def _font() -> pygame.font.Font:
    """The UI font, created once. ``pygame.font`` needs no display."""
    global _FONT
    if _FONT is None:
        if not pygame.font.get_init():
            pygame.font.init()
        _FONT = pygame.font.Font(None, FONT_SIZE)
    return _FONT


class _Text:
    """One rendered line, re-rendered only when its text or colour changes.

    This is what keeps ``draw`` allocation-free on an unchanged frame: a
    widget holds one ``_Text`` per line it draws and calls :meth:`surface`
    every frame, which returns the cached surface unless the key moved.
    """

    __slots__ = ("_key", "_surface")

    def __init__(self) -> None:
        self._key: tuple[str, RGB] | None = None
        self._surface: pygame.Surface | None = None

    def surface(self, text: str, colour: RGB) -> pygame.Surface:
        key = (text, colour)
        if key != self._key:
            self._surface = _font().render(text, True, colour)
            self._key = key
        assert self._surface is not None
        return self._surface


def _wrap(text: str, width: int) -> list[str]:
    """Break ``text`` into lines no wider than ``width`` pixels, on spaces.

    A single word wider than ``width`` is left on its own line rather than
    split mid-word, because the message is the user's and must stay readable
    (constraint 14's posture: the fix is named in words they can act on).
    """
    font = _font()
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.size(candidate)[0] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _clip(text: str, width: int) -> str:
    """``text`` shortened with an ellipsis so it fits in ``width`` pixels."""
    font = _font()
    if font.size(text)[0] <= width:
        return text
    for end in range(len(text) - 1, 0, -1):
        candidate = text[:end] + "…"
        if font.size(candidate)[0] <= width:
            return candidate
    return "…"


# ---------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------


class Widget:
    """What every widget has: a rect, ``draw``, ``handle``.

    Args:
        rect: anything :class:`pygame.Rect` accepts — ``(x, y, w, h)`` or a
            ``Rect``. A :class:`Panel` overwrites ``x``, ``y`` and ``width``
            when it lays the widget out and keeps ``height``.

    Attributes:
        rect: where the widget draws and what it hit-tests against.
        focused: whether keyboard events are routed here. Set by a mouse
            click inside the rect, by :class:`Panel`'s tab order, or directly.
        focusable: class flag — whether tab order visits this widget.
    """

    focusable: bool = False

    def __init__(self, rect: pygame.Rect | tuple[int, int, int, int]) -> None:
        self.rect: pygame.Rect = pygame.Rect(rect)
        self.focused: bool = False

    def draw(self, surface: pygame.Surface) -> None:
        """Paint the widget onto ``surface``. Chrome only — never a field."""
        raise NotImplementedError

    def handle(self, events: Iterable[pygame.event.Event]) -> bool:
        """Consume ``events`` in order; return whether the reported state changed."""
        changed = False
        for event in events:
            changed = self.handle_event(event) or changed
        return changed

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Consume one event; return whether the reported state changed."""
        return False

    # -- helpers shared by the concrete widgets ------------------------------

    def _hit(self, event: pygame.event.Event) -> bool:
        """Whether a mouse event landed inside this widget's rect."""
        return self.rect.collidepoint(event.pos)

    def _take_focus_from_click(self, event: pygame.event.Event) -> None:
        """A left click focuses the widget under it and blurs every other one."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self._hit(event)


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


class Label(Widget):
    """One line of text. It reports nothing and consumes nothing.

    Args:
        rect: see :class:`Widget`.
        text: what to show. Clipped with an ellipsis to the rect's width.
        colour: the ink; :data:`MUTED` for a hint, :data:`ERROR` for a verdict.
    """

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        *,
        colour: RGB = INK,
    ) -> None:
        super().__init__(rect)
        self._text: str = text
        self.colour: RGB = colour
        self._line = _Text()

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        self._text = value

    def draw(self, surface: pygame.Surface) -> None:
        shown = _clip(self._text, max(self.rect.width - 2 * PAD, 0))
        line = self._line.surface(shown, self.colour)
        surface.blit(line, (self.rect.x + PAD, self.rect.y + (self.rect.height - line.get_height()) // 2))


# ---------------------------------------------------------------------------
# TextField
# ---------------------------------------------------------------------------


class TextField(Widget):
    """A single-line field for a physical quantity, validated as it is typed.

    Validation is :func:`flow.quantity.parse` with the *same* ``expect`` and
    ``default_unit`` that :class:`flow.case.Case` passes for that input
    (``flow/case.py``: speed is ``expect=SPEED, default_unit="m/s"``, size is
    ``expect=LENGTH, default_unit="m"``). A failure's message is therefore the
    one ``python -m flow`` prints to stderr, obtained from the same code path
    and not re-worded — constraint 14's posture, and
    ``tests/test_widgets.py`` asserts the two strings are identical.

    Args:
        rect: see :class:`Widget`. Height should allow the field row plus
            two rows of error text; :attr:`DEFAULT_HEIGHT` is that.
        expect: the dimension required — :data:`flow.quantity.SPEED` or
            :data:`flow.quantity.LENGTH` for the two inputs the app has.
        default_unit: what a bare number means, exactly as ``Case`` declares it.
        text: initial contents.
        placeholder: hint drawn in :data:`MUTED` while the field is empty.
        on_commit: called with the parsed :class:`~flow.quantity.Quantity`
            when Enter is pressed on a valid field.

    Attributes:
        text: what the user has typed, verbatim.
        value: the parsed quantity, or ``None`` while the text is empty or
            does not parse.
        error: the parser's message in the user's words, or ``None``. An
            empty field has no error — nothing has been asked yet.
    """

    focusable = True
    #: Rows of wrapped error text drawn under the field. The full message is
    #: always in :attr:`error`; this is how much of it the widget shows.
    ERROR_ROWS: int = 3
    #: Field row plus :attr:`ERROR_ROWS` rows of wrapped error text.
    DEFAULT_HEIGHT: int = ROW_HEIGHT + ERROR_ROWS * (FONT_SIZE + 2)

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        *,
        expect: str,
        default_unit: str,
        text: str = "",
        placeholder: str = "",
        on_commit: Callable[[Quantity], None] | None = None,
    ) -> None:
        super().__init__(rect)
        self.expect: str = expect
        self.default_unit: str = default_unit
        self.placeholder: str = placeholder
        self.on_commit = on_commit
        self.text: str = ""
        self.value: Quantity | None = None
        self.error: str | None = None
        self._line = _Text()
        self._error_lines: list[_Text] = [_Text() for _ in range(self.ERROR_ROWS)]
        self._error_wrapped: list[str] = []
        self._error_wrapped_for: tuple[str, int] | None = None
        self.set_text(text)

    # -- the two constructors the app actually needs ------------------------

    @classmethod
    def speed(cls, rect: pygame.Rect | tuple[int, int, int, int], **kw: object) -> "TextField":
        """A field for the speed, validated exactly as ``Case`` validates it."""
        return cls(rect, expect=SPEED, default_unit="m/s", **kw)  # type: ignore[arg-type]

    @classmethod
    def size(cls, rect: pygame.Rect | tuple[int, int, int, int], **kw: object) -> "TextField":
        """A field for the size, validated exactly as ``Case`` validates it."""
        return cls(rect, expect=LENGTH, default_unit="m", **kw)  # type: ignore[arg-type]

    # -- state -----------------------------------------------------------------

    def set_text(self, text: str) -> bool:
        """Replace the contents and re-validate. Returns whether the text changed."""
        changed = text != self.text
        self.text = text
        if not text.strip():
            self.value, self.error = None, None
        else:
            try:
                self.value = parse(text, expect=self.expect, default_unit=self.default_unit)
                self.error = None
            except ValueError as exc:
                # The message, verbatim. It is the same ValueError
                # flow.cli.main prints, because it comes from the same call.
                self.value, self.error = None, str(exc)
        return changed

    def handle_event(self, event: pygame.event.Event) -> bool:
        self._take_focus_from_click(event)
        if not self.focused:
            return False
        if event.type == pygame.TEXTINPUT:
            return self.set_text(self.text + event.text)
        if event.type != pygame.KEYDOWN:
            return False
        if event.key == pygame.K_BACKSPACE:
            return self.set_text(self.text[:-1]) if self.text else False
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.value is not None and self.on_commit is not None:
                self.on_commit(self.value)
            return False
        if event.key == pygame.K_ESCAPE:
            self.focused = False
        return False

    # -- drawing ---------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        row = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, ROW_HEIGHT)
        pygame.draw.rect(surface, FIELD, row)
        border = ERROR if self.error else (FOCUS if self.focused else BORDER)
        pygame.draw.rect(surface, border, row, width=2 if self.focused else 1)

        inner_width = max(row.width - 2 * PAD, 0)
        if self.text:
            line = self._line.surface(_clip(self.text, inner_width), INK)
        else:
            line = self._line.surface(_clip(self.placeholder, inner_width), MUTED)
        text_y = row.y + (row.height - line.get_height()) // 2
        surface.blit(line, (row.x + PAD, text_y))

        if self.focused:
            caret_x = row.x + PAD + (line.get_width() if self.text else 0) + 1
            pygame.draw.line(surface, INK, (caret_x, row.y + PAD), (caret_x, row.bottom - PAD))

        if self.error:
            key = (self.error, inner_width)
            if key != self._error_wrapped_for:
                self._error_wrapped = _wrap(self.error, inner_width)
                self._error_wrapped_for = key
            y = row.bottom + 2
            for cache, text in zip(self._error_lines, self._error_wrapped):
                line = cache.surface(text, ERROR)
                surface.blit(line, (row.x + PAD, y))
                y += line.get_height() + 2


# ---------------------------------------------------------------------------
# Dropdown
# ---------------------------------------------------------------------------


class Dropdown(Widget):
    """Pick one string from a fixed list. Click to open, click a row to choose.

    Args:
        rect: see :class:`Widget`. The list opens *below* the rect, one
            :data:`ROW_HEIGHT` per option, and is drawn by :meth:`draw_overlay`
            so a :class:`Panel` can paint it over the widgets underneath.
        options: the choices, in order. Must be non-empty.
        selected: the initial choice, by index or by value.
        on_change: called with the new value after the user picks one.

    Attributes:
        options: the choices, frozen at construction.
        index: the selected position.
        open: whether the list is showing.
    """

    focusable = True

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        options: Sequence[str],
        *,
        selected: int | str = 0,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(rect)
        if not options:
            raise ValueError("a Dropdown needs at least one option")
        self.options: tuple[str, ...] = tuple(options)
        self.index: int = (
            self.options.index(selected) if isinstance(selected, str) else int(selected)
        )
        if not 0 <= self.index < len(self.options):
            raise ValueError(f"selected={selected!r} is not one of {self.options}")
        self.on_change = on_change
        self.open: bool = False
        self._line = _Text()
        self._rows: list[_Text] = [_Text() for _ in self.options]

    @classmethod
    def fluids(
        cls,
        rect: pygame.Rect | tuple[int, int, int, int],
        *,
        selected: int | str = 0,
        on_change: Callable[[str], None] | None = None,
    ) -> "Dropdown":
        """The fluid picker, populated from :data:`flow.fluids.FLUIDS` **now**.

        The library is read at construction, not at import, so a fluid added
        to ``flow/fluids.py`` is in the widget with no edit here —
        ``tests/test_widgets.py`` adds one and asserts it appears.
        """
        return cls(rect, tuple(FLUIDS), selected=selected, on_change=on_change)

    @property
    def selected(self) -> str:
        return self.options[self.index]

    def select(self, index: int) -> bool:
        """Choose by position. Returns whether the selection moved."""
        index = max(0, min(len(self.options) - 1, index))
        if index == self.index:
            return False
        self.index = index
        if self.on_change is not None:
            self.on_change(self.selected)
        return True

    def row_rect(self, index: int) -> pygame.Rect:
        """Where option ``index`` is drawn while the list is open."""
        return pygame.Rect(
            self.rect.x, self.rect.bottom + index * ROW_HEIGHT, self.rect.width, ROW_HEIGHT
        )

    def captures(self, event: pygame.event.Event) -> bool:
        """Whether an open list should see ``event`` before anyone else.

        While open, every click and key belongs to the list: a click outside
        it closes it and goes no further, so a button underneath cannot fire
        by accident.
        """
        return self.open and event.type in (
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
            pygame.KEYDOWN,
            pygame.TEXTINPUT,
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.open:
                self.open = False
                for i in range(len(self.options)):
                    if self.row_rect(i).collidepoint(event.pos):
                        return self.select(i)
                return False
            self._take_focus_from_click(event)
            if self.focused:
                self.open = True
            return False
        if event.type != pygame.KEYDOWN or not self.focused:
            return False
        if event.key == pygame.K_DOWN:
            return self.select(self.index + 1)
        if event.key == pygame.K_UP:
            return self.select(self.index - 1)
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.open = not self.open
        elif event.key == pygame.K_ESCAPE:
            self.open = False
        return False

    def draw(self, surface: pygame.Surface, *, overlay: bool = True) -> None:
        """The closed box. ``overlay=True`` also draws the open list, for a
        widget used on its own; a :class:`Panel` passes ``False`` and calls
        :meth:`draw_overlay` after every other widget has drawn."""
        pygame.draw.rect(surface, FIELD, self.rect)
        pygame.draw.rect(
            surface, FOCUS if self.focused else BORDER, self.rect, width=2 if self.focused else 1
        )
        inner_width = max(self.rect.width - 2 * PAD - ROW_HEIGHT, 0)
        line = self._line.surface(_clip(self.selected, inner_width), INK)
        surface.blit(line, (self.rect.x + PAD, self.rect.y + (self.rect.height - line.get_height()) // 2))
        # the disclosure mark: a small triangle at the right edge
        cx = self.rect.right - ROW_HEIGHT // 2
        cy = self.rect.centery
        pygame.draw.polygon(surface, INK, [(cx - 5, cy - 3), (cx + 5, cy - 3), (cx, cy + 4)])
        if overlay:
            self.draw_overlay(surface)

    def draw_overlay(self, surface: pygame.Surface) -> None:
        """The open list, if open. Drawn last so it sits over its neighbours."""
        if not self.open:
            return
        for i, (cache, option) in enumerate(zip(self._rows, self.options)):
            row = self.row_rect(i)
            pygame.draw.rect(surface, HOVER if i == self.index else FIELD, row)
            pygame.draw.rect(surface, BORDER, row, width=1)
            line = cache.surface(_clip(option, max(row.width - 2 * PAD, 0)), INK)
            surface.blit(line, (row.x + PAD, row.y + (row.height - line.get_height()) // 2))


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------


class Button(Widget):
    """Press it. A click is a left button down *and* up inside the rect, or
    Enter / Space while focused.

    Args:
        rect: see :class:`Widget`.
        text: the caption.
        on_click: called with no arguments on every click.
        enabled: a disabled button draws :data:`MUTED` and ignores everything.

    Attributes:
        clicks: how many clicks have happened. :meth:`take_click` reads and
            resets it for a poll-style caller.
        pressed: the mouse went down inside and has not come up yet.
    """

    focusable = True

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        *,
        on_click: Callable[[], None] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(rect)
        self.text: str = text
        self.on_click = on_click
        self.enabled: bool = enabled
        self.pressed: bool = False
        self.clicks: int = 0
        self._line = _Text()

    def click(self) -> bool:
        """Register one click. Returns ``True`` — the reported state changed."""
        self.clicks += 1
        if self.on_click is not None:
            self.on_click()
        return True

    def take_click(self) -> bool:
        """Whether a click happened since the last call; resets the count."""
        happened = self.clicks > 0
        self.clicks = 0
        return happened

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._take_focus_from_click(event)
            self.pressed = self.focused
            return False
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed, self.pressed = self.pressed, False
            return self.click() if was_pressed and self._hit(event) else False
        if event.type == pygame.KEYDOWN and self.focused:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return self.click()
        return False

    def draw(self, surface: pygame.Surface) -> None:
        fill = PRESSED if self.pressed else (HOVER if self.focused else FIELD)
        pygame.draw.rect(surface, fill, self.rect)
        pygame.draw.rect(
            surface,
            MUTED if not self.enabled else (FOCUS if self.focused else BORDER),
            self.rect,
            width=2 if self.focused else 1,
        )
        line = self._line.surface(
            _clip(self.text, max(self.rect.width - 2 * PAD, 0)), INK if self.enabled else MUTED
        )
        surface.blit(
            line,
            (
                self.rect.x + (self.rect.width - line.get_width()) // 2,
                self.rect.y + (self.rect.height - line.get_height()) // 2,
            ),
        )


# ---------------------------------------------------------------------------
# DropTarget
# ---------------------------------------------------------------------------


class DropTarget(Widget):
    """Where the picture lands. Consumes :data:`pygame.DROPFILE` and reports the path.

    This is the one widget that is not negotiable (``DOCS/TASKS3.md`` § T206
    Notes): the fall-back for everything else is a file dialog, but *"drops a
    picture on it"* is the phase's own sentence. SDL delivers a drop as
    ``DROPBEGIN`` → one ``DROPFILE`` per file → ``DROPCOMPLETE``; the target
    highlights between the first and the last and records the file. A drop
    carries no position on this SDL (2.28), so any file dropped anywhere on
    the window reaches this widget — and there is one of them per window.

    Args:
        rect: see :class:`Widget`.
        prompt: what to show while nothing has been dropped.
        on_drop: called with the path after each drop.

    Attributes:
        path: the last dropped file, or ``None``. What it *is* — a picture,
            a mask, garbage — is :func:`flow.prepare`'s verdict, not this
            widget's; the widget reports the path and no more.
        hovering: a drag is in progress over the window.
        drops: how many files have been dropped, for a poll-style caller.
    """

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        *,
        prompt: str = "Drop a picture here",
        on_drop: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(rect)
        self.prompt: str = prompt
        self.on_drop = on_drop
        self.path: str | None = None
        self.hovering: bool = False
        self.drops: int = 0
        self._line = _Text()
        self._sub = _Text()

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.DROPBEGIN:
            self.hovering = True
            return False
        if event.type == pygame.DROPCOMPLETE:
            self.hovering = False
            return False
        if event.type == pygame.DROPFILE:
            self.path = event.file
            self.drops += 1
            if self.on_drop is not None:
                self.on_drop(self.path)
            return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, DROP_HOVER if self.hovering else FIELD, self.rect)
        pygame.draw.rect(surface, FOCUS if self.hovering else BORDER, self.rect, width=2)
        inner_width = max(self.rect.width - 2 * PAD, 0)
        if self.path is None:
            line = self._line.surface(_clip(self.prompt, inner_width), MUTED)
            surface.blit(line, (self.rect.centerx - line.get_width() // 2, self.rect.centery - line.get_height() // 2))
        else:
            line = self._line.surface(_clip(self.path, inner_width), INK)
            sub = self._sub.surface("drop another to replace it", MUTED)
            surface.blit(line, (self.rect.centerx - line.get_width() // 2, self.rect.centery - line.get_height()))
            surface.blit(sub, (self.rect.centerx - sub.get_width() // 2, self.rect.centery + 2))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class Panel(Widget):
    """One column of widgets, top to bottom, with tab order and nothing more.

    This is the whole of the layout: each child's ``x`` and ``width`` are the
    panel's inner width, each child's ``y`` is the running total, and each
    child keeps the ``height`` it was built with. It is computed when the
    panel is built and when :meth:`resize` is called — never per frame, and
    there are no rows, grids, anchors or weights. That is the
    ``DOCS/PLAN3.md`` § Risks valve, honoured rather than argued with.

    Event routing:

    * ``Tab`` / ``Shift+Tab`` moves focus through the focusable children in
      order (the one focus chain the contract allows).
    * An **open** :class:`Dropdown` sees every click and key first and
      swallows it (:meth:`Dropdown.captures`), so a button under the list
      cannot fire through it.
    * Keyboard and text events go to the focused child only; mouse and drop
      events go to every child, each of which hit-tests its own rect.

    Args:
        rect: the panel's own area.
        widgets: the children, top to bottom.
        gap: vertical space between children.
        padding: inset from the panel's edge.
    """

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        widgets: Sequence[Widget],
        *,
        gap: int = PAD,
        padding: int = PAD,
    ) -> None:
        # children first: Widget.__init__ assigns ``focused``, whose setter
        # here walks the children.
        self.widgets: tuple[Widget, ...] = tuple(widgets)
        super().__init__(rect)
        self.gap: int = gap
        self.padding: int = padding
        self.resize(self.rect)

    def resize(self, rect: pygame.Rect | tuple[int, int, int, int]) -> None:
        """Move the panel and re-stack its children. Heights are kept."""
        self.rect = pygame.Rect(rect)
        x = self.rect.x + self.padding
        width = max(self.rect.width - 2 * self.padding, 0)
        y = self.rect.y + self.padding
        for widget in self.widgets:
            widget.rect.update(x, y, width, widget.rect.height)
            y += widget.rect.height + self.gap

    @property
    def content_height(self) -> int:
        """The height the column actually needs, padding included."""
        total = sum(w.rect.height for w in self.widgets) + self.gap * max(len(self.widgets) - 1, 0)
        return total + 2 * self.padding

    # -- focus -------------------------------------------------------------------

    @property
    def focusable_widgets(self) -> tuple[Widget, ...]:
        return tuple(w for w in self.widgets if w.focusable)

    @property
    def focused(self) -> Widget | None:  # type: ignore[override]
        for widget in self.widgets:
            if widget.focused:
                return widget
        return None

    @focused.setter
    def focused(self, value: Widget | bool | None) -> None:
        # Widget.__init__ assigns ``focused = False``; a panel is never
        # itself focused, so a bool clears the children.
        for widget in self.widgets:
            widget.focused = widget is value

    def focus(self, widget: Widget | None) -> None:
        """Give ``widget`` the keyboard, and take it from everyone else."""
        self.focused = widget

    def _tab(self, backwards: bool) -> None:
        order = self.focusable_widgets
        if not order:
            return
        current = self.focused
        if current in order:
            step = -1 if backwards else 1
            target = order[(order.index(current) + step) % len(order)]
        else:
            target = order[-1] if backwards else order[0]
        self.focus(target)

    # -- events --------------------------------------------------------------------

    def _open_dropdown(self) -> Dropdown | None:
        for widget in self.widgets:
            if isinstance(widget, Dropdown) and widget.open:
                return widget
        return None

    def handle(self, events: Iterable[pygame.event.Event]) -> bool:
        changed = False
        for event in events:
            changed = self.handle_event(event) or changed
        return changed

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self._tab(bool(event.mod & pygame.KMOD_SHIFT))
            return False
        capturing = self._open_dropdown()
        if capturing is not None and capturing.captures(event):
            return capturing.handle_event(event)
        if event.type in (pygame.KEYDOWN, pygame.KEYUP, pygame.TEXTINPUT):
            target = self.focused
            return target.handle_event(event) if target is not None else False
        changed = False
        for widget in self.widgets:
            changed = widget.handle_event(event) or changed
        return changed

    # -- drawing -----------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, PAPER, self.rect)
        for widget in self.widgets:
            if isinstance(widget, Dropdown):
                widget.draw(surface, overlay=False)
            else:
                widget.draw(surface)
        for widget in self.widgets:
            if isinstance(widget, Dropdown):
                widget.draw_overlay(surface)
