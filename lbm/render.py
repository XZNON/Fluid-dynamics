"""Field -> RGB, and the live pygame window that shows it.

Implements ``DOCS/IDEA2.md`` § "What to actually draw" and the *live* third of
§ "Three output sinks, same frame source".

Two things live here and they are deliberately separate:

* :func:`render` is **the** renderer (``CLAUDE.md`` constraint 10). It turns a
  scalar field into ``uint8`` RGB with a **diverging** colormap and **fixed,
  symmetric** limits the caller supplies. Nothing here autoscales — per-frame
  limits are what make a vorticity animation flicker (constraint 9), so the
  limits are an argument and a non-symmetric pair is refused.
* :class:`LiveSink` *consumes* that output and blits it. It colours nothing and
  computes nothing; give it a different colormap and there is nowhere to put it.

Why vorticity and not speed
---------------------------

``DOCS/IDEA2.md`` § What to actually draw: "Speed magnitude looks like a grey
smear. Vorticity with a diverging colormap (blue/white/red, symmetric limits)
makes the vortex street pop immediately." The field itself is computed by
:func:`lbm.probe.vorticity` (via :meth:`lbm.runner.Sim.vorticity`), which puts
``nan`` on solid cells; :func:`render` paints those a fixed neutral grey rather
than letting them fold into the colour scale.

Threading
---------

``old-Docs/STATE1.md`` **D-024**: ``run(sim, sink, drop=True)`` drains the ring
buffer from one consumer thread, so :meth:`LiveSink.push` runs on *that* thread
and **no pygame call sits inside the physics loop** (constraint 8). The window
is therefore opened lazily, on the first :meth:`LiveSink.push`, which keeps
every SDL call on a single thread — the consumer's — for the life of the run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lbm.runner import Sink

__all__ = [
    "COOLWARM",
    "NAN_RGB",
    "colormap",
    "render",
    "LiveSink",
]


# ---------------------------------------------------------------------------
# The colormap
# ---------------------------------------------------------------------------

#: Anchor colours of the diverging map, ``(position, r, g, b)`` with position in
#: ``[0, 1]``. Moreland's "cool-warm": saturated blue at the low end, a light
#: neutral at the midpoint, saturated red at the high end. The midpoint is what
#: makes it diverging — with symmetric limits it lands exactly on zero
#: vorticity, so the sign of a vortex is readable at a glance
#: (``DOCS/IDEA2.md`` § What to actually draw).
_ANCHORS: tuple[tuple[float, int, int, int], ...] = (
    (0.0, 59, 76, 192),
    (0.25, 122, 149, 231),
    (0.5, 221, 221, 221),
    (0.75, 229, 130, 104),
    (1.0, 180, 4, 38),
)

#: Colour painted on cells whose value is ``nan`` — the solid cells of
#: :func:`lbm.probe.vorticity`. A neutral grey that is not any colour the map
#: produces, so the body reads as a hole in the field rather than as a value.
NAN_RGB: tuple[int, int, int] = (110, 110, 110)


def colormap(n: int = 256) -> NDArray[np.uint8]:
    """The diverging lookup table, shape ``(n, 3)``, ``uint8``.

    Built once at import into :data:`COOLWARM` and indexed per frame; the
    interpolation below never runs in the render path.

    Args:
        n: number of entries.

    Returns:
        ``(n, 3)`` ``uint8``, entry 0 the low end and entry ``n-1`` the high
        end, with the neutral midpoint at ``(n-1)/2``.
    """
    if n < 2:
        raise ValueError(f"colormap needs at least 2 entries (got {n!r}).")
    pos = np.array([a[0] for a in _ANCHORS], dtype=np.float64)
    cols = np.array([a[1:] for a in _ANCHORS], dtype=np.float64)
    x = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.round(np.interp(x, pos, cols[:, c])).astype(np.uint8)
    return lut


#: The lookup table :func:`render` indexes. One definition, one map.
#:
#: **257 entries, not 256**, so the count is odd and the neutral midpoint sits on
#: exactly one index (128). With an even count, zero falls between two entries
#: and the map is very slightly asymmetric — a clockwise and an anticlockwise
#: vortex of the same strength would not be mirror images, which is the one
#: property a diverging map exists to have.
COOLWARM: NDArray[np.uint8] = colormap(257)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def _limits(limits: float | tuple[float, float]) -> tuple[float, float]:
    """Normalise the ``limits`` argument to ``(vmin, vmax)``, symmetric.

    ``CLAUDE.md`` constraint 9 asks for symmetric fixed limits, so a scalar is
    the intended form (``limits=0.05`` means ``-0.05 .. +0.05``) and an
    asymmetric pair is an error rather than a silently skewed colour scale — on
    a diverging map that would move the neutral colour off zero and make a
    clockwise vortex look weaker than an anticlockwise one.
    """
    if np.isscalar(limits):
        vmax = float(limits)  # type: ignore[arg-type]
        vmin = -vmax
    else:
        vmin, vmax = (float(v) for v in limits)  # type: ignore[misc]

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"limits must be finite (got {(vmin, vmax)!r}).")
    if vmax <= vmin:
        raise ValueError(
            f"limits must be increasing (got vmin={vmin!r}, vmax={vmax!r})."
        )
    if abs(vmin + vmax) > 1e-12 * max(abs(vmin), abs(vmax), 1.0):
        raise ValueError(
            f"limits must be symmetric about zero for a diverging colormap "
            f"(got vmin={vmin!r}, vmax={vmax!r}); pass a single number, "
            f"render(field, 0.05), which means -0.05 .. +0.05 "
            f"(CLAUDE.md constraint 9)."
        )
    return vmin, vmax


def render(
    field: NDArray[np.float32],
    limits: float | tuple[float, float],
    *,
    lut: NDArray[np.uint8] = COOLWARM,
    nan_rgb: tuple[int, int, int] = NAN_RGB,
    out: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """Map a scalar field to RGB with a diverging colormap and fixed limits.

    ``DOCS/IDEA2.md`` § What to actually draw: "Clip limits to a fixed range or
    the colours will flicker frame to frame." That is ``CLAUDE.md`` constraint 9
    and it is the reason ``limits`` is a **parameter**: this function never
    looks at ``field.min()`` or ``field.max()``, so the same value maps to the
    same bytes in every frame of a run. ``tests/test_render.py`` asserts exactly
    that across two frames of different data.

    Constraint 10 — there is one renderer and this is it. :class:`LiveSink` and
    the T011 recording sinks all consume this output; none of them colours
    anything itself.

    The mapping, in full::

        t   = clip((value - vmin) / (vmax - vmin), 0, 1)
        i   = round(t * (len(lut) - 1))
        rgb = lut[i]

    Values outside the limits clamp to the end colours; ``nan`` — which
    :func:`lbm.probe.vorticity` writes on solid cells — is painted ``nan_rgb``
    instead of participating in the scale.

    Args:
        field: scalar field, shape ``(ny, nx)``. Usually vorticity.
        limits: a single positive number ``v`` meaning ``-v .. +v``, or an
            explicitly symmetric ``(vmin, vmax)`` pair.
        lut: the colormap, shape ``(n, 3)``, ``uint8``. Defaults to
            :data:`COOLWARM`.
        nan_rgb: colour for ``nan`` cells (the solid body).
        out: optional preallocated ``(ny, nx, 3)`` ``uint8`` destination. One
            frame is many timesteps, so this is a per-frame convenience, not a
            step-loop requirement.

    Returns:
        ``(ny, nx, 3)`` ``uint8``, row 0 at ``y = 0``.

    Raises:
        ValueError: on a non-2D field, non-symmetric or degenerate limits, or an
            ``out`` of the wrong shape or dtype.
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"field must be 2D (ny, nx) (got shape {field.shape}).")

    lut = np.asarray(lut, dtype=np.uint8)
    if lut.ndim != 2 or lut.shape[1] != 3 or lut.shape[0] < 2:
        raise ValueError(f"lut must be (n, 3) uint8 with n >= 2 (got {lut.shape}).")

    vmin, vmax = _limits(limits)
    ny, nx = field.shape

    if out is None:
        out = np.empty((ny, nx, 3), dtype=np.uint8)
    elif out.shape != (ny, nx, 3) or out.dtype != np.uint8:
        raise ValueError(
            f"out must be {(ny, nx, 3)} uint8 (got {out.shape} {out.dtype})."
        )

    n = lut.shape[0]
    scale = (n - 1) / (vmax - vmin)

    finite = np.isfinite(field)
    # nan/inf would produce an undefined uint8 cast, so they are replaced before
    # the cast and overwritten afterwards. +-inf clamp to the end colours, which
    # is what a value past the limits does anyway.
    safe = np.where(finite, field, vmin)
    t = (safe.astype(np.float32) - np.float32(vmin)) * np.float32(scale)
    idx = np.clip(np.rint(t), 0, n - 1).astype(np.uint8 if n <= 256 else np.int32)

    np.take(lut, idx, axis=0, out=out)
    if not finite.all():
        out[~finite] = np.asarray(nan_rgb, dtype=np.uint8)
    return out


# ---------------------------------------------------------------------------
# The live sink
# ---------------------------------------------------------------------------


class LiveSink(Sink):
    """A pygame window fed by the ring buffer's consumer thread.

    The *live* sink of ``DOCS/IDEA2.md`` § Three output sinks, same frame
    source: "pygame surface... Interactive, drop frames if behind." Dropping is
    not this class's job — :class:`lbm.runner.RingBuffer` already does it, and
    :func:`lbm.runner.run` with ``drop=True`` drives this sink from a consumer
    thread (D-024). What this class guarantees is the other half of
    constraint 8: **no pygame call happens on the physics thread**. Every SDL
    call below — ``init``, ``set_mode``, ``blit``, ``flip``, ``quit`` — happens
    inside :meth:`push` or :meth:`close`, both of which the consumer calls.

    The window opens lazily on the first frame, for the same reason: SDL wants
    its display owned by one thread, and the first ``push`` is the first moment
    that thread exists and the frame shape is known.

    ``pygame`` is imported inside the methods, not at module scope, so
    ``import lbm.render`` stays cheap and headless (``validate/cylinder.py
    --headless`` never touches it).

    Attributes:
        scale: integer upscale factor applied to the frame.
        title: window caption.
        frames: frames blitted.
        quit_requested: ``True`` once the window's close button was pressed.
            :func:`lbm.runner.run` takes a ``stop=`` predicate; a caller that
            wants the run to end with the window checks this.
        closed: whether :meth:`close` has run.
    """

    def __init__(
        self,
        *,
        scale: int = 1,
        title: str = "lbm — vorticity",
        flip_y: bool = True,
    ) -> None:
        """Configure the window. Nothing is opened until the first frame.

        Args:
            scale: integer magnification. ``2`` shows a 200x500 grid at
                400x1000 pixels.
            title: window caption.
            flip_y: draw row 0 at the *bottom*. The solver's ``y`` increases
                upward and a screen's increases downward, so the default keeps
                the picture the right way up. Purely a display concern — the
                frame's bytes are untouched (constraint 10).
        """
        if scale < 1:
            raise ValueError(f"scale must be at least 1 (got {scale!r}).")
        self.scale = int(scale)
        self.title = title
        self.flip_y = bool(flip_y)
        self.frames = 0
        self.quit_requested = False
        self.closed = False
        self._screen: Any = None
        self._surface: Any = None
        self._pygame: Any = None

    # -- window lifecycle (consumer thread only) --------------------------

    def _open(self, shape: tuple[int, int]) -> None:
        """Open the window, sized from the first frame."""
        import pygame

        ny, nx = shape
        pygame.display.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((nx * self.scale, ny * self.scale))
        pygame.display.set_caption(self.title)
        # A per-frame scratch surface of the frame's own size; the blit scales
        # it up. Allocated once, like everything else that repeats.
        self._surface = pygame.Surface((nx, ny))

    def push(self, frame: NDArray[np.uint8]) -> None:
        """Blit one rendered frame.

        Args:
            frame: ``(ny, nx, 3)`` ``uint8`` from :func:`render`. This sink does
                not colour, scale values, or read the physics — it draws bytes.
        """
        if self.closed:
            return
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(
                f"LiveSink wants (ny, nx, 3) uint8 from render() "
                f"(got {frame.shape} {frame.dtype})."
            )
        if self._screen is None:
            self._open(frame.shape[:2])

        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE,
                pygame.K_q,
            ):
                self.quit_requested = True

        if self.flip_y:
            frame = frame[::-1]
        # pygame surfaces are indexed (x, y); the frame is (y, x, 3).
        pygame.surfarray.blit_array(self._surface, np.transpose(frame, (1, 0, 2)))
        if self.scale == 1:
            self._screen.blit(self._surface, (0, 0))
        else:
            pygame.transform.scale(self._surface, self._screen.get_size(), self._screen)
        pygame.display.flip()
        self.frames += 1

    def close(self) -> None:
        """Shut the window down. Idempotent."""
        if self.closed:
            return
        self.closed = True
        if self._pygame is not None:
            try:
                self._pygame.display.quit()
            finally:
                self._screen = None
                self._surface = None
