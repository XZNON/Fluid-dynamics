"""The recording sinks: MP4 / GIF, numbered PNGs, and a fan-out.

Implements the *record* and *headless* thirds of ``DOCS/IDEA2.md`` § "Three
output sinks, same frame source":

    - **Live** — pygame surface... Interactive, drop frames if behind.
    - **Record** — ``imageio`` / ``ffmpeg`` writer, fixed framerate, never drop frames.
    - **Headless** — no display, write PNGs or MP4. For long runs on a server.

    Same ``render()`` output feeds all three. Do not write three renderers.

That last line is ``CLAUDE.md`` constraint 10, and it is why there is no colour,
no colormap, no scaling and no physics in this module: every class here takes
``(ny, nx, 3)`` ``uint8`` — exactly what :func:`lbm.render.render` returns — and
puts those bytes somewhere. ``tests/test_record.py`` asserts that
:class:`RecordSink`, :class:`HeadlessSink` and :class:`lbm.render.LiveSink`
receive **byte-identical** frames for the same sim state, which is the
enforceable form of "do not write three renderers".

Dropping, and who is allowed to
-------------------------------

``CLAUDE.md`` constraint 8 says the ring buffer drops *display* frames and never
simulation steps. A recorder is the other case: a video at a fixed framerate
with a frame missing is not a slower video, it is a **wrong** one — the wake
skips — and nothing in the file says so. **D-024** settles it:

* ``run(sim, sink, drop=True)`` drains from a consumer thread and drops the
  oldest display frames when the buffer fills. That is the live mode.
* ``run(sim, sink, drop=False)`` drains inline: the sink sees every frame in
  order and *the sim waits for it*, which is explicitly allowed for "a
  fixed-framerate recorder (T011)".

:class:`RecordSink` therefore does not implement any dropping policy of its
own — it writes every frame it is given, synchronously, and counts them. Handing
one to ``run(..., drop=True)`` is a caller error, so :meth:`RecordSink.push`
records the frames it received and :attr:`RecordSink.frames` is what
``tests/test_record.py`` compares against the frame count of the file on disk.

ffmpeg
------

MP4 needs an ffmpeg binary. ``imageio[ffmpeg]`` ships one (``imageio-ffmpeg``),
so the fix is a one-line install and :func:`check_ffmpeg` is what turns a
missing binary into that line instead of a traceback out of a subprocess pipe.
GIF and PNG need no binary at all — Pillow writes both — so a machine without
ffmpeg can still record.

Pixel dimensions
----------------

H.264 wants frame dimensions that are a multiple of the macro block size, and
``yuv420p`` needs them even; imageio resizes the *video* up to the next multiple
of 16 when they are not (a 61x97 frame is stored as a 64x112 video). The bytes
handed to :meth:`RecordSink.push` are untouched by that — the resize happens
inside the writer — so constraint 10 and the byte-identical test are unaffected.
``macro_block_size=1`` is available for an exact-size video and is not the
default: with odd dimensions libx264 refuses the stream outright.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lbm.runner import Sink

__all__ = [
    "FFMPEG_HINT",
    "IMAGEIO_HINT",
    "VIDEO_SUFFIXES",
    "RecordSink",
    "HeadlessSink",
    "TeeSink",
    "check_ffmpeg",
    "frame_count",
]

#: Install line for the whole recording stack. Quoted verbatim by every error
#: this module raises, because "install imageio" without the extra is the
#: mistake that leaves MP4 broken and GIF working.
IMAGEIO_HINT: str = 'myenv/Scripts/pip.exe install "imageio[ffmpeg]"'

#: What a missing ffmpeg binary should say. ``DOCS/TASKS1.md`` § T011: "Missing
#: ffmpeg produces a clear install message, not a traceback."
FFMPEG_HINT: str = (
    "MP4 recording needs an ffmpeg binary and none was found.\n"
    "    Install one into the project venv with:\n"
    f"        {IMAGEIO_HINT}\n"
    "    and add a row to DOCS/STATE1.md § Environment in the same session.\n"
    "    (A system ffmpeg on PATH works too — set IMAGEIO_FFMPEG_EXE to it.)\n"
    "    Nothing else is blocked: --out with a .gif suffix and --headless PNGs "
    "both write through Pillow and need no binary."
)

#: Suffixes routed to the ffmpeg writer. Everything else imageio accepts as a
#: multi-image format (``.gif``) goes through Pillow.
VIDEO_SUFFIXES: frozenset[str] = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm"})


# ---------------------------------------------------------------------------
# imageio access
# ---------------------------------------------------------------------------


def _imageio() -> Any:
    """Import ``imageio.v2``, or raise with the install line.

    Imported lazily and never at module scope: ``import lbm`` must stay cheap
    and must not require a video stack — the validation ladder does not record
    anything.
    """
    try:
        import imageio.v2 as iio
    except ImportError as exc:  # pragma: no cover - imageio is in myenv
        raise ImportError(
            "Recording needs imageio. Install it into the project venv with:\n"
            f"    {IMAGEIO_HINT}\n"
            "and add a row to DOCS/STATE1.md § Environment in the same session."
        ) from exc
    return iio


def check_ffmpeg() -> str:
    """Path to the ffmpeg binary, or a :class:`RuntimeError` naming the fix.

    Called by :class:`RecordSink` **before the run starts** when the output is a
    video, so a five-minute simulation does not discover at frame 1 that it has
    nowhere to write. The message is :data:`FFMPEG_HINT`.

    Returns:
        Absolute path to the binary imageio would use.

    Raises:
        RuntimeError: if imageio-ffmpeg is absent or has no usable binary.
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(FFMPEG_HINT) from exc
    try:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # imageio-ffmpeg raises a bare RuntimeError here
        raise RuntimeError(FFMPEG_HINT) from exc


def frame_count(path: str | Path) -> int:
    """Number of frames in a written video or GIF, read back off the disk.

    The acceptance criterion for :class:`RecordSink` is "a test writes 50 frames
    and asserts the file has exactly 50", so the count has to come from the
    *file* and not from a counter the writer kept.

    Args:
        path: an MP4/MKV/AVI/MOV/WEBM or a GIF.

    Returns:
        Frames the file actually contains.
    """
    iio = _imageio()
    path = Path(path)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        reader = iio.get_reader(str(path))
        try:
            return int(reader.count_frames())
        finally:
            reader.close()
    return len(iio.mimread(str(path)))


def _check_frame(frame: Any, who: str) -> NDArray[np.uint8]:
    """Validate one frame as :func:`lbm.render.render` output."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"{who} wants (ny, nx, 3) uint8 from lbm.render.render() "
            f"(got {arr.shape} {arr.dtype}). This sink writes bytes; it does "
            f"not colour a field (CLAUDE.md constraint 10)."
        )
    return arr


# ---------------------------------------------------------------------------
# RecordSink
# ---------------------------------------------------------------------------


class RecordSink(Sink):
    """Write every frame to an MP4 or GIF at a fixed framerate.

    The *record* sink of ``DOCS/IDEA2.md`` § Three output sinks: "``imageio`` /
    ``ffmpeg`` writer, fixed framerate, never drop frames." Both halves are
    structural rather than aspirational:

    **Fixed framerate.** ``fps`` is set once, at construction, and written into
    the container. It is not derived from how fast frames arrive, so a run that
    stutters produces a video that plays at the requested rate — which is the
    whole reason ``steps_per_frame`` is computed from ``dt`` (constraint 7,
    **D-023**) instead of the wall clock being sampled.

    **Never drop.** :meth:`push` writes synchronously and returns only when the
    frame is in the writer. Drive it with ``run(..., drop=False)`` (**D-024**),
    which drains the ring buffer inline and lets the sim wait for the writer;
    with ``drop=True`` the ring buffer would discard frames before this class
    ever saw them. :attr:`frames` counts what arrived and
    :func:`frame_count` counts what landed — the acceptance test asserts they
    agree, and that both are 50.

    The writer opens **lazily on the first frame**, because the frame shape is
    not known until then, but :func:`check_ffmpeg` runs in ``__init__`` for a
    video target so an unusable configuration fails at setup rather than after
    the first minute of physics.

    Attributes:
        path: output file.
        fps: container framerate.
        frames: frames pushed (and therefore written).
        closed: whether :meth:`close` has run.
        shape: ``(ny, nx)`` of the first frame; every later frame must match.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float = 60.0,
        quality: float | None = 8.0,
        macro_block_size: int = 16,
        codec: str = "libx264",
        loop: int = 0,
        **writer_kwargs: Any,
    ) -> None:
        """Configure the writer. Nothing is opened until the first frame.

        Args:
            path: output file. The suffix picks the writer: anything in
                :data:`VIDEO_SUFFIXES` goes through ffmpeg, ``.gif`` through
                Pillow.
            fps: frames per second written into the file. Positive.
            quality: imageio's ffmpeg quality knob, 0..10 (higher is better).
                ``None`` leaves the plugin's default. Ignored for GIF.
            macro_block_size: H.264 macro block size — see the module
                docstring. Ignored for GIF.
            codec: video codec. ``libx264`` is what plays everywhere.
            loop: GIF loop count, ``0`` meaning forever. Ignored for video.
            **writer_kwargs: passed through to ``imageio.get_writer``.

        Raises:
            ValueError: on a non-positive ``fps`` or a suffix that is neither a
                known video container nor ``.gif``.
            RuntimeError: for a video target with no ffmpeg binary available —
                message :data:`FFMPEG_HINT`, raised **here**, at setup.
        """
        if fps <= 0.0:
            raise ValueError(f"fps must be positive (got {fps!r}).")

        self.path = Path(path)
        suffix = self.path.suffix.lower()
        self.is_video: bool = suffix in VIDEO_SUFFIXES
        if not self.is_video and suffix != ".gif":
            raise ValueError(
                f"unsupported output suffix {suffix!r} for {self.path}: use "
                f"one of {sorted(VIDEO_SUFFIXES)} for video or .gif. "
                f"Numbered PNGs are HeadlessSink's job."
            )

        self.fps = float(fps)
        self.quality = quality
        self.macro_block_size = int(macro_block_size)
        self.codec = codec
        self.loop = int(loop)
        self._extra = dict(writer_kwargs)

        self.frames: int = 0
        self.closed: bool = False
        self.shape: tuple[int, int] | None = None
        self._writer: Any = None

        # Fail at setup, not at frame 1 of a long run.
        if self.is_video:
            self.ffmpeg_exe: str | None = check_ffmpeg()
        else:
            self.ffmpeg_exe = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writer lifecycle -------------------------------------------------

    def _open(self) -> None:
        """Open the imageio writer with the per-format arguments."""
        iio = _imageio()
        if self.is_video:
            kwargs: dict[str, Any] = {
                "fps": self.fps,
                "codec": self.codec,
                "macro_block_size": self.macro_block_size,
            }
            if self.quality is not None:
                kwargs["quality"] = self.quality
        else:
            # Pillow's GIF writer takes milliseconds per frame, not a rate.
            kwargs = {"mode": "I", "duration": 1000.0 / self.fps, "loop": self.loop}
        kwargs.update(self._extra)
        self._writer = iio.get_writer(str(self.path), **kwargs)

    def push(self, frame: NDArray[np.uint8]) -> None:
        """Write one rendered frame. Synchronous, and never a no-op.

        Args:
            frame: ``(ny, nx, 3)`` ``uint8`` from :func:`lbm.render.render`.

        Raises:
            ValueError: on a frame that is not that, on a frame whose shape
                differs from the first one, or on a push after :meth:`close`.
                A silently ignored frame is a video with a missing frame, which
                is the one failure this sink exists to make impossible.
        """
        if self.closed:
            raise ValueError(
                f"RecordSink({self.path.name}) is closed and cannot take frame "
                f"{self.frames + 1}: a dropped frame in a fixed-framerate "
                f"recording is a wrong video, not a slower one "
                f"(CLAUDE.md constraint 8, D-024)."
            )
        arr = _check_frame(frame, "RecordSink")
        if self.shape is None:
            self.shape = (int(arr.shape[0]), int(arr.shape[1]))
            self._open()
        elif (arr.shape[0], arr.shape[1]) != self.shape:
            raise ValueError(
                f"frame {self.frames + 1} is {arr.shape[:2]} but the video was "
                f"opened at {self.shape}; a video has one frame size."
            )
        self._writer.append_data(arr)
        self.frames += 1

    def close(self) -> None:
        """Finalise the file. Idempotent.

        A video is not readable until its container is closed, so this is not
        optional bookkeeping — :func:`lbm.runner.run` closes the sink it owns,
        and a caller that builds its own sink should use it as a context
        manager (:class:`lbm.runner.Sink` implements the protocol).
        """
        if self.closed:
            return
        self.closed = True
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RecordSink({self.path.name!r}, fps={self.fps:g}, "
            f"frames={self.frames})"
        )


# ---------------------------------------------------------------------------
# HeadlessSink
# ---------------------------------------------------------------------------


class HeadlessSink(Sink):
    """Write numbered PNGs into a directory. No display, no ffmpeg.

    The *headless* sink of ``DOCS/IDEA2.md`` § Three output sinks: "no display,
    write PNGs or MP4. For long runs on a server." Nothing here imports pygame
    or opens a window, and nothing here needs a binary — Pillow writes the PNGs,
    and it is already a dependency of :func:`lbm.geometry.from_png`.

    Numbering is zero-padded and monotonic from :attr:`start`, so
    ``frame_00000.png … frame_00299.png`` sorts correctly in a shell and feeds
    ``ffmpeg -i frame_%05d.png`` unchanged. Like :class:`RecordSink` this sink
    never drops: one push, one file.

    Attributes:
        directory: where the files go; created if absent.
        prefix: filename stem before the number.
        digits: zero-padding width.
        frames: files written.
        paths: every path written, in order.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        prefix: str = "frame",
        digits: int = 5,
        suffix: str = ".png",
        start: int = 0,
    ) -> None:
        """Create the output directory and fix the naming scheme.

        Args:
            directory: destination directory, created with parents.
            prefix: filename stem.
            digits: zero-padding width; must be at least 1.
            suffix: file extension, ``.png`` by default. Anything Pillow can
                write a single image to.
            start: first frame number.
        """
        if digits < 1:
            raise ValueError(f"digits must be at least 1 (got {digits!r}).")
        self.directory = Path(directory)
        self.prefix = str(prefix)
        self.digits = int(digits)
        self.suffix = suffix if suffix.startswith(".") else f".{suffix}"
        self.start = int(start)
        self.frames: int = 0
        self.closed: bool = False
        self.paths: list[Path] = []
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, index: int) -> Path:
        """Filename for frame ``index`` (absolute frame number, not offset)."""
        return self.directory / f"{self.prefix}_{index:0{self.digits}d}{self.suffix}"

    def push(self, frame: NDArray[np.uint8]) -> None:
        """Write one rendered frame as the next numbered file.

        Args:
            frame: ``(ny, nx, 3)`` ``uint8`` from :func:`lbm.render.render`.
        """
        if self.closed:
            raise ValueError(
                f"HeadlessSink({self.directory}) is closed and cannot take "
                f"frame {self.frames + 1}."
            )
        arr = _check_frame(frame, "HeadlessSink")
        iio = _imageio()
        path = self.path_for(self.start + self.frames)
        iio.imwrite(str(path), arr)
        self.paths.append(path)
        self.frames += 1

    def close(self) -> None:
        """Nothing to finalise — each PNG is complete when written."""
        self.closed = True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"HeadlessSink({str(self.directory)!r}, frames={self.frames})"


# ---------------------------------------------------------------------------
# TeeSink
# ---------------------------------------------------------------------------


class TeeSink(Sink):
    """Hand the **same** frame object to several sinks, in order.

    What makes ``--live --record`` composable (``DOCS/TASKS1.md`` § T011). It is
    not a fourth sink in the ``DOCS/IDEA2.md`` sense and not a second frame
    source: it renders nothing, copies nothing, and passes each sink the very
    same array, which is the strongest available form of "same ``render()``
    output feeds all three" (constraint 10).

    It is also not a third *run mode* — **D-024** says there are two and no
    more. A tee is driven by whichever mode the strictest member needs, which
    means ``drop=False`` whenever a :class:`RecordSink` is in the list: the
    recording must not lose frames, and the window is allowed to make the sim
    wait for it (the live sink is ~2% of a step, measured in T007).

    Attributes:
        sinks: the sinks, in push order.
        frames: frames forwarded.
    """

    def __init__(self, *sinks: Sink) -> None:
        """Args: ``sinks`` — one or more sinks, pushed in the order given."""
        flat: list[Sink] = []
        for s in sinks:
            if isinstance(s, Iterable) and not isinstance(s, Sink):  # type: ignore[unreachable]
                flat.extend(s)  # type: ignore[arg-type]
            else:
                flat.append(s)
        if not flat:
            raise ValueError("TeeSink needs at least one sink.")
        self.sinks: Sequence[Sink] = tuple(flat)
        self.frames: int = 0
        self.closed: bool = False

    def push(self, frame: NDArray[np.uint8]) -> None:
        """Forward one frame to every sink, in order."""
        for sink in self.sinks:
            sink.push(frame)
        self.frames += 1

    def close(self) -> None:
        """Close every sink, even if one of them raises."""
        if self.closed:
            return
        self.closed = True
        first: BaseException | None = None
        for sink in self.sinks:
            try:
                sink.close()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                first = first if first is not None else exc
        if first is not None:
            raise first

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TeeSink({', '.join(type(s).__name__ for s in self.sinks)})"
