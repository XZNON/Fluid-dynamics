"""Generate the Rung C shape corpus — deliberately awful pictures, on purpose.

``DOCS/TASKS2.md`` § T107 Notes: *"Generate the corpus programmatically where
possible so the images are auditable and small, and commit the generator
alongside."* Every image in this directory comes from this file and nothing
else, so a reviewer can see what each defect **is** rather than squinting at a
PNG, and a corpus image can be regenerated bit-for-bit.

Run it from the project root:

    myenv/Scripts/python.exe -m tests.data.shapes.generate

The images are greyscale PNGs, **dark = solid**, which is the sense
``lbm.geometry.from_png`` uses for a picture with no alpha channel. They are
small on purpose: the biggest is 400x400 and the whole corpus is a few
kilobytes, because a committed binary is only auditable if it is also cheap.

The expected verdict for each one lives in ``expectations.json`` beside them,
written by ``validate/shapes.py --write-expectations``. **The verdict is the
designed known answer** (``DOCS/IDEA3.md`` § Validation ladder, Rung C row);
the measured properties beside it are pinned so that a change in *how* a shape
is repaired shows up as a diff rather than passing quietly.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

#: Where the corpus lives — this file's own directory.
HERE = Path(__file__).resolve().parent

SOLID = 0  # black
FLUID = 255  # white


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("L", (width, height), FLUID)
    return img, ImageDraw.Draw(img)


def disc() -> Image.Image:
    """A plain disc. The control — and D-017's false-alarm bar."""
    img, d = _canvas(200, 200)
    d.ellipse((30, 30, 169, 169), fill=SOLID)
    return img


def square() -> Image.Image:
    """A plain square. The second control."""
    img, d = _canvas(200, 200)
    d.rectangle((30, 30, 169, 169), fill=SOLID)
    return img


def hairline_appendage() -> Image.Image:
    """A thick body with a hairline **fused** to it — the D-017 limit itself.

    The whisker shares the disc's 8-connected component, so ``min_thickness``
    reports the disc's depth and says nothing about it. This is the image
    Q-102 exists for.
    """
    img, d = _canvas(260, 200)
    d.ellipse((10, 30, 149, 169), fill=SOLID)
    d.rectangle((140, 97, 249, 102), fill=SOLID)
    return img


def specks() -> Image.Image:
    """A body plus detached debris — the dirt a threshold leaves behind."""
    img, d = _canvas(200, 200)
    d.ellipse((40, 40, 159, 159), fill=SOLID)
    for x, y, r in ((12, 14, 3), (186, 20, 4), (24, 180, 3), (180, 186, 5)):
        d.ellipse((x - r, y - r, x + r, y + r), fill=SOLID)
    return img


def donut() -> Image.Image:
    """An annulus: a genuine interior hole the flow can never reach."""
    img, d = _canvas(200, 200)
    d.ellipse((20, 20, 179, 179), fill=SOLID)
    d.ellipse((70, 70, 129, 129), fill=FLUID)
    return img


def unclosed_outline() -> Image.Image:
    """A ring drawn as a stroke, with a gap — the classic traced outline.

    The gap means the interior is *not* enclosed, so there is no hole to fill:
    what is wrong is that the stroke itself is thinner than three cells.
    """
    img, d = _canvas(200, 200)
    d.ellipse((25, 25, 174, 174), outline=SOLID, width=6)
    d.pieslice((25, 25, 174, 174), start=-25, end=25, fill=FLUID)
    return img


def antialiased() -> Image.Image:
    """A disc with a heavily feathered edge — every exported PNG looks like this."""
    img, d = _canvas(200, 200)
    d.ellipse((35, 35, 164, 164), fill=SOLID)
    return img.filter(ImageFilter.GaussianBlur(radius=9))


def huge_margin() -> Image.Image:
    """A small body adrift in a large canvas — what **D-040** is about."""
    img, d = _canvas(400, 400)
    d.ellipse((175, 178, 224, 227), fill=SOLID)
    return img


def tiny_body() -> Image.Image:
    """A body a few pixels across. Refused when too few cells are asked for."""
    img, d = _canvas(400, 400)
    d.ellipse((196, 196, 205, 205), fill=SOLID)
    return img


def extreme_aspect() -> Image.Image:
    """A flat plate broadside to the flow: 160 across, 4 along."""
    img, d = _canvas(200, 200)
    d.rectangle((98, 20, 101, 179), fill=SOLID)
    return img


def diagonal_line() -> Image.Image:
    """A one-cell diagonal — 8-connected, and it leaks like a sieve."""
    img, d = _canvas(200, 200)
    d.line((20, 20, 179, 179), fill=SOLID, width=4)
    return img


def self_touching() -> Image.Image:
    """Two lobes joined by a waist thinner than the grid can hold.

    One 8-connected component, so ``min_thickness`` reports the lobes and says
    nothing about the join — the same blind spot as
    :func:`hairline_appendage`, reached from the other direction: here the thin
    part is in the *middle* of the shape rather than hanging off its end.
    """
    img, d = _canvas(280, 160)
    d.ellipse((10, 20, 129, 139), fill=SOLID)
    d.ellipse((150, 20, 269, 139), fill=SOLID)
    d.rectangle((126, 77, 153, 82), fill=SOLID)
    return img


def two_bodies() -> Image.Image:
    """A body plus a second, detached body far too big to be a speck.

    ``drop_specks`` will not touch it — that is the point. Something has to
    decide *which* of two plausible bodies the flow is about, and
    ``largest_component`` is the repair that does, loudly.
    """
    img, d = _canvas(280, 200)
    d.ellipse((10, 30, 149, 169), fill=SOLID)
    d.ellipse((190, 70, 269, 149), fill=SOLID)
    return img


def all_white() -> Image.Image:
    """No solid at all: whatever the resolution, there is no body."""
    img, _ = _canvas(120, 120)
    return img


def all_black() -> Image.Image:
    """Solid everywhere: whatever the resolution, there is no fluid."""
    img, d = _canvas(120, 120)
    d.rectangle((0, 0, 119, 119), fill=SOLID)
    return img


#: ``name -> (builder, cells_across)``. The body size is part of the case: a
#: tiny body is only a refusal because of the resolution it was asked for, and
#: pinning it here is what makes ``validate/shapes.py`` need no manual step.
CORPUS: dict[str, tuple] = {
    "disc": (disc, 40),
    "square": (square, 40),
    "hairline_appendage": (hairline_appendage, 40),
    "specks": (specks, 40),
    "donut": (donut, 40),
    "unclosed_outline": (unclosed_outline, 40),
    "antialiased": (antialiased, 40),
    "huge_margin": (huge_margin, 40),
    "tiny_body": (tiny_body, 6),
    "extreme_aspect": (extreme_aspect, 40),
    "diagonal_line": (diagonal_line, 40),
    "self_touching": (self_touching, 40),
    "two_bodies": (two_bodies, 40),
    "all_white": (all_white, 40),
    "all_black": (all_black, 40),
}


def write_corpus(into: Path = HERE) -> list[Path]:
    """Write every corpus image, returning the paths in :data:`CORPUS` order."""
    into.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (builder, _cells) in CORPUS.items():
        path = into / f"{name}.png"
        builder().save(path, optimize=True)
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover
    for path in write_corpus():
        print(f"{path.relative_to(Path.cwd()) if path.is_absolute() else path} "
              f"({path.stat().st_size} bytes)")
