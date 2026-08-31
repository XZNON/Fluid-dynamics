# HOWTO — what this thing does, and how to make it do it

A plain-language guide to running the simulator. Not a spec. For the specs see
`DOCS/IDEA2.md` (Phase 0, the solver) and `DOCS/IDEA3.md` (Phase 1, the product layer).

---

## 1. What it does, in one paragraph

You hand it **a picture of a shape**, **a fluid**, **a speed**, and **a real-world size**.
It works out every simulation setting by itself, runs the flow, and gives you a moving
picture of the swirls plus the numbers a fluid dynamicist would ask for — the drag on your
object, and how fast the wake sheds vortices behind it.

Four inputs. No CFD knowledge needed. It refuses badly-posed setups instead of quietly
handing you nonsense, and when it refuses it names the fix.

**How right is it?** Nine independent checks compare its answers against numbers published
in textbooks and papers. All nine pass. Its headline case — a disc in water — lands drag at
1.4040 where the published band is 1.25–1.45, in about 50 seconds from a cold shell.

---

## 2. The one command

```bash
myenv/Scripts/python.exe -m flow --shape PICTURE --fluid NAME --speed "Q" --size "Q" [options]
```

All four are required. Everything else has a sensible default.

Real example, the one that is known to work:

```bash
myenv/Scripts/python.exe -m flow \
  --shape tests/data/shapes/disc.png \
  --fluid water --speed "5 mm/s" --size "2 cm" \
  --out wake.mp4
```

**Two practical notes before you run anything:**

- Add `--backend warp` to run on the graphics card. Much faster. Leave it off to use the
  slower reference path on the CPU.
- Have the laptop **on mains power**. On battery the processor throttles and everything runs
  roughly 40% slower — this has already caused one false failure.

---

## 3. The four inputs

### `--shape` — a picture

A **PNG** or an **SVG**. Dark shape on a light background. If the PNG has transparency, the
opaque region is the shape and the colour does not matter — a cut-out wing works as-is.

Already in the repo, ready to use:

| File | What it is |
|---|---|
| `tests/data/shapes/disc.png` | a circle — the classic cylinder case |
| `tests/data/shapes/square.png` | a square |
| `tests/data/shapes/donut.png` | a ring with a hole |
| `tests/data/shapes/two_bodies.png` | two separate objects |
| `tests/data/shapes/specks.png` | a body plus stray dots (deliberately messy) |
| `tests/data/shapes/hairline_appendage.png` | a body with a hair-thin spike (deliberately messy) |
| `tests/data/shapes/tiny_body.png` | a body too small to resolve (gets refused) |
| `tests/data/shapes/unclosed_outline.png` | an outline that never closes |
| `antialiased.png`, `diagonal_line.png`, `extreme_aspect.png`, `huge_margin.png`, `self_touching.png`, `all_black.png`, `all_white.png` | more edge cases, same folder |

### `--fluid` — one of six

`water`, `air`, `olive oil`, `helium`, `glycerine`, `honey`. All at 20 °C, each with a cited
source in the code. Spelling is forgiving: `H2O`, `He`, `glycerol`, `glycerin`, `olive_oil`,
`Olive Oil` all resolve.

Thickness is what decides whether a case can run at all — but the thickness that matters is
**kinematic** viscosity, `nu = mu / rho`, not the everyday sense of the word. Thin to thick:

| fluid | nu (m^2/s) |
|---|---|
| water | 1.004e-06 |
| air | 1.516e-05 |
| olive oil | 8.400e-05 |
| helium | 1.178e-04 |
| glycerine | 1.120e-03 |
| honey | 7.042e-03 |

**Water is the thinnest thing here, 15x thinner than air.** It is 55x more viscous than air by
`mu`, but 830x denser, and density wins. So swapping air for water at the same speed and size
multiplies the Reynolds number by 15 and will usually get you refused. Helium is thicker than
air for the same reason in reverse: it is a gas, but a very light one.

If a case is refused for being too energetic, moving *down* that list is one of the three
fixes. Moving from air to water is moving **up** it.

You can also pass a viscosity directly instead of a name, e.g. `--fluid "1.5e-5 m^2/s"`.

### `--speed` — with a unit

`"5 mm/s"`, `"2 m/s"`, `"20 km/h"`. The unit is required; bare numbers are not guessed at.

### `--size` — with a unit

How big the object is **across the flow**. `"2 cm"`, `"10 cm"`, `"1.5 m"`.

---

## 4. What comes out

### A moving picture

The colours are **vorticity** — how hard the fluid is spinning at each point, red one way,
blue the other. Deliberately not speed: speed magnitude renders as a grey smear where
vorticity shows the vortex street crisply.

Three ways to get it:

```bash
--out wake.mp4          # save a video (.mp4 or .gif)
--live                  # open a window and watch it happen
--frames-dir frames/    # a folder of numbered PNGs
```

A window opens automatically if you ask for no output at all. `--out` plus `--live` does
both. `--no-live` with no output file runs the numbers and draws nothing — fastest, for
when you only want the figures.

`.mp4` needs **ffmpeg** installed. `.gif` and the frames folder do not.

### The numbers

Printed at the end of every run:

- **Cd** — drag coefficient, with its wobble and its spread
- **Cl** — lift coefficient, mean and amplitude
- **Strouhal number** — how fast the wake sheds vortices, with a confidence figure
- **convergence** — whether the flow settled down
- **stable** — whether the simulation stayed healthy start to finish

Add `--quiet` for numbers only, no prose.

---

## 5. Features worth trying

### Ask before you run — free, instant

```bash
myenv/Scripts/python.exe -m flow --shape tests/data/shapes/square.png --fluid air --speed "20 km/h" --size "10 cm" --explain
```

Prints the entire plan — what it worked out, what it will do, how long it expects to take —
and stops. Runs zero simulation. **This is the cheapest way to explore the tool.** Use it
constantly.

### Trade accuracy against time

```bash
--quality fast        # coarse, quick
--quality balanced    # the default
--quality accurate    # fine, slow
```

One knob, spelled in words rather than grid cells on purpose.

### Run longer or shorter

```bash
--seconds "2 s"
```

Default is however long the plan judges necessary to reach a settled flow.

### Let it fix a messy picture

By default it repairs what it can and **prints every repair it made** — filling holes,
dropping stray specks, keeping the largest body, thickening parts too thin to simulate.

To see the refusals instead of the repairs:

```bash
--no-repair
```

### Let it pick the nearest workable case

```bash
--nearest
```

If your setup is impossible, this runs the closest one that works instead — and every
artifact it produces says so, including the video's own metadata. It is honest about the
substitution rather than pretending you got what you asked for.

### Watch it refuse something on purpose

```bash
myenv/Scripts/python.exe -m flow --shape tests/data/shapes/disc.png --fluid air --speed "20 m/s" --size "1.5 m" --explain
```

That is a Reynolds number of about 2 million — a car on a motorway. This solver has no
turbulence model and tops out in the low thousands, so it **refuses**, and tells you to slow
it down, shrink it, or change the fluid. That refusal is correct behaviour and it stays.

### Compare fluids

Same shape, same speed, same size — swap `--fluid water` for `--fluid glycerine`. Thick
syrup should smother the vortices; water should shed them. Visible with your eyes, no
analysis needed.

### Compare speeds

Same everything, change `--speed`. Slow gives smooth flow hugging the shape. Fast gives a
messy shedding wake. The transition between them is the interesting bit.

---

## 6. Running the built-in validation

These are the nine checks that prove it is right. Each prints pass or fail.

```bash
myenv/Scripts/python.exe -m validate.poiseuille            # flow in a pipe
myenv/Scripts/python.exe -m validate.cavity --re 100       # a stirred box vs published data
myenv/Scripts/python.exe -m validate.cylinder              # vortex street behind a circle
myenv/Scripts/python.exe -m validate.polygons              # square and polygon bodies
myenv/Scripts/python.exe -m validate.parity --backend warp # CPU and GPU agree
myenv/Scripts/python.exe -m validate.autoconfig            # the auto-settings are sane (~23 min)
myenv/Scripts/python.exe -m validate.shapes                # picture handling (~10 s)
myenv/Scripts/python.exe -m validate.refusals              # every refusal names a fix that works
myenv/Scripts/python.exe -m validate.minute --backend warp # the whole product path, timed (~50 s)
```

The last one is the headline claim: a picture and three physical numbers reach published
accuracy in under a minute from a cold start.

Unit tests: `myenv/Scripts/python.exe -m pytest` — currently 772 pass, 1 skipped.

---

## 7. Using it from Python instead of the command line

```python
from flow import Case

case = Case.from_image(
    "tests/data/shapes/disc.png",
    fluid="water", speed="5 mm/s", size="2 cm",
)

print(case.explain())        # the plan, as text — runs nothing
result = case.run()          # actually simulate

print(result.cd, result.strouhal)
result.save("wake.mp4")      # or "wake.gif", or "frames/"
result.plot("history.png")   # drag and lift over time, as a graph
```

`Case.from_array(mask, ...)` takes a boolean array directly if you built the shape in code.

---

# Can I get any diagram or shape?

**Yes — three routes, in order of how much freedom they give you.**

## Route 1 — draw it and export a PNG (easiest, works for anything)

Any image editor. Paint, Figma, Inkscape, Illustrator, a phone drawing app, a photo of an
ink drawing. Rules:

1. **Solid dark shape on a light background.** Black on white is ideal.
2. **Or** a transparent PNG — then the opaque region is the shape and colour is ignored.
   This is the best route for a cut-out wing or a logo.
3. **Close your outlines and fill them.** A hollow outline is a thin ring of solid with
   fluid inside it, not a solid body. `unclosed_outline.png` exists to show what that does.
4. **Keep it chunky.** Anything thinner than about 3 cells at simulation resolution leaks
   fluid through it. The tool will thicken it and tell you, or refuse it.
5. **No signature marks or stray dots.** They get dropped as specks — again, it tells you.

Then just point at it:

```bash
myenv/Scripts/python.exe -m flow --shape my_wing.png --fluid air --speed "5 m/s" --size "10 cm" --explain
```

Start with `--explain` so a bad picture costs you nothing.

## Route 2 — an SVG (vector, scales cleanly)

```bash
myenv/Scripts/python.exe -m flow --shape my_shape.svg --fluid water --speed "5 mm/s" --size "2 cm"
```

**Limits, and they are enforced loudly rather than silently:** the built-in reader handles
simple closed paths — straight lines and Bézier curves, absolute or relative. It does
**not** handle arcs, `transform` attributes, strokes, text, or embedded images. If your file
uses one of those it stops with a clear message naming the feature. It will never silently
drop a `transform` and give you a plausible mask of the wrong shape.

Multiple sub-paths combine with the even-odd rule, so a donut drawn as two rings correctly
gets a hole.

**If your SVG is refused, export it to PNG.** That is the supported path and it always works.

## Route 3 — build it in code (exact and repeatable)

Four primitives, on a grid of your choosing:

```python
import numpy as np
from lbm.geometry import circle, rectangle, polygon, regular_polygon

ny, nx = 200, 600

disc    = circle(ny, nx, cx=150, cy=100, radius=20)
box     = rectangle(ny, nx, x0=140, y0=80, x1=180, y1=120)
hexagon = regular_polygon(ny, nx, nsides=6, cx=150, cy=100, radius=25)
wedge   = polygon(ny, nx, vertices=[(140, 100), (200, 92), (200, 108)])
```

Combine them with `|` (union) and `& ~` (subtract):

```python
solid = disc | box                       # both bodies
solid = circle(...) & ~circle(...)       # a ring
```

Then feed the array straight in:

```python
from flow import Case
case = Case.from_array(solid, fluid="water", speed="5 mm/s", size="2 cm")
print(case.explain())
```

Look at `tests/data/shapes/generate.py` for worked examples — that is the script that
produced every test picture in the repo.

## Checking your shape before you commit to a run

```bash
myenv/Scripts/python.exe -m flow --shape my_shape.png --fluid water --speed "5 mm/s" --size "2 cm" --explain
```

Costs nothing, runs nothing, and tells you exactly what it thinks of your picture: what it
would repair, what it would refuse, and why. Iterate here, not in the simulator.

---

## What it deliberately cannot do yet

Named so you don't go looking:

- **No 3D.** Two dimensions only. No STL files.
- **No graphical interface.** Command line and Python.
- **Not installable with `pip`.** Run it from this folder.
- **No sweeping many speeds at once.** One case per run — no automatic drag polars.
- **No turbulence model**, which is what caps the Reynolds number in the low thousands.

Which of these gets built next is the Phase 2 decision, still open.
