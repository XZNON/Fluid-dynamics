# `examples/`

Ad-hoc geometry used while testing and demonstrating `flow`, plus anywhere else
a hand-made input belongs.

`shapes/` is **not** `tests/data/shapes/`, and the difference is load-bearing:
`validate/shapes.py` (Rung C) iterates *every* image in `tests/data/shapes/` and
`tests/test_prepare.py` cross-checks that corpus against its `generate.py`.
Adding a picture there changes what a validation rung measures. Adding one here
does not.

| File | What it is |
|---|---|
| `shapes/car.png` | a car silhouette — asymmetric about the horizontal axis, which is how issue `9e58e90c9b58` (mirrored output) was found |
| `shapes/test_shape.png` | first ad-hoc body; the `slowmo.py` example |
| `shapes/test2.png` | the body the `windtunnel.py` / `streamlines.py` examples use |
| `shapes/test3.png` | left-pointing wedge with a flat rear face; the repro for issue `5e94900a8170` (sharp corners diverge) |

White is fluid, black is solid, as everywhere else in the project.
