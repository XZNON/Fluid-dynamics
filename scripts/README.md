# `scripts/` — visualisation drivers on top of `flow`

These are not part of the solver and not part of the product. Each one is a thin
driver that builds a `flow.Case` the normal way and then changes **how the run is
drawn or paced** — never what it computes. Every solver parameter still comes
from `flow.autoconfig.plan`; these scripts choose none of it.

They live here rather than at the repo root because they are experiments, and
because `python -m flow` (`flow/cli.py`) remains the product command. `bench.py`
stays at the root — `CLAUDE.md` names it there.

| Script | What it changes | Why it exists |
|---|---|---|
| `slowmo.py` | `Plan.steps_per_frame` only | `PLAYBACK_SPEED = 1.0` renders a physically small, fast case as a 1-frame video. Queued as issue `214ccb320994`. |
| `streamlines.py` | composites advected tracer particles onto the rendered frame | Vorticity of a uniform free stream is zero, so undisturbed flow paints one flat colour. Tracers are the only thing that reads as motion. |
| `windtunnel.py` | composites rake smoke filaments; `--tint` dims the field; `--u-lattice` | The smoke-tunnel picture. `--u-lattice` is the lever that works for sharp-cornered bodies — see issue `5e94900a8170`. |

Constraints 9 and 10 hold in all three: the field drawn is vorticity, computed
in `lbm.probe` and coloured by the one `lbm.render.render`. Smoke, tracers and
dimming are composited onto the RGB that returns — compositing, not a second
renderer.

## Running

Each script puts the repo root on `sys.path` itself, so cwd does not matter:

```bash
myenv/Scripts/python.exe scripts/windtunnel.py --shape examples/shapes/test2.png \
    --fluid air --speed "3 m/s" --size "0.36 mm" --downstream 2.5 --out outputs/tunnel.mp4
```

Drop `--out` for a live window. Shapes are in `examples/shapes/`; rendered runs
belong in `outputs/`, which is gitignored.

**Note.** `windtunnel.py` and `streamlines.py` flip the frame on the way out of
`_compose` to work around issue `9e58e90c9b58` — `RecordSink` does not flip, while
`LiveSink` does, so saved files are vertically mirrored. When that lands fixed
in `lbm/record.py`, remove the flip here or the files invert again.
