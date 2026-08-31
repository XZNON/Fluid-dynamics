"""Run a `flow` case in slow motion.

``python -m flow`` renders at real-time playback (``flow/autoconfig.py``
PLAYBACK_SPEED = 1.0), which is right for a 2 cm disc in water and useless for a
0.36 mm body in air: that flow's whole life is 5.8 ms, so real time is one frame.

This overrides ``Plan.steps_per_frame`` so the run yields a watchable number of
frames instead. It changes nothing about the physics — only how often a frame is
kept. Every solver parameter still comes from ``flow.autoconfig.plan``.

    myenv/Scripts/python.exe scripts/slowmo.py --shape examples/shapes/test_shape.png --fluid air \
        --speed "5 m/s" --size "0.36 mm" --out wake.mp4

Drop ``--out`` for a live window instead.
"""

from __future__ import annotations

import argparse
import dataclasses

import sys
from pathlib import Path

# This script lives in ``scripts/`` but drives the packages at the repo root, so
# put the root on the path before importing them. Keeps ``python scripts/x.py``
# working from anywhere without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flow import Case


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", required=True)
    p.add_argument("--fluid", required=True)
    p.add_argument("--speed", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--quality", default="balanced")
    p.add_argument("--backend", default="warp", choices=("numpy", "warp"))
    p.add_argument("--frames", type=int, default=300, help="how many frames to keep")
    p.add_argument(
        "--downstream",
        type=float,
        default=1.0,
        help="stretch the domain along the flow by this factor, so the wake has "
        "room to develop before it leaves the frame. Costs cells linearly. The "
        "span across the flow is NOT touched -- that one sets the blockage "
        "ratio the plan checked, and changing it would change the answer",
    )
    p.add_argument("--span", type=float, default=None, metavar="D",
                   help="cross-flow extent in body diameters (plan default 24, "
                        "which is Rung 3's own domain, D-075). Blockage ratio is "
                        "D/span: 24 -> 4.2%%, 14 -> 7.1%%, 10 -> 10%%, constraint "
                        "12's ceiling. MEASURED on test2.png in air at 3 m/s, "
                        "against span 24: at 14 D Cd +0.4%% (inside the run-to-run "
                        "scatter) and St +1.7%%; at 10 D Cd +1.9%% and St +5.0%%. "
                        "St is the sensitive one and rises monotonically -- "
                        "confinement changes how the wake oscillates before it "
                        "changes how hard the body is pushed. 14 D costs 42%% "
                        "fewer cells for a change inside the noise; below 12 D "
                        "you are correcting for blockage, not ignoring it")
    p.add_argument("--seconds", default=None, help="physical time; default the plan's own")
    p.add_argument("--out", default=None, help=".mp4 / .gif / a directory")
    p.add_argument("--fps", type=float, default=30.0, help="playback rate of the file")
    args = p.parse_args()

    case = Case.from_image(
        args.shape,
        fluid=args.fluid,
        speed=args.speed,
        size=args.size,
        quality=args.quality,
        backend=args.backend,
    )

    case.explain()  # prints and returns; printing the return would double it

    if not case.runnable:
        return 2

    # Override 1: frame cadence. Size it off the steps this run will actually
    # take -- --seconds may have shortened it, and sizing off plan.steps then
    # would keep a fraction of the frames asked for.
    steps = case._steps(args.seconds)
    case.plan = dataclasses.replace(
        case.plan, steps_per_frame=max(1, steps // args.frames)
    )
    print(f"\nslow motion: {case.plan.steps_per_frame} steps/frame "
          f"-> ~{steps // case.plan.steps_per_frame} frames")

    # Override 2: extra fetch downstream. Case._domain() puts the leading edge
    # UPSTREAM_D diameters from the inlet and centres the body across the flow,
    # both computed from the plan -- so growing nx alone leaves the upstream
    # distance and the blockage ratio exactly as planned and spends every new
    # cell behind the body. More downstream fetch than the plan asked for can
    # only help the outlet; it costs cells, not accuracy.
    if args.span is not None:
        ny0, nx0 = case.plan.domain
        ny_new = int(round(args.span * case.plan.cells_per_length))
        body = case.prepared.mask.shape[0]
        if ny_new <= body + 2:
            print(f"refused: span {args.span} D is {ny_new} cells, not larger "
                  f"than the {body}-cell body.")
            return 2
        blockage = body / ny_new
        flag = "  OVER constraint 12's 10%" if blockage > 0.10 else ""
        print(f"span       {ny0} -> {ny_new} cells across the flow, "
              f"blockage {blockage:.1%}{flag}")
        case.plan = dataclasses.replace(case.plan, domain=(ny_new, nx0))

    if args.downstream != 1.0:
        ny, nx = case.plan.domain
        wide = int(round(nx * args.downstream))
        case.plan = dataclasses.replace(case.plan, domain=(ny, wide))
        print(f"downstream: {nx} -> {wide} cells along the flow "
              f"({ny}x{wide} = {ny * wide / 1e6:.2f}M cells, "
              f"{args.downstream:.2g}x the plan's)")
    print()

    # record= streams each frame to the sink as it is produced, so a long run at
    # a big domain never has to hold them all. keep_frames=False is what keeps
    # this under flow.case.FRAME_MEMORY_BUDGET (512 MB): at 960x1838 a frame is
    # 5.3 MB and 300 of them would be 1.6 GB, which the budget would silently
    # cap -- it warns, but a capped run is the first third of the flow, not the
    # flow. Nothing calls Result.save() afterwards; the file is already written.
    result = case.run(
        seconds=args.seconds,
        live=args.out is None,
        record=args.out,
        keep_frames=args.out is None,
        quiet=True,
    )

    result.summary()  # prints as a side effect
    if args.out:
        print("wrote", args.out)
    return 0 if result.stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
