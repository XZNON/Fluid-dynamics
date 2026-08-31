"""Run a `flow` case in slow motion.

``python -m flow`` renders at real-time playback (``flow/autoconfig.py``
PLAYBACK_SPEED = 1.0), which is right for a 2 cm disc in water and useless for a
0.36 mm body in air: that flow's whole life is 5.8 ms, so real time is one frame.

This overrides ``Plan.steps_per_frame`` so the run yields a watchable number of
frames instead. It changes nothing about the physics — only how often a frame is
kept. Every solver parameter still comes from ``flow.autoconfig.plan``.

    myenv/Scripts/python.exe slowmo.py --shape test_shape.png --fluid air \
        --speed "5 m/s" --size "0.36 mm" --out wake.mp4

Drop ``--out`` for a live window instead.
"""

from __future__ import annotations

import argparse
import dataclasses

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
