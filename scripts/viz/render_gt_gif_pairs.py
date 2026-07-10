#!/usr/bin/env python3
"""Render GT BEV GIFs for the qualitative token set used in ChainFlow figures."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for path in (REPO_ROOT, NUPLAN_DEVKIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from navsim.evaluate.render_gt_gif_from_tokens import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_MAPS_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SENSOR_BLOBS_PATH,
    render_pairs,
)


PAIRS: List[Tuple[str, str]] = [
    ("9b7108902e7158d6", "2021.09.16.19.27.01_veh-45_00472_00711"),
    ("c08cd52346155301", "2021.10.06.08.16.17_veh-52_01430_01579"),
    ("020dee65dab453bb", "2021.06.28.16.29.11_veh-38_03263_03766"),
    ("2c337eb368fb54ca", "2021.09.16.15.12.03_veh-42_01037_01434"),
    ("a2d180a344d15054", "2021.09.29.14.44.26_veh-28_01331_01485"),
    ("00fcad6d092c5e8e", "2021.09.16.19.27.01_veh-45_00472_00711"),
    ("0eb7dda83bbe5fb2", "2021.08.30.13.45.25_veh-40_00878_01104"),
    ("277f191c94b952f3", "2021.09.29.15.23.04_veh-28_00814_01101"),
    ("4376d00ed2245c21", "2021.10.06.08.16.17_veh-52_01590_01725"),
    ("923e4fcf3daa57f8", "2021.09.29.14.44.26_veh-28_01331_01485"),
    ("680c8d90658556da", "2021.10.06.07.26.10_veh-52_02208_02394"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maps-root",
        type=Path,
        default=DEFAULT_MAPS_ROOT,
        help="nuPlan map root. Defaults to $NUPLAN_MAPS_ROOT or /data/download/maps.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="NAVSIM logs split path containing *.pkl files.",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        type=Path,
        default=DEFAULT_SENSOR_BLOBS_PATH,
        help="NAVSIM sensor blobs path. No sensors are loaded, but SceneLoader requires the argument.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated GIFs.",
    )
    parser.add_argument("--gif-steps", type=int, default=8, help="Number of future GT steps to render.")
    parser.add_argument("--gif-fps", type=float, default=4.0, help="GIF playback FPS.")
    parser.add_argument("--bev-x-min", type=float, default=-20.0, help="Longitudinal rear limit in meters.")
    parser.add_argument("--bev-x-max", type=float, default=40.0, help="Longitudinal front limit in meters.")
    parser.add_argument("--bev-y-min", type=float, default=-30.0, help="Lateral right limit in meters.")
    parser.add_argument("--bev-y-max", type=float, default=30.0, help="Lateral left limit in meters.")
    parser.add_argument(
        "--prefix",
        type=str,
        default="GT",
        help="Output filename prefix. Files are {prefix}_{token}_BEV_gt.gif.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.gif_steps < 1:
        raise ValueError("--gif-steps must be >= 1.")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive.")
    if args.bev_x_min >= args.bev_x_max:
        raise ValueError("--bev-x-min must be smaller than --bev-x-max.")
    if args.bev_y_min >= args.bev_y_max:
        raise ValueError("--bev-y-min must be smaller than --bev-y-max.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.maps_root.expanduser())

    print("==============================================")
    print(" ChainFlow qualitative GT BEV GIFs")
    print("==============================================")
    print(f" Maps root:        {args.maps_root.expanduser()}")
    print(f" NAVSIM logs:      {args.data_path.expanduser()}")
    print(f" Sensor blobs:     {args.sensor_blobs_path.expanduser()}")
    print(f" Output dir:       {args.output_dir.expanduser()}")
    print(f" GIF steps / FPS:  {args.gif_steps} / {args.gif_fps}")
    print(f" Pair count:       {len(PAIRS)}")
    print("==============================================")

    ok, fail = render_pairs(
        pairs=PAIRS,
        data_path=args.data_path,
        sensor_blobs_path=args.sensor_blobs_path,
        output_dir=args.output_dir,
        gif_steps=args.gif_steps,
        gif_fps=args.gif_fps,
        prefix=args.prefix,
        bev_x_min=args.bev_x_min,
        bev_x_max=args.bev_x_max,
        bev_y_min=args.bev_y_min,
        bev_y_max=args.bev_y_max,
    )
    print(f"Done: ok={ok} fail={fail} -> {args.output_dir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
