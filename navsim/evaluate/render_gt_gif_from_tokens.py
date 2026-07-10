#!/usr/bin/env python3
"""Render green GT BEV GIFs for selected NAVSIM token/log pairs."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for path in (REPO_ROOT, NUPLAN_DEVKIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_MAPS_ROOT = Path(os.environ.get("NUPLAN_MAPS_ROOT", "/data/download/maps")).expanduser()
_OPENSCENE_ROOT = Path(
    os.environ.get(
        "OPENSCENE_DATA_ROOT",
        "/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1",
    )
)
DEFAULT_DATA_PATH = _OPENSCENE_ROOT / "navsim_logs" / "test"
DEFAULT_SENSOR_BLOBS_PATH = _OPENSCENE_ROOT / "sensor_blobs" / "test"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("GT_GIF_OUTPUT_DIR", "./gt_gifs")).expanduser()
GIF_FIGSIZE = (9.0, 9.0)
GIF_DPI = 180
GT_COLOR = "#59a14f"


def _preparse_maps_root() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    args, _ = parser.parse_known_args()
    maps_root = args.maps_root.expanduser()
    os.environ["NUPLAN_MAPS_ROOT"] = str(maps_root)
    return maps_root


MAPS_ROOT = _preparse_maps_root()

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Annotations, Frame, Scene, SceneFilter, SensorConfig, Trajectory
from navsim.planning.scenario_builder.navsim_scenario_utils import normalize_angle, rotate_state_se2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.visualization.bev import (
    add_annotations_to_bev_ax,
    add_ego_box_at_local_pose_to_bev_ax,
    add_map_to_bev_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.config import AGENT_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


GT_STYLE = {
    "fill_color": GT_COLOR,
    "fill_color_alpha": 1.0,
    "line_color": GT_COLOR,
    "line_color_alpha": 1.0,
    "line_width": 2.8,
    "line_style": "-",
    "marker": "o",
    "marker_size": 6,
    "marker_edge_color": "black",
    "zorder": 4,
}


def parse_pair(raw: str) -> Tuple[str, str]:
    parts = [part.strip() for part in raw.split(",", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid pair {raw!r}; expected TOKEN,LOG_NAME.")
    return parts[0], parts[1]


def load_pairs_from_file(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                pairs.append(parse_pair(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return pairs


def _global_pose_from_frame(frame: Frame) -> StateSE2:
    pose = frame.ego_status.ego_pose
    return StateSE2(float(pose[0]), float(pose[1]), float(pose[2]))


def _empty_annotations() -> Annotations:
    return Annotations(
        boxes=np.zeros((0, 7), dtype=np.float32),
        names=[],
        velocity_3d=np.zeros((0, 3), dtype=np.float32),
        instance_tokens=[],
        track_tokens=[],
    )


def _transform_annotations_to_current_ego(future_frame: Frame, current_frame: Frame) -> Annotations:
    """Warp future-frame GT boxes into the current ego frame."""
    if len(future_frame.annotations.boxes) == 0:
        return _empty_annotations()

    future_origin = _global_pose_from_frame(future_frame)
    current_origin = _global_pose_from_frame(current_frame)

    global_poses: List[List[float]] = []
    for box in future_frame.annotations.boxes:
        local_se2 = rotate_state_se2(
            StateSE2(float(box[0]), float(box[1]), float(box[6])),
            angle=future_origin.heading,
        )
        global_poses.append(
            [
                local_se2.x + future_origin.x,
                local_se2.y + future_origin.y,
                normalize_angle(local_se2.heading),
            ]
        )

    local_array = convert_absolute_to_relative_se2_array(
        current_origin,
        np.asarray(global_poses, dtype=np.float64),
    )

    transformed_boxes: List[np.ndarray] = []
    for idx, box in enumerate(future_frame.annotations.boxes):
        transformed = np.array(box, dtype=np.float32, copy=True)
        transformed[0] = float(local_array[idx, 0])
        transformed[1] = float(local_array[idx, 1])
        transformed[6] = float(local_array[idx, 2])
        transformed_boxes.append(transformed)

    return Annotations(
        boxes=np.asarray(transformed_boxes, dtype=np.float32),
        names=list(future_frame.annotations.names),
        velocity_3d=np.asarray(future_frame.annotations.velocity_3d, dtype=np.float32),
        instance_tokens=list(future_frame.annotations.instance_tokens),
        track_tokens=list(future_frame.annotations.track_tokens),
    )


def _filter_annotations_by_track_tokens(
    annotations: Annotations,
    allowed_track_tokens: set[str],
) -> Annotations:
    """Keep annotation rows whose track token was visible at the current frame."""
    if not allowed_track_tokens or len(annotations.track_tokens) == 0:
        return _empty_annotations()

    keep_indices = [
        idx for idx, track_token in enumerate(annotations.track_tokens) if track_token in allowed_track_tokens
    ]
    if not keep_indices:
        return _empty_annotations()

    return Annotations(
        boxes=np.asarray([annotations.boxes[idx] for idx in keep_indices], dtype=np.float32),
        names=[annotations.names[idx] for idx in keep_indices],
        velocity_3d=np.asarray([annotations.velocity_3d[idx] for idx in keep_indices], dtype=np.float32),
        instance_tokens=[annotations.instance_tokens[idx] for idx in keep_indices],
        track_tokens=[annotations.track_tokens[idx] for idx in keep_indices],
    )


def _local_box_center_to_global_xy(box: np.ndarray, ego_pose_global: StateSE2) -> np.ndarray:
    local_center = rotate_state_se2(
        StateSE2(float(box[0]), float(box[1]), float(box[6])),
        angle=ego_pose_global.heading,
    )
    return np.asarray(
        [local_center.x + ego_pose_global.x, local_center.y + ego_pose_global.y],
        dtype=np.float64,
    )


def _compute_static_track_tokens(
    scene: Scene,
    current_frame_idx: int,
    current_track_tokens: set[str],
    ref_future_step: int = 8,
    static_l2_thresh_m: float = 2.0,
) -> set[str]:
    """Treat nearly stationary current-frame agents as fixed background boxes."""
    max_future_idx = len(scene.frames) - 1
    ref_future_idx = min(current_frame_idx + ref_future_step, max_future_idx)
    if ref_future_idx <= current_frame_idx:
        return set()

    current_frame = scene.frames[current_frame_idx]
    ref_future_frame = scene.frames[ref_future_idx]
    current_pose = _global_pose_from_frame(current_frame)
    ref_pose = _global_pose_from_frame(ref_future_frame)

    current_rows = {
        track_token: idx
        for idx, track_token in enumerate(current_frame.annotations.track_tokens)
        if track_token in current_track_tokens
    }
    ref_rows = {
        track_token: idx for idx, track_token in enumerate(ref_future_frame.annotations.track_tokens)
    }

    static_tracks: set[str] = set()
    for track_token in current_track_tokens:
        current_idx = current_rows.get(track_token)
        ref_idx = ref_rows.get(track_token)
        if current_idx is None or ref_idx is None:
            continue

        current_xy = _local_box_center_to_global_xy(current_frame.annotations.boxes[current_idx], current_pose)
        ref_xy = _local_box_center_to_global_xy(ref_future_frame.annotations.boxes[ref_idx], ref_pose)
        if float(np.linalg.norm(ref_xy - current_xy)) < static_l2_thresh_m:
            static_tracks.add(track_token)

    return static_tracks


def _slice_trajectory(trajectory: Trajectory, num_poses: int) -> Trajectory:
    poses = trajectory.poses[:num_poses]
    sampling = TrajectorySampling(
        num_poses=int(poses.shape[0]),
        interval_length=trajectory.trajectory_sampling.interval_length,
    )
    return Trajectory(poses, sampling)


def _fig_to_image(fig: plt.Figure, dpi: int = GIF_DPI) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).copy()
    buf.close()
    return image


def _green_ego_box_style(fill_alpha: float, line_alpha: float, zorder: int) -> Dict[str, Any]:
    style = dict(AGENT_CONFIG[TrackedObjectType.EGO])
    style["fill_color"] = GT_COLOR
    style["line_color"] = "black"
    style["fill_color_alpha"] = fill_alpha
    style["line_color_alpha"] = line_alpha
    style["zorder"] = zorder
    return style


def _add_gt_gif_background(
    ax: plt.Axes,
    scene: Scene,
    current_frame: Frame,
    static_background_annotations: Optional[Annotations],
) -> None:
    """Draw fixed map, static neighbors, and transparent planning-time ego."""
    add_map_to_bev_ax(ax, scene.map_api, _global_pose_from_frame(current_frame))
    if static_background_annotations is not None and len(static_background_annotations.boxes) > 0:
        add_annotations_to_bev_ax(ax, static_background_annotations, add_ego=False)
    add_ego_box_at_local_pose_to_bev_ax(
        ax,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        _green_ego_box_style(fill_alpha=0.25, line_alpha=0.55, zorder=2),
        add_heading=False,
    )


def _apply_bev_limits(
    ax: plt.Axes,
    bev_x_min: Optional[float],
    bev_x_max: Optional[float],
    bev_y_min: Optional[float],
    bev_y_max: Optional[float],
) -> None:
    if None in (bev_x_min, bev_x_max, bev_y_min, bev_y_max):
        configure_bev_ax(ax)
        return
    ax.set_aspect("equal")
    ax.set_xlim(float(bev_y_min), float(bev_y_max))
    ax.set_ylim(float(bev_x_min), float(bev_x_max))
    ax.invert_xaxis()


def render_gt_bev_gif(
    scene: Scene,
    output_path: Path,
    gif_steps: int,
    gif_fps: float,
    bev_x_min: Optional[float] = -20.0,
    bev_x_max: Optional[float] = 40.0,
    bev_y_min: Optional[float] = -30.0,
    bev_y_max: Optional[float] = 30.0,
) -> Path:
    """Render one green GT ego trajectory GIF with moving future GT agents."""
    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    max_future_steps = len(scene.frames) - current_frame_idx - 1
    num_future_steps = min(gif_steps, scene.scene_metadata.num_future_frames, max_future_steps)
    if num_future_steps < 1:
        raise ValueError(f"Scene {scene.scene_metadata.initial_token} has no future frames to render.")

    gt_traj = scene.get_future_trajectory(num_trajectory_frames=num_future_steps)
    current_frame = scene.frames[current_frame_idx]
    duration_ms = int(round(1000.0 / max(gif_fps, 0.1)))
    images: List[Image.Image] = []
    current_track_tokens = set(current_frame.annotations.track_tokens)
    static_track_tokens: set[str] = set()
    static_background_annotations: Optional[Annotations] = None
    if current_track_tokens:
        static_track_tokens = _compute_static_track_tokens(
            scene=scene,
            current_frame_idx=current_frame_idx,
            current_track_tokens=current_track_tokens,
            ref_future_step=8,
            static_l2_thresh_m=2.0,
        )
        static_background_annotations = _filter_annotations_by_track_tokens(
            current_frame.annotations,
            static_track_tokens,
        )
    dynamic_track_tokens = current_track_tokens - static_track_tokens

    for step in range(num_future_steps + 1):
        future_frame = scene.frames[current_frame_idx + step]
        fig, ax = plt.subplots(figsize=GIF_FIGSIZE)
        ax.set_facecolor("white")
        _add_gt_gif_background(ax, scene, current_frame, static_background_annotations)

        warped_anns = _transform_annotations_to_current_ego(future_frame, current_frame)
        warped_dynamic_anns = _filter_annotations_by_track_tokens(warped_anns, dynamic_track_tokens)
        add_annotations_to_bev_ax(ax, warped_dynamic_anns, add_ego=False)

        if step > 0:
            partial_gt = _slice_trajectory(gt_traj, step)
            add_trajectory_to_bev_ax(ax, partial_gt, GT_STYLE, draw_ego_box_at_end=False)
            add_ego_box_at_local_pose_to_bev_ax(
                ax,
                partial_gt.poses[-1],
                _green_ego_box_style(fill_alpha=1.0, line_alpha=1.0, zorder=5),
                add_heading=True,
            )

        _apply_bev_limits(ax, bev_x_min, bev_x_max, bev_y_min, bev_y_max)
        configure_ax(ax)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        images.append(_fig_to_image(fig))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def load_scene(
    data_path: Path,
    sensor_blobs_path: Path,
    token: str,
    log_name: str,
    num_future_frames: int,
) -> Scene:
    scene_filter = SceneFilter(
        tokens=[token],
        log_names=[log_name],
        num_history_frames=4,
        num_future_frames=num_future_frames,
        frame_interval=1,
        has_route=False,
    )
    loader = SceneLoader(
        data_path=data_path.expanduser().resolve(),
        sensor_blobs_path=sensor_blobs_path.expanduser().resolve(),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if token not in loader.tokens:
        raise ValueError(f"Token not found: token={token}, log_name={log_name}, data_path={data_path}")
    return loader.get_scene_from_token(token)


def render_pairs(
    pairs: Iterable[Tuple[str, str]],
    data_path: Path,
    sensor_blobs_path: Path,
    output_dir: Path,
    gif_steps: int = 8,
    gif_fps: float = 4.0,
    prefix: str = "GT",
    bev_x_min: Optional[float] = -20.0,
    bev_x_max: Optional[float] = 40.0,
    bev_y_min: Optional[float] = -30.0,
    bev_y_max: Optional[float] = 30.0,
) -> Tuple[int, int]:
    ok = 0
    fail = 0
    for token, log_name in pairs:
        output_path = output_dir / f"{prefix}_{token}_BEV_gt.gif"
        print(f"=== {token} ===")
        try:
            scene = load_scene(
                data_path=data_path,
                sensor_blobs_path=sensor_blobs_path,
                token=token,
                log_name=log_name,
                num_future_frames=gif_steps,
            )
            saved_path = render_gt_bev_gif(
                scene=scene,
                output_path=output_path,
                gif_steps=gif_steps,
                gif_fps=gif_fps,
                bev_x_min=bev_x_min,
                bev_x_max=bev_x_max,
                bev_y_min=bev_y_min,
                bev_y_max=bev_y_max,
            )
            print(f"Saved: {saved_path}")
            ok += 1
        except Exception as exc:
            print(f"FAILED: token={token}, log_name={log_name}: {exc}")
            fail += 1
    return ok, fail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maps-root",
        type=Path,
        default=MAPS_ROOT,
        help="nuPlan map root. Defaults to $NUPLAN_MAPS_ROOT or /data/download/maps.",
    )
    parser.add_argument("--pair", action="append", default=[], help="TOKEN,LOG_NAME pair. Repeatable.")
    parser.add_argument("--pairs-file", type=Path, default=None, help="Text file with one TOKEN,LOG_NAME per line.")
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


def _resolve_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    pairs = [parse_pair(raw_pair) for raw_pair in args.pair]
    if args.pairs_file is not None:
        pairs.extend(load_pairs_from_file(args.pairs_file))
    if not pairs:
        raise ValueError("Provide at least one --pair TOKEN,LOG_NAME or --pairs-file.")
    return pairs


def main() -> int:
    args = parse_args()
    validate_args(args)
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.maps_root.expanduser())
    pairs = _resolve_pairs(args)

    print("==============================================")
    print(" NAVSIM GT BEV GIF Rendering")
    print("==============================================")
    print(f" Maps root:        {args.maps_root.expanduser()}")
    print(f" NAVSIM logs:      {args.data_path.expanduser()}")
    print(f" Sensor blobs:     {args.sensor_blobs_path.expanduser()}")
    print(f" Output dir:       {args.output_dir.expanduser()}")
    print(f" GIF steps / FPS:  {args.gif_steps} / {args.gif_fps}")
    print(f" Pair count:       {len(pairs)}")
    print("==============================================")

    ok, fail = render_pairs(
        pairs=pairs,
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
