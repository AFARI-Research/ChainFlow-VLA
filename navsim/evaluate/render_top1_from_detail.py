#!/usr/bin/env python3
"""
从单次评测导出的 ``detail.json`` 绘制单模型 top-1 轨迹：BEV 鸟瞰图 + 前视相机投影。

输入路径约定（``token`` / ``log_name`` 从路径自动解析）::

    <exp_root>/<csv_stem>_selected_proposals/<log_name>/<token>/<proposal_role>/detail.json

输出文件（在 ``--output-dir`` 下）::

    <image_prefix>_BEV.<fmt>
    <image_prefix>_front_camera.<fmt>
    <image_prefix>_BEV_future_agents.gif   # 仅当 ``--render-bev-gif`` 时

必填参数
--------
--detail-json          proposal 的 detail.json（如 ``.../proposal_top1/detail.json``）
--data-path            NavSim 日志根目录（含 ``*.pkl``，或 ``navsim_logs/test/`` 等子目录）
--sensor-blobs-path    传感器 blob 根目录
--output-dir           输出目录
--image-prefix 或 --model-name   输出文件名前缀（二选一）

常用可选参数
------------
--traj-field           detail 中要画的轨迹字段（默认 ``pred_traj_local``；
                       也可用 ``simulated_traj_local`` 等，需为 ``[N,>=3]`` 的 ego 局部坐标）
--legend-labels        图例名，逗号分隔：第 1 个为 GT，其后依次为各条轨迹
--traj-colors          轨迹颜色，逗号分隔 RRGGBB（跳过 GT；第 1 个值给第 1 条非 GT 轨迹）
--interval-length      轨迹点时间间隔（秒，默认 0.1）
--bev-size-x / --bev-size-y   BEV 对称范围（米，以 ego 为中心，如 80 → [-40,40]）
--bev-x-min/max, --bev-y-min/max  BEV 非对称范围（米；ego-x 为纵轴前后，ego-y 为横轴左右）
--format               输出格式：pdf、jpg、png，可逗号组合（默认 pdf）
--draw-camera-traj     在前视（及拼接侧视）图像上叠加 GT/预测轨迹（默认关闭）
--concat-side-front    将左前+前视+右前横向拼接为一张 _front_camera 图（默认关闭）
--left-front-width-ratio / --right-front-width-ratio  侧视拼接后宽度相对前视比例（默认 0.3）
--left-front-width-ratio-start/end  左前图横向切片 [start,end]，从左到右，与像素列一致（默认 0/1）
--right-front-width-ratio-start/end 右前图横向切片，语义同上（默认 0/1）
--render-bev-gif           额外输出 BEV 动画 GIF（当前帧背景 + 预测自车轨迹 + 未来他车 GT 位置）
--gif-steps / --gif-fps    GIF 帧数与帧率（仅配合 ``--render-bev-gif``）
--draw-other-agents        GIF 中是否绘制未来他车（默认 true）

基础示例::

    python navsim/evaluate/render_top1_from_detail.py \\
        --detail-json /data/exp/<csv_stem>_selected_proposals/<log_name>/<token>/proposal_top1/detail.json \\
        --data-path /path/to/navsim_logs \\
        --sensor-blobs-path /path/to/sensor_blobs \\
        --output-dir /tmp/bev_vis \\
        --model-name stage_2 \\
        --format pdf,jpg

自定义轨迹字段、图例与颜色::

    python navsim/evaluate/render_top1_from_detail.py \\
        --detail-json .../detail.json \\
        --data-path ... --sensor-blobs-path ... --output-dir /tmp/out \\
        --image-prefix recogdrive \\
        --traj-field pred_traj_local \\
        --legend-labels "Human Expert,RecogDrive" \\
        --traj-colors FF0000 \\
        --bev-size-x 80 --bev-size-y 80

说明
----
* ``detail.json`` 若轨迹在顶层、指标在 ``detail`` 子对象中，会自动合并读取。
* ``--traj-colors`` 未指定时：GT 为绿色，第 1 条轨迹为橙色（与 PDM 默认可视化一致）。
* ``--legend-labels`` 未指定时：GT 为 ``Human Expert``，模型轨迹名由 ``--traj-field`` 推断。
* 优先使用 ``*_traj_local`` 字段；``pred_states`` 等为全局状态，不宜直接用于 BEV。
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Annotations, Frame, Scene, SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.eval_visualization import _configure_bev_ax, _trajectory_from_detail
from navsim.planning.scenario_builder.navsim_scenario_utils import normalize_angle, rotate_state_se2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.visualization.bev import (
    add_annotations_to_bev_ax,
    add_configured_bev_on_ax,
    add_ego_box_at_local_pose_to_bev_ax,
    add_map_to_bev_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG
from navsim.visualization.camera import add_camera_ax, add_trajectory_to_camera_ax


def _parse_detail_json_path(detail_json_path: Path) -> Tuple[Path, str, str]:
    """
    Parse:
    <exp_root>/<csv_stem>_selected_proposals/<log_name>/<token>/<proposal_role>/detail.json
    Return (exp_root, log_name, token).
    """
    if detail_json_path.name != "detail.json":
        raise ValueError(f"Expected detail.json, got: {detail_json_path}")

    proposal_dir = detail_json_path.parent
    token_dir = proposal_dir.parent
    log_dir = token_dir.parent
    selected_proposals_dir = log_dir.parent
    suffix = "_selected_proposals"
    if not selected_proposals_dir.name.endswith(suffix):
        raise ValueError(
            "detail.json path must follow "
            "<csv_stem>_selected_proposals/<log_name>/<token>/<proposal_role>/detail.json"
        )

    exp_root = selected_proposals_dir.parent
    log_name = log_dir.name
    token = token_dir.name
    return exp_root, log_name, token


# Trajectory arrays sometimes live at JSON top-level while PDM metrics sit under "detail".
_TOP_LEVEL_TRAJ_KEYS = (
    "pred_traj_local",
    "simulated_traj_local",
    "pred_traj",
    "simulated_traj",
    "reference_traj",
)


def _load_detail_payload(detail_json_path: Path) -> Dict[str, Any]:
    with open(detail_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected detail.json format: {detail_json_path}")

    detail = raw.get("detail")
    if isinstance(detail, dict):
        merged = dict(detail)
        for key in _TOP_LEVEL_TRAJ_KEYS:
            if key in raw and key not in merged:
                merged[key] = raw[key]
        return merged

    return raw


def _traj_poses_from_payload(payload: Dict[str, Any], traj_field: str) -> np.ndarray:
    if traj_field not in payload:
        raise KeyError(f"detail.json missing key: {traj_field}")
    try:
        poses = np.asarray(payload[traj_field], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{traj_field} is not a numeric trajectory array") from exc
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"{traj_field} must have shape [N, >=3], got {poses.shape}")
    return poses


def _build_scene(
    data_path: Path,
    sensor_blobs_path: Path,
    token: str,
    log_name: str,
    num_future_frames: int,
) -> Scene:
    scene_filter = SceneFilter(
        tokens=[token],
        log_names=[log_name],
        num_future_frames=max(1, num_future_frames),
        frame_interval=1,
        has_route=False,
    )
    loader = SceneLoader(
        data_path=data_path.expanduser().resolve(),
        sensor_blobs_path=sensor_blobs_path.expanduser().resolve(),
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=True,
            cam_l0=True,
            cam_l1=True,
            cam_l2=True,
            cam_r0=True,
            cam_r1=True,
            cam_r2=True,
            cam_b0=True,
            lidar_pc=False,
        ),
    )
    if token not in loader.tokens:
        raise ValueError(
            "Token not found in SceneLoader. "
            f"token={token}, log_name={log_name}, loaded_scenes={len(loader.tokens)}. "
            "Please verify data-path/sensor-blobs-path and whether this token exists in the selected split."
        )
    return loader.get_scene_from_token(token)


def _parse_output_formats(raw: str) -> List[str]:
    """
    Parse comma-separated output formats: pdf, jpg/jpeg, png, or none (no static images).
    Order is preserved; duplicates removed.
    """
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--format must list at least one format (e.g. pdf, pdf,jpg, or none).")
    if len(parts) == 1 and parts[0] == "none":
        return []
    allowed = {"pdf", "jpg", "jpeg", "png", "none"}
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        if p == "jpeg":
            p = "jpg"
        if p == "none":
            raise ValueError(
                "Use --format none alone to skip static images; do not mix none with pdf/jpg/png."
            )
        if p not in allowed:
            raise ValueError(
                f"Unsupported format {p!r}; use one of: pdf, jpg, png, none (comma-separated)."
            )
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _matplotlib_format_and_suffix(fmt: str) -> Tuple[str, str]:
    if fmt == "pdf":
        return "pdf", ".pdf"
    if fmt == "jpg":
        return "jpeg", ".jpg"
    if fmt == "png":
        return "png", ".png"
    raise ValueError(f"Unknown format: {fmt}")


def _pick_camera_by_names(frame, candidate_names: Sequence[str]) -> Any:
    for name in candidate_names:
        cam = getattr(frame.cameras, name, None)
        if cam is not None and getattr(cam, "image", None) is not None:
            return cam
    return None


def _pick_front_camera(frame) -> Any:
    """Prefer cam_f0; fallback to the first available camera with an image."""
    return _pick_camera_by_names(
        frame, ["cam_f0", "cam_r0", "cam_l0", "cam_r1", "cam_l1", "cam_r2", "cam_l2", "cam_b0"]
    )


def _pick_left_front_camera(frame) -> Any:
    """Left-front camera for side montage (prefer cam_l0)."""
    return _pick_camera_by_names(frame, ["cam_l0", "cam_l1", "cam_l2"])


def _pick_right_front_camera(frame) -> Any:
    """Right-front camera for side montage (prefer cam_r0)."""
    return _pick_camera_by_names(frame, ["cam_r0", "cam_r1", "cam_r2"])


_DEFAULT_GT_COLOR = "green"
_DEFAULT_TRAJ_COLORS = ("tab:orange", "tab:blue", "tab:purple", "red")


def _parse_comma_list(raw: Optional[str]) -> List[str]:
    if raw is None or not str(raw).strip():
        return []
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _parse_hex_rgb(raw: str) -> str:
    s = raw.strip().lstrip("#")
    if len(s) != 6 or not re.fullmatch(r"[0-9A-Fa-f]{6}", s):
        raise ValueError(f"Invalid RGB hex {raw!r}; expected RRGGBB (e.g. FF0000 or #FF0000).")
    return f"#{s.lower()}"


def _parse_traj_colors(raw: Optional[str]) -> List[str]:
    parts = _parse_comma_list(raw)
    return [_parse_hex_rgb(p) for p in parts]


def _default_traj_color(traj_index: int) -> str:
    if traj_index < len(_DEFAULT_TRAJ_COLORS):
        return _DEFAULT_TRAJ_COLORS[traj_index]
    return "red"


def _resolve_traj_color(traj_index: int, traj_colors: Sequence[str]) -> str:
    if traj_index < len(traj_colors):
        return traj_colors[traj_index]
    return _default_traj_color(traj_index)


def _default_pred_legend_label(traj_field: str) -> str:
    if traj_field == "pred_traj_local":
        return "Model Prediction"
    if traj_field == "simulated_traj_local":
        return "Simulated (local)"
    return traj_field


def _resolve_legend_labels(
    legend_labels: Sequence[str],
    traj_field: str,
    num_trajs: int,
) -> Tuple[str, List[str]]:
    """
    Resolve legend text: index 0 = GT, then one label per rendered trajectory.
    Missing entries fall back to PDM-style defaults.
    """
    gt_label = legend_labels[0] if len(legend_labels) >= 1 else "Human Expert"
    traj_labels: List[str] = []
    for i in range(num_trajs):
        if i + 1 < len(legend_labels):
            traj_labels.append(legend_labels[i + 1])
        elif i == 0:
            traj_labels.append(_default_pred_legend_label(traj_field))
        else:
            traj_labels.append(f"Trajectory {i + 1}")
    return gt_label, traj_labels


def _gt_bev_style(line_color: str = _DEFAULT_GT_COLOR) -> Dict[str, Any]:
    return {
        "fill_color": line_color,
        "fill_color_alpha": 0.0,
        "line_color": line_color,
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 7,
        "marker_edge_color": "black",
        "zorder": 3,
    }


def _traj_bev_style(line_color: str, zorder: int = 4) -> Dict[str, Any]:
    return {
        "fill_color": line_color,
        "fill_color_alpha": 0.0,
        "line_color": line_color,
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 7,
        "marker_edge_color": "black",
        "zorder": zorder,
    }


def _camera_gt_style(line_color: str = _DEFAULT_GT_COLOR) -> Dict[str, Any]:
    return {
        "line_color": line_color,
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": None,
        "marker_size": 0,
        "zorder": 2,
    }


def _resolve_bev_axis_limits(
    bev_size_x: Optional[float],
    bev_size_y: Optional[float],
    bev_x_min: Optional[float],
    bev_x_max: Optional[float],
    bev_y_min: Optional[float],
    bev_y_max: Optional[float],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Resolve matplotlib axis limits for BEV.

    Ego frame: x = longitudinal (forward +), y = lateral (left +).
    Plot mapping: matplotlib x = ego y, matplotlib y = ego x (see add_trajectory_to_bev_ax).

    Returns ((xlim_lo, xlim_hi), (ylim_lo, ylim_hi)) or None to use default _configure_bev_ax.
    """
    has_explicit_x = bev_x_min is not None or bev_x_max is not None
    has_explicit_y = bev_y_min is not None or bev_y_max is not None
    has_symmetric = bev_size_x is not None and bev_size_y is not None
    if not has_explicit_x and not has_explicit_y and not has_symmetric:
        return None

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]

    if bev_y_min is not None and bev_y_max is not None:
        xlim = (bev_y_min, bev_y_max)
    elif bev_size_y is not None:
        xlim = (-bev_size_y / 2.0, bev_size_y / 2.0)
    else:
        xlim = (-margin_y / 2.0, margin_y / 2.0)

    if bev_x_min is not None and bev_x_max is not None:
        ylim = (bev_x_min, bev_x_max)
    elif bev_size_x is not None:
        ylim = (-bev_size_x / 2.0, bev_size_x / 2.0)
    else:
        ylim = (-margin_x / 2.0, margin_x / 2.0)

    if xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
        raise ValueError(
            f"Invalid BEV limits: xlim={xlim}, ylim={ylim}. Each min must be < max."
        )
    return xlim, ylim


def _apply_bev_limits(ax: plt.Axes, limits: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
    xlim, ylim = limits
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_xaxis()


def _camera_traj_style(line_color: str, zorder: int = 3) -> Dict[str, Any]:
    return {
        "line_color": line_color,
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": None,
        "marker_size": 0,
        "zorder": zorder,
    }


_FRONT_CAMERA_FIGSIZE = (12.0, 7.0)
_FRONT_CAMERA_DPI = 180
_SIDE_PANEL_DARKEN_FACTOR = 0.8


def _render_camera_overlay_rgb(
    camera: Any,
    human_traj: Any,
    pred_traj: Any,
    gt_color: str,
    pred_color: str,
    draw_camera_traj: bool = False,
    figsize: Tuple[float, float] = _FRONT_CAMERA_FIGSIZE,
    dpi: int = _FRONT_CAMERA_DPI,
) -> np.ndarray:
    """Render one camera view; optionally overlay GT + pred trajectories. Return RGB uint8 array."""
    fig, ax = plt.subplots(figsize=figsize)
    add_camera_ax(ax, camera)
    if draw_camera_traj:
        add_trajectory_to_camera_ax(ax, camera, human_traj, _camera_gt_style(gt_color))
        add_trajectory_to_camera_ax(ax, camera, pred_traj, _camera_traj_style(pred_color))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _resize_rgb_panel(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if target_width < 1:
        target_width = 1
    if target_height < 1:
        target_height = 1
    pil = Image.fromarray(img)
    return np.asarray(pil.resize((target_width, target_height), Image.Resampling.LANCZOS))


def _validate_width_ratio_range(start: float, end: float, label: str) -> None:
    if not (0.0 <= start < end <= 1.0):
        raise ValueError(f"{label} must satisfy 0 <= start < end <= 1, got start={start}, end={end}")


def _darken_rgb_uint8(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale all pixel values by factor (e.g. 0.8 to manually darken side panels)."""
    return np.clip(img.astype(np.float32) * float(factor), 0.0, 255.0).astype(np.uint8)


def _crop_rgb_width_range(img: np.ndarray, start_ratio: float, end_ratio: float) -> np.ndarray:
    """
    Crop image columns [start_ratio * W, end_ratio * W) left-to-right (pixel x-axis).
    """
    h, w = img.shape[:2]
    x0 = int(round(w * float(start_ratio)))
    x1 = int(round(w * float(end_ratio)))
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    return img[:, x0:x1]


def _concat_side_front_montage(
    left_rgb: np.ndarray,
    center_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_width_ratio: float,
    right_width_ratio: float,
    left_crop_start: float = 0.0,
    left_crop_end: float = 1.0,
    right_crop_start: float = 0.0,
    right_crop_end: float = 1.0,
) -> np.ndarray:
    """
    Concatenate [left | center | right].
    Side panels are first trajectory-projected, then width-cropped, then resized.
    """
    left_rgb = _crop_rgb_width_range(left_rgb, left_crop_start, left_crop_end)
    right_rgb = _crop_rgb_width_range(right_rgb, right_crop_start, right_crop_end)
    left_rgb = _darken_rgb_uint8(left_rgb, _SIDE_PANEL_DARKEN_FACTOR)
    right_rgb = _darken_rgb_uint8(right_rgb, _SIDE_PANEL_DARKEN_FACTOR)
    h_center = int(center_rgb.shape[0])
    w_center = int(center_rgb.shape[1])
    w_left = max(1, int(round(w_center * left_width_ratio)))
    w_right = max(1, int(round(w_center * right_width_ratio)))
    left_r = _resize_rgb_panel(left_rgb, w_left, h_center)
    right_r = _resize_rgb_panel(right_rgb, w_right, h_center)
    return np.concatenate([left_r, center_rgb, right_r], axis=1)


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
    """
    Map future-frame ego-local GT boxes to current-frame ego-local coordinates.
    Uses GT ego global poses at both timestamps.
    """
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

    global_array = np.asarray(global_poses, dtype=np.float64)
    local_array = convert_absolute_to_relative_se2_array(current_origin, global_array)

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
    """Keep only annotation rows whose track_token is in `allowed_track_tokens`."""
    if not allowed_track_tokens or len(annotations.track_tokens) == 0:
        return _empty_annotations()

    keep_indices = [
        idx for idx, track_token in enumerate(annotations.track_tokens) if track_token in allowed_track_tokens
    ]
    if not keep_indices:
        return _empty_annotations()

    boxes = np.asarray([annotations.boxes[idx] for idx in keep_indices], dtype=np.float32)
    names = [annotations.names[idx] for idx in keep_indices]
    velocity_3d = np.asarray([annotations.velocity_3d[idx] for idx in keep_indices], dtype=np.float32)
    instance_tokens = [annotations.instance_tokens[idx] for idx in keep_indices]
    track_tokens = [annotations.track_tokens[idx] for idx in keep_indices]
    return Annotations(
        boxes=boxes,
        names=names,
        velocity_3d=velocity_3d,
        instance_tokens=instance_tokens,
        track_tokens=track_tokens,
    )


def _local_box_center_to_global_xy(box: np.ndarray, ego_pose_global: StateSE2) -> np.ndarray:
    """Convert one ego-local box center to global XY."""
    local_center = rotate_state_se2(
        StateSE2(float(box[0]), float(box[1]), float(box[6])),
        angle=ego_pose_global.heading,
    )
    return np.array(
        [local_center.x + ego_pose_global.x, local_center.y + ego_pose_global.y],
        dtype=np.float64,
    )


def _compute_static_track_tokens(
    scene: Scene,
    current_frame_idx: int,
    current_track_tokens: set[str],
    ref_future_step: int = 8,
    static_l2_thresh_m: float =2.0,
) -> set[str]:
    """
    Identify static neighboring agents:
    compare global L2 displacement between current frame and future `ref_future_step` frame.
    """
    max_future_idx = len(scene.frames) - 1
    ref_future_idx = min(current_frame_idx + ref_future_step, max_future_idx)
    if ref_future_idx <= current_frame_idx:
        return set()

    current_frame = scene.frames[current_frame_idx]
    ref_future_frame = scene.frames[ref_future_idx]
    current_pose = _global_pose_from_frame(current_frame)
    ref_pose = _global_pose_from_frame(ref_future_frame)

    current_rows = {
        track_token: idx for idx, track_token in enumerate(current_frame.annotations.track_tokens)
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
        current_xy_global = _local_box_center_to_global_xy(current_frame.annotations.boxes[current_idx], current_pose)
        ref_xy_global = _local_box_center_to_global_xy(ref_future_frame.annotations.boxes[ref_idx], ref_pose)
        if float(np.linalg.norm(ref_xy_global - current_xy_global)) < static_l2_thresh_m:
            static_tracks.add(track_token)

    return static_tracks


def _slice_trajectory(trajectory: Trajectory, num_poses: int) -> Trajectory:
    poses = trajectory.poses[:num_poses]
    sampling = TrajectorySampling(
        num_poses=int(poses.shape[0]),
        interval_length=trajectory.trajectory_sampling.interval_length,
    )
    return Trajectory(poses, sampling)


def _fig_to_pil(fig: plt.Figure, dpi: int = 180) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).copy()
    buf.close()
    return image


def _gif_fixed_ego_box_style(pred_color: str) -> Dict[str, Any]:
    """Transparent ego box at planning-time origin, fixed on every GIF frame."""
    style = dict(AGENT_CONFIG[TrackedObjectType.EGO])
    style["fill_color"] = pred_color
    style["line_color"] = pred_color
    style["fill_color_alpha"] = 0.25
    style["line_color_alpha"] = 0.55
    style["zorder"] = 2
    return style


def _gif_predicted_ego_box_style() -> Dict[str, Any]:
    """Solid ego box at predicted pose (same opacity as static BEV ego footprint)."""
    return dict(AGENT_CONFIG[TrackedObjectType.EGO])


def _add_gif_bev_background(
    ax: plt.Axes,
    map_api: Any,
    current_frame: Frame,
    pred_color: str,
    static_background_annotations: Optional[Annotations] = None,
) -> None:
    """Map + light ego at origin + optional static neighbors."""
    add_map_to_bev_ax(ax, map_api, _global_pose_from_frame(current_frame))
    if static_background_annotations is not None and len(static_background_annotations.boxes) > 0:
        add_annotations_to_bev_ax(ax, static_background_annotations, add_ego=False)
    add_ego_box_at_local_pose_to_bev_ax(
        ax,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        _gif_fixed_ego_box_style(pred_color),
        add_heading=False,
    )


def _save_bev_future_agents_gif(
    scene: Scene,
    current_frame: Frame,
    pred_traj: Trajectory,
    output_path: Path,
    *,
    pred_color: str,
    bev_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    gif_steps: int,
    gif_fps: float,
    draw_other_agents: bool,
) -> Path:
    """
    Render GIF with fixed map + transparent ego-at-origin background.
    Neighbors are constrained as:
    1) only agents already visible at current frame can be drawn in future frames;
    2) agents with small displacement (<4m) between current frame and future step 4 are
       treated as static background and removed from dynamic future rendering.
    The output sequence includes current frame (step=0) + future steps.
    """
    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    max_future_idx = len(scene.frames) - 1
    num_future_steps = min(
        gif_steps,
        int(pred_traj.poses.shape[0]),
        max(0, max_future_idx - current_frame_idx),
    )
    if num_future_steps < 1:
        raise ValueError(
            "Cannot render BEV GIF: need at least one future step with prediction poses and scene frames."
        )

    pred_style = _traj_bev_style(pred_color)
    duration_ms = int(round(1000.0 / max(gif_fps, 0.1)))
    images: List[Image.Image] = []
    current_track_tokens = set(current_frame.annotations.track_tokens)
    static_track_tokens: set[str] = set()
    static_background_annotations: Optional[Annotations] = None
    if draw_other_agents and current_track_tokens:
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

    # Frame 0 is current frame, then future frames 1..num_future_steps.
    for step in range(0, num_future_steps + 1):
        future_idx = current_frame_idx + step
        future_frame = scene.frames[future_idx]
        pred_partial = _slice_trajectory(pred_traj, step) if step > 0 else None

        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_facecolor("white")
        _add_gif_bev_background(
            ax,
            scene.map_api,
            current_frame,
            pred_color,
            static_background_annotations=static_background_annotations,
        )
        if draw_other_agents:
            warped_anns = _transform_annotations_to_current_ego(future_frame, current_frame)
            warped_dynamic_anns = _filter_annotations_by_track_tokens(warped_anns, dynamic_track_tokens)
            add_annotations_to_bev_ax(ax, warped_dynamic_anns, add_ego=False)
        if pred_partial is not None:
            add_trajectory_to_bev_ax(ax, pred_partial, pred_style, draw_ego_box_at_end=False)
        if step == 0:
            # Current frame: draw solid ego on top of transparent background ego.
            add_ego_box_at_local_pose_to_bev_ax(
                ax,
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                _gif_predicted_ego_box_style(),
                add_heading=True,
            )
        elif pred_partial is not None and pred_partial.poses.shape[0] > 0:
            add_ego_box_at_local_pose_to_bev_ax(
                ax,
                pred_partial.poses[-1],
                _gif_predicted_ego_box_style(),
                add_heading=True,
            )
        if bev_limits is not None:
            _apply_bev_limits(ax, bev_limits)
        else:
            _configure_bev_ax(ax)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        images.append(_fig_to_pil(fig, dpi=180))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def _save_rgb_array(path: Path, rgb: np.ndarray, mpl_fmt: str) -> None:
    """Save HxWx3 uint8 RGB to jpg/png/pdf via PIL or matplotlib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if mpl_fmt == "pdf":
        h, w = rgb.shape[:2]
        fig, ax = plt.subplots(figsize=(w / _FRONT_CAMERA_DPI, h / _FRONT_CAMERA_DPI), dpi=_FRONT_CAMERA_DPI)
        ax.imshow(rgb)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(path, format="pdf", dpi=_FRONT_CAMERA_DPI, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return
    pil_fmt = "JPEG" if mpl_fmt == "jpeg" else mpl_fmt.upper()
    Image.fromarray(rgb).save(path, format=pil_fmt)


def render_top1_images(
    scene: Scene,
    detail_payload: Dict[str, Any],
    output_dir: Path,
    image_prefix: str,
    interval_length: float,
    traj_field: str = "pred_traj_local",
    legend_labels: Optional[Sequence[str]] = None,
    traj_colors: Optional[Sequence[str]] = None,
    bev_size_x: float | None = None,
    bev_size_y: float | None = None,
    bev_x_min: float | None = None,
    bev_x_max: float | None = None,
    bev_y_min: float | None = None,
    bev_y_max: float | None = None,
    concat_side_front: bool = False,
    left_front_width_ratio: float = 0.3,
    right_front_width_ratio: float = 0.3,
    left_front_width_ratio_start: float = 0.0,
    left_front_width_ratio_end: float = 1.0,
    right_front_width_ratio_start: float = 0.0,
    right_front_width_ratio_end: float = 1.0,
    draw_camera_traj: bool = False,
    output_formats: List[str] | None = None,
    render_bev_gif: bool = False,
    gif_steps: int = 8,
    gif_fps: float = 4.0,
    draw_other_agents: bool = True,
) -> Tuple[List[Path], List[Path], Optional[Path]]:
    poses = _traj_poses_from_payload(detail_payload, traj_field)

    sampling = TrajectorySampling(num_poses=int(poses.shape[0]), interval_length=interval_length)
    pred_traj = _trajectory_from_detail(traj_field, detail_payload, sampling)

    labels_in = list(legend_labels) if legend_labels else []
    colors_in = list(traj_colors) if traj_colors else []
    gt_label, traj_labels = _resolve_legend_labels(labels_in, traj_field, num_trajs=1)
    gt_color = _DEFAULT_GT_COLOR
    pred_color = _resolve_traj_color(0, colors_in)

    n_future = min(scene.scene_metadata.num_future_frames, int(poses.shape[0]))
    human_traj = scene.get_future_trajectory(num_trajectory_frames=n_future)

    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[current_frame_idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    formats = output_formats if output_formats is not None else ["pdf"]
    bev_limits = _resolve_bev_axis_limits(
        bev_size_x, bev_size_y, bev_x_min, bev_x_max, bev_y_min, bev_y_max
    )

    bev_paths: List[Path] = []
    if formats:
        camera = _pick_front_camera(frame)
        if camera is None:
            raise ValueError("No camera image is available in this scene (cam_f0/cam_r0/... all missing).")

        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_facecolor("white")
        add_configured_bev_on_ax(ax, scene.map_api, frame)
        gt_style = _gt_bev_style(gt_color)
        pred_style = _traj_bev_style(pred_color)
        add_trajectory_to_bev_ax(ax, human_traj, gt_style, draw_ego_box_at_end=True)
        add_trajectory_to_bev_ax(ax, pred_traj, pred_style, draw_ego_box_at_end=True)
        if bev_limits is not None:
            _apply_bev_limits(ax, bev_limits)
        else:
            _configure_bev_ax(ax)
        ax.axis("off")
        legend_scale = 1.5
        try:
            base_fs = float(mpl.rcParams["font.size"])
        except (TypeError, ValueError):
            base_fs = 10.0
        legend_fs = base_fs * legend_scale
        lw_legend = 2.0 * legend_scale
        ax.legend(
            handles=[
                Line2D([0], [0], color=gt_color, lw=lw_legend, label=gt_label),
                Line2D([0], [0], color=pred_color, lw=lw_legend, label=traj_labels[0]),
            ],
            loc="upper right",
            frameon=True,
            fontsize=legend_fs,
            handlelength=2.0 * legend_scale,
            handletextpad=0.8 * legend_scale,
            borderpad=0.4 * legend_scale,
            labelspacing=0.5 * legend_scale,
            borderaxespad=0.5 * legend_scale,
        )
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for fmt in formats:
            mpl_fmt, suffix = _matplotlib_format_and_suffix(fmt)
            path = output_dir / f"{image_prefix}_BEV{suffix}"
            fig.savefig(path, format=mpl_fmt, dpi=180, bbox_inches="tight", pad_inches=0)
            bev_paths.append(path)
        plt.close(fig)

    gif_path: Optional[Path] = None
    if render_bev_gif:
        gif_path = output_dir / f"{image_prefix}_BEV_future_agents.gif"
        _save_bev_future_agents_gif(
            scene=scene,
            current_frame=frame,
            pred_traj=pred_traj,
            output_path=gif_path,
            pred_color=pred_color,
            bev_limits=bev_limits,
            gif_steps=gif_steps,
            gif_fps=gif_fps,
            draw_other_agents=draw_other_agents,
        )

    front_paths: List[Path] = []
    if not formats:
        return bev_paths, front_paths, gif_path

    camera = _pick_front_camera(frame)
    if camera is None:
        raise ValueError("No camera image is available in this scene (cam_f0/cam_r0/... all missing).")

    if concat_side_front:
        left_cam = _pick_left_front_camera(frame)
        right_cam = _pick_right_front_camera(frame)
        if left_cam is None or right_cam is None:
            missing = []
            if left_cam is None:
                missing.append("left-front (cam_l0/cam_l1/cam_l2)")
            if right_cam is None:
                missing.append("right-front (cam_r0/cam_r1/cam_r2)")
            raise ValueError(
                f"--concat-side-front requires side camera images; missing: {', '.join(missing)}"
            )
        center_rgb = _render_camera_overlay_rgb(
            camera, human_traj, pred_traj, gt_color, pred_color, draw_camera_traj=draw_camera_traj
        )
        left_rgb = _render_camera_overlay_rgb(
            left_cam, human_traj, pred_traj, gt_color, pred_color, draw_camera_traj=draw_camera_traj
        )
        right_rgb = _render_camera_overlay_rgb(
            right_cam, human_traj, pred_traj, gt_color, pred_color, draw_camera_traj=draw_camera_traj
        )
        montage_rgb = _concat_side_front_montage(
            left_rgb,
            center_rgb,
            right_rgb,
            left_front_width_ratio,
            right_front_width_ratio,
            left_crop_start=left_front_width_ratio_start,
            left_crop_end=left_front_width_ratio_end,
            right_crop_start=right_front_width_ratio_start,
            right_crop_end=right_front_width_ratio_end,
        )
        for fmt in formats:
            mpl_fmt, suffix = _matplotlib_format_and_suffix(fmt)
            path = output_dir / f"{image_prefix}_front_camera{suffix}"
            _save_rgb_array(path, montage_rgb, mpl_fmt)
            front_paths.append(path)
    else:
        fig, ax = plt.subplots(figsize=_FRONT_CAMERA_FIGSIZE)
        add_camera_ax(ax, camera)
        if draw_camera_traj:
            add_trajectory_to_camera_ax(ax, camera, human_traj, _camera_gt_style(gt_color))
            add_trajectory_to_camera_ax(ax, camera, pred_traj, _camera_traj_style(pred_color))
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for fmt in formats:
            mpl_fmt, suffix = _matplotlib_format_and_suffix(fmt)
            path = output_dir / f"{image_prefix}_front_camera{suffix}"
            fig.savefig(path, format=mpl_fmt, dpi=_FRONT_CAMERA_DPI, bbox_inches="tight", pad_inches=0)
            front_paths.append(path)
        plt.close(fig)

    return bev_paths, front_paths, gif_path


def _normalize_model_name_to_prefix(model_name: str) -> str:
    """
    Convert model names like:
      stage2 -> stage_2
      Stage-2 -> stage_2
      stage_2 -> stage_2
    """
    s = model_name.strip()
    if not s:
        raise ValueError("model_name cannot be empty")
    # Keep explicit segment boundaries provided by user/model names.
    # This avoids splitting token-like suffixes such as f7d40806c7045d54.
    if "_" in s or "-" in s:
        s = s.replace("-", "_")
        s = re.sub(r"_+", "_", s)
        return s.lower().strip("_")

    # Only auto-insert underscore for the simple "<letters><digits>" pattern.
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", s)
    if m:
        return f"{m.group(1).lower()}_{m.group(2)}"
    return s.lower()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render top-1 trajectory BEV and front-camera overlay from one detail.json. "
            "See module docstring at top of this file for path layout and examples."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python navsim/evaluate/render_top1_from_detail.py \\\n"
            "    --detail-json .../proposal_top1/detail.json \\\n"
            "    --data-path /path/navsim_logs --sensor-blobs-path /path/sensor_blobs \\\n"
            "    --output-dir /tmp/out --model-name stage_2 \\\n"
            "    --legend-labels 'Human Expert,ChainFlow' --traj-colors FF0000\n"
        ),
    )
    parser.add_argument(
        "--detail-json",
        type=Path,
        required=True,
        help="Path to proposal_top1/detail.json",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Navsim logs root (directory containing *.pkl logs)",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        type=Path,
        required=True,
        help="Sensor blobs root used by SceneLoader",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where output figures are saved",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default=None,
        help="Output image name prefix, e.g. stage_2 (higher priority than --model-name)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name to auto-generate prefix, e.g. stage2 -> stage_2",
    )
    parser.add_argument(
        "--traj-field",
        type=str,
        default="pred_traj_local",
        help=(
            "Trajectory key in detail.json to render (default: pred_traj_local). "
            "Common local-frame keys: pred_traj_local, simulated_traj_local."
        ),
    )
    parser.add_argument(
        "--legend-labels",
        type=str,
        default=None,
        help=(
            "Comma-separated legend names: first = GT, then each trajectory in order. "
            "Example: 'Human Expert,ChainFlow'. If omitted, uses PDM defaults "
            "(Human Expert + field-based model label)."
        ),
    )
    parser.add_argument(
        "--traj-colors",
        type=str,
        default=None,
        help=(
            "Comma-separated trajectory colors as RRGGBB hex (optional #). "
            "Skips GT; first value colors the first non-GT trajectory, etc. "
            "Example: 'FF0000,0000FF' -> red then blue. If omitted, GT stays green "
            "and trajectories use tab:orange, tab:blue, ..."
        ),
    )
    parser.add_argument(
        "--interval-length",
        type=float,
        default=0.1,
        help="Trajectory sampling interval length for --traj-field (default: 0.1)",
    )
    parser.add_argument(
        "--bev-size-x",
        type=float,
        default=None,
        help="Symmetric BEV span along ego-x (longitudinal, m). With --bev-size-y 80 → y in [-40, 40].",
    )
    parser.add_argument(
        "--bev-size-y",
        type=float,
        default=None,
        help="Symmetric BEV span along ego-y (lateral, m). With --bev-size-x 80 → x in [-40, 40].",
    )
    parser.add_argument(
        "--bev-x-min",
        type=float,
        default=None,
        help="Ego-x min (m), rear; use with --bev-x-max. Example: -20 for 20 m behind ego.",
    )
    parser.add_argument(
        "--bev-x-max",
        type=float,
        default=None,
        help="Ego-x max (m), forward; use with --bev-x-min. Example: 40 for 40 m ahead.",
    )
    parser.add_argument(
        "--bev-y-min",
        type=float,
        default=None,
        help="Ego-y min (m), right side in plot; use with --bev-y-max.",
    )
    parser.add_argument(
        "--bev-y-max",
        type=float,
        default=None,
        help="Ego-y max (m), left side in plot; use with --bev-y-max.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        type=str,
        default="pdf",
        help=(
            "Output format(s), comma-separated: pdf, jpg, png (e.g. pdf,jpg). "
            "Use 'none' to skip static BEV/camera images (e.g. with --render-bev-gif only). Default: pdf."
        ),
    )
    parser.add_argument(
        "--draw-camera-traj",
        action="store_true",
        help=(
            "Overlay GT and predicted trajectories on front (and side) camera images "
            "(default: off, camera image only)."
        ),
    )
    parser.add_argument(
        "--concat-side-front",
        action="store_true",
        help=(
            "Concatenate left-front and right-front camera overlays with front camera "
            "into one _front_camera image (default: off, front only)."
        ),
    )
    parser.add_argument(
        "--left-front-width-ratio",
        type=float,
        default=0.3,
        help=(
            "Left panel width as fraction of front panel width after cropping, when "
            "--concat-side-front (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--right-front-width-ratio",
        type=float,
        default=0.3,
        help=(
            "Right panel width as fraction of front panel width after cropping, when "
            "--concat-side-front (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--left-front-width-ratio-start",
        type=float,
        default=0.0,
        help=(
            "Left-front horizontal crop start (0-1, left edge), left-to-right pixel x-axis. "
            "Use with --left-front-width-ratio-end; e.g. 0.3 and 0.7 keep columns [0.3W, 0.7W)."
        ),
    )
    parser.add_argument(
        "--left-front-width-ratio-end",
        type=float,
        default=1.0,
        help="Left-front horizontal crop end (0-1, exclusive-style upper bound via rounding).",
    )
    parser.add_argument(
        "--right-front-width-ratio-start",
        type=float,
        default=0.0,
        help=(
            "Right-front horizontal crop start (0-1, left edge), same convention as left-front. "
            "E.g. 0.3 and 0.7 keep columns [0.3W, 0.7W)."
        ),
    )
    parser.add_argument(
        "--right-front-width-ratio-end",
        type=float,
        default=1.0,
        help="Right-front horizontal crop end (0-1).",
    )
    parser.add_argument(
        "--render-bev-gif",
        action="store_true",
        help=(
            "Also render an animated BEV GIF: fixed current-frame background, growing predicted "
            "ego trajectory, and future other-agent boxes warped to current ego coordinates."
        ),
    )
    parser.add_argument(
        "--gif-steps",
        type=int,
        default=8,
        help="Number of GIF frames (future steps). Only used with --render-bev-gif (default: 8).",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=4.0,
        help="GIF playback rate in frames per second. Only used with --render-bev-gif (default: 4).",
    )
    parser.add_argument(
        "--draw-other-agents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In BEV GIF, overlay other agents from future GT frames (warped to current ego). "
            "Only used with --render-bev-gif (default: true)."
        ),
    )
    args = parser.parse_args()
    if (args.bev_size_x is None) != (args.bev_size_y is None):
        raise ValueError("Please set both --bev-size-x and --bev-size-y together.")
    if args.bev_size_x is not None and (args.bev_size_x <= 0 or args.bev_size_y <= 0):
        raise ValueError("--bev-size-x and --bev-size-y must be positive numbers.")
    if (args.bev_x_min is None) != (args.bev_x_max is None):
        raise ValueError("Please set both --bev-x-min and --bev-x-max together.")
    if (args.bev_y_min is None) != (args.bev_y_max is None):
        raise ValueError("Please set both --bev-y-min and --bev-y-max together.")
    if args.left_front_width_ratio <= 0 or args.right_front_width_ratio <= 0:
        raise ValueError("--left-front-width-ratio and --right-front-width-ratio must be positive.")
    _validate_width_ratio_range(
        args.left_front_width_ratio_start,
        args.left_front_width_ratio_end,
        "--left-front-width-ratio-start/end",
    )
    _validate_width_ratio_range(
        args.right_front_width_ratio_start,
        args.right_front_width_ratio_end,
        "--right-front-width-ratio-start/end",
    )

    detail_json_path = args.detail_json.expanduser().resolve()
    if not detail_json_path.is_file():
        raise FileNotFoundError(f"detail.json not found: {detail_json_path}")

    exp_root, log_name, token = _parse_detail_json_path(detail_json_path)
    print(f"exp_root={exp_root}")
    print(f"log_name={log_name}")
    print(f"token={token}")

    traj_field = args.traj_field.strip()
    if not traj_field:
        raise ValueError("--traj-field cannot be empty")

    detail_payload = _load_detail_payload(detail_json_path)
    poses = _traj_poses_from_payload(detail_payload, traj_field)
    num_future_frames = int(poses.shape[0])
    if args.render_bev_gif:
        num_future_frames = max(num_future_frames, int(args.gif_steps))
    print(f"traj_field={traj_field} ({num_future_frames} poses for scene load)")
    scene = _build_scene(
        data_path=args.data_path,
        sensor_blobs_path=args.sensor_blobs_path,
        token=token,
        log_name=log_name,
        num_future_frames=num_future_frames,
    )

    image_prefix = args.image_prefix
    if image_prefix is None:
        if args.model_name is None:
            raise ValueError("Please provide either --image-prefix or --model-name")
        image_prefix = _normalize_model_name_to_prefix(args.model_name)

    legend_labels = _parse_comma_list(args.legend_labels)
    traj_colors = _parse_traj_colors(args.traj_colors)

    if args.render_bev_gif and args.gif_steps < 1:
        raise ValueError("--gif-steps must be >= 1 when --render-bev-gif is set.")
    if args.render_bev_gif and args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive when --render-bev-gif is set.")

    output_formats = _parse_output_formats(args.output_format)
    if not output_formats and not args.render_bev_gif:
        raise ValueError("Use --format none only together with --render-bev-gif.")

    bev_paths, front_paths, gif_path = render_top1_images(
        scene=scene,
        detail_payload=detail_payload,
        output_dir=args.output_dir.expanduser().resolve(),
        image_prefix=image_prefix,
        interval_length=args.interval_length,
        traj_field=traj_field,
        legend_labels=legend_labels,
        traj_colors=traj_colors,
        bev_size_x=args.bev_size_x,
        bev_size_y=args.bev_size_y,
        bev_x_min=args.bev_x_min,
        bev_x_max=args.bev_x_max,
        bev_y_min=args.bev_y_min,
        bev_y_max=args.bev_y_max,
        concat_side_front=bool(args.concat_side_front),
        left_front_width_ratio=float(args.left_front_width_ratio),
        right_front_width_ratio=float(args.right_front_width_ratio),
        left_front_width_ratio_start=float(args.left_front_width_ratio_start),
        left_front_width_ratio_end=float(args.left_front_width_ratio_end),
        right_front_width_ratio_start=float(args.right_front_width_ratio_start),
        right_front_width_ratio_end=float(args.right_front_width_ratio_end),
        draw_camera_traj=bool(args.draw_camera_traj),
        output_formats=output_formats,
        render_bev_gif=bool(args.render_bev_gif),
        gif_steps=int(args.gif_steps),
        gif_fps=float(args.gif_fps),
        draw_other_agents=bool(args.draw_other_agents),
    )
    for p in bev_paths:
        print(f"Saved BEV: {p}")
    if gif_path is not None:
        print(f"Saved BEV GIF: {gif_path}")
    for p in front_paths:
        print(f"Saved front camera: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
