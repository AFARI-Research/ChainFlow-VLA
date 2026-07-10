from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Scene, Trajectory
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax, add_trajectory_to_camera_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG


def _trajectory_config(name: str) -> Dict[str, Any]:
    if name in TRAJECTORY_CONFIG:
        return TRAJECTORY_CONFIG[name]
    if name == "agent_sim":
        return {
            **TRAJECTORY_CONFIG["agent"],
            "line_color": "tab:orange",
            "fill_color": "tab:orange",
            "line_style": "--",
        }
    return TRAJECTORY_CONFIG["agent"]


def _save_figure(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return str(output_path)


def _trajectory_from_detail(detail_key: str, pdm_details: Dict[str, Any], sampling: TrajectorySampling) -> Trajectory:
    poses = np.asarray(pdm_details[detail_key], dtype=np.float32)
    detail_sampling = TrajectorySampling(
        num_poses=len(poses),
        interval_length=sampling.interval_length,
    )
    return Trajectory(poses, detail_sampling)


def _configure_bev_ax(ax: plt.Axes) -> plt.Axes:
    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_aspect("equal")
    ax.set_xlim(-margin_y / 2, margin_y / 2)
    ax.set_ylim(-margin_x / 2, margin_x / 2)
    ax.invert_xaxis()
    return ax


def _front_camera_trajectory_config() -> Dict[str, Any]:
    """Front camera: single polyline in red, vertices connected, no scatter markers."""
    return {
        "line_color": "red",
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": None,
        "marker_size": 0,
        "zorder": 3,
    }


def _front_camera_gt_trajectory_config() -> Dict[str, Any]:
    """GT / human future on front camera: same hues as BEV human, polyline only."""
    human = _trajectory_config("human")
    return {
        "line_color": human["line_color"],
        "line_color_alpha": human["line_color_alpha"],
        "line_width": 2.0,
        "line_style": "-",
        "marker": None,
        "marker_size": 0,
        "zorder": 2,
    }


def render_eval_visualizations(
    scene: Scene,
    model_trajectory: Trajectory,
    pdm_details: Dict[str, Any],
    output_dir: Path,
    file_stem: str,
) -> Dict[str, str]:
    try:
        current_frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[current_frame_idx]

        camera = frame.cameras.cam_f0
        if camera.image is None:
            return {}

        pred_traj = _trajectory_from_detail(
            "pred_traj_local",
            pdm_details,
            model_trajectory.trajectory_sampling,
        )
        simulated_traj = _trajectory_from_detail(
            "simulated_traj_local",
            pdm_details,
            model_trajectory.trajectory_sampling,
        )
        human_traj = scene.get_future_trajectory(num_trajectory_frames=model_trajectory.trajectory_sampling.num_poses)

        paths: Dict[str, str] = {}

        pred_camera_path = output_dir / f"{file_stem}_pred_cam_f0.jpg"
        fig, ax = plt.subplots(figsize=(12, 7))
        add_camera_ax(ax, camera)
        add_trajectory_to_camera_ax(ax, camera, human_traj, _front_camera_gt_trajectory_config())
        add_trajectory_to_camera_ax(ax, camera, pred_traj, _front_camera_trajectory_config())
        ax.axis("off")
        paths["pred_front_camera_path"] = _save_figure(fig, pred_camera_path)

        sim_camera_path = output_dir / f"{file_stem}_sim_cam_f0.jpg"
        fig, ax = plt.subplots(figsize=(12, 7))
        add_camera_ax(ax, camera)
        add_trajectory_to_camera_ax(ax, camera, human_traj, _front_camera_gt_trajectory_config())
        add_trajectory_to_camera_ax(ax, camera, simulated_traj, _front_camera_trajectory_config())
        ax.axis("off")
        paths["simulated_front_camera_path"] = _save_figure(fig, sim_camera_path)

        bev_path = output_dir / f"{file_stem}_bev.jpg"
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_facecolor("white")
        add_configured_bev_on_ax(ax, scene.map_api, frame)
        add_trajectory_to_bev_ax(ax, human_traj, _trajectory_config("human"))
        add_trajectory_to_bev_ax(ax, pred_traj, {
            "fill_color": "tab:orange",
            "fill_color_alpha": 1.0,
            "line_color": "tab:orange",
            "line_color_alpha": 1.0,
            "line_width": 2.0,
            "line_style": "-",
            "marker": "o",
            "marker_size": 5,
            "marker_edge_color": "black",
            "zorder": 3,
        })
        _configure_bev_ax(ax)
        paths["bev_path"] = _save_figure(fig, bev_path)

        return paths
    except Exception:
        plt.close("all")
        raise


def render_all_proposals_front_camera(
    scene: Scene,
    proposal_details: Dict[str, Any],
    output_dir: Path,
    file_stem: str,
) -> str:
    try:
        current_frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[current_frame_idx]

        camera = frame.cameras.cam_f0
        if camera.image is None:
            return ""

        proposals = proposal_details.get("proposals", [])
        if not proposals:
            return ""

        scores = np.asarray([proposal["pdm_result"]["score"] for proposal in proposals], dtype=np.float32)
        best_idx = int(np.argmax(scores))
        detail = proposals[best_idx]["detail"]

        n_poses = len(detail["simulated_traj_local"])
        human_traj = scene.get_future_trajectory(num_trajectory_frames=n_poses)

        fig, ax = plt.subplots(figsize=(12, 7))
        add_camera_ax(ax, camera)
        add_trajectory_to_camera_ax(ax, camera, human_traj, _front_camera_gt_trajectory_config())
        trajectory = _trajectory_from_detail(
            "simulated_traj_local",
            detail,
            TrajectorySampling(num_poses=n_poses, interval_length=0.1),
        )
        add_trajectory_to_camera_ax(ax, camera, trajectory, _front_camera_trajectory_config())

        ax.axis("off")
        output_path = output_dir / f"{file_stem}_all64_lqr_sim_cam_f0.jpg"
        return _save_figure(fig, output_path)
    except Exception:
        plt.close("all")
        raise
