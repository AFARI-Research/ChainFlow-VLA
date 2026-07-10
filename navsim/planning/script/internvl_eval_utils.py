import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _output_directory_log_name(log_file: str) -> str:
    if not log_file:
        return "unknown_log"
    name = Path(log_file).name
    for suffix in (".pkl", ".pickle"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _resolve_per_token_json_root(cfg: DictConfig) -> Path:
    root = cfg.get("failure_cases_output_dir", None)
    if root:
        return Path(root)
    return Path(cfg.output_dir) / "per_token_json"


def _scorer_config_to_dict(scorer: PDMScorer) -> Dict[str, Any]:
    config = scorer._config
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return {key: value for key, value in vars(config).items() if not key.startswith("_")}
    return {"value": _json_default(config)}


def save_per_token_json(
    cfg: DictConfig,
    token: str,
    log_file: str,
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    pdm_result: PDMResults,
    details: Dict[str, Any],
    scorer: PDMScorer,
) -> Path:
    """Save compact per-token PDM details for InternVL evaluation."""

    exp_name = str(cfg.get("experiment_name", "internvl_agent") or "internvl_agent")
    log_name = _output_directory_log_name(log_file)
    save_dir = _resolve_per_token_json_root(cfg) / exp_name / log_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{token}.json"

    output: Dict[str, Any] = {
        "token": token,
        "log_file": log_file,
        "experiment_name": exp_name,
        "pdm_score": asdict(pdm_result),
        "pred_idx": details.get("pred_idx"),
        "pred_traj_local": np.asarray(model_trajectory.poses).tolist(),
        "reference_traj": details.get("pdm_states"),
        "pred_traj": details.get("pred_states"),
        "simulated_traj": details.get("simulated_states_pred"),
        "raw_details": details,
        "scorer_config": _scorer_config_to_dict(scorer),
        "initial_timestamp_us": int(metric_cache.ego_state.time_point.time_us),
        "route_lane_ids": [str(route_id) for route_id in metric_cache.route_lane_ids],
    }

    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)
    os.replace(tmp_path, save_path)
    return save_path
