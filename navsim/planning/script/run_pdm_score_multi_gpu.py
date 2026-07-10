# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pickle
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"
PDMScoreTask = Dict[str, Any]


def _bool_cfg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def run_pdm_score(args: List[PDMScoreTask]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation.
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info("Starting worker in thread_id=%s, node_id=%s", thread_id, node_id)

    log_names = [str(task["log_file"]) for task in args]
    tokens = [token for task in args for token in task["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    merged_predictions: Dict[str, Any] = {}
    for task in args:
        merged_predictions.update(task["model_predictions"])

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
    )

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(
            "Processing scenario %s / %s in thread_id=%s, node_id=%s",
            idx + 1,
            len(tokens_to_evaluate),
            thread_id,
            node_id,
        )
        scene_dict_list = scene_loader.scene_frames_dicts[token]
        log_name = scene_dict_list[0]["log_name"]
        score_row: Dict[str, Any] = {"token": token, "log_name": log_name, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trajectory = merged_predictions[token]["trajectory"]
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))

        except Exception:
            logger.warning("----------- Agent failed for token %s:", token)
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)

    return pdm_results


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def init_dist_from_torchrun() -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1 or dist_ready():
        return
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def dist_rank() -> int:
    if dist_ready():
        return dist.get_rank()
    return int(os.getenv("RANK", "0"))


def dist_world_size() -> int:
    if dist_ready():
        return dist.get_world_size()
    return int(os.getenv("WORLD_SIZE", "1"))


def broadcast_object(obj: Any, src: int = 0) -> Any:
    if not dist_ready():
        return obj
    payload = [obj if dist_rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _build_inference_dataset(cfg: DictConfig, agent: AbstractAgent) -> Tuple[SceneLoader, DataLoader]:
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    return scene_loader, DataLoader(dataset, **cfg.dataloader.params, shuffle=False)


def _merge_prediction_batches(predictions: Any) -> Dict[str, Any]:
    merged_predictions: Dict[str, Any] = {}
    for prediction_by_token in predictions or []:
        if not isinstance(prediction_by_token, Mapping):
            raise RuntimeError(f"Expected dict from predict_step, got {type(prediction_by_token)!r}")
        merged_predictions.update(prediction_by_token)
    return merged_predictions


def _filter_rank_predictions(
    predictions: Dict[str, Any],
    scene_loader: SceneLoader,
    metric_cache_loader: MetricCacheLoader,
    rank: int,
    world_size: int,
) -> Tuple[Dict[str, Any], List[str]]:
    metric_tokens = set(metric_cache_loader.tokens)
    tokens_to_evaluate = [token for token in scene_loader.tokens if token in metric_tokens]
    rank_tokens = {
        token for idx, token in enumerate(tokens_to_evaluate) if idx % world_size == rank
    }
    return (
        {
            token: prediction
            for token, prediction in predictions.items()
            if token in rank_tokens
        },
        tokens_to_evaluate,
    )


def _build_score_tasks(
    cfg: DictConfig,
    scene_loader: SceneLoader,
    local_predictions: Dict[str, Any],
) -> List[PDMScoreTask]:
    return [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token for token in tokens_list if token in local_predictions],
            "model_predictions": {
                token: local_predictions[token]
                for token in tokens_list
                if token in local_predictions
            },
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
        if any(token in local_predictions for token in tokens_list)
    ]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    """
    # Disable agent-side Ray worker pools in this DDP evaluation entrypoint.
    os.environ["CHAINFLOW_ENABLE_RAY"] = "0"
    init_dist_from_torchrun()
    build_logger(cfg)
    logger.info("CHAINFLOW_ENABLE_RAY=0: agent Ray disabled for multi-GPU evaluation.")
    pl.seed_everything(int(cfg.seed), workers=True)
    logger.info("Global Seed set to %s", cfg.seed)
    torch.backends.cudnn.benchmark = False
    if _bool_cfg(getattr(cfg, "deterministic", False)):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    subscore_root = os.getenv("SUBSCORE_PATH") or str(cfg.output_dir)
    dump_root = Path(subscore_root) / "navsim1_pdm_scores" / cfg.experiment_name

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    scene_loader_inference, dataloader = _build_inference_dataset(cfg, agent)
    trainer = pl.Trainer(**cfg.trainer.params)
    predictions = trainer.predict(
        AgentLightningModule(agent=agent),
        dataloader,
        return_predictions=True,
    )

    if dist_ready():
        dist.barrier()

    rank = dist_rank()
    eval_timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S") if rank == 0 else None
    if rank == 0:
        dump_root.mkdir(parents=True, exist_ok=True)
    eval_timestamp = broadcast_object(eval_timestamp, src=0)
    if rank == 0:
        print(f"Per-rank subscore/trajectory shards saved under {dump_root}")

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    local_predictions, tokens_to_evaluate = _filter_rank_predictions(
        predictions=_merge_prediction_batches(predictions),
        scene_loader=scene_loader_inference,
        metric_cache_loader=metric_cache_loader,
        rank=rank,
        world_size=dist_world_size(),
    )

    rank_dump_dir = dump_root / f"{eval_timestamp}_rank_predictions"
    rank_dump_dir.mkdir(parents=True, exist_ok=True)
    rank_dump_path = rank_dump_dir / f"rank_{rank:05d}.pkl"
    with open(rank_dump_path, "wb") as f:
        pickle.dump(local_predictions, f)

    logger.info(
        "Rank %s starting local pdm scoring for %s / %s predicted scenarios...",
        rank,
        len(local_predictions),
        len(tokens_to_evaluate),
    )

    data_points = _build_score_tasks(cfg, scene_loader_inference, local_predictions)

    score_rows = run_pdm_score(data_points) if data_points else []
    gathered_score_rows = [None for _ in range(dist_world_size())]
    if dist_ready():
        dist.all_gather_object(gathered_score_rows, score_rows)
    else:
        gathered_score_rows = [score_rows]

    if rank != 0:
        return None

    score_rows = [row for rank_rows in gathered_score_rows for row in rank_rows]

    pdm_score_df = pd.DataFrame(score_rows)
    num_successful_scenarios = int(pdm_score_df["valid"].sum())
    num_failed_scenarios = int(len(pdm_score_df) - num_successful_scenarios)
    average_row = pdm_score_df.drop(columns=["token", "log_name", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["log_name"] = ""
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    csv_path = save_path / f"{eval_timestamp}.csv"
    pdm_score_df.to_csv(csv_path, index=False)

    logger.info(
        """
        Finished running evaluation.
            Number of successful scenarios: %s.
            Number of failed scenarios: %s.
            Final average score of valid results: %s.
            Results are stored in: %s.

            All scores:
            %s
        """,
        num_successful_scenarios,
        num_failed_scenarios,
        pdm_score_df["score"].mean(),
        csv_path,
        average_row,
    )


if __name__ == "__main__":
    main()
