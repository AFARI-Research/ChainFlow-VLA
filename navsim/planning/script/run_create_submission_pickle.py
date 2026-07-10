from typing import Any, Dict, List, Mapping, cast
from pathlib import Path
import logging
import os
import pickle

import hydra
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def init_dist_from_torchrun() -> None:
    """Same pattern as run_pdm_score_multi_gpu.py (单机多 torchrun )."""
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


def _bool_cfg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _gather_predictions_on_rank0(local_predictions: Dict[str, Trajectory]) -> Dict[str, Trajectory]:
    """各 rank 预测的 token 互不重叠；在 rank 0 合并成全集（语义对齐 PDMS 多卡 predict）。"""
    world_size = dist_world_size()
    if not dist_ready() or world_size <= 1:
        return local_predictions

    gathered: List[Any] = [None] * world_size
    dist.barrier()
    dist.all_gather_object(gathered, local_predictions)

    merged: Dict[str, Trajectory] = {}
    if dist_rank() == 0:
        for part in gathered:
            if part:
                merged.update(part)
        logger.info("[submission] Rank 0 merged %s trajectories from %s ranks", len(merged), world_size)
    dist.barrier()
    return merged


def _build_predict_trainer_params(cfg: DictConfig) -> Dict[str, Any]:
    raw_params = OmegaConf.to_container(cfg.trainer.params, resolve=True)
    trainer_params = cast(Dict[str, Any], dict(raw_params) if isinstance(raw_params, dict) else {})
    trainer_params["logger"] = False
    trainer_params["enable_checkpointing"] = False

    world_size = dist_world_size()
    if world_size <= 1:
        trainer_params["devices"] = 1
        trainer_params["strategy"] = "auto"
        trainer_params["num_nodes"] = 1
        trainer_params.setdefault("enable_progress_bar", True)
        return trainer_params

    # torchrun/Torchelastic: LightningEnvironment 要求 devices * num_nodes == WORLD_SIZE。
    trainer_params.setdefault("accelerator", "gpu")
    trainer_params["strategy"] = "ddp"
    num_nodes = max(
        1,
        int(os.environ.get("NNODES", os.environ.get("NUM_NODES", os.environ.get("NODE_COUNT", "1")))),
    )
    trainer_params["num_nodes"] = num_nodes
    if world_size % num_nodes != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must divide num_nodes ({num_nodes}) for Lightning cluster check."
        )
    trainer_params["devices"] = world_size // num_nodes
    trainer_params["enable_progress_bar"] = dist_rank() == 0
    return trainer_params


def _collect_trajectory_predictions(predictions: Any) -> Dict[str, Trajectory]:
    local_predictions: Dict[str, Trajectory] = {}
    for batch_prediction in predictions or []:
        if not isinstance(batch_prediction, Mapping):
            raise RuntimeError(f"Expected dict from predict_step, got {type(batch_prediction)!r}")
        for token, payload in batch_prediction.items():
            local_predictions[str(token)] = payload["trajectory"]
    return local_predictions


def run_submission_predictions_like_eval(cfg: DictConfig, agent: AbstractAgent) -> Dict[str, Trajectory]:
    """
    Same sample construction and predict_step as PDMS eval (run_pdm_score_multi_gpu predict phase):

    SceneLoader → Dataset(no disk cache, append_token) → Lightning predict.

    Single process: Trainer 1 GPU.「单机多卡」: torchrun spawn 后本 rank 上用 devices=1 + ddp，
    与常规 PDMS Launcher 用法一致。
    """
    if agent.requires_scene:
        raise ValueError(
            "Submission mode only receives AgentInput (no annotated scene). agent.requires_scene must be False."
        )

    agent.initialize()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader_inference = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    trainer = pl.Trainer(**_build_predict_trainer_params(cfg))
    predictions = trainer.predict(
        AgentLightningModule(agent=agent, for_viz=False),
        dataloader,
        return_predictions=True,
    )
    return _gather_predictions_on_rank0(_collect_trajectory_predictions(predictions))


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    init_dist_from_torchrun()
    pl.seed_everything(int(cfg.seed), workers=True)
    torch.backends.cudnn.benchmark = False
    if _bool_cfg(getattr(cfg, "deterministic", False)):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    if dist_rank() == 0:
        print(cfg.agent)

    agent = instantiate(cfg.agent)
    save_path = Path(cfg.output_dir)

    merged = run_submission_predictions_like_eval(cfg, agent)

    if dist_rank() == 0:
        submission = {
            "team_name": cfg.team_name,
            "authors": cfg.authors,
            "email": cfg.email,
            "institution": cfg.institution,
            "country / region": cfg.country,
            "predictions": [merged],
        }
        save_path.mkdir(parents=True, exist_ok=True)
        filename = save_path / "submission.pkl"
        with open(filename, "wb") as file:
            pickle.dump(submission, file)
        logger.info(f"Your submission filed was saved to {filename}")

    if dist_ready():
        dist.barrier()


if __name__ == "__main__":
    main()
