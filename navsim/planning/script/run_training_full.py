import os
import random
from typing import Tuple
from pathlib import Path
import logging
import pickle
import warnings
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.collate import build_train_val_dataloaders

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in line_locate_point",
    category=RuntimeWarning,
    module=r"shapely\.linear",
)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def log_model_summary(cfg: DictConfig, agent: AbstractAgent) -> None:
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    total_param_size_mb = total_params * 4 / (1024 ** 2)

    logger.info("Model summary:")
    logger.info("  agent=%s", agent.__class__.__name__)
    logger.info("  total_params=%s", f"{total_params:,}")
    logger.info("  trainable_params=%s", f"{trainable_params:,}")
    logger.info("  approx_param_size_mb=%.2f", total_param_size_mb)
    logger.info(
        "  batch_size_per_gpu=%s | num_workers=%s | num_gpus=%s | max_epochs=%s",
        cfg.dataloader.params.batch_size,
        cfg.dataloader.params.num_workers,
        getattr(cfg.agent, "num_gpus", "unknown"),
        cfg.trainer.params.max_epochs,
    )

    agent_cfg = getattr(cfg.agent, "config", None)
    if agent_cfg is not None:
        key_fields = [
            "proposal_num",
            "num_poses",
            "ref_num",
            "tf_d_model",
            "tf_d_ffn",
            "trajectory_head_type",
            "one_token_per_traj",
            "ar_decoder_num_layers",
            "ar_use_bicycle_kinematics",
            "long_trajectory_additional_poses",
        ]
        summary_parts = [f"{key}={agent_cfg[key]}" for key in key_fields if key in agent_cfg]
        if summary_parts:
            logger.info("  config: %s", " | ".join(summary_parts))

def dist_ready():
    return dist.is_available() and dist.is_initialized()

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """

    logger.info("Building datasets without cached features.")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    train_log_names = (
        cfg.train_logs + cfg.val_logs
        if cfg.train_test_split.get("include_val_logs_in_train", False)
        else cfg.train_logs
    )
    allowed_train_logs = set(train_log_names)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names
            if log_name in allowed_train_logs
        ]
    else:
        train_scene_filter.log_names = train_log_names
    

    logger.info("Number of training logs: %s", len(train_scene_filter.log_names))

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    if os.getenv("DEBUG") == "True":
        import debugpy

        debugpy_host = os.getenv("DEBUGPY_HOST", "0.0.0.0")
        debugpy_port = int(os.getenv("DEBUGPY_PORT", "5678"))
        debugpy.listen((debugpy_host, debugpy_port))
        logger.info("Waiting for debugger attach on %s:%s...", debugpy_host, debugpy_port)
        debugpy.wait_for_client()
        logger.info("Debugger attached.")
    # Set NAVSIM_LOG_LEVEL=DEBUG to enable debug logs from modules such as chainflow_vla_model.
    _lvl_name = os.getenv("NAVSIM_LOG_LEVEL", "").strip().upper()
    if _lvl_name:
        _level = getattr(logging, _lvl_name, None)
        if _level is not None:
            logging.getLogger().setLevel(_level)
            logging.getLogger("navsim").setLevel(_level)
            for _h in logging.getLogger().handlers:
                _h.setLevel(_level)

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    logger.info("Initializing Agent (load pretrained checkpoint if configured)")
    agent.initialize()
    log_model_summary(cfg, agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader, val_dataloader = build_train_val_dataloaders(train_data, val_data, cfg, agent)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    if cfg.train_ckpt_path is None:
        # Pretrain-only flow: load AR (etc.) via agent.initialize() from agent.checkpoint_path with
        # strict=False. Do not also auto-pick a Lightning ckpt — full trainer restore conflicts with
        # new diffusion submodule keys and breaks or masks partial init.
        _agent_ckpt = OmegaConf.select(cfg, "agent.checkpoint_path", default=None)
        if _agent_ckpt and str(_agent_ckpt).strip():
            logger.info(
                "agent.checkpoint_path is set (%s); skipping auto train_ckpt_path lookup. "
                "Use train_ckpt_path=... or TRAIN_RESUME_CKPT_PATH only for full Lightning resume.",
                _agent_ckpt,
            )
        else:
            # Pattern to match all .ckpt files in the base_path recursively
            search_pattern = (
                "/".join(str(cfg.output_dir).split("/")[:-1])
                + "/*/lightning_logs/version_*/checkpoints/"
                + "*.ckpt"
            )
            logger.info("Searching latest training checkpoint from pattern: %s", search_pattern)
            cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
            logger.info("Auto-resolved train_ckpt_path: %s", cfg.train_ckpt_path)

    if cfg.train_ckpt_path is not None:
        logger.info("Trainer will restore from ckpt_path: %s", cfg.train_ckpt_path)
        if os.path.exists(cfg.train_ckpt_path):
            try:
                checkpoint = torch.load(cfg.train_ckpt_path, map_location="cpu")
                state_dict = checkpoint.get("state_dict", {})
                logger.info(
                    "Checkpoint readable: state_dict_keys=%d, epoch=%s, global_step=%s",
                    len(state_dict),
                    checkpoint.get("epoch", "unknown"),
                    checkpoint.get("global_step", "unknown"),
                )
            except Exception as exc:
                logger.warning("Failed to inspect checkpoint file before trainer restore: %s", exc)
        else:
            logger.warning("Configured ckpt_path does not exist: %s", cfg.train_ckpt_path)
    else:
        logger.info("Trainer will start without ckpt_path (fresh init unless agent loads its own checkpoint).")

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()
