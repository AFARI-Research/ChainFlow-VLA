# ------------------------------------------------------------------------
# Copyright (c) 2026 AFARI-Model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAVSIM (https://github.com/autonomousvision/navsim)
# ------------------------------------------------------------------------

from typing import Any, Dict, Optional

import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from .builders.feature_builder import ChainFlowVLAFeatureBuilder
from .builders.target_builder import ChainFlowVLATargetBuilder
from .chainflow_vla_model import ChainFlowVLAModel
from navsim.agents.abstract_agent import AbstractAgent
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_enable_ray(enable_ray: Optional[bool]) -> bool:
    """Allow eval scripts to disable Ray workers via argument/env override."""
    if enable_ray is not None:
        return bool(enable_ray)
    return _env_flag("CHAINFLOW_ENABLE_RAY", True)

class LitProgressBar(ProgressBar):

    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.log_every_n_steps = log_every_n_steps
        self._train_epoch_start_t = None
        self._val_epoch_start_t = None

    def disable(self):
        self.enable = False

    @staticmethod
    def _is_rank_zero(trainer) -> bool:
        # Lightning provides is_global_zero in most versions
        if hasattr(trainer, "is_global_zero"):
            try:
                return bool(trainer.is_global_zero)
            except Exception:
                pass
        # Fallbacks
        for attr in ("global_rank", "local_rank", "node_rank"):
            if hasattr(trainer, attr):
                try:
                    return int(getattr(trainer, attr)) == 0
                except Exception:
                    continue
        return True

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds_i = max(0, int(seconds))
        h = seconds_i // 3600
        m = (seconds_i % 3600) // 60
        s = seconds_i % 60
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    @staticmethod
    def _now_ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._train_epoch_start_t = time.monotonic()
        return super().on_train_epoch_start(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._val_epoch_start_t = time.monotonic()
        return super().on_validation_epoch_start(trainer, pl_module)

    def _format_line(self, phase: str, epoch: int, batch_idx: int, total_batches: int, start_t) -> str:
        elapsed_s = None
        eta_s = None
        if start_t is not None and batch_idx >= 0:
            elapsed_s = time.monotonic() - start_t
            done = max(1, batch_idx + 1)
            remain = max(0, total_batches - done)
            eta_s = (elapsed_s / done) * remain

        parts = [f"[{self._now_ts()}] Epoch {epoch} - {phase} {batch_idx} / {total_batches}"]
        if elapsed_s is not None:
            parts.append(f"elapsed {self._fmt_duration(elapsed_s)}")
        if eta_s is not None:
            parts.append(f"ETA {self._fmt_duration(eta_s)}")
        return " - ".join(parts)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if not self._is_rank_zero(trainer):
            return
        if batch_idx % self.log_every_n_steps == 0:
            total = int(getattr(self, "total_train_batches", None) or getattr(trainer, "num_training_batches", 0) or 0)
            total = total if total > 0 else (batch_idx + 1)
            prefix = self._format_line("train", trainer.current_epoch, batch_idx, total, self._train_epoch_start_t)
            print(f"{prefix} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if not self._is_rank_zero(trainer):
            return
        if batch_idx % self.log_every_n_steps == 0:
            total = int(getattr(self, "total_val_batches", None) or 0)
            if total <= 0:
                total = batch_idx + 1
            prefix = self._format_line("val", trainer.current_epoch, batch_idx, total, self._val_epoch_start_t)
            print(f"{prefix} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if not self._is_rank_zero(trainer):
            return
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n[{self._now_ts()}] ###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")

def _get_scheduler_arg(scheduler_args: Any, key: str) -> Any:
    if hasattr(scheduler_args, key):
        return getattr(scheduler_args, key)
    return scheduler_args[key]


class ChainFlowVLAAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: Optional[str] = None,
            checkpoint_strict_load: bool = False,
            loss: Optional[nn.Module] = None,
            progress_bar: bool = True,
            console_log_every_n_steps: int = 100,
            scheduler_args: Optional[dict] = None,
            batch_size: int = 64,
            num_gpus: int = 1,
            enable_ray: Optional[bool] = None,
    ):
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self._checkpoint_strict_load = checkpoint_strict_load
        self.progress_bar = progress_bar
        self.console_log_every_n_steps = console_log_every_n_steps
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self._chainflow_vla_model = ChainFlowVLAModel(config)
        self.bce_logit_loss = nn.BCEWithLogitsLoss()

        self.use_ray_score = _env_flag("CHAINFLOW_USE_RAY_SCORE", True) and _resolve_enable_ray(enable_ray)
        try:
            threads_per_node = int(os.getenv("CHAINFLOW_RAY_SCORE_THREADS", "48"))
            self._ray_threads_per_node = max(1, threads_per_node)
        except ValueError:
            self._ray_threads_per_node = 48
        logger.info(
            "PDM score parallel: use_ray=%s, threads_per_node=%s",
            self.use_ray_score,
            self._ray_threads_per_node,
        )

        self.worker = None
        self.worker_map = None
        self.train_metric_cache_paths = None
        self.test_metric_cache_paths = None
        self.train_metric_cache_paths_synthetic = None
        self.test_metric_cache_paths_synthetic = None
        self.get_scores = None
        self.loss = loss

    def _ensure_metric_cache_paths(self) -> None:
        if self.train_metric_cache_paths is not None and self.get_scores is not None:
            return

        from .utils.pdm_score import get_scores

        metric_cache_env = os.getenv("NAVSIM_METRIC_CACHE_ROOT")
        if metric_cache_env is None:
            raise RuntimeError("NAVSIM_METRIC_CACHE_ROOT must be set.")
        metric_cache_root = Path(metric_cache_env)
        metric_cache = MetricCacheLoader(metric_cache_root / "train_metric_cache")
        try:
            synthetic_cache_paths: Dict[str, Path] = {}
            for cache_idx in range(5):
                cache_name = f"train_metric_synthetic_reaction_pdm_v1.0-{cache_idx}"
                synthetic_cache = MetricCacheLoader(metric_cache_root / cache_name)
                synthetic_cache_paths.update(synthetic_cache.metric_cache_paths)
            self.train_metric_cache_paths_synthetic = synthetic_cache_paths
            self.test_metric_cache_paths_synthetic = synthetic_cache_paths
        except (FileNotFoundError, AssertionError, ValueError) as exc:
            logger.info("Synthetic metric cache unavailable: %s", exc)
            self.train_metric_cache_paths_synthetic = None
            self.test_metric_cache_paths_synthetic = None

        self.train_metric_cache_paths = metric_cache.metric_cache_paths
        self.test_metric_cache_paths = metric_cache.metric_cache_paths
        self.get_scores = get_scores

    def _ensure_score_worker(self) -> None:
        if not self.use_ray_score or self.worker is not None:
            return

        from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
        from nuplan.planning.utils.multithreading.worker_utils import worker_map

        self.worker = RayDistributedNoTorch(
            threads_per_node=self._ray_threads_per_node,
            log_to_driver=False,
        )
        self.worker_map = worker_map

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def _load_checkpoint_state(self) -> Dict[str, torch.Tensor]:
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint must contain a state_dict: {self._checkpoint_path}")
        state_dict = checkpoint["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint state_dict must be a dict: {self._checkpoint_path}")

        # Lightning AgentLightningModule stores the inner agent under the
        # "agent." module prefix. Pretrain loading targets ChainFlowVLAAgent
        # itself, so normalize those keys before matching.
        consume_prefix_in_state_dict_if_present(state_dict, prefix="agent.")
        return state_dict

    def _load_pretrained_checkpoint(self) -> None:
        logger.info("Loading pretrained checkpoint: %s", self._checkpoint_path)
        checkpoint_state = self._load_checkpoint_state()
        incompatible = self.load_state_dict(checkpoint_state, strict=False)

        model_state = self.state_dict()
        matched_keys = [
            key
            for key, value in checkpoint_state.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        ]
        logger.info(
            "Pretrained load summary: loaded=%s, missing=%s, unexpected=%s, model_keys=%s",
            len(matched_keys),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            len(model_state),
        )
        if matched_keys:
            logger.info("First loaded keys: %s", ", ".join(matched_keys[:8]))
        if incompatible.missing_keys:
            logger.warning("First missing keys: %s", ", ".join(incompatible.missing_keys[:8]))
        if incompatible.unexpected_keys:
            logger.warning("First unexpected keys: %s", ", ".join(incompatible.unexpected_keys[:8]))

        if not self._checkpoint_strict_load:
            return
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Strict checkpoint load failed: "
                f"missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}."
            )
        if bool(getattr(self._config, "use_diffusion_decoder", False)):
            diffusion_prefix = "_chainflow_vla_model.trajectory_decoder.diffusion_decoder"
            if not any(key.startswith(diffusion_prefix) for key in matched_keys):
                raise RuntimeError(
                    "Strict checkpoint load failed: "
                    f"no diffusion decoder weights matched prefix '{diffusion_prefix}'."
                )

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._checkpoint_path:
            self._load_pretrained_checkpoint()

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        # return SensorConfig(
        #     cam_f0=[3],
        #     cam_l0=[3],
        #     cam_l1=[],
        #     cam_l2=[],
        #     cam_r0=[3],
        #     cam_r1=[],
        #     cam_r2=[],
        #     cam_b0=[3],
        #     lidar_pc=[],
        # )
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self):
        return [ChainFlowVLATargetBuilder(config=self._config)]

    def get_feature_builders(self):
        return [ChainFlowVLAFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._chainflow_vla_model(features, targets)

    def compute_score(self, targets, proposals, test: bool = True):
        self._ensure_metric_cache_paths()
        assert self.train_metric_cache_paths is not None
        assert self.test_metric_cache_paths is not None
        assert self.get_scores is not None
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic or {}
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic or {}

        target_trajectory = targets["trajectory"]
        proposals = proposals.detach()

        data_points = [
            {
                "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
                "poses": poses,
                "test": test
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self.use_ray_score:
            self._ensure_score_worker()
            assert self.worker_map is not None
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        assert self.loss is not None
        return self.loss(targets, pred, self._config, self.compute_score)

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(self._chainflow_vla_model.parameters(), lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(self._chainflow_vla_model.parameters(), lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:

            dataset_size = _get_scheduler_arg(self.scheduler_args, "dataset_size")
            num_epochs = _get_scheduler_arg(self.scheduler_args, "num_epochs")
            T_max = int(math.ceil(dataset_size / global_batchsize) * num_epochs)

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine, 
                eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )           

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):
        checkpoint_cb_every_epoch = ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True,
            filename='epoch-{epoch:02d}-step-{step}'
        )

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        
        if self.progress_bar:
            return [checkpoint_cb_every_epoch, lr_monitor]
        else:
            progress_bar = LitProgressBar(log_every_n_steps=self.console_log_every_n_steps)
            return [checkpoint_cb_every_epoch, progress_bar, lr_monitor]
