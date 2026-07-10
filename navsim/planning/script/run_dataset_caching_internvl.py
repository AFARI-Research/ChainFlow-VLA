#!/usr/bin/env python3
"""
Multi-GPU dataset caching script for InternVL hidden states.

Uses torchrun for distributed execution. Each GPU loads the VLM backbone
independently and caches `last_hidden_state` for its assigned subset of scenes.

Usage (via torchrun):
  torchrun --nnodes=1 --nproc_per_node=8 \\
      navsim/planning/script/run_dataset_caching_internvl.py \\
      agent=internvl_agent \\
      agent.cache_mode=true \\
      agent.cache_hidden_state=true \\
      agent.checkpoint_path=/path/to/internvl_checkpoint \\
      cache_path=/path/to/cache_output \\
      train_test_split=navtrain
"""
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import sys
import uuid
import os
import gzip
import pickle

import hydra
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

import torch.distributed as dist
import torch


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _dump_feature_target_to_pickle(path: str, data_dict: Dict[str, torch.Tensor]) -> None:
    """Save feature/target dict to a gzipped pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=1) as gz_f:
            pickle.dump(data_dict, gz_f)


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Cache features and targets for assigned scene tokens.
    Each GPU process calls this with its own subset of data via torchrun.

    Uses single-sample VLM forward pass for reproducible hidden-state caching.
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    thread_id = str(uuid.uuid4())
    batch_size = 1

    # Give each rank its own transformers cache to avoid race conditions
    # when multiple processes load trust_remote_code=True models concurrently.
    rank_cache_dir = f"/tmp/transformers_cache_rank_{local_rank}"
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = rank_cache_dir

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )
    logger.info(
        "Extracted %d scenarios for thread_id=%s, node_id=%s.",
        len(scene_loader.tokens),
        thread_id,
        node_id,
    )

    cache_root = str(cfg.cache_path)
    os.makedirs(cache_root, exist_ok=True)

    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()

    total = len(scene_loader.tokens)
    log_interval = max(1, total // 20)

    iterator = tqdm(
        range(0, total, batch_size),
        desc=f"[rank {node_id}] Caching (bs={batch_size})",
        total=(total + batch_size - 1) // batch_size,
        file=sys.stdout,
        mininterval=10.0,
        maxinterval=60.0,
    )

    done_count = 0
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, total)
        batch_tokens = scene_loader.tokens[batch_start:batch_end]

        # Pre-load all scenes and agent inputs in this batch
        batch_scenes = []
        batch_agent_inputs = []
        batch_metadatas = []
        for token in batch_tokens:
            scene = scene_loader.get_scene_from_token(token)
            batch_scenes.append(scene)
            batch_agent_inputs.append(scene.get_agent_input())
            batch_metadatas.append(scene.scene_metadata)

        # Single-sample feature extraction (strictly no multi-batch path).
        for i, token in enumerate(batch_tokens):
            metadata = batch_metadatas[i]
            token_cache_dir = os.path.join(cache_root, metadata.log_name, metadata.initial_token)
            for builder in feature_builders:
                cache_file = os.path.join(token_cache_dir, builder.get_unique_name() + ".gz")
                data_dict = builder.compute_features(batch_agent_inputs[i])
                _dump_feature_target_to_pickle(cache_file, data_dict)

        # Target building (per-scene, not batched — targets come from scene metadata)
        for i, token in enumerate(batch_tokens):
            metadata = batch_metadatas[i]
            token_cache_dir = os.path.join(cache_root, metadata.log_name, metadata.initial_token)
            for builder in target_builders:
                cache_file = os.path.join(token_cache_dir, builder.get_unique_name() + ".gz")
                data_dict = builder.compute_targets(batch_scenes[i])
                _dump_feature_target_to_pickle(cache_file, data_dict)

        done_count += len(batch_tokens)
        if done_count % log_interval < batch_size or done_count >= total:
            logger.info(
                "Caching progress: %d / %d (%.1f%%)",
                done_count,
                total,
                100.0 * done_count / total,
            )

    return []


def _broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """Broadcast a pickle-able object from source rank to all other processes."""
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.tensor(list(buffer), dtype=torch.uint8, device=device)
        size_tensor = torch.tensor(len(tensor), dtype=torch.long, device=device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0, dtype=torch.long, device=device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj


def _init_distributed() -> Tuple[int, int, int, torch.device]:
    """Initialize distributed runtime from torchrun environment variables."""
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return local_rank, world_size, rank, device


class InferenceSampler(torch.utils.data.sampler.Sampler):
    """Splits token indices evenly across distributed processes."""

    def __init__(self, size: int):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size: int, world_size: int, rank: int) -> range:
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for multi-GPU InternVL dataset caching.

    Requires torchrun for distributed execution:
      torchrun --nnodes=1 --nproc_per_node=8 navsim/planning/script/run_dataset_caching_internvl.py ...
    """
    local_rank, world_size, rank, device = _init_distributed()
    logger.info(
        "Distributed setup: rank=%s local_rank=%s world_size=%s master=%s:%s",
        rank,
        local_rank,
        world_size,
        os.getenv("MASTER_ADDR", ""),
        os.getenv("MASTER_PORT", ""),
    )

    logger.info("Global Seed set to 0")
    pl.seed_everything(0)

    # OpenScene stores per-log scene pickles under meta_datas/, not navsim_logs/.
    # Also, logs/sensor blobs are often split by train_test_split.data_split (mini/trainval/test).
    navsim_log_path_str = str(cfg.navsim_log_path)
    if "navsim_logs" in navsim_log_path_str:
        navsim_log_path_str = navsim_log_path_str.replace("navsim_logs", "meta_datas")

    data_split = getattr(cfg.train_test_split, "data_split", None)
    if data_split:
        split_log_path = Path(navsim_log_path_str) / str(data_split)
        if split_log_path.exists():
            navsim_log_path_str = str(split_log_path)

        sensor_blobs_path_str = str(cfg.sensor_blobs_path)
        split_sensor_blobs_path = Path(sensor_blobs_path_str) / str(data_split)
        if split_sensor_blobs_path.exists():
            cfg.sensor_blobs_path = str(split_sensor_blobs_path)

    cfg.navsim_log_path = navsim_log_path_str
    if rank == 0:
        logger.info("Scene pickle root (navsim_log_path): %s", cfg.navsim_log_path)

    logger.info("Building SceneLoader")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    if rank == 0:
        tokens_to_evaluate = list(set(scene_loader.tokens))
        tokens_to_evaluate = sorted(tokens_to_evaluate)
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = _broadcast_object(tokens_to_evaluate, device=device, src=0)

    if len(tokens_to_evaluate) == 0:
        logger.warning("No tokens to cache. Exiting on rank=%s", rank)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token]
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],
        })

    _ = cache_features(data_points)
    logger.info(
        "Finished caching %d scenarios for training/validation dataset on rank=%s",
        len(data_points),
        rank,
    )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
