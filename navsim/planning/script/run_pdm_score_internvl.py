from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
import torch
import torch.distributed as dist
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score_with_details
# build_worker 不在此脚本中使用，避免与 torchrun 多进程冲突（见 main 内注释）
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.internvl_eval_utils import save_per_token_json

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
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


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    rank = dist.get_rank()
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}, rank={rank}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    token_to_log_file = {t: a["log_file"] for a in args for t in a["tokens"]}
    cfg: DictConfig = args[0]["cfg"]
    logger.info(f"[Rank {rank}] Received {len(tokens)} tokens to process")
    save_json = bool(cfg.get("save_per_token_json", True))

    logger.info(f"[Rank {rank}] Instantiating simulator and scorer...")
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    logger.info(f"[Rank {rank}] Instantiating agent...")
    agent: AbstractAgent = instantiate(cfg.agent)
    logger.info(f"[Rank {rank}] Initializing agent (loading checkpoint)...")
    agent.initialize()
    logger.info(f"[Rank {rank}] Agent initialized successfully.")

    logger.info(f"[Rank {rank}] Loading metric cache from {cfg.metric_cache_path}...")
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    logger.info(f"[Rank {rank}] Metric cache loaded, {len(metric_cache_loader.tokens)} tokens available.")

    logger.info(f"[Rank {rank}] Creating SceneLoader...")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )
    logger.info(f"[Rank {rank}] SceneLoader created, {len(scene_loader.tokens)} tokens loaded.")

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    logger.info(f"[Rank {rank}] {len(tokens_to_evaluate)} tokens to evaluate after intersection.")

    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        if rank == 0:
            logger.info(f"Rank {rank} processing scenario {idx+1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}")

        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            agent_input = scene_loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input)
            pdm_result, details = pdm_score_with_details(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            score_row['rank'] = rank
            if save_json:
                try:
                    save_per_token_json(
                        cfg=cfg,
                        token=token,
                        log_file=token_to_log_file.get(token, ""),
                        metric_cache=metric_cache,
                        model_trajectory=trajectory,
                        pdm_result=pdm_result,
                        details=details,
                        scorer=scorer,
                    )
                except Exception:
                    logger.warning(f"Failed to save per-token json for token {token}:")
                    traceback.print_exc()
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    serialized_score_rows = pickle.dumps(pdm_results)
    return serialized_score_rows

def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """
    Helper function to broadcast an object from the source rank to all other processes.
    :param obj: Object to broadcast.
    :param device: Device to use for tensor operations.
    :param src: Source rank.
    :return: Broadcasted object.
    """
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor)).to(device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0).to(device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    build_logger(cfg)
    logger.info(f"[Rank {rank}] Initialized dist process group (world_size={world_size})")

    # NOTE: 不要在此处调用 build_worker(cfg)。
    # build_worker 会通过 ray.init() 启动一个本地 Ray 集群，但实际评分逻辑
    # 是直接调用 run_pdm_score() 而非通过 Ray worker pool 分发。
    # 当 torch.distributed.run --nproc_per_node=8 启动 8 个进程时，
    # 每个进程都会启动独立的 Ray 集群，导致多个 Ray 实例互相冲突，
    # 进程挂死在 ray::ipc::RayletIpcClient::RegisterClient()。
    # worker = build_worker(cfg)
    cfg.navsim_log_path = cfg.navsim_log_path.replace('navsim_logs', 'meta_datas') 

    logger.info(f"[Rank {rank}] Loading SceneLoader from {cfg.navsim_log_path} ...")
    scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_config=SensorConfig.build_no_sensors(),
        )
    logger.info(f"[Rank {rank}] SceneLoader ready, {len(scene_loader.tokens)} scene tokens found.")

    if rank == 0:
        logger.info(f"[Rank 0] Loading MetricCacheLoader from {cfg.metric_cache_path} ...")
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
        tokens_to_evaluate = sorted(tokens_to_evaluate)  
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
        logger.info(f"[Rank 0] {len(tokens_to_evaluate)} tokens to evaluate after intersection.")
    else:
        tokens_to_evaluate = []

    logger.info(f"[Rank {rank}] Broadcasting tokens_to_evaluate ...")
    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)
    logger.info(f"[Rank {rank}] Received {len(tokens_to_evaluate)} tokens to evaluate.")

    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))
    logger.info(f"[Rank {rank}] InferenceSampler assigned {len(sampler)} scenarios to this rank.")

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token] 
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],
        })

    logger.info(f"[Rank {rank}] Calling run_pdm_score with {len(data_points)} data_points ...")
    serialized_score_rows = run_pdm_score(data_points)
    logger.info(f"[Rank {rank}] run_pdm_score completed, result size={len(serialized_score_rows)} bytes.")

    # Convert the serialized result into a tensor (byte string)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    serialized_tensor = torch.ByteTensor(list(serialized_score_rows)).to(device)

    # Step 1: Gather tensor sizes across all GPUs (each GPU may have different size)
    local_size = torch.tensor(len(serialized_tensor), dtype=torch.long).to(device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)

    # Step 2: Determine the maximum size
    max_size = max(s.item() for s in size_list)

    # Step 3: Pad the tensors to match the maximum size
    if len(serialized_tensor) < max_size:
        padded_tensor = torch.cat([serialized_tensor, torch.zeros(max_size - len(serialized_tensor), dtype=torch.uint8).to(device)])
    else:
        padded_tensor = serialized_tensor

    # Step 4: Perform all_gather on the padded tensors
    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    # Step 5: After gathering, deserialize using each GPU's actual size (NOT local_size)
    if dist.get_rank() == 0:
        final_results = []
        for i, gathered_tensor in enumerate(gathered_results):
            actual_size = size_list[i].item()
            trimmed_tensor = gathered_tensor[:actual_size]
            serialized_data = trimmed_tensor.cpu().numpy().tobytes()
            final_results.extend(pickle.loads(serialized_data))

        logger.info(f"Gathered {len(final_results)} total results from {dist.get_world_size()} GPUs")
        pdm_score_df = pd.DataFrame(final_results)

        num_sucessful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
        average_row = pdm_score_df.drop(columns=["token", "valid",'rank']).mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = pdm_score_df["valid"].all()
        average_row["rank"] = "0"
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

        logger.info(
            f"""
            Finished running evaluation.
                Number of successful scenarios: {num_sucessful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
                Final average score of valid results: {pdm_score_df['score'].mean()}.
                Results are stored in: {save_path / f"{timestamp}.csv"}.
            """
        )


if __name__ == "__main__":
    main()
