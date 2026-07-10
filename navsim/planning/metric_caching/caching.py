from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import logging
import os
import gc
import uuid

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)

from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig, Scene

logger = logging.getLogger(__name__)


def _discover_cached_tokens(cache_path: Path) -> Set[str]:
    """Return tokens that already have a completed metric cache file."""
    if not cache_path.is_dir():
        return set()

    return {path.parent.name for path in cache_path.rglob("metric_cache.pkl") if path.is_file()}


def _split_tokens(tokens: List[str], tokens_per_task: int) -> List[List[str]]:
    return [tokens[idx : idx + tokens_per_task] for idx in range(0, len(tokens), tokens_per_task)]


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    Performs the caching of scenario DB files in parallel.
    :param args: A list of dicts containing the following items:
        "scenario": the scenario as built by scenario_builder
        "cfg": the DictConfig to use to process the file.
    :return: A dict with the statistics of the job. Contains the following keys:
        "successes": The number of successfully processed scenarios.
        "failures": The number of scenarios that couldn't be processed.
    """

    # Define a wrapper method to help with memory garbage collection.
    # This way, everything will go out of scope, allowing the python GC to clean up after the function.
    #
    # This is necessary to save memory when running on large datasets.
    def cache_scenarios_internal(args: List[Dict[str, Union[Path, DictConfig]]]) -> List[CacheResult]:

        def cache_single_scenario(
            scene_dict: Dict[str, Any], processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            scene = Scene.from_scene_dict_list(
                scene_dict,
                None,
                num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,
                num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,
                sensor_config=SensorConfig.build_no_sensors(),
            )
            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")

            return processor.compute_metric_cache(scenario)

        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        log_names = list(dict.fromkeys(a["log_file"] for a in args))
        tokens = [t for a in args for t in a["tokens"]]
        cfg: DictConfig = args[0]["cfg"]

        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = log_names
        scene_filter.tokens = tokens
        scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )

        # Create feature preprocessor
        assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

        processor = MetricCacheProcessor(
            cache_path=cfg.cache.cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
        )

        logger.info(f"Extracted {len(scene_loader)} scenarios for thread_id={thread_id}, node_id={node_id}.")
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scene_dict in enumerate(scene_loader.scene_frames_dicts.values()):
            logger.info(
                f"Processing scenario {idx + 1} / {len(scene_loader)} in thread_id={thread_id}, node_id={node_id}"
            )
            file_cache_metadata = cache_single_scenario(scene_dict, processor)
            gc.collect()

            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        logger.info(f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}")
        return [
            CacheResult(
                failures=num_failures,
                successes=num_successes,
                cache_metadata=all_file_cache_metadata,
            )
        ]

    result = cache_scenarios_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Build the lightning datamodule and cache all samples.
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    """
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    cache_path = Path(cfg.cache.cache_path)
    force_feature_computation = bool(cfg.cache.force_feature_computation)
    cached_tokens = set() if force_feature_computation else _discover_cached_tokens(cache_path)
    tokens_per_task = int(cfg.cache.get("tokens_per_task", 128))
    if tokens_per_task < 1:
        raise ValueError(f"cache.tokens_per_task must be >= 1, got {tokens_per_task}")

    data_points = []
    existing_metadata = []
    num_total_tokens = 0
    for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items():
        num_total_tokens += len(tokens_list)
        if force_feature_computation:
            tokens_to_cache = tokens_list
        else:
            tokens_to_cache = [token for token in tokens_list if token not in cached_tokens]
            existing_metadata.extend(
                CacheMetadataEntry(path)
                for path in (cache_path / log_file).glob("*/**/metric_cache.pkl")
                if path.parent.name in set(tokens_list)
            )

        data_points.extend(
            {
                "cfg": cfg,
                "log_file": log_file,
                "tokens": token_chunk,
            }
            for token_chunk in _split_tokens(tokens_to_cache, tokens_per_task)
        )

    num_cached_tokens = num_total_tokens - sum(len(data_point["tokens"]) for data_point in data_points)
    logger.info(
        "Starting metric caching with %s tasks for %s / %s missing scenarios; %s scenarios already cached.",
        str(len(data_points)),
        str(num_total_tokens - num_cached_tokens),
        str(num_total_tokens),
        str(num_cached_tokens),
    )

    cache_results = worker_map(worker, cache_scenarios, data_points) if data_points else []

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    if num_fail == 0:
        logger.info(
            "Completed dataset caching! All %s features and targets were cached successfully.",
            str(num_total),
        )
    else:
        logger.info(
            "Completed dataset caching! Failed features and targets: %s out of %s",
            str(num_fail),
            str(num_total),
        )

    cached_metadata = existing_metadata + [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]

    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
    save_cache_metadata(cached_metadata, Path(cfg.cache.cache_path), node_id)
    logger.info("Done storing metadata csv file.")
