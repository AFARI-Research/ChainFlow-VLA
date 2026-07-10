import logging
import os
import tempfile
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import ray
from psutil import cpu_count

from nuplan.planning.utils.multithreading.ray_execution import ray_map
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool, WorkerResources

logger = logging.getLogger(__name__)

# Silent botocore which is polluting the terminal because of serialization and deserialization
# with following message: INFO:botocore.credentials:Credentials found in config file: ~/.aws/config
logging.getLogger("botocore").setLevel(logging.WARNING)


def _ddp_isolate_ray_session_and_share_cpus(requested_cpus: int) -> int:
    """
    PyTorch Lightning DDP launches one training process per GPU; each used to call
    ray.init() with the full thread budget. That starts competing Ray clusters on one
    machine (GCS / raylet timeouts). Use a separate RAY_TMPDIR per rank and split CPUs.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        return requested_cpus

    if not dist.is_available() or not dist.is_initialized():
        return requested_cpus

    world_size = dist.get_world_size()
    if world_size <= 1:
        return requested_cpus

    rank = dist.get_rank()
    base = os.environ.get("RAY_TMPDIR", "").strip()
    if base:
        ray_home = Path(base) / f"navsim_pl_rank_{rank}"
    else:
        ray_home = Path(tempfile.gettempdir()) / "navsim_ray" / f"rank_{rank}"
    ray_home.mkdir(parents=True, exist_ok=True)
    os.environ["RAY_TMPDIR"] = str(ray_home)

    per_rank_cpus = max(1, int(requested_cpus) // world_size)
    logger.info(
        "PyTorch DDP active (world_size=%s rank=%s): Ray PDM scoring uses RAY_TMPDIR=%s "
        "and num_cpus=%s (was %s).",
        world_size,
        rank,
        ray_home,
        per_rank_cpus,
        requested_cpus,
    )
    return per_rank_cpus


def initialize_ray(
    master_node_ip: Optional[str] = None,
    threads_per_node: Optional[int] = None,
    local_mode: bool = False,
    log_to_driver: bool = True,
    use_distributed: bool = False,
) -> WorkerResources:
    """
    Initialize ray worker.
    ENV_VAR_MASTER_NODE_IP="master node IP".
    ENV_VAR_MASTER_NODE_PASSWORD="password to the master node".
    ENV_VAR_NUM_NODES="number of nodes available".
    :param master_node_ip: if available, ray will connect to remote cluster.
    :param threads_per_node: Number of threads to use per node.
    :param log_to_driver: If true, the output from all of the worker
            processes on all nodes will be directed to the driver.
    :param local_mode: If true, the code will be executed serially. This
            is useful for debugging.
    :param use_distributed: If true, and the env vars are available,
            ray will launch in distributed mode
    :return: created WorkerResources.
    """
    # Env variables which are set through SLURM script
    env_var_master_node_ip = "ip_head"
    env_var_master_node_password = "redis_password"
    env_var_num_nodes = "num_nodes"

    # Read number of CPU cores on current machine
    number_of_cpus_per_node = threads_per_node if threads_per_node else cpu_count(logical=True)
    number_of_gpus_per_node = 0  # no cuda support
    if not number_of_gpus_per_node:
        logger.info("Not using GPU in ray")

    # Find a way in how the ray should be initialized
    if master_node_ip and use_distributed:
        # Connect to ray remotely to node ip
        logger.info(f"Connecting to cluster at: {master_node_ip}!")
        ray.init(address=f"ray://{master_node_ip}:10001", local_mode=local_mode, log_to_driver=log_to_driver)
        number_of_nodes = 1
    elif env_var_master_node_ip in os.environ and use_distributed:
        # In this way, we started ray on the current machine which generated password and master node ip:
        # It was started with "ray start --head"
        number_of_nodes = int(os.environ[env_var_num_nodes])
        master_node_ip = os.environ[env_var_master_node_ip].split(":")[0]
        redis_password = os.environ[env_var_master_node_password].split(":")[0]
        logger.info(f"Connecting as part of a cluster at: {master_node_ip} with password: {redis_password}!")
        # Connect to cluster, follow to https://docs.ray.io/en/latest/package-ref.html for more info
        ray.init(
            address="auto",
            _node_ip_address=master_node_ip,
            _redis_password=redis_password,
            log_to_driver=log_to_driver,
            local_mode=local_mode,
        )
    else:
        # In this case, we will just start ray directly from this script
        number_of_nodes = 1
        number_of_cpus_per_node = _ddp_isolate_ray_session_and_share_cpus(number_of_cpus_per_node)
        logger.info("Starting ray local!")
        ray.init(
            num_cpus=number_of_cpus_per_node,
            include_dashboard=os.getenv("CHAINFLOW_RAY_DASHBOARD", "0").lower()
            in {"1", "true", "yes", "y", "on"},
            local_mode=local_mode,
            log_to_driver=log_to_driver,
        )

    return WorkerResources(
        number_of_nodes=number_of_nodes,
        number_of_cpus_per_node=number_of_cpus_per_node,
        number_of_gpus_per_node=number_of_gpus_per_node,
    )


class RayDistributedNoTorch(WorkerPool):
    """
    This worker uses ray to distribute work across all available threads.
    """

    def __init__(
        self,
        master_node_ip: Optional[str] = None,
        threads_per_node: Optional[int] = None,
        debug_mode: bool = False,
        log_to_driver: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        logs_subdir: Optional[str] = "logs",
        use_distributed: bool = False,
    ):
        """
        Initialize ray worker.
        :param master_node_ip: if available, ray will connect to remote cluster.
        :param threads_per_node: Number of threads to use per node.
        :param debug_mode: If true, the code will be executed serially. This
            is useful for debugging.
        :param log_to_driver: If true, the output from all of the worker
                processes on all nodes will be directed to the driver.
        :param output_dir: Experiment output directory.
        :param logs_subdir: Subdirectory inside experiment dir to store worker logs.
        :param use_distributed: Boolean flag to explicitly enable/disable distributed computation
        """
        self._master_node_ip = master_node_ip
        self._threads_per_node = threads_per_node
        self._local_mode = debug_mode
        self._log_to_driver = log_to_driver
        self._log_dir: Optional[Path] = Path(output_dir) / (logs_subdir or "") if output_dir is not None else None
        self._use_distributed = use_distributed
        super().__init__(self.initialize())

    def initialize(self) -> WorkerResources:
        """
        Initialize ray.
        :return: created WorkerResources.
        """
        # In case ray was already running, shut it down. This occurs mainly in tests
        if ray.is_initialized():
            logger.warning("Ray is running, we will shut it down before starting again!")
            ray.shutdown()

        return initialize_ray(
            master_node_ip=self._master_node_ip,
            threads_per_node=self._threads_per_node,
            local_mode=self._local_mode,
            log_to_driver=self._log_to_driver,
            use_distributed=self._use_distributed,
        )

    def shutdown(self) -> None:
        """
        Shutdown the worker and clear memory.
        """
        ray.shutdown()

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        del verbose
        return ray_map(task, *item_lists, log_dir=self._log_dir)  # type: ignore

    def submit(self, task: Task, *args: Any, **kwargs: Any):
        """Inherited, see superclass."""
        remote_fn = ray.remote(task.fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids: ray._raylet.ObjectRef = remote_fn.remote(*args, **kwargs)
        return object_ids.future()  # type: ignore
