#!/usr/bin/env bash
# ==============================================================================
# InternVL VLM hidden-state caching on one multi-GPU node.
#
# Precompute and cache last_hidden_state from an InternVL checkpoint on NAVSIM.
# Downstream ChainFlow-VLA training can read this cache instead of running VLM inference.
#
# Usage:
#   # Single split
#   bash scripts/cache_dataset/cache_internvl_single_node.sh
#
#   # Multiple splits, processed sequentially
#   TRAIN_TEST_SPLITS="navtrain navtest" bash scripts/cache_dataset/cache_internvl_single_node.sh
#
# Debug mode:
#   DEBUG=1 bash scripts/cache_dataset/cache_internvl_single_node.sh
#
# Common environment overrides:
#   GPUS=4 MODEL_PATH=... CACHE_PATH=... TRAIN_TEST_SPLIT=navtrain \\
#   bash scripts/cache_dataset/cache_internvl_single_node.sh
# ==============================================================================
set -euo pipefail   # Stop on errors, unset variables, and pipeline failures.

# ==============================================================================
# Debug switch
# ==============================================================================
if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x
fi

# ==============================================================================
# 1. Basic settings
# ==============================================================================
GPUS=${GPUS:-8}                          # Number of GPUs
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}    # Master node address

# Pick a free port when MASTER_PORT is not set.
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1]);s.close()')"
else
    MASTER_PORT="${MASTER_PORT}"
fi

# InternVL HF checkpoint, typically after SFT.
MODEL_PATH=${MODEL_PATH:-"/mnt/navsimtraincache/vlm_model/RecogDrive-VLM-2B-original"}

# NAVSIM data root, containing navsim_logs/ and sensor_blobs/.
OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1"}

# nuPlan map data used by NAVSIM scene loading.
NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"/data/download/maps"}
NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}

# NAVSIM experiment root.
NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"/data/download/navsim_dataset/caching_debug"}

# Cache output base. Final path is ${CACHE_PATH_BASE}/${split}_cache.
CACHE_PATH_BASE=${CACHE_PATH_BASE:-"${NAVSIM_EXP_ROOT}/internvl_cache_reproduction"}

# Single split mode.
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navmini}

# Multi-split mode. Example: TRAIN_TEST_SPLITS="navtrain navtest"
TRAIN_TEST_SPLITS=${TRAIN_TEST_SPLITS:-""}

# InternVL image preprocessing settings. Keep them aligned with finetuning.
DYNAMIC_IMAGE_SIZE=${DYNAMIC_IMAGE_SIZE:-true}
USE_THUMBNAIL=${USE_THUMBNAIL:-true}
FORCE_IMAGE_SIZE=${FORCE_IMAGE_SIZE:-448}
MAX_DYNAMIC_PATCH=${MAX_DYNAMIC_PATCH:-12}

# Log output directory.
OUTPUT_DIR=${OUTPUT_DIR:-"${NAVSIM_EXP_ROOT}/caching_debug"}

# ==============================================================================
# 2. Resolve project root
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || { echo "[ERROR] Failed to change to project root: ${PROJECT_ROOT}"; exit 1; }

# ==============================================================================
# 3. CUDA environment
# ==============================================================================
CUDA_HOME="/data/libs/cuda/cuda-12.4/cuda"
export CUDA_HOME
if [[ -d "${CUDA_HOME}/bin" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -d "${CUDA_HOME}/lib64" ]]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ==============================================================================
# 4. Runtime environment
# ==============================================================================
export OPENSCENE_DATA_ROOT
export NAVSIM_EXP_ROOT
export NUPLAN_MAPS_ROOT
export NUPLAN_MAP_VERSION
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MASTER_PORT
export MASTER_ADDR
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_PATH_BASE}"

# ==============================================================================
# 5. Preflight checks
# ==============================================================================
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Invalid MODEL_PATH, config.json not found: ${MODEL_PATH}"
    exit 1
fi

if [[ ! -d "${OPENSCENE_DATA_ROOT}/navsim_logs" ]]; then
    echo "[ERROR] navsim_logs/ not found under OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
    exit 1
fi

if [[ "${GPUS}" -le 0 ]]; then
    echo "[ERROR] Invalid GPUS=${GPUS}"
    exit 1
fi

# ==============================================================================
# 6. Print run summary
# ==============================================================================
echo "=============================================="
echo " InternVL VLM Cache (Single Node ${GPUS}xGPU)"
echo "=============================================="
echo " Project root:       ${PROJECT_ROOT}"
echo " Model:              ${MODEL_PATH}"
echo " NavSim data:        ${OPENSCENE_DATA_ROOT}"
echo " NuPlan maps:        ${NUPLAN_MAPS_ROOT}"
echo " Cache base:         ${CACHE_PATH_BASE}"
echo " Log output:         ${OUTPUT_DIR}"
echo " GPUs:               ${GPUS}"
echo " VLM batch size:     1 (fixed)"
echo " Dynamic img size:   ${DYNAMIC_IMAGE_SIZE}"
echo " Use thumbnail:      ${USE_THUMBNAIL}"
echo " Force image size:   ${FORCE_IMAGE_SIZE}"
echo " Max dynamic patch:  ${MAX_DYNAMIC_PATCH}"
echo " CUDA:               ${CUDA_HOME}"
echo " Master port:        ${MASTER_PORT}"
echo "=============================================="
echo ""

# ==============================================================================
# 7. Preload model code to reduce trust_remote_code races across ranks
# ==============================================================================
echo "[preload] Warming up transformers cache for: ${MODEL_PATH}"
python3 -c "
import torch
from transformers import AutoConfig
_ = AutoConfig.from_pretrained('${MODEL_PATH}', trust_remote_code=True)
print('[preload] Config loaded OK')
from transformers import AutoModel
m = AutoModel.from_pretrained('${MODEL_PATH}', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cpu')
print('[preload] Model loaded OK')
" || echo "[WARN] Model preload failed, continuing anyway (ranks may hit race condition)"

# ==============================================================================
# 8. Cache one split
# ==============================================================================
run_cache_for_split() {
    local split_name="$1"
    local cache_path="${CACHE_PATH_BASE}/${split_name}_cache"
    local log_file="${OUTPUT_DIR}/caching_internvl_${split_name}.txt"

    echo ""
    echo "=============================================="
    echo " Caching split: ${split_name}"
    echo " Cache path:    ${cache_path}"
    echo " Log file:      ${log_file}"
    echo "=============================================="
    echo ""

    mkdir -p "${cache_path}"

    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="${MASTER_ADDR}" \
        --nproc_per_node="${GPUS}" \
        --master_port="${MASTER_PORT}" \
        navsim/planning/script/run_dataset_caching_internvl.py \
        --config-name cache_internvl \
        agent.checkpoint_path="${MODEL_PATH}" \
        agent.cache_mode=true \
        agent.cache_hidden_state=true \
        agent.dynamic_image_size="${DYNAMIC_IMAGE_SIZE}" \
        agent.use_thumbnail="${USE_THUMBNAIL}" \
        agent.force_image_size="${FORCE_IMAGE_SIZE}" \
        agent.max_dynamic_patch="${MAX_DYNAMIC_PATCH}" \
        train_test_split="${split_name}" \
        cache_path="${cache_path}" \
        output_dir="${OUTPUT_DIR}" \
        hydra.output_subdir="${OUTPUT_DIR}/hydra" \
        2>&1 | tee -a "${log_file}"
}

# ==============================================================================
# 9. Run caching
# ==============================================================================
if [[ -n "${TRAIN_TEST_SPLITS}" ]]; then
    # Multi-split mode, sequential execution.
    echo "Multi-split mode: ${TRAIN_TEST_SPLITS}"
    for split in ${TRAIN_TEST_SPLITS}; do
        run_cache_for_split "${split}"
    done
else
    # Single-split mode.
    run_cache_for_split "${TRAIN_TEST_SPLIT}"
fi

echo ""
echo "=============================================="
echo " All caching completed."
echo "=============================================="
