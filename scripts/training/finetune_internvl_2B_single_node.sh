#!/usr/bin/env bash
# ==============================================================================
# InternVL3-2B single-node 8-GPU fine-tuning script (DeepSpeed ZeRO Stage 1)
#
# Usage:
#   bash scripts/training/finetune_internvl_2B_single_node.sh
#
# Debug mode (prints each command):
#   DEBUG=1 bash scripts/training/finetune_internvl_2B_single_node.sh
# ==============================================================================
set -euo pipefail   # Strict mode: stop on errors, unset variables, and pipeline failures.

# ==============================================================================
# Debug switch
# ==============================================================================
if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x
fi

# ==============================================================================
# 1. Basic settings (override with environment variables)
# ==============================================================================
GPUS=${GPUS:-8}                          # Number of GPUs
MASTER_PORT=${MASTER_PORT:-34229}        # Distributed communication port
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}    # Master node address

# Batch size: per_device * grad_accum * gpus = total
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-$((64/ PER_DEVICE_BATCH_SIZE / GPUS))}

# Model and data paths
MODEL_PATH=${MODEL_PATH:-"/mnt/navsimtraincache/vlm_model/InternVL3-2B"}
META_PATH=${META_PATH:-"navsim/agents/chainflow_vla/encoders/vlm/internvl_chat/shell/data_info/internvl_finetune.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/data/debug_internvl_finetune/internvl3_2b_finetune_other_dataset_test_8_gpu"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"navsim/agents/chainflow_vla/encoders/vlm/internvl_chat/zero_stage1_config.json"}

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
# 4. Runtime environment variables
# ==============================================================================
export TRITON_CACHE_DIR="/tmp/triton_cache_internvl"   # Avoid DeepSpeed Triton cache issues on NFS.
export ACCELERATE_USE_DEEPSPEED=true
export DEEPSPEED_STRATEGY="deepspeed_stage_1"
export PYTHONPATH="${PROJECT_ROOT}/navsim/agents/chainflow_vla/encoders/vlm/internvl_chat:${PROJECT_ROOT}:${PYTHONPATH:-}"
export MASTER_PORT
export MASTER_ADDR
export TF_CPP_MIN_LOG_LEVEL=3        # Reduce TensorFlow log noise.
export LAUNCHER=pytorch
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ==============================================================================
# 5. Print run summary
# ==============================================================================
echo "=============================================="
echo " InternVL3-2B Finetune (Single Node ${GPUS}xGPU)"
echo "=============================================="
echo " Project root:    ${PROJECT_ROOT}"
echo " Model:           ${MODEL_PATH}"
echo " Dataset:         ${META_PATH}"
echo " Output:          ${OUTPUT_DIR}"
echo " GPUs:            ${GPUS}"
echo " Per-device BS:   ${PER_DEVICE_BATCH_SIZE}"
echo " Grad accum:      ${GRADIENT_ACC}"
echo " Total BS:        $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACC * GPUS))"
echo " CUDA:            ${CUDA_HOME}"
echo " DeepSpeed:       ${DEEPSPEED_CONFIG}"
echo " OMP_NUM_THREADS: ${OMP_NUM_THREADS:-auto}"
echo "=============================================="
echo ""

# ==============================================================================
# 6. Start training
# ==============================================================================
# Key training settings:
#   epochs=5, lr=2e-5, cosine schedule, warmup=10%
#   image_size=448, max_dynamic_patch=12, down_sample=0.5
#   max_seq_length=10000, save every 400 steps, keep 2 checkpoints
#   bf16, gradient checkpointing, group_by_length
# ==============================================================================

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    navsim/agents/chainflow_vla/encoders/vlm/internvl_chat/internvl/train/internvl_chat_finetune_modified.py \
    --model_name_or_path "${MODEL_PATH}" \
    --conv_style "internvl2_5" \
    --use_fast_tokenizer False \
    --output_dir "${OUTPUT_DIR}" \
    --meta_path "${META_PATH}" \
    --overwrite_output_dir True \
    --force_image_size 448 \
    --max_dynamic_patch 12 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.1 \
    --freeze_llm False \
    --freeze_mlp False \
    --freeze_backbone False \
    --vision_select_layer -1 \
    --dataloader_num_workers 8 \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACC}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 3 \
    --max_seq_length 10000 \
    --do_train True \
    --grad_checkpoint True \
    --group_by_length True \
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version 'v2' \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --report_to "tensorboard" \
    2>&1 | grep --line-buffered -v 'petrel_client is not installed' | tee -a "${OUTPUT_DIR}/training_log.txt"
