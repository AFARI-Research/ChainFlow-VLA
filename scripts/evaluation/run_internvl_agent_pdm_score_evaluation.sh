#!/bin/bash
set -x

TRAIN_TEST_SPLIT=navtest

export TORCH_NCCL_ENABLE_MONITORING=0
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/download/maps"
export NAVSIM_EXP_ROOT="/data/download/navsim_dataset/caching_debug"
export NAVSIM_DEVKIT_ROOT="/home/zhoutingguang/chainflow-vla"
export OPENSCENE_DATA_ROOT="/mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MASTER_PORT=${MASTER_PORT:-63669}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export MASTER_PORT=${MASTER_PORT}

# ============================================================
# Baseline VLM checkpoint (InternVL3-2B finetuned, text traj)
# ============================================================
# CHECKPOINT="/data/download/navsim_dataset/exp/internvl3_2b_finetune_8_machine_64_gpu_bs_256_minibatch_2/checkpoint-3000" # 2B
# CHECKPOINT="/data/download/navsim_dataset/exp/internvl3_8b_finetune_4_machine_32_gpu_bs_128/checkpoint-1000" # 8B


EXPERIMENT_UID=$(date +%Y.%m.%d.%H.%M.%S)
CHECKPOINT="/mnt/navsimtraincache/vlm_model/RecogDrive-VLM-2B-original"
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_internvl.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=internvl_agent \
    agent.checkpoint_path="$CHECKPOINT" \
    agent.prompt_type='base' \
    agent.cam_type='single' \
    experiment_name=internvl_vlm_test \
    metric_cache_path="/data/internvl_cache/metric_cache_navtest" \
    experiment_uid=$EXPERIMENT_UID
