#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "[train $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

hydra_escape_value() {
  printf '%s' "$1" | sed 's/=/\\=/g'
}

PYTHON_BIN="${PYTHON_BIN:-python}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${ROOT_DIR}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${ROOT_DIR}/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${ROOT_DIR}/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_METRIC_CACHE_ROOT="${NAVSIM_METRIC_CACHE_ROOT:-${NAVSIM_EXP_ROOT}}"

AGENT="${AGENT:-chainflow_vla_stage2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TRAIN_RUN_SUBDIR:-${AGENT}}}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrainval}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

cmd=(
  "${PYTHON_BIN}"
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py"
  "--config-name=default_training"
  "agent=${AGENT}"
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "trainer.params.devices=${GPUS_PER_NODE}"
  "agent.num_gpus=${GPUS_PER_NODE}"
)

if [[ -n "${VLM_FEATURE_CACHE_PATH:-}" ]]; then
  cmd+=( "agent.config.vlm_feature_cache_path=$(hydra_escape_value "${VLM_FEATURE_CACHE_PATH}")" )
fi

if [[ -n "${TRAIN_PRETRAIN_CKPT_PATH:-}" ]]; then
  cmd+=( "agent.checkpoint_path=$(hydra_escape_value "${TRAIN_PRETRAIN_CKPT_PATH}")" )
fi

if [[ -n "${TRAIN_RESUME_CKPT_PATH:-}" ]]; then
  cmd+=( "train_ckpt_path=$(hydra_escape_value "${TRAIN_RESUME_CKPT_PATH}")" "agent.checkpoint_path=null" )
fi

if [[ -n "${TRAIN_EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  extra_overrides=( ${TRAIN_EXTRA_OVERRIDES} )
  cmd+=( "${extra_overrides[@]}" )
fi

log "agent=${AGENT} | experiment_name=${EXPERIMENT_NAME} | split=${TRAIN_TEST_SPLIT} | gpus=${GPUS_PER_NODE}"
log "OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
log "NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"
log "NAVSIM_METRIC_CACHE_ROOT=${NAVSIM_METRIC_CACHE_ROOT}"
log "vlm_feature_cache_path=${VLM_FEATURE_CACHE_PATH:-<from yaml>}"
log "pretrain_ckpt=${TRAIN_PRETRAIN_CKPT_PATH:-<none>} | resume_ckpt=${TRAIN_RESUME_CKPT_PATH:-<none>}"
log "CHAINFLOW_RAY_SCORE_THREADS=${CHAINFLOW_RAY_SCORE_THREADS:-<default>}"
log "Executing:"
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
