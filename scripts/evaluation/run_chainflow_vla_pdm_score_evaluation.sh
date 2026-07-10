#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() { echo "[eval $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

hydra_escape_value() {
  printf '%s' "$1" | sed 's/=/\\=/g'
}

sanitize_path_component() {
  local value="$1"
  value="${value##*/}"
  value="${value%.ckpt}"
  value="${value// /_}"
  value="${value//=/-}"
  printf '%s' "${value}"
}

default_eval_output_dir() {
  local exp_root="$1"
  local experiment_name="$2"
  local checkpoint_path="$3"
  local eval_save_name="$4"
  local ckpt_name before_lightning experiment_dir

  ckpt_name="$(sanitize_path_component "${checkpoint_path}")"
  if [[ "${checkpoint_path}" == *"/lightning_logs/"* ]]; then
    before_lightning="${checkpoint_path%%/lightning_logs/*}"
    experiment_dir="$(dirname "${before_lightning}")"
    printf '%s/%s/%s' "${experiment_dir%/}" "${eval_save_name}" "${ckpt_name}"
    return 0
  fi

  printf '%s/chainflow_vla/%s/%s/%s' "${exp_root%/}" "${experiment_name}" "${eval_save_name}" "${ckpt_name}"
}

PYTHON_BIN="${PYTHON_BIN:-python}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${ROOT_DIR}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${ROOT_DIR}/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${ROOT_DIR}/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_METRIC_CACHE_ROOT="${NAVSIM_METRIC_CACHE_ROOT:-${NAVSIM_EXP_ROOT}}"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/path/to/checkpoint.ckpt}"

AGENT="${AGENT:-chainflow_vla_stage2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_${AGENT}}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_METRIC_CACHE_ROOT}/metric_cache}"
EVAL_SAVE_NAME="${EVAL_SAVE_NAME:-${EVAL_OUTPUT_GROUP:-eval}}"
OUTPUT_DIR="${OUTPUT_DIR:-$(default_eval_output_dir "${NAVSIM_EXP_ROOT}" "${EXPERIMENT_NAME}" "${CHECKPOINT_PATH}" "${EVAL_SAVE_NAME}")}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

cmd=(
  "${PYTHON_BIN}"
  -m
  torch.distributed.run
  --standalone
  "--nproc_per_node=${GPUS_PER_NODE}"
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py"
  "--config-name=default_run_pdm_score_gpu"
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "metric_cache_path=${METRIC_CACHE_PATH}"
  "agent=${AGENT}"
  "agent.checkpoint_path=$(hydra_escape_value "${CHECKPOINT_PATH}")"
  "++agent.checkpoint_strict_load=true"
  "output_dir=${OUTPUT_DIR}"
)

if [[ -n "${VLM_FEATURE_CACHE_PATH:-}" ]]; then
  cmd+=( "agent.config.vlm_feature_cache_path=$(hydra_escape_value "${VLM_FEATURE_CACHE_PATH}")" )
fi

if [[ -n "${EVAL_EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  extra_overrides=( ${EVAL_EXTRA_OVERRIDES} )
  cmd+=( "${extra_overrides[@]}" )
fi

mkdir -p "${OUTPUT_DIR}"

log "agent=${AGENT} | split=${TRAIN_TEST_SPLIT} | gpus=${GPUS_PER_NODE}"
log "metric_cache_path=${METRIC_CACHE_PATH}"
log "checkpoint=${CHECKPOINT_PATH}"
log "output=${OUTPUT_DIR}"
log "vlm_feature_cache_path=${VLM_FEATURE_CACHE_PATH:-<from yaml>}"
log "Executing:"
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
