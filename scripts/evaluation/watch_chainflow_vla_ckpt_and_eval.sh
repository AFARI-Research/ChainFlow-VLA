#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() { echo "[watch_eval $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${ROOT_DIR}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${ROOT_DIR}/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${ROOT_DIR}/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_METRIC_CACHE_ROOT="${NAVSIM_METRIC_CACHE_ROOT:-${NAVSIM_EXP_ROOT}}"

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR=/path/to/training/run/or/checkpoints}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  log "ERROR: CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}"
  exit 1
fi

AGENT="${AGENT:-chainflow_vla_stage2}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtest}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_METRIC_CACHE_ROOT}/metric_cache}"
EVAL_SAVE_NAME="${EVAL_SAVE_NAME:-${EVAL_OUTPUT_GROUP:-eval}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-180}"
STATE_FILE="${STATE_FILE:-${CHECKPOINT_DIR%/}/.eval_watcher_processed}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/run_chainflow_vla_pdm_score_evaluation.sh}"

mkdir -p "$(dirname "${STATE_FILE}")"
touch "${STATE_FILE}"

collect_ckpts() {
  if [[ "$(basename "${CHECKPOINT_DIR}")" == "checkpoints" ]]; then
    find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' -size +0c -print
  else
    find "${CHECKPOINT_DIR}" -path '*/lightning_logs/version_*/checkpoints/*.ckpt' ! -name 'last.ckpt' -size +0c -print
  fi | sort
}

run_eval_for_ckpt() {
  local ckpt="$1"
  CHECKPOINT_PATH="${ckpt}" \
  AGENT="${AGENT}" \
  TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT}" \
  METRIC_CACHE_PATH="${METRIC_CACHE_PATH}" \
  EVAL_SAVE_NAME="${EVAL_SAVE_NAME}" \
  VLM_FEATURE_CACHE_PATH="${VLM_FEATURE_CACHE_PATH:-}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" \
  PYTHON_BIN="${PYTHON_BIN:-python}" \
  EVAL_EXTRA_OVERRIDES="${EVAL_EXTRA_OVERRIDES:-}" \
  bash "${EVAL_SCRIPT}"
}

count_processed_ckpts() {
  local n
  n="$(wc -l < "${STATE_FILE}")"
  n="${n//[[:space:]]/}"
  printf '%s' "${n:-0}"
}

log_wait_for_ckpts() {
  local total_ckpts="$1"
  local processed_ckpts="$2"
  local pending_ckpts="$3"

  if (( total_ckpts == 0 )); then
    log "no checkpoint files found under ${CHECKPOINT_DIR}; waiting for training to write ckpt (poll in ${POLL_INTERVAL}s)"
  elif (( pending_ckpts == 0 )); then
    log "all ${total_ckpts} checkpoint(s) evaluated; waiting for new checkpoint files (poll in ${POLL_INTERVAL}s)"
  else
    log "waiting to evaluate ${pending_ckpts} pending checkpoint(s) (${processed_ckpts}/${total_ckpts} done; poll in ${POLL_INTERVAL}s)"
  fi
}

log "watch root=${CHECKPOINT_DIR}"
log "state file=${STATE_FILE}"
log "poll_interval=${POLL_INTERVAL}s"
log "agent=${AGENT} | split=${TRAIN_TEST_SPLIT} | eval_save_name=${EVAL_SAVE_NAME} | gpus=${GPUS_PER_NODE}"
log "metric_cache_path=${METRIC_CACHE_PATH}"
log "vlm_feature_cache_path=${VLM_FEATURE_CACHE_PATH:-<from yaml>}"

while true; do
  total_ckpts=0
  pending_ckpts=0
  evaluated_this_round=0

  while IFS= read -r ckpt; do
    [[ -n "${ckpt}" ]] || continue
    total_ckpts=$((total_ckpts + 1))
    if grep -Fxq "${ckpt}" "${STATE_FILE}"; then
      continue
    fi
    pending_ckpts=$((pending_ckpts + 1))

    log "evaluating ${ckpt}"
    if run_eval_for_ckpt "${ckpt}"; then
      echo "${ckpt}" >> "${STATE_FILE}"
      evaluated_this_round=$((evaluated_this_round + 1))
      log "done ${ckpt}"
    else
      log "evaluation failed; will retry later: ${ckpt}"
      break
    fi
  done < <(collect_ckpts)

  if (( evaluated_this_round == 0 )); then
    log_wait_for_ckpts "${total_ckpts}" "$(count_processed_ckpts)" "${pending_ckpts}"
  fi

  sleep "${POLL_INTERVAL}"
done
