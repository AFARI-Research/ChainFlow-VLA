#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() { echo "[submission $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

hydra_escape_value() {
  printf '%s' "$1" | sed 's/=/\\=/g'
}

hydra_quote_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\'/\\\'}"
  printf "'%s'" "${value}"
}

PYTHON_BIN="${PYTHON_BIN:-python}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${ROOT_DIR}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${ROOT_DIR}/exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${ROOT_DIR}/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH=/path/to/checkpoint.ckpt}"

AGENT="${AGENT:-chainflow_vla_stage2}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-private_test_e2e}"
SUBMISSION_OUTPUT_DIR="${SUBMISSION_OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/submission}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

TEAM_NAME="${TEAM_NAME:-your_team}"
AUTHORS="${AUTHORS:-author_1,author_2}"
EMAIL="${EMAIL:-contact@example.com}"
INSTITUTION="${INSTITUTION:-your_institution}"
COUNTRY="${COUNTRY:-your_country}"

targets=(
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_create_submission_pickle.py"
  "experiment_name=chainflow_vla_submission"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "agent=${AGENT}"
  "agent.checkpoint_path=$(hydra_escape_value "${CHECKPOINT_PATH}")"
  "++agent.checkpoint_strict_load=true"
  "output_dir=${SUBMISSION_OUTPUT_DIR}"
  "team_name=$(hydra_quote_string "${TEAM_NAME}")"
  "authors=$(hydra_quote_string "${AUTHORS}")"
  "email=$(hydra_quote_string "${EMAIL}")"
  "institution=$(hydra_quote_string "${INSTITUTION}")"
  "country=$(hydra_quote_string "${COUNTRY}")"
)

if [[ -n "${VLM_FEATURE_CACHE_PATH:-}" ]]; then
  targets+=( "agent.config.vlm_feature_cache_path=$(hydra_escape_value "${VLM_FEATURE_CACHE_PATH}")" )
fi

if [[ -n "${SUBMISSION_EXTRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  extra_overrides=( ${SUBMISSION_EXTRA_OVERRIDES} )
  targets+=( "${extra_overrides[@]}" )
fi

mkdir -p "${SUBMISSION_OUTPUT_DIR}"

if [[ "${GPUS_PER_NODE}" -gt 1 ]]; then
  cmd=(
    "${PYTHON_BIN}"
    -m
    torch.distributed.run
    --standalone
    "--nproc_per_node=${GPUS_PER_NODE}"
    "${targets[@]}"
  )
else
  cmd=( "${PYTHON_BIN}" "${targets[@]}" )
fi

log "agent=${AGENT} | split=${TRAIN_TEST_SPLIT} | gpus=${GPUS_PER_NODE}"
log "checkpoint=${CHECKPOINT_PATH}"
log "output=${SUBMISSION_OUTPUT_DIR}"
log "vlm_feature_cache_path=${VLM_FEATURE_CACHE_PATH:-<from yaml>}"
log "Executing:"
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

SUBMISSION_PICKLE_PATH="${SUBMISSION_OUTPUT_DIR%/}/submission.pkl"
if [[ ! -f "${SUBMISSION_PICKLE_PATH}" ]]; then
  log "ERROR: submission file was not created: ${SUBMISSION_PICKLE_PATH}"
  exit 1
fi

log "submission.pkl=${SUBMISSION_PICKLE_PATH}"
