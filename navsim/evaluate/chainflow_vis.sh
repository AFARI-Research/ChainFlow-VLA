#!/usr/bin/env bash
# Batch-render BEV + front-camera visualizations for the ChainFlow-VLA
# qualitative token set. Thin wrapper around
# navsim/evaluate/render_top1_from_detail.py.
#
# Required env vars:
#   BASE                 directory of per-token detail.json outputs,
#                        produced by the eval pipeline. Layout:
#                        ${BASE}/${log_name}/${token}/proposal_top1/detail.json
#   OPENSCENE_DATA_ROOT  OpenScene/NAVSIM data root. Used to derive
#                        DATA_PATH and SENSOR_PATH defaults.
#   NUPLAN_MAPS_ROOT     nuPlan map root.
#
# Optional env vars:
#   OUT                  output directory (default: ${PWD}/chainflow_vis_output)
#   COLOR                predicted trajectory color (RRGGBB, default: empty
#                        = renderer default)
#   PYTHON_BIN           python executable (default: python)
#   DATA_PATH            NAVSIM logs split (default: $OPENSCENE_DATA_ROOT/navsim_logs/test)
#   SENSOR_PATH          sensor blobs split (default: $OPENSCENE_DATA_ROOT/sensor_blobs/test)
#   BEV_SIZE_X / BEV_SIZE_Y          symmetric BEV range, meters (default: 60)
#   BEV_X_MIN / BEV_X_MAX            asymmetric longitudinal range (default: -20, 40)
#   BEV_Y_MIN / BEV_Y_MAX            asymmetric lateral range (default: unset)
#   RENDER_BEV_GIF       1 to also write a GIF per token (default: 0)
#   GIF_STEPS / GIF_FPS  GIF length and frame rate (default: 8, 8)
#   GIF_ONLY             1 to skip static images and only emit GIFs (default: 0)
#   OUTPUT_FORMAT        jpg|pdf|png|none (default: jpg)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

: "${BASE:?Set BASE=/path/to/<csv_stem>_selected_proposals}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT=/path/to/openscene-v1.1}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT=/path/to/nuplan_maps}"
export NUPLAN_MAPS_ROOT

COLOR="${COLOR:-}"
OUT="${OUT:-${PWD}/chainflow_vis_output}"
DATA_PATH="${DATA_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/test}"
SENSOR_PATH="${SENSOR_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/test}"

BEV_SIZE_X="${BEV_SIZE_X:-60}"
BEV_SIZE_Y="${BEV_SIZE_Y:-60}"
BEV_X_MIN="${BEV_X_MIN:--20}"
BEV_X_MAX="${BEV_X_MAX:-40}"

RENDER_BEV_GIF="${RENDER_BEV_GIF:-0}"
GIF_STEPS="${GIF_STEPS:-8}"
GIF_FPS="${GIF_FPS:-8}"
GIF_ONLY="${GIF_ONLY:-0}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-jpg}"
if [[ "$GIF_ONLY" == "1" ]]; then
  RENDER_BEV_GIF=1
  OUTPUT_FORMAT="none"
fi

mkdir -p "$OUT"

# token, log_name pairs for the ChainFlow-VLA qualitative set
PAIRS=(
  "9b7108902e7158d6,2021.09.16.19.27.01_veh-45_00472_00711"
  "c08cd52346155301,2021.10.06.08.16.17_veh-52_01430_01579"
  "020dee65dab453bb,2021.06.28.16.29.11_veh-38_03263_03766"
  "2c337eb368fb54ca,2021.09.16.15.12.03_veh-42_01037_01434"
  "a2d180a344d15054,2021.09.29.14.44.26_veh-28_01331_01485"
  "00fcad6d092c5e8e,2021.09.16.19.27.01_veh-45_00472_00711"
  "0eb7dda83bbe5fb2,2021.08.30.13.45.25_veh-40_00878_01104"
  "277f191c94b952f3,2021.09.29.15.23.04_veh-28_00814_01101"
  "4376d00ed2245c21,2021.10.06.08.16.17_veh-52_01590_01725"
  "923e4fcf3daa57f8,2021.09.29.14.44.26_veh-28_01331_01485"
  "680c8d90658556da,2021.10.06.07.26.10_veh-52_02208_02394"
)

ok=0
fail=0
for pair in "${PAIRS[@]}"; do
  IFS=',' read -r token log <<< "$pair"
  detail="${BASE}/${log}/${token}/proposal_top1/detail.json"
  if [[ ! -f "$detail" ]]; then
    echo "MISSING: $detail" >&2
    fail=$((fail + 1))
    continue
  fi
  prefix="ChainFlow_${token}"
  echo "=== $token ==="
  bev_args=(
    --bev-size-x "$BEV_SIZE_X"
    --bev-size-y "$BEV_SIZE_Y"
  )
  if [[ -n "${BEV_X_MIN:-}" && -n "${BEV_X_MAX:-}" ]]; then
    bev_args+=(--bev-x-min "$BEV_X_MIN" --bev-x-max "$BEV_X_MAX")
  fi
  if [[ -n "${BEV_Y_MIN:-}" && -n "${BEV_Y_MAX:-}" ]]; then
    bev_args+=(--bev-y-min "$BEV_Y_MIN" --bev-y-max "$BEV_Y_MAX")
  fi
  gif_args=()
  if [[ "$RENDER_BEV_GIF" == "1" ]]; then
    gif_args+=(--render-bev-gif --gif-steps "$GIF_STEPS" --gif-fps "$GIF_FPS")
  fi
  camera_args=(
    --legend-labels "Human Expert,ChainFlow-VLA"
  )
  if [[ "$GIF_ONLY" != "1" ]]; then
    camera_args+=(
      --concat-side-front
      --left-front-width-ratio 0.35
      --right-front-width-ratio 0.35
      --left-front-width-ratio-start 0.42
      --left-front-width-ratio-end 0.77
      --right-front-width-ratio-start 0.23
      --right-front-width-ratio-end 0.58
      --draw-camera-traj
    )
  fi
  if "$PYTHON_BIN" navsim/evaluate/render_top1_from_detail.py \
    --detail-json "$detail" \
    --traj-field pred_traj_local \
    --data-path "$DATA_PATH" \
    --sensor-blobs-path "$SENSOR_PATH" \
    --traj-colors "$COLOR" \
    --output-dir "$OUT" \
    --image-prefix "$prefix" \
    "${camera_args[@]}" \
    "${bev_args[@]}" \
    "${gif_args[@]}" \
    --format "$OUTPUT_FORMAT"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi
done

echo "Done: ok=$ok fail=$fail -> $OUT"
ls -la "$OUT"/ChainFlow* 2>/dev/null | tail -30 || true
