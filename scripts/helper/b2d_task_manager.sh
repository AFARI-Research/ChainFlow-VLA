#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- Initial argument: always job file
JOB_FILE=$1
shift

if [[ ! -f "$JOB_FILE" ]]; then
    echo "Error: Job file $JOB_FILE not found."
    exit 1
fi

# Extract paths from job file
CKPT=$(grep '^# ckpt:' "$JOB_FILE" | awk -F': ' '{print $2}')
SAVE_ROOT=$(grep '^# save_root:' "$JOB_FILE" | awk -F': ' '{print $2}')

if [[ -z "$CKPT" || -z "$SAVE_ROOT" ]]; then
    echo "Missing ckpt or save_root in $JOB_FILE"
    exit 1
fi

# --- Flags
RESTART_FAILED=0
RESTART_LIST=()
SHOW_LOG=0
SHOW_LOG_TASK=""

# Parse remaining args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--show-log)
            SHOW_LOG=1
            SHOW_LOG_TASK="$2"
            shift 2
            ;;
        --restart-failed)
            RESTART_FAILED=1
            shift
            ;;
        --restart)
            shift
            while [[ "$1" =~ ^[0-9]+$ ]]; do
                RESTART_LIST+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --- Show log
if [[ "$SHOW_LOG" == 1 ]]; then
    LOG_PATH="${SAVE_ROOT}/task_${SHOW_LOG_TASK}/${SHOW_LOG_TASK}.log"
    echo -e "${YELLOW}--- Log for task ${SHOW_LOG_TASK} ---${NC}"
    if [[ -f "$LOG_PATH" ]]; then
        cat "$LOG_PATH"
    else
        echo -e "${RED}Log not found: $LOG_PATH${NC}"
    fi
    exit 0
fi

# --- Normal mode
echo -e "Using checkpoint: ${YELLOW}$CKPT${NC}"
echo -e "Using save root:  ${YELLOW}$SAVE_ROOT${NC}\n"

printf "%-8s %-10s %-12s %-30s\n" "TASK" "JOB_ID" "STATUS" "CURRENT_SCENARIO"
echo "----------------------------------------------------------------------------"

FAILED_TASKS=()

grep -v '^#' "$JOB_FILE" | while read -r TASK_ID JOB_ID; do
    STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | awk '{print $1}')
    COLOR=$NC
    case "$STATUS" in
        COMPLETED) COLOR=$GREEN ;;
        FAILED|TIMEOUT|CANCELLED) COLOR=$RED ;;
        PENDING|RUNNING) COLOR=$YELLOW ;;
    esac

    LOG_PATH="${SAVE_ROOT}/task_${TASK_ID}/${TASK_ID}.log"
    EVAL_JSON="${SAVE_ROOT}/task_${TASK_ID}/eval_bench2drive220_${TASK_ID}.json"

    # Default status message
    SCENARIO="(no eval)"
    CRASHED=0

    # Look for crash-related messages
    if [[ -f "$LOG_PATH" ]]; then
        if tail -n 100 "$LOG_PATH" | grep -qiE 'segmentation fault|agent has crashed|cuda error|fatal'; then
            SCENARIO="CRASH DETECTED"
            CRASHED=1
        elif [[ -f "$EVAL_JSON" ]]; then
            ENTRY_STATUS=$(jq -r '.entry_status' "$EVAL_JSON")
            if [[ "$ENTRY_STATUS" == "Started" ]]; then
                PROGRESS_CURRENT=$(jq -r '._checkpoint.progress[0]' "$EVAL_JSON")
                PROGRESS_TOTAL=$(jq -r '._checkpoint.progress[1]' "$EVAL_JSON")
                RECORDS_LEN=$(jq -r '._checkpoint.records | length' "$EVAL_JSON")
                if (( RECORDS_LEN > 0 )); then
                    LAST_ROUTE=$(jq -r '._checkpoint.records[-1].route_id' "$EVAL_JSON" | sed 's/RouteScenario_//; s/_rep.*//')
                    SCENARIO="Completed Route ${LAST_ROUTE}, Running ${PROGRESS_CURRENT}/${PROGRESS_TOTAL}"
                else
                    SCENARIO="Running ${PROGRESS_CURRENT}/${PROGRESS_TOTAL}"
                fi
            else
                SCENARIO="(no progress)"
            fi
        fi
    fi
    echo -e "${COLOR}$(printf "%-8s %-10s %-12s %-30s" "$TASK_ID" "$JOB_ID" "$STATUS" "$SCENARIO")${NC}"

    if [[ "$RESTART_FAILED" == 1 && ("$STATUS" == "FAILED" || "$STATUS" == "TIMEOUT") ]]; then
        FAILED_TASKS+=("$TASK_ID")
    fi
done

update_job_file() {
    local task_id="$1"
    local new_job_id="$2"

    # Escape slashes for sed
    local new_line="${task_id} ${new_job_id}"
    sed -i.bak "/^${task_id} /s/^.*$/${new_line}/" "$JOB_FILE"
}

# --- Helper function
restart_task() {
    local TASK_ID="$1"
    local COLOR="$2"
    
    # Get old job ID
    OLD_JOB_ID=$(grep -v '^#' "$JOB_FILE" | awk -v id="$TASK_ID" '$1 == id {print $2}')
    
    if [[ -n "$OLD_JOB_ID" ]]; then
        echo -e "${COLOR}Cancelling old job $OLD_JOB_ID for task $TASK_ID...${NC}"
        scancel "$OLD_JOB_ID"
    fi

    echo -e "${COLOR}Restarting task $TASK_ID...${NC}"
    NEW_JOB_ID=$(sbatch Bench2Drive/leaderboard/scripts/launch_b2d_slurm.sh "$CKPT" "$TASK_ID" "${SAVE_ROOT}/task_${TASK_ID}" | awk '{print $NF}')
    echo "New job ID: $NEW_JOB_ID"
    update_job_file "$TASK_ID" "$NEW_JOB_ID"
}

# --- Restarts
for TASK_ID in "${FAILED_TASKS[@]}"; do
    restart_task "$TASK_ID" "$RED"
done

for TASK_ID in "${RESTART_LIST[@]}"; do
    restart_task "$TASK_ID" "$YELLOW"
done