#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <screen_name> <train_log> <summary_log>" >&2
  echo "env: INTERVAL_SECONDS=1800 BASELINE_AVGMAP=63.77 TARGET_AVGMAP=65.00" >&2
  echo "env: PROCESS_PID=<pid> [PROCESS_PATTERN=<extended-regex>]" >&2
  echo "env: PROCESS_PATTERN=<extended-regex>  # fallback when no stable PID is available" >&2
  echo "note: prefer PROCESS_PID plus PROCESS_PATTERN; pattern-only matching is best-effort." >&2
  exit 2
fi

SCREEN_NAME="$1"
TRAIN_LOG="$2"
SUMMARY_LOG="$3"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
BASELINE_AVGMAP="${BASELINE_AVGMAP:-63.77}"
TARGET_AVGMAP="${TARGET_AVGMAP:-65.00}"
PROCESS_PID="${PROCESS_PID:-}"
PROCESS_PATTERN="${PROCESS_PATTERN:-}"
SHUTTING_DOWN=0

mkdir -p "$(dirname "$SUMMARY_LOG")"

screen_running() {
  screen -ls 2>/dev/null | awk -v token=".${SCREEN_NAME}" '
    {
      pos = index($0, token)
      if (pos > 0 && substr($0, pos + length(token), 1) ~ /[[:space:]]/) {
        found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

process_running() {
  if [[ -n "$PROCESS_PID" ]]; then
    [[ "$PROCESS_PID" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$PROCESS_PID" 2>/dev/null || return 1

    if [[ -n "$PROCESS_PATTERN" ]]; then
      local pid_cmd
      pid_cmd="$(ps -p "$PROCESS_PID" -o args= 2>/dev/null || true)"
      [[ -n "$pid_cmd" ]] || return 1
      printf '%s\n' "$pid_cmd" | grep -Eq -- "$PROCESS_PATTERN" || return 1
    fi
    return 0
  fi

  if [[ -z "$PROCESS_PATTERN" ]]; then
    return 1
  fi

  local matches
  if command -v pgrep >/dev/null 2>&1; then
    matches="$(pgrep -af -- "$PROCESS_PATTERN" 2>/dev/null || true)"
  else
    matches="$(ps -eo pid,args 2>/dev/null | grep -E -- "$PROCESS_PATTERN" | grep -v -- "grep -E" || true)"
  fi
  [[ -n "$matches" ]] || return 1
  printf '%s\n' "$matches" \
    | grep -v -- "supervise_adapter_quality.sh" \
    | grep -v -- "screen -dmS" \
    | grep -q . || return 1
  return 0
}

run_running() {
  screen_running || process_running
}

write_summary() {
  local ts
  ts="$(date '+%F %T %z')"
  {
    echo "## Training Check - ${ts}"
    echo "- Run: ${SCREEN_NAME}"
    echo "- Data source: log_fallback"
    echo "- Train log: ${TRAIN_LOG}"
    echo "- Process PID: ${PROCESS_PID:-none}"
    echo "- Process pattern: ${PROCESS_PATTERN:-none}"
    echo "- Baseline Avg-mAP: ${BASELINE_AVGMAP}"
    echo "- Target Avg-mAP: ${TARGET_AVGMAP}"

    if [[ ! -f "$TRAIN_LOG" ]]; then
      echo "- Recent metrics: train log not found yet"
      echo "- Anomalies: unknown"
      echo "- Decision: WAIT"
      echo
      return
    fi

    local recent_loss recent_epoch recent_map recent_map70 hard_errors running_screen running_process
    recent_epoch="$(grep -E "\\[Train\\]: Epoch [0-9]+ started" "$TRAIN_LOG" | tail -1 || true)"
    recent_loss="$(grep -E "Loss=" "$TRAIN_LOG" | tail -1 || true)"
    recent_map="$(grep -E "Average-mAP:" "$TRAIN_LOG" | tail -1 || true)"
    recent_map70="$(grep -E "mAP at tIoU 0\\.70" "$TRAIN_LOG" | tail -1 || true)"
    hard_errors="$(grep -Ei "Traceback|RuntimeError|marked ready|Your training graph has changed|CUDA out of memory|non[- ]finite|loss[=:_ -]*(nan|inf)([^[:alpha:]]|$)|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)" "$TRAIN_LOG" | tail -8 || true)"
    if screen_running; then
      running_screen="yes"
    else
      running_screen="no"
    fi
    if process_running; then
      running_process="yes"
    else
      running_process="no"
    fi

    echo "- Running screen: ${running_screen}"
    echo "- Running process: ${running_process}"
    echo "- Recent epoch: ${recent_epoch:-none}"
    echo "- Recent loss: ${recent_loss:-none}"
    echo "- Recent Avg-mAP: ${recent_map:-not yet}"
    echo "- Recent mAP@0.7: ${recent_map70:-not yet}"
    if [[ -n "$hard_errors" ]]; then
      echo "- Anomalies:"
      echo "$hard_errors" | sed 's/^/  /'
      echo "- Decision: STOP_REVIEW"
      echo "- Reason: hard failure signature found; inspect before continuing."
    elif [[ "${running_screen}" == "no" && "${running_process}" == "no" ]]; then
      echo "- Anomalies: no active training screen/process matched"
      if [[ -n "$recent_map" ]]; then
        echo "- Decision: REVIEW_RESULT"
        echo "- Reason: run is no longer active and evaluation metrics are present."
      else
        echo "- Decision: STOP_REVIEW"
        echo "- Reason: run is no longer active before any Avg-mAP was logged."
      fi
    else
      echo "- Anomalies: none in tracked hard-failure patterns"
      echo "- Decision: CONTINUE"
      echo "- Reason: screen or process is alive and latest losses/metrics are being logged."
    fi
    echo
  } >> "$SUMMARY_LOG"
}

shutdown() {
  if [[ "$SHUTTING_DOWN" == "1" ]]; then
    exit 0
  fi
  SHUTTING_DOWN=1
  write_summary
  echo "supervisor interrupted $(date '+%F %T %z') screen=${SCREEN_NAME}" >> "$SUMMARY_LOG"
  exit 0
}

trap shutdown INT TERM

echo "supervisor start $(date '+%F %T %z') screen=${SCREEN_NAME}" >> "$SUMMARY_LOG"
while run_running; do
  write_summary
  sleep "$INTERVAL_SECONDS"
done
SHUTTING_DOWN=1
write_summary
echo "supervisor end $(date '+%F %T %z') screen=${SCREEN_NAME}" >> "$SUMMARY_LOG"
