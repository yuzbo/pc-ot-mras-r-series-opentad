#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <screen_name> <train_log> <summary_log>" >&2
  exit 2
fi

SCREEN_NAME="$1"
TRAIN_LOG="$2"
SUMMARY_LOG="$3"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
BASELINE_AVGMAP="${BASELINE_AVGMAP:-63.77}"
TARGET_AVGMAP="${TARGET_AVGMAP:-65.00}"

mkdir -p "$(dirname "$SUMMARY_LOG")"

write_summary() {
  local ts
  ts="$(date '+%F %T %z')"
  {
    echo "## Training Check - ${ts}"
    echo "- Run: ${SCREEN_NAME}"
    echo "- Data source: log_fallback"
    echo "- Train log: ${TRAIN_LOG}"
    echo "- Baseline Avg-mAP: ${BASELINE_AVGMAP}"
    echo "- Target Avg-mAP: ${TARGET_AVGMAP}"

    if [[ ! -f "$TRAIN_LOG" ]]; then
      echo "- Recent metrics: train log not found yet"
      echo "- Anomalies: unknown"
      echo "- Decision: WAIT"
      echo
      return
    fi

    local recent_loss recent_epoch recent_map recent_map70 errors running
    recent_epoch="$(grep -E "\\[Train\\]: Epoch [0-9]+ started" "$TRAIN_LOG" | tail -1 || true)"
    recent_loss="$(grep -E "Loss=" "$TRAIN_LOG" | tail -1 || true)"
    recent_map="$(grep -E "Average-mAP:" "$TRAIN_LOG" | tail -1 || true)"
    recent_map70="$(grep -E "mAP at tIoU 0\\.70" "$TRAIN_LOG" | tail -1 || true)"
    errors="$(grep -E "Traceback|RuntimeError|marked ready|Your training graph has changed|CUDA out of memory|non-finite|NaN| nan | inf | loss=nan|loss=inf|Loss=nan|Loss=inf" "$TRAIN_LOG" | tail -8 || true)"
    if screen -ls 2>/dev/null | grep -q "\\.${SCREEN_NAME}[[:space:]]"; then
      running="yes"
    else
      running="no"
    fi

    echo "- Running screen: ${running}"
    echo "- Recent epoch: ${recent_epoch:-none}"
    echo "- Recent loss: ${recent_loss:-none}"
    echo "- Recent Avg-mAP: ${recent_map:-not yet}"
    echo "- Recent mAP@0.7: ${recent_map70:-not yet}"
    if [[ -n "$errors" ]]; then
      echo "- Anomalies:"
      echo "$errors" | sed 's/^/  /'
      if echo "$errors" | grep -Eq "Traceback|RuntimeError|marked ready|Your training graph has changed|CUDA out of memory"; then
        echo "- Decision: STOP_REVIEW"
        echo "- Reason: hard failure signature found; inspect before continuing."
      else
        echo "- Decision: CONTINUE"
        echo "- Reason: only known/non-fatal warning signatures or ordinary numeric strings found."
      fi
    else
      echo "- Anomalies: none in tracked hard-failure patterns"
      echo "- Decision: CONTINUE"
      echo "- Reason: process alive and latest losses/metrics are being logged."
    fi
    echo
  } >> "$SUMMARY_LOG"
}

echo "supervisor start $(date '+%F %T %z') screen=${SCREEN_NAME}" >> "$SUMMARY_LOG"
while screen -ls 2>/dev/null | grep -q "\\.${SCREEN_NAME}[[:space:]]"; do
  write_summary
  sleep "$INTERVAL_SECONDS"
done
write_summary
echo "supervisor end $(date '+%F %T %z') screen=${SCREEN_NAME}" >> "$SUMMARY_LOG"
