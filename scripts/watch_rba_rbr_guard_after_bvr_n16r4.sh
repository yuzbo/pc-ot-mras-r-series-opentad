#!/usr/bin/env bash
set -euo pipefail

PARENT_JOB=${PARENT_JOB:-1118197}
WAIT_STEP=${WAIT_STEP:-1118197.542}
PUBLIC_JOB=${PUBLIC_JOB:-1132718}
WT=${WT:-/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle}
WATCH_DIR=${WATCH_DIR:-/data/run01/sczc063/yuzibo/route_watchers/rba_rbr_guard_after_bvr_1118197_542_public1132718_20260701}
LAUNCH=${LAUNCH:-$WATCH_DIR/launch_rba_rbr_guarddiag_hold_g0_n16r4.sh}
WATCH_LOG=${WATCH_LOG:-$WATCH_DIR/watch.log}

public_sacct_state() {
  sacct -j "$PUBLIC_JOB" --noheader -X --format=State -P 2>/dev/null | head -n 1 | tr -d '[:space:]' || true
}

public_squeue_state() {
  squeue -j "$PUBLIC_JOB" -h -o '%T' 2>/dev/null | head -n 1 | tr -d '[:space:]' || true
}

public_squeue_record() {
  squeue -j "$PUBLIC_JOB" -h -o '%i|%j|%T' 2>/dev/null | head -n 1 || true
}

parent_alive() {
  squeue -j "$PARENT_JOB" -h 2>/dev/null | grep -q "$PARENT_JOB"
}

step_alive() {
  squeue -s -j "$PARENT_JOB" -h -o '%i' 2>/dev/null | grep -qx "$WAIT_STEP"
}

step_alive_fail_closed() {
  if step_alive; then
    return 0
  fi
  sleep 10
  step_alive
}

active_rba_child() {
  squeue -s -j "$PARENT_JOB" -h -o '%i %j' 2>/dev/null | grep -E 'rba_guard_g0|rba_rbr_eval_g0|rba_rbr_short_g0' >/dev/null
}

cancel_pending_public_guard() {
  record="$(public_squeue_record)"
  IFS='|' read -r jid name state <<EOF
$record
EOF
  if [ "$jid" = "$PUBLIC_JOB" ] && [ "$name" = "rba_guarddiag" ] && { [ "$state" = "PENDING" ] || [ "$state" = "CONFIGURING" ]; }; then
    echo "[$(date -Is)] cancel_pending_public_guard_to_avoid_duplicate job=$jid name=$name state=$state" | tee -a "$WATCH_LOG"
    scancel "$PUBLIC_JOB"
    return 0
  fi
  echo "[$(date -Is)] public_guard_not_cancelled_unexpected_record=${record:-missing}" | tee -a "$WATCH_LOG"
  return 1
}

mkdir -p "$WATCH_DIR"
echo "[$(date -Is)] watcher_start parent=$PARENT_JOB wait_step=$WAIT_STEP public_job=$PUBLIC_JOB wt=$WT" | tee -a "$WATCH_LOG"

while step_alive_fail_closed; do
  sacct_state="$(public_sacct_state)"
  queue_state="$(public_squeue_state)"
  state="${queue_state:-$sacct_state}"
  case "$state" in
    RUNNING|COMPLETING|COMPLETED)
      echo "[$(date -Is)] public_guard_already_$state stop_no_hold_launch job=$PUBLIC_JOB" | tee -a "$WATCH_LOG"
      exit 0
      ;;
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
      echo "[$(date -Is)] public_guard_terminal_$state stop_for_manual_review job=$PUBLIC_JOB" | tee -a "$WATCH_LOG"
      exit 4
      ;;
  esac
  echo "[$(date -Is)] waiting_for_bvr step=$WAIT_STEP public_state=${state:-unknown}" | tee -a "$WATCH_LOG"
  sleep 180
done

if ! parent_alive; then
  echo "[$(date -Is)] parent_hold_missing stop_no_launch parent=$PARENT_JOB" | tee -a "$WATCH_LOG"
  exit 3
fi

sacct_state="$(public_sacct_state)"
queue_state="$(public_squeue_state)"
state="${queue_state:-$sacct_state}"
case "$state" in
  RUNNING|COMPLETING|COMPLETED)
    echo "[$(date -Is)] public_guard_already_$state stop_no_hold_launch job=$PUBLIC_JOB" | tee -a "$WATCH_LOG"
    exit 0
    ;;
  FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
    echo "[$(date -Is)] public_guard_terminal_$state stop_for_manual_review job=$PUBLIC_JOB" | tee -a "$WATCH_LOG"
    exit 4
    ;;
esac

if active_rba_child; then
  echo "[$(date -Is)] active_rba_child_exists stop_no_duplicate" | tee -a "$WATCH_LOG"
  squeue -s -j "$PARENT_JOB" -h -o '%i %j %T' | tee -a "$WATCH_LOG"
  exit 0
fi

if [ "$queue_state" = "PENDING" ] || [ "$queue_state" = "CONFIGURING" ]; then
  cancel_pending_public_guard
fi

RUN_TAG="rba_rbr_guard_evaldiag_holdg0_torchrun_$(date +%Y%m%d_%H%M%S_%z)"
LOGDIR="$WT/logs/$RUN_TAG"
mkdir -p "$LOGDIR"
export RBA_RBR_GUARD_LOGDIR="$LOGDIR"
echo "[$(date -Is)] launching_rba_guard_hold_g0 logdir=$LOGDIR" | tee -a "$WATCH_LOG"
srun --overlap \
  --jobid="$PARENT_JOB" \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=120G \
  --job-name=rba_guard_g0 \
  --output="$LOGDIR/srun-%j.out" \
  "$LAUNCH"
rc=$?
echo "[$(date -Is)] rba_guard_hold_g0_finished rc=$rc logdir=$LOGDIR" | tee -a "$WATCH_LOG"
exit "$rc"
