#!/bin/bash
set -euo pipefail

ROOT="/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe"
RUN_SCRIPT="${ROOT}/scripts/run_c3_lowres_action_probe_inside_pcot_dbg2g_v2_20260625.sh"

cd "${ROOT}"
mkdir -p logs outputs/c3_lowres_action_probe_20260625_pcot_dbg2g_v2_progress

TS=$(date +%Y%m%d_%H%M%S)
SCREEN="c3lrprobe_v2_${TS}"
RUN="c3_lowres_probe_pcot_dbg2g_v2_${TS}"
OUT="${ROOT}/logs/${RUN}.out"
ERR="${ROOT}/logs/${RUN}.err"
EXIT="${ROOT}/logs/${RUN}.exit"

screen -dmS "${SCREEN}" bash -lc "cd '${ROOT}' && srun --jobid=1118197 --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --job-name=c3lrprobe_v2 '${RUN_SCRIPT}' > '${OUT}' 2> '${ERR}'; ec=\$?; echo \$ec > '${EXIT}'; exit \$ec"

echo "SCREEN=${SCREEN}"
echo "RUN=${RUN}"
echo "OUT=${OUT}"
echo "ERR=${ERR}"
echo "EXIT=${EXIT}"
echo "RESULT=${ROOT}/outputs/c3_lowres_action_probe_20260625_pcot_dbg2g_v2_progress"
