# DIVERGENT RBA-RBR Guard Final Result - 2026-07-01

Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.

This is a route-owned documentation/evidence update only. It records the completed corrected coverage-guard diagnostic child and does not edit code, sync remote files, submit Slurm, touch C3/BVR/MDL files, or unlock any claim.

## Run

- Child step: `1118197.560`.
- Job name: `rba_guard_g0`.
- Protected parent hold: `1118197 pcot_dbg2g`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Latest GitHub evidence branch: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-complete-39073521-20260701`.

## Result

- First observed Average-mAP line: `0.22%` at `2026-07-01 13:39:19 +08:00`.
- Final observed Average-mAP line: `7.45%` at `2026-07-01 14:47:42 +08:00`.
- Final `mAP@0.3/0.4/0.5/0.6/0.7`: `17.34/11.47/5.69/2.09/0.65`.
- Predictions: `411700`.
- GT instances: `3325`.
- `Training Over` count: `1`.
- Hard error count: `0`.

## Guard Audit

- Rows: `4090`.
- Status: all `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`.
- `raw_valid_k` avg: `84.6235`.
- `mask_true_count` avg: `42.5601`.
- `selected_max_gap_after_guard`: max `16`, avg `15.5897`.
- `max_detector_gap_after_guard`: max `24`, avg `22.5460`.

## Decision

- Formal/full RBA-RBR training is locked.
- No mAP/runtime/FLOPs/sparse-compute/deploy/paper claim is unlocked.
- No route conclusion is unlocked.
- A valid GPT-5.5 Pro severe-result diagnosis is required before any long follow-up or route-level conclusion, unless the user gives an explicit same-scope override.
