# DIVERGENT RBA-RBR Guard Severe-Result Pro Prompt - 2026-07-01

Use this prompt only when GPT-5.5 Pro / Rosetta / Oracle transport is available.
Current transport status is incomplete: Rosetta CDP ports `9223`, `9333`, and
`9222` refused connection, and Oracle Pro provider check reported missing
`OPENAI_API_KEY`.

## Prompt

You are doing a severe-result diagnosis for a temporal action detection sparse
frame acquisition route. Please inspect the GitHub repository and the files
listed below. Do not assume causes. Do not rely on chat-only summaries when code
is available. Be strict and code-grounded.

Repository branch for inspection:

`https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-complete-39073521-20260701`

Route label:

`DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`

Route name:

`RBA-RBR: Regret-Based Recoverable Bracketing`

Idea:

The method should avoid old ABR's irrecoverable first bracket failure. A cheap
low-resolution scout estimates regret/risk, first brackets are soft priors, and
later probes can rescue high-regret/high-uncertainty regions outside the first
bracket. The selector must remain deploy-visible at test time: no GT, teacher
outputs, validation/test raw-prediction shortcuts, or hidden caches.

Key files to inspect:

- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py`
- `opentad/acquisition/rba_rbr/open_tad_bridge.py`
- `opentad/acquisition/rba_rbr/risk_map.py`
- `opentad/acquisition/rba_rbr/soft_bracket.py`
- `opentad/acquisition/rba_rbr/probe_builder.py`
- `opentad/acquisition/rba_rbr/budget_controller.py`
- `opentad/acquisition/rba_rbr/regret_scorer.py`
- `opentad/acquisition/rba_rbr/validators.py`
- `opentad/acquisition/rba_rbr/metadata.py`
- `opentad/datasets/transforms/end_to_end.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `opentad/models/detectors/irregular_actionformer.py`
- `tests/test_rba_rbr_core.py`
- `tests/test_rba_rbr_integration.py`
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`
- `scripts/run_rba_rbr_guard_evaldiag_n16r4.sbatch`

Remote evidence:

- Protected parent hold: `1118197 pcot_dbg2g`; do not release/cancel/replace.
- Corrected guard child: `1118197.560 rba_guard_g0`.
- Remote route tree:
  `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle`.
- Logdir:
  `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Evidence files in that logdir: `srun-1118197.out`, `train.log`,
  `rba_rbr_grid_audit.jsonl`.

Observed result:

- First validation after epoch 1 completed at `2026-07-01 13:39:19 +08:00`.
- First observed `Average-mAP=0.22%`.
- Final observed Average-mAP line completed at `2026-07-01 14:47:42 +08:00`.
- Final observed `Average-mAP=7.45%`.
- Final `mAP@0.3/0.4/0.5/0.6/0.7 = 17.34/11.47/5.69/2.09/0.65`.
- Final evaluator counts: `3325` GT instances and `411700` predictions.
- `Training Over` count: `1`.
- Hard error count: `0`.
- No Traceback/OOM/NaN/non-finite pattern observed in the harvested evidence.

Coverage-guard audit:

- `4090` JSONL rows inspected in the final audit.
- All rows had route label
  `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- All rows had status `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`.
- `raw_valid_k` avg: `84.6235`.
- `mask_true_count` avg: `42.5601`.
- `selected_max_gap_after_guard`: max `16`, avg `15.5897`.
- `max_detector_gap_after_guard`: max `24`, avg `22.5460`.

Current decision state:

- Formal/full RBA-RBR training is locked.
- No mAP/runtime/FLOPs/sparse-compute/deploy/paper claim is unlocked.
- No route conclusion is unlocked.
- A valid GPT-5.5 Pro severe-result diagnosis is required before any long
  RBA-RBR follow-up or route-level conclusion, unless the user gives an explicit
  same-scope override.

Diagnosis needed:

1. Does the implementation preserve the recoverable-bracketing idea, or has it
   become an unstable sparse-to-fixed bridge?
2. Is there a coordinate, mask, native-time, selected-axis, proposal conversion,
   loss assignment, or post-processing bug that can explain near-zero mAP with
   finite training loss?
3. Is `remap_gt_to_selected_axis=False` correct for RBA-RBR with
   `rba_rbr_feature_stride=2`, detector-feature centers, and raw selected
   positions?
4. Are `selected_frame_inds`, `irregular_selected_positions`,
   `rba_rbr_raw_selected_positions`, and
   `rba_rbr_detector_feature_positions` internally consistent through dataset,
   backbone wrapper, detector, projection/neck/head, and evaluator?
5. Does the fixed-length padded adapter bridge destroy the semantics that the
   irregular head expects?
6. Is the deploy-visible raw scout too weak or badly normalized for THUMOS, and
   what replacement would remain deploy-visible and still match the original
   regret/recoverability idea?
7. What are the minimum diagnostic experiments to run next before any formal
   long training?
8. Provide concrete key code changes or diagnostic snippets if you find an
   actionable blocker.

Required response format:

- `Context verdict`
- `Model evidence`
- `Inspected materials`
- `Verdict`
- `Blocking findings`
- `Non-blocking findings`
- `Required fixes or next experiments`
- `Accepted launch/sync/review/Slurm decision`
- `Key code`

Do not pass this route for formal/full training unless the severe-low result has
a concrete diagnosis and the next run is explicitly justified.
