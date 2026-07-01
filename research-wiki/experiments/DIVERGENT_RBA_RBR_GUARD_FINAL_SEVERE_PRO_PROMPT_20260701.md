# DIVERGENT RBA-RBR Guard Final Severe-Result Pro Prompt - 2026-07-01

You are doing a severe-result diagnosis for a temporal action detection sparse
frame acquisition route. Please inspect the GitHub branch and the listed files.
Do not infer causes from this prompt. Do not rely on chat history. Be strict,
code-grounded, and evidence-grounded.

Repository branch for inspection:

`https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-complete-39073521-20260701`

Route label:

`DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`

Route name:

`RBA-RBR: Regret-Based Recoverable Bracketing`

Design intent:

RBA-RBR should avoid the old ABR failure mode where the first bracket
irreversibly misses boundaries. It should use deploy-visible low-cost evidence
to estimate regret/risk, treat first brackets as soft priors, and allow later
probes to recover high-regret or high-uncertainty regions outside the initial
bracket. Test-time selection must not use GT, teacher outputs, validation/test
raw-prediction shortcuts, or hidden caches.

Key files to inspect on the branch:

- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py`
- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py`
- `opentad/acquisition/rba_rbr/open_tad_bridge.py`
- `opentad/acquisition/rba_rbr/risk_map.py`
- `opentad/acquisition/rba_rbr/soft_bracket.py`
- `opentad/acquisition/rba_rbr/probe_builder.py`
- `opentad/acquisition/rba_rbr/budget_controller.py`
- `opentad/acquisition/rba_rbr/regret_scorer.py`
- `opentad/acquisition/rba_rbr/validators.py`
- `opentad/acquisition/rba_rbr/metadata.py`
- `opentad/acquisition/rba_rbr/adapter_bridge.py`
- `opentad/datasets/transforms/end_to_end.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `opentad/models/detectors/irregular_actionformer.py`
- `tests/test_rba_rbr_core.py`
- `tests/test_rba_rbr_integration.py`
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`
- `scripts/run_rba_rbr_guard_evaldiag_n16r4.sbatch`
- `scripts/launch_rba_rbr_guarddiag_hold_g0_n16r4.sh`

Remote run evidence:

- Platform: N16R4 Slurm.
- Protected parent hold: `1118197 pcot_dbg2g`; do not release/cancel/replace.
- Corrected guard child: `1118197.560 rba_guard_g0`.
- Remote route tree:
  `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle`.
- Logdir:
  `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Evidence files in that logdir: `srun-1118197.out`, `train.log`,
  `rba_rbr_grid_audit.jsonl`.
- GPU binding: protected hold GPU0 only. C3/GPU1 was not used by this route.

Observed result:

- The bounded guard diagnostic completed normally and printed `Training Over`.
- Hard error pattern count was zero for Traceback, RuntimeError, CUDA OOM,
  Killed, Loss=nan, non-finite, and No space.
- First validation:
  - `2026-07-01 13:39:19 +08:00`
  - `Average-mAP=0.22%`
  - `mAP@0.3/0.4/0.5/0.6/0.7 = 0.66/0.30/0.09/0.04/0.01`
- Final bounded diagnostic validation:
  - `2026-07-01 14:47:42 +08:00`
  - `Average-mAP=7.45%`
  - `mAP@0.3/0.4/0.5/0.6/0.7 = 17.34/11.47/5.69/2.09/0.65`
  - `3325` validation GT instances
  - `411700` predictions
- Final training losses were finite in the inspected logs. Example epoch-final
  lines:
  - epoch 0: `Loss=2.3574 cls_loss=0.7851 reg_loss=0.5568 boundary_loss=1.0155`
  - epoch 1: `Loss=1.8465 cls_loss=0.5892 reg_loss=0.5459 boundary_loss=0.7114`
  - epoch 2: `Loss=1.6841 ...`
  - epoch 3: `Loss=1.5980 ...`

Coverage-guard audit summary:

- JSONL rows: `4090`.
- Route labels: only `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Statuses: only `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`.
- `raw_valid_k`: min `60`, p05 `78`, p50 `85`, p95 `91`, max `94`,
  avg `84.6235`.
- `mask_true_count`: min `30`, p05 `39`, p50 `43`, p95 `46`, max `47`,
  avg `42.5601`.
- `meta_detector_feature_position_count`: min `30`, p05 `39`, p50 `43`,
  p95 `46`, max `47`, avg `42.5601`.
- `guard_addition_count`: min `3`, p05 `13`, p50 `29`, p95 `49`, max `54`,
  avg `29.8227`.
- `selected_max_gap_before_guard`: min `20`, p05 `39`, p50 `64`, p95 `128`,
  max `128`, avg `73.9286`.
- `selected_max_gap_after_guard`: min `1`, p05 `14`, p50 `16`, p95 `16`,
  max `16`, avg `15.5897`.
- `max_detector_gap_before_guard`: min `12.5`, p05 `36.5`, p50 `70.5`,
  p95 `140.5`, max `214`, avg `75.6500`.
- `max_detector_gap_after_guard`: min `2.0`, p05 `20.5`, p50 `23.0`,
  p95 `24.0`, max `24.0`, avg `22.5460`.

Please answer only from code and evidence:

1. Does the current implementation preserve the recoverable-bracketing idea, or
   has it become an unstable sparse-to-fixed bridge?
2. Is there a coordinate, mask, native-time, selected-axis, proposal conversion,
   loss assignment, checkpoint/pretraining, or post-processing bug that can
   explain severe-low mAP with finite training loss?
3. Is `remap_gt_to_selected_axis=False` correct for RBA-RBR with
   `rba_rbr_feature_stride=2`, detector-feature centers, and raw selected
   positions?
4. Are `selected_frame_inds`, `irregular_selected_positions`,
   `rba_rbr_raw_selected_positions`, and
   `rba_rbr_detector_feature_positions` internally consistent through dataset,
   backbone wrapper, detector, head, and evaluator?
5. Does the fixed-length padded adapter bridge destroy the temporal semantics
   expected by the irregular detector/head?
6. Does the postprocess guard correctly cap proposal/class candidate explosion,
   and is `411700` predictions still suspicious after the guard branch?
7. Is the deploy-visible raw scout too weak or badly normalized for THUMOS?
   If yes, what replacement remains deploy-visible and aligned with regret /
   recoverability?
8. What minimum diagnostic experiments should run next before any formal long
   training?
9. Provide concrete key code changes or diagnostic snippets for actionable
   blockers.

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

Do not pass this route for formal/full training unless the severe-low result
has a concrete diagnosis and the next run is explicitly justified.
