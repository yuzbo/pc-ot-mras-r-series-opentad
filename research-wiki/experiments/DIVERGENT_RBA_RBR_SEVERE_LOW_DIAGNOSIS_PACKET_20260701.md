# DIVERGENT RBA-RBR Severe-Low Diagnosis Packet - 2026-07-01

## Decision Context

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Route name: `RBA-RBR: Regret-Based Recoverable Bracketing`.
- Purpose: recoverable active sparse-frame acquisition, where a cheap first-pass bracket is only a soft prior and later probes can rescue high-regret / high-uncertainty boundary regions outside the first bracket.
- Current stage: bounded eval diagnostic only.
- Not a formal full train.
- No mAP/runtime/FLOPs/deploy/paper claim is unlocked.
- C3/C3-Pro results, selectors, labels, and attribution are not mixed into this route.

## Current Code And Branch State

- Local route-owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`.
- Local branch: `codex/divergent-rba-rbr-20260701`.
- Local evidence commit after first validation: `a8d533d`.
- Local severe-low packet commit before final eval completion: `769415d`.
- Remote N16R4 route-owned worktree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Remote evidence commit after first validation: `152ffed`.
- Remote severe-low packet commit before final eval completion: `0704fd9`.
- GitHub branch exists, but the local/remote route-owned worktrees are ahead of the tracked GitHub branch. Before asking Pro to inspect GitHub, the branch must be synchronized.

## Key Files For Review

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
- `tests/test_rba_rbr_core.py`
- `tests/test_rba_rbr_integration.py`
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`

## Remote Run Evidence

- Protected parent hold: `1118197 pcot_dbg2g`.
- Parent hold status: protected; not released, cancelled, replaced, or modified.
- GPU binding: GPU0 only, `CUDA_VISIBLE_DEVICES=0`.
- Child step: `1118197.539`.
- Job name: `rba_rbr_eval_g0`.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Log file: `srun-1118197.out`.

## Launch And Load Evidence

- Remote route-owned tests before launch: `20 passed`.
- Launch gate before launch: `gate_pass=true`, `full_train_unlocked=false`.
- Eval diagnostic config:
  - `end_epoch=4`.
  - `val_start_epoch=1`.
  - `val_eval_interval=2`.
  - `disable_checkpoint=False`.
  - N16R4 annotation path: `/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json`.
- Pretrained loading:
  - Log contains: `Loads checkpoint by local backend from path: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`.
  - Unexpected keys: classification head weights/bias.
  - Missing keys: adapter parameters, expected because adapter layers are newly added/trainable.
  - Therefore current evidence does not support the simple hypothesis that no pretraining was loaded.

## First Validation Result

- First validation completed all `1645/1645` windows.
- Bad-pattern count for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError, PermissionError, and FileNotFoundError: `0`.
- First diagnostic validation metric after epoch 1:
  - `Average-mAP=0.12%`.
  - `mAP@0.30=0.34%`.
  - `mAP@0.40=0.18%`.
  - `mAP@0.50=0.06%`.
  - `mAP@0.60=0.02%`.
  - `mAP@0.70=0.01%`.
- The child continued into epoch 2; no hard failure was observed at the time of this packet draft.

## Final Bounded Diagnostic Result

- Child step `1118197.539` completed normally.
- Slurm state: `COMPLETED|0:0`.
- Elapsed: `01:57:09`.
- Bad-pattern count for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError, PermissionError, and FileNotFoundError: `0`.
- Checkpoints/logs observed:
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/checkpoint/epoch_1.pth`
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/checkpoint/epoch_3.pth`
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/log.json`
- Final bounded diagnostic metric after epoch 3:
  - `Average-mAP=4.42%`.
  - `mAP@0.30=10.92%`.
  - `mAP@0.40=6.35%`.
  - `mAP@0.50=3.16%`.
  - `mAP@0.60=1.28%`.
  - `mAP@0.70=0.37%`.
- Interpretation: execution is stable and pretraining is loaded, but the detector-health metric remains failure-scale severe-low. This triggers the repository severe-result gate before any RBA-RBR full train or route-level performance interpretation.

## Current Suspicion List

This is a diagnosis checklist, not a conclusion.

- The raw RGB low-resolution scout is deploy-visible but weak: it uses 32 sampled frames and derives actionness/uncertainty/transition from motion, contrast, and mean-change.
- Train/validation selector distribution may be mismatched:
  - train path: `method_base="random_trunc"`, `rba_rbr_train_value_labels=True`.
  - validation/test path: `method_base="sliding_window"`, `rba_rbr_train_value_labels=False`.
- The selected-frame axis and detector physical-time axis may still be inconsistent despite metadata plumbing:
  - generic `irregular_selected_positions` is detector-feature positions.
  - RBA-specific `rba_rbr_raw_selected_positions` is used for backbone time-axis metadata.
  - `remap_gt_to_selected_axis=False`.
- The adapter bridge is fixed-length padded back to `target_len=192`, so the method may currently behave more like a distorted sparse-to-fixed bridge than a clean recoverable detector input.
- The final bounded diagnostic recovered from `0.12%` to `4.42%`, but remains severe-low with normal loss and no hard runtime failure. This suggests a mechanism/geometry/protocol issue is more likely than simple launch failure.

## Questions For Pro / Severe Diagnosis

Ask GPT-5.5 Pro to inspect the synchronized GitHub branch and answer code-grounded, not speculative:

1. Does the current RBA-RBR implementation actually preserve the original recoverable-bracketing idea, or did the fixed-length adapter bridge / weak raw-scout turn it into an unstable sparse sampler?
2. Is there any coordinate, mask, physical-time, selected-axis, or post-processing bug that could explain near-zero mAP despite finite loss?
3. Is `remap_gt_to_selected_axis=False` correct for this route given `rba_rbr_feature_stride=2`, detector-feature centers, and raw-frame selected positions?
4. Does the validation/test pipeline feed the detector with the same temporal geometry assumed by the Adapter/Head?
5. Are the `selected_frame_inds`, `irregular_selected_positions`, `rba_rbr_raw_selected_positions`, and `rba_rbr_detector_feature_positions` internally consistent?
6. Is the deploy-visible raw-scout actionness/uncertainty/transition too weak or badly normalized for THUMOS, and what minimal replacement would remain deploy-visible?
7. What concrete diagnostic experiments should be run next before any formal long train?
8. Provide key code changes if there is an obvious blocker.

## Launch Decision Before Pro

- Do not launch RBA-RBR formal full training.
- Do not interpret the bounded diagnostic metric as a final route result or paper claim.
- Synchronize GitHub before any Pro request if using repository URLs, because local and N16R4 route-owned worktrees are ahead of the tracked GitHub branch.
- Send this packet plus current files/logs to Pro/Oracle for severe-result diagnosis before any further RBA-RBR long run or route-level conclusion.
