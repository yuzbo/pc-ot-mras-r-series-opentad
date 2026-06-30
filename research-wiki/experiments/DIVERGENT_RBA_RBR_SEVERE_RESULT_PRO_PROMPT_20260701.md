# DIVERGENT RBA-RBR Severe-Result Pro Prompt - 2026-07-01

You are asked to perform a strict, read-only, code-grounded severe-result diagnosis for a temporal action detection sparse acquisition route.

Do not infer a root cause from narrative. Inspect the GitHub branch and the named files. Treat this as a line-by-line / function-by-function diagnosis request.

## Repository

- Repository: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad`
- Branch: `codex/divergent-rba-rbr-20260701`
- Branch URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-20260701`
- GitHub ref confirmed after evidence sync: `b4ac9b42146e465ca350571a7ec572ae0fb4b5c5`

## Route Boundary

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`
- Route name: `RBA-RBR: Regret-Based Recoverable Bracketing`
- Do not mix this route with C3/C3-Pro/CADF/PQR/GlobalRank-ST results or attribution.
- This is not a formal full train.
- No deploy/runtime/FLOPs/paper claim is unlocked.

## Observed Phenomenon

The current RBA-RBR implementation is stable but produces severe-low diagnostic detector performance.

Remote run:

- Platform: N16R4 protected hold `1118197 pcot_dbg2g`.
- GPU binding: GPU0 only, `CUDA_VISIBLE_DEVICES=0`.
- Child step: `1118197.539`, job name `rba_rbr_eval_g0`.
- Remote worktree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Slurm state: `COMPLETED|0:0`.
- Elapsed: `01:57:09`.
- Bad-pattern count for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError, PermissionError, and FileNotFoundError: `0`.
- Pretraining evidence: log contains `Loads checkpoint by local backend from path: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`; unexpected keys are classification head weights/bias; missing keys are newly added adapter parameters.

Metrics:

- Epoch-1 validation:
  - `Average-mAP=0.12%`
  - `mAP@0.30=0.34%`
  - `mAP@0.40=0.18%`
  - `mAP@0.50=0.06%`
  - `mAP@0.60=0.02%`
  - `mAP@0.70=0.01%`
- Epoch-3/final bounded diagnostic validation:
  - `Average-mAP=4.42%`
  - `mAP@0.30=10.92%`
  - `mAP@0.40=6.35%`
  - `mAP@0.50=3.16%`
  - `mAP@0.60=1.28%`
  - `mAP@0.70=0.37%`

This is a severe-result trigger: stable training/evaluation, but failure-scale mAP.

## Files To Inspect

Please inspect at minimum:

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
- `opentad/datasets/transforms/end_to_end.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `tests/test_rba_rbr_core.py`
- `tests/test_rba_rbr_integration.py`
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`
- `research-wiki/experiments/DIVERGENT_RBA_RBR_SEVERE_LOW_DIAGNOSIS_PACKET_20260701.md`
- `research-wiki/experiments/DIVERGENT_RBA_RBR_LOCAL_IMPLEMENTATION_20260701.md`

## Required Review Questions

Please answer code-grounded, not speculatively:

1. Is the current RBA-RBR implementation actually preserving the recoverable-bracketing idea, or has the fixed-length adapter bridge / current raw scout turned it into an unstable sparse sampler?
2. Is there a coordinate, mask, physical-time, selected-axis, feature-stride, GT remapping, or post-processing bug that can explain severe-low mAP despite finite loss?
3. Is `remap_gt_to_selected_axis=False` correct for this route given `rba_rbr_feature_stride=2`, detector-feature centers, raw-frame selected positions, and ActionFormer-style temporal geometry?
4. Does validation/test feed the detector with the same temporal geometry assumed by the Adapter/Head?
5. Are `selected_frame_inds`, `irregular_selected_positions`, `rba_rbr_raw_selected_positions`, and `rba_rbr_detector_feature_positions` internally consistent through transforms, backbone wrapper, adapter bridge, head, and postprocess?
6. Is the deploy-visible raw-scout actionness/uncertainty/transition implementation too weak or badly normalized for THUMOS? If yes, provide a minimal deploy-visible replacement that does not use GT, teacher, cache, or detector predictions at test time.
7. What is the minimum diagnostic experiment tree before any formal RBA-RBR long train?
8. Provide concrete key code changes for any blocker you find.

## Required Output Format

Return:

- `Context verdict`
- `Model evidence`
- `Inspected materials`
- `Verdict`
- `Blocking findings`
- `Non-blocking findings`
- `Required fixes or next experiments`
- `Accepted launch/sync/review/Slurm decision`

If you cannot inspect the GitHub branch/files, say `CONTEXT_INSUFFICIENT_FOR_CODE_GROUNDED_DIAGNOSIS` and list exactly what was missing.
