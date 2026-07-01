# DIVERGENT RBA-RBR Control Diagnostics 20260701

## 2026-07-01 15:35:09 +08:00 - CONTROL_DIAGNOSTIC_ONLY knobs staged locally

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_ControlDiag_Worktree_20260701`.
- Owned branch: `codex/divergent-rba-rbr-control-diagnostics-20260701`.
- Purpose: separate selector quality from budget/bridge/native-axis geometry after the severe-low RBA-RBR diagnostic result.
- Changed surface:
  - input sampling / selector bypass: yes, via `rba_rbr_control_mode="uniform_raw"`.
  - dynamic budget policy: no deployable policy change; this is forced control only.
  - token compression: no.
  - Adapter/backbone internals: no.
  - detector head logic: no.
  - losses/assignment: no.
  - post-processing/evaluator: no behavior change.
- Control A: `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_control_uniform_full.py`.
  - Raw valid budget: `192`.
  - Detector feature stride: `2`.
  - Expected detector valid features: `96`.
  - Same RBA bridge/native-axis metadata path is retained.
- Control B: `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_control_uniform_lowbudget.py`.
  - Raw valid budget: `85`, approximating the observed corrected guard raw budget.
  - Detector feature stride: `2`.
  - Expected detector valid features: `43`.
  - Same RBA bridge/native-axis metadata path is retained.
- Leakage/claim locks:
  - No GT, teacher, raw-prediction, proposal-cache, oracle-boundary, or oracle-residual signal is used by the selector/control.
  - `full_train_unlocked=False`.
  - `no_metric_claim=True`, `no_runtime_claim=True`, `no_deploy_claim=True`, `no_paper_claim=True`, `no_sparse_compute_claim=True`.
- Intended next stage after local verification and review only: `SHORT_DIAGNOSTIC_ONLY`.
- Not allowed by this implementation alone: full training, remote sync, Slurm launch, `tools/test.py`, final mAP claim, runtime/FLOPs claim, deploy claim, paper claim, or sparse-compute claim.

## Self-check notes

- The forced uniform controls return a RBA ledger with `selector_method="forced_uniform_control_diagnostic"` and `budget_stop_reason="control_forced_uniform"`.
- `LoadFrames` still builds the existing `adapter_fixed_length_padded_bridge`, detector masks, RBA raw positions, detector feature positions, and native-axis metadata.
- The launch gate now audits control configs by synthesizing a dense 384-window, validating selected raw positions, adapter valid masks, detector masks, native-axis metadata, raw/detector density summaries, and claim locks.
- Focused pytest coverage was added in `tests/test_rba_rbr_control_diagnostics.py`.

## Local verification

- `python -m py_compile` on changed Python/config/test files: passed.
- `python -m pytest tests\test_rba_rbr_core.py tests\test_rba_rbr_integration.py tests\test_rba_rbr_control_diagnostics.py -q`: `25 passed, 7 skipped`.
- `python tools\rba_rbr\validate_rba_rbr_launch_gate.py --config configs\adatad\thumos\input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`: `gate_pass=true`, next action `FINAL_READ_ONLY_REVIEW_THEN_LOCAL_PRECHECK_ONLY`.
- `python tools\rba_rbr\validate_rba_rbr_launch_gate.py --config configs\adatad\thumos\input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_control_uniform_full.py`: `gate_pass=true`, control audit `raw_valid_k=192`, `detector_feature_valid_k=96`, `detector_mask_len=96`, next action `FINAL_READ_ONLY_REVIEW_THEN_SHORT_DIAGNOSTIC_ONLY`.
- `python tools\rba_rbr\validate_rba_rbr_launch_gate.py --config configs\adatad\thumos\input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_control_uniform_lowbudget.py`: `gate_pass=true`, control audit `raw_valid_k=85`, `detector_feature_valid_k=43`, `detector_mask_len=96`, next action `FINAL_READ_ONLY_REVIEW_THEN_SHORT_DIAGNOSTIC_ONLY`.
- `git diff --check`: passed with line-ending warnings only.
- No remote sync, Slurm, training/evaluation launch, Pro/Oracle/Rosetta call, mAP/runtime/FLOPs/deploy/paper/sparse-compute claim, or full-train unlock occurred.
