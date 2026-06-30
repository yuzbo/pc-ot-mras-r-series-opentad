# DIVERGENT ABR Short Diagnostic Gate 2026-06-30

Route label: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_ShortDiagGate_Worktree_20260630`

Owned branch: `codex/divergent-abr-shortdiag-gate-20260630`

## Scope

This package prepares ABR for a bounded `SHORT_DIAGNOSTIC_ONLY` decision. It does not introduce a formal full-train config, result claim, deployment claim, paper claim, or sparse-compute claim.

Changed surface:

- Config only: `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py`
- Gate validator only: `tools/abr/validate_abr_shortdiag.py`
- Focused tests only: `tests/test_abr_shortdiag.py`
- Child-context wrapper only: `logs/run_abr_shortdiag_n16r4.sh`
- Evidence file list: `logs/abr_shortdiag_filelist_20260630.txt`

Not changed:

- Adapter internals
- Detector head logic
- Loss or assignment logic
- Test-time post-processing
- Evaluator behavior
- C3, BVR, MDL, combo, or any non-ABR route files
- Remote sync, Slurm submission, full training, `tools/test.py`, or Pro submission

## Gate Semantics

The short diagnostic config extends the existing ABR candidate config:

`./input_abr_active_bracket_refinement_adapter_irregular_headv3.py`

It explicitly sets:

- `diagnostic_only=True`
- `full_train_unlocked=False`
- `metric_claim=False`
- `sparse_compute_claim=False`
- `runtime_claim=False`
- `deploy_claim=False`
- `paper_claim=False`
- `workflow.end_epoch=1`
- `workflow.disable_checkpoint=True`
- `workflow.val_start_epoch=999999`
- `workflow.val_loss_interval=-1`
- `workflow.val_eval_interval=-1`
- `inference.save_raw_prediction=False`
- `post_processing.save_dict=False`

The validator fails closed for:

- missing config or wrong config name
- missing ABR route label
- missing `SHORT_DIAGNOSTIC_ONLY` locks
- full-train, deploy, metric, runtime, sparse-compute, or paper claim unlock markers
- missing finite `Loss=...` line when a train log is provided
- NaN, Inf, non-finite cost, Traceback, RuntimeError, CUDA OOM, killed process, no-space, or no-GPU markers
- evaluation, mAP, `result_detection`, `tools/test.py`, `eval_one_epoch`, or evaluator markers
- C3, BVR, MDL, combo route, GlobalRank, or Boundary Microscope route drift markers

## Wrapper Semantics

`logs/run_abr_shortdiag_n16r4.sh` is only for an already allocated child GPU context.

It does not call `sbatch`, `scancel`, `scontrol release`, or `scontrol hold`.

It runs:

1. `python -m py_compile` on the shortdiag config, validator, and focused test.
2. Focused pytest for ABR shortdiag, ABR core, and ABR pipeline/gate tests.
3. Existing ABR launch gate on the base ABR config.
4. ABR shortdiag validator on the shortdiag config.
5. Only in explicit `train` mode, and only with `SLURM_JOB_ID` or `ABR_CHILD_GPU_CONTEXT=1` plus `ABR_SHORTDIAG_ACK=1`, one epoch of `torchrun ... tools/train.py ... --not_eval`.
6. ABR shortdiag validator on the captured train log.

No evaluation is run.
No `tools/test.py` command is run.
No mAP is produced.
No checkpoint claim is made.

## Allowed Next Action

Allowed next action after local verification:

`ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY` inside an already allocated child GPU context, if the operator explicitly acknowledges the diagnostic-only semantics.

## Still Locked

- Formal full training
- Evaluation and `tools/test.py`
- mAP or metric claim
- Runtime/FLOPs/sparse-compute claim
- Deployment claim
- Paper claim
- Remote sync or Slurm submission
- Parent hold release/cancel/replace behavior

## Claims

There is no metric result.
There is no runtime result.
There is no deployment result.
There is no paper claim.
There is no sparse-compute claim.
There is no ABR full-train approval.

## Local Verification

Timestamp: `2026-06-30 08:03:40 +08:00`

Commands run from the owned worktree:

- `python -m py_compile configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py tools/abr/validate_abr_shortdiag.py tests/test_abr_shortdiag.py`
  - Result: PASS
- `python -m pytest tests/test_abr_shortdiag.py tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q`
  - Result: PASS, `35 passed, 1 skipped in 1.61s`
- `python tools/abr/validate_abr_shortdiag.py --config configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py`
  - Result: PASS
  - Validator decision: `allowed_next_action=ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY`, `diagnostic_only=True`, `full_train_unlocked=False`, `metric_claim=False`, `sparse_compute_claim=False`
  - Still locked: formal full train, `tools/test.py`, evaluation, checkpoint claim, mAP/paper claim, runtime/sparse-compute claim, deploy or remote sync
- `git diff --check`
  - Result: PASS
- `bash -n logs/run_abr_shortdiag_n16r4.sh`
  - Result: PASS

Git status note:

- Ordinary untracked files: config, validator, focused test, route report.
- `logs/run_abr_shortdiag_n16r4.sh` and `logs/abr_shortdiag_filelist_20260630.txt` exist but are ignored by the repository-level `/logs/` ignore rule.
