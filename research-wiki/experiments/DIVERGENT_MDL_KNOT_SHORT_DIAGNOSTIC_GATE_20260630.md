# DIVERGENT MDL-Knot Short Diagnostic Gate

Timestamp: 2026-06-30 Asia/Shanghai

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_ShortDiagGate_Worktree_20260630`

Owned branch: `codex/divergent-mdl-knot-shortdiag-gate-20260630`

## Scope

This package moves MDL-Knot from `PRECHECK_ONLY` candidate evidence toward `ready-for-bounded-diagnostic-review`.

The gate is `SHORT_DIAGNOSTIC_ONLY`. It permits only a bounded one-epoch diagnostic launch inside an already allocated child GPU context after local gates pass. It does not unlock remote sync by itself, Slurm submission, parent hold release, formal full training, evaluation, checkpoint claims, deployment, paper claims, metric claims, runtime/FLOPs claims, or sparse-compute claims.

## Changed Files

- `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py`
- `tools/mdl_knot/validate_mdl_knot_shortdiag.py`
- `tests/test_mdl_knot_shortdiag.py`
- `logs/run_mdl_knot_shortdiag_n16r4.sh`
- `logs/mdl_knot_shortdiag_filelist_20260630.txt`
- `logs/mdl_knot_shortdiag_verification_20260630.txt`
- `research-wiki/experiments/DIVERGENT_MDL_KNOT_SHORT_DIAGNOSTIC_GATE_20260630.md`

## Config Contract

- Extends `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`.
- Preserves route label `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Uses `raw_frame_motion_scout_with_metadata_fallback`.
- Keeps `synthetic_fallback_allowed=False`.
- Sets `diagnostic_only=True`.
- Sets `full_train_unlocked=False`.
- Sets `metric_claim=False`.
- Sets `sparse_compute_claim=False`.
- Sets one-epoch train-only workflow.
- Keeps evaluation, checkpoint, `tools/test.py`, mAP, runtime, deploy, paper, and sparse-compute claims locked.

## Validator Contract

`tools/mdl_knot/validate_mdl_knot_shortdiag.py` validates the merged shortdiag config and optionally a one-epoch train log.

When a log is provided, it fails closed on:

- missing finite `Loss` values;
- `NaN`, `Inf`, or `cost nan`;
- `Traceback`, `RuntimeError`, CUDA OOM, killed process, no-space, or no-GPU markers;
- evaluation, mAP, `result_detection.json`, or `tools/test.py` markers;
- C3, BVR, ABR, Event-Surprise, or combo route drift;
- full-train, deploy, paper, metric, runtime, or sparse-compute claim markers;
- epochs beyond one.

## N16R4 Wrapper Contract

`logs/run_mdl_knot_shortdiag_n16r4.sh` is for an already allocated child GPU context only.

It does not call `sbatch`, `scancel`, or any parent-hold release command.

It runs:

- Python compile checks for the shortdiag config, validator, and test;
- focused pytest for MDL-Knot shortdiag/core/tool integration tests;
- the existing MDL-Knot launch gate on the base MDL config;
- the shortdiag validator on the shortdiag config;
- only when `RUN_SHORTDIAG_TRAIN=1`, a one-epoch `torch.distributed.run` train command and post-log shortdiag validation.

No evaluation command is included.

## Current Local Verification Freeze

Timestamp: 2026-06-30 09:23:07 +08:00 Asia/Shanghai

Verification count: `5/5` local non-GPU verification commands rerun successfully in the owned worktree on branch `codex/divergent-mdl-knot-shortdiag-gate-20260630`.

Preserved command summary: `logs/mdl_knot_shortdiag_verification_20260630.txt`.

Commands and results:

- `python -m pytest tests/test_mdl_knot_shortdiag.py tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py -q`
  - Exit code: `0`.
  - Result: `27 passed, 1 skipped in 18.12s`.
- `python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py`
  - Exit code: `0`.
  - Result: `SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED`; still locked full training, evaluation, checkpoints, `tools/test.py`, mAP and sparse-compute claims.
- `python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`
  - Exit code: `0`.
  - Result: `PRECHECK_ONLY_REQUEST_ALLOWED`; still locked remote sync, Slurm, training, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claims.
- `C:\Windows\System32\bash.exe -lc "cd /mnt/e/DeskTop/TAD/temrefuse-tad/OpenTAD_MDLKnot_ShortDiagGate_Worktree_20260630 && bash -n logs/run_mdl_knot_shortdiag_n16r4.sh"`
  - Exit code: `0`.
  - Result: script syntax check passed. This was a local syntax-only check; no SSH, sync, remote control, Slurm, or training command was run.
- `git diff --check`
  - Exit code: `0`.
  - Result: no whitespace error output.

Claim state after rerun:

- Only `SHORT_DIAGNOSTIC_ONLY` remains allowed.
- Formal training remains locked.
- Evaluation, `tools/test.py`, mAP, metric, runtime/FLOPs, deploy, paper, and sparse-compute claims remain locked.
- Remote sync and Slurm remain locked for this task.

## Allowed Next Action

Allowed next action after local verification: bounded read-only review or route-owner self-check evidence collection for `SHORT_DIAGNOSTIC_ONLY`.

Actual N16R4 one-epoch diagnostic execution remains gated by an already allocated child GPU context and explicit `RUN_SHORTDIAG_TRAIN=1`.

## Still Locked

- Formal full train.
- Evaluation and `tools/test.py`.
- mAP reporting or metric claim.
- Runtime/FLOPs or sparse-compute claim.
- Deployment claim.
- Paper claim.
- Slurm submission and remote sync from this task.
- Parent hold cancellation, release, replacement, or self-termination.
- C3/BVR/ABR/Event-Surprise/combo mixing.

## Claims

No mAP claim exists.

No runtime or FLOPs claim exists.

No deployment claim exists.

No paper claim exists.

No sparse-compute claim exists.

No formal full-train approval exists.
