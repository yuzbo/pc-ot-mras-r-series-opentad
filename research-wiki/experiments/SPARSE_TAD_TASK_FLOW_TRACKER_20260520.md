# Sparse TAD Task Flow Tracker

Route-owned copy for `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.

## All Required Experiments and Current Status

| Experiment / config | Changed surface | Current status | Review / gate state | Deployment / result state | Next action |
| --- | --- | --- | --- | --- | --- |
| MDL-Knot sampled_raw full-observation edge repair, `input_mdl_knot_dynamic_adapter_irregular_headv3*.py` | Input sampling handoff validator, sampled_raw audit metadata, profile instrumentation, diagnostics counters | `SHORT_DIAGNOSTIC_ONLY` Slurm job running | Local and remote py_compile/gates passed; remote focused pytest passed `60 passed in 204.98s`; final read-only subagent returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_SHORT_DIAGNOSTIC_ONLY`; clean restart gate passed | Remote commit `9183fb7`; active Slurm job `1132502` (`mdl_shortdiag`) running on `g0032`; clean logdir `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_shortdiag_sbatch_9183fb7_20260701_091642_+0800_torchrun_port29683_exclude_g0030`; prior children `1132480/1132482/1132483` are deployment-entry corrections only; no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim | Monitor first finite loss and post-shortdiag validator; formal/full long training remains locked |
| MDL-Knot formal/full train lock consistency fix, `input_mdl_knot_dynamic_adapter_irregular_headv3.py` | Config lock state, launch gate claim locks, tests | Local blocker fix complete; remote PRECHECK_ONLY rerun passed under locked formal state | py_compile passed; launch gate rejects stale unlock/status/sparse-claim states; remote launch gate and launch gate with summary both returned `PRECHECK_ONLY_REQUEST_ALLOWED` | Remote logs under `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_sampledraw_fix_precheck_f5fe3a7/`; `.535` remains failed historical user-override formal evidence only | One-epoch shortdiag only; formal/full long training remains locked |
| MDL-Knot precheck sparse-compute claim-lock schema fix, `audit_mdl_knot_pipeline_precheck.py` | Precheck summary schema, launch gate regression tests | Remote PRECHECK_ONLY passed after schema fix | py_compile passed; generated summary validator emitted `VALIDATED_PRECHECK_SUMMARY`; remote `tests/test_mdl_knot_core.py tests/test_mdl_knot_shortdiag.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py -q` passed as `60 passed in 204.98s` | No GPU/Slurm/train/eval was run by precheck; no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim | Queue one-epoch `SHORT_DIAGNOSTIC_ONLY` when GPU0 is available and BVR is no longer occupying it; formal/full long training remains locked |

## Timeline

### 2026-07-01 09:09:10 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`.
- Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`.
- Final read-only subagent review: `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_SHORT_DIAGNOSTIC_ONLY`; formal/full training, evaluation, `tools/test.py`, checkpoints, mAP/runtime/FLOPs/deploy/paper/sparse-compute claims remain locked.
- Remote runtime worktree: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7`.
- Remote code evidence commit: `9183fb7` (`DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3 remote shortdiag sync`).
- Slurm deployment: submitted bounded one-epoch `SHORT_DIAGNOSTIC_ONLY` as job `1132480` (`mdl_shortdiag`) with `--exclude=g0030`; initial `sacct` shows `RUNNING` on `g0053`, so the protected hold node `g0030` and C3/BVR hold GPUs were not used.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_shortdiag_sbatch_9183fb7_20260701_090910_+0800_exclude_g0030`.
- Strict random-fixed 50% contract: not applicable to this MDL-Knot dynamic route; fixed-pad bridge still makes no sparse-compute claim.
- GT/teacher leakage risk: no new GT, teacher, prediction cache, evaluator, or post-processing access added.
- Current mAP evidence: none.
- Decision: monitor launch sanity and post-shortdiag validator only. Do not promote to formal/full training from this launch without a new gate decision.

### 2026-07-01 09:12:59 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Deployment correction: ordinary child `1132480` was cancelled because the sbatch carried a stale `--request-stage PRECHECK_ONLY` argument that made `launch_gate.log` invalid; ordinary child `1132482` was cancelled because `torch.distributed.run` collided with an already used default port `29400` on shared node `g0053`.
- Clean restart: submitted child `1132483` with direct single-GPU `python tools/train.py` entry, `--exclude=g0030`, and `set -euo pipefail`.
- Clean logdir: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_shortdiag_sbatch_9183fb7_20260701_091259_+0800_direct_python_exclude_g0030`.
- Clean gate evidence: py_compile pass; `launch_gate.log` returned `PRECHECK_ONLY_REQUEST_ALLOWED`; `shortdiag_static_gate.log` returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`; `CUDA_VISIBLE_DEVICES_INITIAL=0` inside Slurm allocation on `g0053`.
- Current status: child `1132483` running at launch window; no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim.
- Decision: monitor first finite loss and post-shortdiag validator. Previous children are deployment-script evidence only, not route failure evidence.

### 2026-07-01 09:16:42 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Deployment correction: ordinary child `1132483` reached `tools/train.py` but failed immediately with `KeyError: 'LOCAL_RANK'` because direct single-process Python does not satisfy this repo's DDP entrypoint contract.
- Clean restart: submitted child `1132502` with `torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29683`, `--exclude=g0030`, and `set -euo pipefail`.
- Clean logdir: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_shortdiag_sbatch_9183fb7_20260701_091642_+0800_torchrun_port29683_exclude_g0030`.
- Clean gate evidence: py_compile pass; `launch_gate.log` returned `PRECHECK_ONLY_REQUEST_ALLOWED`; `shortdiag_static_gate.log` returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`; initial `sacct` shows `RUNNING` on `g0032`, not protected hold node `g0030`.
- Training startup evidence: pretrained checkpoint loaded from `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`; AMP and EMA enabled; `Epoch 0 started`; first `MDL_KNOT_PROFILE` line emitted for sampled_raw scout/handoff. No first loss yet at this record.
- Current mAP evidence: none.
- Decision: continue monitoring first finite loss and post-shortdiag validator. Previous children are deployment-entry evidence only, not route model failure evidence.

### 2026-07-01 08:36:33 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`.
- Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`.
- Remote PRECHECK_ONLY rerun path: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7`.
- Remote logs: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_sampledraw_fix_precheck_f5fe3a7/`.
- Changed files for this record: this tracker, route report, and `research-wiki/log.md`.
- Strict random-fixed 50% contract: not applicable to this MDL-Knot dynamic route; fixed-pad bridge still makes no sparse-compute claim.
- GT/teacher leakage risk: no new GT, teacher, prediction cache, evaluator, or post-processing access added.
- Current mAP evidence: none.
- Remote verification: py_compile pass; base launch gate `PRECHECK_ONLY_REQUEST_ALLOWED`; generated summary plus launch gate `PRECHECK_ONLY_REQUEST_ALLOWED`; shortdiag static gate `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`; focused pytest `60 passed in 204.98s`.
- Decision: remote precheck is passed. Next allowed action is one bounded `SHORT_DIAGNOSTIC_ONLY` attempt when GPU0 is free. Formal/full long training, evaluation, checkpoints, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper/sparse-compute claims remain locked.

### 2026-07-01 07:53:15 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`.
- Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`.
- Remote evidence recorded: N16R4 formal child `1118197.535` failed with `ValueError: selected_inputs must be shorter than dense_T for sparse selected-only audit`.
- Changed files: `opentad/acquisition/mdl_knot/validators.py`, `opentad/acquisition/mdl_knot/selector.py`, `opentad/acquisition/mdl_knot/handoff.py`, `opentad/acquisition/mdl_knot/diagnostics.py`, `opentad/datasets/transforms/end_to_end.py`, `tests/test_mdl_knot_core.py`, `tests/test_mdl_knot_tools_and_integration.py`, this tracker, route report, and `research-wiki/log.md`.
- Strict random-fixed 50% contract: not applicable to this MDL-Knot dynamic route; fixed-pad bridge still makes no sparse-compute claim.
- GT/teacher leakage risk: no new GT, teacher, prediction cache, evaluator, or post-processing access added.
- Current mAP evidence: none.
- Verification: py_compile relevant files passed; `python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_shortdiag.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py -q` passed as `55 passed, 3 skipped in 96.63s`.
- Gate state: `validate_mdl_knot_launch_gate.py` returned `PRECHECK_ONLY_REQUEST_ALLOWED`; `validate_mdl_knot_shortdiag.py` returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` with `validated=false`.
- Decision: remote `PRECHECK_ONLY` is allowed by local gates; one-epoch shortdiag may follow only after remote precheck passes; formal/full long training remains locked.

### 2026-07-01 08:06:19 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`.
- Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`.
- Final review blocker: main formal config still carried stale `formal_train_unlocked=True` / user-override formal queued semantics after failed child `1118197.535`.
- Changed files: `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`, `tools/mdl_knot/validate_mdl_knot_launch_gate.py`, `tests/test_mdl_knot_tools_and_integration.py`, route report, tracker, and `research-wiki/log.md`.
- Strict random-fixed 50% contract: not applicable to this MDL-Knot dynamic route; fixed-pad bridge still makes no sparse-compute claim.
- GT/teacher leakage risk: no new GT, teacher, prediction cache, evaluator, or post-processing access added.
- Current mAP evidence: none.
- Verification: py_compile relevant config/gate/test files passed; `python -m pytest tests/test_mdl_knot_tools_and_integration.py -q` passed as `19 passed, 3 skipped in 19.18s`.
- Gate state: `validate_mdl_knot_launch_gate.py` returned `PRECHECK_ONLY_REQUEST_ALLOWED` with `formal_train_unlocked=false`; `validate_mdl_knot_shortdiag.py` returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` with `validated=false`.
- Decision: historical `.535` formal user-override run is failed evidence only; no full-train permission exists. Next action remains remote `PRECHECK_ONLY` then one-epoch shortdiag only after precheck passes; formal/full long training remains locked.

### 2026-07-01 08:24:45 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`.
- Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`.
- Remote PRECHECK_ONLY feedback: py_compile, launch gate, and shortdiag gate passed; pytest failed at `test_launch_gate_unlocks_only_for_valid_precheck_summary` because generated precheck summary did not explicitly lock `no_claims.sparse_compute`.
- Changed files: `tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py`, `tests/test_mdl_knot_tools_and_integration.py`, route report, tracker, and `research-wiki/log.md`.
- Strict random-fixed 50% contract: not applicable to this MDL-Knot dynamic route; fixed-pad bridge still makes no sparse-compute claim.
- GT/teacher leakage risk: no new GT, teacher, prediction cache, evaluator, or post-processing access added.
- Current mAP evidence: none.
- Verification: py_compile relevant files passed; targeted `test_launch_gate_unlocks_only_for_valid_precheck_summary` passed; generated precheck summary plus launch gate passed as `PRECHECK_ONLY_REQUEST_ALLOWED`; `tests/test_mdl_knot_tools_and_integration.py` passed as `19 passed, 3 skipped in 19.96s`.
- Decision: rerun remote `PRECHECK_ONLY` only. One-epoch shortdiag may follow only after remote precheck passes; formal/full long training remains locked.
