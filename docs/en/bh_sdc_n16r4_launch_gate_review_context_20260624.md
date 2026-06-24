# BH-SDC Follow-Up Pro-Fix Review Context

Date: 2026-06-25

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

Decision requested from Pro: review the owner-fix commit after the 2026-06-24
blocking review. Decide whether the BH-SDC implementation is fixed for local
smoke and whether any additional local fixes are required. Do not approve
remote sync, Slurm, full training, detector mAP, runtime/FLOPs, deployment, or
paper claims from this context.

## Branch And Base

- Owned branch: `codex/bh-sdc-pro-fix-20260625`
- Owned worktree: `OpenTAD_BHSDC_ProFix_Worktree_20260625`
- Reviewed implementation base commit: the prior launch-gate review inspected
  `02c6b983ec6dd29b2890309c33f536f6a4cbc612`.
- Owner-fix commit under review: use the pushed branch HEAD commit that
  contains this document.

The previous Pro review found that the full-train gate provenance was invalid
and that the bridge build path needed direct proof. This commit is not a
launch-gate candidate. It is a fail-closed implementation/provenance repair.

## Files To Inspect

- `configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py`
- `tools/bata/validate_bh_sdc_full_train_gate.py`
- `tests/test_bh_sdc_config_gate.py`
- `tests/test_bh_sdc_actionformer_integration.py`
- `tests/test_bh_sdc_core.py`
- `tests/test_bh_sdc_metadata_contract.py`
- `docs/en/bh_sdc_route_original_purpose_and_drift_check_20260624.md`
- `docs/en/bh_sdc_n16r4_launch_gate_review_context_20260624.md`

Relevant implementation context from the reviewed base:

- `opentad/models/selectors/bh_sdc_frame_selector.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/utils/training_guard.py`

## Why Full Training Is Locked

After the Pro review, the full-train candidate is locked again:

- `launch_gate_passed=false`
- `allowed_entrypoints=()`
- `command_whitelist=()`
- remote sync, Slurm, GPU, `tools/train.py`, full-train, dataset access,
  train-validation mAP, and checkpoint writes are false
- `reviewed_impl_commit=FOLLOWUP_PRO_REQUIRED_AFTER_BH_SDC_PRO_FIX`

The validator rejects every launch payload before schema validation with a
pending follow-up Pro error. Old implementation reviews, old launch payloads,
and the former decision string
`ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1` do not authorize any
execution.

## Fixed Local Scope

The current status string is:

`FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`

The implementation now has a real local build/forward smoke for:

- BH-SDC selector;
- compact per-sample backbone path;
- `PCOTMRASBoundaryHazardSparseToDenseBridge`;
- projection;
- detector head.

It also adds tests for route isolation, mask/selected-index/physical-time
metadata, and no GT/teacher/oracle/cache/raw-prediction payloads in test mode.

## Still Locked

The following remain locked until a new substantive follow-up Pro review
explicitly says otherwise and the local tracker/report is updated:

- remote sync;
- Slurm;
- GPU use;
- `tools/train.py`;
- `tools/test.py`;
- full train;
- detector mAP or train-validation mAP claim;
- checkpoint writes;
- runtime/FLOPs/deploy/paper claims.

## Local Verification To Check

Expected local checks for this owner-fix commit:

```powershell
python -m pytest tests\test_bh_sdc_config_gate.py tests\test_bh_sdc_actionformer_integration.py tests\test_bh_sdc_core.py tests\test_bh_sdc_metadata_contract.py -q
python tools/bata/validate_bh_sdc_full_train_gate.py configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --json
python -m py_compile tools/bata/validate_bh_sdc_full_train_gate.py
```

The validator command without `--gate-json` is expected to fail closed and exit
nonzero while printing JSON with `authorized=false` and the
`FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO` status.

## Requested Pro Answer Schema

- `Context verdict`
- `Model evidence`
- `Inspected materials`
- `Verdict`
- `Blocking findings`
- `Non-blocking findings`
- `Required fixes or next experiments`
- `Accepted launch/sync/Slurm decision`

Pro should answer whether this commit is sufficient as a local-smoke repair and
which minimum fixes, if any, are still required before a separate future launch
permission discussion. Pro should not approve sync, Slurm, full training, or
mAP claims in this review.
