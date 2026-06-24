# BH-SDC N16R4 Launch-Unlock Review Context

Date: 2026-06-25

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

This document records the launch-unlock state after explicit user/coordinator
override. It is not a Pro-review request and does not depend on a Pro follow-up.
The only approved route label is
`DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`.

## Branch And Base

- Owned branch: `codex/bh-sdc-launch-unlock-20260625`
- Owned worktree: `OpenTAD_BHSDC_LaunchUnlock_Worktree_20260625`
- Reviewed base implementation commit:
  `2cebd955c2df7552f1291a179bfab9c892e9c5f5`

The branch changes only the BH-SDC launch gate, validator, launcher, tests, and
review documents. It must not be described as C3/C3-Pro, interval packet,
dynamic budget guard, physical-grid ActionFormer, or a combo route.

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

## Launch-Unlock Contract

The full-train candidate config is statically unlocked for BH-SDC formal
training only:

- `launch_gate_passed=true`
- `static_authorization=true`
- `external_payload_required=true`
- `allowed_entrypoints=("tools/train.py",)`
- command whitelist:
  `REMOTE_SYNC_TO_N16R4:~/run/yuzibo/OpenTAD_Back_check`,
  `sbatch scripts/run_bh_sdc_full_train_n16r4.sbatch`, and
  `python tools/train.py configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --id 0`
- allowed: remote sync, Slurm, one GPU, `tools/train.py`, standard
  train-time validation through `tools/train.py`, dataset access, route-workdir
  checkpoint writes, long/full training
- forbidden: `tools/test.py`, direct detector mAP claims, metric/paper/runtime/
  deploy claims, checkpoint load, pretrained initialization, resume,
  raw-prediction cache, test-time GT, validation/test teacher, oracle use

The validator command without a gate payload is expected to pass static config
validation and return `authorized=false`. Formal training is authorized only
when the payload, payload SHA256, resolved config SHA256, and active manifest
SHA256 all match.

Required launch payload fields:

- `schema_version=1`
- `decision=ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1`
- `explicit_user_override_decision=ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1`
- `route=bh_sdc_boundary_hazard_sparse_dense`
- `route_label=DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`
- `stage=bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4`
- `reviewed_impl_commit=2cebd955c2df7552f1291a179bfab9c892e9c5f5`
- `launch_gate_commit=<40 hex>`
- `final_read_only_review_verdict=PASS_SUBAGENT_FINAL_REVIEW_ONLY`
- `final_read_only_review_id=<non-empty id>`
- `resolved_config_sha256=<64 hex>`
- `active_sha256_manifest_sha256=<64 hex>`
- exact command whitelist, remote workspace, Slurm partition `gpu`,
  `max_gpus=1`, `max_nodes=1`, `max_time_hours<=48`, `max_epochs<=60`,
  train command, checkpoint/data policies, and all required true/false
  permission and leakage flags.

## N16R4 Data Staging Gate

`PRECHECK_ONLY=1` remains the default and does not require non-empty data
directories. It exists to generate the resolved config hash and active manifest
hash before formal training.

`PRECHECK_ONLY=0` requires:

- `ALLOW_BH_SDC_FULL_TRAIN=1`
- gate JSON and SHA256
- payload validation success
- `~/run/yuzibo/thumos14/train` containing 200 staged/symlinked mp4 files by
  default
- `~/run/yuzibo/thumos14/test` containing 211 staged/symlinked mp4 files by
  default

The expected counts may be overridden with
`EXPECTED_THUMOS14_TRAIN_COUNT` and `EXPECTED_THUMOS14_TEST_COUNT`, but they
must remain positive. The current raw source locations are
`~/run/yuzibo/raw/Validation Data/validation` for the 200 training-subset videos
and `~/run/yuzibo/raw/Test Data/TH14_test_set_mp4` for the 213 raw test videos,
of which 211 are expected in the annotated validation/test manifest.

## Local Verification To Check

Expected local checks for this launch-unlock commit:

```powershell
python -m pytest tests\test_bh_sdc_config_gate.py tests\test_bh_sdc_actionformer_integration.py tests\test_bh_sdc_core.py tests\test_bh_sdc_metadata_contract.py -q
python tools/bata/validate_bh_sdc_full_train_gate.py configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --json
python -m py_compile tools/bata/validate_bh_sdc_full_train_gate.py
```

The validator command without `--gate-json` is expected to pass static config
validation, exit zero, and print JSON with `authorized=false` and
`payload_required_for_full_train=true`.
