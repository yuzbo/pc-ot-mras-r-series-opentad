# BH-SDC N16R4 Launch-Gate Review Context

Date: 2026-06-24

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

Decision requested from Pro: review this launch-gate commit and decide whether
it is safe to permit N16R4 remote sync plus one Slurm full-train candidate for
the BH-SDC route.

## Branch And Base

- Owned branch: `codex/bh-sdc-launch-gate-20260624`
- Reviewed implementation base commit: `ae4354307d903f537e2be78723c39a3e19787f9b`
- Launch-gate commit under review: use the pushed branch HEAD commit that
  contains this document.

The reviewed implementation base passed the second GPT-5.5 Pro implementation
review and final read-only implementation review. That base did not approve
remote sync, Slurm, GPU training, direct detector evaluation, result claims, or
paper claims. This commit is a separate launch-gate candidate and must receive a
new Pro launch review before any real execution.

## Files To Inspect

- `configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py`
- `tools/bata/validate_bh_sdc_full_train_gate.py`
- `scripts/run_bh_sdc_full_train_n16r4.sbatch`
- `tests/test_bh_sdc_config_gate.py`
- `tests/test_bh_sdc_launcher_gate.py`
- `docs/en/bh_sdc_route_original_purpose_and_drift_check_20260624.md`
- `docs/en/bh_sdc_n16r4_launch_gate_review_context_20260624.md`

Relevant implementation context from the reviewed base:

- `opentad/models/selectors/bh_sdc_frame_selector.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/utils/training_guard.py`

## Why The Previous State Could Not Launch

Before this launch-gate commit, the BH-SDC candidate was intentionally locked
for implementation review only:

- `launch_gate_passed=false`
- `allowed_entrypoints=()`
- remote sync, Slurm, GPU, full-train, and dataset access flags were false
- the sbatch script requested no GPU
- the sbatch script called only the static validator and never called the train
  entrypoint

That state prevented accidental training and was correct for code review. It
was not a launch request.

## New Gate Semantics

The new decision string is:

`ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1`

The config is now an entrypoint-gated full-train candidate:

- `tools/train.py` is the only detector entrypoint allowed by the config.
- `tools/test.py` remains rejected.
- direct detector mAP, runtime/FLOPs, deploy, paper, and metric claims remain
  rejected before results exist.
- raw prediction cache loading/saving remains disabled.
- checkpoint load, pretrained initialization, and resume remain disabled.
- checkpoint writing is allowed only in the route work directory during the
  reviewed train run.
- the config still has `static_authorization=false` and requires an external
  launch payload.

No payload means no authorization. Old implementation/precheck decisions do not
authorize training.

## Required Payload Evidence

The validator requires an exact JSON object with all required keys and rejects
unknown keys. Required evidence includes:

- route and route label
- reviewed implementation commit
- launch-gate commit under review
- Pro implementation review verdict/session
- final read-only implementation review verdict/id
- Pro launch-gate verdict/session
- explicit user/Pro launch decision
- resolved config SHA256
- active manifest SHA256
- exact command whitelist
- N16R4 remote workspace path
- Slurm partition/resource bounds
- no GT/test leakage assertion
- no teacher/oracle assertion
- no raw prediction cache assertion
- no checkpoint/pretrain/resume policies except route-scoped checkpoint writing

The exact allowed command whitelist is:

- `REMOTE_SYNC_TO_N16R4:~/run/yuzibo/OpenTAD_Back_check`
- `sbatch scripts/run_bh_sdc_full_train_n16r4.sbatch`
- `python tools/train.py configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --id 0`

Any additional command, including a direct evaluation entrypoint, must be
rejected.

## Sbatch Behavior

The sbatch script is conservative:

- single node, single GPU, public `gpu` partition
- default `PRECHECK_ONLY=1`, which performs static checks and exits before
  training
- `PRECHECK_ONLY=0` requires `ALLOW_BH_SDC_FULL_TRAIN=1`
- training mode requires `OPENTAD_BH_SDC_GATE_JSON` and
  `OPENTAD_BH_SDC_GATE_SHA256`
- the script computes a resolved config dump and active SHA256 manifest
- the script validates the payload against the manifest, resolved config hash,
  requested action, and reviewed train command before exporting gate env vars
  for `training_guard`
- arbitrary cfg-options, checkpoint/load/resume shortcuts, and raw prediction
  cache shortcuts are refused

The script has not been submitted. No remote sync, Slurm job, or full training
has been started by this commit.

## Local Verification To Check

Expected local checks for this launch-gate commit:

```powershell
conda run -n torch_1 python -m pytest tests/test_bh_sdc_config_gate.py tests/test_bh_sdc_launcher_gate.py -q
python tools/bata/validate_bh_sdc_full_train_gate.py configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --json
python -m py_compile tools/bata/validate_bh_sdc_full_train_gate.py
bash -n scripts/run_bh_sdc_full_train_n16r4.sbatch
git diff --check -- configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py tools/bata/validate_bh_sdc_full_train_gate.py scripts/run_bh_sdc_full_train_n16r4.sbatch tests/test_bh_sdc_config_gate.py tests/test_bh_sdc_launcher_gate.py docs/en/bh_sdc_route_original_purpose_and_drift_check_20260624.md docs/en/bh_sdc_n16r4_launch_gate_review_context_20260624.md
```

The validator command without `--gate-json` is expected to fail closed and exit
nonzero while printing JSON with `authorized=false`.

## Requested Pro Answer Schema

- `Context verdict`
- `Model evidence`
- `Inspected materials`
- `Verdict`
- `Blocking findings`
- `Non-blocking findings`
- `Required fixes or next experiments`
- `Accepted launch/sync/Slurm decision`

Pro should answer whether this commit is sufficient to allow remote sync to
`~/run/yuzibo/OpenTAD_Back_check` and one N16R4 Slurm full-train candidate for
BH-SDC, or whether fixes are required first.
