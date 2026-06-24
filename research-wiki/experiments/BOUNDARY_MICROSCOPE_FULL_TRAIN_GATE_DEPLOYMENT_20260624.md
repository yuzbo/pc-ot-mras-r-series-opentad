# Boundary Microscope Full-Train Gate Deployment 2026-06-24

## Scope

- Route label: `DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3`
- Stage: full-train gate/deployment owner action.
- Writable owner worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BoundaryMicroscope_PrecheckDeploy_Worktree_20260624`
- Branch: `codex/boundary-microscope-precheck-deploy-20260624`
- Prior pushed head: `b2a6a6a90dc617a2fed8eef52b9e5518cae4d320`
- Prior remote precheck path: `/data/home/sczc063/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf`
- Prior remote precheck commit actually run: `db33bafdbe398b15f425028800ae3f2da5d6d9be`
- Prior remote precheck result: `PASS`, `BOUNDARY_MICROSCOPE_PRECHECK_ONLY_PASS_NO_TRAIN`
- Changed surface: launcher/gate enforcement, tests, and route-owned deployment documentation.
- Selector core: unchanged.
- Detector head/loss/assignment/post-processing/evaluator: unchanged.
- Route separation: this is Boundary Microscope only and must not be merged with C3, BH-SDC, Frame, Event, or combo attribution.

## Gate Design

- Default state remains locked: `PRECHECK_ONLY=1` exits before `tools/train.py`.
- Full train requires all of the following:
  - `PRECHECK_ONLY=0`;
  - `ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1`;
  - `BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON`;
  - `BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256`;
  - validator action `--action full-train`;
  - exact `RUN_TAG` match;
  - exact active SHA256 manifest hash match;
  - exact resolved config SHA256 match;
  - exact route label match;
  - exact user override statement:
    `USER_EXPLICITLY_REQUESTED_BOUNDARY_MICROSCOPE_FULL_TRAIN_CANDIDATE_ON_2026-06-24`.
- Full-train gate continues to reject:
  - `tools/test.py`;
  - raw-prediction load/save or cache shortcuts;
  - test-time GT, teacher, or oracle shortcuts;
  - detector-mAP, runtime, deploy, or paper claims.

## Local Verification Log

Initial RED check before implementation:

```text
python -m pytest tests/test_boundary_microscope_config_gate.py -q -rs
3 failed, 10 passed in 1.55s
Expected failures: validator lacked --action/--run-tag support and full-train launcher file was absent.
```

Post-implementation focused test:

```text
python -m pytest tests/test_boundary_microscope_config_gate.py -q -rs
13 passed in 2.33s
```

Additional local verification:

```text
python -m py_compile tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py tests/test_boundary_microscope_config_gate.py opentad/models/selectors/boundary_microscope_acquisition_route.py
exit=0
```

```text
python tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py --json
status=BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS
allows_full_train=false
allows_slurm=false
allows_gpu=false
allows_tools_train=false
```

```text
bash -n scripts/run_boundary_microscope_precheck_n16r4.sbatch
exit=0
```

```text
bash -n scripts/run_boundary_microscope_full_train_n16r4.sbatch
exit=0
```

```text
python tools/bata/validate_boundary_microscope_gate.py --action full-train --gate-json <temp> --gate-sha256 <temp_sha> --active-manifest-sha256 manifest-sha --resolved-config-sha256 resolved-sha --run-tag boundary_microscope_full_train_fixed_run_tag --json
status=BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_VALIDATION_PASS
allows_tools_train=true
allows_slurm=true
allows_gpu=true
allows_full_train=true
metric_claim_allowed=false
paper_claim_allowed=false
```

Commit, push, remote deployment, gate JSON, and Slurm fields are recorded below after execution.

## Remote Deployment Fields

- Final implementation commit: pending.
- Push result: pending.
- Remote path: pending.
- Sync method: pending.
- Remote code commit used for full-train launcher: pending.
- `RUN_TAG`: pending.
- Gate JSON path: pending.
- Gate JSON SHA256: pending.
- Resolved config SHA256: pending.
- Active manifest SHA256: pending.
- Validator action result: pending.
- Slurm job id/status/log path: pending.

## Decisions

- Pro/Claude/Gemini review gates were not used for this action by user instruction.
- Rosetta/Oracle Pro transport failure is not a blocker for this route-owned full-train gate/deployment action.
- Shared repo remains read-only.
