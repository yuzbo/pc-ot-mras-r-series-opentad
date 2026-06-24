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

Commit, push, remote deployment, gate JSON, and Slurm fields are recorded below.

## Remote Deployment Fields

- Final implementation commit: `1e286e702dfe111becc5f367263834d06250812f`
- Push result:
  `b2a6a6a..1e286e7 codex/boundary-microscope-precheck-deploy-20260624 -> codex/boundary-microscope-precheck-deploy-20260624`
- Remote path:
  `/data/home/sczc063/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf`
- Sync method:
  GitHub `git fetch` reached `origin/codex/boundary-microscope-precheck-deploy-20260624`.
  `git pull --ff-only` hit `GnuTLS recv error (-110)`. The deployment used a
  non-zip git-native fallback: `git merge --ff-only origin/codex/boundary-microscope-precheck-deploy-20260624`.
- Remote code commit used for full-train launcher:
  `1e286e702dfe111becc5f367263834d06250812f`
- Fixed `RUN_TAG`:
  `boundary_microscope_full_train_candidate_20260624_1e286e7`
- Precheck manifest command:
  `ALLOW_LOGIN_NODE_DEBUG=1 PRECHECK_ONLY=1 RUN_TAG=boundary_microscope_full_train_candidate_20260624_1e286e7 bash scripts/run_boundary_microscope_full_train_n16r4.sbatch`
- Precheck manifest result:
  `BOUNDARY_MICROSCOPE_PRECHECK_ONLY_PASS_NO_TRAIN`, `13 passed in 3.41s`
- Gate JSON path:
  `/data/home/sczc063/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf/logs/boundary_microscope_full_train_candidate_20260624_1e286e7/boundary_microscope_full_train_gate.json`
- Gate JSON SHA256:
  `d33ca02a9f94dc2ea3f0e8a133941bca7c16d8549543d5316801223e4a9fec70`
- Resolved config SHA256:
  `35008d241952bad31bc05c329ffacdf86f887fe7f9f300c17445f85e59e1094d`
- Active manifest SHA256:
  `70e53ac4eb87165e4d10bdc79967652f7a3c8bed90a7e0852321304c0b7d7afa`
- Validator action result:
  `BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_VALIDATION_PASS`
- Slurm submission command:
  `sbatch --export=ALL,PRECHECK_ONLY=0,ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1,RUN_TAG=boundary_microscope_full_train_candidate_20260624_1e286e7,BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON=logs/boundary_microscope_full_train_candidate_20260624_1e286e7/boundary_microscope_full_train_gate.json,BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256=d33ca02a9f94dc2ea3f0e8a133941bca7c16d8549543d5316801223e4a9fec70,TRAIN_ID=0 scripts/run_boundary_microscope_full_train_n16r4.sbatch`
- Slurm job id/status:
  `1117987`, `PENDING`, reason `Priority`
- Slurm stdout path:
  `/data/home/sczc063/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf/logs/bm_fulltrain-1117987.out`
- Slurm stdout head:
  not created yet at first post-submit check because the job was still pending.
- Run log path:
  `/data/home/sczc063/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf/logs/boundary_microscope_full_train_candidate_20260624_1e286e7/boundary_microscope_full_train_candidate_20260624_1e286e7.log`

## Decisions

- Pro/Claude/Gemini review gates were not used for this action by user instruction.
- Rosetta/Oracle Pro transport failure is not a blocker for this route-owned full-train gate/deployment action.
- Shared repo remains read-only.
