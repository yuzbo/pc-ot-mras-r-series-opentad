# Boundary Microscope Deployment Precheck 2026-06-24

## Scope

- Route label: `DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3`
- Stage: deployment precheck only.
- Writable owner worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BoundaryMicroscope_PrecheckDeploy_Worktree_20260624`
- Branch: `codex/boundary-microscope-precheck-deploy-20260624`
- Base commit: `1328663e611c935e2f7adc1d635e4abb8372cdb1`
- Deployment commit: `db33bafdbe398b15f425028800ae3f2da5d6d9be`
- Changed surface: launcher/gate enforcement and deployment-precheck documentation only.
- Selector core: unchanged.
- Full train: locked.

## Changed Files

- `scripts/run_boundary_microscope_precheck_n16r4.sbatch`
- `tests/test_boundary_microscope_config_gate.py`
- `docs/en/boundary_microscope_acquisition_route_review_context_20260624.md`

## Local Verification

Passed:

```text
python -m pytest tests/test_boundary_microscope_config_gate.py -q -rs
10 passed in 1.36s
```

```text
python tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py --json
status=BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS
allows_full_train=false
allows_remote_sync=false
allows_slurm=false
allows_gpu=false
```

```text
python tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py --json
status=BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS
allows_full_train=false
allows_remote_sync=false
allows_slurm=false
allows_gpu=false
```

```text
python -m py_compile tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py tests/test_boundary_microscope_config_gate.py opentad/models/selectors/boundary_microscope_acquisition_route.py
exit=0
```

```text
bash -n scripts/run_boundary_microscope_precheck_n16r4.sbatch
exit=0
```

Blocked by local Windows torch DLL:

```text
python -m pytest tests/test_boundary_microscope_acquisition_route.py tests/test_boundary_microscope_actionformer_integration.py tests/test_boundary_microscope_registry_build.py -q -rs
2 failed, 1 passed, 8 skipped
Failure reason: Windows torch import OSError / access violation while loading torch\lib\c10.dll.
```

## Remote Sync

- Method: GitHub branch clone, no zip.
- GitHub branch reachability:

```text
git ls-remote https://github.com/yuzbo/pc-ot-mras-r-series-opentad.git refs/heads/codex/boundary-microscope-precheck-deploy-20260624
db33bafdbe398b15f425028800ae3f2da5d6d9be
```

- Remote path:

```text
/data/home/sczc063/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf
```

- Remote HEAD:

```text
db33bafdbe398b15f425028800ae3f2da5d6d9be
```

## Remote PRECHECK_ONLY

Command:

```bash
cd ~/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf
ALLOW_LOGIN_NODE_DEBUG=1 PRECHECK_ONLY=1 RUN_TAG=boundary_microscope_precheck_remote_20260624_db33baf bash scripts/run_boundary_microscope_precheck_n16r4.sbatch
```

Result:

```text
status=PASS
decision=BOUNDARY_MICROSCOPE_PRECHECK_ONLY_PASS_NO_TRAIN
slurm_job_id=null
host=ln01
reason=PRECHECK_ONLY=1; launcher exited before tools/train.py, detector eval, data access, or metric claim
```

Remote log root:

```text
/data/home/sczc063/run/yuzibo/OpenTAD_BoundaryMicroscope_PrecheckDeploy_20260624_db33baf/logs/boundary_microscope_precheck_remote_20260624_db33baf
```

Remote artifacts:

```text
active_sha256_manifest.txt
boundary_microscope_precheck_remote_20260624_db33baf.log
boundary_microscope_precheck_summary.json
config_gate_report.json
resolved_config.py
```

Remote PRECHECK_ONLY checks completed:

- fail-closed config validator;
- resolved-config audit;
- active SHA256 manifest;
- static `py_compile`;
- focused Boundary Microscope config gate pytest: `10 passed in 2.64s`.

## Decision

- Remote sync succeeded.
- Remote `PRECHECK_ONLY=1` succeeded.
- No Slurm full-training job was submitted.
- `tools/train.py` was not executed.
- `tools/test.py` was not executed.
- Full train remains locked and requires a separate code-owner full-train gate before any future execution.
