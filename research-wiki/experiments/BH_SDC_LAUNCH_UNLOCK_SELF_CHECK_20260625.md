# BH-SDC Launch-Unlock Self-Check

Timestamp: 2026-06-25 Asia/Shanghai

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

Owned worktree:
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BHSDC_LaunchUnlock_Worktree_20260625`

Owned branch: `codex/bh-sdc-launch-unlock-20260625`

Reviewed base implementation commit:
`2cebd955c2df7552f1291a179bfab9c892e9c5f5`

## Purpose

Unlock the BH-SDC full-train candidate after explicit user/coordinator override
without depending on a Pro follow-up. The unlock is for BH-SDC formal training
only and must remain auditable:

- static config validation passes;
- `PRECHECK_ONLY=1` remains the launcher default and exits before training after
  producing resolved config and active manifest hashes;
- `PRECHECK_ONLY=0` requires `ALLOW_BH_SDC_FULL_TRAIN=1`, gate JSON, gate
  SHA256, matching resolved config SHA256, matching active manifest SHA256, and
  staged THUMOS14 data counts before `torchrun`;
- no direct `tools/test.py`, detector mAP claim, metric/paper/runtime/deploy
  claim, checkpoint load, pretrained initialization, resume, raw-prediction
  cache, test-time GT, validation/test teacher, or oracle path is authorized.

## Changed Files

- `configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py`
- `tools/bata/validate_bh_sdc_full_train_gate.py`
- `scripts/run_bh_sdc_full_train_n16r4.sbatch`
- `tests/test_bh_sdc_config_gate.py`
- `tests/test_bh_sdc_launcher_gate.py`
- `docs/en/bh_sdc_route_original_purpose_and_drift_check_20260624.md`
- `docs/en/bh_sdc_n16r4_launch_gate_review_context_20260624.md`
- `research-wiki/experiments/BH_SDC_LAUNCH_UNLOCK_SELF_CHECK_20260625.md`

## Surface Attribution

- Input sampling / dynamic acquisition mechanism: unchanged.
- Adapter/backbone/neck/head internals: unchanged.
- Detector loss/assignment/post-processing: unchanged.
- Launch/deployment gate logic: changed.
- Config authorization metadata: changed.
- Tests/docs: changed.

Any future metric change from this commit alone should be attributed to enabling
the already implemented BH-SDC route to run, not to a detector or acquisition
mechanism change.

## Static Gate Contract

The full-train candidate config now carries:

- `launch_gate_passed=True`
- `static_authorization=True`
- `external_payload_required=True`
- `allow_remote_sync=True`
- `allow_slurm=True`
- `allow_gpu=True`
- `allow_tools_train=True`
- `allow_train_validation_map=True`
- `allow_long_training=True`
- `allow_full_train=True`
- `allow_dataset_access=True`
- `allow_checkpoint_write=True`
- `allowed_entrypoints=("tools/train.py",)`
- exact command whitelist for N16R4 sync, sbatch submission, and
  `tools/train.py`

Still false:

- `allow_tools_test`
- `allow_detector_map`
- `allow_metric_claim`
- `allow_paper_claim`
- `runtime_flops_claim_allowed`
- `deploy_claim_allowed`
- `allow_checkpoint_load`
- `allow_pretrained_initialization`
- `allow_resume`
- `allow_raw_prediction_cache`

## Payload Contract

`validate_launch_gate_payload` no longer unconditionally raises. It requires a
strict JSON payload binding:

- route, stage, and route label;
- explicit user override decision;
- final read-only review verdict and id;
- reviewed base implementation commit
  `2cebd955c2df7552f1291a179bfab9c892e9c5f5`;
- launch gate commit;
- resolved config SHA256;
- active manifest SHA256;
- exact command whitelist;
- remote workspace `~/run/yuzibo/OpenTAD_Back_check`;
- `max_gpus=1`, `max_nodes=1`, `max_time_hours<=48`, `max_epochs<=60`;
- checkpoint write policy `route_work_dir_only`;
- checkpoint load, pretrained initialization, and resume policies `none`;
- dataset scope `THUMOS14_TAD_ONLY_TRAIN200_VALTEST211`;
- all required false leakage/claim/cache flags.

## N16R4 Data Gate

Formal training checks staged data only after payload validation and only when
`PRECHECK_ONLY=0`:

- train directory default: `~/run/yuzibo/thumos14/train`, expected 200 mp4
  files;
- test directory default: `~/run/yuzibo/thumos14/test`, expected 211 mp4 files;
- count overrides:
  `EXPECTED_THUMOS14_TRAIN_COUNT` and `EXPECTED_THUMOS14_TEST_COUNT`;
- overrides must be positive.

`PRECHECK_ONLY=1` does not require non-empty data directories, so the
coordinator can run static precheck before manifest/symlink staging.

## Leakage And Claim Risk

No new raw prediction shortcut, validation/test teacher/oracle path, test-time
GT path, `tools/test.py` entrypoint, checkpoint load, pretrained init, or resume
path was added. The route continues to rely on the standard OpenTAD
`tools/train.py` train/validation protocol only.

## Verification

Executed from the owned worktree.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py tools/bata/validate_bh_sdc_full_train_gate.py tests/test_bh_sdc_config_gate.py tests/test_bh_sdc_launcher_gate.py
```

Result: passed.

```powershell
$bash = (Get-Command bash -ErrorAction SilentlyContinue).Source; if ($bash -and ($bash.ToLower() -notlike '*windows\system32\bash*')) { & $bash -n scripts/run_bh_sdc_full_train_n16r4.sbatch } else { Write-Output 'bash unavailable or WSL bash skipped' }
```

Result: `bash unavailable or WSL bash skipped`. The WSL bash path was not used,
following the workspace rule.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_bh_sdc_config_gate.py tests/test_bh_sdc_launcher_gate.py -q -rs
```

Result: `17 passed in 55.58s`.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_bh_sdc_config_gate.py tests/test_bh_sdc_launcher_gate.py tests/test_bh_sdc_actionformer_integration.py tests/test_bh_sdc_core.py tests/test_bh_sdc_metadata_contract.py -q -rs
```

Result: `29 passed in 55.22s`.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe tools/bata/validate_bh_sdc_full_train_gate.py configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --json
```

Result: passed static config validation with:

- `ok=true`
- `authorized=false`
- `payload_required_for_full_train=true`
- `claims_allowed=false`
- `launch_gate_passed=true`
- `allowed_entrypoints=["tools/train.py"]`
- `resolved_config_sha256=87378150220ca66ccc2df17134647777bf68d92485114a29ba18ac9387b3fbf6`

```powershell
git diff --check
```

Result: passed; only Windows line-ending warnings were printed.
