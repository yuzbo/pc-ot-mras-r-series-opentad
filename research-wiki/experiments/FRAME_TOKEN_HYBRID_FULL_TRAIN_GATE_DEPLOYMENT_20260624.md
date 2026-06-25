# Frame/Token Hybrid Full-Train Gate Deployment

Route label:
`DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`

This report is scoped only to the divergent Frame/Token Hybrid route. It is not
C3, BH-SDC, Event, Boundary, or a combo route.

## Local Full-Train Gate Implementation

Timestamp: 2026-06-24T21:05:00+08:00

Owner scope:

- Owned local worktree:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_PrecheckDeploy_Worktree_20260624`
- Branch: `codex/frame-token-precheck-deploy-20260624`
- Base before this gate implementation:
  `7dcb6aec5fccab0b022df7d018c70853b450a7cf`
- Shared repo was not written.
- Pro/Claude/Gemini review or transport was not used.

Changed surface:

- `tools/bata/validate_frame_token_hybrid_gate.py`
  - Added explicit gate actions: `precheck` and `full-train`.
  - `precheck` remains bound to `ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`.
  - `full-train` is bound to `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN`.
  - Full-train gate requires exact active manifest SHA256, resolved config
    SHA256, RUN_TAG, route label, user override statement, coordinator override
    statement, and false GT/teacher/oracle/raw-cache/metric/paper flags.
- `scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch`
  - New fail-closed N16R4 full-train launcher.
  - Defaults to `PRECHECK_ONLY=1` and exits before training.
  - With `PRECHECK_ONLY=0`, requires
    `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1`,
    `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON`, and
    `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256` before running
    `tools/train.py`.
- `scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch`
  - Updated default route-owned repo path to
    `OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27`.
- `tests/test_frame_token_hybrid_config_gate.py`
  - Added missing full-train gate JSON fail-closed coverage.
  - Added valid full-train gate JSON authorization coverage.
  - Added wrong RUN_TAG and missing override rejection coverage.
  - Added full-train launcher default-lock and gate-binding coverage.
- `docs/en/frame_token_hybrid_acquisition_route_review_context_20260624.md`
  - Updated the route context from precheck-only to default-locked plus
    external full-train gate authorization semantics.

Protocol and leakage status:

- Config default remains fail-closed:
  `ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`, `allow_full_train=False`,
  `allow_tools_train=False`, `allow_slurm=False`, `allow_gpu=False`.
- Full train is still locked by default.
- Full train can open only via the external full-train gate JSON plus matching
  SHA256 inputs and `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1`.
- No GT/test teacher/oracle/raw-prediction-cache shortcut is allowed by the
  gate.
- Metric and paper claims remain locked.

Local verification:

```text
python tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py --json
PASS: emitted fail-closed precheck-only config JSON.

python -m py_compile tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py tests/test_frame_token_hybrid_config_gate.py opentad/models/selectors/frame_token_hybrid_acquisition_route.py
PASS.

python -m pytest tests/test_frame_token_hybrid_acquisition_route.py tests/test_frame_token_hybrid_metadata_contract.py tests/test_frame_token_hybrid_actionformer_integration.py tests/test_frame_token_hybrid_config_gate.py -q
PASS: 13 passed, 15 skipped in 24.32s.

bash -n scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch
PASS.
```

Launch decision after local gate:

- Local gate status: `PASS`.
- Allowed next action: deploy route-owned repo update, generate a fixed RUN_TAG
  precheck manifest and matching full-train gate JSON on N16R4, validate both
  gate actions, and submit one Slurm full-train candidate if validation passes.

## Remote Deployment And Full-Train Submission

Timestamp: 2026-06-24T20:45:00+08:00

Local commit and push:

- Implementation commit:
  `c1d41a86527acd9d80762f012ac12fe4aa8b7b92`
  (`Add frame token full train gate launcher`)
- Push target:
  `origin/codex/frame-token-precheck-deploy-20260624`
- Push result: `7dcb6ae..c1d41a8`, successful.

Remote route-owned repo update:

- Remote path:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27`
- Sync method: direct remote `git fetch origin
  codex/frame-token-precheck-deploy-20260624` plus `git merge --ff-only
  FETCH_HEAD`.
- Bundle/format-patch fallback was not used because GitHub fetch/push succeeded.
- Remote HEAD before: `7dcb6aec5fccab0b022df7d018c70853b450a7cf`
- Remote HEAD after: `c1d41a86527acd9d80762f012ac12fe4aa8b7b92`
- Remote tracked status before update: clean.

Fixed RUN_TAG gate artifacts:

- RUN_TAG: `frame_token_hybrid_full_train_gate_20260624_204304`
- Precheck/gate run root:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/frame_token_hybrid_full_train_gate_20260624_204304`
- Precheck log:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/frame_token_hybrid_full_train_gate_20260624_204304/frame_token_hybrid_full_train_gate_20260624_204304.log`
- Active manifest:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/frame_token_hybrid_full_train_gate_20260624_204304/active_sha256_manifest.txt`
- active_manifest_sha:
  `44de7c9474eee45b96ea618ec4f33a2903123a9f47b85d289493034fa07730ba`
- resolved_config_sha:
  `f67c3bfa33e6cfbc3b416cc7823ed27b1c68baea03a5f3607aae083aa0c3b2cc`
- Full-train gate JSON:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/frame_token_hybrid_full_train_gate_20260624_204304/frame_token_hybrid_full_train_gate.json`
- full_train_gate_sha:
  `3dcfeddd4eaa84d85754db33c79daba337aaddd847673d8f165701db58afb5a2`

Remote validator results:

```text
FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_PRECHECK_PASS
FRAME_TOKEN_HYBRID_GATE_VALIDATION_PASS
13 passed in 4.36s
FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN
FRAME_TOKEN_HYBRID_GATE_VALIDATION_PASS
FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_VALIDATION_PASS
```

Submitted Slurm full-train candidate:

```text
sbatch --export=ALL,PRECHECK_ONLY=0,ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1,RUN_TAG=frame_token_hybrid_full_train_gate_20260624_204304,FRAME_TOKEN_HYBRID_REPO=/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27,FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON=/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/frame_token_hybrid_full_train_gate_20260624_204304/frame_token_hybrid_full_train_gate.json,FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256=3dcfeddd4eaa84d85754db33c79daba337aaddd847673d8f165701db58afb5a2,FRAME_TOKEN_HYBRID_CONFIG=configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py,FRAME_TOKEN_HYBRID_TRAIN_ID=0 scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch
```

- Slurm job id: `1117996`.
- Slurm job name: `ft_hybrid_full`.
- Queue status at first check: `PENDING`.
- Reason at first check: `None` as printed by `parajobs`.
- Slurm stdout path:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/ft_hybrid_full-1117996.out`
- Log-head check: file not created yet while pending.
- No other Slurm jobs were cancelled or modified.

Deployment decision:

- Remote gate status: `PASS`.
- Full-train submission status: `SUBMITTED_PENDING`.
- Next action: do not poll repeatedly; wait for normal long-duration monitor or
  next material status change.

## Pro-Finding Fix Superseding Status

Timestamp: 2026-06-25T00:57:13+08:00

Status:
`FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`

Current owner/worktree:

- Owned worktree:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_ProFix_Worktree_20260625`
- Owned branch:
  `codex/frame-token-pro-fix-20260625`
- Route label:
  `DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`

Superseding decision:

- The earlier full-train gate/submission notes above are historical evidence
  only.
- Current full-train candidate is locked again until a follow-up Pro review
  accepts this fix.
- No remote sync, Slurm, long training, `tools/test.py`, mAP, runtime/FLOPs,
  deploy, raw-decode-saving, or paper claim is allowed from this local fix.

Fixes implemented for the Frame/Token Pro findings:

- Added deploy-visible preview metadata source:
  `FrameTokenHybridPreviewProbe` writes
  `frame_token_hybrid_preview_signal`,
  `frame_token_hybrid_preview_positions`, and
  `frame_token_hybrid_preview_source` from `LoadFrames` `frame_inds` and
  prefix masks before decode/augmentation. It does not read GT, teacher,
  oracle, detector outputs, result JSON, checkpoints, or raw-prediction caches.
- Wired the preview hook into the normal train/val/test pipelines of the
  Frame/Token configs and added the preview keys to `Collect.meta_keys`.
- Extended the selector preview contract so `preview_probe` records
  `source_meta_key` and `source`.
- Preserved the raw-observation/span-token bridge contract:
  observed raw positions are copied exactly; stable gaps remain span tokens
  with `span_start`, `span_end`, `role`, `visibility`, and
  `compression_confidence`; dense completion masks distinguish observed,
  span-derived, and completed positions.
- Kept raw decode/runtime claims locked:
  `actual_decode_saving_in_current_actionformer_pipeline=False`,
  `raw_decode_saving_claim_allowed=False`,
  `pre_decode_loader_hook_reviewed=False`.
- Kept full training fail-closed in config:
  `ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`, `allow_tools_train=False`,
  `allow_slurm=False`, `allow_gpu=False`, `allow_full_train=False`,
  `metric_claim_allowed=False`, `paper_claim_allowed=False`.
- Narrowed `opentad/models/selectors/__init__.py` in this route worktree so
  the Frame/Token package import no longer imports PC-OT/MRAS/C3 route classes.

Changed files:

- `opentad/models/selectors/frame_token_hybrid_acquisition_route.py`
- `opentad/models/selectors/__init__.py`
- `opentad/datasets/transforms/frame_token_hybrid.py`
- `opentad/datasets/transforms/__init__.py`
- `configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py`
- `configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py`
- `tools/bata/validate_frame_token_hybrid_gate.py`
- `tools/bata/frame_token_hybrid_build_forward_smoke.py`
- `tests/test_frame_token_hybrid_metadata_contract.py`
- `tests/test_frame_token_hybrid_config_gate.py`
- `tests/test_frame_token_hybrid_build_forward_smoke.py`

Local verification:

```text
python -m py_compile opentad/models/selectors/frame_token_hybrid_acquisition_route.py opentad/models/selectors/__init__.py opentad/datasets/transforms/frame_token_hybrid.py opentad/datasets/transforms/__init__.py configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py tools/bata/validate_frame_token_hybrid_gate.py tools/bata/frame_token_hybrid_build_forward_smoke.py tests/test_frame_token_hybrid_metadata_contract.py tests/test_frame_token_hybrid_config_gate.py tests/test_frame_token_hybrid_build_forward_smoke.py tests/test_frame_token_hybrid_actionformer_integration.py tests/test_frame_token_hybrid_acquisition_route.py
PASS

python tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py --json
PASS: emitted fail-closed precheck-only JSON with preview_source_meta_key and all train/Slurm/claim flags false.

C:/Users/skywalker/.conda/envs/torch_1/python.exe -m pytest tests/test_frame_token_hybrid*.py -q
PASS: 31 passed, 1 skipped in 44.07s.

C:/Users/skywalker/.conda/envs/torch_1/python.exe tools/bata/frame_token_hybrid_build_forward_smoke.py
PASS: FRAME_TOKEN_HYBRID_BUILD_FORWARD_SMOKE_PASS.
```

Self-check:

- Strict random-fixed 50% contract: not a random-fixed selector. The current
  contract is up to 384 raw observations plus stable-gap span tokens from a
  768-step dense window, reconstructed to a 768-step ActionFormer-compatible
  axis.
- GT/teacher/oracle/cache risk: test-time selector rejects forbidden metadata
  and the preview hook uses only deploy-visible `frame_inds`/mask geometry.
- Tensor and mask reasoning: selector requires 5D `[B,C,T,H,W]`, prefix
  contiguous `[B,T]` masks, preserves observed positions, zeroes invalid mask
  suffixes, and returns dense `[B,C,768,H,W]` plus `[B,768]` masks.
- Changed surface: input sampling/controller metadata, token-compression span
  schema, pre-backbone dense completion bridge, fail-closed config/gate, tests,
  and local smoke tooling. It does not modify Adapter internals, detector head,
  loss/assignment, or post-processing.
- Attribution: any future metric gain would be attributable to Frame/Token
  preview-driven acquisition and span-conditioned dense completion only after
  follow-up Pro and full review gates pass. It is not attributable to C3.

Allowed next action:

- Follow-up Pro review with this complete fix context.
- Local static/smoke tests only.

Still locked:

- Remote sync, Slurm, long training, `tools/train.py`, `tools/test.py`, mAP,
  runtime/FLOPs, deploy/raw-decode-saving, paper claims, and any C3/combo
  attribution.

## Full-Train Local Guard Fix

Timestamp: 2026-06-25T10:23:20+08:00

Status:
`FIXED_FOR_SYNC_DEPLOY_PENDING_MAIN_PROCESS_ACCEPTANCE`

User override for this narrow stage:

- This is a small startup gate repair after the external full-train JSON gate
  had already passed but `tools/train.py` was blocked by the config local-only
  guard.
- The user explicitly requested no Pro/Gemini/Claude gate for this repair.
- Route label remains
  `DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`.
- This change does not touch BH-SDC, C3, C3-Pro, or GlobalRank-ST files.

Root cause:

- `scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch`
  correctly validated `--action full-train`, but
  `configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py`
  still advertised `ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY` and
  `allow_tools_train=False`.
- `opentad/utils/training_guard.py` therefore rejected `tools/train.py`
  before DDP, dataset, model, or runner construction.

Fix:

- The full-train candidate config now allows only `tools/train.py` and requires
  a strict `entrypoint_gate_context` bound to:
  `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON`,
  `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256`,
  `FRAME_TOKEN_HYBRID_ACTIVE_MANIFEST_SHA256`,
  `FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_SHA256`, and `RUN_TAG`.
- The guard now supports generic payload-to-environment value bindings and
  uses that to reject a reused or mismatched full-train gate `run_tag`.
- A precheck-only JSON, missing JSON/SHA/env, mismatched manifest/resolved config,
  mismatched `RUN_TAG`, direct `tools/test.py`, raw prediction cache, metric
  claim, or paper claim remains fail-closed.
- The full-train launcher now validates that the config is a gate-bound
  full-train config instead of requiring the old precheck-only config shape, and
  exports the full-train gate variables before `tools/train.py`.

Changed files in this repair:

- `configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py`
- `opentad/utils/training_guard.py`
- `scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch`
- `tools/bata/validate_frame_token_hybrid_gate.py`
- `tests/test_frame_token_hybrid_config_gate.py`
- `research-wiki/experiments/FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_DEPLOYMENT_20260624.md`

Verification:

```text
C:/Users/skywalker/.conda/envs/torch_1/python.exe -m pytest tests/test_frame_token_hybrid_config_gate.py -q
PASS: 17 passed in 3.39s.

C:/Users/skywalker/.conda/envs/torch_1/python.exe -m py_compile opentad/utils/training_guard.py tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py tests/test_frame_token_hybrid_config_gate.py
PASS.

C:/Users/skywalker/.conda/envs/torch_1/python.exe tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py --json
PASS: emitted gate-bound full-train config JSON with only tools/train.py open and tools/test/raw-cache/metric/paper claims closed.

bash -n scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch
NOTE: direct check failed on CRLF line endings in the existing scripts.

bash -lc "tr -d '\r' < scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch | bash -n - && tr -d '\r' < scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch | bash -n -"
PASS.

C:/Users/skywalker/.conda/envs/torch_1/python.exe tools/bata/frame_token_hybrid_build_forward_smoke.py
PASS: FRAME_TOKEN_HYBRID_BUILD_FORWARD_SMOKE_PASS.

C:/Users/skywalker/.conda/envs/torch_1/python.exe -m pytest tests/test_frame_token_hybrid_actionformer_integration.py tests/test_frame_token_hybrid_acquisition_route.py tests/test_frame_token_hybrid_build_forward_smoke.py tests/test_frame_token_hybrid_config_gate.py tests/test_frame_token_hybrid_metadata_contract.py -q
PASS: 33 passed, 1 skipped in 42.04s.

C:/Users/skywalker/.conda/envs/torch_1/python.exe -m pytest tests/test_pc_ot_mras_p2_quality_formal_train_candidate.py tests/test_pc_ot_mras_p2_quality_short_smoke_execution_candidate.py tests/test_pc_ot_mras_r17_formal_train_config.py tests/test_pc_ot_mras_r18_aux_formal_train_config.py tests/test_pc_ot_mras_r35_actionformer_head_formal_train_config.py tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py tests/test_pc_ot_mras_local_lowmem_eval_config.py -q
PASS: 74 passed in 6.15s.
```

## Full-Train Launcher Env Isolation Fix

Timestamp: 2026-06-25T10:58:24+08:00

Status:
`FIXED_LOCALLY_PENDING_MAIN_PROCESS_SYNC_REDEPLOY`

Scope:

- Owned worktree:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_ProFix_Worktree_20260625`
- Owned branch:
  `codex/frame-token-pro-fix-20260625`
- Route label:
  `DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`
- External Pro/Gemini/Claude review was not run per explicit user instruction
  for this narrow launcher repair.
- No C3, BH-SDC, shared-worktree, remote sync, Slurm, or `tools/test.py` action
  was performed.

Root cause:

- The full-train launcher generated and exported/used formal full-train gate
  context in the same process that later ran
  `tests/test_frame_token_hybrid_config_gate.py`.
- When the Slurm submission supplied
  `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON`,
  `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256`, and `RUN_TAG`, the pytest
  subprocess inherited those variables.
- The unit case that must prove "missing gate env fails closed" was therefore
  polluted and failed with `Failed: DID NOT RAISE RuntimeError`; training never
  reached iteration.

Fix:

- `scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch` now runs
  its focused config-gate pytest through an isolated `env -u` subprocess.
- The isolated self-test clears:
  `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN`,
  `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON`,
  `FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256`,
  `FRAME_TOKEN_HYBRID_ACTIVE_MANIFEST_SHA256`,
  `FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_SHA256`, and `RUN_TAG`.
- The parent launcher process does not unset those variables, so after
  full-train gate validation it still exports the gate JSON/SHA and launches
  `tools/train.py` with the bound gate context.
- `tests/test_frame_token_hybrid_config_gate.py` now contains explicit launcher
  coverage that the self-test is env-isolated before train while the train
  invocation remains after the gate exports.

Changed files in this repair:

- `scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch`
- `tests/test_frame_token_hybrid_config_gate.py`
- `research-wiki/experiments/FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_DEPLOYMENT_20260624.md`

Verification:

```text
python -m pytest tests/test_frame_token_hybrid_config_gate.py::test_frame_token_hybrid_full_train_launcher_isolates_self_test_env_before_train -q
RED before launcher fix: failed on missing `env -u` isolation.

python -m pytest tests/test_frame_token_hybrid_config_gate.py::test_frame_token_hybrid_full_train_launcher_isolates_self_test_env_before_train -q
PASS: 1 passed in 0.01s.

python -m pytest tests/test_frame_token_hybrid_config_gate.py -q
PASS: 18 passed in 2.67s.

python -m pytest tests/test_frame_token_hybrid_config_gate.py tests/test_frame_token_hybrid_acquisition_route.py tests/test_frame_token_hybrid_actionformer_integration.py tests/test_frame_token_hybrid_metadata_contract.py tests/test_frame_token_hybrid_build_forward_smoke.py -q
PASS: 19 passed, 16 skipped in 27.21s.
NOTE: local Windows torch DLL initialization printed a fatal-exception trace
inside a skipped/smoke path, but pytest exited 0.

python -m py_compile tests/test_frame_token_hybrid_config_gate.py tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py opentad/models/selectors/frame_token_hybrid_acquisition_route.py
PASS.

(Get-Content -Raw scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch) -replace "`r`n", "`n" | bash -n -
PASS.

git diff --check
PASS.
```

Launch decision:

- Local fix status: `PASS`.
- Allowed next action: main process may push/sync/redeploy this branch's commit
  for the Frame/Token Hybrid full-train launcher repair.
- Still locked in this owner turn: no direct push, no remote sync, no Slurm
  submission, no `tools/test.py`, no mAP/runtime/FLOPs/paper claim.

## Full-Train Dataloader Single-Clip Axis Fix

Timestamp: 2026-06-25T11:36:35+08:00

Status:
`FIXED_LOCALLY_PENDING_MAIN_PROCESS_PUSH_SYNC_REDEPLOY`

Scope:

- Owned worktree:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_ProFix_Worktree_20260625`
- Owned branch:
  `codex/frame-token-pro-fix-20260625`
- Route label:
  `DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`
- User constraint honored: no Pro/Gemini/Claude review, no C3/BH-SDC/shared
  main worktree writes, no remote sync, no Slurm launch, and no push.

Trigger:

- Remote full-train gate had passed and training reached `Training Starts` /
  `Epoch 0 started`, then failed before the first loss with:
  `ValueError: inputs must be [B,C,T,H,W], got (2, 1, 3, 768, 160, 160)`.
- This showed that the real OpenTAD dataloader supplies dense video tensors as
  `[B,N,C,T,H,W]` with `N=1` for this THUMOS full-train path, while
  `FrameTokenHybridAcquisitionRoute._forward_impl` only accepted
  `[B,C,T,H,W]`.

Fix:

- `FrameTokenHybridAcquisitionRoute` now canonicalizes selector inputs at the
  route boundary:
  `[B,C,T,H,W]` remains unchanged, `[B,1,C,T,H,W]` is squeezed to
  `[B,C,T,H,W]`, and `N>1` raises a clear fail-closed error because this route
  does not yet have an explicit metadata flattening or view-fusion contract.
- Masks and metas keep batch semantics unchanged: masks remain `[B,T]`, metas
  length must still equal `B`, observed positions are still per original
  sample, dense completion still emits the ActionFormer-compatible 5D tensor,
  and GT segment/label passthrough remains unchanged.

Changed files in this repair:

- `opentad/models/selectors/frame_token_hybrid_acquisition_route.py`
- `tests/test_frame_token_hybrid_acquisition_route.py`
- `research-wiki/experiments/FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_DEPLOYMENT_20260624.md`

Verification:

```text
C:/Users/skywalker/.conda/envs/torch_1/python.exe -m pytest tests/test_frame_token_hybrid_acquisition_route.py -k "single_clip_dataloader_axis or multi_clip_axis"
PASS: 2 passed, 11 deselected in 9.88s.

C:/Users/skywalker/.conda/envs/torch_1/python.exe -m pytest tests/test_frame_token_hybrid_acquisition_route.py tests/test_frame_token_hybrid_metadata_contract.py tests/test_frame_token_hybrid_config_gate.py tests/test_frame_token_hybrid_build_forward_smoke.py tests/test_frame_token_hybrid_actionformer_integration.py
PASS: 36 passed, 1 skipped in 50.57s.
NOTE: the skipped pytest path requires full `mmaction.registry`; the independent build-forward smoke below passed.

C:/Users/skywalker/.conda/envs/torch_1/python.exe -m py_compile opentad/models/selectors/frame_token_hybrid_acquisition_route.py tests/test_frame_token_hybrid_acquisition_route.py tools/bata/frame_token_hybrid_build_forward_smoke.py
PASS.

C:/Users/skywalker/.conda/envs/torch_1/python.exe tools/bata/frame_token_hybrid_build_forward_smoke.py
PASS: FRAME_TOKEN_HYBRID_BUILD_FORWARD_SMOKE_PASS.
```

Local environment note:

- Default `C:/ProgramData/anaconda3/python.exe` still cannot import torch
  because `c10.dll` fails Windows DLL initialization; torch verification used
  `C:/Users/skywalker/.conda/envs/torch_1/python.exe`, which imports torch and
  runs pytest successfully.

Launch decision:

- Local fix status: `PASS`.
- Allowed next action: main process may push/sync/redeploy this branch's new
  commit for the Frame/Token Hybrid full-train dataloader shape repair.
- Still locked in this owner turn: no direct push, no remote sync, no Slurm
  submission, no `tools/test.py`, no mAP/runtime/FLOPs/paper claim.
