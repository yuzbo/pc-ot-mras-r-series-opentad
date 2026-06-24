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
