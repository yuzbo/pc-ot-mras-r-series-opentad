# Frame/Token Hybrid Deployment Precheck Report

Route label:
`DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`

Timestamp: 2026-06-24T20:01:33+08:00

## Boundary

This report is scoped only to the divergent Frame/Token Hybrid route. It must
not be merged with or attributed to C3, Event, BH-SDC, or Boundary routes.

Frame/Token Hybrid keeps boundary/anchor positions as raw frame observations,
represents stable gaps as span tokens with span start/end, role, visibility, and
compression confidence, and uses a span-conditioned bridge to form the dense
representation. The current pipeline is still post-decode and pre-backbone, so
raw decode saving, runtime, metric, deploy, and paper claims remain locked.

## Deployment-Precheck Plan

- Owned worktree:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_PrecheckDeploy_Worktree_20260624`
- Owned branch: `codex/frame-token-precheck-deploy-20260624`
- Base commit: `17d958ee34716dc34fc11ba726747d3a0e8e33ec`
- New launcher:
  `scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch`
- Local config:
  `configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py`
- Full-train candidate config:
  `configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py`
- Validator: `tools/bata/validate_frame_token_hybrid_gate.py`

## Initial Status

- PRECHECK_ONLY launcher: prepared.
- Selector core: unchanged.
- Gate stance: fail-closed, `ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY` only.
- Full train: locked.
- Pro/Claude/Gemini review: intentionally not run for this precheck stage per
  route owner instruction.

## Local Verification

- `python -m py_compile tools/bata/validate_frame_token_hybrid_gate.py
  configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py
  configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py
  tests/test_frame_token_hybrid_config_gate.py
  opentad/models/selectors/frame_token_hybrid_acquisition_route.py`: PASS.
- `python tools/bata/validate_frame_token_hybrid_gate.py
  configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py
  --json`: PASS, fail-closed JSON preserved.
- `python tools/bata/validate_frame_token_hybrid_gate.py
  configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py
  --json`: PASS, fail-closed JSON preserved.
- `python -m pytest tests/test_frame_token_hybrid_config_gate.py -q`: PASS,
  9 passed.
- `python -m pytest tests/test_frame_token_hybrid_acquisition_route.py
  tests/test_frame_token_hybrid_metadata_contract.py
  tests/test_frame_token_hybrid_actionformer_integration.py
  tests/test_frame_token_hybrid_config_gate.py -q`: PASS, 9 passed and
  15 skipped by local dependency/runtime guards.
- Local Windows `bash -n` was not accepted because the available `bash.exe` is
  the Windows system/WSL shim and did not resolve the worktree-relative script
  path. Shell syntax will be checked on the Linux remote before PRECHECK_ONLY.
- First N16R4 dry invocation found an environment-source ordering issue:
  `set -u` was active before `/etc/profile` and caused the launcher to exit
  before logging. The launcher was fixed to match existing N16R4 scripts:
  `set -eo pipefail`, source profile, then `set -u`.

## Evidence To Fill After Execution

- Commit and push:
  - `ecb7c27a303c06cdb7c216e5be7e6ce713751ca3`
    (`Add frame token hybrid N16R4 precheck launcher`) pushed to
    `origin/codex/frame-token-precheck-deploy-20260624`.
  - `3df6597` (`Fix frame token precheck profile sourcing`) pushed to the same
    branch after N16R4 showed `/etc/profile` must be sourced before `set -u`.
- Remote sync method:
  - GitHub branch clone/fetch only; no zip/scp fallback used.
- Remote path:
  - Initial route-owned clone succeeded at
    `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27`
    with HEAD `ecb7c27a303c06cdb7c216e5be7e6ce713751ca3`.
  - Final intended route-owned clone path
    `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_3df6597`
    did not complete because GitHub TLS fetch/clone failed.
- Remote PRECHECK_ONLY command/log:
  - Not executed on final commit `3df6597`.
  - `bash -n` on the initial clone's launcher passed, but direct execution of
    the initial launcher failed before logging because of the fixed
    `/etc/profile`/`set -u` ordering issue.
- Final deployment-precheck decision at 2026-06-24T20:12:58+08:00:
  `BLOCKED_REMOTE_GITHUB_TLS_UNREACHABLE_AFTER_PUSH`; full train remains
  locked.

## Git-Native Recovery Update

Timestamp: 2026-06-24T20:20:18+08:00

Recovery owner action:

- GitHub HTTPS/TLS remained bypassed. No zip or tar artifact was used.
- Local authoritative worktree:
  `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_PrecheckDeploy_Worktree_20260624`
- Local branch: `codex/frame-token-precheck-deploy-20260624`
- Local HEAD:
  `7dcb6aec5fccab0b022df7d018c70853b450a7cf`
- Git-native artifact:
  `logs/frame_token_hybrid_precheck_recovery_20260624_201720.bundle`
- Local bundle SHA256:
  `583c58fa854a096074a468c4bc4ddcd781dd9072d7f8fef54f989e90699c902d`
- Remote bundle path:
  `/data/home/sczc063/run/yuzibo/frame_token_hybrid_recovery/frame_token_hybrid_precheck_recovery_20260624_201720.bundle`
- Remote bundle SHA256:
  `583c58fa854a096074a468c4bc4ddcd781dd9072d7f8fef54f989e90699c902d`

Remote repo recovery:

- Route-owned remote repo:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27`
- Before recovery HEAD:
  `ecb7c27a303c06cdb7c216e5be7e6ce713751ca3`
- Recovery command shape:
  `git fetch <bundle> refs/heads/codex/frame-token-precheck-deploy-20260624`
  followed by `git merge --ff-only FETCH_HEAD`
- Merge result: fast-forward `ecb7c27..7dcb6ae`.
- Remote HEAD after recovery:
  `7dcb6aec5fccab0b022df7d018c70853b450a7cf`

PRECHECK_ONLY execution:

- Slurm command submitted:
  `sbatch --export=ALL,ALLOW_LOGIN_NODE_DEBUG=1,PRECHECK_ONLY=1,RUN_TAG=frame_token_hybrid_precheck_recovery_20260624_201720,FRAME_TOKEN_HYBRID_REPO=/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27 scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch`
- Slurm job id: `1117984`.
- Slurm state at evidence collection: `PD (Priority)`.
- Slurm log path: `logs/ft_hybrid_pre-1117984.out`; no Slurm log existed yet
  while the job was still pending.
- To avoid queue delay while preserving the no-train contract, the same
  launcher script was also run once as a login-node PRECHECK_ONLY debug command:
  `ALLOW_LOGIN_NODE_DEBUG=1 PRECHECK_ONLY=1 RUN_TAG=frame_token_hybrid_precheck_recovery_login_20260624_201720 FRAME_TOKEN_HYBRID_REPO=/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27 bash scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch`
- Remote login PRECHECK log:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/frame_token_hybrid_precheck_recovery_login_20260624_201720/frame_token_hybrid_precheck_recovery_login_20260624_201720.log`
- Local copied log:
  `logs/frame_token_hybrid_precheck_recovery_login_20260624_201720.remote.log`
- Remote summary JSON:
  `/data/home/sczc063/run/yuzibo/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27/logs/frame_token_hybrid_precheck_recovery_login_20260624_201720/frame_token_hybrid_precheck_summary.json`
- Local copied summary:
  `logs/frame_token_hybrid_precheck_recovery_login_20260624_201720.summary.json`

PASS evidence:

```text
git_branch=codex/frame-token-precheck-deploy-20260624 git_head=7dcb6aec5fccab0b022df7d018c70853b450a7cf
FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_PRECHECK_PASS
FRAME_TOKEN_HYBRID_GATE_VALIDATION_PASS
9 passed in 4.20s
FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN
```

Summary decision:

- Status: `PASS`.
- Decision: `FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN`.
- Full train remains locked: `true`.
- `tools_train=false`, `tools_test=false`.
- `metric_claim_allowed=false`, `paper_claim_allowed=false`.

## 2026-06-25 Pro-Finding Fix Superseding Note

Status:
`FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`

The current Frame/Token owner fix is in
`E:/DeskTop/TAD/temrefuse-tad/OpenTAD_FrameToken_ProFix_Worktree_20260625` on
branch `codex/frame-token-pro-fix-20260625`. It adds a normal-pipeline
`FrameTokenHybridPreviewProbe` metadata source, preserves the
raw-observation/span-token/dense-completion bridge, narrows route imports away
from PC-OT/MRAS/C3, and keeps decode/runtime/mAP/deploy/paper/full-train claims
locked until follow-up Pro review.

Canonical current self-check and verification evidence are recorded in
`research-wiki/experiments/FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_DEPLOYMENT_20260624.md`
under `Pro-Finding Fix Superseding Status`.
