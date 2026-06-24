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
