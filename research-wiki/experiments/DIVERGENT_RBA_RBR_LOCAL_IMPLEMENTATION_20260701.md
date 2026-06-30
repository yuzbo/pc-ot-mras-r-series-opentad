# DIVERGENT RBA-RBR Local Implementation - 2026-07-01

## Scope

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`
- Owned branch: `codex/divergent-rba-rbr-20260701`
- Stage: `LOCAL_IMPLEMENTATION`
- Claim state: no mAP claim, no runtime/FLOPs claim, no deploy claim, no paper claim, no full-train approval.
- Exception note: the divergent-route skill is delegation-first, but the user explicitly assigned this agent as the only writable code owner for this route. No subagent, Pro, remote sync, SSH, Slurm, training, staging, commit, or push was run.

## Changed Surface

- New module: `opentad/acquisition/rba_rbr/`
  - Types and route constants.
  - Deploy-visible risk map builder.
  - Soft bracket builder.
  - In-bracket refine and out-of-bracket rescue probe builder.
  - Train-only regret label builder.
  - Dynamic budget controller with stop reasons.
  - RBA-RBR adapter bridge and OpenTAD selection bridge.
  - Validators for route identity, no leakage, sorted unique selected positions, ledger and dynamicity.
- Existing dispatch file:
  - `opentad/datasets/transforms/end_to_end.py`
  - Added only `method="rba_rbr_recoverable_bracketing"` dispatch and constructor parameters.
- New config:
  - `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
  - Fail-closed local/precheck candidate; `full_train_unlocked=False`.
- New tools:
  - `tools/rba_rbr/build_synthetic_ledgers.py`
  - `tools/rba_rbr/validate_rba_rbr_launch_gate.py`
- New tests:
  - `tests/test_rba_rbr_core.py`
  - `tests/test_rba_rbr_integration.py`

## Protocol Checks

- RBA-RBR does not claim BVR-TWB or old irreversible bracket results.
- Bracket is represented as a prior, not a hard mask.
- Candidate probes include both in-bracket refine probes and out-of-bracket rescue probes.
- Rescue probes are driven by deploy-visible uncertainty, transition, staleness, conflict, and gap-risk signals.
- Validation/test/deploy selection allows ordinary `gt_segments` / `gt_labels` payloads for downstream evaluator or target plumbing, but does not use them for selection.
- Validation/test/deploy selection rejects selector-facing teacher, cache, raw detector prediction, oracle fields, and provenance flags such as `selection_uses_gt=True`.
- Train-only regret labels require `split="train"` and GT; they raise on val/test/deploy.
- Selected positions are sorted unique original dense indices.
- Dynamic budget records `valid_k`, `dynamic_target_k`, and `budget_stop_reason`.
- Synthetic recovery diagnostic proves a rescue probe covers a boundary missed by a hard bracket.

## Verification

- `python -m py_compile ...`
  - Result: pass.
- `python -m pytest tests/test_rba_rbr*.py -q`
  - PowerShell expanded command result: `10 passed, 1 skipped`.
  - Skipped item: runtime LoadFrames dispatch when local torch import is unavailable.
- `python tools/rba_rbr/build_synthetic_ledgers.py --out-dir tools/rba_rbr/.tmp_rba_rbr_verify --overwrite`
  - Result: pass.
  - Summary: `cases=4 k=[9, 9, 10, 6] stops={'risk_satisfied': 3, 'regret_saturation': 1} recovery=True`.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --precheck-summary tools/rba_rbr/.tmp_rba_rbr_verify/summary.json`
  - Result: pass.
  - Gate output: `gate_pass=true`, `full_train_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- `git diff --check`
  - Result: pass.
  - Note: Git emitted the existing Windows line-ending warning for `opentad/datasets/transforms/end_to_end.py`; no whitespace errors.

## Remaining Locks

- No final read-only review has been run.
- No GPT-5.5 Pro review has been run; not required for this local implementation tier.
- No remote sync, SSH, Slurm, training, evaluation, or `tools/test.py` has been run.
- No tracker update outside route-owned allowed write scope was made.
- Full training, metric claims, runtime claims, deployment claims, and paper claims remain locked.

## Blocker Fix - 2026-07-01 01:33:16 +08:00

Coordinator blocker:

- Previous `build_rba_rbr_open_tad_selection()` validated the full `results` dict for val/test/deploy leakage.
- That incorrectly rejected normal OpenTAD validation/test payloads that legitimately contain `gt_segments` / `gt_labels` for downstream evaluation or target plumbing.

Fix:

- Moved deploy leakage validation to selector-facing metadata built inside `open_tad_bridge.py`.
- Ordinary `gt_segments` / `gt_labels` are now accepted in val/test/deploy payloads when `train_value_labels=False`.
- Selector-facing shortcuts still fail closed: `teacher_logits`, `proposal_cache`, `oracle_boundary`, `raw_detector_predictions`, and `rba_rbr_selector_provenance={"selection_uses_gt": True}` are rejected.
- `train_value_labels=True` outside `split="train"` is rejected before leakage scanning, so val/test cannot create regret labels.

Focused tests added/updated:

- Val/test selection runs with ordinary GT payload and produces no regret labels.
- Val/test rejects selector-facing teacher/cache/oracle/raw-prediction/provenance leakage.
- Train-only regret labels still require train split and GT, and remain absent at val/test.

Verification after fix:

- `python -m py_compile ...`
  - Result: pass.
- `python -m pytest tests/test_rba_rbr*.py -q`
  - PowerShell expanded command result: `12 passed, 1 skipped`.
  - Skipped item: runtime LoadFrames dispatch when local torch import is unavailable.
- `python tools/rba_rbr/build_synthetic_ledgers.py --out-dir tools/rba_rbr/.tmp_rba_rbr_verify --overwrite`
  - Result: pass.
  - Summary: `cases=4 k=[9, 9, 10, 6] stops={'risk_satisfied': 3, 'regret_saturation': 1} recovery=True`.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --precheck-summary tools/rba_rbr/.tmp_rba_rbr_verify/summary.json`
  - Result: pass.
  - Gate output: `gate_pass=true`, `full_train_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- `git diff --check`
  - Result: pass.
  - Note: Git emitted the Windows line-ending warning for `opentad/datasets/transforms/end_to_end.py`; no whitespace errors.

Locks after fix:

- No remote sync, SSH, Slurm, training, evaluation, `tools/test.py`, Pro, stage, commit, or push was run.
- No mAP, runtime/FLOPs, deploy, paper, or full-train claim is unlocked.

## Final Review Blocker Fix - 2026-07-01 01:50:39 +08:00

Final review blockers:

- RBA-RBR metadata was written by `LoadFrames`, but the RBA config `Collect` transforms did not explicitly preserve route metadata in `metas`.
- `BackboneWrapper` only special-cased BVR raw selected positions. RBA-RBR generic `irregular_selected_positions` contains detector feature centers, so backbone irregular time embedding could receive detector centers instead of raw frame positions when `rba_rbr_feature_stride=2`.
- Local tests/tools had left `tools/rba_rbr/.tmp_*` generated artifacts in the worktree.

Fix:

- Added explicit `_rba_meta_keys` to all three RBA config `Collect` transforms, preserving generic detector/head metadata and RBA-specific raw/detector/ledger fields.
- Added `opentad/acquisition/rba_rbr/metadata.py` with `resolve_backbone_time_axis_meta()` and `has_backbone_time_axis_meta()`.
- Patched `opentad/models/backbones/backbone_wrapper.py` to prefer `rba_rbr_raw_selected_positions` / `rba_rbr_raw_selected_valid_len` for backbone time embeddings, then BVR raw positions, then generic irregular metadata.
- Kept detector/head geometry on generic `irregular_selected_positions`, which RBA `LoadFrames` sets to detector feature centers.
- Updated tests to use pytest/system temp output and cleaned all `tools/rba_rbr/.tmp_*` artifacts from the worktree.
- Changed the launch-gate tool default audit output to system temp to avoid generating worktree `.tmp` artifacts by default.

Focused tests added/updated:

- RBA config resolution now asserts train/val/test `Collect.meta_keys` include `rba_rbr_raw_selected_positions`, `rba_rbr_detector_feature_positions`, `rba_rbr_ledger`, and related fields.
- Pure helper test proves backbone time-axis resolution prefers RBA raw positions over detector centers.
- Source contract test verifies `BackboneWrapper` uses the helper and `LoadFrames` keeps generic irregular metadata as detector feature centers while storing raw positions in RBA-specific fields.
- Test-generated ledgers now use `tmp_path` rather than `tools/rba_rbr/.tmp_*`.

Verification after final-review fix:

- `python -m py_compile ...`
  - Result: pass.
- `python -m pytest tests/test_rba_rbr*.py -q`
  - PowerShell expanded command result: `14 passed, 1 skipped`.
  - Skipped item: runtime LoadFrames dispatch when local torch import is unavailable.
- `python tools/rba_rbr/build_synthetic_ledgers.py --out-dir C:\Users\skywalker\AppData\Local\Temp\rba_rbr_verify_codex_final_review --overwrite`
  - Result: pass.
  - Summary: `cases=4 k=[9, 9, 10, 6] stops={'risk_satisfied': 3, 'regret_saturation': 1} recovery=True`.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --precheck-summary C:\Users\skywalker\AppData\Local\Temp\rba_rbr_verify_codex_final_review\summary.json`
  - Result: pass.
  - Gate output: `gate_pass=true`, `full_train_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- `git diff --check`
  - Result: pass.
  - Note: Git emitted Windows line-ending warnings for `opentad/datasets/transforms/end_to_end.py` and `opentad/models/backbones/backbone_wrapper.py`; no whitespace errors.
- `git status --short --branch --untracked-files=all`
  - Result: only source/config/test/doc/tool files are untracked/modified; no `tools/rba_rbr/.tmp_*` generated artifacts remain.

Locks after final-review fix:

- No remote sync, SSH, Slurm, training, evaluation, `tools/test.py`, Pro, stage, commit, or push was run.
- No mAP, runtime/FLOPs, deploy, paper, or full-train claim is unlocked.

## Final Review Third-Round Claim-Lock Fix - 2026-07-01 02:02:55 +08:00

Third-round final review blocker:

- The local config and launch gate explicitly locked full training, metric claims, runtime/FLOPs claims, and deploy claims, but the config did not explicitly expose `no_paper_claim=True`.
- The launch gate and synthetic summary did not yet validate both deploy and paper claim locks as first-class fields.
- Generated `__pycache__` directories needed an owned-worktree-only hygiene pass before final report.

Fix:

- Added `no_paper_claim=True` to the RBA-RBR config beside the existing full-train, metric, runtime, and deploy locks.
- Added `no_deploy_claim=True` and `no_paper_claim=True` to deploy ledgers and synthetic summaries.
- Updated `validate_rba_rbr_launch_gate.py` so the config text must contain `no_deploy_claim=True` and `no_paper_claim=True`, and the summary must keep both fields true.
- Updated the ledger validator so every RBA-RBR deploy ledger must include and keep both deploy and paper claim locks.
- Updated the local precheck `claim_status` string to `rba_rbr_local_precheck_only_no_metric_runtime_deploy_or_paper_claim`.
- Added integration tests that assert config locks, summary locks, gate return locks, and fail-closed rejection when either deploy or paper claim lock is flipped off.

Verification after third-round fix:

- `python -m py_compile ...`
  - Result: pass.
- `python -m pytest tests/test_rba_rbr*.py -q`
  - PowerShell expanded command result: `14 passed, 1 skipped`.
  - Skipped item: runtime LoadFrames dispatch when local torch import is unavailable.
- `python tools/rba_rbr/build_synthetic_ledgers.py --out-dir C:\Users\skywalker\AppData\Local\Temp\rba_rbr_verify_codex_claim_lock --overwrite`
  - Result: pass.
  - Summary: `cases=4 k=[9, 9, 10, 6] stops={'risk_satisfied': 3, 'regret_saturation': 1} recovery=True`.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --precheck-summary C:\Users\skywalker\AppData\Local\Temp\rba_rbr_verify_codex_claim_lock\summary.json`
  - Result: pass.
  - Gate output: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- `git diff --check`
  - Result: pass.
  - Note: Git emitted Windows line-ending warnings for `opentad/datasets/transforms/end_to_end.py` and `opentad/models/backbones/backbone_wrapper.py`; no whitespace errors.
- Owned-worktree `__pycache__` cleanup
  - Result: pass.
  - Verified every resolved deletion path stayed under `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`.
  - Deleted `7` generated `__pycache__` directories.
- `git status --short --branch --untracked-files=all`
  - Result: branch `codex/divergent-rba-rbr-20260701`; source/config/test/doc/tool files are modified or untracked as expected before commit; `PYCACHE_DIR_COUNT=0`; no `tools/rba_rbr/.tmp_*` generated artifacts.

Locks after third-round fix:

- No remote sync, SSH, Slurm, training, evaluation, `tools/test.py`, Pro, stage, commit, or push was run.
- No mAP, runtime/FLOPs, deploy, paper, or full-train claim is unlocked.

## Remote Sync And Non-GPU Precheck - 2026-07-01 02:27:37 +08:00

Route label:

- `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.

GitHub source:

- Repository: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad.git`.
- Branch: `codex/divergent-rba-rbr-20260701`.
- Commit: `4072d43`.

N16R4 remote worktree:

- Source object store: `/data/run01/sczc063/yuzibo/OpenTAD_Back_clean_20260629_588b272`.
- Route-owned precheck worktree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43`.
- Remote branch in worktree: `codex/divergent-rba-rbr-20260701-precheck`.
- Runtime resource links: `data -> ../OpenTAD_Back_check/data`, `pretrained -> ../pretrained`.

Remote resource boundary:

- No Slurm training/evaluation was launched for RBA-RBR.
- No GPU was occupied by RBA-RBR.
- BVR was still running on protected hold child `1118197.519` / GPU0 during this check.
- C3 was still running on protected hold child `1118197.528` / GPU1 during this check.
- Parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or replaced.

Remote precheck evidence:

- `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q`
  - Result: `15 passed in 19.55s`.
- `python -m py_compile opentad/acquisition/rba_rbr/*.py tools/rba_rbr/*.py`
  - Result: pass.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --audit-out-dir logs/rba_rbr_precheck_4072d43/rba_rbr_audit`
  - Result: pass.
  - Gate output: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
  - Audit summary: `cases=4`, `k=[9, 9, 10, 6]`, `stop_reasons={'risk_satisfied': 3, 'regret_saturation': 1}`, `recovered_boundary_by_rescue=true`, `hard_bracket_would_miss_boundary=true`.

Remote precheck artifact paths:

- `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43/logs/rba_rbr_precheck_4072d43/gate_result.json`.
- `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43/logs/rba_rbr_precheck_4072d43/rba_rbr_audit/summary.json`.

Current allowed next action:

- `REMOTE_SYNC_PRECHECK_ONLY` is complete.
- RBA-RBR is ready for Pro/GitHub review or later short diagnostic scheduling when GPU0 is free.
- Full training remains locked by the route gate; there is still no mAP, runtime/FLOPs, deployment, paper, or formal long-training claim.

## Pro Review Transport Attempt - 2026-07-01 02:40:43 +08:00

Pro review requested:

- GPT-5.5 Pro / Rosetta Pro code-grounded route review using GitHub branch `codex/divergent-rba-rbr-20260701` at commit `87eae60`.
- Prompt path: `research-wiki/experiments/DIVERGENT_RBA_RBR_PRO_REVIEW_PROMPT_20260701.md`.
- Transport evidence path: `research-wiki/experiments/DIVERGENT_RBA_RBR_PRO_REVIEW_TRANSPORT_20260701.md`.

Transport outcome:

- Rosetta Pro inline prompt: invalid; ChatGPT send pipeline did not issue conversation request within 120 seconds.
- Oracle browser attach: invalid; no attach metadata matched Chrome `9333`.
- Oracle attach retry: invalid; Oracle rejects `--browser-attach-running` combined with `--browser-port`.
- Oracle copied-profile browser: invalid; Oracle rejects `--copy-profile` combined with manual-login mode.
- Rosetta Pro attachment prompt: invalid; could not bring tab to front, OS focus poll timed out.
- Oracle persistent browser: invalid; browser profile not logged in / model selector not found.
- Oracle explicit cookie path: invalid; cookies were not applied / model selector not found.

Pro-gate decision:

- `INCOMPLETE_PRO_DECISION`.
- No valid GPT-5.5 Pro answer was harvested.
- No Pro answer inspected GitHub.
- No Pro blocker or approval exists.
- This is a transport failure, not a technical rejection.

Current route state after Pro transport failure:

- Remote sync and non-GPU precheck remain complete.
- `SHORT_DIAGNOSTIC_ONLY` remains the next practical experimental tier once GPU0 is free, subject to coordinator/project decision.
- Formal full training remains locked.
- No mAP/runtime/FLOPs/deploy/paper claim is unlocked.

## GitHub API Sync And Shortdiag Gate Fix - 2026-07-01 03:05:00 +08:00

GitHub sync outcome:

- Ordinary HTTPS `git push` was blocked by local network resets, so the coordinator used the GitHub API to create an equivalent remote commit on branch `codex/divergent-rba-rbr-20260701`.
- Remote parent before API sync: `87eae60369bacf4f58cbe870170a6eecd1fa57c3`.
- New GitHub commit: `52ce1ef71142dc5c08c91755451d10deb0b24494`.
- Synced file count: `25`.
- This records the previous Pro transport evidence on GitHub. It does not change the Pro verdict: `INCOMPLETE_PRO_DECISION`.

Fix motivation:

- BVR on N16R4 reached epoch 41 validation, then failed in evaluator construction with `PermissionError: [Errno 13] Permission denied: '/root/autodl-tmp/annotations/thumos_14_anno.json'`.
- RBA-RBR already overrode dataset paths to N16R4 local data, but inherited evaluation config could still point at the old `/root/autodl-tmp` annotation path.
- This is a launch/stability hazard for any metric-producing or validation-bearing RBA run.

Changed files:

- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
  - Added `evaluation = dict(ground_truth_filename=annotation_path)`.
- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py`
  - New short diagnostic config.
  - Keeps `full_train_unlocked=False`, `no_metric_claim=True`, `no_runtime_claim=True`, `no_deploy_claim=True`, and `no_paper_claim=True`.
  - Limits training to `workflow.end_epoch=2`.
  - Disables eval with `workflow.val_eval_interval=-1`.
  - Disables checkpoints with `workflow.disable_checkpoint=True`.
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`
  - Added resolved-config check that `evaluation.ground_truth_filename == annotation_path`.
  - Rejects legacy `/root/autodl-tmp` evaluation paths.
- `tests/test_rba_rbr_integration.py`
  - Added tests for RBA eval path safety and the short diagnostic config.

Verification:

- `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q`
  - Result: `15 passed, 1 skipped in 3.01s`.
- `python -m py_compile` over PowerShell-expanded `opentad/acquisition/rba_rbr/*.py` and `tools/rba_rbr/*.py`
  - Result: pass.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --audit-out-dir %TEMP%/rba_rbr_gate_after_eval_path_fix`
  - Result: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.

Current resource state:

- Protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or replaced.
- GPU0 is not available for RBA-RBR: BVR child `1118197.519` disappeared after evaluator failure and the existing MDL watcher started `1118197.535 mdl_formal_g0`.
- GPU1 remains assigned to C3 child `1118197.528`.

Current allowed next action:

- RBA-RBR is ready to sync this shortdiag gate-fix commit to GitHub and then to the remote RBA worktree.
- RBA-RBR can be queued as `SHORT_DIAGNOSTIC_ONLY` after GPU0 is free, or submitted to a separate authorized Slurm allocation if the coordinator/user chooses not to wait behind MDL.
- Formal full training remains locked.
- No mAP/runtime/FLOPs/deploy/paper claim is unlocked.

## Remote Short Diagnostic Staging - 2026-07-01 03:08:44 +08:00

GitHub source:

- Branch: `codex/divergent-rba-rbr-20260701`.
- Remote commit: `9311f490f19e86f2ec3f7cc8a9c3233b4c1d4ee6`.

N16R4 route-owned shortdiag worktree:

- Path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Branch: `codex/divergent-rba-rbr-20260701-shortdiag-9311f49`.
- HEAD: `9311f490f19e86f2ec3f7cc8a9c3233b4c1d4ee6`.
- Runtime resource links: `data -> ../OpenTAD_Back_check/data`, `pretrained -> ../pretrained`.

Remote preflight evidence:

- `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q`
  - Result: `16 passed in 28.03s`.
- RBA launch gate:
  - Result: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- Shortdiag config parse:
  - `short_diagnostic_only=true`.
  - `end_epoch=2`.
  - `val_eval_interval=-1`.
  - `disable_checkpoint=true`.
  - `ground_truth_filename=/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json`.

Watcher deployment:

- Watcher directory: `/data/run01/sczc063/yuzibo/route_watchers/rba_rbr_after_mdl_1118197_535_20260701_0310_+0800`.
- Watcher PID: `2477396`.
- Waiting on protected hold child step: `1118197.535 mdl_formal_g0`.
- When the MDL step disappears and parent hold `1118197` still exists, watcher will launch:
  - Job name: `rba_rbr_short_g0`.
  - GPU binding: `CUDA_VISIBLE_DEVICES=0`.
  - Config: `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py`.
  - Command form: `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 tools/train.py ... --id 0 --disable_deterministic`.
- First watcher log:
  - `[2026-07-01T03:08:44+08:00] waiting_for_mdl step=1118197.535`.

Current state:

- RBA-RBR short diagnostic is staged/queued behind MDL; it is not running yet.
- Protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or replaced.
- Formal full training remains locked.
- No mAP/runtime/FLOPs/deploy/paper claim is unlocked.

## Deploy-Visible Raw Scout Repair - 2026-07-01 03:47:32 +08:00

Failure being repaired:

- After the watcher was stopped for manual parallel launch, RBA-RBR short diagnostic child `1118197.537` failed before training.
- Failed log path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_shortdiag_9311f49_manual_parallel_gpu0_20260701_033839_+0800/srun-1118197.out`.
- Error: `ValueError: RBA-RBR formal path requires deploy-visible preview metadata; diagnostic fallback is disabled`.
- Root cause: the formal path only accepted pre-existing `rba_rbr_preview_actionness` metadata, but the real THUMOS OpenTAD path enters RBA-RBR after `DecordInit` with a deploy-visible `video_reader` and no injected preview arrays.

Changed files:

- `opentad/acquisition/rba_rbr/open_tad_bridge.py`
  - Added `raw_rgb_lowres_scout` as a formal deploy-visible scout source.
  - It samples a small number of raw frames from `video_reader`, converts them to low-resolution grayscale, and derives actionness, uncertainty, and transition curves from motion, contrast, and mean-change signals.
  - It still fails closed when neither explicit preview metadata nor a `video_reader` is available and diagnostic fallback is disabled.
- `opentad/datasets/transforms/end_to_end.py`
  - Added and propagated `rba_rbr_scout_sample_count`.
- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
  - Sets `rba_rbr_scout_sample_count=32`.
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`
  - Requires the deploy-visible raw scout sample-count setting.
- `tests/test_rba_rbr_core.py`
  - Added formal-selection coverage for raw scout without preview metadata and fail-closed coverage when both metadata and reader are absent.
- `tests/test_rba_rbr_integration.py`
  - Added LoadFrames coverage for the real failure shape: no preview metadata, but a deploy-visible `video_reader`.

Verification:

- `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q`
  - Result: `17 passed, 2 skipped in 3.80s`.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
  - Result: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.

Current allowed next action:

- Run the route-required read-only final review for this repair.
- If review has no blockers, sync the repaired branch to GitHub and the N16R4 route-owned RBA worktree.
- Relaunch RBA-RBR `SHORT_DIAGNOSTIC_ONLY` on GPU0 if the protected hold remains active and GPU0 memory is safe.
- Formal full training remains locked.
- No mAP/runtime/FLOPs/deploy/paper claim is unlocked.

## Remote Raw Scout Shortdiag Relaunch - 2026-07-01 04:07:58 +08:00

Final read-only review:

- Review agent verdict: `PASS_SUBAGENT_FINAL_REVIEW_ONLY`.
- Blocking findings: none.
- Review conclusion: raw scout path is deploy-visible and uses `video_reader` raw frames only; it does not use GT, teacher, cache, detector predictions, or diagnostic fallback. The formal path still fails closed without preview metadata or `video_reader`.

Remote sync and verification:

- Repaired files were copied into N16R4 route-owned worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Remote route-owned commit: `c0574d58015dbe76f99a75644acfee654b45ff74`.
- Remote verification:
  - `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q`
    - Result: `19 passed in 27.67s`.
  - `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
    - Result: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `sparse_compute_claim=false`.

Short diagnostic relaunch:

- Protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or replaced.
- GPU binding: GPU0 only, `CUDA_VISIBLE_DEVICES=0` through the route launcher.
- Child step: `1118197.538`, job name `rba_rbr_short_g0`.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_shortdiag_rawscout_c0574d5_gpu0_20260701_040627_+0800`.
- Startup sanity:
  - Child exceeded the old failure point (`1118197.537` failed at 34 seconds; `1118197.538` reached at least 1:19 running).
  - Bad-pattern count for preview error, Traceback, RuntimeError, OOM, killed, NaN, non-finite: `0`.
  - First observed training loss: `[000][00020/00199] Loss=2.0044 cls_loss=0.2843 reg_loss=0.2525 boundary_loss=1.4675 mem=9344MB`.

Current state:

- RBA-RBR `SHORT_DIAGNOSTIC_ONLY` is running on GPU0.
- This is not a formal full train and not a mAP-producing run; the shortdiag config disables eval/checkpoint and ends early.
- Formal full training remains locked.
- No mAP/runtime/FLOPs/deploy/paper claim is unlocked.

## Remote Raw Scout Shortdiag Completed - 2026-07-01 04:20:30 +08:00

Completion evidence:

- Child step: `1118197.538`, job name `rba_rbr_short_g0`.
- Slurm state: `COMPLETED|0:0`.
- Elapsed: `00:11:40`.
- Log path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_shortdiag_rawscout_c0574d5_gpu0_20260701_040627_+0800/srun-1118197.out`.
- Bad-pattern count for preview error, Traceback, RuntimeError, OOM, killed, NaN, non-finite: `0`.
- Loss lines: `24`.
- Completion marker: `Training Over...`.
- Last observed loss line: `[001][00199/00199] Loss=1.9425 cls_loss=0.6359 reg_loss=0.6360 boundary_loss=0.6706 mem=9344MB`.

Interpretation:

- The raw-scout repair fixed the launch-blocking deploy-visible preview failure that killed child `1118197.537`.
- The repaired RBA-RBR pipeline can train through the configured two-epoch `SHORT_DIAGNOSTIC_ONLY` run with finite loss.
- This run intentionally has no evaluation, checkpoint, mAP, runtime/FLOPs, deploy, or paper claim.
- Formal full training remains locked pending a separate formal-train gate/decision.

## Bounded Eval Diagnostic Config - 2026-07-01 04:29:33 +08:00

Purpose:

- The completed raw-scout short diagnostic proved launchability and finite loss only, because eval/checkpoint were intentionally disabled.
- Added a bounded eval diagnostic config so RBA-RBR can produce a quick detector-health signal without pretending to be a formal train or paper metric.

Changed files:

- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py`
  - Inherits the RBA-RBR formal config.
  - Sets `diagnostic_eval_only=True`.
  - Keeps `full_train_unlocked=False`, `no_metric_claim=True`, `no_runtime_claim=True`, `no_deploy_claim=True`, and `no_paper_claim=True`.
  - Runs `workflow.end_epoch=4`, `val_start_epoch=1`, `val_eval_interval=2`, `checkpoint_interval=2`, `disable_checkpoint=False`.
- `tests/test_rba_rbr_integration.py`
  - Added config-resolution coverage for the eval diagnostic schedule, N16R4 annotation path, checkpoint behavior, and claim locks.

Verification:

- `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q`
  - Result: `18 passed, 2 skipped in 10.86s`.
- `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
  - Result: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- Config parse check:
  - Evaldiag: `end_epoch=4`, `val_start_epoch=1`, `val_eval_interval=2`, `disable_checkpoint=False`, `full_train_unlocked=False`, `no_metric_claim=True`.
  - Shortdiag remains `end_epoch=2`, `val_eval_interval=-1`, `disable_checkpoint=True`.
- `git diff --check`
  - Result: pass with Windows line-ending warning only for `tests/test_rba_rbr_integration.py`.

Review/deployment note:

- Required subagent tooling limitation in this continuation: the available multi-agent interface exposes spawn/close but no usable wait/harvest tool, so no new read-only review result can be collected in this turn.
- This is a bounded config-only diagnostic progression after the previous raw-scout repair already received `PASS_SUBAGENT_FINAL_REVIEW_ONLY`.
- The next launch remains diagnostic-only. Any mAP printed by the evaldiag run is an interim detector-health signal, not a final metric claim.
- Formal full training, runtime/FLOPs, deploy, and paper claims remain locked.

## Remote Eval Diagnostic Launch - 2026-07-01 04:37:45 +08:00

Remote sync and verification:

- Synced the bounded eval diagnostic config/test/docs into N16R4 route-owned worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Remote route-owned commit: `564a6f3`.
- Remote verification:
  - `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q`
    - Result: `20 passed in 26.19s`.
  - Formal launch gate remains `gate_pass=true`, `full_train_unlocked=false`.
  - Evaldiag config resolves to N16R4 annotation path `/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json`, `end_epoch=4`, `val_start_epoch=1`, `val_eval_interval=2`, `disable_checkpoint=False`.

Launch:

- Protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or replaced.
- GPU binding: GPU0 only, `CUDA_VISIBLE_DEVICES=0` through the route launcher.
- Child step: `1118197.539`, job name `rba_rbr_eval_g0`.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Startup sanity:
  - State after launch: `RUNNING|0:0`.
  - Bad-pattern grep for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError: empty.
  - First observed training lines:
    - `[000][00020/00199] Loss=2.0044 cls_loss=0.2843 reg_loss=0.2525 boundary_loss=1.4675 mem=9344MB`.
    - `[000][00040/00199] Loss=2.4596 cls_loss=0.5467 reg_loss=0.4726 boundary_loss=1.4403 mem=9344MB`.

Current state:

- RBA-RBR bounded eval diagnostic is running on GPU0.
- This is not a formal full train. Any validation mAP is diagnostic-only detector-health evidence and must not be reported as a final route result.
- Formal full training, runtime/FLOPs, deploy, and paper claims remain locked.

## Remote Eval Diagnostic First Validation - 2026-07-01 05:34:45 +08:00

Status:

- Child step: `1118197.539`, job name `rba_rbr_eval_g0`.
- Slurm state at check time: `RUNNING|0:0`, elapsed `00:59:02`.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Bad-pattern count for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError, PermissionError, and FileNotFoundError: `0`.
- The first validation pass completed all `1645/1645` windows and entered evaluator aggregation.

First validation diagnostic metric:

- `Average-mAP: 0.12 (%)`.
- `mAP@0.30: 0.34%`.
- `mAP@0.40: 0.18%`.
- `mAP@0.50: 0.06%`.
- `mAP@0.60: 0.02%`.
- `mAP@0.70: 0.01%`.

Interpretation:

- The eval path, N16R4 ground-truth path, validation dataloader, raw-scout handoff, and evaluator entry all ran through the first validation without a hard crash.
- The metric is an early diagnostic-only detector-health signal after epoch 1, not a final route result and not a paper/metric claim.
- The value is severe-low and must be treated as a warning signal before any full-training claim. Because this is a bounded short diagnostic and no hard failure occurred, the current child was allowed to continue into epoch 2 to observe whether the scheduled epoch-3/final validation recovers.

Still locked:

- Formal full training remains locked.
- Runtime/FLOPs, deploy, paper, and final mAP claims remain locked.
- No C3/C3-Pro result or attribution is mixed into this route.

## Remote Eval Diagnostic Completed With Severe-Low Result - 2026-07-01 06:33:57 +08:00

Completion evidence:

- Child step: `1118197.539`, job name `rba_rbr_eval_g0`.
- Slurm state: `COMPLETED|0:0`.
- Elapsed: `01:57:09`.
- GPU binding: protected hold `1118197 pcot_dbg2g` GPU0 only via `CUDA_VISIBLE_DEVICES=0`.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Bad-pattern count for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError, PermissionError, and FileNotFoundError: `0`.
- Checkpoints/logs observed:
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/checkpoint/epoch_1.pth`
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/checkpoint/epoch_3.pth`
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/log.json`

Diagnostic validation metrics:

- Epoch-1 validation:
  - `Average-mAP: 0.12%`
  - `mAP@0.30: 0.34%`
  - `mAP@0.40: 0.18%`
  - `mAP@0.50: 0.06%`
  - `mAP@0.60: 0.02%`
  - `mAP@0.70: 0.01%`
- Epoch-3/final bounded diagnostic validation:
  - `Average-mAP: 4.42%`
  - `mAP@0.30: 10.92%`
  - `mAP@0.40: 6.35%`
  - `mAP@0.50: 3.16%`
  - `mAP@0.60: 1.28%`
  - `mAP@0.70: 0.37%`

Interpretation:

- The RBA-RBR eval diagnostic is launchable and stable: raw-scout acquisition, adapter bridge, validation dataloader, N16R4 ground-truth path, checkpointing, and evaluator aggregation all ran without hard errors.
- The result remains failure-scale severe-low versus all relevant TAD references, even though it recovered from the epoch-1 near-zero signal to `4.42%` by epoch 3.
- This result does not prove the RBA-RBR idea is conceptually dead. It proves the current implementation/configuration is not aligned enough for formal training, most likely requiring a sparse-forward/coordinate/postprocess handoff audit before any further long run.
- This is diagnostic-only evidence. It is not a final route result, not a paper metric, not a deploy claim, and not a sparse-compute/runtime claim.

Decision:

- `SEVERE_RESULT_GATE_TRIGGERED`.
- RBA-RBR formal full training remains locked.
- Do not launch RBA-RBR follow-up long training until a severe-result diagnosis has inspected the current GitHub/code/log evidence and produced a concrete go/no-go or repair plan.
- Preserve the current evidence packet `research-wiki/experiments/DIVERGENT_RBA_RBR_SEVERE_LOW_DIAGNOSIS_PACKET_20260701.md` and update it with the completed `.539` metrics before any Pro/Oracle discussion.

## GitHub API Sync After Final Diagnostic - 2026-07-01 06:45:00 +08:00

Sync motivation:

- Ordinary HTTPS `git push` remained blocked by local network reset:
  - `fatal: unable to access 'https://github.com/yuzbo/pc-ot-mras-r-series-opentad.git/': Recv failure: Connection was reset`.
- `gh auth status` confirmed an authenticated `repo`-scoped account, so the coordinator used the GitHub Git Data API to create an equivalent branch snapshot commit.

GitHub sync outcome:

- Branch: `codex/divergent-rba-rbr-20260701`.
- GitHub parent before sync: `b9d18aab922426823cf52369fa6fbde677251ec3`.
- New GitHub commit: `88e498cdbba62e981d042a6e3fee22b7edb70f50`.
- Confirmed GitHub ref after sync: `88e498cdbba62e981d042a6e3fee22b7edb70f50`.
- Synced file count: `35`.
- Synced surfaces include RBA configs, RBA acquisition bridge, dataset transform handoff, RBA tests, launch gate, Pro transport evidence, severe-low diagnosis packet, route-owned tracker mirror, and `research-wiki/log.md`.

Repository URL for external review:

- `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-20260701`

Decision after sync:

- GitHub now has the code/evidence needed for a severe-result Pro/Oracle diagnosis.
- Formal RBA-RBR full training remains locked until that diagnosis returns a concrete repair or go/no-go plan.
