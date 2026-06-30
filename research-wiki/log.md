# Research Log

## 2026-07-01 01:22:40 +08:00 - RBA-RBR local implementation

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Worktree/branch: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`, `codex/divergent-rba-rbr-20260701`.
- Implemented first complete local RBA-RBR acquisition route: risk map, soft brackets, refine/rescue probes, train-only regret labels, dynamic budget controller, validators, OpenTAD LoadFrames dispatch, fail-closed config, synthetic ledger builder, launch gate validator, and focused tests.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `10 passed, 1 skipped`; synthetic ledger builder -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`; `git diff --check` -> pass with Windows line-ending warning only.
- No remote sync, SSH, Slurm, training, evaluation, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.
- Next allowed action: final read-only review, then local/precheck-only decision; full training remains locked.

## 2026-07-01 01:33:16 +08:00 - RBA-RBR selector-facing leakage blocker fix

- Fixed coordinator blocker in `build_rba_rbr_open_tad_selection()`: val/test/deploy no longer validate the full OpenTAD `results` dict, so ordinary downstream `gt_segments` / `gt_labels` payloads are accepted when `train_value_labels=False`.
- Leakage validation now applies to selector-facing metadata/provenance only, while still rejecting teacher/cache/oracle/raw-prediction fields and `selection_uses_gt=True` provenance.
- Added focused tests proving val/test GT payload acceptance without regret labels, selector-facing leakage rejection, and train-only regret-label behavior.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `12 passed, 1 skipped`; synthetic ledger builder -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`; `git diff --check` -> pass with Windows line-ending warning only.
- No remote sync, SSH, Slurm, training, evaluation, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.

## 2026-07-01 01:50:39 +08:00 - RBA-RBR final-review metadata/backbone blocker fix

- Fixed final-review blocker: RBA config train/val/test `Collect` transforms now explicitly preserve RBA raw/detector/ledger metadata in `metas`.
- Added RBA metadata helper and patched `BackboneWrapper` so backbone irregular time embedding prefers `rba_rbr_raw_selected_positions` / `rba_rbr_raw_selected_valid_len`, while detector/head generic `irregular_selected_positions` remains detector feature centers.
- Updated tests to prove config metadata propagation, raw-axis preference for backbone, and generic detector-center metadata preservation.
- Cleaned `tools/rba_rbr/.tmp_*` generated artifacts and changed test/verify output to pytest/system temp paths.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `14 passed, 1 skipped`; synthetic ledger builder in system temp -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`; `git diff --check` -> pass with Windows line-ending warnings only; `git status --untracked-files=all` -> no `tools/rba_rbr/.tmp_*` artifacts.
- No remote sync, SSH, Slurm, training, evaluation, `tools/test.py`, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.

## 2026-07-01 02:02:55 +08:00 - RBA-RBR third-round claim-lock/hygiene blocker fix

- Fixed third-round final-review blocker: config now explicitly sets `no_paper_claim=True`, and launch gate requires both `no_deploy_claim=True` and `no_paper_claim=True`.
- Added deploy/paper claim locks to RBA-RBR deploy ledgers, synthetic summary, ledger validator, launch-gate output, and integration tests; `claim_status` is now `rba_rbr_local_precheck_only_no_metric_runtime_deploy_or_paper_claim`.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `14 passed, 1 skipped`; synthetic ledger builder in system temp -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`; `git diff --check` -> pass with Windows line-ending warnings only; owned-worktree `__pycache__` cleanup deleted 7 generated dirs and final count is `PYCACHE_DIR_COUNT=0`; `git status --untracked-files=all` -> no `.tmp_*` or `__pycache__` artifacts.
- No remote sync, SSH, Slurm, training, evaluation, `tools/test.py`, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.

## 2026-07-01 02:27:37 +08:00 - RBA-RBR remote sync and non-GPU precheck

- Synced `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3` from GitHub branch `codex/divergent-rba-rbr-20260701` at commit `4072d43` into N16R4 route-owned precheck worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43`.
- Used existing remote clean clone `/data/run01/sczc063/yuzibo/OpenTAD_Back_clean_20260629_588b272` as the Git object source to avoid a full new clone after the first clone attempt hit `Disk quota exceeded`.
- Remote precheck passed without GPU or Slurm: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `15 passed in 19.55s`; py_compile -> pass; RBA-RBR launch gate -> `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- Remote audit summary path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43/logs/rba_rbr_precheck_4072d43/rba_rbr_audit/summary.json`; gate result path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43/logs/rba_rbr_precheck_4072d43/gate_result.json`.
- Resource boundary: BVR child `1118197.519` was still running on GPU0 and C3 child `1118197.528` was still running on GPU1; RBA-RBR did not occupy either GPU and did not modify/release/cancel protected parent hold `1118197 pcot_dbg2g`.
- Current next action: RBA-RBR may proceed to Pro/GitHub review or later short diagnostic scheduling when GPU0 is free; full training and all mAP/runtime/deploy/paper claims remain locked.

## 2026-07-01 02:40:43 +08:00 - RBA-RBR Pro review transport incomplete

- Created Pro review prompt `research-wiki/experiments/DIVERGENT_RBA_RBR_PRO_REVIEW_PROMPT_20260701.md` for GitHub branch `codex/divergent-rba-rbr-20260701` at commit `87eae60`.
- Rosetta/Oracle transport attempts are recorded in `research-wiki/experiments/DIVERGENT_RBA_RBR_PRO_REVIEW_TRANSPORT_20260701.md`.
- No valid GPT-5.5 Pro answer was harvested: Rosetta inline failed with stuck send pipeline; Rosetta attachment failed with focus timeout; Oracle attach/copy-profile/persistent/cookie attempts all failed before submission or model selection.
- Pro state is `INCOMPLETE_PRO_DECISION`: no GitHub inspection, no code-grounded Pro blocker, and no Pro approval exists.
- This is transport failure only, not a technical rejection of RBA-RBR. Remote sync and non-GPU precheck remain valid; full training and all mAP/runtime/deploy/paper claims remain locked.

## 2026-07-01 03:05:00 +08:00 - RBA-RBR GitHub API sync and shortdiag gate fix

- Advanced GitHub branch `codex/divergent-rba-rbr-20260701` by API because ordinary HTTPS `git push` kept failing locally; remote branch moved from `87eae60369bacf4f58cbe870170a6eecd1fa57c3` to `52ce1ef71142dc5c08c91755451d10deb0b24494` with 25 synced Pro transport evidence files.
- Fixed RBA-RBR inherited evaluation path risk after BVR failed at epoch-41 evaluation on legacy `/root/autodl-tmp/annotations/thumos_14_anno.json`: RBA config now explicitly sets `evaluation.ground_truth_filename=annotation_path`, and launch gate rejects legacy evaluation paths.
- Added `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py` as a two-epoch, no-eval, no-checkpoint, claim-locked short diagnostic config.
- Verification passed: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `15 passed, 1 skipped`; PowerShell-expanded py_compile -> pass; RBA launch gate -> `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`.
- Resource state: protected hold `1118197` unchanged; BVR child disappeared after evaluator failure and MDL child `1118197.535 mdl_formal_g0` started on GPU0; C3 child `1118197.528` remains on GPU1. RBA-RBR short diagnostic is staged but not running.
