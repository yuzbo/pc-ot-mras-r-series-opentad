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
