# Sparse TAD Task Flow Tracker

## All Required Experiments and Current Status

| Experiment / Config | Changed Surface | Current Status | Review / Gate State | Deployment / Result State | Next Action |
|---|---|---|---|---|---|
| C3 oracle-shell indirect precheck / `configs/adatad/thumos/c3_oracle_shell_indirect_precheck.py` | Input sampling only: coarse score `keep_positions` inside oracle shell; no Adapter/head/loss/postprocess change | Local implementation complete | Focused pytest 13 passed; py_compile, bash -n, and config validator PASS; subagent final review pending | Not deployed; no mAP | Commit/push, export real `C3_COARSE_SCORE_CACHE_DIR`, then run precheck on GPU1 |
| C3 oracle-shell indirect full train / `configs/adatad/thumos/c3_oracle_shell_indirect_full_train.py` | Input sampling only; fixed 384/768 Original AdaTAD backend; validation epoch2 then every5 | Staged but locked | Validator and launcher fail-closed; full train requires explicit main-process unlock env | Not launched; no final Avg-mAP, no high-IoU result | After precheck and cache validation, main process may launch on N16R4 GPU1 |
| Oracle-vs-indirect distribution diagnostic / `tools/diagnose_c3_oracle_shell_indirect_distribution.py` | Diagnostic only; compares indirect positions with GT oracle positions offline from annotation + deploy cache | Local tool implemented | TDD coverage confirms diagnostic does not mutate `frame_inds` and can read annotation/cache | No dataset-wide output yet | Run after coarse cache exists to report overlap/Jaccard, boundary/action/gap/histogram differences |

## Timeline

### 2026-07-01 Asia/Shanghai

- Implemented local C3 oracle-shell indirect candidate in owned worktree `OpenTAD_C3OracleShellIndirect_Worktree_20260701` on branch `codex/c3-oracle-shell-indirect-20260701`.
- Changed files: `opentad/datasets/transforms/end_to_end.py`, `opentad/datasets/transforms/pseudo_boundary.py`, `tools/build_c3_coarse_score_cache_from_records.py`, `tools/diagnose_c3_oracle_shell_indirect_distribution.py`, `tools/validate_c3_oracle_shell_indirect.py`, configs, launchers, tests, and route docs.
- Strict random-fixed 50% contract status: fixed 384 selected from dense 768; dense window and padding/mask/remap/meta semantics follow oracle shell. This is not dynamic-budget evidence yet.
- GT/teacher leakage risk: deploy selection uses coarse score cache only. Manifest rejects `uses_gt=true`; validator rejects teacher/raw cache/P2/PQR/forbidden route tokens. GT is used only for training annotation remap and offline oracle comparison diagnostics.
- Current mAP evidence: none. Local implementation only; no remote deploy, no Slurm, no long training.
- Decision: stage candidate for main-process N16R4 GPU1 deployment after real coarse score cache is provided and precheck passes.

### 2026-07-01 Asia/Shanghai - Exporter/Gate Update

- Added `tools/export_c3_coarse_score_cache_from_selector_checkpoint.py` to export deploy-visible coarse action score cache from a trained C3/CADF selector checkpoint. Cache axis is now explicitly `global_snippet_index`; manifest rejects `uses_gt=true`.
- Extended distribution diagnostic to support `--ann-file` + `--cache-dir` so oracle-vs-indirect comparisons can be generated without video decoding and without feeding GT into selection.
- Verification: focused pytest `13 passed`, py_compile pass, `bash -n` pass for both GPU1 launchers, and `tools/validate_c3_oracle_shell_indirect.py` pass for precheck/full-train configs.
- Current mAP evidence: none. This is implementation/deployment preparation only.
