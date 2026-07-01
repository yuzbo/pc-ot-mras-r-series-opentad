# Sparse TAD Task Flow Tracker

## All Required Experiments and Current Status

| Experiment / Config | Changed Surface | Current Status | Review / Gate State | Deployment / Result State | Next Action |
|---|---|---|---|---|---|
| C3 oracle-shell indirect precheck / `configs/adatad/thumos/c3_oracle_shell_indirect_precheck.py` | Input sampling only: coarse score `keep_positions` inside oracle shell; no Adapter/head/loss/postprocess change | Remote GPU1 precheck PASS | Focused pytest 14 passed; py_compile, bash -n, config validator PASS; subagent blockers fixed locally; remote static gates PASS at `d9a7481` | N16R4 precheck run `logs/c3_oracle_shell_indirect_precheck_gpu1_d9a7481_20260701_135939_+0800`; first finite loss `0.8188`; no mAP by design | Complete; monitor full train |
| C3 oracle-shell indirect full train / `configs/adatad/thumos/c3_oracle_shell_indirect_full_train.py` | Input sampling only; fixed 384/768 Original AdaTAD backend; validation epoch2 then every5 | Full training started on GPU1 | Validator and launcher fail-closed; full train explicitly unlocked by main process after cache + diagnostic + precheck evidence | N16R4 child `1118197.571 c3_oracle_full_g1`, run dir `logs/c3_oracle_shell_indirect_full_train_gpu1_d9a7481_20260701_140057_+0800`; first finite loss `1.2794`; no final Avg-mAP yet | Low-frequency monitor for crash/OOM/NaN, first scheduled validation, Training Over, final mAP |
| Oracle-vs-indirect distribution diagnostic / `tools/diagnose_c3_oracle_shell_indirect_distribution.py` | Diagnostic only; compares indirect positions with GT oracle positions offline from annotation + deploy cache | Completed for train and validation splits | TDD coverage confirms diagnostic does not mutate `frame_inds` and can read annotation/cache; manifest `uses_gt=false`, axis `global_snippet_index` | Output dir `logs/c3_oracle_shell_oracle_vs_indirect_diag_epoch41_d9a7481_20260701`; train Jaccard `0.3922`, validation Jaccard `0.3824`; no official mAP claim | Use distribution gap to interpret final full-train result when available |

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

### 2026-07-01 Asia/Shanghai - Final Review Blocker Fixes

- Final read-only subagent review found two blockers: test split used `test_mode=False`, and the score-cache exporter could inherit `ThumosPaddingDataset` for train export.
- Fixed test protocol by setting `dataset.test.test_mode=True` and adding validator/test coverage.
- Fixed exporter by forcing `type=\"ThumosSlidingDataset\"` for all deploy-visible score-cache export splits.
- Verification after fixes: focused pytest `14 passed`, py_compile pass, `bash -n` pass, and precheck/full-train validators pass.

### 2026-07-01 Asia/Shanghai - Remote Cache, Diagnostics, Precheck, Full Train

- Remote clean clone: `/data/run01/sczc063/yuzibo/OpenTAD_C3OracleShellIndirect_69aa934_20260701`, HEAD `d9a7481d0d5ba058cc48cb970fe8cd5e0c3b4997`, GitHub branch `codex/c3-oracle-shell-indirect-20260701`.
- Runtime resources: repository-local `data/thumos-14/raw_data/video` contains 411 symlinks to the authorized TAD-only THUMOS videos; `pretrained` points to `/data/run01/sczc063/yuzibo/pretrained`.
- Remote static gates passed with `/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python`: py_compile, precheck/full-train validator, and focused pytest `14 passed`.
- Coarse score cache exported from CADF loss-select V2 checkpoint `epoch_41.pth` into `logs/c3_oracle_shell_score_cache_epoch41_d9a7481_20260701_r2/cache`; manifest reports `411` videos, `uses_gt=false`, `axis=global_snippet_index`, route labels `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`, and child `1118197.564 c3_oracle_cache_g1` completed with exit `0`.
- Oracle-vs-indirect diagnostic completed in `logs/c3_oracle_shell_oracle_vs_indirect_diag_epoch41_d9a7481_20260701`: train summary `mean_jaccard=0.3922`, indirect action coverage `0.5182`, oracle action coverage `0.8891`, indirect boundary-near selected rate `0.0789`, oracle boundary-near selected rate `0.1438`; validation summary `mean_jaccard=0.3824`, indirect action coverage `0.5173`, oracle action coverage `0.8831`, indirect boundary-near selected rate `0.0808`, oracle boundary-near selected rate `0.1465`.
- GPU1 precheck `logs/c3_oracle_shell_indirect_precheck_gpu1_d9a7481_20260701_135939_+0800` entered `tools/train.py`, produced finite loss `0.8188`, and stopped after the configured two iterations with exit `0`.
- Full train launched on protected parent hold `1118197 pcot_dbg2g` as child `1118197.571 c3_oracle_full_g1`, run dir `logs/c3_oracle_shell_indirect_full_train_gpu1_d9a7481_20260701_140057_+0800`, with `CUDA_VISIBLE_DEVICES=1`. First train loss was finite: `Loss=1.2794`, `cls_loss=0.6824`, `reg_loss=0.5971`, `mem=2144MB`.
- Current result status: full training is running; no final Avg-mAP, no high-IoU conclusion, and no paper/final route claim yet. The run is a diagnostic full train comparing a deploy-visible coarse-classifier indirect selector against the manual oracle-shell processing contract.
