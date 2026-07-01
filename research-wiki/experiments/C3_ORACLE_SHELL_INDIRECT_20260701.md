# C3 Oracle-Shell Indirect 2026-07-01

## Status

- Timestamp: 2026-07-01 Asia/Shanghai
- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`
- Worktree: `OpenTAD_C3OracleShellIndirect_Worktree_20260701`
- Branch: `codex/c3-oracle-shell-indirect-20260701`
- Base commit: `c9b8cd6cdc26307048356258d4a9f2dc92b72b1f`
- Current evidence: implementation and gates complete; remote score cache export, oracle-vs-indirect distribution diagnostic, GPU1 precheck, and GPU1 full-train launch completed. Full train is running; no final mAP yet.

## Purpose

This route learns from the manual oracle frame-selection experiment, but replaces only the manual oracle `keep_positions` decision with deployable coarse per-frame action scores. The dense window construction, selected frame indices, mask padding, GT remap for training/evaluation annotations, and irregular-axis metadata follow the oracle shell contract.

Deploy selection must use only the coarse score cache. Validation/test GT is allowed only in offline oracle-vs-indirect diagnostic reports and must not feed `frame_inds`.

## Changed Surface

- Input sampling: changed. New `coarse_score_oracle_shell_subsample` LoadFrames method.
- Dynamic budget policy: unchanged; fixed 384 from dense 768.
- Token compression: unchanged.
- Adapter/backbone internals: unchanged.
- Detector head logic: unchanged.
- Loss/assignment: unchanged.
- Test-time post-processing: unchanged.

## Cache Contract

The coarse cache requires `manifest.json` plus one `.npz` per video. Manifest must include:

- `uses_gt=false`
- `axis="global_snippet_index"`
- C3 route labels
- `videos[video_name].file`

Per-video `.npz` supports `action_score` or `action_logit`. If only logits are present, probabilities are derived by sigmoid. Entropy/uncertainty and absolute action-score change are derived locally. The cache exporter `tools/export_c3_coarse_score_cache_from_selector_checkpoint.py` scans dense sliding windows with a trained coarse selector checkpoint, averages overlapping deploy-visible action logits/scores by global snippet index, and writes `uses_gt=false`.

## Local Verification

RED:

```text
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests\test_c3_oracle_shell_indirect.py -q
ERROR ModuleNotFoundError: No module named 'opentad.datasets.transforms.pseudo_boundary'
```

Launcher RED:

```text
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests\test_c3_oracle_shell_indirect.py::test_gpu1_launchers_fail_closed_and_preserve_parent_hold -q
FAILED FileNotFoundError: scripts/run_c3_oracle_shell_indirect_precheck_gpu1.sh
```

GREEN:

```text
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests\test_c3_oracle_shell_indirect.py -q
14 passed
```

Additional GREEN:

```text
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile opentad\datasets\transforms\end_to_end.py opentad\datasets\transforms\pseudo_boundary.py opentad\models\utils\post_processing\utils.py tools\build_c3_coarse_score_cache_from_records.py tools\diagnose_c3_oracle_shell_indirect_distribution.py tools\export_c3_coarse_score_cache_from_selector_checkpoint.py tools\validate_c3_oracle_shell_indirect.py
pass

bash -n scripts/run_c3_oracle_shell_indirect_precheck_gpu1.sh scripts/run_c3_oracle_shell_indirect_full_train_gpu1.sh
pass

python tools\validate_c3_oracle_shell_indirect.py configs\adatad\thumos\c3_oracle_shell_indirect_precheck.py
PASS_C3_ORACLE_SHELL_INDIRECT_CONFIG

python tools\validate_c3_oracle_shell_indirect.py configs\adatad\thumos\c3_oracle_shell_indirect_full_train.py
PASS_C3_ORACLE_SHELL_INDIRECT_CONFIG
```

## Launch State

- Precheck config: `configs/adatad/thumos/c3_oracle_shell_indirect_precheck.py`
- Full train config: `configs/adatad/thumos/c3_oracle_shell_indirect_full_train.py`
- Full train validation cadence: epoch 2 first validation, then every 5 epochs anchored to epoch 2.
- GPU1 scripts: `scripts/run_c3_oracle_shell_indirect_precheck_gpu1.sh`, `scripts/run_c3_oracle_shell_indirect_full_train_gpu1.sh`
- Parent hold: `1118197` is documented as protected. Scripts do not cancel Slurm jobs.
- Current allowed action: monitor the N16R4 GPU1 full train for crash/OOM/NaN, first scheduled validation, Training Over, and final mAP. Full mAP is pending complete training.

## Final Review Fixes

Subagent final review initially returned blockers:

- `dataset.test.test_mode=False` could let validation/test GT enter the pipeline before `Collect`.
- The score-cache exporter inherited source split dataset type and could build `ThumosPaddingDataset` for train, which does not accept sliding-window export arguments.

Fixes applied:

- `dataset.test.test_mode=True` is now required by config and validator.
- Exporter forces `ThumosSlidingDataset` for all export splits, so default `train test` cache export is deploy-visible sliding-window scoring.
- Focused tests now cover both failures.

## Remote Deployment Evidence

- GitHub branch: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/c3-oracle-shell-indirect-20260701`
- Remote clean clone: `/data/run01/sczc063/yuzibo/OpenTAD_C3OracleShellIndirect_69aa934_20260701`
- Remote HEAD: `d9a7481d0d5ba058cc48cb970fe8cd5e0c3b4997`
- Parent hold: `1118197 pcot_dbg2g` on `g0030`, preserved. This route used child `srun` steps only.
- GPU policy: C3 mainline work launched with `CUDA_VISIBLE_DEVICES=1`; GPU0 remains reserved for divergent innovation work.

Remote static gates passed with `/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python`:

```text
py_compile changed Python files: pass
tools/validate_c3_oracle_shell_indirect.py precheck config: PASS
tools/validate_c3_oracle_shell_indirect.py full-train config: PASS
pytest tests/test_c3_oracle_shell_indirect.py -q: 14 passed
```

## Coarse Score Cache

- Export run dir: `/data/run01/sczc063/yuzibo/OpenTAD_C3OracleShellIndirect_69aa934_20260701/logs/c3_oracle_shell_score_cache_epoch41_d9a7481_20260701_r2`
- Cache dir: `/data/run01/sczc063/yuzibo/OpenTAD_C3OracleShellIndirect_69aa934_20260701/logs/c3_oracle_shell_score_cache_epoch41_d9a7481_20260701_r2/cache`
- Source checkpoint: `/data/run01/sczc063/yuzibo/OpenTAD_C3CADFLossSelectV2_cf3bc13_20260630/exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_formal/gpu1_id225813/checkpoint/epoch_41.pth`
- Slurm child: `1118197.564 c3_oracle_cache_g1`
- Result: exit `0`, `411` `.npz` files plus `manifest.json`.
- Manifest: `uses_gt=false`, `axis=global_snippet_index`, route labels `C3_MAINLINE_OPTIMIZATION` and `C3_ORIGINAL_OPTIMIZATION_ROUTE`.

## Oracle-Vs-Indirect Diagnostic

- Output dir: `/data/run01/sczc063/yuzibo/OpenTAD_C3OracleShellIndirect_69aa934_20260701/logs/c3_oracle_shell_oracle_vs_indirect_diag_epoch41_d9a7481_20260701`
- This is diagnostic-only and is not an official mAP claim.

```text
train windows: 470
train mean_jaccard: 0.3922
train indirect action coverage: 0.5182
train oracle action coverage: 0.8891
train indirect boundary-near selected rate: 0.0789
train oracle boundary-near selected rate: 0.1438

validation windows: 515
validation mean_jaccard: 0.3824
validation indirect action coverage: 0.5173
validation oracle action coverage: 0.8831
validation indirect boundary-near selected rate: 0.0808
validation oracle boundary-near selected rate: 0.1465
```

Interpretation before final mAP: the coarse-classifier indirect selector is not matching the manual oracle selection distribution closely. Boundary recall is high because the shell still enforces dense local continuity, but selected-frame mass near boundaries and action coverage are much lower than the oracle shell. The full train will test how much performance remains when only the oracle keep-position decision is replaced by the deploy-visible coarse classifier cache.

## Remote Precheck And Full Train

Precheck:

- Run dir: `/data/run01/sczc063/yuzibo/OpenTAD_C3OracleShellIndirect_69aa934_20260701/logs/c3_oracle_shell_indirect_precheck_gpu1_d9a7481_20260701_135939_+0800`
- Result: entered `tools/train.py`, produced finite loss `0.8188`, and stopped after the configured two iterations with exit `0`.

Full train:

- Slurm child: `1118197.571 c3_oracle_full_g1`
- Run dir: `/data/run01/sczc063/yuzibo/OpenTAD_C3OracleShellIndirect_69aa934_20260701/logs/c3_oracle_shell_indirect_full_train_gpu1_d9a7481_20260701_140057_+0800`
- First finite loss: `Loss=1.2794`, `cls_loss=0.6824`, `reg_loss=0.5971`, `mem=2144MB`.
- Current status: running. No final Average-mAP or high-IoU conclusion yet.
