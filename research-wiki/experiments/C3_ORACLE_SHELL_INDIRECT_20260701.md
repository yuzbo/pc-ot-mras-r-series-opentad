# C3 Oracle-Shell Indirect 2026-07-01

## Status

- Timestamp: 2026-07-01 Asia/Shanghai
- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`
- Worktree: `OpenTAD_C3OracleShellIndirect_Worktree_20260701`
- Branch: `codex/c3-oracle-shell-indirect-20260701`
- Base commit: `c9b8cd6cdc26307048356258d4a9f2dc92b72b1f`
- Current evidence: local implementation and gate candidate complete; no remote deploy, no Slurm, no long training, no mAP.

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
13 passed
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
- Next allowed action: after final read-only review, commit/push the branch, clone a clean remote directory, export a real coarse score cache from the trained coarse selector checkpoint, run oracle-vs-indirect distribution diagnostics, then run N16R4 GPU1 precheck and start full training. Full mAP is pending complete training.
