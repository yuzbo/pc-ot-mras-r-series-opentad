# DIVERGENT BVR-TWB Bounded Diagnostic Evidence 20260630

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630`

Owned branch: `codex/divergent-bvr-twb-final-20260630`

Local HEAD at diagnostic time: `e00d75adc864d9c57e4348deef88313e1561ed63`

Remote Linux CPU clone checked: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024`

Remote commit checked: `92ec024d1821897245a56f993ed040d97e894831`

Timestamp: 2026-06-30 Asia/Shanghai

## Decision Scope

This was a bounded diagnostic stage only. No training, no Slurm child, no GPU, no `tools/test.py`, no official evaluation, no mAP, no checkpoint-producing run, no runtime/FLOPs claim, no deploy claim, and no paper claim were made.

The diagnostic question was whether the latest BVR-TWB final branch is still blocked by implementation or protocol issues before any future launch decision.

## Changed Files

- `tools/bvr_twb/audit_pretrain_load.py`
- `tools/bvr_twb/dump_bridge_roundtrip.py`
- `tests/test_bvr_twb_bounded_diagnostics.py`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_BOUNDED_DIAGNOSTIC_EVIDENCE_20260630.md`

No BVR config, model, detector head, loss, post-processing, full-train launcher, Slurm launcher, C3/ABR/MDL route file, or shared worktree file was modified.

## Diagnostic Outcomes

## Focused Pretrain-Config Repair Update

Timestamp: 2026-06-30 18:17:18 +08:00

Stage: `focused pretrain-config repair`

Writable owner exception: the divergent-route skill is normally delegation-first, but the user explicitly assigned this turn's BVR focused repair to the current agent as the only writable code owner. No other writable agent was used.

Changed files:

- `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`
- `tools/bvr_twb/audit_pretrain_load.py`
- `tests/test_bvr_twb_bounded_diagnostics.py`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_BOUNDED_DIAGNOSTIC_EVIDENCE_20260630.md`

Repair:

- The final BVR config still needs `_delete_=True` for `model.backbone.custom` because it replaces the inherited fixed-length Adapter pipeline with the BVR fixed-length padded bridge pipeline.
- The deleted inherited custom block also contained the canonical VideoMAE-S pretrain path.
- The repair explicitly restores `pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"` inside the final BVR custom block.

Before:

```json
{
  "resolved_pretrain": null,
  "verdict": "BLOCKER_PRETRAIN_MISSING_IN_RESOLVED_CONFIG"
}
```

After:

```json
{
  "blocked": false,
  "resolved_pretrain": "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
  "pretrain_resolves_videomae_s": true,
  "pretrain_file_check_required": false,
  "pretrain_file_exists": false,
  "verdict": "PASS_PRETRAIN_RESOLVED_STATIC_NO_TRAINING"
}
```

Local pretrain audit log:

```text
E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630\logs\bvr_twb_pretrain_config_repair_audit_20260630\summary.json
```

The local audit intentionally did not require checkpoint-file presence because this Windows worktree has no local `pretrained/` checkpoint directory. The static config blocker is cleared locally; file/readability/runtime loading should be rechecked on N16R4 or any environment that has the actual pretrained checkpoint by adding `--check-file` or `--require-runtime`.

Audit/test strengthening:

- `audit_pretrain_load.py` now blocks non-null but wrong pretrain paths with `BLOCKER_PRETRAIN_NOT_VIDEOMAE_S`.
- It distinguishes static resolved-config evidence from file/readability evidence through `--check-file` and `--require-runtime`.
- It accepts repeated `--config` arguments so local and future N16R4 BVR config variants can be audited in one no-training pass.
- `tests/test_bvr_twb_bounded_diagnostics.py` now discovers all `configs/adatad/thumos/*bvr_twb*.py` variants and requires each to resolve the canonical VideoMAE-S pretrain path. At this commit, only `input_bvr_twb_dynamic_adapter_irregular_headv3.py` exists; no separate BVR N16R4 variant exists in this worktree.

Commands and results:

```powershell
python -m py_compile tools\bvr_twb\audit_pretrain_load.py
```

Result: passed.

```powershell
python tools\bvr_twb\audit_pretrain_load.py --out-dir logs\bvr_twb_pretrain_config_repair_audit_20260630
```

Result: exit code 0; `PASS_PRETRAIN_RESOLVED_STATIC_NO_TRAINING`.

```powershell
python -m py_compile tools\bvr_twb\audit_pretrain_load.py tools\bvr_twb\dump_bridge_roundtrip.py tools\bvr_twb\validate_bvr_twb_geometry_contracts.py tools\bvr_twb\audit_opentad_bvr_twb_pipeline.py tools\bvr_twb\audit_sparse_forward_precheck.py
```

Result: passed.

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_bvr_twb_bounded_diagnostics.py tests\test_bvr_twb_geometry_contracts.py tests\test_bvr_twb_opentad_pipeline.py tests\test_bvr_twb_sparse_forward_audit.py tests\test_bvr_twb_validators.py
```

Result: `36 passed, 12 skipped, 1 warning in 4.75s`.

```powershell
python tools\bvr_twb\validate_bvr_twb_geometry_contracts.py
```

Result: passed numpy/source contracts; Windows torch runtime skipped with `WinError 1114` while preserving `no_training=true` and `no_metric_claim=true`.

```powershell
python tools\bvr_twb\audit_opentad_bvr_twb_pipeline.py --out-dir logs\bvr_twb_pretrain_config_repair_pipeline_audit_20260630 --overwrite
```

Result: `ledgers=3 all_validated=True sparse_compute_claim=False blocked=False`.

```powershell
python tools\bvr_twb\audit_sparse_forward_precheck.py --mode fake_raw --out-dir logs\bvr_twb_pretrain_config_repair_sparse_fake_raw_20260630 --overwrite
```

Result: `PASS_LOCAL_SHAPE_ONLY_NO_SPARSE_COMPUTE_CLAIM` for 3 ledgers.

```powershell
python tools\bvr_twb\audit_sparse_forward_precheck.py --mode module_fake_forward --out-dir logs\bvr_twb_pretrain_config_repair_sparse_module_fake_forward_20260630 --overwrite
```

Result: `PASS_REAL_MODULE_FORWARD_NO_METRIC_CLAIM` for 3 ledgers.

Remote note:

- No remote CPU rerun was performed in this repair pass. The next bounded diagnostic may sync/pull the repaired branch into the existing clean remote clone and rerun `audit_pretrain_load.py --check-file` without GPU, Slurm, training, or evaluation.

Tracker/log note:

- This owned worktree does not contain `research-wiki/log.md` or `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.
- The user-provided write whitelist for this stage allowed BVR route reports but did not allow tracker/log writes. To avoid shared-worktree or out-of-scope writes, only this BVR route report was updated.

Current decision:

- `BLOCKER_PRETRAIN_MISSING_IN_RESOLVED_CONFIG` is cleared by local static resolved-config evidence.
- BVR formal/full training remains locked.
- Allowed next action is bounded diagnostic rerun / read-only review / optional N16R4 CPU pretrain-file audit, not Slurm/full training.

## N16R4 CPU Pretrain File Audit After Repair

Timestamp: 2026-06-30 18:26:39 +08:00

Scope: `PRECHECK_ONLY / CPU / no Slurm / no GPU / no training / no evaluation`.

Remote clone:

```text
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024
```

Remote branch state:

```text
HEAD=af00464
```

The clone was fast-forwarded from `92ec024` to `af00464`. The code resolved
`cfg.model.backbone.custom.pretrain` to the canonical VideoMAE-S path, but the
clone initially lacked its runtime resource symlink:

```text
pretrained -> ../pretrained
```

The parent resource directory and checkpoint were already present under the
authorized N16R4 workspace:

```text
../pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
```

The coordinator created only the missing clone-local resource symlink and then
ran:

```bash
/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python \
  tools/bvr_twb/audit_pretrain_load.py \
  --check-file \
  --out-dir logs/bvr_twb_pretrain_repair_af00464_cpu_check_symlink_20260630_182639_+0800/pretrain_check_file
```

Audit result:

```json
{
  "blocked": false,
  "pretrain_file_check_required": true,
  "pretrain_file_exists": true,
  "pretrain_resolves_videomae_s": true,
  "resolved_pretrain": "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
  "has_backbone_like_keys": true,
  "state_dict_num_keys": 163,
  "verdict": "PASS_PRETRAIN_RESOLVED_AND_READABLE_NO_TRAINING"
}
```

Log paths:

```text
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_pretrain_repair_af00464_cpu_check_symlink_20260630_182639_+0800/precheck.log
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_pretrain_repair_af00464_cpu_check_symlink_20260630_182639_+0800/pretrain_check_file/summary.json
```

Wrapper note:

- The audit itself printed `AUDIT_RC=0` and wrote the pass summary above.
- The outer SSH command returned nonzero because a CR character in the shell
  wrapper made `exit 0` parse as a non-numeric argument. This is recorded as a
  wrapper anomaly, not as an audit failure.

Current decision after this audit:

- The resolved-config pretrain blocker is cleared locally and on N16R4.
- The actual VideoMAE-S checkpoint is present and readable in the N16R4 clone
  through the restored resource symlink.
- BVR formal/full training remains locked pending the remaining bounded
  diagnostic/review/launch decision gates; this audit does not unlock Slurm
  training, `tools/test.py`, official mAP, runtime/FLOPs, deploy, paper, sparse
  compute, or combo claims.

### 1. Historical Runtime Pretrain-Load Audit Before Repair

Historical status before focused repair: **BLOCKED**

The final BVR config resolves `cfg.model.backbone.custom` without a `pretrain` key:

```json
"custom_cfg_keys": [
  "freeze_backbone",
  "norm_eval",
  "post_processing_pipeline",
  "pre_processing_pipeline",
  "trainable_backbone_keywords"
],
"resolved_pretrain": null,
"verdict": "BLOCKER_PRETRAIN_MISSING_IN_RESOLVED_CONFIG"
```

Interpretation: `BackboneWrapper` would take its random-initialization warning path instead of loading the expected VideoMAE-S checkpoint `vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`. This is a real implementation/protocol blocker and is sufficient to keep BVR formal/full training locked.

Local log:

```text
E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630\logs\bvr_twb_bounded_diagnostic_pretrain_audit_20260630\summary.json
```

Remote Linux CPU log:

```text
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/bvr_twb_pretrain_static.out
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/bvr_twb_pretrain_static/summary.json
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/bvr_twb_pretrain_static.exitcode = 2
```

### 2. Small Validation-Window Ledger / Bridge Dump

Status: **PASS as bounded geometry evidence**

`dump_bridge_roundtrip.py` wrote dynamic-BVR and forced-uniform-through-BVR-bridge ledgers. Both ledgers validated, preserved native GT axis, produced selected raw-frame indices, detector feature positions, detector valid lengths, adapter bridge fields, and seconds round-trip fields.

Local log:

```text
E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630\logs\bvr_twb_bounded_diagnostic_bridge_roundtrip_20260630\bridge_roundtrip_summary.json
E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630\logs\bvr_twb_bounded_diagnostic_bridge_roundtrip_20260630\bridge_roundtrip_rows.json
E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630\logs\bvr_twb_bounded_diagnostic_bridge_roundtrip_20260630\bridge_roundtrip_ledgers.jsonl
```

Key summary:

```json
{
  "all_ledgers_validated": true,
  "all_seconds_roundtrip_match": true,
  "cases": [
    "dynamic_bvr",
    "forced_uniform_through_bvr_bridge"
  ],
  "forced_uniform_raw_valid_k": 192,
  "forced_uniform_detector_feature_valid_k": 96,
  "forced_uniform_adapter_padding_duplicate_count": 0,
  "no_metric_claim": true
}
```

### 3. Forced-Uniform-Through-BVR-Bridge Sanity Gate

Status: **PASS as bounded bridge/control geometry evidence**

The forced-uniform diagnostic used 192 uniformly selected native positions from a dense length of 384, passed them through the BVR adapter bridge, and produced 96 detector feature centers at feature stride 2 without coordinate collapse. This is a bridge sanity/control check only. It does not claim BVR selector quality.

### 4. Single-Batch Geometry Round-Trip

Status: **PASS as bounded geometry evidence**

The local round-trip dump checked:

- native detector-grid probe segments;
- selected raw positions and selected frame indices;
- detector feature positions and valid mask length;
- native-axis GT segments;
- seconds conversion formula under `irregular_native_axis=True`;
- local torch skip reason when Windows torch DLL import fails.

Linux CPU torch runtime contract also passed on the remote clone:

```text
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/geometry_require_torch.out
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/geometry_require_torch.exitcode = 0
```

Remote geometry output included:

```json
{
  "torch_runtime_contract": "passed",
  "torch_runtime_skipped": false,
  "loader_frame_count": 192,
  "detector_feature_valid_k": 14,
  "no_training": true,
  "no_metric_claim": true
}
```

### 5. Proposal / NMS / Score Diagnostic

Status: **PENDING**

This requires checkpoint/evaluation artifacts or proposal outputs. It was not run because `tools/test.py`, official evaluation, checkpoint-producing runs, and mAP/runtime claims are forbidden in this bounded diagnostic stage.

## Commands And Results

Local commands:

```powershell
python -m py_compile tools\bvr_twb\audit_pretrain_load.py tools\bvr_twb\dump_bridge_roundtrip.py tools\bvr_twb\validate_bvr_twb_geometry_contracts.py tools\bvr_twb\audit_opentad_bvr_twb_pipeline.py tools\bvr_twb\audit_sparse_forward_precheck.py
```

Result: passed.

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_bvr_twb_bounded_diagnostics.py tests\test_bvr_twb_geometry_contracts.py tests\test_bvr_twb_opentad_pipeline.py tests\test_bvr_twb_sparse_forward_audit.py tests\test_bvr_twb_validators.py
```

Result: `36 passed, 12 skipped, 1 warning in 4.60s`.

```powershell
python tools\bvr_twb\audit_pretrain_load.py --out-dir logs\bvr_twb_bounded_diagnostic_pretrain_audit_20260630
```

Result: exit code nonzero by design; blocker `BLOCKER_PRETRAIN_MISSING_IN_RESOLVED_CONFIG`.

```powershell
python tools\bvr_twb\dump_bridge_roundtrip.py --out-dir logs\bvr_twb_bounded_diagnostic_bridge_roundtrip_20260630 --overwrite
```

Result: pass; dynamic and forced-uniform bridge rows written.

Remote Linux CPU commands were run with Windows native OpenSSH only, no WSL, no Slurm, no GPU:

```text
cd /data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024
/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch
/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python tools/bvr_twb/audit_opentad_bvr_twb_pipeline.py --out-dir logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/bvr_twb_pipeline --overwrite
/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python tools/bvr_twb/audit_sparse_forward_precheck.py --mode fake_raw --out-dir logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/bvr_twb_sparse_fake_raw --overwrite
/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python tools/bvr_twb/audit_sparse_forward_precheck.py --mode module_fake_forward --out-dir logs/bvr_twb_bounded_diagnostic_20260630_remote_cpu/bvr_twb_sparse_module_fake_forward --overwrite
```

Remote results:

```text
geometry_require_torch.exitcode = 0
bvr_twb_pipeline.exitcode = 0
bvr_twb_sparse_fake_raw.exitcode = 0
bvr_twb_sparse_module_fake_forward.exitcode = 0
bvr_twb_pretrain_static.exitcode = 2
```

## Self-Check

Changed surface:

- Diagnostic tools only.
- Focused diagnostic tests only.
- Route report only.

No changed surface:

- Input sampling implementation: unchanged.
- Dynamic budget policy: unchanged.
- Token compression: unchanged.
- Adapter/backbone internals: unchanged.
- Detector head logic: unchanged.
- Loss/assignment: unchanged.
- Test-time post-processing: unchanged.
- Full-training launcher / Slurm: unchanged.

Strict random-fixed 50% contract status:

- Not modified. This diagnostic route is BVR-specific and no random-fixed baseline/config was edited.

GT/teacher/cache leakage risk:

- No validation/test GT was used for selection.
- No teacher, prediction cache, raw detector prediction, oracle boundary, or oracle residual source was used.
- Train-only value labels were not used at validation/test/deploy selection time.

Tensor/mask reasoning:

- Adapter bridge distinguishes padded raw inputs from valid raw observations.
- Padding duplicates are not counted as valid.
- Detector valid mask counts detector feature centers, not padded Adapter inputs.
- BVR detector feature centers are grouped from raw selected positions by feature stride.
- Native-axis post-processing remains required when BVR detector feature positions are present.

Review gate note:

- Exact required subagent configuration (`model=gpt-5.5`, `reasoning_effort=high`, normal speed) was not available through the callable review tools in this environment. Claude/Gemini/weak/default review tools were deliberately not used. Because the required final read-only subagent review is unavailable and the pretrain blocker is real, this report does not unlock launch or deployment.

## Historical Blocker And Current Next Action

Historical blocking finding, now cleared by the focused repair above:

`input_bvr_twb_dynamic_adapter_irregular_headv3.py` deletes inherited `model.backbone.custom` and does not restore the VideoMAE-S `pretrain` path. The resolved final config would silently skip pretrain loading and random-initialize the backbone.

Allowed next action after repair:

`BOUNDED_DIAGNOSTIC_ONLY` rerun and read-only review, optionally including N16R4 CPU `audit_pretrain_load.py --check-file` in the existing clean remote clone. Do not proceed to Slurm/full training until the required gates explicitly allow it.

Still locked:

- BVR formal/full training.
- Slurm long train.
- GPU1.
- `tools/test.py`.
- Official evaluation / mAP.
- Proposal/NMS/score claim.
- Sparse compute claim.
- Runtime/FLOPs claim.
- Deploy claim.
- Paper claim.
- Any C3/BVR/ABR/MDL combo or merge.

## 2026-06-30 19:28:58 +08:00 Pro-Requested Severe-Collapse D0-D5 Local Diagnostic Tooling

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630`

Owned branch: `codex/divergent-bvr-twb-final-20260630`

Writable owner exception: the divergent-route skill is delegation-first, but the user explicitly assigned this stage to the current agent as the only writable BVR-TWB code owner. No other writable worker was used.

Pro decision being satisfied:

- Severe BVR collapse remains frozen for formal training.
- New training must not proceed until runtime provenance proves the loaded `LoadFrames.__call__` contains `bvr_twb_dynamic_subsample` and calls `build_bvr_twb_open_tad_selection`.
- A one-sample pipeline trace must prove BVR-selected raw frame indices are the indices handed to decode/backbone metadata, with ledger, mask, and GT native-axis evidence.
- Detector grid/mask audit must prove native-axis detector feature positions enter the model.
- Post-processing audit must explain or prove the proposal-count expansion behind `365940` predictions.

Changed files in this stage:

- `tools/bvr_twb/audit_runtime_provenance.py`
- `tools/bvr_twb/trace_one_sample_pipeline.py`
- `tools/bvr_twb/audit_postprocess_proposals.py`
- `tests/test_bvr_twb_runtime_diagnostics.py`
- `opentad/datasets/transforms/end_to_end.py`
- `opentad/models/detectors/irregular_actionformer.py`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_BOUNDED_DIAGNOSTIC_EVIDENCE_20260630.md`

Implemented D0-D5 bounded diagnostics:

- D0 runtime provenance audit tool: `audit_runtime_provenance.py` checks source tokens, loaded function provenance when torch runtime is safe, function hashes, dispatch ledger, and no-training/no-video/no-metric status.
- D1 fail-closed dispatch: unsupported `LoadFrames.method` now raises `ValueError("Unsupported LoadFrames method: ...")` instead of falling through to an implicit undefined-variable failure.
- D2 one-sample pipeline trace: `trace_one_sample_pipeline.py` uses synthetic metadata, BVR `LoadFrames`, fake decode, NCTHW formatting, and collected meta keys to prove `dispatch_hit`, ledger, `frame_inds`, selected-frame subset, mask count, detector positions, and native-axis GT preservation when CPU torch runtime is available.
- D3 detector temporal-grid audit: `IrregularActionFormer` now has env-gated `BVR_TWB_GRID_AUDIT=1` / `BVR_TWB_GRID_AUDIT_PATH=...` JSONL audit rows proving BVR detector feature positions become model temporal-grid centers. Default behavior is unchanged.
- D4 post-processing proposal-count audit: `IrregularActionFormer.post_processing` now has env-gated `BVR_TWB_POSTPROCESS_AUDIT=1` / `BVR_TWB_POSTPROCESS_AUDIT_PATH=...` JSONL count rows. `audit_postprocess_proposals.py` constructs the diagnostic-only formula `19260 raw proposals * 19 classes = 365940 flattened candidates`, then records threshold/top-k/NMS/result counts when CPU torch runtime is available.
- D5 tests and documentation: `tests/test_bvr_twb_runtime_diagnostics.py` covers source provenance, unsupported method fail-closed, diagnostic output keys without GPU, grid audit hook, and postprocess audit helper/source behavior.

Local commands and results:

```powershell
python -m py_compile tools\bvr_twb\audit_runtime_provenance.py tools\bvr_twb\trace_one_sample_pipeline.py tools\bvr_twb\audit_postprocess_proposals.py tests\test_bvr_twb_runtime_diagnostics.py opentad\models\detectors\irregular_actionformer.py opentad\datasets\transforms\end_to_end.py
```

Result: passed.

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_bvr_twb_runtime_diagnostics.py
```

Result: `3 passed, 2 skipped in 3.75s`.

Focused BVR suite rerun after the new diagnostics:

```powershell
$env:PYTHONPATH=(Get-Location).Path; pytest -q tests\test_bvr_twb_bounded_diagnostics.py tests\test_bvr_twb_geometry_contracts.py tests\test_bvr_twb_opentad_pipeline.py tests\test_bvr_twb_sparse_forward_audit.py tests\test_bvr_twb_validators.py tests\test_bvr_twb_runtime_diagnostics.py
```

Result: `39 passed, 14 skipped, 1 warning in 9.67s`.

The two skipped runtime tests are expected on this Windows environment because torch import is not safe here. The tools now probe torch in a child process first, so local source/schema checks do not crash or falsely pass runtime provenance.

```powershell
python tools\bvr_twb\audit_runtime_provenance.py --out-dir tools\bvr_twb\.tmp_bvr_twb_runtime_provenance_cli --overwrite
```

Result: `runtime=skipped dispatch_hit=False blocked=False`.

```powershell
python tools\bvr_twb\trace_one_sample_pipeline.py --out-dir tools\bvr_twb\.tmp_bvr_twb_one_sample_trace_cli --overwrite
```

Result: `runtime=skipped dispatch_hit=False trace_keys=False`.

```powershell
python tools\bvr_twb\audit_postprocess_proposals.py --out-dir tools\bvr_twb\.tmp_bvr_twb_postprocess_cli --overwrite
```

Result: `runtime=skipped flattened=None explains_365940=False`.

```powershell
git diff --check -- tools/bvr_twb/audit_runtime_provenance.py tools/bvr_twb/trace_one_sample_pipeline.py tools/bvr_twb/audit_postprocess_proposals.py tests/test_bvr_twb_runtime_diagnostics.py opentad/models/detectors/irregular_actionformer.py opentad/datasets/transforms/end_to_end.py
```

Result: exit code `0`; only Windows LF-to-CRLF warnings.

Generated `.tmp_bvr_twb*` and `__pycache__` directories from local verification were removed after the checks.

Current blocker:

- This Windows environment still cannot provide the required runtime proof because torch import fails in the child probe. Therefore D0-D5 runtime evidence is implemented but not fully proven locally.
- No mAP, runtime, sparse compute, deploy, or paper claim is made.

Allowed next action:

- Run the three new tools in a Linux CPU environment with torch available, using `--require-runtime`, plus the focused pytest file. This remains bounded diagnostic-only and does not require GPU, real THUMOS videos, Slurm, checkpoint, or `tools/test.py`.

Suggested Linux CPU commands:

```bash
python tools/bvr_twb/audit_runtime_provenance.py --out-dir logs/bvr_twb_runtime_provenance_d0 --overwrite --require-runtime
python tools/bvr_twb/trace_one_sample_pipeline.py --out-dir logs/bvr_twb_one_sample_trace_d1 --overwrite --require-runtime
python tools/bvr_twb/audit_postprocess_proposals.py --out-dir logs/bvr_twb_postprocess_count_d4 --overwrite --require-runtime
PYTHONPATH="$PWD" pytest -q tests/test_bvr_twb_runtime_diagnostics.py
```

Still locked:

- BVR formal/full training.
- Slurm or remote sync for training.
- GPU allocations.
- `tools/test.py` and official evaluation.
- mAP, runtime/FLOPs, sparse compute, deploy, or paper claims.
- Any C3/BVR/ABR/MDL combo or merge.

## 2026-06-30 N16R4 Runtime Diagnostics After GitHub Proxy Sync

Remote GitHub synchronization was performed on N16R4 with the platform academic
proxy. The existing remote clone
`/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024`
fast-forwarded to branch `codex/divergent-bvr-twb-final-20260630`, commit
`46337e2`.

Bounded Linux diagnostics were then run without GPU training, `tools/test.py`,
official evaluation, checkpoint claims, or sparse-compute claims:

```bash
python tools/bvr_twb/audit_runtime_provenance.py --out-dir logs/bvr_twb_runtime_bounded_diagnostics_46337e2_20260630_195852_+0800_r2/runtime_provenance_d0 --require-runtime
python tools/bvr_twb/trace_one_sample_pipeline.py --out-dir logs/bvr_twb_runtime_bounded_diagnostics_46337e2_20260630_195852_+0800_r2/one_sample_trace_d1 --require-runtime
python tools/bvr_twb/audit_postprocess_proposals.py --out-dir logs/bvr_twb_runtime_bounded_diagnostics_46337e2_20260630_195852_+0800_r2/postprocess_count_d4 --require-runtime
PYTHONPATH="$PWD" pytest -q tests/test_bvr_twb_runtime_diagnostics.py
```

Results:

- Runtime provenance: `runtime_contract=passed`, `dispatch_hit=True`, `blocked=False`.
- One-sample trace: `dispatch_hit=True`, `trace_keys=True`, `blocked=False`.
- Focused pytest: `5 passed in 46.07s`.
- Postprocess proposal-count audit: `runtime_contract=passed`, `blocked=False`,
  `raw_proposal_count=19260`, `num_classes=19`,
  `flattened_candidate_count=365940`, `above_threshold_count=365940`,
  `pre_nms_selected_count=2000`, `post_nms_count=2000`,
  `explains_365940_predictions=True`.

Evidence root:

`/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_runtime_bounded_diagnostics_46337e2_20260630_195852_+0800_r2`

Interpretation:

- The severe-run `365940` prediction count is now localized to the
  postprocess raw-proposal by class flattening stage:
  `19260 raw proposals * 19 classes = 365940 flattened candidates`.
- This does not prove the BVR idea has failed. It supports the earlier Pro
  diagnosis that the severe result is an implementation/protocol/postprocess
  failure mode rather than route rejection.
- BVR formal/full training remains locked. The next BVR work should focus on
  proposal/postprocess control or a reviewed repair before any new long run.
