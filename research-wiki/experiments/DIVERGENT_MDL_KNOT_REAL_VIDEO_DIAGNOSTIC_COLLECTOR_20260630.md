# MDL-Knot Real-Video Diagnostic Collector

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`

Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`

Timestamp: 2026-06-30 Asia/Shanghai

## Scope

This stage adds a real-video pipeline diagnostic collector for MDL-Knot. It is a diagnostic/gate evidence stage only. It does not run training, evaluation, `tools/test.py`, Slurm, SSH, remote sync, checkpoint loading, mAP reporting, runtime/FLOPs claims, deployment claims, paper claims, or sparse-compute claims.

MDL-Knot remains positioned as a stable-span compression and dynamic candidate generator. It is not promoted to a direct long-training mainline route by this change.

## Changed Files

- `tools/mdl_knot/collect_mdl_knot_real_video_diagnostics.py`
- `tools/mdl_knot/validate_mdl_knot_shortdiag.py`
- `opentad/acquisition/mdl_knot/diagnostics.py`
- `tests/test_mdl_knot_realdiag.py`
- `research-wiki/experiments/DIVERGENT_MDL_KNOT_REAL_VIDEO_DIAGNOSTIC_COLLECTOR_20260630.md`

## Collector Behavior

The collector reads the MDL-Knot config, runs the configured `LoadFrames(method="mdl_knot_dynamic_subsample")` path when the OpenTAD transform stack can be imported, and otherwise uses a torch-free equivalent of the same MDL-Knot scout/selector/fixed-pad handoff for local diagnostic collection.

It emits a formal-readiness summary JSON with:

- `synthetic=false`
- raw-frame scout count and ratio
- metadata fallback count and ratio
- synthetic fallback count and ratio
- `valid_k` min, max, mean, std, unique count, and nonconstant marker
- selected gap stats
- mask/meta alignment status
- short/transition guard coverage fields
- fixed-pad sparse-compute claim lock
- route drift and claim lock
- no mAP, no `tools/test.py`, and no train markers

The formal validator can accept the summary as `FORMAL_READINESS_DIAGNOSTICS_PRESENT`, but formal training remains locked and still requires an explicit later decision.

## Example Usage

Fixture/local schema check:

```powershell
python tools\mdl_knot\collect_mdl_knot_real_video_diagnostics.py `
  --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3.py `
  --out <out_dir>\mdl_knot_realdiag_formal_summary.json `
  --dry-run-fixture `
  --window-count 4 `
  --shortdiag-log <path_to_one_epoch_shortdiag_log>
```

Real THUMOS windows:

```powershell
python tools\mdl_knot\collect_mdl_knot_real_video_diagnostics.py `
  --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3.py `
  --annotation <path_to_thumos_14_anno.json> `
  --video-root <path_to_thumos_videos> `
  --window-count 8 `
  --out <out_dir>\mdl_knot_realdiag_formal_summary.json `
  --shortdiag-log <path_to_one_epoch_shortdiag_log>
```

Validate the summary:

```powershell
python tools\mdl_knot\validate_mdl_knot_launch_gate.py `
  --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3.py `
  --formal-readiness-summary <out_dir>\mdl_knot_realdiag_formal_summary.json
```

Expected validator state: `FORMAL_READINESS_DIAGNOSTICS_PRESENT`, followed by a still-locked formal training statement.

## Verification

- `python -m py_compile opentad\acquisition\mdl_knot\diagnostics.py tools\mdl_knot\collect_mdl_knot_real_video_diagnostics.py tools\mdl_knot\validate_mdl_knot_launch_gate.py tests\test_mdl_knot_realdiag.py`
  - Passed.
- `python -m pytest tests\test_mdl_knot_realdiag.py -q`
  - Passed: `6 passed`.
- `python -m pytest tests\test_mdl_knot_core.py tests\test_mdl_knot_shortdiag.py tests\test_mdl_knot_tools_and_integration.py tests\test_mdl_knot_realdiag.py -q`
  - Passed: `49 passed, 1 skipped`.

## Remaining Locks

- No formal full training approval exists.
- No remote sync, SSH, Slurm, or GPU launch was performed.
- No evaluation or `tools/test.py` was run.
- No mAP, runtime/FLOPs, sparse-compute, deployment, or paper claim exists.
- Real-video summary quality depends on actual THUMOS videos being visible to the collector; if videos are missing or Decord cannot open them, raw-frame scout ratio may be too low and the formal validator will stay locked.
- GPT-5.5 Pro was not invoked in this stage. A later Pro packet can use the emitted formal-readiness JSON plus this report, but Pro is still needed before any paper/deploy/final metric claim or formal full-train decision.

## Final Review Blocker Fix

The read-only final review reported that `--dry-run-fixture` output could masquerade as real-video formal-readiness evidence because the summary hardcoded `synthetic=false` and lacked source provenance. This blocker is fixed.

The collector now records:

- `source_mode`
- `dry_run_fixture`
- `annotation_path`
- `video_root`
- `reader_backend`
- `real_video_reader_window_count`
- `real_video_raw_scout_count`
- `fixture_window_count`

The formal-readiness validator now rejects fixture/dry-run summaries, missing source provenance, missing annotation/video-root provenance, fixture reader backends, and summaries with no real-video reader windows. Fixture output remains available for schema tests only and cannot produce `FORMAL_READINESS_DIAGNOSTICS_PRESENT`.
