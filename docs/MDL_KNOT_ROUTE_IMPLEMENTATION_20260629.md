# MDL-Knot Route Implementation Record

Route label: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Implementation target: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_FullCode_Worktree_20260629`

Superseded evidence-only worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_Worktree_20260629`

This first implementation remains fail-closed. Linux/N16R4 PRECHECK_ONLY passed on 2026-06-30 after the clean-clone dependency fix, but this does not approve remote sync, Slurm, training, evaluation, runtime/FLOPs, deploy, paper, sparse-compute, or metric claims.

Implemented surfaces:

- Input sampling: yes, via `method="mdl_knot_dynamic_subsample"`.
- Dynamic budget policy: yes, greedy MDL-Knot with residual, gap, transition, and short-island stopping.
- Token compression: no.
- Adapter/backbone internals: no new model logic.
- Detector head logic: no.
- Loss/assignment: no.
- Test-time post-processing: no.

Protocol boundaries:

- No validation/test ground truth enters the selector.
- No teacher outputs enter the selector.
- No prediction cache enters the selector.
- No dense raw backbone handoff is accepted by the real-sparse validator.
- Selected positions are original dense-time indices; padding duplicates do not count as valid sparse entries.
- Current Adapter compatibility uses `fixed_pad`: MDL chooses dynamic `valid_k`, then raw frame indices are padded to the inherited Adapter target length while masks and sparse metadata keep only true selected entries valid.
- Scout provenance is fail-closed. Any incoming scout marked as GT, teacher, prediction-cache, or dense-backbone sourced is rejected before ledger creation.
- The current full-code config still uses `deploy_scout_source="fallback_synthetic_precheck_only"` and `real_scout_unavailable=True`; this permits local/remote precheck only and is not a deploy-visible scout claim.
- Clean-clone dependency: `opentad/datasets/transforms/pseudo_boundary.py` is included because the real `end_to_end.py` imports the shared pseudo-boundary helper. This is a transform dependency only; it does not change MDL-Knot selection semantics or merge MDL with C3/BVR/ABR routes.

Local gates:

- `python -m pytest tests -q`
- `python tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py --out-dir .tmp_mdl_knot_precheck --overwrite`
- `python tools/mdl_knot/validate_mdl_knot_launch_gate.py --route-label DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3 --precheck-summary .tmp_mdl_knot_precheck/mdl_knot_precheck_summary.json`
- `python -m py_compile` on route files and tools.

Linux/N16R4 PRECHECK_ONLY evidence:

Recorded 2026-06-30 Asia/Shanghai.

- Remote logdir: `/data/home/sczc063/run/yuzibo/mdl_knot_precheck_logs/mdl_gpu1_precheck_20260630_010030`
- Remote clean clone: `/data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_Precheck_20260630_6e8b698`
- Commit tested: `6e8b698 Fix MDL-Knot clean-clone transform dependency`
- Protected hold: `1118197 pcot_dbg2g`, node `g0030`, `CUDA_VISIBLE_DEVICES=1`; parent hold was not released or cancelled.
- Result: `20 passed in 46.43s`
- Precheck summary: `logs/mdl_knot_precheck_gpu1/mdl_knot_precheck_summary.json`
- Gate output: `PRECHECK_ONLY_REQUEST_ALLOWED`
- Gate still locks: remote_sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, push, mAP/runtime/FLOPs/deploy/paper claims.

Interpretation: Linux/N16R4 PRECHECK_ONLY passed after the clean-clone dependency fix. This still does not unlock full train, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper, or sparse-compute claims.
