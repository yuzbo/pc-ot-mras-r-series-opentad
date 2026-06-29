# MDL-Knot Route Implementation Record

Route label: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Implementation target: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_FullCode_Worktree_20260629`

Superseded evidence-only worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_Worktree_20260629`

This first implementation is local-only and fail-closed. It implements a trainable/precheckable sparse acquisition surface, but it does not approve remote sync, Slurm, training, evaluation, runtime/FLOPs, deploy, paper, or metric claims.

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
