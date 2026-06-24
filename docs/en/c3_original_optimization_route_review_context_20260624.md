# C3 Original Optimization Route Review Context - 2026-06-24

Route label: `C3_ORIGINAL_OPTIMIZATION_ROUTE` / `C3_MAINLINE_OPTIMIZATION`.

This file is intentionally scoped to the C3/C3-Pro route only. It must not be merged with or used to review divergent routes such as `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`, `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`, `DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3`, or `DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`. Any future combined experiment requires an explicit `COMBO_ROUTE_APPROVED` gate first.

## Original Intent

C3 was created to test whether a deployable pre-backbone frame selector can improve temporal action detection while preserving the original AdaTAD/ActionFormer backend as much as possible.

The intended C3 contract is:

- The selector sees only deploy-visible compressed low-resolution video pixels before the VideoMAE backbone.
- The hard forward path must select real frames, not interpolated features, raw prediction cache rows, teacher rows, or test-time GT.
- Training and test-time hard selected indices must follow the same distribution.
- Downstream detector loss should be able to send gradient signal back through a straight-through surrogate.
- P2 heads, offline ledgers, teachers, validation/test GT, raw prediction cache, and direct `tools/test.py` shortcuts are forbidden for deployable claims.
- Selection must be closed-loop with detector temporal geometry, masks, GT assignment, and post-processing.
- High-IoU localization and proposal score/rank quality matter more than candidate recall alone.
- Fixed `384/768` is only the controlled first-stage budget; the long-term target is dynamic budget allocation.

## Implemented C3 Candidates In This Branch

The current C3 optimization branch adds local candidates for review:

1. `C3-Pro global-rank ST`
   - Config: `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_global_rank_st_full_train_candidate_n16r4.py`
   - Keeps hard selection as `frame_score_topk`.
   - Changes only the straight-through surrogate from local neighborhood softmax to global candidate softmax.

2. `C3 interval boundary packet`
   - Config: `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_interval_packet_full_train_candidate_n16r4.py`
   - Uses deploy-visible dense heads to form start/end/action interval records.
   - Fills boundary and interior packets, then sorts selected real dense-frame positions for the hard detector input.
   - `interval_candidate_topk` now truly bounds start/end pair enumeration, and metadata records `interval_pair_limit`, `interval_record_count`, and `interior_candidate_count`.

3. `C3 dynamic marginal-budget guard`
   - Config: `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_dynamic_marginal_budget_guard_candidate_n16r4.py`
   - Uses deploy-visible reader heads to choose per-sample budgets in a capped `320/384/448` frontier.
   - Capacity is explicitly `448` across selector target length, reader slots, backbone total frames, and projection max sequence length.
   - The max-gap guard is a safety gate, not evidence of intelligent acquisition.

4. `C3 physical-grid ActionFormer`
   - Config: `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_physical_grid_actionformer_fixed384_candidate_n16r4.py`
   - Keeps selector hard path as frame-score-first fixed `384/768`.
   - Enables a physical temporal grid in `ActionFormerHead` so assignment/decode uses real selected dense positions.
   - The config sets `frame_selector.remap_gt_to_selected_axis=False`.
   - Runtime fail-fast rejects physical-grid training if metadata says selected-axis GT remapping is still enabled.

5. `C3 post-training selector diagnostics`
   - Tool: `tools/bata/analyze_pc_ot_mras_selector_posttrain_diagnostics.py`
   - Reads metadata/result dumps after full C3/C3-Pro training.
   - Reports selected indices, valid lengths, gaps, duplicate rates, boundary-near support, packet roles, dynamic budget distribution, physical-grid metadata consistency, score/rank hooks, and NaN/Inf paths.

## Known Review Questions

Please review this GitHub branch strictly from the C3 original route perspective:

1. Does the current implementation still match the original C3 purpose, or has it drifted into a different route?
2. Are the hard selected frames, masks, metadata, GT remapping, physical time mapping, assignment, and post-processing contracts internally consistent?
3. Does the interval-packet implementation have acceptable complexity for fixed `384/768` training after the `interval_candidate_topk` bound?
4. Does the physical-grid ActionFormer path fully avoid the selected-axis GT vs dense-axis points mismatch?
5. Could the current straight-through surrogate, auxiliary losses, dynamic budget mapping, or max-gap guard cause finite-loss but collapsed mAP behavior?
6. What diagnostics are mandatory before a long run result can be interpreted, especially for score/rank quality and high-IoU localization?
7. Is it acceptable to sync this C3 branch only for remote `PRECHECK_ONLY` and later queue isolated candidates behind the current running jobs?

Do not approve direct replacement of the current running/queued C3-CNN-Lite and C3-Pro sequence. Do not approve detector mAP claims, runtime/FLOPs claims, deployment claims, or paper claims from this local implementation alone.

Expected verdict options:

- `APPROVE_REMOTE_PRECHECK_ONLY_DO_NOT_REPLACE_CURRENT_RUN`
- `APPROVE_AFTER_LOCAL_FIXES_THEN_PRECHECK_ONLY`
- `FIX_BEFORE_REMOTE_PRECHECK`
- `REJECT_CURRENT_C3_MECHANISM_ROUTE`
- `CONTEXT_INSUFFICIENT`

## Local Verification Before Commit

Latest focused C3 checks after the two blocking fixes:

```text
conda run -n torch_1 python -m pytest -q tests\test_pc_ot_mras_prebackbone_interval_rank_transport.py tests\test_pc_ot_mras_prebackbone_next_mechanism_configs.py tests\test_physical_grid_actionformer_head.py tests\test_pc_ot_mras_prebackbone_c3_pro_dynamic_budget.py
16 passed in 27.94s
```

Also run:

```text
conda run -n torch_1 python -m py_compile opentad\models\selectors\pc_ot_mras_prebackbone_frame_selector.py opentad\models\dense_heads\anchor_free_head.py tests\test_physical_grid_actionformer_head.py tests\test_pc_ot_mras_prebackbone_interval_rank_transport.py
PASS

git diff --check
PASS, with LF/CRLF warnings only
```
