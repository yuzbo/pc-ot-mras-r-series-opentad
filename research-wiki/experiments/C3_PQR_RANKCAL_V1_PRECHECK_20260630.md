# C3 PQR RankCal V1 Precheck 20260630

## Self-check: SparseIrregularQCV2 diagnostic enhancement, 2026-07-01 Asia/Shanghai

- Route: `C3_MAINLINE_OPTIMIZATION / C3_ORIGINAL_OPTIMIZATION_ROUTE`, SparseIrregularQCV2 diagnostic-only enhancement.
- Changed files: `opentad/models/dense_heads/anchor_free_head.py`, `opentad/models/detectors/single_stage.py`, `tools/analyze_c3_pqr_rankcal_proposals.py`, `tests/test_c3_pqr_rankcal_proposal_diagnostics.py`, `tests/test_c3_pqr_rankcal_v1_quality_head.py`.
- Experimental purpose: implement R17 QC V2 failure-attribution matrix next-step diagnostics for pre-NMS/per-candidate survival visibility and selected-time stride>1 normalization.
- Strict random-fixed 50% contract: unchanged; this patch does not modify input sampling, selector policy, config inheritance, launcher behavior, training, or evaluation protocol.
- GT/teacher leakage risk: none added; new diagnostic fields use deploy-time proposals, scores, point metadata, and post-processing NMS/top-k state only.
- Changed surface attribution: detector post-processing diagnostics, analyzer diagnostics, and quality-head geometry feature normalization. No Adapter internals, losses/assignment, selector files, remote launchers, Slurm scripts, or official evaluator logic changed.
- Tensor/shape reasoning: QC V2 point geometry now normalizes `selected_time` by deploy-visible selected-axis length from `irregular_selected_positions`, clamped to `[0, 1]`, avoiding FPN level-length normalization for stride>1 points. Pre-NMS diagnostic records are built after threshold/top-k and before NMS, matched back to NMS survivors by segment and class, then serialized as a nested diagnostic-only candidate dump on the first retained prediction so retained prediction count/order/score remain unchanged.
- Local verification:
  - `python -m py_compile opentad/models/dense_heads/anchor_free_head.py opentad/models/detectors/single_stage.py tools/analyze_c3_pqr_rankcal_proposals.py tests/test_c3_pqr_rankcal_proposal_diagnostics.py tests/test_c3_pqr_rankcal_v1_quality_head.py` -> PASS.
  - `python -m pytest tests/test_c3_pqr_rankcal_proposal_diagnostics.py::test_pqr_proposal_diagnostic_reads_qc_v2_pre_nms_survival_dump -q` -> PASS, 1 passed.
  - `python -m pytest tests/test_c3_pqr_rankcal_proposal_diagnostics.py -q` -> PASS, 15 passed, 3 skipped. Skips are torch/OpenTAD-extension dependent tests; this Windows environment prints torch DLL access-violation traces during skipped imports.
  - `python -m pytest tests/test_c3_pqr_rankcal_v1_quality_head.py -q` -> PASS status with 1 passed, 9 skipped; torch-dependent tests skipped because local torch import triggers Windows DLL access violation.
- Remaining risk: torch-dependent behavioral tests for `single_stage.py` and `_sparse_irregular_qc_v2_point_geometry` could not execute locally because importing torch on this Windows environment triggers access-violation output and the existing tests skip. They should be rerun in the project Linux/conda environment before sync, training, or launch.
- Launch decision: no launch, no sync, no training, no remote action, no commit/push in this task.

## Self-check addendum: review non-blocking follow-up, 2026-07-01 Asia/Shanghai

- Accepted review finding 1: analyzer pre-NMS survival completeness should not require optional `quality_score` or `fused_score`. Handling: `_qc_v2_pre_nms_diagnostic_state` now grades survival core coverage separately from quality/fused coverage. Survival core requires selected/physical proposal, `cls_score` or `score`/`pre_nms_score`, `level_id`, `point_index`, `pre_nms_rank`, `survived_after_topk`, and `survived_after_nms`; optional quality and fused scores are reported as independent coverage counts.
- Accepted review finding 2: rank semantics were ambiguous for single-class/no-topk versus multi-class top-k paths. Handling: detector diagnostic records now include `rank_semantics`; multi-class uses `score_sorted_after_threshold_topk`, while single-class uses `traversal_proposal_order_single_class_no_topk`. Analyzer preserves the field and reports coverage.
- Changed files in this addendum: `opentad/models/detectors/single_stage.py`, `tools/analyze_c3_pqr_rankcal_proposals.py`, `tests/test_c3_pqr_rankcal_proposal_diagnostics.py`, and this report.
- Local verification:
  - `python -m pytest tests/test_c3_pqr_rankcal_proposal_diagnostics.py::test_pqr_proposal_diagnostic_pre_nms_survival_core_does_not_require_quality_or_fused -q` -> PASS, 1 passed.
  - `python -m pytest tests/test_c3_pqr_rankcal_proposal_diagnostics.py::test_pqr_proposal_diagnostic_reads_qc_v2_pre_nms_survival_dump -q` -> PASS, 1 passed.
- Protocol and launch decision unchanged: diagnostic-only, no GT/teacher leakage added, no remote action, no training, no sync, no commit/push.

## Final read-only review and local full focused checks, 2026-07-01 Asia/Shanghai

- Final read-only subagent review: `PASS_SUBAGENT_FINAL_REVIEW_ONLY`, no blocking findings.
- Reviewer allowed next action: stage/commit/push current code, tests, analyzer, and this evidence file; then run remote Linux PRECHECK before any bounded QC V2 shortdiag.
- Reviewer locked actions: official mAP claim, long training, GPU/Slurm training, `tools/test.py`, BH-SDC / `DIVERGENT_INNOVATION_*`, and any test-time GT/teacher/cache leakage path.
- Main-process verification after final review:
  - `git diff --check` -> PASS, only Windows LF/CRLF warnings.
  - `python -m py_compile opentad/models/dense_heads/anchor_free_head.py opentad/models/detectors/single_stage.py tools/analyze_c3_pqr_rankcal_proposals.py tests/test_c3_pqr_rankcal_proposal_diagnostics.py tests/test_c3_pqr_rankcal_v1_quality_head.py` -> PASS.
  - `python -m pytest tests/test_c3_pqr_rankcal_proposal_diagnostics.py -q` -> PASS, `16 passed, 3 skipped`.
  - `python -m pytest tests/test_c3_pqr_rankcal_v1_quality_head.py -q` -> PASS, `1 passed, 9 skipped, 1 warning`.
- Windows limitation: skipped tests are torch/OpenTAD extension dependent because local torch import prints Windows DLL access-violation traces. They must be covered by remote Linux/conda PRECHECK before any GPU shortdiag.
- Launch decision: commit/push is allowed; remote Linux PRECHECK is the next gate. No GPU, Slurm child, training/evaluation, official mAP, runtime/FLOPs, deploy, or paper claim is unlocked by this review.

## Linux PRECHECK fix: QC V2 pre-NMS candidate semantics, 2026-07-01 06:26:19 +08:00

- Remote Linux PRECHECK evidence: commit `96d982bfd416138fd802bd435295e604b6287974`, `pytest tests/test_c3_pqr_rankcal_proposal_diagnostics.py` failed with `1 failed, 18 passed`; failing case was `test_single_stage_post_processing_dumps_pre_nms_qc_v2_survival_fields`, where the test expected `len(qc_v2_pre_nms_candidates) == 2` but Linux produced `4`.
- Root cause: the implementation uses the existing multi-class post-processing contract. For `num_classes > 1`, `rpn_scores` with shape `[N, C]` is flattened to proposal-class candidates, then thresholded and score-sorted before NMS. With the test fixture's 2 proposals and 2 classes plus `pre_nms_thresh=0.0`, the correct pre-NMS diagnostic dump contains `2 * 2 = 4` class-specific candidates.
- Semantic decision: keep the detector/analyzer code unchanged. `qc_v2_pre_nms_candidates` means post-threshold/top-k, class-specific, pre-NMS candidates, not proposal-level candidates. Filtering this dump back down to 2 proposal-level entries would hide class-specific false positives and make `rank_semantics="score_sorted_after_threshold_topk"` misleading.
- Changed files for this fix: `tests/test_c3_pqr_rankcal_proposal_diagnostics.py` and this evidence report only.
- Test update: `test_single_stage_post_processing_dumps_pre_nms_qc_v2_survival_fields` now expects 4 candidates and asserts class order `[Diving, Diving, BaseballPitch, BaseballPitch]`, class indices `[0, 0, 1, 1]`, pre-NMS/post-top-k ranks `[1, 2, 3, 4]`, shared `rank_semantics`, all candidates surviving threshold/top-k, and only the first candidate surviving the mocked NMS.
- Protocol status: diagnostic-only test/evidence fix; no input sampling, dynamic policy, token compression, Adapter/backbone/neck/head internals, detector score logic, losses/assignment, post-processing behavior, configs, launcher, sync, training, Slurm, GPU, official evaluation, or GT/teacher path changed.
- Local verification in this owner task:
  - `git diff --check` -> PASS, only Windows LF/CRLF warnings for the changed files.
  - `python -m py_compile tests/test_c3_pqr_rankcal_proposal_diagnostics.py` -> PASS.
  - `python -m pytest tests/test_c3_pqr_rankcal_proposal_diagnostics.py -q` -> PASS, `16 passed, 3 skipped`; skipped tests are torch/OpenTAD-extension dependent in this Windows environment, where torch import prints access-violation traces before the existing skip path.
- Required next gate: main process may commit/push and retry remote Linux PRECHECK. No remote action is performed in this owner task.
