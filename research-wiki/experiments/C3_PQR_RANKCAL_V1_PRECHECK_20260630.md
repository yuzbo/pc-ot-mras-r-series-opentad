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
