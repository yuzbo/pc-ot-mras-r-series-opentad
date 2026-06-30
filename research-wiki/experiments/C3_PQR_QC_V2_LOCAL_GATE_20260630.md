# C3 PQR Sparse/Irregular-Aware QC V2 Local Gate - 2026-06-30

## Verdict

`PASS_LOCAL_IMPLEMENTATION_AND_SUBAGENT_FINAL_REVIEW_ONLY`.

This is a local/precheck candidate for `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`. It is not a full training result, not an official mAP result, and not a paper/deploy claim.

## Purpose

PQR V1 diagnostics showed weak score-IoU/rank calibration and proposal-tail overload under sparse/selected-axis conditions. QC V2 adds an optional, default-off sparse/irregular-aware quality calibration path and richer proposal diagnostics so we can distinguish weak proposal learning from postprocess/ranking overload without changing the selector route or claiming a performance fix.

## Changed Surface

- Detector head quality calibration only: `quality_head_cfg.mode="sparse_irregular_qc_v2"`.
- Training target: train-only GT may supervise a `sparse_physical_iou_visibility` quality target using selected-position metadata.
- Test path: no GT, no teacher, no raw prediction cache; diagnostics use model outputs and deploy-visible `metas`.
- Diagnostic fields: `cls_score`, `quality_score`, `selected_segment`, `physical_segment`, selected/physical length, gap, visibility, coverage, endpoint support.
- Input sampling, Adapter internals, ordinary assignment/regression/classification losses, NMS score formula, and evaluator remain unchanged.

## Fail-Closed Boundary

The new config `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck.py` is diagnostic-only:

- `diagnostic_only=True`
- `formal_fulltrain=False`
- `user_override_fulltrain=False`
- `remote_launch_locked=True`
- `official_map_claim=False`
- `claim_map_improvement=False`
- `use_teacher=False`
- `use_test_gt=False`
- `use_raw_prediction_cache=False`

The validator rejects route drift into BH-SDC/DIVERGENT labels and rejects unlocked QC V2 fulltrain configs.

## Verification

- `python -m py_compile ...` passed for changed implementation, tool, validator, and test files.
- `python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck.py` passed.
- `C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest -q tests\test_c3_pqr_rankcal_v1_quality_head.py tests\test_c3_pqr_rankcal_v1_config.py tests\test_c3_pqr_rankcal_proposal_diagnostics.py` returned `37 passed, 8 skipped`.
- Default `python` focused pytest returned `37 passed, 8 skipped, 1 warning`; Windows torch DLL emitted access-violation text but pytest exited 0.
- `git diff --check` exited 0 with LF/CRLF warnings only.
- Final read-only subagent review returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY`, with no blockers.

## Remaining Locks

- No remote sync or remote run has been performed for QC V2 in this gate.
- Remote Linux PRECHECK must rerun real torch/NMS paths because local Windows lacks `nms_1d_cpu` and the default Python torch DLL path is unstable.
- Fulltrain, official mAP claim, paper claim, deploy claim, route-quality judgment, `tools/test.py` official evaluation, and raw-prediction-cache use remain locked.

## Next Action

Allowed next step: remote PRECHECK_ONLY and a bounded 2-iteration smoke, respecting the GPU1-only C3/PQR/CADF mainline training rule. If PRECHECK/smoke pass, a later decision can consider whether QC V2 should move to a diagnostic training run; it still does not unlock PQR formal full training by itself.
