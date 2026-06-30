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

## Remote PRECHECK R1

R1 remote PRECHECK at commit `9081d9a2202a` used a fresh clone under `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_9081d9a_20260630_20260630_135857_+0800`. It did not run Slurm, `srun`, `tools/train.py`, `tools/test.py`, or any GPU job. `py_compile`, the QC V2 validator, and `git diff --check` passed, but focused Linux pytest failed one test:

`tests/test_c3_pqr_rankcal_v1_quality_head.py::test_sparse_irregular_qc_v2_returns_optional_deploy_visible_diagnostics`.

Root cause: the test fixture used `reg_pred` as if the last dimension were `[left, right]` per point, but `AnchorFreeHead.get_refined_proposals` consumes regression as `[B, 2, T]` and then permutes it to `[B, T, 2]`. The fixture therefore produced real selected lengths `[1.0, 1.0]` while the test expected `[0.0, 2.0]`. This is a test-fixture bug, not a QC V2 model-logic bug.

Local fix: encode the intended offsets as left `[0, 1]`, right `[0, 1]`, and add a direct `selected_segments == [[0, 0], [0, 2]]` assertion.

## Remote PRECHECK R2

R2 remote PRECHECK at commit `43ef497d05785fc845f98bd0e7f71f65bfffcd25` used fresh clone `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_43ef497_20260630_20260630_140649_+0800`. It again did not run Slurm, `srun`, `tools/train.py`, `tools/test.py`, or any GPU job. `py_compile`, the QC V2 validator, and `git diff --check` passed, but focused Linux pytest still failed one test:

`tests/test_c3_pqr_rankcal_v1_quality_head.py::test_sparse_irregular_qc_v2_returns_optional_deploy_visible_diagnostics`.

R2 failure detail: `selected_segments` was `[[0, 1], [1, 2]]`, while the test expected `[[0, 0], [0, 2]]`.

Root cause: the first local fixture fix was accidentally applied to the earlier `test_quality_score_fusion_uses_low_alpha_model_score_only` fixture, not to the QC V2 diagnostics fixture. The QC V2 diagnostics fixture still contained the old regression layout at commit `43ef497`.

Local fix after R2: restore the quality-fusion fixture to its original values, update the QC V2 diagnostics fixture to left `[0, 1]`, right `[0, 1]`, and keep the direct selected-segment assertion. Local verification returned `torch_1` focused pytest `37 passed, 8 skipped`, QC V2 validator PASS, py_compile PASS, and `git diff --check` PASS.

## Remote PRECHECK R3

R3 remote PRECHECK at commit `f827c50538c6cda68756c3d37cd11a53c5b3e314` used fresh clone `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_f827c50_20260630_20260630_141515_+0800`.

Log path:

`/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_f827c50_20260630_20260630_141515_+0800/logs/c3_pqr_qcv2_precheck_f827c50_20260630_141515_+0800.log`

Evidence:

- `git clone`: PASS.
- `git checkout f827c50538c6cda68756c3d37cd11a53c5b3e314`: PASS.
- `data` and `pretrained` symlinks: PASS.
- `python -m py_compile ...`: PASS.
- `python tools/validate_c3_pqr_rankcal_v1_config.py ...qc_v2_precheck.py`: PASS, printed `PASS_C3_PQR_RANKCAL_V1_CONFIG`.
- Focused Linux pytest: PASS, `45 passed in 27.89s`.
- `git diff --check`: PASS.

Boundary: R3 did not run Slurm, `srun`, `tools/train.py`, `tools/test.py`, or any GPU job. It did not touch the parent hold, BH-SDC, or any divergent route.

## Resource Ownership Rule

From `2026-06-30 14:19:26 +08:00`, all future C3/PQR/CADF mainline training, diagnostic training, short smoke, and bounded `tools/train.py` runs must use GPU1 only. GPU0 is reserved for divergent innovation and must not be used as a fallback. If GPU1 is busy, the allowed action is to wait, queue, or report the blocker.

## Updated Next Action

QC V2 now has a passing remote PRECHECK_ONLY gate. This still does not unlock full training, official evaluation, paper/deploy claims, or route-quality judgment. The next possible execution step is a bounded smoke/diagnostic run on GPU1 only after confirming GPU1 is free or route-owned.

## 2026-06-30 Execution Status

Read-only remote monitoring at `2026-06-30 14:32:56 +08:00` found GPU1 occupied by CADF formal child `1118197.433 cadf_formal_g1`, still running with finite loss and no validation result. GPU0 is occupied by `1118197.467 bvr_twb_g0_r4` and is reserved for divergent innovation. Therefore QC V2 bounded smoke/diagnostic is queued, not started. The next action remains GPU1-only after CADF formal completes, fails, or otherwise releases GPU1.

## 2026-06-30 14:42:58 +08:00 QC V2 Shortdiag Hold-Child Local Implementation

Purpose: add the missing GPU1-only bounded diagnostic launch path that can
generate `result_detection.json` and QC V2 proposal diagnostics after GPU1 is
free. This is a short diagnostic for fields and mechanism evidence, not final
route-quality evidence.

Changed files:

- `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_shortdiag.py`
- `scripts/run_c3_pqr_qc_v2_shortdiag_hold_child.sh`
- `tests/test_c3_pqr_rankcal_v1_config.py`
- `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`
- `research-wiki/log.md`
- `research-wiki/experiments/C3_PQR_QC_V2_LOCAL_GATE_20260630.md`

Config contract:

- `diagnostic_only=True`, `formal_fulltrain=False`,
  `official_map_claim=False`, `claim_map_improvement=False`.
- `use_teacher=False`, `use_test_gt=False`,
  `use_raw_prediction_cache=False`.
- `quality_head_cfg.mode="sparse_irregular_qc_v2"` and
  `diagnostic_dump=True`.
- `post_processing.save_dict=True` and
  `post_processing.qc_v2_diagnostic_dump=True`.
- `workflow.end_epoch=2`, `val_eval_interval=1`, `val_start_epoch=0`,
  `max_train_iters=None`, and `disable_checkpoint=True`, so evaluation/dump is
  not skipped by the smoke-iteration gate.

Launcher contract:

- Default `PQR_ROOT` points at the existing PQR remote clean clone path and can
  be overridden by environment.
- Runtime config is written under a unique `logs/<run_name>/` directory, and
  `work_dir` points at a unique run directory.
- It prints BEGIN/config/runtime_config/run_dir/CUDA visibility metadata.
- It fails closed unless `CUDA_VISIBLE_DEVICES=1`, printing
  `PQR_QCV2_GPU_GUARD_FAIL`; there is no GPU0 fallback.
- It runs `py_compile`, the PQR config validator, `tools/train.py`, and then
  `tools/analyze_c3_pqr_rankcal_proposals.py`.
- If `result_detection.json` is missing after training returns, it writes
  `pqr_qc_v2_proposal_diagnostic_missing.json` through the analyzer path and
  exits non-zero.
- Main-process review found and fixed one launcher issue: OpenTAD single-GPU
  distributed evaluation can write `result_detection.json` under nested
  `gpu*_id*/` directories, so the launcher now discovers exactly one nested
  result path before declaring the artifact missing. Ambiguous or absent result
  artifacts remain fail-closed.

Local verification:

- TDD RED: focused pytest first failed because the shortdiag config and launcher
  were missing.
- TDD RED: launcher contract test then failed on the nested runtime-config
  `_base_` path, which was fixed to `../../$CONFIG_REL`.
- `python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_shortdiag.py`
  printed `PASS_C3_PQR_RANKCAL_V1_CONFIG`.
- `python -m py_compile tools\validate_c3_pqr_rankcal_v1_config.py tools\analyze_c3_pqr_rankcal_proposals.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_shortdiag.py`
  passed.
- Local `bash -n scripts/run_c3_pqr_qc_v2_shortdiag_hold_child.sh` is not
  counted as evidence because Windows resolved `bash` to a WSL shim and path
  parsing failed. Linux syntax/execution remains a remote PRECHECK or bounded
  diagnostic responsibility.
- `python -m pytest -q tests/test_c3_pqr_rankcal_v1_config.py` returned
  `28 passed, 1 warning`.
- `python -m pytest -q tests/test_c3_pqr_rankcal_v1_config.py tests/test_c3_pqr_rankcal_v1_quality_head.py tests/test_c3_pqr_rankcal_proposal_diagnostics.py`
  returned `41 passed, 8 skipped, 1 warning` despite repeated local Windows
  torch DLL fatal-exception prints; pytest exited 0, and Linux remote PRECHECK
  remains authoritative for torch/NMS-backed behavior.
- `git diff --check` exited 0 with LF/CRLF warnings only.

Boundary:

- No SSH, remote sync, Slurm, `srun`, `tools/train.py` real run,
  `tools/test.py`, checkpoint, result JSON, mAP evidence, or route-quality
  judgment was produced by this local implementation.
- No CADF, BH-SDC, DIVERGENT, evaluator, GT/teacher/cache, postprocess
  scoring/NMS, Adapter internals, or input sampler files were edited.
- Main-process review/commit/sync remains pending before any GPU1 diagnostic
  launch.

## 2026-06-30 14:52:34 +08:00 User-Reaffirmed GPU1-Only Training Rule

The user explicitly reaffirmed that this C3/PQR/CADF mainline must continue to
train only on GPU1, because GPU0 is reserved for the innovation/divergent
experiment owner. This is now treated as a hard route-ownership rule:

- all C3/PQR/CADF mainline training, diagnostic training, short smoke, bounded
  `tools/train.py`, and future Slurm child launches must use GPU1 only;
- GPU0 must not be used as a fallback, even if GPU1 is busy;
- if GPU1 is occupied, the allowed action is to wait, queue, or report the
  blocker;
- the QC V2 launcher already implements this by requiring
  `CUDA_VISIBLE_DEVICES=1` and failing closed with
  `PQR_QCV2_GPU_GUARD_FAIL`.
