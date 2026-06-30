# Research Log

## 2026-06-30 14:00:00 +08:00

C3 PQR Sparse/Irregular-Aware QC V2 local gate passed implementation and
read-only review in the route-owned worktree
`OpenTAD_C3PQRRankCal_Worktree_20260629`.

- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.
- Config: `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck.py`.
- Changed surface: default-off detector-head sparse/irregular quality calibration
  V2, proposal diagnostic fields, validator, and focused tests. Input sampling,
  Adapter internals, ordinary detection losses, evaluator, and post-processing
  score/NMS logic are not claimed as changed.
- Protocol: diagnostic-only and fail-closed; no teacher, no test GT, no raw
  prediction cache, no official mAP claim, no fulltrain unlock.
- Verification: py_compile PASS; QC V2 validator PASS; default Python focused
  pytest `37 passed, 8 skipped, 1 warning`; `torch_1` focused pytest
  `37 passed, 8 skipped`; `git diff --check` exit 0 with LF/CRLF warnings
  only.
- Review: read-only subagent final review returned
  `PASS_SUBAGENT_FINAL_REVIEW_ONLY`; no blocking findings. A suggested
  `test_engine` diagnostic extras preservation test was added before this final
  record.
- Limitation: Windows local tests still skip true OpenTAD NMS paths when
  `nms_1d_cpu` is unavailable or the default torch DLL import fails, so Linux
  remote PRECHECK must rerun those paths.
- Next decision: QC V2 remains locked for fulltrain and metric claims. Allowed
  next action is remote PRECHECK_ONLY / bounded smoke on the C3 mainline GPU1
  rule; no `tools/test.py` official eval or long training is unlocked by this
  local gate alone.

## 2026-06-30 14:03:22 +08:00

C3 PQR Sparse/Irregular-Aware QC V2 remote PRECHECK R1 failed on one focused
Linux pytest assertion, and the root cause was identified as a test-fixture
layout bug rather than model logic.

- Remote fresh clone:
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_9081d9a_20260630_20260630_135857_+0800`.
- Commit tested: `9081d9a2202a`.
- Boundary: no Slurm, `srun`, `tools/train.py`, `tools/test.py`, GPU use,
  parent-hold action, BH-SDC action, or DIVERGENT route action occurred.
- Passing parts: clone/checkout/resource symlinks, Python `3.10.20`,
  py_compile, QC V2 validator, and `git diff --check`.
- Failing part: focused pytest retried with `/tmp` tempdir returned
  `44 passed, 1 failed`, no skipped tests. The failed test expected selected
  lengths `[0.0, 2.0]` but the fixture produced `[1.0, 1.0]`.
- Root cause: `AnchorFreeHead.get_refined_proposals` consumes `reg_pred` as
  `[B, 2, T]`, but the test fixture encoded offsets as if the last dimension
  were per-point `[left, right]`. The model decode was correct.
- Local fix: update the fixture to encode intended left `[0, 1]`, right
  `[0, 1]`, and add a direct selected-segment assertion. Local verification
  after the fix: `torch_1` focused pytest `37 passed, 8 skipped`; py_compile
  PASS; QC V2 validator PASS; `git diff --check` PASS.
- Next action: commit/push the test-fixture fix and rerun remote PRECHECK_ONLY
  in a fresh clone. Fulltrain and metric claims remain locked.

## 2026-06-30 14:10:00 +08:00

C3 PQR Sparse/Irregular-Aware QC V2 remote PRECHECK R2 still failed the same
focused Linux pytest because the first fixture fix was applied to the wrong
test block.

- Remote fresh clone:
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_43ef497_20260630_20260630_140649_+0800`.
- Commit tested: `43ef497d05785fc845f98bd0e7f71f65bfffcd25`.
- Boundary: no Slurm, `srun`, `tools/train.py`, `tools/test.py`, GPU use,
  parent-hold action, BH-SDC action, or DIVERGENT route action occurred.
- Passing parts: clone/checkout/resource symlinks, py_compile, QC V2 validator
  after operator retry with config argument, and `git diff --check`.
- Failing part: `/tmp` focused pytest returned `44 passed, 1 failed`, no
  skipped tests. The failed QC V2 diagnostics test saw selected segments
  `[[0, 1], [1, 2]]` while expecting `[[0, 0], [0, 2]]`.
- Root cause: the first local test-fixture fix patched
  `test_quality_score_fusion_uses_low_alpha_model_score_only`, not
  `test_sparse_irregular_qc_v2_returns_optional_deploy_visible_diagnostics`.
- Correct local fix: restore the quality-fusion fixture and patch the QC V2
  diagnostics fixture exactly. Local verification after the correct fix:
  `torch_1` focused pytest `37 passed, 8 skipped`; py_compile PASS; QC V2
  validator PASS; `git diff --check` PASS.
- Next action: commit/push the corrected fixture fix and rerun remote
  PRECHECK_ONLY R3. Fulltrain and metric claims remain locked.

## 2026-06-30 04:28:44 +08:00

C3 PQR RankCal V1 pseudo-boundary clean-clone runtime dependency blocker fixed
locally in the route-owned worktree
`OpenTAD_C3PQRRankCal_Worktree_20260629`.

- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.
- Remote evidence: clean clone
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQRRankCal_Precheck_20260630/github_clean_c3_pqr_rankcal_v1`
  at HEAD `8cb6b64f83c1b9e86d887978accf1cabfe6f7d34` passed Linux PRECHECK
  but failed the 2-iter runtime smoke before the train loop with
  `ModuleNotFoundError: No module named
  'opentad.datasets.transforms.pseudo_boundary'` from
  `opentad.datasets.transforms.end_to_end`.
- Root cause: `end_to_end.py` hard-imports a real transform helper module that
  was missing from the PQR branch/clean clone.
- Fix: added `opentad/datasets/transforms/pseudo_boundary.py`; added a PQR
  validator/test clean-clone dependency guard; updated PQR config metadata so it
  no longer says pseudo_boundary is missing after local restoration.
- Changed surface: transform helper dependency restoration plus validator/test
  and metadata only. No CADF selector, BH-SDC, evaluator, postprocess, sampler,
  PQR scoring math, quality-head math, training launcher, or runtime gate
  behavior changed.
- Strict random-fixed 50% contract: unchanged.
- GT/teacher leakage risk: unchanged; `load_boundary_scores` rejects manifests
  with `uses_gt=True`, and no teacher/test-GT/cache shortcut was added.
- Local verification: RED dependency test failed before the module and passed
  after; py_compile exit 0; focused PQR pytest `17 passed, 3 skipped`; three
  PQR validators all printed `PASS_C3_PQR_RANKCAL_V1_CONFIG`; pseudo-boundary
  pure module behavior tests `5 passed`.
- Local limitation: three torch-backed quality-head tests still skip on Windows
  due local torch DLL import failure; Linux PRECHECK must rerun them.
- Review attempt: Claude review failed twice with no JSON output; GPT-4o and
  MiniMax fallbacks were unavailable due missing API keys. No review PASS is
  claimed.
- No SSH, Slurm, remote PRECHECK, remote smoke, training, evaluation, or remote
  write was performed by this owner.

## 2026-06-30 04:08:38 +08:00

C3 PQR RankCal V1 clean-clone dependency completeness blocker fixed locally in
the route-owned worktree `OpenTAD_C3PQRRankCal_Worktree_20260629`.

- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.
- Remote evidence: clean clone
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQRRankCal_Precheck_20260630/github_clean_c3_pqr_rankcal_v1`
  at HEAD `30ea4f1a4970416ad744a2d5429b8b37fddb547d` failed focused pytest
  with `ModuleNotFoundError` for
  `opentad.models.backbones.time_aligned_rasterizer` from
  `opentad/models/backbones/vit_adapter.py:18`.
- Root cause: `vit_adapter.py` imports a real Adapter TARA dependency, but the
  module file was missing from the PQR branch/clean clone.
- Fix: added `opentad/models/backbones/time_aligned_rasterizer.py` from the
  existing mainline local implementation.
- Changed surface: backbone dependency restoration only; no CADF selector,
  BH-SDC, evaluator, postprocess, sampler, PQR scoring math, configs, or
  launch/runtime control changed.
- Strict random-fixed 50% contract: unchanged.
- GT/teacher leakage risk: unchanged.
- Local verification: py_compile exit 0; focused PQR pytest `16 passed, 3
  skipped`; three PQR validators all printed
  `PASS_C3_PQR_RANKCAL_V1_CONFIG`.
- Local limitation: three torch-backed quality-head tests still skipped on
  Windows due torch DLL import failure; Linux clean clone PRECHECK must rerun
  those tests.
- No SSH, Slurm, remote PRECHECK, smoke, training, evaluation, or remote write
  was performed by this owner.

## 2026-06-30 03:48:16 +08:00

C3 PQR RankCal V1 runtime gate blocker fixed locally in the route-owned
worktree `OpenTAD_C3PQRRankCal_Worktree_20260629`.

- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.
- Fixed Pro blocker: `workflow.max_train_iters` is now consumed by the
  standard local training launcher and `train_one_epoch` runtime path.
- Runtime semantics: unset or `<=0` keeps default full-epoch behavior; `N > 0`
  caps the whole run to at most `N` attempted train iterations per process, then
  exits before checkpoint/validation/evaluation branches.
- Changed surface: `tools/train.py`, `opentad/cores/train_engine.py`, PQR
  validator/tests/docs only.
- Strict random-fixed 50% contract: unchanged.
- GT/teacher leakage risk: unchanged; validator still rejects teacher/test-GT
  and raw-prediction-cache paths.
- Local verification: focused runtime gate pytest `3 passed`; focused PQR pytest
  `16 passed, 3 skipped`; py_compile exit 0; three PQR validators all printed
  `PASS_C3_PQR_RANKCAL_V1_CONFIG`.
- Evidence report:
  `research-wiki/experiments/c3_pqr_rankcal_v1_runtime_gate_fix_20260630.md`.
- Still locked: required read-only subagent final review, remote PRECHECK,
  remote 2-iter smoke, 8-epoch short diagnostic, formal/full train,
  `tools/test.py`, mAP/checkpoint/paper/deploy claims.
- No SSH, Slurm, real `tools/train.py` run, evaluation, remote write, or N16R4
  long-held allocation action was performed. N16R4 `1118197 pcot_dbg2g` was not
  touched.

## 2026-06-30 03:38:34 +08:00

PQR GitHub sync and GPT-5.5 Pro launch-decision gate completed in the route
owned worktree `OpenTAD_C3PQRRankCal_Worktree_20260629`.

- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.
- Branch pushed: `pcot-yuzbo/codex/c3-pqr-rankcal-v1-20260629`.
- Implementation commit: `dca62cc24c0bef50e53135709820b6a66afe1422`.
- GitHub branch: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/c3-pqr-rankcal-v1-20260629
- GitHub commit: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/commit/dca62cc24c0bef50e53135709820b6a66afe1422
- Staged/committed implementation files: three PQR configs, PQR validator, two PQR tests, and implementation report.
- PRECHECK_ONLY evidence carried into Pro prompt: remote py_compile PASS, three validators PASS, focused pytest `16 passed in 15.44s`.
- Pro transport: Rosetta `--port 9333 --host 127.0.0.1` failed with `ECONNREFUSED`; Oracle browser fallback succeeded with `gpt-5.5-pro`, model resolved/verified as Pro extended, exitcode 0.
- Valid Pro verdict: `FIX_BEFORE_RUNTIME`.
- Pro blocker: `workflow.max_train_iters=2` is not consumed by the standard `tools/train.py` / `train_one_epoch` runtime path, so an exact 2-iteration smoke is not currently enforceable.
- Allowed next action: fix the runtime iteration gate and add a test proving it is consumed.
- Still locked: 2-iter smoke until gate fix, 8-epoch short diagnostic, formal/full train, `tools/test.py`, mAP claim, checkpoint claim, paper/deploy claim.
- No SSH, Slurm, training, evaluation, remote write, or N16R4 long-held allocation action was performed.

## 2026-06-30 13:43:26 +08:00

Sparse/Irregular-Aware QC V2 local implementation completed in the route owned
worktree `OpenTAD_C3PQRRankCal_Worktree_20260629`.

- Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.
- Config: `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck.py`.
- Changed surface: default-off head support for `mode="sparse_irregular_qc_v2"`, train-only physical IoU/visibility quality target, deploy-visible proposal diagnostics, fail-closed validator/config gates, and focused tests.
- Leakage boundary: no test-time GT, teacher, raw prediction cache, or official mAP claim; test diagnostics use only model predictions plus selected-coordinate metadata.
- Local verification: py_compile PASS; validator PASS; focused config/diagnostics pytest `36 passed, 1 skipped`; focused quality-head pytest `1 passed, 6 skipped`; `git diff --check` exit 0 with line-ending warnings only.
- Still locked: remote launch, Slurm, `tools/train.py`, `tools/test.py`, fulltrain, official mAP/checkpoint/paper/deploy claims, and Pro/subagent gates. No stage/commit/push was performed.

## 2026-06-30 14:19:26 +08:00

C3/PQR/CADF mainline GPU ownership rule and QC V2 remote PRECHECK R3 result
recorded.

- Resource rule: all future C3/PQR/CADF mainline training, diagnostic training,
  short smoke, and bounded `tools/train.py` runs must use GPU1 only. GPU0 is
  reserved for divergent innovation and must not be used as a fallback; if GPU1
  is busy, wait, queue, or report instead.
- Protected resources: do not modify/cancel BH-SDC or any
  `DIVERGENT_INNOVATION_*` route, and do not release/cancel/replace parent hold
  `1118197 pcot_dbg2g`.
- Remote PRECHECK R3 commit:
  `f827c50538c6cda68756c3d37cd11a53c5b3e314`.
- Fresh remote clone:
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_f827c50_20260630_20260630_141515_+0800`.
- R3 log:
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_f827c50_20260630_20260630_141515_+0800/logs/c3_pqr_qcv2_precheck_f827c50_20260630_141515_+0800.log`.
- Evidence: clone/checkout/resource symlinks PASS; py_compile PASS; QC V2
  validator PASS; focused Linux pytest `45 passed in 27.89s`; `git diff --check`
  PASS.
- Boundary: no Slurm, `srun`, `tools/train.py`, `tools/test.py`, GPU use,
  parent-hold action, BH-SDC action, or DIVERGENT route action occurred.
- Current mAP evidence: none. PRECHECK_ONLY is not a final route result.
- Next decision: QC V2 PRECHECK gate passed; bounded smoke/diagnostic may be
  considered only on GPU1 after current GPU1 run ownership is checked. Fulltrain
  and official metric claims remain locked.

## 2026-06-30 14:32:56 +08:00

PQR backend-control completion and current GPU1 blocker recorded from read-only
subagent evidence.

- Parent hold `1118197 pcot_dbg2g` remains `RUNNING` on `g0030`; it was not
  released, cancelled, or modified.
- GPU0 is occupied by `1118197.467 bvr_twb_g0_r4` and remains reserved for the
  divergent-innovation side. C3/PQR/CADF must not use it as fallback.
- GPU1 is occupied by `1118197.433 cadf_formal_g1`; latest CADF formal evidence
  reached epoch `[005][00040/00099]` around `2026-06-30 14:30:14 +08:00` with
  finite loss and no NaN/OOM/Traceback/RuntimeError/Killed/No space. No
  validation mAP or `Training Over` exists yet.
- PQR backend-control child `1118197.459 pqr_ctrl_g0_r5` completed
  `COMPLETED 0:0`. It is diagnostic-only/short training, not a final route
  result. Evidence: interim Average-mAP `15.15%`, final diagnostic Average-mAP
  `24.95%`, `Training Over`, `result_detection.json`, and diagnostic JSON files.
- Local analysis synthesis: current evidence most strongly supports
  proposal/ranking/cap overload: high-IoU proposals can exist, but classification
  score, quality/rank calibration, and per-video cap behavior are weak. This does
  not prove proposal learning is healthy; it only says the clearest observed
  bottleneck is calibration/overload rather than total proposal absence.
- Next decision: do not start new PQR/QC V2 smoke or training while GPU1 is held
  by CADF formal. When GPU1 is free, the most informative next step is QC V2
  bounded proposal-dump diagnostics on GPU1 only; short diagnostics remain
  non-final unless they reveal NaN, persistent non-finite behavior, OOM, or a
  protocol error.
