# Research Log

## 2026-06-30 14:20:00 +08:00

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
