# Sparse TAD Task Flow Tracker - C3 PQR Worktree Snapshot

This route-owned tracker snapshot records only the PQR branch state in
`OpenTAD_C3PQRRankCal_Worktree_20260629`. The shared repository tracker remains
the broader source for cross-route orchestration.

## All Required Experiments and Current Status

| Time (+08:00) | Experiment / Config | Changed Surface | Status | Review / Gate State | Deployment / Result State | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-30 04:28:44 | `C3_PQR_RankCalV1_MaxIoU` / pseudo-boundary clean-clone dependency fix | Added missing transform helper dependency `opentad/datasets/transforms/pseudo_boundary.py`; added PQR validator/test dependency guard; updated PQR config metadata only; no PQR scoring math, CADF selector, BH-SDC, sampler, evaluator, postprocess, launcher, or runtime-gate behavior change | Remote clean clone 2-iter smoke import blocker addressed locally; no remote/runtime training launched | Local RED/GREEN dependency test; py_compile PASS; focused PQR pytest `17 passed, 3 skipped` with Windows torch-backed tests skipped due local torch DLL failure; 3 validators PASS; pseudo-boundary pure module behavior `5 passed`; review tool attempts failed and are not counted as PASS | No deployment, no SSH, no Slurm, no `tools/train.py` real run, no `tools/test.py`, no mAP evidence; N16R4 untouched | Commit/push this dependency completeness fix, then allow separate remote agent to rerun Linux PRECHECK and 2-iter smoke; keep 8-epoch shortdiag/formal training/eval/claims locked |
| 2026-06-30 04:08:38 | `C3_PQR_RankCalV1_MaxIoU` / clean-clone dependency completeness fix | Added missing backbone dependency module `opentad/models/backbones/time_aligned_rasterizer.py` only; no PQR scoring math, CADF selector, BH-SDC, sampler, evaluator, postprocess, or launch logic change | Remote clean clone PRECHECK import blocker addressed locally; no remote/runtime training launched | Local py_compile PASS; focused PQR pytest `16 passed, 3 skipped` with Windows torch-backed tests skipped due local torch DLL failure; 3 validators PASS; Linux clean clone must rerun failed torch-backed PRECHECK | No deployment, no SSH, no Slurm, no `tools/train.py` real run, no `tools/test.py`, no mAP evidence; N16R4 untouched | Commit/push this dependency completeness fix, then allow separate remote PRECHECK agent to rerun PRECHECK; keep smoke/training/eval locked here |
| 2026-06-30 03:48:16 | `C3_PQR_RankCalV1_MaxIoU` / `workflow.max_train_iters` runtime gate | Standard local training launcher and train loop only; PQR validator/tests/docs; no CADF selector, no BH-SDC, no evaluator/postprocess change | Pro blocker fixed at local implementation layer; no remote/runtime training launched | Local focused runtime gate tests PASS (`3 passed`); full focused PQR pytest PASS (`16 passed, 3 skipped`); py_compile PASS; 3 validators PASS; still needs read-only subagent final review | No deployment, no Slurm, no SSH, no `tools/train.py` real run, no `tools/test.py`, no mAP evidence; N16R4 `1118197 pcot_dbg2g` untouched | Run required read-only subagent final review, then remote PRECHECK/2-iter smoke only if review passes; keep 8-epoch shortdiag and formal/full train locked |
| 2026-06-30 03:38:34 | `C3_PQR_RankCalV1_MaxIoU` / `c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py` | Detector-head quality/ranking calibration only; no CADF selector, no BH-SDC, no evaluator/postprocess change | GitHub-synced PRECHECK_ONLY evidence; Pro gate valid | Remote PRECHECK_ONLY evidence: py_compile PASS, 3 validators PASS, focused pytest `16 passed in 15.44s`; GPT-5.5 Pro verdict `FIX_BEFORE_RUNTIME` | Branch pushed to GitHub at `dca62cc24c0bef50e53135709820b6a66afe1422`; no training launched; no mAP/runtime/deploy/paper claim | Fix runtime gate so `workflow.max_train_iters=2` is consumed before any 2-iter smoke; keep 8-epoch shortdiag and formal/full train locked |

## 2026-06-30 GitHub Sync And Pro Gate

- Branch: `codex/c3-pqr-rankcal-v1-20260629`
- Pushed remote: `pcot-yuzbo/codex/c3-pqr-rankcal-v1-20260629`
- Implementation commit: `dca62cc24c0bef50e53135709820b6a66afe1422`
- Branch URL: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/c3-pqr-rankcal-v1-20260629
- Commit URL: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/commit/dca62cc24c0bef50e53135709820b6a66afe1422
- Strict random-fixed 50% contract: preserved for the main PQR Adapter backend configs; exact-uniform file is explicitly a stride-2 uniform backend control.
- GT/teacher leakage risk: validator requires `use_teacher=False`, `use_test_gt=False`, raw prediction cache disabled, and no physical-time postprocess claim.
- Current mAP evidence: none. PRECHECK_ONLY only.
- Pro prompt: `research-wiki/experiments/c3_pqr_rankcal_v1_pro_gate_20260630_prompt.md`
- Pro output: `research-wiki/experiments/c3_pqr_rankcal_v1_pro_gate_20260630_oracle_output.md`
- Pro transport: Rosetta 9333 invalid (`ECONNREFUSED`), Oracle browser valid (`gpt-5.5-pro`, model resolved/verified as Pro extended, exitcode 0).
- Pro verdict: `FIX_BEFORE_RUNTIME`.

## Current Locks

- This local owner must not start remote PRECHECK, smoke, SSH, Slurm, training,
  or evaluation.
- A separate remote agent may rerun Linux PRECHECK and then the 2-iteration
  runtime smoke after this commit is pushed.
- No 8-epoch short diagnostic is allowed until the rerun 2-iteration runtime
  smoke passes.
- Formal/full training remains locked.
- No `tools/test.py`, no metric claim, no official result claim, no paper/deploy claim.
- N16R4 long-held parent allocation `1118197 pcot_dbg2g` was not touched.

## 2026-06-30 Pseudo-Boundary Clean Clone Dependency Fix

- Evidence report:
  `research-wiki/experiments/c3_pqr_rankcal_v1_pseudo_boundary_dependency_fix_20260630.md`
- Remote failed evidence: clean clone
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQRRankCal_Precheck_20260630/github_clean_c3_pqr_rankcal_v1`
  at branch HEAD `8cb6b64f83c1b9e86d887978accf1cabfe6f7d34` passed Linux
  PRECHECK but failed 2-iter runtime smoke before the train loop with
  `ModuleNotFoundError: No module named
  'opentad.datasets.transforms.pseudo_boundary'`.
- Root cause: `opentad/datasets/transforms/end_to_end.py` hard-imports the real
  pseudo-boundary transform helper, but the module file was absent from this PQR
  branch/clean clone.
- Changed files: `opentad/datasets/transforms/pseudo_boundary.py`,
  `tools/validate_c3_pqr_rankcal_v1_config.py`,
  `tests/test_c3_pqr_rankcal_v1_config.py`, and the three PQR config metadata
  files.
- Changed surface: clean-clone dependency restoration and metadata guard only.
  No CADF selector, BH-SDC, evaluator, postprocess, sampler, PQR scoring math,
  quality-head math, training launcher, or runtime gate behavior was changed.
- Strict random-fixed 50% contract: unchanged.
- GT/teacher leakage risk: unchanged; pseudo-boundary cache loader rejects
  `uses_gt=True` manifests.
- Local verification: RED/GREEN dependency test; py_compile PASS; focused PQR
  pytest `17 passed, 3 skipped`; three validators PASS; pseudo-boundary pure
  module behavior `5 passed`.
- Review gate status: attempted but incomplete due tool failures; no PASS review
  claimed.
- Next launch decision: this owner does not launch remote PRECHECK, smoke, or
  training. A separate remote agent may rerun Linux PRECHECK and 2-iter smoke
  after this commit is pushed.

## 2026-06-30 Clean Clone Dependency Completeness Fix

- Remote failed evidence: clean clone
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQRRankCal_Precheck_20260630/github_clean_c3_pqr_rankcal_v1`
  at branch HEAD `30ea4f1a4970416ad744a2d5429b8b37fddb547d` failed focused
  PRECHECK pytest with `ModuleNotFoundError: No module named
  'opentad.models.backbones.time_aligned_rasterizer'` from
  `opentad/models/backbones/vit_adapter.py:18`.
- Root cause: `vit_adapter.py` has a real hard dependency on
  `TimeAlignedRasterizer`, but the module file was absent from this PQR branch.
  This was a clean-clone dependency completeness gap, not a PQR ranking/scoring
  bug.
- Changed file: `opentad/models/backbones/time_aligned_rasterizer.py` added
  from the existing mainline local implementation used by Adapter TARA paths.
- Changed surface: backbone dependency restoration only. No CADF selector,
  BH-SDC, evaluator, postprocess, sampler, PQR scoring math, configs, or
  launcher behavior was changed.
- Strict random-fixed 50% contract: unchanged.
- GT/teacher leakage risk: unchanged; no data path, teacher/cache, or GT
  access code was touched.
- Local verification: py_compile PASS; focused PQR pytest PASS with
  `16 passed, 3 skipped`; three validators printed
  `PASS_C3_PQR_RANKCAL_V1_CONFIG`.
- Local limitation: Windows torch DLL import still skips the three torch-backed
  quality-head tests locally, so Linux clean clone PRECHECK must rerun them.
- Next launch decision: this owner does not launch remote PRECHECK, smoke, or
  training. A separate remote PRECHECK agent may rerun PRECHECK after this
  commit is pushed.

## 2026-06-30 Runtime Gate Fix

- Evidence report: `research-wiki/experiments/c3_pqr_rankcal_v1_runtime_gate_fix_20260630.md`
- Changed files: `tools/train.py`, `opentad/cores/train_engine.py`,
  `tools/validate_c3_pqr_rankcal_v1_config.py`,
  `tests/test_c3_pqr_rankcal_v1_config.py`, and PQR docs.
- Strict random-fixed 50% contract: unchanged. The runtime gate only caps train
  iterations for smoke/precheck use.
- GT/teacher leakage risk: unchanged; validator still requires teacher/test-GT
  disabled and raw-prediction cache disabled.
- Current mAP evidence: none.
- Next launch decision: still locked pending subagent final review and remote
  PRECHECK/2-iter smoke. No remote action was performed in this fix.

## 2026-06-30 13:43:26 +08:00 Sparse/Irregular-Aware QC V2 Local Implementation

- Experiment/config: `c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck.py`.
- Changed surface: detector head sparse/irregular quality calibration V2, optional proposal diagnostic dump, PQR validator, focused tests, and local route records. Input sampling, Adapter internals, loss assignment outside the quality target, and post-processing score/NMS logic remain unchanged.
- Changed files: `opentad/models/dense_heads/anchor_free_head.py`, `opentad/models/detectors/actionformer.py`, `opentad/models/detectors/single_stage.py`, `opentad/cores/test_engine.py`, `tools/analyze_c3_pqr_rankcal_proposals.py`, `tools/validate_c3_pqr_rankcal_v1_config.py`, new QC V2 precheck config, three focused test files, this tracker, and `research-wiki/log.md`.
- Strict random-fixed 50% contract: preserved; QC V2 config inherits the PQR random-fixed Adapter 50% backend and validator still checks train/val/test loaders.
- GT/teacher leakage risk: test-time GT/teacher/cache paths are rejected; QC V2 test diagnostics use only deploy-visible `metas` geometry plus model predictions. Training quality target may use training GT with selected-position metadata.
- Current mAP evidence: none. This is local/precheck candidate code only; no `tools/train.py`, `tools/test.py`, remote sync, Slurm, checkpoint, or result claim was run.
- Local verification: py_compile PASS; QC V2 validator PASS; default Python focused pytest `37 passed, 8 skipped, 1 warning`; `torch_1` focused pytest `37 passed, 8 skipped`; `git diff --check` exit 0 with CRLF warnings only. The extra focused test covers `test_engine` NMS-result diagnostic-field preservation when the local NMS extension is available.
- Review gate: read-only subagent final review returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY` with no blocking findings. Non-blocking findings were: test-time GT/teacher/cache assertion is a safety behavior change for illegal test calls, Linux precheck should rerun real torch/NMS paths, and the new `gather_ddp_results` extras coverage was desirable; the last item has been covered locally.
- Local limitation: Windows default Python torch DLL initialization emits access-violation text and skips torch-backed tests; `torch_1` skips OpenTAD import paths that require the unavailable local `nms_1d_cpu` extension. Linux/remote PRECHECK must rerun these paths before any Slurm child.
- Next launch decision: remain locked for fulltrain. Allowed next action is remote PRECHECK_ONLY / bounded 2-iter smoke on GPU1 or CPU-login checks as appropriate, with no official mAP, checkpoint, paper/deploy claim, or route-quality judgment. PQR formal full training remains locked until diagnostic gates explicitly unlock it.

## 2026-06-30 14:03:22 +08:00 Sparse/Irregular-Aware QC V2 Remote PRECHECK R1 Failure And Test-Fixture Fix

- Remote PRECHECK R1 at commit `9081d9a2202a` used fresh clone `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_9081d9a_20260630_20260630_135857_+0800`; no Slurm, `srun`, `tools/train.py`, `tools/test.py`, GPU use, parent-hold action, BH-SDC action, or DIVERGENT route action occurred.
- R1 evidence: clone/checkout/resource symlinks succeeded; Python `3.10.20`; py_compile PASS; QC V2 validator PASS; `git diff --check` PASS; focused pytest retried with `/tmp` tempdir and returned `44 passed, 1 failed`, no skipped tests.
- Failure: `test_sparse_irregular_qc_v2_returns_optional_deploy_visible_diagnostics` expected selected lengths `[0.0, 2.0]`, but Linux produced `[1.0, 1.0]`.
- Root cause: the test fixture wrote `reg_pred` with the wrong layout. `AnchorFreeHead.get_refined_proposals` consumes regression as `[B, 2, T]`; the fixture intended left/right offsets per point but encoded left `[0, 0]`, right `[1, 1]`. Model code behaved correctly.
- Local fix: update the fixture to left `[0, 1]`, right `[0, 1]`, and assert the intended selected segments directly. Local `torch_1` focused pytest returned `37 passed, 8 skipped`; QC V2 validator PASS; py_compile PASS; `git diff --check` PASS.
- Next launch decision: commit/push the test-fixture fix and rerun remote PRECHECK_ONLY in a fresh clone. Fulltrain, official eval, paper/deploy claim, and route-quality judgment remain locked.

## 2026-06-30 14:10:00 +08:00 Sparse/Irregular-Aware QC V2 Remote PRECHECK R2 Failure And Correct Fixture Fix

- Remote PRECHECK R2 at commit `43ef497d05785fc845f98bd0e7f71f65bfffcd25` used fresh clone `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_43ef497_20260630_20260630_140649_+0800`; no Slurm, `srun`, `tools/train.py`, `tools/test.py`, GPU use, parent-hold action, BH-SDC action, or DIVERGENT route action occurred.
- R2 evidence: clone/checkout/resource symlinks succeeded; py_compile PASS; QC V2 validator PASS after one operator retry with the config argument; `git diff --check` PASS; focused pytest retried with `/tmp` tempdir and returned `44 passed, 1 failed`, no skipped tests.
- Failure: the same QC V2 diagnostics test now showed selected segments `[[0, 1], [1, 2]]` while the test expected `[[0, 0], [0, 2]]`.
- Root cause: the first local fix accidentally patched the earlier quality-fusion fixture, not the QC V2 diagnostics fixture. The model code was still behaving correctly.
- Local fix: restore the quality-fusion fixture and patch the QC V2 diagnostics fixture exactly. Local `torch_1` focused pytest returned `37 passed, 8 skipped`; QC V2 validator PASS; py_compile PASS; `git diff --check` PASS.
- Next launch decision: commit/push the corrected fixture fix and rerun remote PRECHECK_ONLY R3 in a fresh clone. Fulltrain, official eval, paper/deploy claim, and route-quality judgment remain locked.

## 2026-06-30 14:19:26 +08:00 C3/PQR/CADF GPU Ownership Rule And QC V2 Remote PRECHECK R3 PASS

- Resource ownership rule: all future C3/PQR/CADF mainline training, diagnostic
  training, short smoke, or bounded `tools/train.py` runs must use GPU1 only.
  GPU0 is reserved for the divergent-innovation owner and must not be used as a
  fallback. If GPU1 is busy, wait, queue, or report the blocker instead of
  occupying GPU0. CPU-only/login-node static PRECHECK remains allowed when it
  does not launch training or use GPU.
- Route boundary: this applies to `C3_MAINLINE_OPTIMIZATION` /
  `C3_ORIGINAL_OPTIMIZATION_ROUTE` work. Do not modify or cancel BH-SDC or any
  `DIVERGENT_INNOVATION_*` route, and do not release/cancel/replace protected
  parent hold `1118197 pcot_dbg2g`.
- Remote PRECHECK R3 at commit
  `f827c50538c6cda68756c3d37cd11a53c5b3e314` used fresh clone
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_f827c50_20260630_20260630_141515_+0800`.
- R3 log:
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_f827c50_20260630_20260630_141515_+0800/logs/c3_pqr_qcv2_precheck_f827c50_20260630_141515_+0800.log`.
- R3 evidence: `git clone` PASS; checkout `f827c50538c6cda68756c3d37cd11a53c5b3e314`
  PASS; `data` and `pretrained` symlinks PASS; py_compile PASS; QC V2 validator
  printed `PASS_C3_PQR_RANKCAL_V1_CONFIG`; focused Linux pytest returned
  `45 passed in 27.89s`; `git diff --check` PASS.
- R3 boundary: no Slurm, `srun`, `tools/train.py`, `tools/test.py`, GPU use,
  parent-hold action, BH-SDC action, or DIVERGENT route action occurred.
- Current mAP evidence: none. This is a PRECHECK_ONLY result, not a metric result
  and not a route-quality judgment.
- Next launch decision: QC V2 remote PRECHECK gate is passed. The next allowed
  step is a bounded smoke/diagnostic only on GPU1 after checking GPU1 ownership
  and current child-run status. Fulltrain, official eval, paper/deploy claim, and
  route-quality judgment remain locked until the required diagnostic gates unlock
  them explicitly.
