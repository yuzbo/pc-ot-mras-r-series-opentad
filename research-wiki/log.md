# Research Log

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
