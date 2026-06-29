# C3 PQR RankCal V1 Pro Gate Summary - 2026-06-30

Timestamp: 2026-06-30 03:38:34 +08:00.

Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.

Branch: `codex/c3-pqr-rankcal-v1-20260629`.

Implementation commit reviewed by Pro:
`dca62cc24c0bef50e53135709820b6a66afe1422`
(`C3_PQR_RankCalV1 PRECHECK_ONLY implementation evidence`).

GitHub URLs:

- Branch: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/c3-pqr-rankcal-v1-20260629
- Commit: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/commit/dca62cc24c0bef50e53135709820b6a66afe1422

Transport:

- Rosetta first attempt: `rosetta probe --port 9333 --host 127.0.0.1`
- Rosetta result: `INVALID_TRANSPORT`, `ECONNREFUSED 127.0.0.1:9333`
- Oracle fallback: browser engine, model `gpt-5.5-pro`, slug `c3-pqr-rankcal-v1-pro`
- Oracle model evidence: requested `Pro`, resolved `Pro extended`, verified `yes`
- Oracle elapsed/model footer: `8m56s`, `gpt-5.5-pro[browser]`
- Oracle exitcode: `0`

Artifacts:

- Prompt: `research-wiki/experiments/c3_pqr_rankcal_v1_pro_gate_20260630_prompt.md`
- Pro output: `research-wiki/experiments/c3_pqr_rankcal_v1_pro_gate_20260630_oracle_output.md`
- Oracle stdout: `logs/c3_pqr_rankcal_v1_pro_gate_20260630_oracle_stdout.log`
- Oracle stderr: `logs/c3_pqr_rankcal_v1_pro_gate_20260630_oracle_stderr.log`
- Oracle exitcode: `logs/c3_pqr_rankcal_v1_pro_gate_20260630_oracle_exitcode.log`

Valid Pro Verdict:

`FIX_BEFORE_RUNTIME`.

Pro accepted that the PQR branch is detector-head ranking calibration, not CADF
selector/input sampler, and did not find a route-drift or leakage blocker. Pro
did find a launch-gate blocker: `workflow.max_train_iters=2` is present in the
precheck config and tests, but the standard `tools/train.py` /
`train_one_epoch` path does not consume it. Therefore the current branch cannot
reliably enforce an exact 2-iteration runtime smoke.

Allowed next action:

- Code fix only, inside the PQR route-owned worktree, to make the 2-iteration
  smoke limit a real consumed runtime contract.
- Required before any runtime smoke: add/verify consumption of
  `workflow.max_train_iters` in the standard launcher or a dedicated smoke
  harness, and add a test proving that the precheck limit is consumed.

Still locked:

- 2-iter runtime smoke: locked until the runtime-iteration gate is fixed.
- 8-epoch short diagnostic: locked until a valid 2-iter smoke fix and passing
  smoke evidence exist.
- Formal/full train: locked.
- `tools/test.py`, mAP claim, checkpoint claim, paper/deploy claim: locked.

Attribution boundary:

PQR future evidence must be attributed only to sparse head/ranking calibration.
It must not be attributed to CADF selector, CADF input sampling, BH-SDC, dynamic
budgeting, physical-time post-processing, evaluator changes, or hidden
raw-prediction/teacher/test-GT shortcuts.
