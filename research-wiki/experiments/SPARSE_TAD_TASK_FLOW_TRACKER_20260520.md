# Sparse TAD Task Flow Tracker - C3 PQR Worktree Snapshot

This route-owned tracker snapshot records only the PQR branch state in
`OpenTAD_C3PQRRankCal_Worktree_20260629`. The shared repository tracker remains
the broader source for cross-route orchestration.

## All Required Experiments and Current Status

| Time (+08:00) | Experiment / Config | Changed Surface | Status | Review / Gate State | Deployment / Result State | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
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

- No runtime smoke is allowed until the iteration-limit gate is fixed.
- No 8-epoch short diagnostic is allowed until a fixed 2-iter runtime smoke passes.
- Formal/full training remains locked.
- No `tools/test.py`, no metric claim, no official result claim, no paper/deploy claim.
- N16R4 long-held parent allocation `1118197 pcot_dbg2g` was not touched.
