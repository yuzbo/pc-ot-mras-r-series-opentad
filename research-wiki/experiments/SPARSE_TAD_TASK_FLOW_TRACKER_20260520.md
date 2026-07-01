# Sparse TAD Task Flow Tracker

Last updated: 2026-07-01 15:34:10 +08:00 Asia/Shanghai

This owned-worktree tracker snapshot records only the MDL-Knot handoff speedfix and interval allowlist repair stage for `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.

## All Required Experiments and Current Status

| Experiment / config | Changed surface | Current status | Review / gate state | Deployment / result state | Next action |
| --- | --- | --- | --- | --- | --- |
| MDL-Knot handoff speedfix GitHub evidence sync, commit `493ed2a` | GitHub evidence synchronization only; no code/config behavior change beyond already-committed validator/formal-readiness allowlist repair. | GitHub branch pushed successfully. | Local tests and validator evidence remain as recorded for commit `493ed2a`; no new Pro gate, remote precheck, Slurm, or formal-readiness unlock occurred. | Branch `codex/divergent-mdl-knot-handoff-speedfix-493ed2a-20260701`, URL `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-mdl-knot-handoff-speedfix-493ed2a-20260701`. No N16R4 remote sync/rerun yet; no mAP, runtime/FLOPs, deploy, paper, or sparse-compute claim. | Use this branch for a bounded remote `SHORT_DIAGNOSTIC_ONLY` resync/rerun only. Formal/full train remains locked. |
| MDL-Knot handoff speedfix, `input_mdl_knot_dynamic_adapter_irregular_headv3.py` and `input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py` | Input sampling/acquisition policy, route-local sparse decode handoff, diagnostics, validator/test evidence. No Adapter internals, detector head, losses/assignment, token compression, or test-time post-processing change. | `PRECHECK/SHORT_DIAGNOSTIC_ONLY`; local code and focused tests pass. | Local validator passes. `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` without train log. Synthetic allowed-interval train log passes as `SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED`. Formal full train remains locked. | No remote sync, Slurm, training, evaluation, `tools/test.py`, mAP, runtime/FLOPs, deploy, paper, or sparse-compute claim. One GitHub push attempt to `pcot-yuzbo` failed with port 443 connectivity timeout; no retry. | Keep local owned-worktree commit as source of truth until network push is available. Any formal/full train launch requires separate gate/review/user authorization. |

## Timeline

- 2026-07-01 11:36:55 +08:00 Asia/Shanghai: commit `1d40254` implemented MDL-Knot sparse handoff speedfix with `MDLKnotDecordDecode`, sample-local raw scout probe cache, profile split, and launch/precheck gate updates. Strict route boundary: no C3/BVR/RBA/ABR/combo mixing. No GT/teacher/prediction-cache/evaluator/post-processing shortcut introduced. Formal full training locked.
- 2026-07-01 14:37:21 +08:00 Asia/Shanghai: repaired shortdiag validator and formal-readiness log revalidation `INTERVAL` false-positive risk. Allowed only schedule-style `checkpoint/eval/evaluation/val_eval/val_loss/logging interval` text before route-drift and eval/checkpoint claim scanning. Real `INTERVAL selector route drift` remains rejected.
- 2026-07-01 14:37:21 +08:00 Asia/Shanghai: one `git push pcot-yuzbo codex/divergent-mdl-knot-handoff-speedfix-20260701` attempt failed with GitHub port 443 connectivity timeout. No retry was made.
- 2026-07-01 15:34:10 +08:00 Asia/Shanghai: retry GitHub sync succeeded. Commit `493ed2a850e15c67cd0be1dfffe9a1792c239065` is available at `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-mdl-knot-handoff-speedfix-493ed2a-20260701`. This is evidence sync only; N16R4 resync/rerun and all claims remain locked.

## Current Evidence

- `python -m py_compile opentad/acquisition/mdl_knot/diagnostics.py tools/mdl_knot/validate_mdl_knot_shortdiag.py tests/test_mdl_knot_shortdiag.py`: pass, exit code `0`.
- `python -m pytest tests/test_mdl_knot_shortdiag.py -q`: pass, `17 passed in 3.93s`.
- `python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py`: pass, `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`.
- `python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_shortdiag.py -q`: pass, `47 passed, 2 skipped in 24.75s`.
- `python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py --train-log <temp allowed interval log>`: pass, `SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED`.

## Locks

- `formal_train_unlocked=false`.
- `full_train_unlocked=false`.
- Remote sync to N16R4 locked; GitHub evidence sync for commit `493ed2a` is complete.
- Slurm locked.
- Training locked.
- Evaluation and `tools/test.py` locked.
- mAP/runtime/FLOPs/deploy/paper/sparse-compute claims locked.
