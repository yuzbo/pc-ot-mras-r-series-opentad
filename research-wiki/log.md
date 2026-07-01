# Research Log

## 2026-07-01 14:37:21 +08:00 Asia/Shanghai

Route: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_HandoffSpeedFix_Worktree_20260701`

Owned branch: `codex/divergent-mdl-knot-handoff-speedfix-20260701`

Event: MDL-Knot shortdiag validator and formal-readiness interval allowlist repair.

Summary:

- Synchronized the CLI shortdiag validator and `validate_shortdiag_train_log_content()` formal-readiness revalidation.
- Allowed only schedule-style `checkpoint/eval/evaluation/val_eval/val_loss/logging interval` log terms before route-drift and eval/checkpoint claim scanning.
- Kept real `INTERVAL selector route drift` rejected in both validator and formal-readiness revalidation.
- No C3/BVR/RBA/ABR files were modified.
- No remote sync, Slurm, training, evaluation, `tools/test.py`, Pro/Oracle/Rosetta, mAP/runtime/FLOPs/deploy/paper claim, or sparse-compute claim was run or unlocked.

Verification:

- `python -m py_compile opentad/acquisition/mdl_knot/diagnostics.py tools/mdl_knot/validate_mdl_knot_shortdiag.py tests/test_mdl_knot_shortdiag.py` passed with exit code `0`.
- `python -m pytest tests/test_mdl_knot_shortdiag.py -q` passed: `17 passed in 3.93s`.
- `python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py` passed as `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`.
- `python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_shortdiag.py -q` passed: `47 passed, 2 skipped in 24.75s`.
- `python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py --train-log <temp allowed interval log>` passed as `SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED`.
- One `git push pcot-yuzbo codex/divergent-mdl-knot-handoff-speedfix-20260701` attempt failed with GitHub port 443 connectivity timeout. No retry was made.

Next action:

- Keep the local owned-worktree commit as the source of truth until network push is available.
- Formal/full training remains locked.

## 2026-07-01 11:16:42 +08:00 Asia/Shanghai

Route: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_HandoffSpeedFix_Worktree_20260701`

Owned branch: `codex/divergent-mdl-knot-handoff-speedfix-20260701`

Event: local MDL-Knot handoff speed fix implementation and verification.

Summary:

- Preserved `mdl_knot_scout_max_frames=96`; no coarse scout-frame reduction was made.
- Added sample-local raw scout probe frame cache and route-local `MDLKnotDecordDecode`.
- Split profile fields for raw scout decode, selector, structural handoff, validation, and sparse decode.
- Preserved `mdl_knot_profile` in collected metadata and backfilled decoder-side sparse decode fields into
  `mdl_knot_pipeline_diagnostic.profile` for remote short-diagnostic attribution.
- Hardened the short-diagnostic validator so common `*_interval` config keys in train logs do not trigger
  false Interval route-drift failures; real Interval-route drift strings are still rejected.
- Updated MDL config, validator, audit evidence, and focused tests.
- No dense raw backbone handoff was introduced.
- No GT, teacher, prediction cache, evaluator, or post-processing shortcut was introduced.
- No remote sync, Slurm, training, evaluation, `tools/test.py`, Pro/Oracle/Claude/Gemini, commit, push, mAP/runtime/FLOPs/deploy/paper claim, or sparse-compute claim was run or unlocked.

Verification:

- Raw wildcard py_compile command failed locally because PowerShell did not expand wildcard paths for Python: `[Errno 22] Invalid argument: 'opentad/acquisition/mdl_knot/*.py'`.
- Expanded py_compile passed with exit code `0`.
- Focused pytest passed: `43 passed, 2 skipped in 23.13s`. The two skips are torch-import gated OpenTAD dataset smoke tests in this Windows environment.
- Launch gate passed as `PRECHECK_ONLY_REQUEST_ALLOWED` and reports `decoder_type=MDLKnotDecordDecode`; all training/evaluation/claim locks remain closed.

Next action:

- Stage-appropriate read-only review or coordinator integration.
- Formal/full training remains locked and cannot be claimed from this local speed-fix stage.
