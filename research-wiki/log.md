# Research Log

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
