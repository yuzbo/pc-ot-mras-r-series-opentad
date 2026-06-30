# DIVERGENT MDL-Knot Formal Readiness Hardening

Timestamp: 2026-06-30 Asia/Shanghai

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_FormalGate_Worktree_20260630`

Owned branch: `codex/divergent-mdl-knot-formal-gate-20260630`

## Purpose

This change hardens MDL-Knot formal-readiness evidence without unlocking formal training.

The route remains an input-sampling and dynamic-budget-policy route. It does not change Adapter/backbone internals, detector head logic, losses/assignment, evaluation, or test-time post-processing.

## Added Diagnostics

Sample-level pipeline diagnostics now record:

- raw-frame scout usage;
- metadata fallback usage and ratio after aggregation;
- synthetic fallback rejection;
- dynamic `valid_k`;
- `max_gap` and `gap_p95`;
- mask, sparse metadata, selected positions, and `valid_k` alignment;
- short-island guard coverage;
- transition guard coverage;
- fixed-pad bridge compute boundary.

Aggregated diagnostics distinguish synthetic/offline precheck evidence from real-video pipeline evidence. Synthetic precheck summaries explicitly keep `real_video_pipeline_diagnostics=None`.

## Formal Gate State

`tools/mdl_knot/validate_mdl_knot_launch_gate.py` still allows local `PRECHECK_ONLY_REQUEST_ALLOWED` when the config and synthetic/offline precheck are valid.

The new `--formal-readiness-summary` path is fail-closed. It stays locked unless a summary contains:

- real-video pipeline diagnostics with at least one raw-frame scout window;
- zero synthetic fallback windows;
- nonconstant `valid_k` distribution;
- `max_gap` and `gap_p95` distributions;
- passing mask/metadata alignment;
- measurable short-action/boundary guard evidence;
- zero uncovered short-island and transition guard counts;
- validated shortdiag execution evidence with a real one-epoch train-log path and finite train losses;
- explicit locked actions for remote sync, Slurm, training, evaluation, and `tools/test.py`;
- explicit no-claim locks for mAP, runtime, FLOPs, deployment, paper, and sparse-compute claims.

Current state: formal train remains locked. Complete diagnostic evidence can only prove `FORMAL_READINESS_DIAGNOSTICS_PRESENT`; it does not unlock formal training, remote sync, Slurm, evaluation, deployment, or claims. No full-train user/coordinator approval exists in this stage.

## 2026-06-30 Final Review Blocker Fixes

The 075553b final read-only review blockers were fixed in this owned worktree only:

- Route drift detection now rejects `C3`, `C3-Pro`, `C3_MAINLINE`, and GlobalRank spellings after stripping the legal MDL route label.
- Formal readiness evidence now rejects contradictory summaries, including `formal_train_unlocked=True`, unlocked training/evaluation/tools actions, and open mAP/runtime/FLOPs/deploy/paper/sparse-compute claims.
- Short diagnostic validation without `--train-log` is now a static config check only: it exits successfully for local inspection but emits `validated=false`, `log_evidence=null`, and `evidence_scope=static_config_only`, so it cannot satisfy formal readiness.
- Short-action and boundary guard uncovered counts are now readiness blockers. Any uncovered short-island or transition band keeps formal readiness locked.

## 2026-06-30 GlobalRank Shortdiag Drift Blocker Fix

The commit `1f50abf` final read-only review found one remaining blocker: the short diagnostic train-log route-drift check did not reject GlobalRank spellings, so a log containing `GlobalRank diagnostic drift marker` could be accepted as `validated=true` and later trusted as shortdiag execution evidence.

This stage fixes that blocker in the owned worktree only:

- `tools/mdl_knot/validate_mdl_knot_shortdiag.py` now reuses the launch-gate `FORBIDDEN_ROUTE_DRIFT` tuple instead of maintaining a shorter local token list.
- Shortdiag config text and train logs now reject `GLOBALRANK`, `GLOBAL_RANK`, `GLOBAL-RANK`, and `GLOBAL RANK`, along with the existing C3/BVR/ABR/combo drift tokens.
- Regression tests confirm a GlobalRank drift log exits `LOCKED`, emits no `SHORTDIAG_EVIDENCE`, and therefore cannot be assembled into a formal-readiness summary as validated shortdiag evidence.

The follow-up final read-only review found a deeper stale-summary risk: even if the current shortdiag validator refuses GlobalRank drift, an external or stale formal-readiness JSON could still set `shortdiag_evidence.validated=true` and point `log_evidence.train_log` at a route-drift log. This stage closes that evidence-chain gap:

- `opentad/acquisition/mdl_knot/diagnostics.py` now revalidates the referenced shortdiag train log before accepting formal-readiness evidence.
- The revalidation checks the same route-drift family, fatal markers, non-finite loss markers, evaluation/checkpoint/claim markers, finite loss count, and one-epoch limit.
- `tools/mdl_knot/validate_mdl_knot_launch_gate.py` resolves relative train-log paths against the formal-readiness summary directory before falling back to the repo root.
- Regression tests now construct a stale/forged formal summary pointing at a `GlobalRank` train log and require the formal-readiness gate to stay `LOCKED`.

State after this fix remains locked: `full_train_unlocked=false`. No remote sync, GPU, Slurm, `tools/test.py`, training, evaluation, mAP, runtime/FLOPs, deployment, paper, or sparse-compute claim is made or unlocked.

## Fixed-Pad Compute Boundary

MDL-Knot currently uses `fixed_pad` to preserve the inherited Adapter input length. Dynamic `valid_k` is measured and padding is masked, but the detector still receives the fixed target tensor length.

Therefore:

- no sparse-compute claim is made;
- no runtime/FLOPs claim is made;
- no deployment claim is made;
- no paper claim is made;
- no mAP claim is made.

This bridge is a compatibility bridge for acquisition diagnostics, not evidence that the current route reduces backbone/Adapter compute.

## Verification Scope

Allowed local verification only:

- Python compile;
- focused pytest;
- synthetic/offline precheck validator;
- shortdiag config/log validator;
- formal-readiness gate rejection checks.

Forbidden in this stage:

- training;
- `tools/test.py`;
- evaluation;
- remote sync;
- Slurm/GPU;
- Pro/Oracle/Rosetta;
- C3/BVR/ABR/combo route changes.

## Verification Rerun

Commands rerun locally in the owned worktree on branch `codex/divergent-mdl-knot-formal-gate-20260630`:

- `python -m py_compile opentad\acquisition\mdl_knot\diagnostics.py tools\mdl_knot\validate_mdl_knot_launch_gate.py tools\mdl_knot\validate_mdl_knot_shortdiag.py tools\mdl_knot\audit_mdl_knot_pipeline_precheck.py tests\test_mdl_knot_shortdiag.py tests\test_mdl_knot_tools_and_integration.py` passed.
- `python -m pytest tests\test_mdl_knot_shortdiag.py tests\test_mdl_knot_tools_and_integration.py -q` passed: `21 passed, 1 skipped`.
- `python -m pytest tests\test_mdl_knot_core.py tests\test_mdl_knot_shortdiag.py tests\test_mdl_knot_tools_and_integration.py -q` passed: `34 passed, 1 skipped`.
- `python tools\mdl_knot\validate_mdl_knot_launch_gate.py --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3.py` passed as `PRECHECK_ONLY_REQUEST_ALLOWED` and kept all locks.
- `python tools\mdl_knot\validate_mdl_knot_shortdiag.py --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py` passed as `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` with `validated=false`.
- `python tools\mdl_knot\validate_mdl_knot_shortdiag.py --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py --train-log <temporary finite-loss one-epoch log>` passed as `SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED` with `validated=true`.
- `python tools\mdl_knot\validate_mdl_knot_launch_gate.py --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3.py --formal-readiness-summary __missing_formal_readiness_summary__.json` locked as expected.

Additional blocker-fix verification for the commit `1f50abf` GlobalRank review finding:

- `python -m py_compile tools\mdl_knot\validate_mdl_knot_shortdiag.py tools\mdl_knot\validate_mdl_knot_launch_gate.py tests\test_mdl_knot_shortdiag.py tests\test_mdl_knot_tools_and_integration.py` passed.
- `python -m pytest tests\test_mdl_knot_shortdiag.py tests\test_mdl_knot_tools_and_integration.py -q` passed: `30 passed, 1 skipped`.
- `python tools\mdl_knot\validate_mdl_knot_shortdiag.py --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py` passed as `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`, with `validated=false`, `log_evidence=null`, and all full-train/metric/sparse-compute locks preserved.
- `python tools\mdl_knot\validate_mdl_knot_launch_gate.py --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3.py` passed as `PRECHECK_ONLY_REQUEST_ALLOWED`, while preserving remote sync, Slurm, training, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper locks.
- `python tools\mdl_knot\validate_mdl_knot_launch_gate.py --config configs\adatad\thumos\input_mdl_knot_dynamic_adapter_irregular_headv3.py --formal-readiness-summary __missing_formal_readiness_summary__.json` returned `LOCKED` as expected.

No training, evaluation, GPU, Slurm, remote sync, `tools/test.py`, Pro/Oracle/Rosetta, mAP claim, runtime/FLOPs claim, deploy claim, paper claim, or sparse-compute claim was run or unlocked.
