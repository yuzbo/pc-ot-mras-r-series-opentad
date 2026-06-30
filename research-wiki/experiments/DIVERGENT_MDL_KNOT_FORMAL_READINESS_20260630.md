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
- validated shortdiag evidence.

Current state: formal train remains locked because no real-video pipeline diagnostic summary and no validated shortdiag execution evidence are present in this worktree.

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
