# DIVERGENT MDL-Knot Implementation

Date: 2026-06-29 Asia/Shanghai

Route label: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Owned full-code implementation worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_FullCode_Worktree_20260629`

Superseded evidence-only worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_Worktree_20260629`

Status: local implementation and precheck tooling only. Remote sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, push, mAP/runtime/FLOPs/deploy/paper claims remain locked.

Changed surface:

- Input sampling: MDL-Knot dynamic sparse selection before decode.
- Dynamic budget policy: greedy minimum-description-length knot selection.
- Token compression: unchanged.
- Adapter/backbone internals: unchanged.
- Detector head logic: unchanged.
- Loss/assignment: unchanged.
- Test-time post-processing: unchanged.

Implemented components:

- `ScoutCurve`, `KnotLedger`, `SparseTemporalMeta`, objective terms, provenance checks.
- Deploy-visible scout builder from actionness, uncertainty, change, persistence, optional motion.
- Weighted reconstruction objective, complexity penalty, long-gap risk, short-island duration risk, transition risk.
- Greedy selector with candidate pool from residuals, transitions, uncertainty, gap midpoints, short islands, endpoints, and scaffold anchors.
- Matched controls: per-video same-K uniform, mean-K exact-uniform, random same-K, scaffold-only, MDL-only, MDL plus transition/gap/duration risk.
- Real sparse handoff validator and OpenTAD `LoadFrames`/`Collect` integration surface.
- Synthetic/offline ledger builder, precheck audit tool, launch gate validator.

No claims:

- No detector training was run.
- No detector evaluation was run.
- No mAP, runtime, FLOPs, deploy, or paper claim is made from this implementation.

## Hume Final Review Fixes

Recorded 2026-06-30 Asia/Shanghai.

Accepted blockers fixed:

- Linux true `LoadFrames` test now asserts fixed-pad Adapter bridge semantics: padded `frame_inds` length equals Adapter target, only first `valid_k` are valid, padding duplicates the last valid frame, and padding does not count as valid.
- `LoadFrames(method="mdl_knot_dynamic_subsample")` keeps dynamic MDL `valid_k` but pads raw indices to the fixed Adapter length required by inherited `window_size=384` contracts.
- Launch gate now requires `tools_test_py` to remain locked, rejects `COMBO`, and checks explicit random-fixed/C3 drift tokens in config evidence in addition to the split load methods.
- Selector no longer overwrites unsafe scout provenance. If a scout curve declares GT, teacher, prediction-cache, or dense-backbone provenance, ledger creation fails before validation.

Remaining locks:

- Remote sync and N16R4 PRECHECK_ONLY remain blocked until the latest read-only review no longer reports blockers.
- Training, evaluation, `tools/test.py`, stage/commit/push, mAP/runtime/FLOPs/deploy/paper claims remain locked.
