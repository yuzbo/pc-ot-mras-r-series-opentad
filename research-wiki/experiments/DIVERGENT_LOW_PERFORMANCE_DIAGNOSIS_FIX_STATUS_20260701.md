# Divergent Low-Performance Diagnosis Fix Status - 2026-07-01

Status time: 2026-07-01 17:45 +0800

Scope: read-only diagnosis followed by route-owned fixes for the divergent
innovation routes. The shared repository was not used for route code edits.

## Summary

Two concrete root causes were fixed in isolated route worktrees:

- BVR-TWB: low effective detector-token budget was hidden by raw-frame budget
  checks and fixed Adapter padding.
- MDL-Knot: very short windows could select all dense inputs, violating the
  true sparse selected-only handoff invariant.

No final metric, runtime, FLOPs, deploy, sparse-compute, or paper claim is
unlocked by this status update.

## BVR-TWB Budget Fix

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree:
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_BudgetFix_Worktree_20260701`

Owned branch: `codex/divergent-bvr-twb-budgetfix-20260701`

Commit: `895474b Fix BVR TWB effective budget floor`

Root cause: the previous route enforced a raw frame floor, not an effective
detector-token floor. With `feature_stride=2`, a low raw `valid_k` could become
an extremely low detector `valid_k`, while fixed padding hid the under-budget
selection.

Changed surfaces:

- BVR budget controller
- BVR OpenTAD bridge and metadata
- BVR validators and launch gate
- BVR config
- BVR pipeline/audit tests

Owner verification:

- Focused red/green tests passed.
- All BVR route tests: `66 passed, 17 skipped, 1 warning`.
- BVR pipeline audit: `ledgers=3 all_validated=True sparse_compute_claim=False blocked=False`.
- Launch gate: `gate_pass=true`, `full_train_unlocked=false`,
  `remote_sync_unlocked_by_local_gate=false`.

Final read-only review:

- Verdict: `PASS_SUBAGENT_FINAL_REVIEW_ONLY`
- Blocking findings: none
- Remaining locks: Linux precheck required before any sync; full training,
  remote sync, Slurm, `tools/test.py`, sparse-compute claim, metric claim,
  and learned-regret claim remain locked.

## MDL-Knot Sparse Handoff Fix

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree:
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_SparseHandoffFix_Worktree_20260701`

Owned branch: `codex/divergent-mdl-knot-sparse-handoff-fix-20260701`

Commit: `4bedd4b DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3 fix sparse handoff cap`

Root cause: selector allowed `valid_k == dense_T` on short real windows when
`dense_t=4`, `min_k=4`, and `max_k=384`, selecting `[0,1,2,3]`. That is dense
passthrough, so the selected-only sparse audit correctly failed.

Changed surfaces:

- MDL-Knot selector cap logic
- MDL-Knot validators
- MDL-Knot sparse handoff tests
- MDL route evidence document

Owner verification:

- Reproduction tests failed before the fix and passed after the fix.
- MDL tests: `49 passed, 2 skipped`.
- MDL pipeline precheck audit exited `0`.
- `py_compile` passed for changed source files.

Final read-only review:

- Verdict: `PASS_SUBAGENT_FINAL_REVIEW_ONLY`
- Blocking findings: none
- Non-blocking note: `dense_T < 3` now fail-closes, which is correct because
  the route cannot both keep endpoints and remain sparse.
- Remaining locks: remote sync, Slurm, training, evaluation, `tools/test.py`,
  and all metric/runtime/deploy/paper/sparse-compute claims remain locked.

## Remote State Observed

At 2026-07-01 17:33 +0800:

- Protected hold `1118197 pcot_dbg2g` remained active.
- BVR child step `1118197.542 bvr_twb_fix2_g0` was running on GPU0.
- C3 child step `1118197.571 c3_oracle_full_g1` was running on GPU1.
- RBA controls `1133000 rba_ctrlfull` and `1133001 rba_ctrllow` were pending.
- Original AdaTAD baseline `1133021 adatad_orig2g` was pending.
- PHASER jobs `1133022-1133026` were pending.

## Next Allowed Actions

1. Run Linux precheck for BVR budgetfix only; do not launch full training yet.
2. Run Linux precheck or short diagnostic for MDL sparse handoff fix only; do
   not launch full training yet.
3. Wait for RBA control diagnostics before changing RBA-RBR further.
4. Keep the current BVR long run as historical evidence; do not reinterpret it
   as the fixed implementation.
