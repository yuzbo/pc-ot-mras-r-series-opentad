# Divergent Low-Performance Diagnosis Fix Status - 2026-07-01

Status time: 2026-07-01 18:28 +0800

Scope: read-only diagnosis followed by route-owned fixes for the divergent
innovation routes. The shared repository was not used for route code edits.

## Summary

Two concrete root causes were fixed in isolated route worktrees, and the
remote Linux checks now split the routes into different states:

- BVR-TWB: low effective detector-token budget was hidden by raw-frame budget
  checks and fixed Adapter padding. The first fix passed locally but failed a
  real Linux/OpenTAD pipeline precheck because selected valid frames were still
  too few for the fixed Adapter length. A second fix now enforces the raw-frame
  floor implied by the duplicate-padding guard and passes Linux PRECHECK_ONLY.
- MDL-Knot: very short windows could select all dense inputs, violating the
  true sparse selected-only handoff invariant. The fix also passed the remote
  Linux static/synthetic sparse-handoff precheck.

No final metric, runtime, FLOPs, deploy, sparse-compute, or paper claim is
unlocked by this status update. In particular, the currently running old BVR
job is historical pre-fix evidence and must not be used as proof of the new
budgetfix branch.

## BVR-TWB Budget Fix

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree:
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_BudgetFix_Worktree_20260701`

Owned branch: `codex/divergent-bvr-twb-budgetfix-20260701`

Initial commit: `895474b Fix BVR TWB effective budget floor`

Follow-up commit: `8a02314 fix bvr twb adapter raw budget floor`

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

Remote Linux/OpenTAD precheck update:

- Remote clone:
  `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_BudgetFix_Precheck_20260701_895474b`
- Remote log dir:
  `logs/bvr_budgetfix_linux_precheck_895474b_20260701_180138_+0800`
- Result: `3 failed, 28 passed` in the real Linux/OpenTAD pipeline tests.
- Main blocker: `validate_bvr_twb_budget_floor` rejected duplicate padding
  dominance, e.g. `duplicate_ratio=0.5625` and `0.5938` while
  `max_adapter_padding_duplicate_ratio=0.5000`. This means the first fix
  raised the detector-facing budget floor but did not fully enforce a raw
  valid-frame floor derived from the fixed Adapter length and padding guard.
- Secondary tooling blocker: `tools/audit_bvr_twb_pipeline.py --overwrite`
  refused an existing output directory without a `bvr_twb` marker, so the
  precheck summary was not produced.
- Follow-up fix: commit `8a02314` adds a raw-frame floor
  `ceil(target_frame_num * (1 - max_adapter_padding_duplicate_ratio))`, records
  it in the ledger as `min_adapter_raw_keep_for_padding_guard`, passes the
  chosen `max_adapter_padding_duplicate_ratio` through `LoadFrames`, and scopes
  the audit overwrite path to the tool's own output files.
- Final local checks after the follow-up fix: focused tests
  `26 passed, 7 skipped`; all BVR-matching tests
  `68 passed, 17 skipped, 42 deselected, 1 warning`; `py_compile` PASS; final
  read-only review verdict `PASS_SUBAGENT_FINAL_REVIEW_ONLY`.
- GitHub sync: branch `codex/divergent-bvr-twb-budgetfix-20260701` pushed to
  `8a02314`.
- Final Linux PRECHECK_ONLY on N16R4:
  remote clone fast-forwarded to `8a02314`, run dir
  `logs/bvr_budgetfix_linux_precheck_8a02314_20260701_182139_+0800`;
  Linux pipeline pytest `33 passed in 22.48s`; OpenTAD pipeline audit printed
  `ledgers=3 all_validated=True sparse_compute_claim=False blocked=False`;
  launch gate rerun printed `gate_pass=true`, `full_train_unlocked=false`,
  `remote_sync_unlocked_by_local_gate=false`.
- Summary evidence from the Linux audit: `min_raw_valid_k=32`,
  `configured_min_raw_keep=32`, `min_detector_feature_valid_k=16`,
  `max_adapter_padding_duplicate_ratio=0.3333333333333333`,
  `max_allowed_adapter_padding_duplicate_ratio=0.5`,
  `preview_sources=['deploy_visible_metadata_actionness']`,
  `value_modes=['deploy_heuristic_voi']`.
- Action: BVR is no longer blocked at Linux PRECHECK_ONLY, but it is still not
  unlocked for formal/full training. The next useful step is a bounded short
  diagnostic or an explicit launch gate decision that cites commit `8a02314`,
  not the older running BVR job.

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

Remote Linux precheck update:

- Remote clone:
  `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SparseHandoffFix_Precheck_20260701_4bedd4b`
- Log dir:
  `logs/mdl_sparse_handoff_linux_precheck_4bedd4b_20260701_180138_+0800`
- Result: `51 passed in 88.22s`.
- Summary:
  `logs/mdl_sparse_handoff_linux_precheck_4bedd4b_20260701_180138_+0800/pipeline_precheck/mdl_knot_precheck_summary.json`
- Summary verdict: `validated=true`, `validation_scope=synthetic_offline_real_sparse_handoff_only`,
  `formal_train_unlocked=false`, and all mAP/runtime/FLOPs/deploy/paper/
  sparse-compute claims remain locked.

## Per-Route Low-Performance Diagnosis

### ABR

ABR does not have a detector mAP result. Its failure is a route-structure
diagnostic: first-round bracket recall remained far below the formal threshold.
The latest meaningful repair3 diagnostic on 32 validation videos recorded
`first_round_bracket_recall=0.6980`, `transition_coverage=0.6305`, and
`273/904` missed transitions. Because later ABR rounds only refine inside the
first bracket, missed first-round boundaries are mostly unrecoverable. The
route was frozen by user decision; low mAP is not the evidence here.

### RBA-RBR

RBA-RBR has implementation, tests, remote precheck, and short diagnostics, but
not a formal train. The coverage-guard short diagnostic completed with
failure-scale Average-mAP `7.45%` and vector
`17.34 / 11.47 / 5.69 / 2.09 / 0.65`. Guard audit showed coverage/gap was
substantially restored (`raw_valid_k` mean about `84.62`, `mask_true_count`
mean about `42.56`, raw max gap `16`, detector max gap `24`), so the remaining
low result is no longer explained by a simple no-coverage bug. Current
suspects are detector-facing geometry, fixed padded bridge effects,
assignment/loss/postprocess compatibility, or weak regret signal attribution.
Formal training is locked pending severe-result diagnosis and matched controls.

### BVR-TWB / VOI-BBC

BVR has real low-platform evidence from the old pre-budgetfix branch. The
currently running child step `1118197.542 bvr_twb_fix2_g0` uses remote tree
`OpenTAD_BVR_TWB_Final_20260630_92ec024` and launch
`bvr_twb_pathfix_restart2_gpu0_5d11ffd_20260701_070013_+0800`; it is not the
new `895474b` budgetfix branch. Its intermediate Average-mAP sequence observed
so far is `21.78 -> 22.08 -> 22.55 -> 22.86 -> 23.09 -> 23.41 -> 23.92 -> 23.97`.
Training is stable but low. This evidence must be interpreted as pre-fix
budget/geometry evidence, not as the final BVR design result.

The budgetfix diagnosis found that prior BVR could satisfy raw-frame-looking
budget checks while presenting very few detector tokens after feature stride
and fixed padding. Commit `8a02314` now also enforces the raw valid-frame floor
needed to keep fixed-Adapter duplicate padding below the configured guard. This
changes BVR's effective budget semantics: future runs must report detector-token
budget and adapter-padding ratio, not just raw selected-frame count. BVR has now
passed Linux PRECHECK_ONLY at the repaired commit, but this does not unlock
formal training or metric claims.

### C3/CADF Boundary Context

C3/CADF is not part of the divergent route and must not be mixed into BVR/MDL/
RBA attribution. Read-only tracker evidence is still relevant for the global
low-performance question: CADF loss-select V2 plateaued in the low-to-mid 20s
Average-mAP, and a bounded train selector dump after child stop found action
occupancy `29.2%`, selected-action precision `30.4%`, action coverage `55.8%`,
boundary recall `97.7%`, boundary-near selected rate `8.26%`, and max-gap p95
`4`. This suggests weak task-aware enrichment plus possible backend/ranking/
geometry risks. It does not prove the divergent routes are broken.

### Original AdaTAD Baseline Sanity

Because several sparse/acquisition experiments are unexpectedly low, an
original AdaTAD VideoMAE-S adapter two-GPU baseline sanity run was queued as
Slurm job `1133021 adatad_orig2g` on public N16R4 `gpu`, excluding protected
hold node `g0030`. It uses remote clean clone commit `588b272` and official
style config `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`
with only `work_dir` overridden. As of `2026-07-01 18:07 +0800` it remains
`PENDING (Priority)`, so there is still no evidence whether the global
data/pretrain/evaluator stack can reproduce the expected original AdaTAD
performance.

### MDL-Knot

MDL-Knot has no detector mAP result yet. Its current status is code-contract
repair plus Linux precheck pass. The fixed route prevents short-window dense
passthrough and preserves selected-only sparse handoff semantics. Its main
remaining risks are not currently "low performance" but "not yet evaluated":
fixed-pad still gives the detector a fixed 384-length tensor, raw scout decode
can be slow, average dynamic `valid_k` distribution is unproven on real videos,
and the detector geometry path has not been validated by a real short
diagnostic. It may proceed only to restricted short diagnostic, not formal
training.

## Remote State Observed

At 2026-07-01 17:33 +0800:

- Protected hold `1118197 pcot_dbg2g` remained active.
- BVR child step `1118197.542 bvr_twb_fix2_g0` was running on GPU0.
- C3 child step `1118197.571 c3_oracle_full_g1` was running on GPU1.
- RBA controls `1133000 rba_ctrlfull` and `1133001 rba_ctrllow` were pending.
- Original AdaTAD baseline `1133021 adatad_orig2g` was pending.
- PHASER jobs `1133022-1133026` were pending.

## Next Allowed Actions

1. Treat BVR commit `8a02314` as PRECHECK_ONLY-passed and ready for a bounded
   short diagnostic or explicit launch-gate discussion; do not launch full
   training yet.
2. Treat MDL commit `4bedd4b` as Linux synthetic/offline sparse-handoff
   precheck passed; next action is a restricted one-epoch short diagnostic
   only, not formal training.
3. Wait for RBA control diagnostics before changing RBA-RBR further.
4. Keep the current BVR long run as historical evidence; do not reinterpret it
   as the fixed implementation.
