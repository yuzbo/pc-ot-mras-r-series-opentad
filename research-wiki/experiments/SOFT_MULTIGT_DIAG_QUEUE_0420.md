---
type: experiment
node_id: exp:SOFT_MULTIGT_DIAG_QUEUE_0420
title: "Post-mainline multi-GT soft-bridge diagnostics: keep2gt and detach-cls"
date: 2026-04-20
status: queued
created_at: 2026-04-20T18:58:00+08:00
updated_at: 2026-04-20T18:58:00+08:00
---

# SOFT_MULTIGT_DIAG_QUEUE_0420

## Scope

This queue is the direct follow-up to the Gemini discussion on the key discrepancy:

- `single_instance` joint overfit can fit
- `single_video` joint overfit fails badly

The purpose is to narrow that gap without launching another broad full-train sweep.

## Important Clarification

The current `single_instance` control is already close to the Gemini proposal
\"single positive + same-video negatives\":

- it keeps the same fixed sparse clip
- it only drops extra GT labels via `KeepSingleGT`
- background points remain in the sample

So the new missing variable is **how failure emerges as the number of supervised GTs increases**, not whether negatives exist at all.

## Newly Implemented Pieces

### 1. New transform

- `KeepGTSubset`
  - file: `OpenTAD_Back/opentad/datasets/transforms/formatting.py`
  - purpose: keep the first `N` GTs after fixed truncation so we can build a clean `1 -> 2 -> full` ladder

### 2. New configs

- `input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_keep2gt_joint.py`
  - tests whether failure already appears with only `2` supervised GTs in the same fixed clip

- `input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_keep2gt_joint_detachcls.py`
  - same `2-GT` setup, but detaches `cls` from the shared trunk

### 3. Reused existing config

- `input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_true_overfit_single_video_joint_detachcls.py`
  - full single-video joint supervision with `cls` detached from trunk

## Queue Layout

### Server `24013`

- screen: `y_soft_multigt_diag_s1`
- waiting behind the active no-warmup mainline
- queued run:
  - `single_video_joint_detachcls`

### Server `25876`

- screen: `y_soft_multigt_diag_s3`
- waiting behind the active `regwarm10` mainline
- queued runs:
  - `single_video_keep2gt_joint`
  - `single_video_keep2gt_joint_detachcls`

## Runtime Refresh

Timestamp:

- `2026-04-20T19:00:00+08:00`

Current waiting state:

- `24013`
  - queue log confirms `single_video_joint_detachcls` passed config check
  - still waiting on GPU release from the no-warmup mainline
  - observed GPU memory while waiting: about `3197 MiB`

- `25876`
  - queue log confirms both `keep2gt` configs passed config check
  - still waiting on GPU release from the `regwarm10` mainline
  - observed GPU memory while waiting: about `3197 MiB`

Interpretation:

- the new diagnostic queue is operational
- it has not preempted or disturbed either active full-train mainline
- the next action is still to let the two mainlines finish or at least reach the first trustworthy validation read

## Runtime Refresh 2

Timestamp:

- `2026-04-20T19:46:00+08:00`

Current waiting state:

- `24013`
  - queue is still waiting behind the active no-warmup mainline
  - latest observed GPU memory while waiting moved from about `3197 MiB` to about `3215 MiB`
  - no config failure or premature launch occurred

- `25876`
  - queue is still waiting behind the active `regwarm10` mainline
  - latest observed GPU memory while waiting also moved from about `3197 MiB` to about `3215 MiB`
  - no config failure or premature launch occurred

Read:

- the queued diagnosis remains cleanly staged
- no accidental preemption happened
- the next meaningful event is still the mainline finishing or the first validation output appearing

## Runtime Refresh 3

Timestamp:

- `2026-04-20T21:46:00+08:00`

Current waiting state:

- `24013`
  - `single_video_joint_detachcls` queue still waiting
  - active mainline GPU memory while waiting is about `3217 MiB`

- `25876`
  - `keep2gt_joint -> keep2gt_joint_detachcls` queue still waiting
  - active mainline GPU memory while waiting is about `3215 MiB`

Interpretation:

- even after the first validation reads landed, neither mainline has finished
- so the queued diagnostic ladder still has not started
- the next highest-value event is now either:
  - the final full-train comparison closing
  - or one of the two mainlines releasing GPU so the queued multigt ladder can begin

## Diagnostic Read Logic

### If `keep2gt_joint` already fails

- then the failure boundary is very low: going from `1 GT` to `2 GTs` is enough to break optimization
- this points to multi-positive competition / support conflict rather than only dense background pressure

### If `keep2gt_joint` succeeds but full `single_video` fails

- then the instability scales with GT density / proposal crowding, not just the existence of multiple positives

### If `detachcls` rescues either `keep2gt` or full `single_video`

- then the current leading hypothesis becomes much stronger:
  - early `cls` gradients are the main destructive force on the shared trunk

### If `detachcls` still does not rescue

- then the problem is deeper than cls-trunk interference alone
- next suspects become assignment instability, target competition, or a masking / collation bug

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
