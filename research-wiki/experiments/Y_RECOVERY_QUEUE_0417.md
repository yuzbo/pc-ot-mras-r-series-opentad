---
type: experiment
node_id: exp:Y_RECOVERY_QUEUE_0417
title: "Y-trunk recovery queue after Gemini 0417: normalization, point-set decoder, and immediate AMP observations"
date: 2026-04-17
status: completed
created_at: 2026-04-17T17:36:30+08:00
updated_at: 2026-04-18T11:11:49+08:00
---

# Y_RECOVERY_QUEUE_0417

## Scope

This page freezes the first recovery queue launched directly from the corrected Gemini 0417 direction review.

The goal is no longer generic "improve irregular head performance". It is narrower:

1. test whether the true irregular `y` trunk mainly fails because its output distribution is numerically hostile to the downstream decoder
2. test whether forcing the `y` trunk into a dense-grid head is the main contract mismatch
3. separate structural mismatch from pure AMP instability

## Current Snapshot

Timestamp:

- `2026-04-18T11:11:49+08:00`

Current server state:

- server `24013`
  - completed: `input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check_norm_ampoff = 10.34`
  - no active recovery screen remains on this server
  - retired as invalid runtime attempt: `input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check_norm` under AMP
- server `25876`
  - completed: `input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity = 44.06`
  - completed: `input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity_ampoff = 44.07`
  - no active recovery screen remains on this server

Why the queue changed immediately:

- the first AMP launch of `y_dense_grid_sanity_check_norm` no longer crashed at optimizer construction after the `output_norms.*` parameter-group fix
- but it still produced projection-wide non-finite gradients almost immediately
- the first AMP launch of `y_pointset_softsym_sanity` did enter training, but it already showed repeated non-finite gradients on `reg_head.weight`
- the first `pointset_ampoff` waiting command also had a deployment bug: the shell-side `while pgrep -f tools/train.py` matched its own command line and would wait forever; this was manually corrected and the run was relaunched directly

So this queue now carries two meanings:

- **method sanity**: normalization vs point-set decoding
- **numeric sanity**: AMP-on vs AMP-off on the true irregular `y` trunk

## Experiment Cards

### 1. `y_dense_grid_sanity_check_norm`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check_norm.py`
- Server:
  - `24013`
- Current status:
  - invalid runtime attempt under AMP; replaced by `norm_ampoff`
- Purpose:
  - test the smallest possible repair to the failed dense-grid adapter path
  - keep the same `y -> dense adapter -> ActionFormerHead` contract, but normalize the resampled dense features before decoding
- Main hypothesis:
  - if `13.86` was mainly a feature-scale / distribution issue, then masked output normalization should materially stabilize and raise this line
  - if it still stays extremely low, then dense-grid decoding itself is the dominant mismatch
- Immediate runtime observation:
  - the first AMP run produced widespread non-finite gradients on projection and neck parameters almost immediately
  - therefore the AMP result itself is not clean evidence for or against the method idea

### 2. `y_dense_grid_sanity_check_norm_ampoff`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check_norm_ampoff.py`
- Server:
  - `24013`
- Current status:
  - completed
- Purpose:
  - re-run the same normalized dense-grid recovery line without AMP, so the result is no longer confounded by early fp16 instability
- Main hypothesis:
  - if this run trains stably and rises well above `13.86`, then the second-ranked Gemini cause (`y`-output distribution / optimization instability) is real
  - if it remains near collapse even in fp32, then normalization is not enough and the dense-grid contract remains the main blocker
- Final result:
  - Average-mAP: `10.34`
- Interpretation:
  - `10.34` is even lower than `y_dense_grid_sanity_check = 13.86`
  - so simple output normalization does not rescue the dense-grid adapter path
  - this sharply lowers the priority of further dense-grid normalization micro-tuning on the true `y` trunk

### 3. `y_pointset_softsym_sanity`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity.py`
- Server:
  - `25876`
- Current status:
  - completed
- Purpose:
  - stop forcing the true irregular `y` trunk into a dense-grid ActionFormer head
  - replace that path with a minimal point/set-style bridge decoder:
    - irregular points
    - soft assignment
    - symmetric regression
    - point-wise tower (`tower_kernel_size=1`, `predictor_kernel_size=1`)
- Main hypothesis:
  - if this line climbs toward the old `headv2_y / headv3_y` regime (`~49`), then the dense-grid decoder contract was the main mismatch
  - if it still fails badly, then the true `y` trunk itself is not yet producing a decodable representation even for a point-wise irregular head
- Current runtime note:
  - completed
- Final result:
  - Average-mAP: `44.06`
  - tIoU 0.3/0.4/0.5/0.6/0.7: `66.73 / 58.19 / 46.74 / 31.96 / 16.68`
- Interpretation:
  - this is a decisive recovery over `y_dense_grid_sanity_check = 13.86`
  - so replacing dense-grid decoding with a point-wise irregular decoder is a real positive move on the true `y` trunk
  - but it still remains below `headv2_y = 49.00` and `headv3_y = 49.37`, so the current bridge-style point decoder is not yet the final answer
  - because repeated AMP non-finite skips were present during this run, the exact ceiling still needs the fp32 paired control

### 4. `y_pointset_softsym_sanity_ampoff`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_sanity_ampoff.py`
- Server:
  - `25876`
- Current status:
  - completed
- Purpose:
  - measure the same point-set decoder line without AMP confounding
- Main hypothesis:
  - if AMP-off removes the repeated skipped steps and gives a cleaner climb, then numeric instability is a real part of the current `y`-line failure
  - if AMP-off changes little, then the failure is more structural than numeric
- Final result:
  - Average-mAP: `44.07`
  - tIoU 0.3/0.4/0.5/0.6/0.7: `67.00 / 59.03 / 47.14 / 31.09 / 16.11`
- Interpretation:
  - the fp32 run is effectively identical to the AMP run `44.06`
  - so AMP is not the main blocker on the current `y_pointset_softsym` line
  - the decoder itself is still too weak to recover the old `~49` `y`-line regime

## Current Working Interpretation

Two conclusions are now already justified:

- the first `y`-recovery attempts did **not** falsify the Gemini 0417 ranking; they strengthened it
- and the strongest completed evidence inside this queue is now:
  - `y_dense_grid_sanity_check = 13.86`
  - `y_dense_grid_sanity_check_norm_ampoff = 10.34`
  - `y_pointset_softsym_sanity = 44.06`

More specifically:

1. dense-grid decoding is indeed a major mismatch, because moving to a point-wise irregular decoder recovers more than `30 mAP`
2. simple dense-grid feature normalization is not a meaningful recovery path on the true `y` trunk
3. the current point-wise bridge decoder is structurally too weak, because fp32 does not improve `44.06`
4. AMP instability existed in the runtime logs, but it is not the main explanation for the `y` ceiling
5. the next native-`y` diagnosis should move from numeric confounds to causal decoder diagnostics such as valid tiny-overfit tests and stronger support interfaces

## Immediate Next-Step Plan

1. stop treating AMP as the first-class blocker on the current `y_pointset` line
2. move the native-`y` diagnosis to valid tiny-overfit controls and stronger decoder / support-interface experiments
3. keep dense-grid normalization deprioritized unless a new hypothesis changes the adapter contract itself
4. do not spend more coding time on new `x`-line OABS/OAA micro-variants as a substitute for true `y`-line diagnosis

## What Would Count as Success

- `pointset_ampoff` reaching the old `~49` regime
  - would have supported the "mostly numeric ceiling" story
- `pointset_ampoff` staying near `44`
  - now confirmed; the current bridge-style point decoder is still structurally too weak even after removing AMP confounding
- both dense-grid runs failing badly (`13.86`, `10.34`)
  - keep the diagnosis one layer deeper: the true irregular `y` trunk representation is not usable under the current dense-head adapter path
