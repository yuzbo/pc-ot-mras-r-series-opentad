---
type: review
node_id: review:GEMINI_DIRECTION_0417
title: "Gemini direction review after y_dense_grid_sanity_check closed at 13.86"
date: 2026-04-17
status: completed
created_at: 2026-04-17T17:18:12+08:00
updated_at: 2026-04-17T17:18:12+08:00
---

# GEMINI_DIRECTION_REVIEW_0417

## Scope

This review was run after the repair queue fully closed and the strongest remaining sanity check also failed:

- `step0_dense_head_baseline = 53.72`
- `step0b_dense_points_soft_sym_repaired = 47.52`
- `headv3_oabs_full_x_legacysingleton = 53.48`
- `headv3_oabs_full_x_fixsingleton_repaired = 52.88`
- `y_dense_grid_sanity_check = 13.86`

The goal was to answer a narrower and sharper question than the 0416 review:

1. why have the recent irregular-head / irregular-trunk lines produced almost no effective improvement?
2. after `y_dense_grid_sanity_check = 13.86`, what is now the strongest supported root-cause ranking?
3. what should become the next experimental priority, and what should be deprioritized?

## Important Correction Applied Before Accepting Gemini's Diagnosis

Before the final diagnosis was accepted, one earlier causal reading was explicitly corrected:

- `63.12 -> 53.72` must **not** be treated as a clean shell / bridge overhead number

Reason:

- `53.72` is only a repaired **remapped selected-axis dense-head control**
- it is **not** a true no-remap shell-exact baseline

So this review does **not** rely on the unsupported claim that the current bridge alone costs `9.40 mAP`.

## Main Evidence Sent to Gemini

- dense random-fixed baseline: `63.12`
- boundary oracle dense: `66.01`
- `headv2_x = 51.04`
- `headv2_y = 49.00`
- `headv3_x = 52.60`
- `headv3_y = 49.37`
- `headv3_oabs_full_x = 53.16`
- `headv3_oabs_full_oaa_x = 53.20`
- clean singleton A/B: `53.48 -> 52.88`
- repaired Step0 / Step0b: `53.72 -> 47.52`
- true irregular-trunk sanity check: `y_dense_grid_sanity_check = 13.86`
- auxiliary evidence: `y_dense_grid_sanity_check` stayed low throughout formal validations
  - `13.84 -> 13.97 -> 13.86`
- training also showed non-finite gradients in `cls_head.weight`

## Gemini's Corrected Main Judgment

After the `53.72` interpretation was corrected, Gemini still kept the same core conclusion:

- the strongest supported diagnosis is now **semantic mismatch between the true irregular `y` trunk and the current dense-head decoding contract**

This conclusion is now anchored primarily by:

1. `y_dense_grid_sanity_check = 13.86`
2. the flat low validation trajectory
3. the non-finite gradients on the dense-head classification layer

Gemini's formulation was direct:

- the `y` trunk is producing feature distributions that the current dense-grid head cannot consume stably
- this is no longer a "maybe bad tuning" story
- it is a structural incompatibility story

## Strong Conclusions vs Tentative Conclusions

### Strong

1. the true irregular `y` trunk is severely incompatible with the current dense-head decoding contract
2. the current `x` line is not a valid carrier for proving full irregular-geometry benefit
3. soft assignment is still clearly negative inside the repaired remapped control family
   - `53.72 -> 47.52`
4. singleton handling is not a serious bottleneck on the current `x`-line carrier
   - `53.48 -> 52.88`

### Tentative

1. how much loss comes specifically from remap / bridge logic is still unresolved
2. whether soft assignment is intrinsically weak, or only weak under the current mismatched carrier, is still unresolved
3. how much of the dense-to-irregular gap can be recovered by a better decoder alone is still unresolved

## Current Root-Cause Ranking

### 1. `y`-trunk vs dense-head semantic mismatch

This is now the top supported root cause.

Why:

- `y_dense_grid_sanity_check = 13.86`
- the run never meaningfully recovered
- the classification head exhibited non-finite gradients

Interpretation:

- the true irregular trunk is not just under-performing
- it is feeding the dense head a feature manifold that violates the head's decoding assumptions

### 2. output-distribution / optimization instability on the `y` line

This is related to, but narrower than, the first point.

Gemini's reading was:

- the head is "choking" on the `y`-trunk output distribution
- scale / variance / non-stationarity mismatch is likely part of the failure mechanism

This is why simple normalization tests now have high value.

### 3. `x` line is the wrong carrier for proving irregular benefit

The `x` line can still be useful as a dense-trunk + irregular-head carrier, but it should not keep being over-read.

Why:

- its best result is still only `53.48`
- all gains from `HeadV2 -> HeadV3 -> OABS/OAA` are small
- it does not tell us whether true irregular feature processing can work

### 4. soft assignment / current head semantics are negative, but secondary

`step0b = 47.52` still matters, but Gemini no longer treated it as the main story.

Correct reading:

- soft assignment is a real penalty in the repaired remapped control family
- but the larger structural failure is the `y`-trunk / dense-head contract itself

### 5. remap / bridge overhead is still unresolved

This stays on the list, but only as a tentative item.

What changed:

- we no longer accept `63.12 -> 53.72` as a clean overhead number
- so the right attitude is: bridge remains suspicious, but not yet proven

## Rewritten Next-Step Priority

### Priority 1. Build a null bridge control

Goal:

- isolate whether the bridge / remap path itself destroys dense semantics

Cleanest version:

- take dense-trunk outputs
- pass them through the same bridge / remap logic
- feed them to the original dense-style head

This is the highest-value control because it resolves a major remaining uncertainty without mixing in true irregular-trunk failure.

### Priority 2. Normalize the `y`-trunk output before decoding

Goal:

- test whether the immediate failure mode is feature-distribution mismatch

Minimal intervention:

- add `LayerNorm` or similar normalization directly before the head consumes `y`-trunk outputs

Why this is high value:

- if gradients become stable and mAP lifts materially, then the failure is at least partly a distribution-matching problem rather than a deeper logic break

### Priority 3. Stop forcing `y` into a dense-grid head; try a point / set decoder

Goal:

- directly test whether the current dense-grid decoding contract is the wrong one

Minimal version:

- a point-based MLP or small set-style decoder over irregular tokens

Why this matters:

- if it clearly beats `13.86`, then the dense-head contract itself is proven to be the wrong frame for the true irregular trunk

### Priority 4. Keep completion as a parallel mainline

Gemini did not walk back the completion direction.

But after this review, completion should be motivated more carefully:

- not by a fake `9.40 mAP shell-overhead` number
- but by the now-strong evidence that direct true-irregular-trunk decoding is failing badly

## Directions Now Explicitly Deprioritized

1. more `OABS / OAA` micro-tuning on the `x` line
2. more singleton / boundary-aux style local fixes on the current carrier
3. more soft-assignment variants before establishing a stable and causally clean baseline
4. treating the `x` line as if it were evidence for full irregular-aware detection

## What This Changes Relative to the 0416 Review

The 0416 Gemini review still contained one over-strong causal reading:

- it treated `53.72` too much like a shell-overhead measurement

This 0417 review is stricter.

Its better final message is:

1. the main issue is still semantic rather than a hidden global bug
2. but the best-supported semantic failure is now specifically:
   - **true irregular trunk output cannot be decoded by the current dense-head contract**
3. therefore the immediate next work should target:
   - bridge isolation
   - `y`-output normalization
   - decoder contract change

## Thread Reference

- Reviewer: `gemini-2.5-pro`
- Gemini thread id: `29c4356396ac4266a0339c9ce83740ee`
