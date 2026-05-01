---
type: review
node_id: review:GEMINI_DIRECTION_0418
title: "Gemini reset review after 0417 recovery results and current middle-state diagnosis"
date: 2026-04-18
status: completed
created_at: 2026-04-18T00:24:39+08:00
updated_at: 2026-04-18T00:24:39+08:00
---

# GEMINI_DIRECTION_REVIEW_0418

## Scope

This review was run after three facts became simultaneously clear:

- dense random-fixed 50% baseline remains strong at `63.12`
- current best dense-trunk + irregular-head carrier still stalls near `53.48`
- true irregular-trunk recovery remains split between:
  - dense-grid decoding collapse: `13.86` and `10.34`
  - point-wise irregular decoding partial recovery: `44.06`

The review prompt deliberately asked Gemini to challenge the current narrative rather than merely summarize it.

## Evidence Sent to Gemini

- dense random-fixed 50%: `63.12`
- uniform 50%: `65.09`
- old irregular detector: `39.04`
- native remap failure: `4.13`
- `HeadV2_x = 51.04`
- `HeadV3_x = 52.60`
- best repaired `x`-carrier region: about `53.48`
- repaired `step0_dense_head_baseline = 53.72` with the explicit warning that it is only a remapped selected-axis control
- repaired `step0b_dense_points_soft_sym = 47.52`
- `denseexact = 51.04 = HeadV2_x`
- `y_dense_grid_sanity_check = 13.86`
- `y_dense_grid_sanity_check_norm_ampoff = 10.34`
- `y_pointset_softsym_sanity = 44.06`
- current negative directions:
  - frozen backbone time embedding hurts
  - direct boundary-aware inference hurts
  - final predictor `k=3` is worse than `k=1`
  - `topk1` is worse than `topk9`

## Gemini's Main Criticism

Gemini's tone was intentionally severe. The most useful content was not the rhetoric but the reset in priority.

Its strongest criticism was:

1. we are still over-narrating around too many weak controls
2. we do not yet have a sufficiently stupid but clean dense-semantics-preserving anchor
3. too much time has been spent on secondary ablations while the main contract failure remains unresolved

The thread did **not** claim that the entire project is wrong. It claimed that the current experimental ladder is too complex and too easy to over-interpret.

## What Gemini Got Right

### 1. Sparse random-fixed input is not the main problem

This was the one point Gemini accepted without pushback:

- `63.12` vs `65.09` means the sparse input protocol itself is not the main bottleneck

### 2. Middle-state hybrids are the current failure pattern

Gemini agreed with the practical substance of the current diagnosis:

- the current `x` line is a middle-state hybrid
- the current `y_dense_grid` path is also a middle-state contract mismatch

The cleaner rephrasing is:

- we are currently losing dense semantics without yet having a genuinely strong native irregular decoder

### 3. Dense-grid normalization should be deprioritized

Gemini treated:

- `13.86`
- `10.34`

as enough evidence that simple normalization is not the rescue path.

### 4. The next experiments should become simpler and more causal

This is the biggest actionable improvement from the review.

## Where Gemini Overreached

Gemini used language such as:

- "you have zero viable strategic directions"
- "kill them all"

This is too strong.

It is more accurate to say:

- the current evidence still supports two *hypotheses*:
  - preserve dense semantics and inject sparse information
  - build a genuinely strong native irregular decoder
- but neither path yet has a paper-worthy winning result

So the real takeaway is not to abandon the project; it is to reset the diagnostic ladder.

## Corrected 0418 Root-Cause Ranking

### 1. Detector contract mismatch is still the top supported failure

Why:

- random-fixed sparse input itself is strong under dense detection
- current irregular-head mainline stalls in the low `50s`
- true irregular trunk collapses under dense-grid decoding

### 2. Lack of a stupid but causally clean anchor is still hurting interpretation

This is Gemini's strongest valid process criticism.

Why:

- repaired `53.72` is useful but still not a no-remap shell-exact baseline
- later conclusions are too easy to over-read without a simpler anchor

### 3. Current native irregular decoder remains too weak

Why:

- `44.06` is a major recovery over `13.86`
- but it still trails the established `y`-line head results around `49`

### 4. AMP instability is real but secondary

Why:

- it clearly contaminates some `y`-line reads
- but it cannot by itself explain the full gap from `63.12`

### 5. Secondary `x`-line micro-optimizations are not the main battlefield

Why:

- they move the low-50s carrier only marginally

## New 0418 Priority Order

### Priority 1. Finish the pending fp32 `y_pointset` control

Why:

- it cleanly answers whether the current `44.06` ceiling is mostly numeric or structural

### Priority 2. Build the stupidest possible dense-semantics-preserving baseline

Concrete version:

- sparse-to-dense nearest-neighbor / zero-order hold completion
- sparse-to-dense linear interpolation
- unchanged original dense detector

Why:

- this creates the new hard anchor for the entire project

### Priority 3. Simplify the native `y` line before adding any more modules

Concrete version:

- tiny overfit ladder
- stripped-down decoder
- instrumentation-first debugging

Why:

- if the minimal decoder cannot overfit, complexity is hiding a deeper problem

### Priority 4. Run oracle interface diagnostics

Why:

- to separate head capacity from support-finding / pooling interface weakness

### Priority 5. Keep learned completion as the mainline only if Priority 2 is strong

Why:

- completion is still attractive, but it should now be justified by a strong simple baseline, not by over-reading ambiguous bridge numbers

## Directions Explicitly Deprioritized

1. more `x`-line OABS / OAA micro-variants
2. more dense-grid normalization
3. more boundary-aware inference variants
4. more frozen-backbone time-embedding variants
5. more rhetoric around `53.72` as if it were a clean shell-overhead number

## Deliverables Triggered by This Review

This review directly triggers:

- `refine-logs/EXPERIMENT_PLAN_0418_RESET.md`
- `refine-logs/EXPERIMENT_TRACKER_0418_RESET.md`

Those files convert the critique into a concrete run order.

## Thread Reference

- Reviewer: `gemini-2.5-pro`
- Gemini thread id: `6292733486f04885afff08174b34da2f`
