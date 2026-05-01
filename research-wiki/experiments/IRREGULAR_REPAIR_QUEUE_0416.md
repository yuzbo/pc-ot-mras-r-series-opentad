---
type: experiment
node_id: exp:IRREGULAR_REPAIR_QUEUE_0416
title: "Current repaired irregular-head queue: active runs, queued controls, hypotheses, and result slots"
date: 2026-04-16
status: completed
created_at: 2026-04-16T23:53:30+08:00
updated_at: 2026-04-17T17:00:23+08:00
---

# IRREGULAR_REPAIR_QUEUE_0416

## Scope

This page records the current repaired irregular-head queue after the upstream audit correction.

It is meant to do two things:

1. freeze the exact run order, purpose, and hypothesis of the currently active repair experiments
2. provide a single place to append final results and post-run discussion once each job finishes

## Current Snapshot

Timestamp:

- `2026-04-17T17:00:23+08:00`

Current server state:

- server `24013`: `step0_dense_head_baseline = 53.72`, `step0b_dense_points_soft_sym_repaired = 47.52`, and `y_dense_grid_sanity_check = 13.86`; the server1 repair queue has fully completed
- server `25876`: clean singleton A/B is fully closed; `headv3_oabs_full_x_legacysingleton = 53.48`, `headv3_oabs_full_x_fixsingleton_repaired = 52.88`; the server3 repair queue is idle

Queue order:

- server `24013`
  - completed: `input_random_fixed_50pct_irregular_actionformer_step0_dense_head_baseline`
  - completed: `input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym_repaired`
  - completed: `input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check`
- server `25876`
  - completed: `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_legacysingleton`
  - completed: `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_fixsingleton_repaired`

## Experiment Cards

### 1. `step0_dense_head_baseline`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step0_dense_head_baseline.py`
- Server:
  - `24013`
- Current status:
  - completed
- Purpose:
  - establish the repaired selected-axis dense-head control after fixing the GT-axis mismatch bug
  - verify whether the old `13.79` collapse was caused by broken `remap_gt_to_selected_axis=False` semantics rather than by optimization instability
- Main hypothesis:
  - if this run returns to the historical `~53.72` regime, then the old `13.79` readout was invalid and should be retired
  - if it still collapses, then the remapped selected-axis control itself is unstable and cannot be used as a bridge anchor
- What result would mean:
  - `~53-55`: the repaired remapped control is stable again, but this still does **not** answer the missing no-remap shell-exact question
  - repeated collapse: even the repaired selected-axis control is numerically contaminated
- Result slot:
  - final Avg-mAP: `53.72`
  - delta vs dense random-fixed baseline `63.12`: `-9.40`
  - final discussion: this run cleanly reproduces the historical remapped selected-axis bridge result, so the old `13.79` readout is invalid; however, it should still be interpreted as a **remapped selected-axis control**, not as a true no-remap shell-exact baseline

### 2. `step0b_dense_points_soft_sym_repaired`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym_repaired.py`
- Server:
  - `24013`
- Current status:
  - completed
- Purpose:
  - provide the clean paired control for `dense points + soft assignment + symmetric regression`
  - re-measure soft-assignment overhead against the repaired Step0 baseline after the GT-axis mismatch fix
- Main hypothesis:
  - if this run still stays far below repaired Step0, then cost-based soft assignment is a first-order negative factor even before moving onto the native irregular axis
  - if this run moves close to repaired Step0, then the earlier `47.52` penalty was partly inflated by the dirty control chain
- What result would mean:
  - large gap vs repaired Step0: soft assignment remains a major carrier problem
  - small gap vs repaired Step0: previous soft-assignment penalty was overstated
- Result slot:
  - final Avg-mAP: `47.52`
  - delta vs repaired Step0: `-6.20`
  - final discussion: the repaired rerun lands on exactly the old `47.52` value, so the soft-assignment penalty is not a dirty-control artifact; even on the repaired remapped control, cost-based soft assignment remains strongly negative

### 3. `y_dense_grid_sanity_check`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check.py`
- Server:
  - `24013`
- Current status:
  - completed
- Purpose:
  - test whether the true irregular `y` trunk can be converted back onto a dense grid cleanly enough for a dense head to use it
  - isolate whether part of the `y` weakness is feature-path geometry mismatch rather than only head semantics
- Main hypothesis:
  - if this run stays clearly below the repaired Step0 line, the true irregular trunk is still not producing dense-head-compatible features
  - if it approaches repaired Step0, then the `y` trunk may be more viable than its current head-side results suggest
- What result would mean:
  - near repaired Step0: trunk-side irregular features are not the main blocker
  - far below repaired Step0: the true irregular trunk still carries substantial mismatch
- Result slot:
  - final Avg-mAP: `13.86`
  - delta vs repaired Step0: `-39.86`
  - final discussion: the run finished at essentially the same low level as its mid-train validations (`13.84 -> 13.97 -> 13.86`), so the true irregular `y` trunk is not merely under-optimized; under this dense-grid adapter path it remains severely incompatible with dense-head decoding

### 4. `headv3_oabs_full_x_legacysingleton`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_legacysingleton.py`
- Server:
  - `25876`
- Current status:
  - completed
- Purpose:
  - rerun the legacy singleton handling on a clean config path so that the subsequent singleton-fix comparison becomes attribution-safe
  - replace the old `52.88` reading, which was not a clean config-level ablation
- Main hypothesis:
  - this run should reproduce the old OABS-full region (`~53`) if the baseline branch itself is stable
- What result would mean:
  - `~53`: the legacy branch is stable enough to support a clean A/B singleton comparison
  - much lower / unstable: the old singleton story was mixed with other code-path drift or numerical noise
- Result slot:
  - final Avg-mAP: `53.48`
  - delta vs old `headv3_oabs_full_x = 53.16`: `+0.32`
  - final discussion: clean legacy rerun itself is already above the previously recorded OABS-full and OAA numbers, so it becomes the new comparison anchor for the repaired singleton A/B

### 5. `headv3_oabs_full_x_fixsingleton_repaired`

- Config:
  - `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_fixsingleton_repaired.py`
- Server:
  - `25876`
- Current status:
  - completed
- Purpose:
  - run the explicit config-level singleton fix against the clean legacy rerun
  - finally answer whether the OABS singleton bug matters materially once the comparison is made clean
- Main hypothesis:
  - if the repaired fix is only marginally different from clean legacy, singleton handling is second-order
  - if the repaired fix is clearly higher, the old singleton bug was masking a real gain
- What result would mean:
  - `fix > legacy` by a clear margin: singleton fix is real and useful
  - `fix ≈ legacy`: singleton is not the main bottleneck
  - `fix < legacy`: the repair changes supervision geometry but does not help the current carrier
- Result slot:
  - final Avg-mAP: `52.88`
  - delta vs clean legacy rerun: `-0.60`
  - final discussion: once the singleton A/B is made attribution-safe, the explicit singleton fix is slightly worse than clean legacy; singleton handling is therefore not the main bottleneck on the current carrier

## Current Watchpoints

### A. Numerical stability

- `step0_dense_head_baseline` has already shown sporadic non-finite-gradient events during training
- but the final run still converged back to `53.72`, so those events no longer dominate the causal reading

### B. Evaluation timing

- `headv3_oabs_full_x_legacysingleton` has finished cleanly at `53.48`
- `headv3_oabs_full_x_fixsingleton_repaired` has also finished, and the clean singleton A/B is now closed at `53.48 -> 52.88`
- `step0_dense_head_baseline`, `step0b_dense_points_soft_sym_repaired`, and `y_dense_grid_sanity_check` have all finished; the repair queue is now fully closed

### C. What this queue is really for

This queue is primarily a **repair / attribution** queue, not a new-method queue.

Its first job is to answer:

- was the old shell-overhead story real?
- was the old singleton-fix story real?
- how much of the earlier Step0 / Step0b reading was caused by dirty controls?

So a "good" outcome from this queue is not necessarily a new best score.  
A good outcome is a cleaner causal map.

## Post-Completion Update Template

When each run finishes, append:

1. final `Average-mAP`
2. delta vs the relevant reference
3. whether the run is causally trustworthy
4. one short conclusion sentence

Recommended update block format:

```markdown
### Update: <experiment_name>

- final Avg-mAP: `XX.XX`
- reference: `YY.YY`
- delta: `+/-ZZ.ZZ`
- trustworthy for causal interpretation: `yes/no/partial`
- conclusion:
  - ...
```

## Provisional Expected Discussion After Completion

The most important post-run discussion questions will be:

1. Does repaired Step0 remain far below `63.12`, or not?
2. Does repaired Step0b still show a large soft-assignment penalty, or not?
3. Does the clean singleton A/B show any real gain?
4. Does the `y_dense_grid_sanity_check` suggest that the true irregular trunk is still incompatible with dense-style detection?

## Result Updates

### Update: `headv3_oabs_full_x_legacysingleton`

- final Avg-mAP: `53.48`
- reference: old `headv3_oabs_full_x = 53.16`
- delta: `+0.32`
- trustworthy for causal interpretation: `yes`
- conclusion:
  - the clean legacy rerun is already stronger than the previously recorded `OABS-full = 53.16`
  - it is also slightly above the previously best completed OAA variant `53.20`
  - therefore the old `52.88` singleton-fix reading should not be used anymore
  - the only clean singleton comparison that matters now is `fixsingleton_repaired` versus clean legacy `53.48`

### Update: `step0_dense_head_baseline`

- final Avg-mAP: `53.72`
- reference: dense random-fixed baseline `63.12`
- delta: `-9.40`
- trustworthy for causal interpretation: `partial`
- conclusion:
  - the old `13.79` collapse was indeed caused by the GT-axis mismatch bug
  - this repaired run restores the stable remapped selected-axis bridge result `53.72`
  - however, this still does not answer the missing no-remap shell-exact question, so `53.72` should not be used as final shell-overhead proof

### Update: `step0b_dense_points_soft_sym_repaired`

- final Avg-mAP: `47.52`
- reference: repaired `step0_dense_head_baseline = 53.72`
- delta: `-6.20`
- trustworthy for causal interpretation: `yes` on the remapped selected-axis control family
- conclusion:
  - the repaired rerun exactly matches the old `47.52`
  - therefore the soft-assignment penalty is real within this control family and is not an artifact of the broken old Step0 run

### Update: `headv3_oabs_full_x_fixsingleton_repaired`

- final Avg-mAP: `52.88`
- reference: clean legacy rerun `53.48`
- delta: `-0.60`
- trustworthy for causal interpretation: `yes`
- conclusion:
  - the clean singleton A/B is now closed and slightly favors legacy singleton handling
  - the singleton fix changes supervision geometry, but it does not improve the current `x`-line carrier

### Update: `y_dense_grid_sanity_check`

- final Avg-mAP: `13.86`
- reference: repaired `step0_dense_head_baseline = 53.72`
- delta: `-39.86`
- trustworthy for causal interpretation: `yes`
- conclusion:
  - the true irregular `y` trunk does not recover late; it finishes at the same collapsed level already seen at epochs `39` and `44`
  - the dense-grid adapter does not rescue it
  - this is now strong evidence that the true irregular trunk is severely incompatible with dense-head decoding under the current carrier

## Second-Round Readout

1. The repaired queue has now invalidated the pathological old `step0_dense_head_baseline = 13.79`; the correct repaired selected-axis dense-head control is back at `53.72`.
2. This does **not** overturn the upstream audit. The repaired `53.72` remains a remapped selected-axis control, not a true no-remap shell-exact baseline.
3. `step0b_dense_points_soft_sym_repaired = 47.52` lands `6.20 mAP` below repaired Step0, so soft assignment remains a first-order negative factor even after fixing the control bug.
4. The clean singleton story is now settled in the opposite direction from the old optimistic reading: `53.48 -> 52.88`, so singleton repair is not a useful lever on the present carrier.
5. `y_dense_grid_sanity_check` has now closed at `13.86`, almost identical to its earlier `13.84 / 13.97` validations. This makes the strongest queue-level conclusion clear: the true irregular `y` trunk is severely incompatible with dense-head decoding under the current dense-grid adapter path.
6. The repair queue therefore ends with a clean split:
   - remapped selected-axis dense-like controls can still reach `53.72`
   - the true irregular trunk collapses to `13.86`
   - so the main unresolved issue is not minor head tuning, but the representational / semantic mismatch between the true irregular trunk and dense-head decoding

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
