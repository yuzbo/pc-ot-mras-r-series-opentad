---
type: experiment
node_id: exp:SOFT_BRIDGE_MAINLINE_0420
title: "Soft bridge full-train mainline reruns: no-warmup vs regwarm10"
date: 2026-04-20
status: running
created_at: 2026-04-20T18:40:00+08:00
updated_at: 2026-04-20T18:40:00+08:00
---

# SOFT_BRIDGE_MAINLINE_0420

## Scope

This page tracks the current highest-priority full-train verification line for the native `y` point-set bridge decoder.

The question is no longer whether the toy diagnosis is real. The toy line already showed:

- `reg-only` can fit
- `cls-only` can fit
- naive joint training fails
- `reg-first warmup -> joint` can recover on toy controls

So the mainline question is now narrower:

1. does the same `reg-first` idea survive on the real full-train THUMOS run
2. does it stay numerically stable after classification is turned back on
3. is it materially better than the same model trained without warmup

## Skill Choice For This Stage

The most suitable skills right now are:

1. `monitor-experiment`
   - because the current bottleneck is no longer code implementation but live training-state verification
2. `research-wiki`
   - because the active priority order has changed and the wiki should reflect the real mainline rather than older reset-queue priorities
3. `analyze-results`
   - but only after the first trustworthy validation mAP or final checkpoint appears

The least useful skill right now is `run-experiment`: new branches should stay paused until one clean full-train read closes.

## Active Mainline Runs

### 1. No-warmup baseline rerun

- Server:
  - `24013`
- Screen:
  - `y_soft_mainline_s1`
- Config:
  - `configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls.py`
- Log:
  - `/root/autodl-tmp/OpenTAD_Back_check/logs/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_rerun2_20260420_172900.log`
- Current runtime snapshot:
  - active through `epoch 18`
  - latest visible train read:
    - `epoch 17 end: Loss=0.6617, cls=0.3984, reg=0.2633`
    - `epoch 18 mid: Loss=0.6729, cls=0.4106, reg=0.2623`
  - cumulative `non-finite gradients detected`: `3`
- Immediate interpretation:
  - this line is not collapsing
  - it has entered a fairly stationary loss band around `0.66 ~ 0.70`
  - so the current no-warmup full-train baseline is operationally valid and worth keeping as the control

### 2. Reg-warmup mainline rerun

- Server:
  - `25876`
- Screen:
  - `y_soft_mainline_s3`
- Config:
  - `configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_regwarm10.py`
- Log:
  - `/root/autodl-tmp/OpenTAD_Back_check/logs/input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_topk1_binarycls_regwarm10_rerun_20260420_172655.log`
- Current runtime snapshot:
  - active through `epoch 20`
  - the schedule switch has already happened:
    - `bridge_cls_loss_weight_effective=0.0` in the early warmup
    - `bridge_cls_loss_weight_effective=1.0` by `epoch 13`
  - latest visible train read:
    - `epoch 18 end: Loss=0.6989, cls=0.4430, reg=0.2559`
    - `epoch 19 end: Loss=0.7003, cls=0.4362, reg=0.2641`
    - `epoch 20 started`
  - cumulative `non-finite gradients detected`: `4`
- Immediate interpretation:
  - the reg-warmup line did not destabilize after the `cls` branch was re-enabled
  - this is the first full-train evidence that the toy `reg-first` diagnosis survives contact with the real training loop at least operationally
  - whether it improves mAP is still open because no meaningful evaluation read has landed yet

## Current Read

What is already supported:

1. the `regwarm10` implementation is working end-to-end on the real training path
2. the `cls` weight switch is actually taking effect
3. post-switch training remains numerically controlled rather than immediately exploding

What is **not** supported yet:

1. that `regwarm10` improves final mAP
2. that it improves over the no-warmup line at equal checkpoints
3. that the remaining failure is solved rather than merely made trainable

## Runtime Refresh

Timestamp:

- `2026-04-20T19:00:00+08:00`

Latest observed state:

### No-warmup baseline rerun

- server:
  - `24013`
- progress:
  - active through about `epoch 27 mid`
- latest visible train read:
  - `epoch 26 end: Loss=0.6390, cls=0.3872, reg=0.2518`
  - `epoch 27 mid: Loss=0.6381, cls=0.3802, reg=0.2579`
- cumulative `non-finite gradients detected`:
  - `5`
- read:
  - still stable in the same `~0.63-0.65` band
  - one more skipped-step event appeared, but the run did not derail

### Reg-warmup mainline rerun

- server:
  - `25876`
- progress:
  - active through about `epoch 29 start`
- latest visible train read:
  - `epoch 27 end: Loss=0.6691, cls=0.4056, reg=0.2636`
  - `epoch 28 end: Loss=0.6390, cls=0.3871, reg=0.2520`
  - `epoch 29 started`
- cumulative `non-finite gradients detected`:
  - `4`
- latest schedule evidence:
  - `bridge_cls_loss_weight_effective=1.0` is still the active post-switch regime
- read:
  - the post-switch phase continues to train normally rather than collapsing
- at the current stage, the main distinction versus no-warmup is still a hypothesis about later validation quality, not a visible train-loss separation

## Runtime Refresh 2

Timestamp:

- `2026-04-20T19:46:00+08:00`

Latest observed state:

### No-warmup baseline rerun

- server:
  - `24013`
- progress:
  - active through about `epoch 39 end`
- latest visible train read:
  - `epoch 38 end: Loss=0.6101, cls=0.3545, reg=0.2556`
  - `epoch 39 mid: Loss=0.5554, cls=0.3224, reg=0.2329`
  - `epoch 39 end: Loss=0.5997, cls=0.3459, reg=0.2538`
- cumulative `non-finite gradients detected`:
  - `5`
- read:
  - the run is still healthy
  - by late-30s epochs the train loss has continued to drop into the high-`0.5x` / low-`0.6x` band

### Reg-warmup mainline rerun

- server:
  - `25876`
- progress:
  - active through about `epoch 39 end`
- latest visible train read:
  - `epoch 37 end: Loss=0.6181, cls=0.3585, reg=0.2596`
  - `epoch 38 end` optimizer step was skipped once due to a non-finite `reg_head.weight` gradient
  - `epoch 39 mid: Loss=0.5700, cls=0.3370, reg=0.2330`
  - `epoch 39 end: Loss=0.6089, cls=0.3556, reg=0.2533`
- cumulative `non-finite gradients detected`:
  - `4`
- latest schedule evidence:
  - `bridge_cls_loss_weight_effective=1.0` remains the active regime
- read:
  - the post-switch line is still stable despite the extra skipped step at `epoch 38`
  - up to this point, train-loss behavior remains very close to the no-warmup control

### Current operational interpretation

- there is still **no visible late-train loss separation** between no-warmup and `regwarm10`
- this does **not** invalidate the warmup hypothesis, but it means the next decisive read is increasingly the first real validation metric rather than additional train-loss snapshots
- both runs remain worth keeping alive because neither has collapsed and neither has yet falsified the optimization-path story

## First Validation Read

Timestamp:

- `2026-04-20T21:46:00+08:00`

Raw table:

| Line | Validation read sequence | Latest confirmed read |
| --- | --- | --- |
| no-warmup | `34.44 -> 35.32 -> 36.59` | `36.59` |
| regwarm10 | `33.35 -> 34.73 -> 35.96 -> 37.00` | `37.00` |

Latest available tIoU breakdown:

### no-warmup

- Average-mAP:
  - `36.59`
- tIoU `0.3 / 0.4 / 0.5 / 0.6 / 0.7`
  - `61.33 / 51.42 / 37.42 / 22.60 / 10.20`

### regwarm10

- Average-mAP:
  - `37.00`
- tIoU `0.3 / 0.4 / 0.5 / 0.6 / 0.7`
  - `61.00 / 51.45 / 38.30 / 23.67 / 10.58`

Interpretation:

1. `regwarm10` is now ahead on the latest confirmed validation read, but only slightly:
   - `37.00 - 36.59 = +0.41`
2. the gain is directionally consistent with the warmup hypothesis, because the latest matched trajectory is:
   - no-warmup: `34.44 -> 35.32 -> 36.59`
   - regwarm10: `34.73 -> 35.96 -> 37.00` in the overlapping later region
3. however, this is still a **small** gain, not a decisive recovery
4. more importantly, both lines remain far below the earlier `y_pointset_softsym_sanity = 44.06 / 44.07`

Current judgment:

- `regwarm10` is no longer just “operationally stable”; it now has the first positive validation evidence
- but the current magnitude is too small to claim that optimization-path repair alone solves the native-`y` bridge failure
- the optimization story is therefore strengthened but not closed

## Immediate Next-Step Plan

1. keep both runs alive to the first trustworthy evaluation read or a much later epoch snapshot
2. prioritize the `25876` reg-warmup line, because the key uncertainty is now post-switch quality rather than pre-switch stability
3. do not launch new method branches before one clean full-train comparison closes
4. only escalate to `regwarm20` or `ampoff` if the `regwarm10` line visibly destabilizes or underperforms in the first real validation read

## Queued Follow-Up

The next diagnosis stage has now been implemented and deployed in waiting mode:

- `exp:SOFT_MULTIGT_DIAG_QUEUE_0420`

These runs do not interrupt the current full-train mainlines. They wait for GPU release and then test:

1. whether the failure already appears at `2 GTs`
2. whether detaching `cls` from the shared trunk rescues `2 GT` or full single-video joint overfit

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
