---
type: experiment
node_id: exp:RESET_QUEUE_0418_STATUS
title: "0418 reset queue status after first closure: AMP control, stupid completion anchor, and repaired overfit redeploy"
date: 2026-04-18
status: running
created_at: 2026-04-18T11:11:49+08:00
updated_at: 2026-04-19T14:45:00+08:00
---

# RESET_QUEUE_0418_STATUS

## Scope

This page freezes the first real closure of the 0418 reset plan.

It answers three immediate questions:

1. did fp32 rescue the native `y` point decoder?
2. did the stupid dense-semantics baseline beat the current irregular-head plateau?
3. were the first tiny overfit diagnostics valid enough to read causally?

## Current Snapshot

Timestamp:

- `2026-04-19T14:45:00+08:00`

Server state:

- server `25876`
  - completed: `G0418-R001 = y_pointset_softsym_sanity_ampoff = 44.07`
  - completed but invalid: `G0418-R004/R005` old overfit ladder
  - stopped as still impure: `G0418-R009/R010` repaired overfit ladder
  - completed: `G0418-R011 = true fixed-sample overfit single-video`, final late loss `0.6693`
  - completed: `G0418-R012 = true fixed-sample overfit single-instance`, final late loss `0.6534`
  - running: branch-decoupled overfit ladder `G0418-R013 -> R016`
- server `24013`
  - completed: `G0418-R002 = completion_nearest_dense_baseline = 54.03`
  - interrupted: `G0418-R003 = completion_linear_dense_baseline`, latest `52.56`
  - stopped: `G0418-R006` AMP-on oracle diagnostic
  - currently idle while native `y` overfit branch diagnosis takes priority

## Run Table

| Run | System | Status | Main read |
|-----|--------|--------|-----------|
| `G0418-R001` | `y_pointset_softsym_sanity_ampoff` | done | `44.07 / 47.14@0.5 / 16.11@0.7`; no meaningful gain over AMP `44.06` |
| `G0418-R002` | nearest completion + unchanged dense detector | done | `54.03 / 57.08@0.5 / 29.32@0.7`; modestly above irregular-head mainline, far below `60+` hope |
| `G0418-R003` | linear completion + unchanged dense detector | interrupted | latest epoch-50 eval `52.56 / 55.23@0.5 / 28.20@0.7`; currently behind nearest |
| `G0418-R004` | tiny overfit single-video | done-invalid | inherited a 200-epoch cosine schedule with one iter/epoch, so LR collapsed to `~1e-8` |
| `G0418-R005` | tiny overfit single-instance | done-invalid | same invalid schedule issue as `R004` |
| `G0418-R006` | oracle action+boundary support diagnostic | stopped | relaunched run again showed early `non-finite gradients`, so we stopped it after Gemini review rather than treating it as a trustworthy read |
| `G0418-R008` | completion oracle upper bound (`amp-off`) | queued | not started yet |
| `G0418-R007` | learned minimal completion (`amp-off`) | queued | not started yet |
| `G0418-R009` | repaired tiny overfit single-video | stopped | no longer treated as valid because `random_trunc` still changed the clip every epoch |
| `G0418-R010` | repaired tiny overfit single-instance | stopped | same impurity as `R009`; superseded before completion |
| `G0418-R011` | true fixed-sample overfit single-video | done | deterministic random-fixed 50% sample still failed to memorize; final line `[499] Loss=0.6693 cls_loss=0.0538 reg_loss=0.6156` |
| `G0418-R012` | true fixed-sample overfit single-instance | done | even the clean single-instance variant failed to memorize; final line `[499] Loss=0.6534 cls_loss=0.0663 reg_loss=0.5870` |
| `G0418-R013` | true fixed-sample single-video `cls-only` | running | first branch-decoupled control; `reg_loss_weight=0`, periodic runtime debug every 25 epochs |
| `G0418-R014` | true fixed-sample single-video `reg-only` | queued | follows `R013` |
| `G0418-R015` | true fixed-sample single-instance `cls-only` | queued | follows `R014` |
| `G0418-R016` | true fixed-sample single-instance `reg-only` | queued | follows `R015` |

## Main Conclusions

### 1. AMP is not the main explanation for the native `y` ceiling

- `R001 = 44.07`
- previous AMP run = `44.06`

So the current point-wise native irregular decoder is structurally weak; fp32 alone does not lift it into the old `~49` regime.

### 2. Dense-semantics preservation still helps, but not enough

- `R002 = 54.03`
- best repaired irregular-head carrier was around `53.48`

The stupid nearest-completion baseline does beat the irregular-head region, but only by a small margin. This supports the contract-mismatch story, but it does not yet establish completion as a decisive route to `60+`.

### 3. The first overfit ladder was not a valid diagnostic

`R004/R005` did not overfit, but that negative read cannot be trusted because the run inherited the wrong schedule for a one-sample regime.

The correct next action is not to interpret those runs; it is to rerun them under a repaired overfit setup.

### 4. The repaired true-overfit read is now decisive

`R011/R012` remove the remaining confounds:

- deterministic fixed truncation window
- deterministic 50% sparse mask on the selected clip
- checkpoint saving disabled
- repaired schedule and optimizer

Yet both runs still fail to fit even one fixed sparse sample.

This is now the strongest evidence that the current native `y` decoder path is not merely under-tuned. Its shared decoder / assignment / regression contract is structurally incapable of memorizing the target under the current formulation.

## Immediate Next Actions

1. read `R013-R016` to separate `cls-only` vs `reg-only` memorization ability
2. use the periodic runtime debug snapshots to inspect candidate counts, positive counts, per-level support, and regression target scale on the fixed sample
3. only after branch-decoupled overfit closes, decide whether the next reset should simplify assignment, simplify regression, or replace the native `y` decoder contract entirely

## Why This Matters

The 0418 reset plan is now already more disciplined:

- the native `y` point decoder is not being artificially capped by AMP
- the dense-semantics route is directionally better than direct irregular-head decoding, but the current simple completion anchor is still far from paper-ready
- the decisive evidence is now the failure of the *true* deterministic overfit diagnostic
- the next highest-value evidence is the branch-decoupled overfit ladder, not another paper-scale ablation queue
