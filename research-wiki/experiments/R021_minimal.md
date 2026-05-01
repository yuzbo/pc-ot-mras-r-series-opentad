---
type: experiment
node_id: exp:R021
title: "Gumbel + Soft Gating minimal control"
date: 2026-04-09
status: completed
config: v3_gumbel_softgate_minimal.py
---

# R021: Gumbel + Soft Gating minimal control

## Purpose

Answer the cleanest version of the question:

- can learned frame selection beat the `uniform 50%` baseline if we only replace the sampler policy and keep everything else minimal?

## Key Config

- `mode=gumbel`
- `soft_gating=True`
- `lambda_scorer=0.05`
- no extra aux loss
- no coarse detection loss

## Original Result

- `Avg-mAP = 53.98`
- `mAP@0.5 = 57.04`

## Original Conclusion

The minimal soft-gating line was already clearly below the `uniform 50%` reference and therefore did not support the claim that end-to-end learned soft gating could beat the fixed baseline in this setup.

## 2026-04-23 bugfix rerun note

A new rerun with the latest implementation fixes is being tracked separately in:

- `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423`

Current bugfix-rerun reads from `v3_gumbel_softgate_minimal_bugfix0420`:

- first eval: `52.43`
- second eval: `53.17`
- third eval: `53.28`
- fourth eval: `53.02`
- fifth eval: `53.75`
- sixth eval: `54.32`
- seventh eval: `54.07`
- eighth eval: `53.87`
- ninth eval: `53.73`
- tenth eval: `54.22`

This does not materially overturn the original `R021 = 53.98` conclusion.

The current supported reading is:

- the rerun is cleaner and more trustworthy
- the rerun is slightly stronger than the early four-read snapshot and briefly peaks at `54.32`
- the completed rerun closes at `54.22`, slightly above the old `R021 = 53.98`
- the performance regime is still the same broad low-to-mid `50s` band rather than a true recovery toward the fixed `uniform 50%` baseline
