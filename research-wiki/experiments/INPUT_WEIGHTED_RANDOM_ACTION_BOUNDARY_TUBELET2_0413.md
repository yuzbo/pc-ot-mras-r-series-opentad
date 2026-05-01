---
type: experiment
node_id: exp:INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413
title: "OpenTAD_Back input weighted-random action+boundary tubelet2 50%"
config: configs/adatad/thumos/input_weighted_random_action_boundary_tubelet2_50pct.py
server: Server 1 (24013)
date: 2026-04-13
status: completed
created_at: 2026-04-13T22:55:59+08:00
updated_at: 2026-04-13T22:55:59+08:00
---

# INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413

## Purpose

Check whether switching the sampling unit from frame-level selection to 2-frame tubelet selection can rescue the previously weak `action+boundary` weighted-random input strategy under the same 50% budget.

## Setup

- config: [input_weighted_random_action_boundary_tubelet2_50pct.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/configs/adatad/thumos/input_weighted_random_action_boundary_tubelet2_50pct.py)
- sampling unit: contiguous 2-frame tubelets
- selection rule: weighted random without replacement
- semantic bias: action + boundary higher than background
- budget: total kept frames remain 50%
- detector: unchanged dense AdaTAD / ActionFormer-style detector

## Result

| Experiment | Avg-mAP | Delta |
|---|---:|---:|
| input random-fixed 50% | 63.12 | +4.58 |
| input oracle boundary-dense 50% | 66.01 | +7.47 |
| input oracle action+boundary-dense 50% | 62.04 | +3.50 |
| input weighted-random action+boundary 50% | 59.24 | +0.70 |
| **input weighted-random action+boundary tubelet2 50%** | **58.54** | 0.00 |

## Interpretation

1. `tubelet2` did not rescue the `action+boundary` weighted-random idea.
2. It is even slightly worse than the frame-level version (`58.54` vs `59.24`, `-0.70`).
3. But this does **not** yet prove that tubelet semantics are irrelevant.
4. There are still two live explanations:
   - budget allocation is wrong: too much density is spent inside the action interior
   - tubelet granularity hurts boundary precision: boundary frames may be bundled together with nearby non-boundary frames, weakening localization
5. The cleanest discriminator is still `input_oracle_boundary_dense_tubelet2_50pct`:
   - if boundary-only tubelet2 stays near `66.01`, the main problem is budget allocation
   - if boundary-only tubelet2 also drops clearly, tubelet granularity itself is part of the loss

## What This Supports

- supports `claim:C11`
- weakens the hypothesis that a simple shift to 2-frame tubelet selection is enough to fix the action+boundary weighted-random line
- keeps open the more precise question of whether tubelet granularity itself damages boundary localization

## Logs

- remote log: `/root/autodl-tmp/OpenTAD_Back_check/logs/input_weighted_random_action_boundary_tubelet2_50pct_queue_20260413_170641.log`

## Related

- exp:INPUT_RANDOMFIX_0410
- exp:INPUT_ORACLE_BOUNDARY_DENSE_0411
- exp:INPUT_ORACLE_ACTION_BOUNDARY_DENSE_0411
- claim:C11

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
