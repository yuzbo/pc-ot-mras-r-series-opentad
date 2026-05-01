---
type: experiment
node_id: exp:IRREGULAR_COMPLETION_UPDATE_0415B
title: "Latest completed irregular-head batch: OABS full/visible + Step0 shell-exact"
date: 2026-04-15
status: completed
created_at: 2026-04-15T17:20:46+08:00
updated_at: 2026-04-15T17:20:46+08:00
---

# IRREGULAR_COMPLETION_UPDATE_0415B

## Scope

本页只记录 `IRREGULAR_MODEL_AUDIT_0415` 之后新增完成、且已经确认 `Training Over` 的一小批关键实验，用来更新当前 irregular-head 主线判断。

本批次只纳入：

- `step0_dense_points_hard_shell_exact`
- `headv3_oabs_visible_x`
- `headv3_oabs_full_x`

同时显式说明未完成项：

- `headv3_oabs_full_oaa_x`：已部署运行，但尚无完整结果
- `step0b_dense_points_soft_sym`：`Epoch 0` 触发 `IndexError`，当前不构成可信结果

## Completed Results

| Experiment | Config | Avg-mAP | Delta vs `63.12` | Direct readout |
|---|---|---:|---:|---|
| `step0_dense_points_hard_shell_exact` | `input_random_fixed_50pct_irregular_actionformer_step0_dense_points_hard_shell_exact.py` | **53.72** | -9.40 | dense-style semantics in the current irregular detector shell/path |
| `headv3_oabs_visible_x` | `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_visible_x.py` | **51.49** | -11.63 | visible-span-only OABS variant |
| `headv3_oabs_full_x` | `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x.py` | **53.16** | -9.96 | full OABS = visible span + boundary confidence + uncertainty weighting |

Reference lines:

- dense baseline on the same random-fixed 50% input: `63.12`
- previous best irregular-head before this batch: `headv3_x = 52.60`

## Immediate Comparisons

### 1. Step0 shell-exact vs dense baseline

- `63.12 -> 53.72` leaves a stable `9.40 mAP` residual
- This result now matches the earlier `step0_densehead_remap = 53.72`

Stricter reading:

- The large residual is **not** a remap accident
- Once dense-style semantics are moved into the current irregular detector shell/path/contract, performance drops to the low `53` range
- This confirms a real **structural overhead** exists before any further irregular-head innovation is even counted

What this does **not** prove yet:

- It does not isolate whether the entire `9.40` is shell-only overhead
- It only shows the loss is already present at the `dense semantics inside current irregular detector path` level

### 2. OABS-visible vs HeadV3-X

- `headv3_x = 52.60`
- `headv3_oabs_visible_x = 51.49`
- delta: `-1.11`

Reading:

- Simply shrinking classification/regression targets to the visible span is not enough
- A visible-only target introduces a short-span bias that hurts overall detection quality

### 3. OABS-full vs HeadV3-X

- `headv3_x = 52.60`
- `headv3_oabs_full_x = 53.16`
- delta: `+0.56`

Reading:

- Observation-aware target design is currently the only newly completed head-side direction in this batch that produces a clean positive gain
- The gain does **not** come from visible-span shrinkage alone
- The useful part is the full package:
  - visible-span handling
  - boundary confidence terms `c_start / c_end`
  - uncertainty-aware weighting

## Updated Structural Judgment

This batch tightens the story in four ways.

### A. There is a real dense-to-irregular structural overhead

`step0_shell_exact = 53.72` means the current irregular detector path itself carries a substantial residual relative to the dense baseline `63.12`.

The key implication is:

- the remaining gap is not only about “making HeadV3 slightly better”
- there is a larger detector-path / semantics-contract overhead that must be explained

### B. OABS is a real positive direction, but only in full form

The pair

- `OABS-visible = 51.49`
- `OABS-full = 53.16`

shows that observation-aware supervision is not equivalent to “just use visible spans”.

The currently supported claim is narrower and stronger:

- **full observation-aware target design helps**
- **visible-only target shrinkage does not**

### C. Current best irregular-head result is now 53.16

The strongest completed irregular-head result is no longer `headv3_x = 52.60`, but:

- `headv3_oabs_full_x = 53.16`

This makes `OABS-full` the new head-side mainline reference point.

### D. One key diagnosis gap remains open

`step0b_dense_points_soft_sym` was supposed to answer how much loss comes from soft assignment itself under dense points, but it currently crashes in `irregular_actionformer_bridge_head.py`.

So this batch does **not** yet settle:

- whether soft assignment itself is weak under dense points
- or whether the main loss only appears once soft assignment is combined with native irregular points / contracts

## What This Batch Supports

Supported:

1. Observation-aware supervision remains the strongest active positive direction on the irregular-head line.
2. The dense baseline gap contains a real pre-head or shell-level structural overhead, not just a weak OABS/HeadV3 design.
3. Visible-only supervision shrinkage is insufficient and can be harmful.

Not yet supported:

1. That the remaining `9.40` gap is purely shell overhead rather than a broader detector-path contract overhead.
2. That soft assignment itself is good or bad in the dense-points setting; `step0b` must be fixed first.
3. That OAA helps; it is deployed but still running.

## Current Priority After This Batch

1. Wait for `headv3_oabs_full_oaa_x`
2. Fix and rerun `step0b_dense_points_soft_sym`
3. Use `headv3_oabs_full_x = 53.16` as the new head-side baseline for any further observation-aware extension

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
