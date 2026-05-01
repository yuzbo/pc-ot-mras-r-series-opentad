---
type: experiment
node_id: exp:IRREGULAR_ROOTCAUSE_UPDATE_0416
title: "Root-cause update after Step0b and OABS fixsingleton"
date: 2026-04-16
status: synthesized
created_at: 2026-04-16T10:05:47+08:00
updated_at: 2026-04-16T16:55:00+08:00
---

# IRREGULAR_ROOTCAUSE_UPDATE_0416

## 2026-04-16 upstream audit correction

This page is no longer the authoritative source for shell-overhead attribution.

The later upstream comparison audit showed three important corrections:

- `step0_dense_points_hard_shell_exact = 53.72` is **not** a valid shell-exact baseline because it inherits a remap config
- `headv3_oabs_full_x_fixsingleton = 52.88` is **not** a clean singleton ablation because the config only changed `work_dir`
- the `x` line should be read as **dense-trunk + irregular-head**, not as evidence for a full irregular-aware trunk

For the corrected interpretation, use:

- `exp:IRREGULAR_UPSTREAM_AUDIT_0416`

## Scope

This page records the two newly completed results that materially changed the diagnosis of the current irregular-head line:

- `step0b_dense_points_soft_sym = 47.52`
- `headv3_oabs_full_x_fixsingleton = 52.88`

For the root-cause update, it also folds in the already completed but previously unrecorded `headv3_oabs_full_oaa_x = 53.20`, because the new ranking and next-step decision depend on it.

## Newly Integrated Results

Reference anchors:

- dense baseline on the same random-fixed 50% input: `63.12`
- previous shell-exact bridge control: `step0_dense_points_hard_shell_exact = 53.72`
- previous best completed HeadV3 result before OABS/OAA: `headv3_x = 52.60`

| Experiment | Avg-mAP | Delta vs reference | Direct readout |
|---|---:|---:|---|
| `step0_dense_points_hard_shell_exact` | **53.72** | `-9.40` vs `63.12` | dense-style semantics inside the current irregular detector shell |
| `step0b_dense_points_soft_sym` | **47.52** | `-6.20` vs `53.72` | soft assignment already hurts even under dense points |
| `headv3_oabs_full_x` | **53.16** | `+0.56` vs `52.60` | full observation-aware targets help a little |
| `headv3_oabs_full_oaa_x` | **53.20** | `+0.04` vs `53.16` | OAA is effectively flat relative to OABS-full |
| `headv3_oabs_full_x_fixsingleton` | **52.88** | `-0.28` vs `53.16` | singleton bug was real but not the main bottleneck |

## What These Results Close

### 1. Soft assignment is no longer an open question

The earlier 0415B page left one key uncertainty open: whether soft assignment was only harmful once combined with native irregular points, or whether it was already weak by itself.

`step0b_dense_points_soft_sym = 47.52` closes this directly:

- `step0_shell_exact = 53.72`
- `step0b_dense_points_soft_sym = 47.52`
- delta: `-6.20`

This means cost-based soft assignment is already a major negative factor even before the detector is moved onto native irregular points.

### 2. The OABS singleton bug was real, but not decisive

The singleton regression bug in OABS existed and needed to be fixed. But after the fix:

- `headv3_oabs_full_x = 53.16`
- `headv3_oabs_full_x_fixsingleton = 52.88`

The result did not improve. So the bug was not the reason the line plateaued around the low `53` range.

### 3. OAA is not a main breakthrough

- `headv3_oabs_full_x = 53.16`
- `headv3_oabs_full_oaa_x = 53.20`
- delta: `+0.04`

Observation-aware assignment/gating is not useless, but at this stage it behaves like a tiny local tweak, not like the mechanism that will recover the remaining dense-baseline gap.

## Updated Root-Cause Ranking

### 1. Detector shell / path / contract overhead

The strongest direct evidence is still:

- dense baseline: `63.12`
- `step0_shell_exact`: `53.72`

So even when dense-style semantics are injected into the current irregular detector shell, performance is still capped almost `10 mAP` below the real dense baseline. This is now the clearest structural ceiling in the whole line.

### 2. Cost-based soft assignment as the core carrier

`step0b = 47.52` proves that soft assignment is not just a rescue mechanism on the native irregular axis; it is also a direct negative factor under dense points.

The line now has two confirmed structural losses:

- `63.12 -> 53.72`: shell/path overhead
- `53.72 -> 47.52`: soft-assignment overhead under otherwise dense points

These are the two biggest directly measured negative factors.

### 3. Native irregular hard ownership is also unusable

The bridge table still matters:

- `step1_irregular_points_hard_sym = 37.10`
- `step2_irregular_points_hard_asym = 34.41`

So the answer is not "just go back to hard assignment on native irregular points." Naive hard semantics collapse badly on the native irregular axis.

This is the key constraint for the next design:

- soft assignment is too lossy
- native irregular hard assignment is too brittle

The new carrier has to avoid both.

### 4. OABS / OAA are second-order gains only

The observation-aware line is still the best-performing branch *inside the current irregular shell*:

- `headv3_x = 52.60`
- `headv3_oabs_full_x = 53.16`
- `headv3_oabs_full_oaa_x = 53.20`

But these gains are only around `+0.56 ~ +0.60`. They are too small to attack the main residual once the new shell and assignment evidence is included.

### 5. No longer primary suspects

The following directions now have direct negative or near-flat evidence and should not be treated as primary explanations:

- OABS singleton bug
- asymmetric `log1p` regression
- final predictor `k=3`
- frozen-backbone time embedding
- post-hoc boundary-aware inference

## Updated Next Model Direction

The correct pivot is now stronger than "keep improving the irregular shell."

### Main decision

- Stop treating the current irregular detector shell as the long-term carrier.
- Stop using cost-based soft assignment as the main semantics carrier.
- Move to **sparse-to-dense completion in front of an otherwise unchanged dense detector path**.

### Why this pivot is now justified

1. `step0_shell_exact = 53.72` suggests the current irregular shell has a hard ceiling in the low `50s`.
2. `step0b = 47.52` shows soft assignment adds another large penalty inside that shell.
3. `step1 / step2` show that native irregular hard semantics are not a clean fallback.

So the clean path is:

- restore dense semantics before detection
- then let a proven dense detector do the actual TAD job

instead of continuing to rewrite assignment / regression / decoding inside the irregular shell.

## Minimal Next Two-Step Plan

### Step A. Oracle completion upper bound

Use sparse input, fill missing positions with the true dense features, and feed the completed dense sequence into the original dense detector path.

This answers the most important next question:

- if the detector sees a perfectly completed dense sequence, can it return to the `63`-range?

### Step B. Minimal learned completion module

Build a small temporal completion module that takes:

- sparse features
- observation mask

and outputs a densified feature sequence for the unchanged dense detector.

This is the cleanest implementation of the new mainline:

- handle sparsity by completion
- handle detection with the proven dense detector

## Deprioritized Directions

- more soft-assignment variants or cost-function tuning
- more OABS/OAA micro-ablations inside the current shell
- asymmetric regression as a mainline
- frozen-backbone time embedding
- post-hoc boundary-aware inference

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
