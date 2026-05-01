---
type: review
node_id: review:GEMINI_DIRECTION_0416
title: "Gemini direction review after Step0b and OABS fixsingleton"
date: 2026-04-16
status: completed
created_at: 2026-04-16T10:05:47+08:00
updated_at: 2026-04-16T10:05:47+08:00
---

# GEMINI_DIRECTION_REVIEW_0416

## Scope

This review was run after two previously open questions were closed:

- `step0b_dense_points_soft_sym = 47.52`
- `headv3_oabs_full_x_fixsingleton = 52.88`

The goal was to answer three things more sharply than the 0415 review:

1. Are the current non-improvements more likely caused by hidden code / setting mistakes, or by wrong model semantics?
2. After `step0b`, what is the updated root-cause ranking?
3. Should the next pivot stay inside the current irregular detector shell, or move to sparse-to-dense completion ahead of a normal dense detector?

## Main Evidence Sent to Gemini

- dense baseline: `63.12`
- boundary oracle dense: `66.01`
- `headv2_x = 51.04`
- `headv2_denseexact = 51.04`
- `headv3_x = 52.60`
- `step0_shell_exact = 53.72`
- `step0b_dense_points_soft_sym = 47.52`
- `step1_irregular_points_hard_sym = 37.10`
- `step2_irregular_points_hard_asym = 34.41`
- `step3_irregular_points_soft_sym = 49.86`
- `step4_irregular_points_soft_asym = 49.79`
- `headv3_oabs_visible_x = 51.49`
- `headv3_oabs_full_x = 53.16`
- `headv3_oabs_full_oaa_x = 53.20`
- `headv3_oabs_full_x_fixsingleton = 52.88`

## Gemini Main Judgment

Gemini's answer was explicit:

- this is **overwhelmingly a model-semantic failure**
- it no longer looks like a hidden implementation bug is the main reason the line fails to improve

The supporting reasoning was:

1. `step0_shell_exact = 53.72` already exposes a large shell/path overhead relative to the real dense baseline `63.12`
2. `step0b = 47.52` proves soft assignment is itself a major negative factor even under dense points
3. the singleton bug existed, but the fixed result `52.88` did not improve, so it was not the main bottleneck

## Gemini's Updated Root-Cause Ranking

### 1. Detector shell / contract inefficiency

Gemini treated

- `63.12 -> 53.72`

as the clearest architectural ceiling currently measured.

### 2. Hard assignment failure on native irregular geometry

Gemini pointed to

- `53.72 -> 37.10`

as the strongest evidence that naive hard semantics do not survive on the native irregular axis.

### 3. Soft assignment as a lossy crutch

Gemini treated

- `53.72 -> 47.52`

as direct proof that soft assignment is not a clean fix, but a high-cost workaround.

## Stronger Pivot Recommended by Gemini

Gemini explicitly chose a side:

- **do not keep the current irregular detector shell as the main carrier**
- **move to sparse-to-dense completion in front of an otherwise unchanged dense detector**

The reasoning was:

1. `step0_shell_exact = 53.72` suggests the current shell is structurally capped far below the dense baseline
2. any completion or observation-aware tweak added *inside* that shell would still start from this handicap
3. `step1 = 37.10` shows that going back to native hard semantics on the irregular axis is also not enough
4. therefore the cleaner move is to make the detector input dense again and let the proven dense detector do detection

## Gemini's Cleanest 2-Step Plan

### Step 1. Oracle completion upper bound

Take the 50% sparse features, fill the missing 50% with the true dense features, and run the unchanged dense detector.

Purpose:

- determine whether sparse-to-dense completion is a viable mainline with a high ceiling

### Step 2. Minimal learned completion module

Build a small completion model that takes:

- sparse features
- observation mask

and outputs dense features for the unchanged dense detector.

Purpose:

- directly test whether learned completion can recover a large part of the gap between current `53.x` results and the oracle-completion ceiling

## What Gemini Explicitly Deprioritized

- more soft-assignment variants
- more minor architectural tweaks inside the current shell
- post-hoc / inference-only fixes

Gemini's reading was that these directions are too weak relative to the newly quantified `shell overhead + soft-assignment overhead`.

## What We Accept

This 0416 review is materially stronger than the 0415 version because it uses the newly closed evidence.

The main takeaways worth keeping are:

1. the non-improvement is primarily semantic, not a hidden bug story
2. the next pivot should be stronger than "another irregular-shell improvement"
3. the cleanest mainline is now:
   - sparse-to-dense completion
   - unchanged dense detector path

## Thread Reference

- Reviewer: `gemini-2.5-pro`
- Gemini thread id: `6bc3bda1cb19456e94c2dc426f52c998`

