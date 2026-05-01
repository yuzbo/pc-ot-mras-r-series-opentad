---
type: experiment
node_id: exp:IRREGULAR_UPSTREAM_AUDIT_0416
title: "Upstream OpenTAD comparison audit: what was actually changed, what was not broken, and which irregular-head conclusions were overstated"
date: 2026-04-16
status: audited
created_at: 2026-04-16T16:55:00+08:00
updated_at: 2026-04-17T13:03:19+08:00
---

# IRREGULAR_UPSTREAM_AUDIT_0416

## Scope

This page records a fresh code-level comparison between:

- upstream clean clone: `OpenTAD_UpstreamFresh` at `main@1aa8ca4`
- current working branch: `OpenTAD_Back`

The audit question is narrow:

- are the current irregular-head plateaus mainly caused by hidden global code breakage?
- or by semantic drift inside the newly added irregular branch and its experiment interpretation?

## Direct Diff Summary

### Unchanged upstream ActionFormer core

The following upstream files are unchanged relative to the fresh clone:

- `configs/_base_/models/actionformer.py`
- `opentad/models/dense_heads/anchor_free_head.py`
- `opentad/models/dense_heads/actionformer_head.py`
- `opentad/models/detectors/actionformer.py`

This means the original dense ActionFormer baseline path was not globally broken.

### Original dense projection / neck were not overwritten

`OpenTAD_Back` adds new classes to the existing files, but does not modify the original dense classes:

- `opentad/models/projections/actionformer_proj.py`
  - added: `GridAwareConv1DTransformerProj`
  - added: `DensePassthroughConv1DTransformerProj`
- `opentad/models/necks/fpn.py`
  - added: `GridAwareFPNIdentity`
  - added: `DensePassthroughFPNIdentity`

The original `Conv1DTransformerProj` and `FPNIdentity` remain intact.

### Most irregular logic lives in newly added files

The real irregular branch is isolated in new files such as:

- `opentad/models/detectors/irregular_actionformer.py`
- `opentad/models/projections/irregular_actionformer_proj.py`
- `opentad/models/necks/irregular_fpn.py`
- `opentad/models/dense_heads/irregular_actionformer_head_v2.py`
- `opentad/models/dense_heads/irregular_actionformer_head_v3.py`
- `opentad/models/dense_heads/irregular_actionformer_head_v3_oabs.py`

So the current problem is not "the whole repo was corrupted." It is localized to how the irregular branch was designed and how its results were interpreted.

## Main Findings

### 1. The original OpenTAD / ActionFormer baseline was not broken

No evidence was found that the upstream dense baseline path was silently damaged.

The strongest safe conclusion is:

- dense baseline numbers remain meaningful
- current irregular-head problems are not explained by a globally broken ActionFormer implementation

### 2. The current `x` line is not a full irregular-aware trunk

This is the most important semantic correction.

The `x` line uses:

- `GridAwareConv1DTransformerProj`
- `GridAwareFPNIdentity`

But in the current implementation these modules mainly:

- normalize / pass through `temporal_grid`
- downsample the grid metadata
- return grid-aligned tuples to the head

They do **not** make the feature transform itself consume irregular geometry in the way the true irregular `y` line does.

So the current `x` line should be read as:

- **dense-trunk + irregular-head carrier**

not as:

- **full irregular-aware projection / neck / detector**

### 3. `headv2_denseexact = headv2_x = 51.04` is now explained

This equality should no longer be interpreted as:

- "proj/neck fairness has been fully settled"

The cleaner interpretation is:

- under the current code, the `x`-line `GridAware...` feature path is effectively dense-like
- therefore `headv2_x` and `headv2_denseexact` are expected to match closely

This makes the result understandable, but it also means the `x` line cannot be used as evidence about a truly irregular-aware trunk.

### 4. `step0_shell_exact = 53.72` is not a valid shell-exact baseline

The config:

- `input_random_fixed_50pct_irregular_actionformer_step0_dense_points_hard_shell_exact.py`

inherits from:

- `input_random_fixed_50pct_irregular_actionformer_step0_densehead_remap.py`

and that parent sets:

- `remap_gt_to_selected_axis=True`

Therefore `53.72` is not a clean "dense semantics inside the irregular shell without remap" control.

It should not be used to support claims such as:

- shell overhead = `63.12 - 53.72 = 9.40`
- the current shell has a hard ceiling in the low `50s`

Those claims are currently unsupported.

### 5. `fixsingleton = 52.88` is not a clean singleton ablation

The config:

- `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_fixsingleton.py`

only changes:

- `work_dir`

It does not change any OABS parameter in config space.

So `52.88` may still correspond to a real rerun on a later code snapshot, but it is **not** a clean config-level singleton ablation and should not be interpreted that way.

## Follow-up After Repair Queue

The later repaired queue does **not** overturn the audit. It refines it.

Three follow-up facts are now established:

1. repaired `step0_dense_head_baseline = 53.72`
   - after explicitly fixing the GT-axis mismatch by setting `remap_gt_to_selected_axis=True`
   - this proves the old `13.79` collapse was a broken-config result
   - but it is still a **remapped selected-axis control**, not the missing no-remap shell-exact baseline
2. repaired `step0b_dense_points_soft_sym_repaired = 47.52`
   - this lands `6.20 mAP` below repaired Step0 inside the same remapped control family
   - so the soft-assignment penalty remains real in that family
   - but the fully correct no-remap paired control is still missing
3. clean singleton A/B is now closed:
   - `headv3_oabs_full_x_legacysingleton = 53.48`
   - `headv3_oabs_full_x_fixsingleton_repaired = 52.88`
   - so singleton repair is not a useful lever on the present `x`-line carrier

These follow-up results therefore strengthen the audit's main point:

- the old pathological numbers were partly caused by misconfigured controls
- once controls are repaired, the current `x` line still behaves like a dense-trunk + irregular-head carrier
- and the missing no-remap shell-exact measurement remains genuinely missing

## Corrected Readouts

### Still trustworthy

The following results remain useful, but with corrected scope:

- `headv2_x = 51.04`
- `headv3_x = 52.60`
- `headv3_oabs_full_x = 53.16`
- `headv3_oabs_full_oaa_x = 53.20`
- `headv3_oabs_full_x_legacysingleton = 53.48`
- clean singleton A/B: `53.48 -> 52.88`

Correct scope:

- these are results on a **dense-trunk + irregular-head** carrier
- they show modest positive gains from V2 -> V3 -> OABS/OAA
- they do **not** prove a full irregular-aware trunk plateau

### No longer trustworthy for causal attribution

- `step0_shell_exact = 53.72` as shell-overhead evidence
- `fixsingleton = 52.88` as a singleton-fix ablation

## What Was Overstated Before

The following earlier readings should be retired:

1. "the current irregular shell overhead is directly measured as `9.40 mAP`"
2. "the shell itself has already been proven to hard-cap in the low `50s`"
3. "`x`-line plateau means a full irregular-aware trunk has failed"

The safer current statement is:

- the best completed **dense-trunk + irregular-head** carrier is now `53.48`
- the true irregular `y` line remains weaker
- but the clean shell-overhead number still has to be re-measured

## What This Means for the Main Hypothesis

The audit does **not** say the completion pivot is wrong.

It says something narrower but important:

- completion may still be the best next line
- but it should no longer be justified by an invalid `9.40 mAP` shell-overhead estimate

The current code comparison supports:

- original dense semantics remain strong
- current irregular-head gains are limited on the dense-trunk carrier
- true irregular trunk evidence is still incomplete or weaker than previously claimed

## Repair Experiments Needed

### 1. True shell-exact rerun

Need a real no-remap control:

- dense points
- hard assignment
- unchanged dense semantics
- **no `remap_gt_to_selected_axis=True` inheritance**

This is the missing experiment needed to measure actual shell / contract overhead.

### 2. True dense-points soft-assignment rerun

Need the matching no-remap version of:

- dense points
- soft assignment
- symmetric regression

Only then can the soft-assignment overhead be quantified cleanly against the corrected shell-exact baseline.

### 3. Either fix or relabel the `x` line

One of the following must happen:

- make `GridAware...` truly consume irregular geometry in the feature path
- or explicitly relabel the `x` line as dense-trunk + irregular-head and stop using it for trunk claims

### 4. Optional clean rerun of singleton ablation

If singleton handling still matters, it needs:

- an explicit config-level parameter change
- or a pinned code hash

Otherwise that comparison remains attribution-unsafe.

## Current Best Summary

1. Upstream dense ActionFormer path is intact.
2. The main issue is not global code destruction.
3. The main semantic mistake was over-interpreting the current `x` line.
4. The best trustworthy `x`-line result is now `53.48`, but that is still a dense-trunk carrier result.
5. The shell-overhead story and singleton-ablation story both need correction.

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
