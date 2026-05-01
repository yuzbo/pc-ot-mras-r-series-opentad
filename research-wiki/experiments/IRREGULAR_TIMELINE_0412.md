---
type: experiment
node_id: exp:IRREGULAR_TIMELINE_0412
title: "OpenTAD_Back irregular timeline detector traceback and implementation snapshot"
config: configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer*.py
server: Server 3 (25876)
date: 2026-04-12
status: audited
created_at: 2026-04-12T22:20:00+08:00
updated_at: 2026-04-12T23:59:00+08:00
---

# IRREGULAR_TIMELINE_0412

## Purpose

Trace back the old `input_random_fixed_50pct_irregular_actionformer.py` run on Server 3, verify the real logged result, and separate:

- the earliest local snapshot in `.codex_sync_irregular/`
- the actual server code used by the `...full_20260411.log` run
- the later native-irregular `remap*` series

## Code Locations

- [OpenTAD_Back/configs/_base_/models/irregular_actionformer.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/configs/_base_/models/irregular_actionformer.py)
- [OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer.py)
- [OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_remap.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_remap.py)
- [OpenTAD_Back/opentad/models/projections/irregular_actionformer_proj.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/opentad/models/projections/irregular_actionformer_proj.py)
- [OpenTAD_Back/opentad/models/necks/irregular_fpn.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/opentad/models/necks/irregular_fpn.py)
- [OpenTAD_Back/opentad/models/dense_heads/prior_generator/irregular_point_generator.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/opentad/models/dense_heads/prior_generator/irregular_point_generator.py)
- [OpenTAD_Back/opentad/models/dense_heads/irregular_actionformer_head.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/opentad/models/dense_heads/irregular_actionformer_head.py)

## Raw Log Audit

Confirmed on Server 3:

- raw log exists at `/root/autodl-tmp/OpenTAD_Back_check/logs/input_random_fixed_50pct_irregular_actionformer_full_20260411.log`
- work dir exists at `/root/autodl-tmp/OpenTAD_Back_check/exps/thumos/adatad/input_random_fixed_50pct_irregular_actionformer`
- only saved checkpoint is `epoch_59.pth`

Real eval trajectory from the raw log:

| Run | Avg-mAP | mAP@0.7 |
|---|---:|---:|
| epoch 39 eval | 34.62 | 8.90 |
| epoch 44 eval | 36.32 | 10.37 |
| epoch 49 eval | 37.43 | 10.95 |
| epoch 54 eval | 38.19 | 11.81 |
| epoch 59 eval | 39.04 | 12.35 |

This directly invalidates the previously repeated `64.62 / 44.86` number from the monitoring aggregate log.

## Important Correction

The old claim

- "`input_random_fixed_50pct_irregular_actionformer.py` reached `64.62`"

is not supported by the raw training log.

Current corrected statement:

- the audited Server 3 raw run of `input_random_fixed_50pct_irregular_actionformer.py` reached only `39.04 Avg-mAP`

## Traceback Result

### 1. Old local snapshot vs current local config

The config file itself is unchanged:

- local `.codex_sync_irregular/.../input_random_fixed_50pct_irregular_actionformer.py`
- current local `configs/.../input_random_fixed_50pct_irregular_actionformer.py`

Both use:

- `remap_gt_to_selected_axis=False`
- `IrregularConvTransformerProj + IrregularFPN + IrregularActionFormerHead`

So the config name was not silently swapped.

### 2. But the audited Server 3 code was not the earliest snapshot

The server code used by the `39.04` run had already drifted away from the earliest `.codex_sync_irregular/` snapshot.

#### A. Head drifted heavily

Relative to the earliest snapshot, the audited Server 3 head had already added:

- `use_regress_range`
- debug state collection
- decode changed to use symmetric scale on both sides
- center radius changed from `min(cell_left, cell_right)` to a larger aligned radius
- classification target changed from shortest-duration mask to `matched_mask`
- regression GT selection changed from shortest-duration exclusive match to `scale_cost` assignment
- regression normalization changed from left/right asymmetric normalization to symmetric normalization by one scale

Interpretation:

- the `39.04` run was already using a partially rewritten head semantics
- it was no longer the original dense-like irregular bridge head

#### B. Point generator also drifted

Relative to the earliest snapshot, the audited Server 3 point generator had already changed from:

- `level_scale` based regression ranges

to:

- `point_scale = cell_left + cell_right`
- `range_mode = hard / overlap_band`

Interpretation:

- level ownership semantics had already been rewritten toward native irregular axis logic
- this is a major semantic break from the earliest snapshot

#### C. Projection and neck mostly changed for stability/diagnosis

Relative to the earliest snapshot, the audited Server 3 projection/neck mainly added:

- `safe_geometry`
- `geometry_fp32`
- `rel_dt_clip`, `rel_span_clip`
- extensive non-finite/debug hooks

These changes look like:

- numerical stabilization
- runtime diagnostics

not like the main reason for the collapse from a hypothetical high score to `39.04`.

## Structural Judgment

The audited `39.04` run is best described as:

- `pre-remap`, because `remap_gt_to_selected_axis=False`
- but already `post-head/post-point rewrite`, because head and point semantics had already diverged strongly from the earliest snapshot

So it is not valid to treat this run as:

- the original earliest irregular bridge version

It is more accurate to treat it as:

- a transitional version between the earliest dense-like irregular bridge and the later fully native-irregular `remap*` line

## Current Results Summary

| Experiment | Meaning | Result | Judgment |
|---|---|---:|---|
| `input_random_fixed_50pct_irregular_actionformer` | pre-remap transitional irregular detector on Server 3 raw audited run | 39.04 | weak, far below input-side dense baseline |
| `input_random_fixed_50pct_irregular_actionformer_remap` | native irregular detector + GT remap | diag only | coarse-level positive collapse |
| `..._remap_no_regrange` | remove hard regress-range gate | 4.13 / 3.92 / 3.92 | positives restored, semantics still broken |
| `..._remap_overlap` | overlap-band point ranges | 1.94 at first eval | worse than no_regrange |
| `headv2_x` | dense-like bridge proj/neck + HeadV2 | 51.04 / 25.04 | major recovery; detector is learnable again |
| `headv2_y` | irregular proj/neck + HeadV2 | 49.00 / 22.71 | clear recovery, but still below X |

## Takeaway

What is now firmly established:

1. The famous `64.62` number is not supported by the raw log.
2. The audited old run is real, but its true result is only `39.04`.
3. That `39.04` run was already using a rewritten head and point generator, so it cannot be treated as the pristine earliest irregular bridge implementation.
4. The main semantic break across versions is in `head + point_generator`, not primarily in `projection + neck`.

## Follow-up on 2026-04-13

The new HeadV2 line substantially changed the picture:

- `headv2_x = 51.04 / 25.04`
- `headv2_y = 49.00 / 22.71`

This means the detector is no longer in a pure semantic-collapse regime. The current bottleneck has shifted from "cannot assign / cannot regress" to "still structurally behind the dense detector family by about 12~14 mAP on the same random-fixed input, with the current irregular projection/FPN branch still about 2 mAP behind the dense-like bridge path".

See also:

- `exp:HEADV2_0413`

## Related Existing Evidence

- `idea:011`
- `exp:INPUT_RANDOMFIX_0410`
- `exp:INPUT_ORACLE_BOUNDARY_DENSE_0411`
- `claim:C11`

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
