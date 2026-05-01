---
type: experiment
node_id: exp:TEMPORALGRIDFIX_FULL_QUEUE_0423
title: "OpenTAD_Back temporal-grid-fix formal training deployment audit on 25876"
date: 2026-04-23
status: completed
created_at: 2026-04-23T09:52:00+08:00
updated_at: 2026-04-24T11:26:05+08:00
---

# TEMPORALGRIDFIX_FULL_QUEUE_0423

## Scope

This page records the exact audit that justifies launching the formal `OpenTAD_Back` temporal-grid-fix rerun on server `25876`.

The goal is narrow and explicit:

- verify that the repaired `temporal_grid.py` is actually exercised by the current `headv3_x` carrier
- confirm that the deployed full-train config is only a `work_dir` alias over the already trusted `headv3_x` inheritance chain
- make clear what this run does and does **not** validate

## Current live status

- server: `25876`
- screen: `temporalgridfix_full_s3`
- queue log: `/root/autodl-tmp/OpenTAD_Back_check/logs/temporalgridfix_full_server3.log`
- latest confirmed state:
  - `2026-04-23 16:01:13 starting input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix`
  - `2026-04-23 20:40:11 finished input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix`
  - final validation read: `52.40`

## Config inheritance audit

The deployed formal config is:

- [input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix.py)

Its inheritance chain is:

- `temporalgridfix.py`
  - only overrides `work_dir`
- `headv3_x.py`
  - projection = `GridAwareConv1DTransformerProj`
  - neck = `GridAwareFPNIdentity`
- `headv3_base.py`
  - head = `IrregularActionFormerHeadV3`
  - prior generator = `IrregularPointGeneratorV2`
  - boundary loss enabled
- `irregular_actionformer.py`
  - frozen `VisionTransformerCP`
  - THUMOS `random_fixed_subsample` with `keep_ratio=0.5`

This means the formal rerun changes no model semantics, no data semantics, and no optimization settings beyond the new output directory.

## Post-hoc confirmation

The final post-deployment audit remains consistent with the launch-time reading:

- `temporalgridfix.py` is only a clean naming alias over the already trusted `headv3_x` chain
- the repaired `temporal_grid.py` is definitely exercised by the current `GridAwareConv1DTransformerProj.forward` path
- feature downsample and grid downsample are index-aligned on `(input[2i], input[2i+1])`
- but this carrier still uses dense `TransformerBlock` `MaxPool1d(stride=2)` for feature downsample, not `IrregularConvTransformerProj.downsample_features`

So the formal `52.40` result should still be read narrowly:

- it is a clean end-to-end check for repaired temporal-grid semantics on the present `x` carrier
- it is not yet the decisive experiment for the separate "feature/grid weighting mismatch" (`BUG-3`) story

## Temporal-grid fix path audit

The repaired file is:

- [temporal_grid.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/opentad/models/utils/temporal_grid.py)

The current `headv3_x` path uses it as follows:

1. `GridAwareConv1DTransformerProj.forward` calls `downsample_temporal_grid(temporal_grid)`.
2. The import path is `from ..utils import downsample_temporal_grid`, which resolves to the repaired `opentad/models/utils/temporal_grid.py`.
3. Feature/grid pairing is structurally aligned at the index level:
   - feature downsample uses `TransformerBlock` internal `MaxPool1d(stride=2)`
   - grid downsample merges `(input[2i], input[2i+1])` into `merged[i]`
4. `GridAwareFPNIdentity` is pass-through plus norm and does not rewrite `temporal_grid_list`.
5. `IrregularActionFormerHeadV3` therefore receives the repaired multi-level `temporal_grid_list`, and `_apply_geometry_modulation` consumes the fixed `cell_left`, `cell_right`, and `fresh_mask`.

## Deployment audit

The deployment synced exactly these files to server `25876`:

- [temporal_grid.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/opentad/models/utils/temporal_grid.py)
- [input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix.py](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_x_temporalgridfix.py)
- [run_temporalgridfix_full_server3.sh](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/scripts/run_temporalgridfix_full_server3.sh)
- [deploy_temporalgridfix_full_server3.ps1](E:/DeskTop/TAD/temrefuse-tad/OpenTAD_Back/scripts/deploy_temporalgridfix_full_server3.ps1)

The queue script behavior is also verified:

- waits for GPU availability with `memory.used < 500 MiB`
- validates the config with `mmengine.Config.fromfile(...)`
- launches the run through single-GPU `torchrun`
- uses `master_port=29835`, which does not collide with the currently active frame-sampling job

## Important interpretation boundary

This formal run **does** validate the practical effect of the repaired temporal-grid semantics on the current `x`-line carrier, especially:

- repaired `fresh_mask` propagation
- repaired coarse-level `cell_left/right` construction

But this run does **not** fully exercise the previously identified `BUG-3` story:

- in this carrier, `GridAwareConv1DTransformerProj` still downsamples features through dense `MaxPool1d`
- it does **not** use `IrregularConvTransformerProj.downsample_features`
- so the "feature downsample weight vs center downsample weight" mismatch is not the variable being tested here

The correct reading is therefore:

- this queued full-train rerun is the clean verification for `BUG-1/2` on the current `headv3_x` path
- it is **not** the final experiment for judging `BUG-3`

## Runtime readout

Observed train band:

- `Loss ~= 1.22 ~ 1.28`
- `cls_loss ~= 0.26 ~ 0.27`
- `reg_loss ~= 0.35 ~ 0.38`
- `boundary_loss ~= 0.61 ~ 0.64`

Numeric stability watch:

- `2026-04-23 16:01:23` reported `non-finite gradients detected at epoch=0 iter=0` on `module.rpn_head.reg_head.weight`
- `2026-04-23 16:01:56` reported `non-finite gradients detected at epoch=0 iter=17` on `module.rpn_head.reg_head.weight`
- `2026-04-23 16:31:07` reported `non-finite gradients detected at epoch=9 iter=17` on `module.rpn_head.reg_head.weight`
- `2026-04-23 17:36:35` reported `non-finite gradients detected at epoch=29 iter=62` on `module.rpn_head.reg_head.weight`
- `2026-04-23 19:15:50` reported `non-finite gradients detected at epoch=49 iter=71` on `module.rpn_head.reg_head.weight`
- the run skipped that optimizer step, continued training, and advanced normally into later epochs

This should be read as:

- a repeated but still sparse numeric warning that did not prevent full completion
- not a hard failure, because the run kept training and finished normally

## Final readout

Reference comparison:

- old `headv3_x = 52.60`
- new `headv3_x_temporalgridfix = 52.40`
- delta: `-0.20`

The correct conclusion is therefore:

- the repaired temporal-grid semantics were valid code fixes
- but on the current `GridAware...` `x` carrier they did not produce a measurable performance gain
- so `BUG-1/2` were not the dominant limiter of the current `x`-line ceiling
- a later `IrregularConvTransformerProj`-based formal rerun is still needed to isolate the separate `BUG-3` question

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
