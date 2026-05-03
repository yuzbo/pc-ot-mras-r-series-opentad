# SimOTA Compact-Space Detection Head

**Date**: 2026-05-04
**Status**: Approved for implementation
**Reviewers**: Gemini 3 Pro, Codex GPT-5.5

## Problem

FCOS (ActionFormerHead + 6-level FPN) achieves 63.12% mAP on 50% random-fixed sparse THUMOS14 via DensePassthrough+FPNIdentity. However, FCOS's hard `regression_range` breaks in compact-index-space: physically long actions may have short compact length, causing them to be routed to wrong FPN levels.

## Design

Replace hard `regression_range` routing with SimOTA (cost-based dynamic assignment). The model keeps ActionFormerHead + 6-level FPN intact but lets features decide which FPN level handles each GT.

### Key Changes

1. **Delete `regression_range`**: `use_regress_range=False` in head config
2. **Expand candidate pool**: `center_sample_radius=1000.0` to effectively disable spatial prior
3. **Activate SimOTA**: Use existing `AnchorFreeSimOTAAssigner` with cost=cls_loss+IoU_loss
4. **All levels compete**: Every GT evaluated on all 6 FPN levels; SimOTA picks top-K lowest-cost points

### Config (inherits from step0_densehead, the 63.12% baseline)

```python
_base_ = ["./input_random_fixed_50pct_irregular_actionformer_step0_densehead.py"]

model = dict(
    rpn_head=dict(
        use_regress_range=False,
        center_sample_radius=1000.0,  # effectively disable spatial prior
        assigner=dict(
            type="AnchorFreeSimOTAAssigner",
            cls_weight=1.0,
            iou_weight=3.0,
            center_radius=1000.0,
            topk=9,
            dynamic_k=dict(
                type="dynamic_k_matching",
                keep_percent=0.3,
            ),
        ),
    ),
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_simota_compact"
```

### Implementation Plan

1. **Config**: Create simota config inheriting step0_densehead
2. **Assigner fix**: If needed, patch `AnchorFreeSimOTAAssigner.within_center()` to respect `center_radius=1000.0`
3. **Dynamic K clamp**: Ensure `dynamic_k_matching` has `clamp(min=1)` to prevent zero-assignment collapse
4. **Deploy**: Sync config + any assigner fixes, launch on Server (port 35407)
5. **Evaluate**: Compare against 63.12% baseline at epoch 39

### Success Criteria

- **>63.12%** mAP at epoch 39 → SimOTA is better than baseline → scale to full training
- **~63.12%** → SimOTA neutral, but validates assignment hypothesis (no harm)
- **<60%** → SimOTA degrades baseline → investigate cost weights or revert

### Risks

| Risk | Mitigation |
|------|-----------|
| `center_radius=1000` not respected by assigner | Patch `within_center()` if needed |
| `dynamic_k` collapses to 0 for short actions | Add `clamp(min=1)` guard |
| IoU cost mismatch with stride normalization | Verify assigner's `gt_offset /= stride` matches head's reg_pred |
| SimOTA overhead slows training | Monitor; assigner is O(K×N) per GT, K=total points, N=FPN levels |
