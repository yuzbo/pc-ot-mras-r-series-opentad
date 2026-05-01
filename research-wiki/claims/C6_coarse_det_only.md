---
type: claim
node_id: claim:C6
title: "纯 coarse_det 可独立训练 scorer"
status: invalidated
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# C6: 纯 coarse_det 可独立训练 scorer (已推翻)

**Statement**: 仅用 coarse detection loss (scorer CNN → coarse_feat → frozen projection+rpn_head) 可以训练 scorer。

**Status**: ❌ invalidated

**Evidence**:
- R018: lambda_scorer=0.0, lambda_coarse_det=1.0 → 47.68%
- 严重退化，比 uniform (63.74%) 低 16 个点

**Root Cause**: coarse_feat 经过 frozen projection+rpn_head 的检测信号太弱，无法有效指导选帧。
