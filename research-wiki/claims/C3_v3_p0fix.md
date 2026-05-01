---
type: claim
node_id: claim:C3
title: "V3 MaskedViT P0fix 是有效的 50% 采样架构"
status: supported
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# C3: V3 MaskedViT P0fix 是有效的 50% 采样架构

**Statement**: V3 (q=k=v=kept, mask_adapter, no attn_scale) + uniform 50% 采样可达到 63.74% Avg-mAP，仅比全帧 stride=1 (68.97%) 低 5.2%，训练完全稳定。

**Status**: ✅ supported

**Evidence**:
- R008: Avg-mAP=63.74%, mAP@0.5=66.72%
- 零 NaN，训练完全稳定
- 关键设计: use_attn_scale=False, mask_adapter=True, scatter 保留上一层值

**Significance**: 63.74% 是 uniform 采样的天花板，所有 learned scorer 必须超过此值。
