---
type: claim
node_id: claim:C2
title: "Stride=2 错过边界帧导致性能下降"
status: invalidated
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# C2: Stride=2 错过边界帧导致性能下降 (已推翻)

**Statement**: stride=2 降采样导致边界帧被跳过，是性能下降的主要原因。

**Status**: ❌ invalidated

**Evidence**:
- EXP-002: stride=2 边界命中率仍为 100%
- 边界定位误差翻倍 (0.033s → 0.066s)，但未错过边界
- 真实边界仅占总帧数 8.63%

**Corrected Understanding**: 性能下降源于整体时序分辨率降低，而非"漏掉边界"。
