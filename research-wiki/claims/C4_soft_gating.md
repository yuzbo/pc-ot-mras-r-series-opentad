---
type: claim
node_id: claim:C4
title: "Soft gating 可用于训练 TAD learned scorer"
status: invalidated
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# C4: Soft gating 可用于训练 TAD learned scorer (已推翻)

**Statement**: DynamicViT-style soft gating (训练 soft, 推理 hard) 可以端到端训练 scorer 超过 uniform 采样。

**Status**: ❌ invalidated

**Evidence**:
- R012 (纯 E2E): ~52-55%
- R013 (+ BCE=1.0): ~53%
- R014 (+ BCE + aux): 53.73%
- **R021 (最小对照, 判定实验): 53.98%** — 即使极低 BCE (0.05)，仍比 uniform (63.74%) 低 ~10%

**Root Cause**: Train-test gap。训练时 soft mask 门控 (O(N²)) ≠ 推理时 hard sparse (O(K²))。12 层累积的 soft mask 缩放导致特征幅度系统性偏移。

**Impact**: 所有依赖 soft gating 的方案 (R019, R022 原方案) 均需取消或改为 hard mode。
