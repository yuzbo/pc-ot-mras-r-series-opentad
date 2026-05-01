---
type: claim
node_id: claim:C5
title: "Hard mode + BCE + coarse_det 可训练 scorer"
status: invalidated
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# C5: Hard mode + BCE + coarse_det 可训练 scorer (已推翻)

**Statement**: 在 hard mode 下，通过 BCE boundary 监督 + coarse detection 短路径可以训练 scorer 超过 uniform。

**Status**: ❌ invalidated

**Evidence**:
- R017: hard + BCE=1.0 + coarse_det=0.5 = 57.55%
- Scorer entropy=0.62 接近 uniform (0.693)，说明 scorer 几乎没学到有意义的策略

**Root Cause**: hard mode 下 `frame_scores.detach()` 切断了 det loss 到 scorer 的梯度。BCE 和 coarse_det 的信号不足以独立训好 scorer。
