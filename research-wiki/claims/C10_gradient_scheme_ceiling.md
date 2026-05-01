---
type: claim
node_id: claim:C10
title: "所有可微梯度方案都在 ~54% 收敛"
status: supported
created_at: 2026-04-09T17:00:00Z
updated_at: 2026-04-09T17:00:00Z
---

# C10: 所有可微梯度方案都在 ~54% 附近收敛 (已支持)

**Statement**: 无论用何种可微近似方案，learned scorer 在 TAD 中的性能上界约为 ~54%，远低于 uniform (64.51%)。

**Status**: ✅ supported

**Evidence**:
| 梯度方案 | 实验 | Avg-mAP |
|---------|------|---------|
| Gumbel-ST + soft gating | R012 | ~52-55% |
| Gumbel-ST + soft gating (最小) | R021 | 53.98% |
| REINFORCE policy gradient | R020 | 53.96% |
| ST top-k | ST_TOPK_E2E | 53.57% |

四种完全不同的梯度方案收敛到几乎相同的性能，说明瓶颈不在梯度传递方式，而在于：
1. 低分辨率帧 (56×56) 上轻量 CNN 无法提取有效的选帧特征
2. TAD 任务中"最优选帧"可能与"按内容选帧"不同（需要 Oracle 验证）
3. Scorer 训练与 backbone 训练的耦合/干扰

**Impact**: 改进梯度方案不再是有效方向。应转向验证 Oracle 上界、或采用两阶段训练。
