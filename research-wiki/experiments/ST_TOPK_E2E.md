---
type: experiment
node_id: exp:ST_TOPK_E2E
title: "ST top-k tubelet 端到端"
config: v3_st_topk_tubelet_e2e.py
server: Server 2
date: 2026-04-09
status: completed
created_at: 2026-04-09T17:00:00Z
updated_at: 2026-04-09T17:00:00Z
---

# ST-TopK E2E: Straight-Through Top-K 端到端训练

## 目的
用 Straight-Through top-k 替代 Gumbel-Softmax，实现 hard mask 的可微近似。

## 结果
| Metric | Value |
|--------|-------|
| Avg-mAP | 53.57% |
| mAP@0.5 | 56.13% |
| mAP@0.7 | 27.77% |

## 分析
- 与 R021 (53.98%) 和 R020 (53.96%) 几乎一致
- 进一步确认：无论用何种可微近似 (Gumbel-ST / REINFORCE / ST-TopK)，learned scorer 在 ~54% 附近收敛
- 问题不在梯度方案，而在更深层原因
