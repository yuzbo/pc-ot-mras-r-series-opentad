---
type: experiment
node_id: exp:R022_COARSE
title: "Gumbel + SoftGate + Coarse Inject (chain)"
config: v3_gumbel_softgate_coarse_inject.py
server: Server 1
date: 2026-04-09
status: completed
created_at: 2026-04-09T17:00:00Z
updated_at: 2026-04-09T17:00:00Z
---

# R022-coarse: Coarse Feature Injection

## 目的
将 scorer 的 coarse_feat 注入 backbone adapter (chain 模式)，改善 dropped 位置特征。

## 结果
| Metric | Value |
|--------|-------|
| Avg-mAP | 56.28% |
| mAP@0.5 | 59.34% |
| mAP@0.7 | 31.78% |

## 失败分析
- 比 R017 (57.55%) 还低，coarse inject 没有起到正面作用
- 仍使用 soft gating，受 train-test gap 影响
