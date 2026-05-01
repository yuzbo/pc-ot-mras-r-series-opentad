---
type: idea
node_id: idea:003
title: "Hard Mode + REINFORCE Policy Gradient"
stage: proposed
outcome: pending
created_at: 2026-04-07T00:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# REINFORCE Policy Gradient for Scorer Training

## Core Idea
把 scorer 当作 RL policy，用 det loss 作为 reward signal，通过 REINFORCE policy gradient 桥接 hard mode 下的梯度断裂。

## Design
- 训练推理都用 hard gather/scatter (无 train-test gap)
- π(keep_i=1|s) = sigmoid(tubelet_score_i)
- R = -det_loss (lower det loss = higher reward)
- ∇θ J ≈ E[(R - baseline) · ∇θ log π(a|s)]
- EMA baseline for variance reduction (α=0.99)
- 极低 BCE (0.01) 辅助冷启动

## Target Gap
G4: Hard mode 下 det loss 到 scorer 的梯度断裂

## Why This Might Work
- 训练推理完全一致，消除 train-test gap
- REINFORCE 在 Uni-AdaFocus 中已验证有效
- 不需要修改 backbone 任何部分

## Risk
- REINFORCE 方差大，可能需要大量调参
- Reward signal (det loss) 可能太 noisy
- 冷启动期 scorer 随机选帧，reward 信号弱

## Tested By
exp:R020 (待运行)
