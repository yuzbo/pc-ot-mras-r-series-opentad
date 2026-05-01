---
type: experiment
node_id: exp:R020
title: "Hard + REINFORCE Policy Gradient"
date: 2026-04-08
status: completed
config: v3_hard_reinforce.py
avg_map: 53.96
map_05: 56.81
confidence: medium
---

# R020: Hard + REINFORCE Policy Gradient

## Purpose
验证在 hard sampling 条件下，是否可以用 policy gradient 把 det reward 直接传回 scorer，从而绕过 hard mask 的不可导问题。

## Key Config
- mode=hard
- lambda_reinforce > 0
- scorer uses policy-gradient reward rather than only BCE / coarse_det

## Results
- Avg-mAP = 53.96
- mAP@0.5 = 56.81

## Conclusion
REINFORCE 没有超过 uniform 50%，也没有明显优于较强 learned baseline。它证明“hard mode 可导替代”是可跑的，但方差和学习效率仍是问题。
