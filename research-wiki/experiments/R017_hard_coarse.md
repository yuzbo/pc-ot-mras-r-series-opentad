---
type: experiment
node_id: exp:R017
title: "Hard + BCE + Coarse Detection"
date: 2026-04-07
status: completed
config: v3_hard_scorer_aux.py
---

# R017: Hard + BCE + Coarse Detection

## Purpose
用 coarse detection 提供检测级短路径信号。

## Key Config
- mode=hard, lambda_scorer=1.0, lambda_coarse_det=0.5

## Results
- Avg-mAP = 57.55% ❌
- Scorer entropy=0.62 (接近 uniform 的 0.693)

## Conclusion
Hard mode 下 frame_scores.detach() 切断 det→scorer 梯度。Scorer 几乎没学到有意义的选帧策略。
