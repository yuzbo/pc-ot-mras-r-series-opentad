---
type: experiment
node_id: exp:R022
title: "Coarse Inject / MobileNet Temporal Coarse Injection"
date: 2026-04-09
status: running
config: v3_mobilenet_temporal_coarse_inject.py
avg_map: 57.27
map_05: 59.91
map_07: 32.34
confidence: medium
---

# R022: Coarse Inject

## Purpose
把 coarse temporal feature 作为 dropped / weak positions 的补充语义，缓解 scorer 与 backbone 表示脱节、dropped token stale feature 污染等问题。

## Key Config
- learned scorer + coarse feature injection
- MobileNet temporal scorer / coarse branch
- keep ratio ≈ 50%

## Current Result
- Latest observed Average-mAP = 57.27
- mAP@0.5 = 59.91
- mAP@0.7 = 32.34

## Current Conclusion
它是当前较有前景的 learned 方向之一，但暂时仍低于 R008 uniform 50%（63.74）。说明 coarse inject 有帮助，但还不足以证明 learned sampling 已经超过 uniform。
