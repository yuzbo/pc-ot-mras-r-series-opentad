---
type: claim
node_id: claim:C9
title: "mask_adapter=True 有害"
status: supported
created_at: 2026-04-09T17:00:00Z
updated_at: 2026-04-09T17:00:00Z
---

# C9: mask_adapter=True 反而有害 (已支持)

**Statement**: P0fix 中的 `mask_adapter=True`（adapter 前零化 dropped token）反而降低性能，关掉后性能更好。

**Status**: ✅ supported

**Evidence**:
- R015 (mask_adapter=False): **64.51%** — 新最佳
- R008 (mask_adapter=True): 63.74%
- 差距 +0.77%

**Root Cause**: Adapter temporal conv 在 full sequence 上运行时，dropped 位置的 stale scatter 值实际提供了有用的时序上下文。零化这些值反而引入了"时序空洞"，破坏卷积核连续性。

**Impact**: 所有后续实验应以 `mask_adapter=False` 为默认。当前项目最佳基准从 R008 (63.74%) 更新为 R015 (64.51%)。
