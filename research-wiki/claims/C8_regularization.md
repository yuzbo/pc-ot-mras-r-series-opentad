---
type: claim
node_id: claim:C8
title: "50% 采样具有正则化效果"
status: invalidated
created_at: 2026-04-09T13:00:00Z
updated_at: 2026-04-09T17:30:00Z
---

# C8: 50% 采样具有正则化效果 (已推翻)

**Statement**: 50% 采样可能起到正则化/去噪作用，提升泛化能力。

**Status**: ❌ invalidated

**原始"证据" (错误推理)**:
- EXP-005 V3 baseline 100%: 55.99%
- EXP-008 V3 无零填充 50%: 60.68%
- 表面看 50% 反超 100%，似乎有正则化效果

**推翻原因**:
真正的基准是 AdaTAD 原始 baseline，而非 V3 框架的 100% 配置：

| 方法 | Avg-mAP |
|------|---------|
| AdaTAD stride=1 全帧 (真基准) | 68.97% |
| AdaTAD stride=2 | 65.20% |
| V3 uniform 50% R015 (最佳) | 64.51% |
| V3 baseline 100% EXP-005 | 55.99% |

50% 采样 (64.51%) 始终低于全帧 baseline (68.97%)，差 4.46%。不存在正则化提升。

EXP-005 的 55.99% 是因为 V3 框架引入了额外性能损耗 (adapter 配置未对齐、P0 级 bug 等)，不能作为 100% 全帧的合理代表。60.68% 反超 55.99% 反映的是 P0 bug 修复的效果，而非采样的正则化作用。
