---
type: claim
node_id: claim:C15
title: "DenseAdapter 14% 根因已确认: 跨轴插值 bug + 不规则几何信息有害"
status: supported
evidence_for: [exp:A0_NOINTERP_0430, exp:DADBUG_B1, exp:CODEX_REVIEW_0430, doc:CODE_AUDIT_0430]
evidence_against: []
created_at: 2026-05-01T10:00:00+08:00
updated_at: 2026-05-01T10:00:00+08:00
---

# Claim C15: 跨轴插值 bug + 不规则几何信息有害

## 声明
1. DenseAdapter 13.86% 的主因是 `linear_interpolate_features` 在不同坐标轴间做 searchsorted
2. A0 消除跨轴插值后恢复至 51.28%, 但仍差 dense baseline 12 个点
3. 剩余差距来自 IrregularConvTransformerProj/IrregularFPN 的不规则几何感知破坏了均匀网格假设
4. **忽略不规则性 (63.12%) > 使用不规则信息 (43-51%)**
