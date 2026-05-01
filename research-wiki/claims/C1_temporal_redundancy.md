---
type: claim
node_id: claim:C1
title: "TAD 任务存在适度时序冗余"
status: supported
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# C1: TAD 任务存在适度时序冗余

**Statement**: 在 THUMOS14 上，将采样率从 100% 降到 50%，性能仅下降约 5.5%，说明 TAD 任务中存在适度时序冗余。

**Status**: ✅ supported

**Evidence**:
- EXP-001: stride=1 (68.97%) → stride=2 (65.20%)，降 5.5%
- R008: uniform 50% masking = 63.74%，仅比 stride=1 低 5.2%
- 非线性下降曲线: stride=2→4 降 18%，stride=4→8 降 38%

**Implications**: 50% 是效率-精度平衡点，进一步降采样性能快速恶化。
