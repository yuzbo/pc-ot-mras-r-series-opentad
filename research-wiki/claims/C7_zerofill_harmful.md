---
type: claim
node_id: claim:C7
title: "零填充 dropped 位置有害"
status: supported
created_at: 2026-04-09T13:00:00Z
updated_at: 2026-04-09T13:00:00Z
---

# C7: 零填充 dropped 位置有害

**Statement**: 在 V3 MaskedViT 中，将 dropped 位置填零会严重损害性能，保留上一层 scatter 值是更优选择。

**Status**: ✅ supported

**Evidence**:
- EXP-007 (零填充): 47.89%
- EXP-008 (保留 scatter 值): 60.68%
- 差距 12.8%

**Root Cause**: 零值干扰 adapter Conv1d 的时序建模，破坏局部时序连续性。
