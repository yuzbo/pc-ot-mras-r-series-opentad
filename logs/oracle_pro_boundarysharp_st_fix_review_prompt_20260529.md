请用中文做只读聚焦复审，不要修改文件。

这是对 `BoundarySharp-ST` 实现的 GPT-5.5 Pro `WARN` 后修复复审。上一轮 Pro 审查没有代码级 blocker，但建议补强：

1. st_topk invalid dense tail 测试；
2. short valid_len 测试；
3. prefix dense mask 合同；
4. quota target weights 非负校验；
5. merged config quota 检查从文件名包含 `"quota"` 改成基于 `selection_mode`。

我已经做了这些修复。请只审查附件中的最新相关文件和自检报告，判断：

- 这些修复是否正确解决上一轮 WARN；
- 是否引入新的行为风险或测试错误；
- 是否仍保持旧 AB/BH quota 配置向后兼容；
- BoundarySharp-ST config 是否仍保持 384-of-768 fixed 50% budget、train-only GT auxiliary、无 test-time GT/teacher/cache；
- 是否可以进入 Gemini/DeepSeek review 和 N16R4 Linux preflight。

当前 post-fix 本地验证：

- `py_compile`: PASS。
- Windows pytest: `1 passed, 20 skipped`，Linux/torch/mmengine 测试按预期在 Windows 跳过。
- `git diff --check`: PASS，仅 LF/CRLF warning。

请输出：

- Verdict: PASS / WARN / FAIL
- Blocking findings
- Non-blocking findings
- Required fixes before deployment
- Whether Gemini/DeepSeek + N16R4 preflight can proceed
