请用中文做只读代码审查，不要修改文件。

背景：我们正在 THUMOS14 TAD/BATA 帧选择方向中添加一个新的端到端训练分支
`BoundarySharp-ST`。前一个 GPT-5.5 Pro 讨论已经确认当前 quota selector 是端到端的：
selector 改变 raw-frame positions，selected frame values/positions 能把 detection loss 梯度传回
selector；但 epoch-19 诊断显示当前 AB/BH quota selector 的 boundary 通道很平，不能证明学到了
boundary-aware allocation。因此新实现希望更激进：boundary/action quota chunk 前向使用 hard top-k
位置，反向用 soft-quantile surrogate 保留梯度。

请逐函数/逐关键逻辑审查附件中的实现，重点回答：

1. `TemporalDensityRawFrameSelector` 新增的 `st_topk` quota position mode 是否实现正确？
   - hard forward 是否真按 valid dense prefix 内 top-k logits 选择位置；
   - `hard.detach() - soft.detach() + soft` 是否能把 detection/selector loss 的梯度回传到 logits；
   - 与后续 concat/sort/min-gap/pad 的交互是否有形状、梯度或边界条件问题；
   - dense tail mask、short valid_len、selected_valid_lens 和 target_len 是否安全。
2. 新增 `quota_action_target_weight` / `quota_boundary_target_weight` 是否保持旧配置向后兼容，并且能让新 config 的 boundary target 更尖锐。
3. `e2e_rawdensel_384of768_boundarysharp_st_adapter.py` 是否严格保持 384-of-768 fixed 50% backbone budget、train-only GT auxiliary、无 test-time GT/teacher/cache。
4. 新测试是否覆盖了关键合同，有没有错误预期、遗漏或会在 Linux/N16R4 上失败的地方。
5. 该实现是否符合最初设计目的：让端到端 selector 的帧选择结果可以影响检测性能并获得反向反馈，同时避免把采样退化成 uniform-like coverage。
6. 如果未来该配置 mAP 提升，是否可以归因到 selector 内部/ST quota 位置和 train-only target shaping，而不是 Adapter/head/post-processing 或输入预算变化。
7. 是否允许进入 Gemini/DeepSeek review 和 N16R4 preflight；如有 blocker，请给出必须修复的文件/行级问题。

当前验证：

- 本地 `py_compile`: PASS。
- Windows pytest: `1 passed, 19 skipped`，torch/mmengine 测试按预期在 Windows 跳过。
- `git diff --check`: PASS，仅 LF/CRLF warning。
- N16R4 Linux preflight 尚未运行，等待本次 Pro 审查后再同步。

请输出：

- Verdict: PASS / WARN / FAIL
- Blocking findings
- Non-blocking findings
- Required fixes before deployment
- Whether Gemini/DeepSeek + N16R4 preflight can proceed
