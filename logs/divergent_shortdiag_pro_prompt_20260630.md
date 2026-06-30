你是 GPT-5.5 Pro / ChatGPT Pro 路线级研究评审。请用中文评审三个 THUMOS14 Sparse TAD divergent short diagnostic gate 分支：ABR、MDL-Knot、BVR-TWB/VOI-BBC。

本次请求是路线级技术准入讨论，不是代码 owner 任务。请不要给出代码补丁，不要建议启动完整训练，不要把短诊断结果解释成正式 mAP、runtime、deploy 或 paper claim。

## 当前决策

请判断这三个分支是否可以按优先级进入 `SHORT_DIAGNOSTIC_ONLY`，是否存在隐藏 blocker，是否需要代码修正或进一步 Pro 评审。

## 研究目标

ABR 和 MDL-Knot 是创新性 second-stage routes，但仍需要完整实现和更长证据链；BVR-TWB/VOI-BBC 是当前主路线。现在三条路线都已有 short diagnostic gate，目标是用短诊断验证门控、协议边界和最小运行可解释性，而不是给出正式训练结论。

最终 THUMOS14 TAD 目标是 task-aware dynamic temporal acquisition：动态决定每个视频、窗口、动作区域或难例应该处理多少 frames/snippets/tokens，减少持续时间冗余，把更多关键信息送给 detector，并保持或提升 TAD，尤其是高 IoU 定位。

## 协议边界

- 所有路线必须保持 DIVERGENT labels。
- 不允许与 C3/C3-Pro 混合，除非未来有单独记录的 combo approval；本次没有 combo approval。
- 这些 short diagnostic gates 不能产生完整训练、正式 mAP、runtime/FLOPs、deployment、paper claim 或 sparse-compute frontier claim。
- 不能使用 validation/test GT、teacher leakage、raw-prediction shortcut、hidden cache decision、evaluator/post-processing shortcut。
- 如果 GitHub 分支或关键文件不可见，请明确回答 `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`，不要假装已检查。

## 资源 caveat

N16R4 protected hold `1118197` 不能释放。当前 child step 可能阻塞 GPU，所以你的任务是判断技术 readiness 和路线优先级，不要把资源拥塞解释为必须强行 launch，也不要建议释放 protected hold。

## 三个 GitHub 分支和证据

### 1. ABR

分支：
https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-abr-shortdiag-gate-20260630

关键文件：
- `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py`
- `tools/abr/validate_abr_shortdiag.py`
- `tests/test_abr_shortdiag.py`
- `logs/run_abr_shortdiag_n16r4.sh`
- `research-wiki/experiments/DIVERGENT_ABR_SHORT_DIAGNOSTIC_GATE_20260630.md`

Base：
- `codex/divergent-abr-fullcode-20260629`
- commit `6a5acb2`

已有本地验证：
- `35 passed, 1 skipped`

当前审计结论：
- `COMPLETE_FOR_PRECHECK_ONLY`
- 不是 formal train
- 需要 Pro 决策是否可进入 `SHORT_DIAGNOSTIC_ONLY`

### 2. MDL-Knot

分支：
https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-mdl-knot-shortdiag-gate-20260630

关键文件：
- `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py`
- `tools/mdl_knot/validate_mdl_knot_shortdiag.py`
- `tests/test_mdl_knot_shortdiag.py`
- `logs/run_mdl_knot_shortdiag_n16r4.sh`
- `research-wiki/experiments/DIVERGENT_MDL_KNOT_SHORT_DIAGNOSTIC_GATE_20260630.md`

Base：
- `codex/divergent-mdl-knot-fullcode-20260629`
- commit `c32c8a2`

已有本地验证：
- `27 passed, 1 skipped`

当前审计结论：
- `COMPLETE_FOR_PRECHECK_ONLY`
- 不是 formal train
- 需要 Pro 决策是否可进入 `SHORT_DIAGNOSTIC_ONLY`

### 3. BVR-TWB / VOI-BBC

分支：
https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-bvr-twb-shortdiag-execfix-20260630

关键文件：
- `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py`
- `tools/bvr_twb/validate_bvr_twb_shortdiag.py`
- `tests/test_bvr_twb_shortdiag.py`
- `logs/run_bvr_twb_478325a_shortdiag_n16r4.sh`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_478325A_SHORT_DIAGNOSTIC_PLAN_20260630.md`

Expected base：
- `478325af`

Package/validator 状态：
- package branch descends from base
- validator reports `shortdiag_execution_mode=descendant_shortdiag_package`

已有本地验证：
- `35 passed, 12 skipped`

当前审计结论：
- `COMPLETE_FOR_PRECHECK_ONLY`
- 不是 formal train
- 之前有效 Pro 已要求先 short diagnostic

## 请重点检查

1. 是否实际检查了上述 GitHub 分支和关键文件。
2. 三条路线是否保持各自 divergent purpose，是否有 C3 混合或路线漂移。
3. short diagnostic gate 是否足够 fail-closed，是否避免 validation/test GT、teacher/cache、raw-prediction、post-processing/evaluator shortcut。
4. 本地验证数量是否足以作为进入 `SHORT_DIAGNOSTIC_ONLY` 的证据，还是必须先修正代码/脚本/validator/doc。
5. 三条路线进入短诊断的优先级。
6. 是否任何路线可以被判定为 `READY_FOR_FORMAL_TRAIN`。预期答案大概率是 no，除非你看到非常强的反证。
7. 是否需要进一步 Pro：例如 GitHub 未检查、关键文件不可见、BVR 仍需同线程 follow-up、或者短诊断后再评审。

## 输出格式要求

请用中文，严格使用以下标题：

Context verdict

Inspected GitHub branches/files

Verdict per route

Blocking findings

Non-blocking findings

Short diagnostic launch priority

Required fixes before SHORT_DIAGNOSTIC_ONLY

Whether any route is READY_FOR_FORMAL_TRAIN

Accepted next action labels

在 `Accepted next action labels` 中请使用明确标签，例如：
- `PASS_TO_SHORT_DIAGNOSTIC_ONLY`
- `FIX_BEFORE_SHORT_DIAGNOSTIC_ONLY`
- `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`
- `REQUIRES_FURTHER_PRO_BEFORE_SHORT_DIAGNOSTIC_ONLY`
- `NOT_READY_FOR_FORMAL_TRAIN`

