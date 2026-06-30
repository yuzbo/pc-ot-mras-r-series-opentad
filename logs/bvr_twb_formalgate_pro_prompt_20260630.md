# GPT-5.5 Pro Code-Grounded Review Prompt: BVR-TWB / VOI-BBC Formal-Readiness Gate

请作为 GPT-5.5 Pro / ChatGPT Pro，对一个 THUMOS14 Temporal Action Detection 动态采样路线的 formal-readiness gate 分支做代码落地审查。请优先阅读 GitHub 分支源码，而不是只根据本提示泛泛判断。

GitHub branch URL:

https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-bvr-twb-formalgate-20260630

路线标签：

`DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

本次审查对象：

BVR-TWB / VOI-BBC dynamic sparse acquisition route 的 formal-readiness gate branch。它不是 C3、不是 C3-Pro、不是 combo route。本阶段目标不是证明最终平均检测精度、FLOPs、延迟、部署安全或论文 claim，而是判断这个 formal gate 是否已经足够严谨，可以进入 remote `PRECHECK_ONLY` 或 `SHORT_DIAGNOSTIC_ONLY`，同时 formal full train 是否仍应保持 locked。

请重点检查以下文件：

1. `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`
2. `tools/bvr_twb/validate_bvr_twb_formal_readiness.py`
3. `tools/bvr_twb/validate_bvr_twb_launch_gate.py`
4. `tools/bvr_twb/validate_bvr_twb_geometry_contracts.py`
5. `tools/bvr_twb/audit_opentad_bvr_twb_pipeline.py`
6. `opentad/acquisition/bvr_twb/open_tad_bridge.py`
7. `opentad/acquisition/bvr_twb/validators.py`
8. `opentad/datasets/transforms/end_to_end.py`
9. `tests/test_bvr_twb_formal_readiness.py`
10. `tests/test_bvr_twb_opentad_pipeline.py`
11. `research-wiki/experiments/DIVERGENT_BVR_TWB_FORMAL_READINESS_GATE_20260630.md`

请同时检查这些文件的相互调用关系、配置继承、测试覆盖、gate 条件是否一致。若 GitHub 页面无法访问、分支不可见、或你没有实际阅读关键文件，请把结论写成 `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`，不要给出 launch 许可。

## 当前研究目标和边界

最终研究目标是 task-aware dynamic temporal acquisition for TAD：按视频、窗口、动作区域或难度动态决定处理多少帧、snippet 或 token，减少持续时间冗余，把更关键的信息送入 detector，并尽量保持或提高 TAD，特别是 high-IoU localization。

固定预算、固定 padded bridge 或 50% 输入只可作为 controlled implementation target / attribution baseline / safety gate，不是最终论文贡献本身。任何 sparse-compute、FLOPs、latency、deployment 或 paper claim 都必须有额外证据；本分支目前不应声称已有这类证据。

BVR-TWB / VOI-BBC 路线意图：

- 用 regret / value-of-information / boundary-benefit-cost 思路做 task-aware sparse acquisition。
- 在 deploy/test 时不能使用 GT、validation/test teacher leakage、raw prediction shortcut、hidden cache decision 或 evaluator shortcut。
- 关键机制应尽量体现：低成本 preview 或 policy 估计哪些时刻/区域更值得获取；将 selected raw frames 在 decode/backbone 前交给 OpenTAD；下游 detector/adapter/head 必须知道真实 temporal geometry、mask、selected indices 或有效观测，而不能把 padding 当作真实观测。
- BVR-TWB 不应混入 C3/C3-Pro/GlobalRank/Interval/combo route 的配置、selector token、checkpoint 假设或 attribution。

## 已知 formal-readiness gate 设计摘要

根据 route report，本分支声称：

- formal full training remains locked。
- short diagnostic / precheck 只产生 formal-readiness evidence，不解锁 full train。
- config 仍使用 VideoMAE-S pretrain path：`pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`。
- pipeline ledger 记录 selected raw frames before decode/backbone handoff，例如：
  - `raw_frame_handoff_stage=pre_decode_selected_raw_frames`
  - `selected_raw_frames_before_decode=true`
  - `decode_input_frame_inds` 的 prefix 必须等于 `selected_frame_inds`
  - `fixed_padded_bridge_sparse_compute_claim=false`
  - `adapter_padding_role=fixed_length_decode_backbone_compatibility_invalid_observation`
  - `adapter_padding_invalid_for_detector=true`
- fixed padded bridge 只允许作为 decode/backbone/adapter 兼容桥，不能声称 sparse compute；padding 不能计为 detector/head 有效观测。
- formal readiness validator 要求 Linux torch geometry artifact、pretrain load evidence、finite `Loss`/`reg_loss`、HeadV3 runtime debug evidence with nonzero regression samples，以及显式：
  - `[bvr_twb_formal_precheck] finite_gradients=true no_skipped_optimizer_step=true no_skipped_reg_head=true`
- 即使 gate 通过，预期输出仍应是：
  - `allowed_next_action=FORMAL_REVIEW_ONLY_FULL_TRAIN_STILL_LOCKED`
  - `formal_train_unlocked=false`
  - `full_train_unlocked=false`
  - `sparse_compute_claim=false`

本地 route report 中记录的 local evidence：

```text
python -m pytest tests/test_bvr_twb_formal_readiness.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_geometry_contracts.py tests/test_bvr_twb_shortdiag.py tests/test_bvr_twb_sparse_forward_audit.py -q

38 passed, 12 skipped in 5.85s
```

```text
python -m py_compile tools/bvr_twb/validate_bvr_twb_formal_readiness.py tools/bvr_twb/validate_bvr_twb_launch_gate.py tools/bvr_twb/audit_opentad_bvr_twb_pipeline.py tools/bvr_twb/validate_bvr_twb_geometry_contracts.py tests/test_bvr_twb_formal_readiness.py

pass
```

```text
python tools/bvr_twb/validate_bvr_twb_launch_gate.py --config configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py --audit-out-dir .tmp_bvr_twb_launch_gate_formalreadiness

{"adapter_bridge_mode": "adapter_fixed_length_padded_bridge", "allowed_next_action": "FINAL_READ_ONLY_REVIEW_THEN_LINUX_PRECHECK_ONLY", "full_train_unlocked": false, "gate_pass": true, "remote_sync_unlocked_by_local_gate": false, "route_label": "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3", "sparse_compute_claim": false}
```

Windows local geometry check reportedly skipped torch runtime because of `c10.dll` load failure, so Linux `--require-torch` evidence remains required before formal readiness can pass.

## Critical Review Questions

请基于 GitHub branch 的代码逐项回答：

1. Implementation 是否真实匹配 BVR-TWB / VOI-BBC 的 regret / value-of-information / boundary-benefit-cost 动态采样设计，还是只是一个 cosmetic wrapper / uniform-like fallback / C3 drift？
2. “true sparse raw-frame handoff” 的说法是否诚实？特别是：如果当前系统仍通过 fixed padded bridge 给 VideoMAE/Adapter 固定长度输入，那么它是否只能声称 selected raw frames are handed off before decode/backbone，而不能声称 end-to-end sparse compute / FLOPs reduction？
3. Formal gate 是否 fail-closed：在缺少 Linux torch geometry、缺少 pretrain load、缺少 finite gradient evidence、缺少 nonzero reg-head runtime evidence、或出现 skipped optimizer/reg-head 时，是否会明确拒绝？
4. `validate_bvr_twb_formal_readiness.py` 是否真的拒绝 NaN/Inf、历史 skip optimizer、missing pretrain、eval/mAP/result marker、formal train unlock marker、deploy/paper/sparse-compute/FLOPs claim 等危险日志？
5. `validate_bvr_twb_geometry_contracts.py` 是否覆盖 source、numpy bridge、Linux torch runtime geometry，并且 `--require-torch` 能在 torch runtime skipped 时 fail closed？
6. `audit_opentad_bvr_twb_pipeline.py`、`open_tad_bridge.py`、`validators.py`、`end_to_end.py` 是否保证 selected indices、valid mask、padding role、decode input frame inds、detector/head 可见性语义一致？
7. 是否存在任何 C3 mixing、C3 selector/config/head token、C3-Pro attribution leakage、combo route 暗合并，或 evaluator / post-processing / raw-prediction shortcut？
8. `tests/test_bvr_twb_formal_readiness.py` 和 `tests/test_bvr_twb_opentad_pipeline.py` 是否覆盖了关键 fail-closed 场景，还是仍有必须补的阻断测试？
9. 在当前证据下，是否允许 remote `PRECHECK_ONLY` 或 `SHORT_DIAGNOSTIC_ONLY`？如果允许，请限定前置条件；如果不允许，请列出 blocking fixes。
10. Formal full train 是否仍必须 locked？是否需要 Linux precheck artifact、subagent final review、human/coordinator decision 或其他材料后才能讨论 full train？

请不要根据“本地测试通过”直接给结论。请以代码检查为主，给出阻断项、非阻断项、必修修复或下一步实验，并明确每一项结论对应的文件或逻辑。

## Required Answer Headings

请严格按以下标题回答，标题英文保持原样：

### Context verdict

说明你是否成功检查了 GitHub branch 和关键文件；若没有，请输出 `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`。

### Model evidence

说明你当前使用的模型/Pro 能力证据；若无法确认，也请明说。

### Inspected GitHub branches/files

列出实际检查过的 GitHub branch、commit 或文件。请不要列未检查文件。

### Verdict

请给出一个清晰标签，优先使用：

- `PASS_ALLOW_REMOTE_PRECHECK_ONLY_FULL_TRAIN_LOCKED`
- `PASS_ALLOW_SHORT_DIAGNOSTIC_ONLY_FULL_TRAIN_LOCKED`
- `FIX_BEFORE_REMOTE_PRECHECK`
- `FIX_BEFORE_SUBAGENT_REVIEW`
- `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`
- `REJECT_ROUTE_DRIFT_OR_LEAKAGE`

### Blocking findings

按严重程度列出阻断问题。每条尽量指向文件、函数、配置键、测试或 gate 条件。

### Non-blocking findings

列出不会阻止 precheck/short diagnostic 但应修正或记录的问题。

### Required fixes or next experiments

列出最小修复树或下一步 evidence tree。请区分 code fix、test fix、Linux precheck artifact、short diagnostic evidence、subagent final review、human/coordinator decision。

### Accepted launch/sync/review/Slurm decision

必须明确回答：

- remote sync 是否允许？
- `PRECHECK_ONLY` 是否允许？
- `SHORT_DIAGNOSTIC_ONLY` 是否允许？
- formal full train 是否仍 locked？
- Slurm full training 是否禁止？
- 是否允许任何 mAP、runtime/FLOPs、deploy 或 paper claim？

请保持中文回答，但保留上面的英文标题和关键 verdict 标签。
