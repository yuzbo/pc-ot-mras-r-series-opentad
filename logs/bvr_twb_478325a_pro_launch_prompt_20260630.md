# GPT-5.5 Pro Launch-Gate Prompt: BVR-TWB / VOI-BBC 478325a

请使用 GPT-5.5 Pro，高强度推理，正常速度。请不要用 Fast/priority。请用中文回答。

这是一个 GitHub/file-name based launch-gate review，不依赖 zip 附件。请优先打开并检查下面 GitHub 分支和文件名。如果你没有实际检查 GitHub 分支/文件，请明确回答 `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`，不要给出放行结论。

## 需要决策的问题

请判断：在 `478325af8da10646f747f955a54378d53fffd3ef` 之后，BVR-TWB / VOI-BBC 路线是否可以进入正式 full training；还是必须先做一个短 diagnostic/smoke；或者仍然需要代码/证据修复。

这不是 mAP、部署、FLOPs、runtime 或 paper claim。它只是 full-train launch gate。

## 路线身份与边界

路线标签：

`DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

路线名称：

BVR-TWB / VOI-BBC。VOI-BBC 是 Value-of-Information Boundary Belief Controller；BVR-TWB 是 regret/value-of-information/boundary-wavelet or boundary-witness temporal scaffold 路线。

请专门判断：当前实现是否真的符合 regret/value-of-information/boundary-belief/witness temporal scaffold 的设计目的，而不是只做 metadata correlation、actionness top-k、均匀 scaffold、或 C3/C3-Pro/GlobalRank/Interval/dynamic-budget/ABR/MDL 的变体。

禁止混合：

- 不允许 C3 / C3-Pro / GlobalRank / Interval 混入；
- 不允许 ABR / MDL 混入；
- 不存在 `COMBO_ROUTE_APPROVED`；
- 不允许把这个路线解释成 C3 的自然延续或 combo route；
- 不允许从本包得出 mAP、runtime/FLOPs、deploy 或 paper claim。

## GitHub 分支与 commit

请检查这个分支：

https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-bvr-twb-pretrainfix-20260630

当前实现 commit：

`478325af8da10646f747f955a54378d53fffd3ef`

该 commit 替代过时的 `92ec024d1821897245a56f993ed040d97e894831` package。

## 478325a 相比 92ec024 的关键修复

旧 blocker：

`92ec024` 的正式 BVR 配置中 `model.backbone.custom` 使用 `_delete_=True`，但没有重新声明 VideoMAE-S pretrain，所以继承来的 `custom.pretrain` 可能在 resolved config 里被删除，导致 backbone 无预训练初始化。

新修复：

`478325a` 在 `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py` 中显式恢复：

`pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"`

并且在 launch gate 和测试中 fail-closed：

- config 文本必须有正式 pretrain 声明；
- mmengine resolved config 必须保留 `cfg.model.backbone.custom.pretrain`；
- 测试会删除该 pretrain 行并要求 launch gate 拒绝。

## 必查文件

请至少检查这些文件：

1. `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`
2. `tools/bvr_twb/validate_bvr_twb_launch_gate.py`
3. `tests/test_bvr_twb_opentad_pipeline.py`
4. `research-wiki/experiments/DIVERGENT_BVR_TWB_PRETRAIN_GATE_FIX_20260630.md`

请同时抽查这些路线目的和风险相关文件：

5. `docs/DIVERGENT_BVR_TWB_LOCAL_IMPLEMENTATION_20260629.md`
6. `opentad/acquisition/bvr_twb/types.py`
7. `opentad/acquisition/bvr_twb/boundary_belief.py`
8. `opentad/acquisition/bvr_twb/witness_packets.py`
9. `opentad/acquisition/bvr_twb/budget_controller.py`
10. `opentad/acquisition/bvr_twb/value_predictor.py`
11. `opentad/acquisition/bvr_twb/open_tad_bridge.py`
12. `opentad/acquisition/bvr_twb/validators.py`
13. `opentad/acquisition/bvr_twb/regret_labels.py`
14. `opentad/acquisition/bvr_twb/adapter_bridge.py`
15. `opentad/models/detectors/irregular_actionformer.py`
16. `opentad/models/utils/post_processing/utils.py`
17. `tools/bvr_twb/validate_bvr_twb_geometry_contracts.py`
18. `tests/test_bvr_twb_voi_bbc.py`
19. `tests/test_bvr_twb_validators.py`
20. `tests/test_bvr_twb_geometry_contracts.py`
21. `tests/test_bvr_twb_regression_stability.py`

## 本地证据

本地 focused route tests 记录为：

`26 passed, 12 skipped`

本地 launch gate 通过但 fail-closed：

```json
{"adapter_bridge_mode": "adapter_fixed_length_padded_bridge", "allowed_next_action": "FINAL_READ_ONLY_REVIEW_THEN_LINUX_PRECHECK_ONLY", "full_train_unlocked": false, "gate_pass": true, "remote_sync_unlocked_by_local_gate": false, "route_label": "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3", "sparse_compute_claim": false}
```

请注意：

- `gate_pass=true` 不等于 full train unlocked；
- `full_train_unlocked=false`；
- `remote_sync_unlocked_by_local_gate=false`；
- `sparse_compute_claim=false`。

## 远程 PRECHECK_ONLY 证据

以下为 coordinator 提供的远程证据路径和事实；本 prompt 准备者没有 SSH 验证，因为本任务禁止 SSH：

- N16R4 clean tree: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_PretrainFix_20260630_478325a`
- Log dir: `/data/home/sczc063/run/yuzibo/bvr_twb_pretrainfix_precheck_logs_20260630_478325a`
- Main pass log: `precheck_retry_py310_20260630_065630_+0800.log`
- remote commit exactly `478325a`
- resource links: `data -> ../OpenTAD_Back_check/data`, `pretrained -> ../pretrained`
- required pretrained file exists, about `87M`
- validator passed
- geometry precheck passed
- focused pytest `2 passed`
- no training
- no video decode claim beyond PRECHECK_ONLY
- no mAP
- no runtime/FLOPs claim
- no sparse compute claim

请判断这些证据是否足以放行正式 full train，或者是否仍需要一个短 diagnostic/smoke 来确认 pretrain load、资源、几何、mask、native-axis、Adapter bridge、head stability。

## 已知剩余锁与风险

- true sparse raw-frame handoff / sparse-forward compute 尚未证明；
- BVR 尚未由本地 gate 解锁 long train；
- 没有 metric claim；
- 没有 runtime/FLOPs claim；
- 先前 `cf662dd` formal attempt 严重崩溃，可能被 missing pretrain invalidated，但仍提示 geometry/mask/native-axis/post-processing 风险；
- Slurm 前必须重新检查资源可用性；
- protected hold `1118197` 不能释放、取消、替换或让其自终止，除非用户明确授权该 job/policy；
- 若你认为 full training 不应直接启动，请给出最小 short diagnostic/smoke 树。

## 请按以下固定结构回答

### Context verdict

请说明你是否检查了 GitHub branch、commit 和文件名。若未检查，回答 `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`。

### Model evidence

请说明你是 GPT-5.5 Pro 还是其他模型；如果不是 GPT-5.5 Pro，请明确说明不能作为有效 Pro gate。

### Inspected materials

列出你实际检查过的 GitHub 文件。

### Verdict

请选择并解释一个结论：

- `ALLOW_FORMAL_FULL_TRAIN_AFTER_478325A`
- `REQUIRE_SHORT_DIAGNOSTIC_SMOKE_FIRST`
- `FIX_BEFORE_FULL_TRAIN`
- `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`
- `REJECT_ROUTE_DRIFT_OR_LEAKAGE`

### Blocking findings

列出任何必须修复或必须先跑的 blocker。没有则写无。

### Non-blocking findings

列出建议但非阻塞项。

### Required fixes or next experiments

如果你要求短 diagnostic/smoke，请给出最小命令/检查树和 stop/go 条件。请特别覆盖 pretrain load、resource link、geometry precheck、mask/native-axis、Adapter bridge、HeadV3 stability、no GT/teacher/test leakage、no sparse-compute claim。

### Accepted launch/sync/review/Slurm decision

请明确：

- 是否允许 full train；
- 是否只允许 short diagnostic/smoke；
- 是否禁止 Slurm；
- 是否需要重新检查 N16R4 资源；
- 是否保持 protected hold `1118197` 不释放；
- 是否仍禁止 mAP/runtime/FLOPs/deploy/paper claim。
