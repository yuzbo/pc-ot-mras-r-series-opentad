## 1. `Context verdict`

我已直接检查 GitHub branch `codex/c3-pqr-rankcal-v1-20260629` 与 commit `dca62cc24c0bef50e53135709820b6a66afe1422`。GitHub commit 页面可见该提交为 `C3_PQR_RankCalV1 PRECHECK_ONLY implementation evidence`，包含 7 个新增文件、870 行新增，文件树与 prompt 列出的 changed surface 一致。([GitHub][1])

上下文**足以做 launch-decision gate**，但结论不是通过 runtime：我能确认 PQR route 本身没有 CADF/selector 漂移，也没有明显 teacher/cache/test-GT 捷径；但我发现一个 launch-gate 级阻塞：`workflow.max_train_iters=2` 虽然写入 precheck config 并被测试检查，但标准 `tools/train.py` / `train_one_epoch` 路径没有消费这个字段，因此“2-iter runtime smoke”在当前标准 launcher 下不能被可靠限制为 2 iter。([GitHub][1])

## 2. `Inspected materials`

我实际检查了以下材料：

* GitHub branch 页面：`codex/c3-pqr-rankcal-v1-20260629`。([GitHub][2])
* GitHub commit 页面：`dca62cc24c0bef50e53135709820b6a66afe1422`。([GitHub][1])
* `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py`。([GitHub][3])
* `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py`。([GitHub][4])
* `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py`。([GitHub][5])
* `tools/validate_c3_pqr_rankcal_v1_config.py`。([GitHub][6])
* `tests/test_c3_pqr_rankcal_v1_config.py`。([GitHub][7])
* `tests/test_c3_pqr_rankcal_v1_quality_head.py`。([GitHub][8])
* `research-wiki/experiments/C3_PQR_RANKCAL_V1_IMPLEMENTATION_20260629.md`。([GitHub][9])
* 复用模型路径：`opentad/models/dense_heads/anchor_free_head.py`、`opentad/models/detectors/actionformer.py`、`opentad/cores/train_engine.py`、`tools/train.py`、`opentad/cores/optimizer.py`、base Adapter configs。([GitHub][10])

## 3. `Verdict`

`FIX_BEFORE_RUNTIME`

理由：PQR 机制本身没有看到需要拒绝路线的泄漏或 route drift；但当前 branch **不能可靠执行“2-iter runtime smoke”**，因为 `max_train_iters=2` 只是 config/test 级字段，标准训练入口没有消费它。这个问题属于 launch-gate ambiguity / unconsumed runtime-control key，必须在任何 runtime smoke 前修复。([GitHub][1])

## 4. `Blocking findings`

1. **2-iter runtime smoke 的硬边界当前不成立。**
   Precheck config 中确实设置了 `workflow.max_train_iters=2`、`end_epoch=1`、`disable_checkpoint=True`、`val_eval_interval=-1`，测试也断言了这些字段；但 `tools/train.py` 调用 `train_one_epoch(...)` 时没有传入 `max_train_iters`，`train_one_epoch` 内部按 `len(train_loader)` 完整遍历 epoch，没有看到基于 `cfg.workflow.max_train_iters` 的 break。因此用标准 `tools/train.py` 运行所谓 precheck config，不等价于 2-iter smoke。([GitHub][1])

2. **`remote_launch_locked=True` / `diagnostic_only=True` 当前是元数据，不是 runtime fail-closed gate。**
   配置和 validator 确实保留并检查 `remote_launch_locked=True`、`diagnostic_only=True`、`claim_map_improvement=False`、`official_map_claim=False`；但 `tools/train.py` 只加载 config、build dataset/model、进入 DDP 和训练循环，没有看到对 `pqr_rankcal_v1.remote_launch_locked` 或 metric-claim flags 的 runtime 拦截。这个问题不代表 PQR 技术泄漏，但代表 launch gate 不能仅靠 config 字段自锁。([GitHub][3])

3. **8-epoch short diagnostic 还缺少实际 build/forward/backward 证据。**
   当前 R2 证据是 py_compile、config validators 和 focused pytest；我没有看到本 commit 下已有完整 dataset build、model build、optimizer build、2-step forward/backward、finite loss、quality head parameter update 的 runtime 日志。因此不能直接放行 8-epoch。commit 文档也把该阶段描述为 PRECHECK_ONLY，并明确 no training / no metric claim。([GitHub][9])

4. **未发现足以 `REJECT_ROUTE_DRIFT_OR_LEAKAGE` 的 blocker。**
   配置没有引入 `model.frame_selector`，validator 会拒绝 forbidden route tokens、raw prediction cache、teacher/test-GT、physical-time postprocess claim，并限定 `quality_head_cfg` 为 `max_iou` 小 alpha ranking calibration。这个方向仍是 detector-head ranking calibration，不是 CADF selector / input sampler。([GitHub][6])

## 5. `Non-blocking findings`

PQR changed surface 基本干净：commit 只新增 7 个配置/validator/test/doc 文件，未改 evaluator、NMS、sampler、selector、backbone 或模型实现文件。([GitHub][1])

`quality_head_cfg` 的关键字段在 `AnchorFreeHead` 中确实有消费路径：`enabled`、`loss_weight`、`score_alpha`、`target_mode`、`positive_weight`、`negative_weight`、`loss_normalizer` 等被读取，并校验 `target_mode` 属于 `assigned_iou/max_iou/positive_max_iou`。([GitHub][10])

质量头初始化是相对温和的：`weight_init=0.0`、`bias_init=4.59511985013459` 在 config 中设置，`AnchorFreeHead` 用这些值初始化 `quality_head`，初始 sigmoid 接近 0.99；再配合 `score_alpha=0.10`，初始对分类分数接近 no-op。([GitHub][1])

`max_iou` target 的实现没有直接给回归分支传梯度：`_max_iou_quality_target` 是 `@torch.no_grad()`，并对 `all_pred_segments` / GT 做 detach；这降低了 quality target 对主回归路径的扰动风险。([GitHub][10])

## 6. `Required fixes`

**Before 2-iter smoke：必须修。**

* 在 `opentad/cores/train_engine.py` 中给 `train_one_epoch` 增加可选 `max_train_iters=None` 参数，并在训练循环中执行硬 break，例如 `if max_train_iters is not None and iter_idx + 1 >= max_train_iters: break`。当前代码没有这个 break。([GitHub][11])
* 在 `tools/train.py` 中把 `cfg.workflow.get("max_train_iters", None)` 传入 `train_one_epoch`，否则 precheck config 的 `workflow.max_train_iters=2` 仍然是未消费字段。([GitHub][12])
* 在 `tests/test_c3_pqr_rankcal_v1_config.py` 或新增 runtime-gate test 中增加 fail-closed 检查：precheck config 的 `max_train_iters=2` 必须被标准 launcher 或 dedicated smoke launcher 消费；不能只断言 config 字段存在。现有测试只检查字段值和 validator。([GitHub][7])

**Before 8-epoch short diagnostic：必须先有 2-iter smoke 证据。**

* 2-iter smoke 日志必须证明：dataset build 成功、model build 成功、optimizer build 成功、`quality_loss` 出现在 loss dict、`cls_loss/reg_loss/quality_loss/cost` 全部 finite、无 non-finite grad、训练确实在 2 iter 后停止。`train_engine.py` 已有 non-finite cost/grad 检查，可用于 smoke 日志核验。([GitHub][11])
* shortdiag 前必须明确保留 `diagnostic_only=True`、`claim_map_improvement=False`、`official_map_claim=False`，且任何 validation/eval 输出只能作为 diagnostic signal，不得作为正式 mAP claim。相关 flags 已在 config/validator 中存在，但 launcher 不会自动执行语义锁。([GitHub][4])

**Before formal/full training：仍需更高门槛。**

* 需要完整 8-epoch diagnostic 日志、PQR 与 exact-uniform backend control 的可比性说明、loss/score 分布稳定性、proposal count / quality score 分布、无 evaluator/postprocess 改动证明、以及单独 formal gate。当前材料远远不够解锁 formal/full train。

## 7. `2-iter runtime smoke allowed?`

**No，当前 branch/standard launcher 下不允许。**

原因不是 PQR 方法本身失败，而是“2-iter”这个 runtime contract 当前没有被代码保证。`workflow.max_train_iters=2` 写在 config 里，但 `tools/train.py` 没有传递它，`train_one_epoch` 也没有消费它。([GitHub][1])

修复后允许的 command scope 应限制为：

* 只使用 `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py`；
* 先跑 `tools/validate_c3_pqr_rankcal_v1_config.py`；
* 只允许标准 train smoke 或 dedicated smoke harness 执行 **exactly 2 train iterations**；
* 不允许 `tools/test.py`；
* 不允许 shortdiag config；
* 不允许 checkpoint claim、mAP claim、official result claim；
* 不允许改 `post_processing`、`inference.load_from_raw_predictions`、teacher/test-GT/cache flags。

## 8. `8-epoch short diagnostic allowed?`

**No，当前不允许。**

条件：只有在上述 2-iter runtime smoke 修复并通过后，才能重新评估是否允许 8-epoch short diagnostic。8-epoch 必须仍然是 diagnostic-only；即使 shortdiag config 设置了 `val_eval_interval=2`、`val_start_epoch=4`、`end_epoch=8`，产生的任何 mAP/validation 数字也只能用于诊断 ranking calibration 是否崩坏，不能作为正式提升声明。([GitHub][4])

## 9. `Formal/full train remains locked?`

**Yes，formal/full train 必须继续锁定。**

当前证据只支持“配置与单元测试层面的 PRECHECK_ONLY 有限通过”，不支持 formal/full train。尤其是：标准 runtime smoke 的 2-iter 边界尚未成立；`remote_launch_locked` 只是配置元数据而非 `tools/train.py` 的硬拦截；还没有真实 runtime 的 finite-loss / no-NaN / no-ranking-collapse 证据。([GitHub][3])

## 10. `Hidden NaN/ranking/postprocess risks`

`max_iou` quality targets：实现上相对安全。`_pairwise_segment_iou_1d` 对 union 做 `clamp(min=eps)`，`_max_iou_quality_target` 对 empty GT / no valid mask 做 continue，并在 no-grad/detach 下计算 target；这降低了 NaN 和主回归梯度污染风险。风险主要不是泄漏，而是 `max_iou` target 对所有 valid proposals 监督，可能让 quality head 学到“全局 proposal overlap prior”，需要通过 score 分布和 ranking 分布监控。([GitHub][10])

低 alpha fusion：`get_valid_proposals_scores` 中 quality score 经过 sigmoid、clamp 到 `[1e-6, 1.0]`，再以 `quality_score.pow(score_alpha)` 乘到分类 score 上；`score_alpha=0.10` 比较保守，初始 bias 接近 0.99 时几乎是 no-op。隐藏风险是训练后 quality score 若极端偏低，仍会压缩部分 proposal 分数并改变排序，但低 alpha 与 clamp 使其不太可能直接产生 NaN。([GitHub][10])

Quality loss normalization：`loss_normalizer="valid"` 时 denominator 用 `valid_mask.sum().clamp(min=1)`，因此空 valid 不会除零；`binary_cross_entropy_with_logits` 本身也比先 sigmoid 再 BCE 更稳定。风险是 valid proposals 数量远大于 positives 时，quality loss 可能偏 background/global-overlap calibration，而不是精细 positive-only calibration；这属于 ranking behavior 风险，不是 immediate NaN blocker。([GitHub][10])

Neutral init：`weight_init=0.0`、`bias_init=4.59511985013459` 会让初始 quality score 接近 0.99；配合 `score_alpha=0.10`，初始阶段对 post-NMS 排序扰动很小。风险是初始负样本 BCE 会有一定梯度，但 `loss_weight=0.03` 控制了幅度。([GitHub][1])

`max_seg_num=2000` 与 unchanged post-processing：validator 明确检查 `cfg.post_processing.nms.max_seg_num == 2000` 且不允许 `pvr_qc_diagnostics` 出现在 post_processing；base Adapter config 的 post_processing 仍是 soft-NMS、`max_seg_num=2000`。因此目前没有 evaluator/NMS shortcut。风险是 PQR 改变进入同一 postprocess 的 score scale 与排序，但这正是 detector-head ranking calibration 的诊断对象，不能归因于 CADF/input selection。([GitHub][6])

[1]: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/commit/dca62cc24c0bef50e53135709820b6a66afe1422 "C3_PQR_RankCalV1 PRECHECK_ONLY implementation evidence · yuzbo/pc-ot-mras-r-series-opentad@dca62cc · GitHub"
[2]: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/c3-pqr-rankcal-v1-20260629 "GitHub - yuzbo/pc-ot-mras-r-series-opentad at codex/c3-pqr-rankcal-v1-20260629 · GitHub"
[3]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/configs/adatad/thumos/c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/tools/validate_c3_pqr_rankcal_v1_config.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/tests/test_c3_pqr_rankcal_v1_config.py "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/tests/test_c3_pqr_rankcal_v1_quality_head.py "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/research-wiki/experiments/C3_PQR_RANKCAL_V1_IMPLEMENTATION_20260629.md "raw.githubusercontent.com"
[10]: https://github.com/yuzbo/pc-ot-mras-r-series-opentad/blob/codex/c3-pqr-rankcal-v1-20260629/opentad/models/dense_heads/anchor_free_head.py "pc-ot-mras-r-series-opentad/opentad/models/dense_heads/anchor_free_head.py at codex/c3-pqr-rankcal-v1-20260629 · yuzbo/pc-ot-mras-r-series-opentad · GitHub"
[11]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/opentad/cores/train_engine.py "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/yuzbo/pc-ot-mras-r-series-opentad/codex/c3-pqr-rankcal-v1-20260629/tools/train.py "raw.githubusercontent.com"
