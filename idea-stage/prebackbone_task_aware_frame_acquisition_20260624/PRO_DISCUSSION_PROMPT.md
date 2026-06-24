# Pro Discussion Prompt

下面是一份可直接给 GPT-5 Pro / GPT-5.5 Pro / Oracle Pro / ChatGPT Pro 的中文 prompt 草稿。它不是要 Pro rubber-stamp 当前 C3-Pro；它要求 Pro 自由批判当前实现，并发散提出更强路线。

---

你是一个极其严厉、实现感很强的 CVPR/ICCV 级 temporal action detection 研究审稿人和系统设计顾问。请用中文回答。不要默认当前实现是正确方向；你可以自由否定、重排或替代当前路线。

## 0. 本次讨论的决策

本次不是请求启动训练，也不是请求写论文 claim。请做三件事：

1. 严厉评审当前 C3-Pro frame-score-first pre-backbone selector 的理论、机制、实现、NaN、训练稳定性、定位几何和性能崩溃风险。
2. 发散提出更创新的 reader / selector / dynamic budget / token-frame hybrid / irregular ActionFormer adaptation 路线。
3. 指出为了做可信判断，你还必须看到哪些关键代码、配置、测试、结果证据。

请不要把 fixed 384/768 或 50% frame budget 当作最终目标。最终目标是 deployable task-aware dynamic temporal acquisition for TAD：系统要按视频、窗口、动作区域、边界敏感度、难度和冗余程度动态分配帧/片段/令牌预算，减少持续时间冗余，保护或提升 TAD 性能，尤其高 IoU 定位。

## 1. Context-integrity check

请先回答以下上下文完整性问题。若你认为上下文不足，请明确列出缺失材料；不要硬给有效结论。

1. 本次提供的上下文是否足够支持当前“路线/实现/风险/下一步”判断？
2. 你是否还需要更多代码、配置、日志、JSON 结果、数据说明、历史结论或 stdout/stderr 才能可靠讨论？
3. 你实际检查了哪些文件、代码片段、结果和报告？
4. 是否存在缺失、不可见、截断、过期、只被口头描述但没有真正提供的关键材料？
5. 你是否确实是本次要求的 Pro 级模型？请说明界面或运行日志中的模型证据。

如果关键代码不可见，请把 verdict 标成 `CONTEXT_INSUFFICIENT_FOR_FINAL_VERDICT`，但仍可提供初步假设和需要索要的材料。

## 2. 当前方法摘要

当前候选称为 C3-Pro / BoundaryDifficulty / frame-score-first pre-backbone selector。

高层机制：

- 输入：低分辨像素/压缩 scout descriptors，来自 dense 768-frame window。
- Reader：temporal encoder / temporal conv stack。
- Dense heads：`action_logits`, `start_logits`, `end_logits`, `boundary_logits`, `uncertainty_logits`, `redundancy_logits`, `frame_selection_logits`。
- Selection：`selection_strategy="frame_score_topk"`，hard top-k 选择 384 个真实帧。
- Gradient：训练时用 straight-through surrogate，把 detector loss 梯度回传到 `frame_selection_logits`。
- Backend：保持原始 AdaTAD/ActionFormer detector head，不启用 P2，不使用 teacher，不使用 raw prediction cache，不使用 validation/test GT。
- Current fixed-budget stage：384/768 是受控实现目标，用来做 attribution/safety/failure diagnosis；不是最终动态 acquisition 目标。

当前配置关键信息：

- C3-Pro config: `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py`
- Reader type: `PCOTMRASBoundaryDifficultyTemporalFrameScout`
- Selection: `frame_score_topk`
- Budget: fixed 384 over dense 768
- selection unit: 1 real frame
- max gap guard/protected uniform guard: disabled in C3-Pro candidate
- Detector: original `ActionFormerHead`

## 3. Baseline and project constraints

Known project references for fixed-budget progress:

- random-fixed Adapter baseline: 63.77
- strict EMA reference: 63.85
- stratified sampling best: 64.64
- uniform stride-2 50% reference: 65.09
- exact-uniform family: 65.57 / 65.73
- residual64 fallback: 65.46
- oracle residual64: 66.61
- oracle-boundary Adapter: about 76-78, recorded as 77.62

Interpretation:

- Recovering exact-uniform-like performance is useful but not sufficient.
- The input-side north star is deployable action/boundary/difficulty-aware allocation that approaches the oracle-boundary insight without test-time GT.
- Any dynamic-budget claim must report compute-performance frontier: average selected frames/snippets/tokens, budget distribution, FLOPs/latency if available, Avg-mAP, mAP@0.6, mAP@0.7, and whether high-IoU localization is preserved.

Hard protocol boundaries:

- No validation/test GT at test time.
- No validation/test teacher leakage.
- No raw-prediction shortcut or hidden cache decision.
- Train-only GT/teacher targets are allowed only if split boundary is explicit and deploy-time inputs remain clean.
- No paper claim from oracle/diagnostic/broken/unaudited runs.

## 4. Candidate idea set to critique and improve

请不要只评 C3-Pro。请同时评估以下发散方向，允许你重排、合并、否定或提出更强新方向。

1. Boundary-risk frame score selector
   - Calibrate `frame_selection_logits` as action/boundary/uncertainty/redundancy utility density.
2. Differentiable temporal transport selector
   - Replace independent hard top-k training with Sinkhorn/OT transport, hard export at inference.
3. Segment-first scout
   - Predict temporal evidence intervals/packets rather than isolated frames.
4. Scout-then-refine multi-round acquisition with one detector forward
   - Scout candidate boundary/action bands, allocate packets, merge, detector forward once.
5. Dynamic budget controller
   - Allocate 288/320/352/384/416 or continuous budget based on boundary risk/redundancy/difficulty.
6. Irregular-time ActionFormer with physical temporal grid
   - Modify point generation, assignment, regression, postprocess to consume selected-frame physical coordinates/cell widths.
7. Dual-stream selector with train-only utility distillation
   - Low-res deploy reader trained from train-only detector utility/sensitivity, no deploy-time teacher.
8. Redundancy-aware token-frame hybrid acquisition
   - Combine frame selection with token/tubelet compression and spend saved compute near boundaries.
9. Acquisition-detector consistency loss
   - Align reader score distribution with detector proposal/boundary utility during train.
10. Uncertainty-triggered fallback selector
   - Reader abstains to exact-uniform/coverage-protective fallback when confidence is low.
11. Boundary-dense action-interior sparse allocation
   - Dense near predicted start/end bands, sparse in stable interiors/background.
12. Proposal-aware low-res reader
   - Predict coarse deployable proposal/actionness density before detector forward.

For each idea you keep or modify, please provide:

- 一句话方法
- 核心假设
- 最小可跑实验
- 贡献类型
- 风险等级
- 预计工程量
- 关键代码需求
- 最可能失败原因
- 如果成功，最合理 paper claim 是什么；如果失败，能得到什么有价值负结果

## 5. Required review tasks

请按以下角度严厉审当前 C3-Pro：

### A. 理论/机制风险

- frame-score-first 是否比 slot-based selector 更合理，还是只是把结构化选择退化成孤立帧排序？
- `action/start/end/boundary/uncertainty/redundancy` heads 是否足以定义 TAD utility？
- hard top-k + ST 的梯度是否会系统性错配真实离散选择？
- 选帧是否会过度选择 action peaks 或 motion bursts，导致边界/内部/上下文不足？
- 固定 384/768 是否掩盖了动态预算问题？
- 当前路线是否有真实 CVPR-level mechanism，还是只是 engineering variant？

### B. 实现/训练风险

- masked logits、AMP、softmax、invalid positions、clamp、ST surrogate 是否有 NaN/Inf 隐患？
- `frame_selection_logits` 到 selected real frames 的索引、mask、valid length 是否严格正确？
- dense-to-selected GT remap 是否破坏边界坐标或导致 high-IoU collapse？
- 是否存在 config inheritance 暗改 detector/head/loss/postprocess？
- launcher gate 是否真的禁止 direct test, checkpoint resume, raw cache, teacher/offline ledger？

### C. detector geometry / ActionFormer risk

- 原始 ActionFormer 是否还能在 non-uniform selected axis 上可靠解释 regression distances？
- 如果 selected positions 非规则，point generation、assignment、regression targets、postprocess 是否需要 physical temporal grid？
- 若不改 ActionFormer，C3-Pro 的性能变化是否可归因于 acquisition？
- 最小什么实验能区分 selector failure 与 detector-geometry failure？

### D. performance collapse / severe-result risk

- 如果 Avg-mAP 掉 5 点以上、mAP@0.7 接近零、loss 正常但 eval 崩溃，应优先查什么？
- 哪些日志/JSON/诊断能最快定位是 NaN、坐标、mask、GT remap、postprocess、selector collapse、还是 detector incompatibility？

## 6. Minimum experiment tree to judge next action

请给出你认可的最小实验树，不要无限增加诊断。要求：

- 1-3 个必要 gates 后就应能决定是否正式跑 fixed-budget mAP 或转向别的路线。
- 必须包含 no-leakage/protocol gate。
- 必须包含 geometry/mask/NaN gate。
- 必须包含至少一个可以区分 C3-Pro、uniform、slot selector、dynamic budget 或 irregular ActionFormer 的高信息实验。
- 请明确哪些动作仍然 locked，哪些动作可以 launch。

## 7. GitHub 地址占位符

请假定代码会通过以下 GitHub 链接或附件提供。请在回答中说明你实际看到了哪些，不要仅根据路径名判断。

- Repository root: `[GITHUB_REPO_URL]`
- Branch: `[GITHUB_REPO_URL]/tree/codex/prebackbone-pcotmras-scout-selector`
- Compare against base clean branch: `[GITHUB_REPO_URL]/compare/[BASE_CLEAN_COMMIT_OR_BRANCH]...codex/prebackbone-pcotmras-scout-selector`
- Current HEAD: `12f84d56d15f15c284f08431bb64d5de49df8185`
- C3-Pro implementation diff: `[GITHUB_REPO_URL]/compare/[BASE]...12f84d56d15f15c284f08431bb64d5de49df8185`

Key file links to inspect:

- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/configs/adatad/thumos/pc_ot_mras_prebackbone_c3_hybrid_reader_full_train_candidate_n16r4.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/configs/adatad/thumos/pc_ot_mras_prebackbone_c3_f1_lr_tinytransformer_st_original_adatad_full_train_candidate_n16r4.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/scripts/run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/opentad/models/detectors/actionformer.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/opentad/models/dense_heads/actionformer_head.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/opentad/models/utils/temporal_grid.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/opentad/cores/train_engine.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/opentad/utils/training_guard.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/tests/test_pc_ot_mras_prebackbone_pro_reader_design.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/tests/test_pc_ot_mras_prebackbone_nan_guards.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/tests/test_pc_ot_mras_prebackbone_selector_behavior.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/tests/test_pc_ot_mras_actionformer_selected_axis_point_geometry.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/tests/test_pc_ot_mras_actionformer_selected_axis_target_assignment.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/tests/test_pc_ot_mras_actionformer_result_detection_geometry.py`
- `[GITHUB_REPO_URL]/blob/codex/prebackbone-pcotmras-scout-selector/PC_OT_MRAS_R16_R18_R20_CLEAN_MANIFEST.md`

Result/evidence placeholders to attach if available:

- `[RESULTS_URL_OR_FILE]` C3-Pro local pytest output
- `[RESULTS_URL_OR_FILE]` precheck summary JSON
- `[RESULTS_URL_OR_FILE]` first full-train stdout/stderr tail
- `[RESULTS_URL_OR_FILE]` validation mAP vector
- `[RESULTS_URL_OR_FILE]` selected-frame diagnostics: boundary support, redundancy, max gap, role counts
- `[RESULTS_URL_OR_FILE]` selected-axis coordinate audit
- `[RESULTS_URL_OR_FILE]` no-leakage/static protocol audit

## 8. Required final answer schema

请严格按以下结构回答：

1. `Context verdict`
   - SUFFICIENT / INSUFFICIENT / PARTIAL
2. `Model evidence`
   - 说明你是否为 Pro 级模型，以及证据。
3. `Inspected materials`
   - 明确列出你真正看过的代码/配置/测试/结果。
4. `Current C3-Pro verdict`
   - PASS_TO_PRECHECK / REVISE_BEFORE_PRECHECK / BLOCK_LONG_RUN / RETHINK_ROUTE / CONTEXT_INSUFFICIENT_FOR_FINAL_VERDICT
5. `Blocking findings`
   - 按严重程度列出，必须引用文件/函数/逻辑位置。
6. `Non-blocking findings`
7. `Risk analysis`
   - 理论、实现、NaN、ST、geometry、performance collapse、leakage、attribution。
8. `Divergent idea ranking`
   - 对 8-12 个 idea 排序，允许新增/合并/删除。
9. `Minimum experiment tree`
   - 只列真正必要的 gates 和 launch order。
10. `Required fixes or requested code`
   - 如果材料不足，明确索要。
11. `Accepted next action`
   - 可以做什么，不可以做什么，哪些仍 locked。

请保持严厉。不要为了礼貌保留当前路线。如果你认为 frame-score-first、slot selector、dynamic budget、ActionFormer irregular adaptation 中任一方向不成立，请直接说，并说明最小反证实验。
