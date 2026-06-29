# DIVERGENT BVR-TWB Geometry Contract Fix 20260630

## 路线与边界

- Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630`
- Owned branch: `codex/divergent-bvr-twb-final-20260630`
- 本阶段唯一 writable code owner: 当前 BVR-TWB route owner。
- 共享主仓 `E:\DeskTop\TAD\temrefuse-tad` 保持只读；本阶段未在共享主仓写代码、切分支、stage、commit、push 或写日志。
- 本报告为 documentation-only follow-up；本轮未改代码、未 stage/commit/push、未 remote/SSH/Slurm、未运行训练或 `tools/test.py`。

## Severe Result 背景

旧稳定分支 `cf662dd` 的 BVR-TWB/VOI-BBC 首次验证出现严重异常：Average-mAP 约 `1.40`，训练 loss 为 finite。GPT-5.5 Pro severe-result 诊断已接受：该结果更像 detector-facing temporal geometry、mask、native-axis 或 post-processing 秒转换集成失败，而不是 BVR-TWB 思路本身无效。

因此 final BVR long train 继续冻结。Pro 要求在任何新训练前完成最小本地证明：BVR detector temporal grid 必须使用 native/detector feature positions；mask/valid length 必须一致；`remap_gt_to_selected_axis=False` 必须保持 native dense axis；post-processing 秒转换必须合理；forced-uniform BVR bridge 必须证明 bridge/head/eval 几何不会因桥接而坍塌。

## 设计目的

本修复包的目的不是提升选择器质量，也不是声明 mAP、runtime、deploy 或 paper claim。它只建立本地 fail-closed 几何合同：

1. loader 在 synthetic dense window `384 -> target 192`、非均匀 selected positions 下，输出 frame inds、raw selected valid length、detector feature positions、detector feature valid length、padding duplicate invalid mask，并保证 val/test 不生成 train value labels。
2. `IrregularActionFormer` 的 BVR 路径必须优先使用 `bvr_twb_detector_feature_positions` / `bvr_twb_detector_feature_valid_len`，不能回退到 raw selected positions、legacy `irregular_selected_positions` 或 arange；BVR 元数据缺失时 fail closed。
3. `remap_gt_to_selected_axis=False` 时 GT 与 HeadV3/IrregularActionFormer 的 native temporal grid 对齐，不落到 selected index axis。
4. BVR post-processing 秒转换在 native-axis 下直接使用 detector/native dense coordinates；若 BVR 元数据存在但 native-axis 未开启，直接报错。
5. forced-uniform BVR bridge sanity 证明：即使选择器退化为均匀桥接，adapter padded input、detector feature centers、mask true count、native-axis GT 和秒转换仍自洽，避免在 selector quality 归因前发生 bridge/head/eval geometry collapse。

## 具体修复文件

- `opentad/models/detectors/irregular_actionformer.py`
  - 新增 BVR metadata 判定。
  - 新增 `_bvr_twb_temporal_grid_from_meta`。
  - BVR route 强制读取 `bvr_twb_detector_feature_positions` / `bvr_twb_detector_feature_valid_len`。
  - 缺字段、`irregular_native_axis=False`、空 detector feature positions、mask true count 与 detector feature position count 不一致、native valid length 不覆盖最后 feature position 时均 fail closed。

- `opentad/models/utils/post_processing/utils.py`
  - `convert_to_seconds` 中增加 BVR native-axis fail-closed gate。
  - BVR 元数据存在但 `irregular_native_axis=False` 时抛错，避免 selected-axis remap 错误污染 seconds conversion。

- `tools/bvr_twb/validate_bvr_twb_geometry_contracts.py`
  - 新增本地 validator。
  - 默认执行 source contract 与 numpy forced-uniform bridge contract。
  - 若 torch 可用，执行 loader/detector/post-processing runtime contract。
  - `--require-torch` 用于 Linux torch 环境 precheck；torch 不可用时必须失败，不能被 Windows skip 当成放行。

- `tests/test_bvr_twb_geometry_contracts.py`
  - 新增 focused tests，覆盖 loader、detector temporal-grid、native-axis GT、post-processing seconds、forced-uniform bridge、validator smoke。
  - 当前 Windows 环境 torch DLL 不可用时，runtime tensor tests 显式 skip；source/numpy/validator tests 仍执行。

## Geometry / Native-Axis / Forced-Uniform Bridge 合同

### BVR detector temporal grid 合同

- BVR route 以 `bvr_twb_*` 元数据识别。
- detector grid center 必须来自 `bvr_twb_detector_feature_positions`。
- detector native valid length 必须来自 `bvr_twb_detector_feature_valid_len`。
- `mask.sum()` 必须等于 detector feature positions 数量。
- padding duplicate 只能作为 fixed-length Adapter compatibility input，不能算作 detector valid observation。
- BVR metadata 缺失时不允许静默回退到 arange、raw selected positions 或 legacy irregular selected positions。

### Native-axis / `remap_gt_to_selected_axis=False` 合同

- BVR formal path 使用 `remap_gt_to_selected_axis=False`。
- GT segments 保持 dense/native coordinate，例如 synthetic `[[104, 143], [238, 290]]` 不被映射到 selected index。
- `irregular_native_axis=True` 是 detector 与 post-processing 的强制条件。
- Head-facing temporal grid 的 centers 是 detector feature positions，而不是 raw selected index list。

### Post-processing seconds 合同

- BVR native-axis 下，proposal segments 被视为 native detector-feature coordinates。
- 秒转换使用 `(segments * snippet_stride + window_start_frame + offset_frames) / fps`。
- 若 BVR 元数据存在但 native-axis 关闭，直接 fail closed，避免 selected-axis interpolation 误用。

### Forced-uniform bridge 合同

- forced-uniform 只用于证明 bridge/head/eval geometry 不会坍塌，不代表 BVR selection 贡献。
- 合成例：`dense_T=384`，raw valid positions `96`，Adapter target frame count `192`，feature stride `2`。
- Adapter input 长度为 `192`；padding duplicate count 为 `96`；padding 不计 valid。
- detector feature valid count 为 `48`，来自每两个 raw positions 的 grouped center。
- 该合同用于确认：如果 selector 被替换为均匀选择，bridge 与 detector geometry 仍然能保持一致；若后续 mAP 仍崩，才可继续定位 head/eval 或其他 detector-facing 问题。

## 已跑本地检查

### Pytest

命令：

```powershell
python -m pytest tests/test_bvr_twb_geometry_contracts.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_validators.py tests/test_bvr_twb_matched_controls.py -q
```

结果：

```text
29 passed, 12 skipped in 3.02s
```

说明：`12 skipped` 来自当前 Windows Python 环境 torch DLL 加载失败；非 torch 的 source/numpy/ledger/bridge/launch-gate 合同已通过。本结果不能替代 Linux torch runtime precheck。

### Validator

命令：

```powershell
python tools/bvr_twb/validate_bvr_twb_geometry_contracts.py
```

结果摘要：

```json
{
  "validator": "bvr_twb_geometry_contracts",
  "source_contract": "passed",
  "numpy_bridge_contract": "passed",
  "dense_T": 384,
  "target_frame_num": 192,
  "raw_valid_k": 96,
  "detector_feature_valid_k": 48,
  "torch_runtime_skipped": true,
  "full_training_unlocked": false,
  "no_training": true,
  "no_video_decode": true,
  "no_metric_claim": true
}
```

torch runtime skip reason:

```text
OSError: [WinError 1114] ... Error loading c10.dll or one of its dependencies.
```

### Py Compile / No-Write Compile Check

首先尝试在不生成 `.pyc` 的前提下调用 `py_compile.compile(..., cfile=os.devnull, doraise=True)`，但 Windows Python 拒绝将 `NUL` 用作 py_compile 输出：

```text
FileExistsError: nul is a non-regular file and will be changed into a regular one if import writes a byte-compiled file to it
```

为遵守“本轮只允许写报告文件”的限制，未生成 `.pyc`、未创建 `__pycache__`。随后执行无文件写入的等价源文件内存编译：

```powershell
python -c "from pathlib import Path; files=['opentad/models/detectors/irregular_actionformer.py','opentad/models/utils/post_processing/utils.py','tools/bvr_twb/validate_bvr_twb_geometry_contracts.py','tests/test_bvr_twb_geometry_contracts.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('in_memory_compile passed:', len(files), 'files')"
```

结果：

```text
in_memory_compile passed: 4 files
```

### Diff Check

命令：

```powershell
git diff --check -- opentad/models/detectors/irregular_actionformer.py opentad/models/utils/post_processing/utils.py tests/test_bvr_twb_geometry_contracts.py tools/bvr_twb/validate_bvr_twb_geometry_contracts.py
```

结果：exit code `0`。

仅有 Git 的 Windows line-ending 提示：

```text
LF will be replaced by CRLF the next time Git touches it
```

未发现 whitespace error。

### Final Read-Only Reviewer

Final read-only reviewer gate: `PASS` / no blocking findings recorded for this local geometry-contract fix stage.

该 PASS 只覆盖本地 source/numpy/focused-test/validator 证据，不解锁 full train、remote sync、Slurm、mAP claim、runtime claim、deploy claim 或 paper claim。

## 当前锁定状态

- full training: 仍然锁定。
- remote sync: 本报告不解锁。
- Slurm: 本报告不解锁。
- `tools/test.py`: 未运行，仍不应用于本阶段 claim。
- mAP claim: 无。
- runtime/FLOPs claim: 无。
- deploy claim: 无。
- paper claim: 无。
- C3/C3-Pro/GlobalRank/Interval/dynamic-budget/PhysicalGrid/PQR/CADF/ABR/MDL: 未触碰，未混合。

## 下一步必须动作

在任何 BVR full train、remote formal launch 或 mAP-producing run 之前，必须在 Linux 且 torch 可用的环境中运行：

```bash
python tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch
```

或等价 precheck，要求至少证明：

1. torch runtime contract 不再 skip；
2. loader 384 -> 192 合同通过；
3. detector 使用 BVR detector feature positions；
4. native-axis GT 与 temporal grid 对齐；
5. BVR post-processing seconds conversion 通过；
6. forced-uniform BVR bridge/head/eval geometry sanity 通过；
7. full_training_unlocked 仍只能由后续 review/precheck gate 明确解锁。

若 Linux torch precheck 失败，继续冻结 BVR full train，并先修复 geometry/mask/native-axis/post-processing 合同。
