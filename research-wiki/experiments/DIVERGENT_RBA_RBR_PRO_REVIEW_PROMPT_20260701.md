# RBA-RBR Pro Review Prompt - 2026-07-01

请作为 GPT-5.5 Pro / Rosetta Pro 对以下 GitHub 分支做严厉、代码级、路线级审查。不要基于我下面的文字替代码下结论；请优先打开 GitHub 分支和列出的文件逐行/逐函数检查。若你无法访问 GitHub 或无法检查关键文件，请明确返回 `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`，不要假装完成审查。

Repository:

- `https://github.com/yuzbo/pc-ot-mras-r-series-opentad.git`

Branch:

- `codex/divergent-rba-rbr-20260701`

Current commit:

- `87eae60`

GitHub branch URL:

- `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-20260701`

Route label:

- `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`

Route name:

- RBA-RBR: Regret-Based Recoverable Bracketing.

Route purpose:

- 这是一个发散创新路线，不属于 C3。
- 核心目标不是直接预测 boundary，也不是只做均匀覆盖。
- 小模型/低成本预览侧要估计 detector 最可能后悔的位置，即“如果不看这里，最终 TAD detector 最容易丢高 IoU 或错过关键动作/边界的风险”。
- 第一轮 bracket 只能是软先验，不是硬裁剪。后续 probe 必须能在 bracket 外进行 rescue，避免 ABR 那种第一轮漏边界后无法恢复的问题。
- 采样应结合 actionness、uncertainty、transition、staleness/conflict/gap risk，并有 dynamic budget stop reason；需要保留 scaffold/refine/rescue 证据。
- 训练期可以有 train-only regret/value 标签；val/test/deploy 不能使用 GT、teacher/cache/raw-prediction shortcut 或任何验证/测试泄漏。

Please inspect these files first:

- `opentad/acquisition/rba_rbr/__init__.py`
- `opentad/acquisition/rba_rbr/types.py`
- `opentad/acquisition/rba_rbr/risk_map.py`
- `opentad/acquisition/rba_rbr/bracketing.py`
- `opentad/acquisition/rba_rbr/probe_policy.py`
- `opentad/acquisition/rba_rbr/dynamic_budget.py`
- `opentad/acquisition/rba_rbr/regret_labels.py`
- `opentad/acquisition/rba_rbr/selection.py`
- `opentad/acquisition/rba_rbr/adapter_bridge.py`
- `opentad/acquisition/rba_rbr/validators.py`
- `opentad/datasets/transforms/end_to_end.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
- `tools/rba_rbr/build_synthetic_ledgers.py`
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`
- `tests/test_rba_rbr_core.py`
- `tests/test_rba_rbr_integration.py`
- `research-wiki/experiments/DIVERGENT_RBA_RBR_LOCAL_IMPLEMENTATION_20260701.md`

Observed evidence only:

- Local worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`.
- Local branch: `codex/divergent-rba-rbr-20260701`.
- Local latest commit pushed to GitHub: `87eae60`.
- Local tests before push: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `14 passed, 1 skipped` before final commit; later remote environment reports `15 passed`.
- Remote N16R4 route-owned precheck worktree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43`.
- Remote non-GPU precheck: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `15 passed in 19.55s`.
- Remote gate command: `python tools/rba_rbr/validate_rba_rbr_launch_gate.py --config configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --audit-out-dir logs/rba_rbr_precheck_4072d43/rba_rbr_audit`.
- Remote gate output: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- Remote synthetic audit summary: `cases=4`, `k=[9, 9, 10, 6]`, `stop_reasons={'risk_satisfied': 3, 'regret_saturation': 1}`, `recovered_boundary_by_rescue=true`, `hard_bracket_would_miss_boundary=true`.
- No RBA-RBR Slurm job, no training, no evaluation, no `tools/test.py`, no mAP, no runtime/FLOPs, no deploy claim, no paper claim.
- N16R4 GPU state at remote precheck time: BVR child `1118197.519` was running on GPU0; C3 child `1118197.528` was running on GPU1; RBA-RBR did not occupy GPU.

Questions:

1. Does the implementation actually match the RBA-RBR design purpose above, or did it drift into a cosmetic selector/scaffold method?
2. Is the bracket recoverability real? Specifically, can rescue probes operate outside hard bracket assumptions, and does the code avoid ABR's unrecoverable first-round failure mode?
3. Is the deploy-time information boundary clean? Check val/test/deploy GT, teacher, cache, raw-prediction shortcut, and provenance leakage.
4. Is raw-frame handoff / irregular metadata correct? Check raw selected positions versus detector feature centers, `LoadFrames`, `BackboneWrapper`, masks, sorted/unique index contracts, and fixed-length bridge behavior.
5. Are train-only regret/value labels isolated from validation/test/deploy?
6. Is the dynamic budget controller meaningful enough for a short diagnostic, or is it too synthetic/cosmetic to justify GPU time?
7. What are the exact blocking fixes, if any, before `SHORT_DIAGNOSTIC_ONLY`?
8. If short diagnostic is allowed, provide an executable minimal diagnostic experiment plan and any key code/config changes needed before launch.
9. Is formal full training still locked? If not locked, explain exactly why; otherwise state `NOT_READY_FOR_FORMAL_TRAIN`.

Required answer format:

- `Context verdict`: one of `CONTEXT_SUFFICIENT_GITHUB_INSPECTED`, `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`, or `CONTEXT_INSUFFICIENT_MISSING_FILES`.
- `Model evidence`: state the actual model/channel evidence available to you.
- `Inspected materials`: list the GitHub files/commit inspected.
- `Verdict`: use one of `PASS_TO_SHORT_DIAGNOSTIC_ONLY`, `FIX_BEFORE_SHORT_DIAGNOSTIC`, `REJECT_ROUTE_DRIFT_OR_LEAKAGE`, or `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`.
- `Blocking findings`: code-grounded, file/function-specific.
- `Non-blocking findings`: code-grounded.
- `Required fixes or next experiments`: executable and minimal.
- `Accepted launch/sync/review/Slurm decision`: explicitly state whether remote sync, local/remote precheck, short diagnostic, formal full train, and any claims are allowed.
