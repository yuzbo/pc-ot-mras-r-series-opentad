# Agent Notes

## Current Remote Training

- Use the old AutoDL servers for the active THUMOS14 Adapter + ActionFormer runs:
  - `ssh -p 35407 root@connect.cqa1.seetacloud.com`
  - `ssh -p 25876 root@connect.cqa1.seetacloud.com`
- Active remote project path: `/root/autodl-tmp/OpenTAD_Back_check`.
- Completed config on both servers:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py`.
- 2026-05-17 process check: both servers are idle. There are no `screen`
  sessions, no `tools/train.py` / `tools/test.py` / `torchrun` processes, and
  both RTX 4080 SUPER GPUs report 0 MiB memory use and 0% utilization.
- Do not launch the next long training run until the quality-failure discussion,
  local checks, and read-only code review requirements are satisfied.
- Small isolated `non-finite gradients` events were expected training noise for
  this run. Record them, but do not treat a single event as a blocker if loss
  continues normally.

## Review Policy

- Use Claude CLI for direction discussion and code review.
- Do not use the `claude-review` MCP path unless the user explicitly changes this rule.
- After code review and local fixes, run local checks before any deployment.
- Before launching any innovative experiment, first complete a Claude CLI direction-and-implementation discussion and write the discussion record here.
- The discussion must include the agent's own reasoning and reflection, not just Claude's recommendation. Claude's advice is an input to challenge and refine, not a command to follow blindly.
- A valid discussion record must separate:
  - Claude's concrete claims or objections.
  - The agent's own agreement, disagreement, or partial acceptance, grounded in observed metrics, training contracts, tensor/label semantics, and leakage risk.
  - The final decision rule and the falsifiable metric gate for the next run.
  - What will not be changed in the next experiment, so one-variable ablations remain clean.
- Do not mark a Claude exchange as complete if the agent only copies Claude's conclusion without independent critique.
- If Claude CLI returns a transient error such as API 429, do not treat the discussion as complete and do not launch the innovative experiment until a later successful discussion.

## Current Direction

- Current active experiment: detached quality reranking on Adapter + ActionFormer while keeping random-fixed 50% input unchanged.
- Historical anchors:
  - random-fixed Adapter baseline: about 61.95 Average-mAP.
  - NMS/EMA audited baseline: about 63.75-63.85 Average-mAP.
  - oracle boundary dense Adapter: about 75.50 Average-mAP.
- Next candidate ablation prepared locally but not deployed:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py`.
  This uses `target_mode="positive_max_iou"`, `negative_weight=0.0`, and `loss_normalizer="positive"` to avoid dense negative BCE pressure on the quality head.

## Claude CLI Discussion Record

- A formal Claude CLI discussion is required before launching `input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py` or any other new innovative experiment.
- 2026-05-16: three Claude CLI attempts were made for the positive-max-IoU direction discussion:
  - full direction prompt: API 429 peak-load rejection.
  - shortened direction prompt: API 429 peak-load rejection.
  - minimal availability probe: socket connection closed unexpectedly.
- Because no successful Claude CLI discussion completed, the next innovative experiment must remain pending. Continue supervising the already-running assigned-IoU quality experiments only.

### 2026-05-16 Successful Quality-Target Discussion

- Later Claude CLI discussion completed successfully.
- Claude's main objection: `positive_max_iou` with positive-only supervision and bias-initialized quality near 0.99 can leave background locations high-quality, making quality fusion inert or poorly calibrated.
- Agent reflection:
  - This objection is valid for background calibration: if negatives are not supervised, score fusion may not suppress background proposals.
  - However, dense max-IoU on all valid points is also not automatically correct, because negative classification locations can decode proposals that overlap GT; forcing all such locations through dense quality BCE may conflict with the existing assignment/classification contract.
  - The cleaner next training ablation is therefore not to switch to dense max-IoU immediately, but to keep the assigned-IoU target and reduce negative pressure with `negative_weight=0.25` plus a weighted normalizer. This changes one variable: negative quality supervision strength.
- Final gate:
  - If current detached-quality run is below the random-fixed Adapter baseline (~61.95 Average-mAP), first re-evaluate the same checkpoint with `score_alpha=0` to separate training harm from inference fusion harm.
  - If it is between ~61.95 and the NMS/EMA audited baseline (~63.75-63.85), launch the reduced-negative assigned-IoU training ablation.
  - If it reaches ~63.85-65.20, run inference-only `score_alpha` sweeps before any new training.
  - If it exceeds ~65.20, lock the config and measure seed variance.
- Current decision: do not deploy `positive_max_iou_posonly` yet. Continue the two active runs until first eval, then apply the gate above.

### 2026-05-17 Gemini CLI Discussion After Failed Quality Run

- The detached quality-rescore run completed far below baseline:
  - Server 35407 final Average-mAP: 50.05.
  - Server 25876 final Average-mAP: 49.87.
- Re-evaluating the same final checkpoints with `score_alpha=0.0` did not
  recover performance:
  - Server 35407 alpha-zero Average-mAP: 49.65.
  - Server 25876 alpha-zero Average-mAP: 49.79.
- Therefore the failure is not primarily inference-time `quality^alpha` fusion;
  the trained checkpoint itself is degraded.
- Gemini CLI was run as requested by the user. Long prompts either drifted into
  unrelated repository context or attempted unavailable tool calls, so the
  successful discussion used a compact single-turn prompt with `gemini.cmd
  --skip-trust -e none --output-format text -p "<prompt>"`.
- Gemini's usable recommendations:
  - Do not launch `positive_max_iou_posonly` next, because positive-only
    supervision can leave background proposals high-quality and poorly
    calibrated.
  - Run a neutral branch diagnostic: quality head enabled, `loss_weight=0.0`,
    `score_alpha=0.0`, from scratch.
  - In parallel, run reduced-negative assigned-IoU with
    `negative_weight=0.25` and `loss_normalizer="weighted"`; first evaluate with
    `score_alpha=0.0`, then sweep alpha only if the checkpoint recovers the
    baseline band.
- Agent reflection:
  - Gemini's "gradient leakage" warning is a check item, not established
    evidence; the known code path uses `reg_feat.detach()`.
  - The stronger current hypothesis is that the added quality branch perturbs
    training through loss scale, optimizer/grad clipping, AMP/GradScaler, EMA,
    DDP/static graph behavior, or dense negative quality BCE.
  - The next two-server package should isolate branch integration effects and
    reduced-negative quality supervision before any positive-only target.
- Full record:
  `../research-wiki/experiments/ADAPTER_QUALITY_FAILURE_GEMINI_DISCUSSION_20260517.md`.

### 2026-05-18 Launch Review Update

- Gemini CLI read-only launch review of the final neutral/reduced-negative
  patch returned `LAUNCH_OK`.
- Claude CLI review is still unavailable because the last attempt failed with
  quota error `401 {"message":"额度不足"}`.
- GPT-5-Pro is configured but expensive. Use it only for unresolved critical
  blockers or one-shot comprehensive arbitration with complete context; do not
  use repeated short probe calls for this track.
- Current launch decision: proceed with the two-server package after remote
  preflight and launcher self-checks:
  - 35407: `input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0`
  - 25876: `input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0`

### 2026-05-18 Active Runs

- 35407 launched in screen `adapter_quality_neutral`.
  - Config:
    `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0.py`
  - Log:
    `logs/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0_20260518_011721.log`
  - First epoch sanity:
    `Loss=2.4231 cls_loss=1.3694 reg_loss=1.0537 quality_loss=0.0000`.
- 25876 launched in screen `adapter_quality_neg025`.
  - Config:
    `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0.py`
  - Log:
    `logs/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0_20260518_011722.log`
  - First epoch sanity:
    `Loss=2.8682 cls_loss=1.3693 reg_loss=1.0537 quality_loss=0.4451`.
- Monitor gates:
  - First validation starts at epoch 40 and then every 2 epochs.
  - If either run falls below 30 Average-mAP at first validation, stop that run
    and audit implementation/training loop.
  - If neutral finishes below 60 Average-mAP, treat quality-branch integration
    as unsafe and do not interpret quality-supervision ablations as modeling
    evidence.
  - If neutral recovers and neg025 reaches the random-fixed baseline band,
    perform inference-only `score_alpha` sweep before any further training.
- Consolidated route and gate document:
  `../research-wiki/experiments/ADAPTER_ACTIONFORMER_ROADMAP_20260518.md`.
- Training-system audit:
  `../research-wiki/experiments/ADAPTER_QUALITY_TRAINING_SYSTEM_AUDIT_20260518.md`.
- 2026-05-18 latest monitor:
  - Use local helper:
    `powershell -ExecutionPolicy Bypass -File logs/monitor_adapter_quality_active.ps1 -Tail 20`.
  - 35407 neutral reached epoch 23:
    `Loss=0.5457 cls_loss=0.2914 reg_loss=0.2543 quality_loss=0.0000`.
  - 25876 neg025 reached epoch 23:
    `Loss=0.5779 cls_loss=0.2951 reg_loss=0.2545 quality_loss=0.0284`.
  - Both runs still have only one recorded non-finite-gradient skip at epoch 1
    iter 18 and continue normally.
  - 25876 disk remains tight at about 12G available on `/root/autodl-tmp`.
  - Each epoch 9 checkpoint is about 595M; expected remaining scheduled
    checkpoints should add roughly 3G plus any best checkpoint/logs.
  - Alpha-sweep command generator is prepared at
    `logs/prepare_adapter_quality_alpha_sweep.ps1`; use only after neg025
    recovers baseline gate.
  - Local mAP parser for copied logs:
    `logs/parse_adapter_quality_metrics.ps1`.
  - Remote mAP collector:
    `logs/collect_adapter_quality_remote_metrics.ps1`.
  - Optional first-eval watcher:
    `logs/watch_adapter_quality_first_eval.ps1`.
  - Epoch 9 and epoch 19 checkpoints have been written; each is about 595M.
  - Next scheduled checkpoint is expected after epoch 29.
