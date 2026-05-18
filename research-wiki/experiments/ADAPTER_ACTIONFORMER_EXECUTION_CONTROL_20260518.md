# Adapter + ActionFormer Execution Control, 2026-05-18

## Objective

Run separate Adapter-side and ActionFormer/head-side optimization tracks on
the two AutoDL servers. The immediate target is to exceed the current
random-fixed Adapter baseline while preventing invalid runs from being
misread as model-route successes or failures.

Remote workspace:

- 35407: `/root/autodl-tmp/OpenTAD_Back_check`
- 25876: `/root/autodl-tmp/OpenTAD_Back_check`

Baseline anchors:

| Anchor | Average-mAP | Role |
|---|---:|---|
| random-fixed Adapter first eval | 61.95 | preservation gate |
| NMS/EMA audited baseline | 63.75-63.85 | competitive gate |
| oracle boundary dense Adapter | 75.50 | input-selection headroom |

## Current Server Assignment

| Server | Active run | Route | Current status | Next queued run |
|---|---|---|---|---|
| 35407 | `adapter_quality_neutral` | quality-head structural diagnostic | clean bs2, running, no eval yet | `adapter_pseudo_snap_q64_after_quality` |
| 25876 | `adapter_quality_neg025` | reduced-negative quality supervision | clean bs2, running, no eval yet | `adapter_regloss15_after_quality` |

Latest checked state at 2026-05-18 07:22:

- 35407 neutral reached epoch 16; latest complete train line is epoch 15
  `[00099/00099]`, `Loss=0.5946`, `quality_loss=0.0000`.
- 25876 neg025 reached epoch 15; latest train line is epoch 15 `[00050/00099]`,
  `Loss=0.6272`, `quality_loss=0.0251`.
- No first validation has appeared in either clean bs2 log.
- 25876 disk remains the operational risk: `/root/autodl-tmp` has about 9.7GB
  free.

Monitoring note at 2026-05-18 08:10:

- Direct SSH temporarily started failing on both ports with
  `kex_exchange_identification: Connection closed by remote host`.
- `Test-NetConnection` still reports both TCP ports reachable, so this looks
  like SeetaCloud proxy/session throttling or a transient SSH gateway issue,
  not evidence that training crashed.
- `logs/collect_adapter_quality_remote_metrics.ps1` now emits structured
  `REMOTE_STATUS=SSH_FAILED` and retries; `logs/evaluate_adapter_quality_gate.ps1`
  maps this to `DECISION=RETRY_REMOTE_MONITORING`.
- Do not make experiment decisions from an unreachable monitor result.
- 2026-05-18 09:29 and a 5-minute backoff retry still failed on both ports
  with the same KEX-close symptom. Treat remote monitoring/deployment as
  blocked until the AutoDL SSH gateway or instances become reachable again.

Queue safety update at 2026-05-18 07:47:

- A GPT-5.5 xhigh read-only critique flagged that "screen disappearance" is
  not a valid success gate: a crash, manual stop, or invalid run would also
  make the waiting screen disappear.
- Added and deployed `scripts/wait_for_screen_and_gate_then_run.sh` to both
  servers. Remote `bash -n` passed and the deployed hash is
  `da62e91ce475f6a6ebc34e3469acfda922637fb27b2893da15df68be8d799431`.
- Restarted both queued screens so they require two conditions before launch:
  the quality screen must be absent, and a gate approval sentinel must exist.
- Local commit recording the queue guard:
  `c852982 guard queued adapter followups`.
- Approval sentinels:
  - 35407:
    `/root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_pseudo_snap_q64_after_quality.ok`
  - 25876:
    `/root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_regloss15_after_quality.ok`
- Do not create these files until the quality-run gate has been written into
  the experiment record.

Pseudo-boundary cache provenance update:

- Local commit `1b042d8 guard pseudo boundary cache manifests` adds launcher
  checks to `scripts/run_adapter_pseudo_boundary_snap_pair.sh`.
- `CHECK_ONLY=1` now validates existing train/validation cache manifests when
  `SKIP_CACHE_BUILD=1`:
  - `uses_gt` must be false;
  - axis must be `global_snippet_index`;
  - source must be `postprocessed_teacher_detections`;
  - subset must match `training` or `validation`;
  - `videos_written` must be positive.
- Local verification: `pytest tests/test_adapter_safety_contracts.py
  tests/test_adapter_quality_rescore_contracts.py -q` passed with `29 passed`,
  and `bash -n scripts/run_adapter_pseudo_boundary_snap_pair.sh
  scripts/wait_for_screen_and_gate_then_run.sh` passed.
- Remote sync is still pending because the AutoDL SSH gateway is currently
  closing KEX before authentication. Do not approve q64 until this launcher is
  synced and `CHECK_ONLY=1 START_INDEX=1 END_INDEX=1 SKIP_CACHE_BUILD=1` passes
  on 35407 again.
- Recovery helper prepared:
  `logs/sync_adapter_followup_guards_after_ssh.ps1`. Once SSH recovers, run it
  to sync the queue guard and q64 launcher, execute remote `bash -n`, rerun q64
  `CHECK_ONLY=1`, and confirm that no approval sentinel was created.

Verification after hardening:

- `pytest tests/test_adapter_quality_rescore_contracts.py tests/test_adapter_safety_contracts.py -q`
  passed with `28 passed`.
- PowerShell parser checks passed for:
  - `logs/evaluate_adapter_quality_gate.ps1`
  - `logs/watch_adapter_quality_first_eval.ps1`
  - `logs/collect_adapter_quality_remote_metrics.ps1`
  - `logs/monitor_adapter_quality_active.ps1`
- Remote queued screens were checked after restart; no approval sentinel exists,
  so q64/regloss15 cannot launch automatically.

## Completed Evidence And What It Means

### 1. Detached quality-rescore failure

Config:

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py`

Final results:

| Server | Average-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 |
|---|---:|---:|---:|---:|---:|---:|
| 35407 | 50.05 | 69.69 | 62.39 | 51.87 | 40.13 | 26.16 |
| 25876 | 49.87 | 70.44 | 61.89 | 51.39 | 40.55 | 25.10 |

Alpha-zero re-evaluation of the same checkpoints:

| Server | Average-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 |
|---|---:|---:|---:|---:|---:|---:|
| 35407 | 49.65 | 69.70 | 61.79 | 51.53 | 39.70 | 25.53 |
| 25876 | 49.79 | 70.49 | 61.71 | 51.24 | 40.33 | 25.18 |

Interpretation:

- Failure is not only inference-time `score * quality^alpha`.
- The checkpoint itself was degraded during training.
- However, this does not yet prove that all quality-head routes are bad,
  because later diagnostics found a separate batch-size contract violation.

### 2. Invalidated bs8 diagnostics

The 2026-05-18 early neutral/neg025 diagnostics produced only `37-39`
Average-mAP at first/second eval. These results are invalid as quality-head
evidence because the runs used:

- `batch_size=8`
- `24` train iterations per epoch

The healthy random-fixed Adapter baseline used:

- `batch_size=2`
- `99` train iterations per epoch

Therefore the low bs8 results are evidence of a training-contract bug, not a
model conclusion.

Fix:

- Commit `a5e476c fix adapter quality batch size contract`
- Restored Adapter base train/val/test `batch_size=2`.
- Added launcher assertion `EXPECT_BATCH_SIZE=2`.
- Remote `CHECK_ONLY=1` confirmed `batch_size= 2 2 2` before relaunch.

### 3. Quality-head code-path audit

Evidence from `opentad/models/dense_heads/anchor_free_head.py`:

- Quality head uses `reg_feat.detach()` in train/test paths, so quality loss
  should not directly backpropagate into base detector features.
- `score_alpha=0.0` disables quality-score fusion, so current first evals
  measure checkpoint health, not reranking.
- `quality_keep_loss_graph_when_weight_zero=True` keeps the neutral quality
  head in the graph while contributing zero scalar loss.
- Weighted normalizer with `negative_weight=0.25` reduces dense negative BCE
  pressure but does not remove background calibration.

Residual risks:

- AMP/GradScaler skip behavior can still affect the whole optimizer step.
- EMA and AdamW include quality-head parameters, though this should not affect
  neutral outputs while `score_alpha=0.0`.
- Non-finite gradient skips have occurred early in both clean bs2 runs. They
  are monitor items, not a stop condition unless they repeat or correlate with
  validation collapse.

## Current Experiments

### 35407: Neutral Quality Branch

Config:

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0.py`

Purpose:

- Test whether adding the detached quality branch and keeping it in the DDP
  graph changes base Adapter + ActionFormer training, even with no quality
  objective and no score fusion.

Key settings:

- `quality_head_cfg.enabled=True`
- `keep_loss_graph_when_weight_zero=True`
- `loss_weight=0.0`
- `score_alpha=0.0`

Decision:

- If neutral does not recover near baseline, do not interpret neg025 as quality
  supervision evidence. Debug training-system/branch integration first.

### 25876: Reduced-Negative Assigned-IoU Quality

Config:

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0.py`

Purpose:

- Test whether gentler assigned-IoU quality supervision can preserve checkpoint
  health before any inference alpha sweep.

Key settings:

- `target_mode="assigned_iou"`
- `negative_weight=0.25`
- `loss_normalizer="weighted"`
- `loss_weight=0.10`
- `score_alpha=0.0`

Decision:

- Interpret only after neutral passes preservation gate.
- If checkpoint recovers baseline, run inference-only alpha sweep.
- If checkpoint remains low while neutral recovers, quality target/loss design
  is the likely failure mode.

## Queued Experiments

### 35407 Queue: Pseudo-Boundary Snap Q64

Screen:

- `adapter_pseudo_snap_q64_after_quality`

Config:

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_snap_q64.py`

Purpose:

- Adapter-side attempt to close input-selection headroom suggested by the
  oracle boundary dense result.
- Uses teacher pseudo-boundary cache and falls back to random-fixed where cache
  evidence is absent.

Why reasonable:

- It changes sampling/input selection, not the ActionFormer head.
- It targets the largest known headroom source.
- Historical q32/q64 low results used the invalid bs8/24-iteration contract and
  are not decisive after the bs2 fix.

Risk:

- Teacher cache may encode teacher bias; validation sampling must not use GT.
- Must confirm first eval under restored bs2 before deciding.
- If q64 improves, label the protocol carefully: because val/test sampling also
  uses pseudo-boundary cache, it is a teacher-assisted input-selector protocol
  unless a companion random-fixed evaluation also confirms the gain.

### 25876 Queue: ActionFormer Regression Loss 1.5

Screen:

- `adapter_regloss15_after_quality`

Config:

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_regloss15.py`

Purpose:

- Minimal ActionFormer-side localization calibration by increasing regression
  loss weight to `1.5`.

Why reasonable:

- It is one-variable and head-side only.
- It targets the persistent high-tIoU weakness without changing sampling,
  assignment, backbone, quality head, or inference.

Risk:

- Average-mAP can improve at high tIoU while sacrificing low tIoU. Judge the
  full mAP@0.3-0.7 vector, not only Average-mAP.

## Gate Logic

First validation is expected after training epoch `[041]`, because
`tools/train.py` checks:

- `epoch >= val_start_epoch`
- `(epoch + 1) % val_eval_interval == 0`

Gates:

| Condition | Action |
|---|---|
| first eval Average-mAP < 30 | stop that run and audit implementation |
| neutral latest Average-mAP < 60 | stop interpreting quality-supervision variants |
| neutral >= 61.95 and neg025 missing | wait for neg025 |
| neutral >= 61.95 and neg025 < 60 | quality branch likely safe, quality loss route weak |
| neutral >= 61.95 and neg025 >= 61.95 | prepare inference-only alpha sweep |
| neg025 >= 63.75-63.85 | competitive; run alpha sweep and seed variance |
| neg025 > 65.20 | lock config, measure seed variance before adding mechanisms |

Additional hard pre-mAP gates:

- `batch_size=2` for train/val/test.
- `99` train iterations per epoch.
- Expected sampling method and cache provenance for the active route.
- No abnormal non-finite skip frequency. A small number of early skips is a
  monitor item; repeated skips or step-collapse is a stop/audit condition.

Current helper:

```powershell
powershell -ExecutionPolicy Bypass -File logs/collect_adapter_quality_remote_metrics.ps1 -Tail 200 |
  powershell -ExecutionPolicy Bypass -File logs/evaluate_adapter_quality_gate.ps1
```

Watcher:

```powershell
powershell -ExecutionPolicy Bypass -File logs/watch_adapter_quality_first_eval.ps1 -IntervalSeconds 600 -MaxChecks 24
```

Dry-run status at 2026-05-18 07:22:

```text
neutral: STATUS=NO_EVAL_YET
neg025: STATUS=NO_EVAL_YET
DECISION=WAIT_FOR_NEUTRAL_FIRST_EVAL
```

## Current Algorithm Problems

1. Quality-head route is not yet proven safe.
   - The first failed checkpoints stayed near `50` even with `score_alpha=0`.
   - Clean bs2 neutral is the required preservation test before any modeling
     conclusion.

2. Dense negative quality supervision is suspicious.
   - The failed route used dense negatives with `negative_weight=1.0` and
     `loss_normalizer="valid"`.
   - The current `negative_weight=0.25` run tests a smaller background pressure,
     but still needs neutral to pass first.

3. High-tIoU localization is still the main metric bottleneck.
   - Prior results show strong low-tIoU but weak mAP@0.7.
   - This supports the queued regloss15 route, but only as a conservative
     calibration, not a guaranteed improvement.

4. Input selection has real oracle headroom.
   - Oracle boundary dense Adapter at about `75.50` Average-mAP suggests the
     model is not only head-limited.
   - Pseudo-boundary snap is the most direct non-GT attempt queued now.

5. Experiment validity is fragile.
   - The bs8 bug shows that config inheritance can silently change epoch
     semantics.
   - Every future launcher needs explicit checks for batch size, eval cadence,
     score fusion settings, and one-variable isolation.

6. Queued follow-up automation can bypass scientific gates if not guarded.
   - The original waiting wrappers launched as soon as a screen disappeared.
   - The corrected wrappers now require explicit approval sentinel files after
     a written gate interpretation.

## Immediate Plan

1. Continue current clean bs2 diagnostics until first eval unless crash, OOM,
   repeated non-finite skips, or disk exhaustion occurs.
2. Use the gate script above as soon as any `Average-mAP` appears.
3. Do not start alpha sweeps until neutral passes and neg025 recovers baseline.
4. Keep queued pseudo-boundary/regloss wrappers waiting behind approval
   sentinels; they are intentionally post-quality runs to avoid GPU contention
   and route confounding.
5. After first eval, update this report, `OpenTAD_Back/agent.md`, and
   `ADAPTER_ACTIONFORMER_NEXT_ROUTES_20260518.md` with raw numbers and the gate
   decision.

Approval commands, only after the written gate permits launch:

```bash
# 35407, Adapter-side q64
touch /root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_pseudo_snap_q64_after_quality.ok

# 25876, ActionFormer-side regloss15
touch /root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_regloss15_after_quality.ok
```

## External Review Status

- Gemini CLI produced a useful earlier discussion for the failed quality route,
  but the 2026-05-18 follow-up attempts ignored the supplied context and are
  not counted as valid review.
- Claude CLI is unavailable due quota errors.
- `llm-chat` is configured for `gpt-5-pro` through the Responses API with high
  reasoning and 8192 max output tokens, but should be used only for one
  complete, critical arbitration prompt.
- A GPT-5.5 xhigh read-only advisor review returned a useful critique. Accepted
  fixes:
  - harden queued wrappers with explicit gate approval sentinels;
  - add hard pre-mAP validity gates beyond the weak `<30` collapse threshold;
  - treat q64 gains as teacher-assisted unless a companion non-teacher eval
    supports the same conclusion;
  - require alpha sweep gains to improve Average-mAP by at least `0.5` without
    hurting low-tIoU bands by more than `0.5`;
  - require regloss15 to improve Average-mAP by at least `0.5` and mAP@0.7 by
    at least `1.0` without losing mAP@0.3/0.4.
