# Adapter + ActionFormer Completion Audit, 2026-05-18

## Restated Objective

The active objective is to improve Adapter + ActionFormer performance on two
servers by:

1.整理当前行进路线、未完成实验和方向。
2.完整分析已完成实验暴露的算法问题。
3.使用外部模型讨论或交叉审查提出更好的优化方向。
4.定位问题并制定计划。
5.完整实现代码、检查代码、部署到两个服务器并启动实验。
6.分别推进 Adapter-side 和 ActionFormer/head-side 更高性能路线。
7.最终不能只启动实验；需要真实评估证明性能提升，或给出被证伪后的下一步。

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Read project instructions and current process | `OpenTAD_Back/agent.md`; remote screen/log checks | done |
| Check running server experiments | SSH checks on 35407/25876; active `adapter_quality_*` screens | done |
| Do not request permissions | Commands run under current `approval_policy=never`; no escalation used | done |
| Configure `llm-chat.chat` for GPT-5-Pro | `~/.codex/config.toml` has `llm-chat` env with `gpt-5-pro`, Responses API, high reasoning | done |
| Avoid repeated expensive GPT-5-Pro calls | No new GPT-5-Pro probe was made after configuration check | done |
| Use Gemini CLI discussion | Attempted; 2026-05-18 follow-ups were unusable and documented as invalid review | partially done |
| Use stronger external discussion | GPT-5.5 xhigh read-only advisor completed; concrete critique integrated | done |
| Analyze completed failed experiments | Detached quality failure, alpha-zero re-eval, invalid bs8 diagnostics documented | done |
| Identify algorithm problems | Quality integration risk, dense negative BCE, high-tIoU localization, input-selection headroom, validity fragility documented | done |
| Implement code | Quality-head diagnostics, batch-size checks, regloss config, queue guard scripts | done for current package |
| Run local checks | `pytest tests/test_adapter_quality_rescore_contracts.py tests/test_adapter_safety_contracts.py -q` -> `28 passed`; PowerShell parser checks passed | done |
| Deploy to servers | Active configs/launchers deployed earlier; queue guard script deployed to both servers with hash check | done |
| Start experiments on two servers | Clean bs2 neutral on 35407 and neg025 on 25876 running; q64/regloss queued behind sentinels | done |
| Adapter-side higher-performance route | `pseudo_boundary_snap_q64` queued on 35407, but not approved until quality gate is interpreted | incomplete |
| ActionFormer/head-side higher-performance route | `regloss15` queued on 25876, but not approved until quality gate is interpreted | incomplete |
| Prove higher performance | No clean bs2 first eval yet; no final improved metric exists | missing |
| Completion gate | Objective cannot be marked complete until real eval evidence supports or redirects the route | not achieved |

## Current Evidence Snapshot

Latest gate dry-run:

```text
neutral: STATUS=NO_EVAL_YET
neg025: STATUS=NO_EVAL_YET
DECISION=WAIT_FOR_NEUTRAL_FIRST_EVAL
```

Latest remote state before this audit:

- 35407 neutral: running, clean bs2, around epoch 24, no `Average-mAP` yet.
- 25876 neg025: running, clean bs2, around epoch 23, no `Average-mAP` yet.
- 25876 disk: about 9.1GB free on `/root/autodl-tmp`.
- Queued q64/regloss screens are guarded by approval sentinels and cannot
  launch automatically.
- Subsequent monitoring hit transient SSH gateway failures on both AutoDL
  ports: TCP was reachable, but SSH KEX was closed by the remote host. This is
  a monitoring blocker only; it is not evidence of training failure.
- 2026-05-18 09:29 and a later 5-minute backoff retry still failed on both
  ports with the same KEX-close symptom, so remote evidence remains unavailable.

## Missing Or Weakly Verified Items

1. No clean bs2 first validation result yet.
   - This blocks interpretation of neutral preservation.
   - It also blocks interpretation of neg025 quality supervision.

2. No proof of improved Adapter-side performance yet.
   - q64 is queued but intentionally gated.
   - It must not launch merely because a quality run exits.
   - Local q64 launcher now has manifest provenance checks in commit `1b042d8`,
     but this still needs remote sync/check-only once SSH recovers.
   - Recovery helper prepared at
     `logs/sync_adapter_followup_guards_after_ssh.ps1`.

3. No proof of improved ActionFormer/head-side performance yet.
   - regloss15 is queued but intentionally gated.
   - It must not launch until quality-route interpretation is written.

4. Gemini CLI is currently unreliable.
   - The earlier Gemini discussion was useful, but the latest current-context
     attempts were invalid.
   - Treat GPT-5.5 critique as the current usable external review.

5. Top-level monitoring/report files are untracked in the outer repository.
   - This is acceptable as working documentation, but should be committed or
     otherwise preserved later if the user wants a fully clean archival state.

## Continue Criteria

Continue monitoring until at least the first clean bs2 validation appears.

After first eval:

1. Run:

```powershell
powershell -ExecutionPolicy Bypass -File logs/collect_adapter_quality_remote_metrics.ps1 -Tail 2600 |
  powershell -ExecutionPolicy Bypass -File logs/evaluate_adapter_quality_gate.ps1
```

2. Write raw metrics and gate decision into:

- `OpenTAD_Back/agent.md`
- `research-wiki/experiments/ADAPTER_ACTIONFORMER_EXECUTION_CONTROL_20260518.md`
- `research-wiki/experiments/ADAPTER_ACTIONFORMER_NEXT_ROUTES_20260518.md`

3. Only create queued approval sentinels if the written gate permits it.

## Verdict

The objective is not complete. Implementation, checks, deployment, monitoring,
and queue hardening are complete for the current diagnostic package, but the
central performance objective has not been achieved or falsified because clean
bs2 evaluation metrics are still pending.
