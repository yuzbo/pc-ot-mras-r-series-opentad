# DIVERGENT RBA-RBR Guard Final Pro Transport - 2026-07-01

Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`

Decision: severe-result diagnosis for completed RBA-RBR guard diagnostic child
`1118197.560`.

Latest GitHub evidence branch:

`https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-final-7fdd2dab-20260701`

Branch head after GitHub Contents API sync:

`0fda1af286c5e7a80976cba691e138d5469624a8`

Prompt file:

`research-wiki/experiments/DIVERGENT_RBA_RBR_GUARD_FINAL_SEVERE_PRO_PROMPT_20260701.md`

## Attempt 1

- Command route: Oracle browser, `gpt-5.5-pro`, `--browser-port 9333`.
- Output log:
  `research-wiki/experiments/pro_review_outputs/rba_rbr_guard_final_severe_oracle_9333_20260701_console.log`
- Exit code:
  `research-wiki/experiments/pro_review_outputs/rba_rbr_guard_final_severe_oracle_9333_20260701_exitcode.txt`
- Result: `INVALID/INCOMPLETE`.
- Reason: CLI invocation did not receive matching `--file` attachments.
- Interpretation: no Pro model response was produced.

## Attempt 2

- Command route: Oracle browser, `gpt-5.5-pro`, `--browser-port 9333`.
- Attachments: 21 files, previewed at about 77k tokens before send.
- Output log:
  `research-wiki/experiments/pro_review_outputs/rba_rbr_guard_final_severe_oracle_9333_retry_20260701_console.log`
- Exit code:
  `research-wiki/experiments/pro_review_outputs/rba_rbr_guard_final_severe_oracle_9333_retry_20260701_exitcode.txt`
- Result: `INVALID/INCOMPLETE`.
- Reason: Oracle acquired and released a ChatGPT browser slot, then failed with
  `Unable to locate the ChatGPT model selector button`; Oracle also reported
  that no cookies were applied and login or inline cookies are required.
- Interpretation: browser transport failure, not a technical route rejection.

## CDP Check

Local CDP probes after the Oracle retry:

- `http://127.0.0.1:9333/json/version`: connection failed.
- `http://127.0.0.1:9223/json/version`: connection failed.
- `http://127.0.0.1:9222/json/version`: connection failed.

## Current Gate State

- No valid GPT-5.5 Pro severe-result diagnosis has been harvested.
- RBA-RBR formal/full training remains locked.
- No mAP/runtime/FLOPs/sparse-compute/deploy/paper claim is unlocked.
- Lower-risk local evidence organization and non-training diagnostic design may
  continue, but a post-processing/evaluator-affecting follow-up diagnostic
  should wait for a valid Pro response or an explicit same-turn user override.
