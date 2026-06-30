# RBA-RBR Pro Review Transport Evidence - 2026-07-01

Route label:

- `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.

Decision requested:

- GPT-5.5 Pro / Rosetta Pro route-level GitHub review for RBA-RBR implementation correctness, design-purpose alignment, leakage risk, metadata/raw-frame handoff correctness, and short-diagnostic/full-train launch decision.

Prompt path:

- `research-wiki/experiments/DIVERGENT_RBA_RBR_PRO_REVIEW_PROMPT_20260701.md`.

GitHub target:

- `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-20260701`
- Commit at submission time: `87eae60`.

## Attempts

1. Rosetta Pro inline prompt
   - Command family: `rosetta run --pro --port 9333 --host 127.0.0.1 --recall divergent-rba-rbr-20260701`.
   - Exit code: `1`.
   - Stdout path: `research-wiki/experiments/pro_review_outputs/rba_rbr_rosetta_pro_20260701_stdout.md`.
   - Stderr path: `research-wiki/experiments/pro_review_outputs/rba_rbr_rosetta_pro_20260701_stderr.log`.
   - Failure: send button click registered, but ChatGPT did not issue `/backend-api/f/conversation` within 120 seconds.
   - Verdict: `INVALID_TRANSPORT_NO_PRO_DECISION`.

2. Oracle browser attach to Chrome 9333
   - Command family: `oracle --engine browser --model gpt-5.5-pro --browser-attach-running --remote-chrome 127.0.0.1:9333`.
   - Exit code: `1`.
   - Stdout path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_20260701_stdout.log`.
   - Stderr path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_20260701_stderr.log`.
   - Failure: no running browser with attach metadata matched `127.0.0.1:9333`.
   - Verdict: `INVALID_TRANSPORT_NO_PRO_DECISION`.

3. Oracle browser attach retry with `--browser-port 9333`
   - Command family: `oracle --engine browser --model gpt-5.5-pro --browser-attach-running --browser-port 9333 --remote-chrome 127.0.0.1:9333`.
   - Exit code: `1`.
   - Stdout path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_20260701_retry_stdout.log`.
   - Stderr path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_20260701_retry_stderr.log`.
   - Failure: Oracle rejects combining `--browser-attach-running` with `--browser-port`.
   - Verdict: `INVALID_TRANSPORT_NO_PRO_DECISION`.

4. Oracle browser with copied Rosetta profile
   - Command family: `oracle --engine browser --model gpt-5.5-pro --browser-port 9333 --copy-profile C:\Users\skywalker\.rosetta\chrome-profile-9333`.
   - Exit code: `1`.
   - Stdout path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_9333_copyprofile_20260701_stdout.log`.
   - Stderr path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_9333_copyprofile_20260701_stderr.log`.
   - Failure: Oracle rejects combining `--copy-profile` with its manual-login browser mode.
   - Verdict: `INVALID_TRANSPORT_NO_PRO_DECISION`.

5. Rosetta Pro attachment prompt
   - Command family: `rosetta run --pro --port 9333 --host 127.0.0.1 --recall divergent-rba-rbr-20260701 --attach <prompt>`.
   - Exit code: `1`.
   - Stdout path: `research-wiki/experiments/pro_review_outputs/rba_rbr_rosetta_pro_attach_20260701_stdout.md`.
   - Stderr path: `research-wiki/experiments/pro_review_outputs/rba_rbr_rosetta_pro_attach_20260701_stderr.log`.
   - Failure: could not bring tab to front; OS focus poll timed out.
   - Verdict: `INVALID_TRANSPORT_NO_PRO_DECISION`.

6. Oracle browser persistent profile on 9333
   - Command family: `oracle --engine browser --model gpt-5.5-pro --browser-port 9333`.
   - Exit code: `1`.
   - Stdout path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_9333_persistent_20260701_stdout.log`.
   - Stderr path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_9333_persistent_20260701_stderr.log`.
   - Failure: unable to locate ChatGPT model selector; no cookies were applied; browser profile was not logged in.
   - Verdict: `INVALID_TRANSPORT_NO_PRO_DECISION`.

7. Oracle browser with explicit Rosetta cookie DB
   - Command family: `oracle --engine browser --model gpt-5.5-pro --browser-port 9333 --browser-cookie-path C:\Users\skywalker\.rosetta\chrome-profile-9333\Default\Network\Cookies`.
   - Exit code: `1`.
   - Stdout path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_cookie_9333_20260701_stdout.log`.
   - Stderr path: `research-wiki/experiments/pro_review_outputs/rba_rbr_oracle_browser_cookie_9333_20260701_stderr.log`.
   - Failure: unable to locate ChatGPT model selector; no cookies were applied.
   - Verdict: `INVALID_TRANSPORT_NO_PRO_DECISION`.

## Current Pro-Gate State

- No valid GPT-5.5 Pro / Rosetta Pro answer was harvested.
- No answer inspected GitHub.
- No code-grounded Pro verdict exists.
- This is a transport/infrastructure failure, not a technical rejection of RBA-RBR.
- Formal full training remains locked.
- Existing local and remote non-GPU precheck evidence remains valid for `REMOTE_SYNC_PRECHECK_ONLY`.
- Next allowed route action remains: wait for GPU0 availability and, subject to coordinator decision/project rules, run `SHORT_DIAGNOSTIC_ONLY`; or rerun Pro review after browser login/transport is repaired.
