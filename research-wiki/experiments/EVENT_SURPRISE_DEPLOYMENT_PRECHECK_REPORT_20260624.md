# EVENT_SURPRISE Deployment Precheck Report 2026-06-24

Timestamp: 2026-06-24 20:30 +08:00

Route: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Boundary: Event-Surprise is a divergent innovation route. This deployment
precheck does not merge it with C3, BH-SDC, Boundary, or Frame routes. No
evaluator, postprocess, full training, Pro, Claude, or Gemini gate was invoked.

## Local Owner Worktree

- Worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_EventSurprise_OwnerFix_Worktree_20260624`
- Branch: `codex/event-surprise-owner-fix-20260624`
- Base owner-fix head: `126e04ee43aa44204bf8ea09a0146780dd9653f0`
- Deployment commits:
  - `0768971` Add Event-Surprise N16R4 precheck launcher
  - `8718669` Scope Event-Surprise precheck to launcher gate
- Push target: `origin/codex/event-surprise-owner-fix-20260624`

Changed files in the deployment-precheck stage:
- `scripts/run_event_surprise_temporal_acquisition_precheck_n16r4.sbatch`
- `tests/test_event_surprise_config_gate.py`
- `docs/en/event_surprise_temporal_acquisition_route.md`
- `research-wiki/experiments/EVENT_SURPRISE_DEPLOYMENT_PRECHECK_REPORT_20260624.md`

## Local Verification

Commands rerun in the owned worktree:
- `python tools\bata\validate_event_surprise_gate.py --config configs\adatad\thumos\event_surprise_temporal_acquisition_local_precheck.py --action precheck-only`
- `python tools\bata\validate_event_surprise_gate.py --config configs\adatad\thumos\event_surprise_temporal_acquisition_local_precheck.py`
- `python tools\bata\validate_event_surprise_gate.py --config configs\adatad\thumos\event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`
- `python tools\bata\validate_event_surprise_gate.py --config configs\adatad\thumos\event_surprise_temporal_acquisition_full_train_candidate_n16r4.py --action full-train`
- `python -m py_compile tools\bata\validate_event_surprise_gate.py configs\adatad\thumos\event_surprise_temporal_acquisition_local_precheck.py configs\adatad\thumos\event_surprise_temporal_acquisition_full_train_candidate_n16r4.py opentad\models\selectors\event_surprise_acquisition_route.py tests\test_event_surprise_config_gate.py tests\test_event_surprise_acquisition_route.py`
- `python -m pytest tests\test_event_surprise_config_gate.py -q`

Results:
- Local precheck action validator: PASS, `launch_allowed=true`,
  `train_command_allowed=false`.
- Local locked precheck/full-candidate config validation: PASS.
- Local full-train action without gate JSON: expected fail,
  `full_train launch requires --gate-json`.
- Local py_compile: PASS.
- Local launcher/gate pytest: `10 passed`.

Earlier local owner-fix evidence also had
`tests\test_event_surprise_config_gate.py tests\test_event_surprise_acquisition_route.py`
as `10 passed, 1 skipped` because the local Windows torch import skipped the
Torch-dependent acquisition route tests.

## Remote Sync

N16R4 login:
- Host observed: `ln01`
- Workspace root: `/data/home/sczc063/run/yuzibo`
- Route-owned deploy path:
  `/data/home/sczc063/run/yuzibo/OpenTAD_EventSurprise_PrecheckDeploy_20260624_126e04e`

Sync method:
- GitHub branch clone/fetch only; no zip upload.
- Repository URL:
  `https://github.com/yuzbo/pc-ot-mras-r-series-opentad.git`
- Branch: `codex/event-surprise-owner-fix-20260624`
- Final remote HEAD:
  `87186697fedd3b9c609ca6c82b5ff82393b245fb`
- Remote status after fast-forward: clean tracked tree.

## Remote PRECHECK_ONLY

Slurm attempt:
- Submitted `sbatch` job: `1117983`
- Command scope: `PRECHECK_ONLY=1`
- Status at check time: `PENDING`, reason `Priority`
- Slurm estimated start: `2026-06-26T15:59:27`
- Decision: not canceled, per instruction. This pending GPU job is not counted
  as the completed precheck result.

Completed lightweight login-node precheck:
- Command scope: validator/import/compile/launcher-gate pytest only.
- No `tools/train.py`, no `tools/test.py`, no detector mAP, no dataset access,
  no checkpoint, no raw-prediction cache, no claims.
- Log:
  `/data/home/sczc063/run/yuzibo/OpenTAD_EventSurprise_PrecheckDeploy_20260624_126e04e/logs/event_surprise_login_precheck_20260624_8718669/precheck.log`
- Result: PASS.

Key remote log evidence:
- Remote branch:
  `codex/event-surprise-owner-fix-20260624`
- Remote head:
  `87186697fedd3b9c609ca6c82b5ff82393b245fb`
- Python:
  `/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python`
  (`Python 3.10.20`)
- Local precheck config with `--action precheck-only`: PASS,
  `launch_allowed=true`, `train_command_allowed=false`.
- Locked full-train candidate config validation: PASS as locked candidate.
- Full-train action without gate JSON: expected fail,
  `full_train launch requires --gate-json`.
- Remote launcher/gate pytest: `10 passed in 13.52s`.
- Final remote marker:
  `PASS PRECHECK_ONLY no_train no_test no_map`.

One remote exploratory precheck before scoping the launcher to gate tests ran the
Torch-dependent acquisition route pytest and failed on an assertion message only:
the selector rejected `{"safe": {"oracle_hint": true}}`, but the message was
`unexpected deploy meta key` instead of the test's expected `forbidden deploy
meta`. This is outside the deployment-precheck writable scope and was not
changed in this owner stage.

## Gate State And Next Action

Allowed now:
- Remote GitHub branch sync to the route-owned path.
- Remote PRECHECK_ONLY validator/import/compile/launcher-gate checks.

Still locked:
- Full training.
- Detector evaluation or mAP.
- Raw prediction cache load/save.
- Runtime/deploy/metric/paper claims.
- C3, BH-SDC, Boundary, Frame, evaluator, and postprocess changes.

Full train is not currently queued and should remain locked until a coordinator
or user explicitly provides an external Event-Surprise full-train gate JSON,
matching SHA256, and override decision. The current precheck launcher validates
that gate path but still does not run `tools/train.py`; a later train-capable
launcher or explicit coordinator launch should be reviewed before any full run.
