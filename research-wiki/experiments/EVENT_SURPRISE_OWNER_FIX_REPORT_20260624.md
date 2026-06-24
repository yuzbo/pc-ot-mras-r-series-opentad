# EVENT_SURPRISE Owner Fix Report 2026-06-24

Timestamp: 2026-06-24 20:05:18 +08:00

Route: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

This report records the single-owner fix for the substantive GPT-5.5 Pro blockers on the Event-Surprise divergent acquisition route. No GPT-5.5 Pro, Claude, Gemini, remote sync, Slurm, training, or evaluator run was invoked in this fix stage.

## Changed Files

- `opentad/models/selectors/event_surprise_acquisition_route.py`
- `configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py`
- `configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`
- `tools/bata/validate_event_surprise_gate.py`
- `tests/test_event_surprise_acquisition_route.py`
- `tests/test_event_surprise_config_gate.py`
- `docs/en/event_surprise_temporal_acquisition_route.md`
- `logs/event_surprise_owner_fix_self_check_20260624.md`

## Fix Summary

Strict test-time meta guard:
- Runtime/test metadata now uses a strict allowlist.
- Unknown metadata keys are rejected by default.
- Known protocol keys that mention GT/oracle/teacher/raw-prediction concepts are only allowed as explicit false flags.

Train-time GT remap:
- Segment and label counts are checked per batch item before remapping.
- Segment filtering and label filtering use one shared keep mask.
- Non-string label sequences are filtered explicitly instead of being silently left unfiltered.

Launcher enforcement:
- The validator now has launch-action checks.
- `precheck-only` is allowed only for the local precheck config and never permits a training command.
- `full-train` fails closed without an explicit external launch gate JSON.
- A valid external gate JSON must match route label, action, allow flag, passed gate, and `full_train` entrypoint.
- The repository does not store a full-train allow gate JSON.

## Gate State After Fix

Allowed:
- local config validation;
- local precheck action through `tools/bata/validate_event_surprise_gate.py --action precheck-only`.

Still locked:
- remote sync;
- Slurm;
- GPU detector training;
- detector mAP;
- long/full training without explicit external Event-Surprise launch gate JSON;
- raw-prediction cache load/save;
- runtime/deploy/metric/paper claims.

## Verification Evidence

See `logs/event_surprise_owner_fix_self_check_20260624.md` for command output summaries.

Key result:
- Event config/validator pytest: `9 passed, 1 skipped`.
- Torch-dependent acquisition tests skipped because local Windows torch import fails with `WinError 1114` loading `c10.dll`.
- Validator locked configs pass.
- Validator precheck action passes for local precheck.
- Validator full-train action fails without `--gate-json`.
- Validator full-train action passes with a temporary explicit self-check gate JSON, which was deleted and not committed.
- `py_compile` exits 0 for touched Python/config/test files.
