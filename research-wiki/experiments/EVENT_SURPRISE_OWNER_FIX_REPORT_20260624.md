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

---

# EVENT_SURPRISE Pro-Fix Self-Check 2026-06-25

Timestamp: 2026-06-25 Asia/Shanghai

Route: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Status label: `FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`

Writable worktree:
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_EventSurprise_ProFix_Worktree_20260625`

Branch: `codex/event-surprise-pro-fix-20260625`

No Claude, Gemini, DeepSeek, remote sync, Slurm, long training,
`tools/test.py`, detector mAP, raw-prediction cache, or deploy/runtime/paper
claim action was invoked.

## Pro Findings Addressed

1. Selected-axis inference mapping:
   - Added selected-axis proposal to dense-window-axis inverse mapping in
     `EventSurpriseTemporalAcquisitionSelector`.
   - `ActionFormer.forward_test` now applies this mapping to Event-Surprise
     selected-axis head proposals before returning predictions.
   - Training GT remap remains train-only and is no longer treated as inference
     proof.

2. Full-train candidate fail-closed provenance:
   - The full-train candidate config is still a candidate container, but
     `allow_slurm`, `allow_gpu`, `allow_tools_train`,
     `allow_detector_training`, `allow_train_validation_map`,
     `allow_long_training`, `allow_full_train`, and `launch_gate_passed` are
     all false.
   - Added hard provenance fields:
     `review_status=FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`,
     `reviewed_impl_commit=FOLLOWUP_PRO_REQUIRED`,
     `selector_type=EventSurpriseTemporalAcquisitionSelector`,
     `no_c3_mixing=True`, `no_gt_teacher_cache_leakage=True`, and
     `requires_followup_pro_review=True`.
   - `tools/bata/validate_event_surprise_gate.py --action full-train` now fails
     even with gate JSON until follow-up Pro review clears the lock.

3. 5D detector-input preview claim:
   - 5D preview is explicitly recorded as
     `loaded_dense_detector_input_prototype`.
   - `decode_saving_claim_allowed=False` and
     `runtime_flops_claim_allowed=False` are written into route metadata.
   - Runtime/FLOPs/deploy/paper claims remain forbidden.

4. Additional tests:
   - Route-isolation/no-C3 config checks.
   - selected-axis/dense-axis GT and proposal roundtrip.
   - no-GT/no-teacher/no-oracle/no-cache/no-raw-prediction metadata guard.
   - synthetic ActionFormer build/forward smoke with Event-Surprise selector.
   - full-train guard fail-closed with and without gate JSON.

## Changed Files

- `opentad/models/selectors/event_surprise_acquisition_route.py`
- `opentad/models/detectors/actionformer.py`
- `configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py`
- `configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`
- `tools/bata/validate_event_surprise_gate.py`
- `tests/test_event_surprise_acquisition_route.py`
- `tests/test_event_surprise_config_gate.py`
- `docs/en/event_surprise_temporal_acquisition_route.md`
- `research-wiki/experiments/EVENT_SURPRISE_OWNER_FIX_REPORT_20260624.md`

## Local Verification Evidence

Default `python` environment note:
- `python -c "import torch; print(torch.__version__)"` fails with Windows
  torch DLL error `WinError 1114` for
  `C:\Users\skywalker\AppData\Roaming\Python\Python311\site-packages\torch\lib\c10.dll`.
- Therefore torch-dependent tests were run with
  `C:\Users\skywalker\.conda\envs\torch_1\python.exe`, which has
  `pytest 8.3.2`, `mmengine 0.10.5`, and `torch 2.3.0+cu121`.

Commands and results:
- Red test gate before implementation:
  `python -m pytest tests/test_event_surprise_acquisition_route.py tests/test_event_surprise_config_gate.py -q`
  failed on missing provenance/full-train-lock fields, then later on the
  selected implementation smoke until fixed.
- Compile verification:
  `C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile opentad\models\selectors\event_surprise_acquisition_route.py opentad\models\detectors\actionformer.py configs\adatad\thumos\event_surprise_temporal_acquisition_local_precheck.py configs\adatad\thumos\event_surprise_temporal_acquisition_full_train_candidate_n16r4.py tools\bata\validate_event_surprise_gate.py tests\test_event_surprise_acquisition_route.py tests\test_event_surprise_config_gate.py`
  -> exit 0.
- Validator verification:
  local config validation PASS; full candidate config validation PASS; local
  `--action precheck-only` PASS with `train_command_allowed=false`; full
  candidate `--action full-train` expected-fails with
  `full_train remains locked pending follow-up Pro review`.
- Focused torch verification:
  `C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_event_surprise_acquisition_route.py tests/test_event_surprise_config_gate.py -q`
  -> `29 passed in 20.79s`.

## Still Locked

- Follow-up GPT-5.5 Pro review is still required before any full-train,
  remote-sync, Slurm, detector-training, train-validation-mAP, long-training,
  detector-evaluation, or paper/deploy/runtime claim action.
- Event-Surprise must remain separate from C3:
  `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`.
