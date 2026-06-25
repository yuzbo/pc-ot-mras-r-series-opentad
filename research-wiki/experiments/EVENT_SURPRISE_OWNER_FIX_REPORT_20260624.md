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

Route: Event-Surprise urgent full-train inheritance fix.

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

---

# EVENT_SURPRISE Full-Train Gate-Bound Fix 2026-06-25

Timestamp: 2026-06-25 12:45:13 +08:00

Route: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Writable worktree:
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_EventSurprise_ProFix_Worktree_20260625`

Branch: `codex/event-surprise-pro-fix-20260625`

This stage fixes the full-train blocker created by the previous follow-up-Pro
hard lock. No GPT-5 Pro, Gemini, Claude, remote sync, Slurm, `tools/test.py`,
metric evaluation, raw-prediction cache, or paper/deploy/runtime claim was
invoked.

## Gate-Bound Status

The full-train candidate config is now gate-bound instead of Pro-hard-locked:

- `stage=full_train_candidate_gate_bound`
- `review_status=GATE_BOUND_FULL_TRAIN_CANDIDATE_NO_FOLLOWUP_PRO_REQUIRED`
- `reviewed_impl_commit=EVENT_SURPRISE_FULL_TRAIN_GATE_BOUND_FIX`
- `requires_followup_pro_review=False`
- train-only config entrypoint: `allowed_entrypoints=("tools/train.py",)`

It remains fail-closed without all external launch bindings:

- external full-train gate JSON plus SHA256;
- active manifest SHA256;
- resolved config SHA256;
- `EVENT_SURPRISE_RUN_TAG`;
- route label and route id;
- fixed-budget contract:
  `budget_protocol=fixed384_over_dense768_event_surprise_train_only`,
  `dense_window_size=768`, `selected_length=384`, `target_len=384`,
  `selection_ratio=0.5`;
- user plus coordinator override statement:
  `USER_COORDINATOR_APPROVED_EVENT_SURPRISE_GATE_BOUND_FULL_TRAIN`.

The valid-gate path allows `tools/train.py` only. It continues to forbid
`tools/test.py`, raw-prediction cache load/save, GT/teacher/oracle shortcuts,
detector-map/eval entrypoints, metric claims, paper claims, runtime/FLOPs
claims, and deploy claims.

This is not a C3 optimization and must not be merged into C3 attribution. Any
future combined route still requires a separate explicit combo gate.

## Changed Files In This Stage

- `configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`
- `opentad/utils/training_guard.py`
- `tools/bata/validate_event_surprise_gate.py`
- `scripts/run_event_surprise_temporal_acquisition_full_train_n16r4.sbatch`
- `scripts/run_event_surprise_temporal_acquisition_precheck_n16r4.sbatch`
- `tests/test_event_surprise_config_gate.py`
- `docs/en/event_surprise_temporal_acquisition_route.md`
- `research-wiki/experiments/EVENT_SURPRISE_OWNER_FIX_REPORT_20260624.md`

## Local Verification Snapshot

Red test before implementation:

- `python -m pytest tests/test_event_surprise_config_gate.py -q`
  -> failed on the old follow-up-Pro hard lock and old false train contract.
- RUN_TAG regression test:
  `python -m pytest tests/test_event_surprise_config_gate.py::test_event_surprise_full_train_config_is_guarded_by_external_entrypoint_gate -q`
  -> failed because `training_guard.py` did not require `EVENT_SURPRISE_RUN_TAG`.
- External binding regression test:
  `python -m pytest tests/test_event_surprise_config_gate.py::test_event_surprise_launch_action_full_train_rejects_missing_external_bindings -q`
  -> failed because validator accepted a gate JSON without requiring
  `--gate-sha256`, `--active-manifest-sha256`,
  `--resolved-config-sha256`, and `--run-tag`.

Green evidence collected during implementation:

- `python -m pytest tests/test_event_surprise_config_gate.py -q`
  -> `17 passed`.

Final verification commands are recorded in the commit/task response for this
stage. No metric or paper claim is derived from these local gates.

## 2026-06-25 Full-Train Config Inheritance Blocker Fix

Timestamp: 2026-06-25 18:22 +08:00

Route: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Owned worktree:
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_EventSurprise_ProFix_Worktree_20260625`

Branch: `codex/event-surprise-pro-fix-20260625`

Observed blocker:
`tools/train.py configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`
crashed on N16R4 with `AttributeError: 'ConfigDict' object has no attribute
'dataset'` because the full-train candidate config only carried the
Event-Surprise route/gate/model delta and did not inherit the THUMOS14
AdaTAD/ActionFormer Adapter base.

Fix:

- `event_surprise_temporal_acquisition_local_precheck.py` now inherits
  `./e2e_thumos_videomae_s_768x1_160_adapter.py`.
- `event_surprise_temporal_acquisition_full_train_candidate_n16r4.py` now
  inherits `./event_surprise_temporal_acquisition_local_precheck.py`.
- The validator now requires the full-train candidate to resolve
  `dataset.train/val/test`, `solver.train/val/test`, `model.type=ActionFormer`,
  `VisionTransformerAdapter`, `ActionFormerHead`, and the Event-Surprise
  `model.frame_selector`.
- Local tests now assert the inheritance chain, base dataset/solver presence,
  base Adapter/ActionFormer detector stack, route label, and selected length.
- Event-Surprise launch manifests now include the inherited local/base
  configs so the resolved full-train config hash is bound to the base stack.

Protocol boundary:
No non-Event divergent-route implementation token was added outside the
existing forbidden-token check lists. No Pro, external advisory model, browser
automation, remote sync, Slurm, training, evaluation, raw-prediction cache,
mAP/runtime/deploy claim, or paper claim was invoked.

Local verification:

- `python -m pytest tests/test_event_surprise_config_gate.py -q`
  -> `18 passed in 24.20s`.
- `python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`
  -> PASS.
- `python -c "... Config.fromfile(...full_train_candidate...) ..."`
  -> `Config.fromfile full Event-Surprise inheritance assertion PASS`.
- `python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py`
  -> PASS.
- `python -m pytest tests/test_event_surprise_acquisition_route.py -q -rs`
  -> skipped in default env because `torch` import was unavailable.
- `conda run -n torch_1 python -m pytest tests/test_event_surprise_config_gate.py -q`
  -> `18 passed in 29.28s`.
- `conda run -n torch_1 python -m pytest tests/test_event_surprise_acquisition_route.py -q`
  -> `16 passed in 8.21s`.
- `conda run -n torch_1 python -c "... Config.fromfile(...full_train_candidate...) ..."`
  -> `torch_1 Config.fromfile full Event-Surprise inheritance assertion PASS`.
- `git diff --check`
  -> PASS; only line-ending warnings from Windows Git were reported.

Remaining risk:
This is a local config/gate fix only. It does not prove remote dataset paths,
GPU runtime, training stability, detector accuracy, runtime/FLOPs, deploy
safety, or paper claims. Full training remains controlled by the existing
external Event-Surprise full-train gate and coordinator/user launch procedure.
