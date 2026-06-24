# Event-Surprise Owner Fix Self-Check 2026-06-24

Timestamp: 2026-06-24 20:05:18 +08:00

Route label: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

Changed surface:
- input sampling selector runtime/test metadata guard;
- train-time selected-axis GT remap;
- config/launcher gate logic;
- Event-Surprise tests and route documentation.

Protocol boundary:
- local code/config/precheck gate only;
- no remote sync;
- no Slurm;
- no detector mAP;
- no full training;
- no runtime/deploy/paper claim.

## Blocker Fix Mapping

1. Strict deploy/test metadata allowlist:
   - Replaced blacklist-only runtime/test metadata validation with an explicit allowlist for deploy-visible dataset metadata and Event-Surprise selector metadata.
   - Unknown keys fail closed even when their names do not contain GT/teacher/oracle/cache tokens.
   - Protocol flags that mention forbidden concepts are allowed only as known keys and must remain false.

2. Segment/label alignment in train GT remap:
   - Added per-sample segment/label count checks before remap.
   - Uses the same `keep` mask to filter remapped segments and labels.
   - Supports tensor labels and non-string sequence labels without silently preserving removed labels.

3. End-to-end launch enforcement:
   - Added `--action precheck-only` and `--action full-train` to `tools/bata/validate_event_surprise_gate.py`.
   - Precheck action is allowed only for the local precheck config.
   - Full train fails closed unless a separate Event-Surprise launch gate JSON explicitly matches route label, action, allow flag, passed gate, and `full_train` entrypoint.
   - Full-train config remains locked by default.

## Verification Commands And Outputs

Command:
```powershell
python -m pytest tests/test_event_surprise_acquisition_route.py tests/test_event_surprise_config_gate.py
```

Output summary:
```text
collected 9 items / 1 skipped
tests\test_event_surprise_config_gate.py ......... [100%]
9 passed, 1 skipped in 7.19s
```

Note: `tests/test_event_surprise_acquisition_route.py` was skipped because this Windows Python cannot import torch.

Command:
```powershell
python -c "import torch"
```

Output:
```text
OSError: [WinError 1114] DLL initialization routine failed.
Error loading "C:\Users\skywalker\AppData\Roaming\Python\Python311\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

Command:
```powershell
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py
```

Output summary:
```json
{
  "allow_precheck_only": true,
  "pass": true,
  "route_label": "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3",
  "stage": "local_precheck_only"
}
```

Command:
```powershell
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py
```

Output summary:
```json
{
  "allow_precheck_only": false,
  "formal_train_candidate": true,
  "full_train_candidate": true,
  "pass": true,
  "route_label": "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3",
  "stage": "full_train_candidate_locked"
}
```

Command:
```powershell
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py --action precheck-only
```

Output summary:
```json
{
  "launch_action": "precheck_only",
  "launch_allowed": true,
  "train_command_allowed": false
}
```

Command:
```powershell
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py --action full-train
```

Output:
```text
EventSurpriseGateError: full_train launch requires --gate-json
```

Command:
```powershell
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py --action full-train --gate-json C:\Users\SKYWAL~1\AppData\Local\Temp\event_surprise_full_train_gate_selfcheck_20260624.json
```

Output summary:
```json
{
  "launch_action": "full_train",
  "launch_allowed": true,
  "train_command_allowed": true
}
```

The temporary gate JSON was deleted after this self-check and is not stored in the repository.

Command:
```powershell
python -m py_compile opentad/models/selectors/event_surprise_acquisition_route.py tools/bata/validate_event_surprise_gate.py configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py tests/test_event_surprise_acquisition_route.py tests/test_event_surprise_config_gate.py
```

Output:
```text
exit code 0, no stdout/stderr
```
