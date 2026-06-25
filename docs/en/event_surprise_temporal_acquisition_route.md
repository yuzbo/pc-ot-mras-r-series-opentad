# Event-Surprise Temporal Acquisition Route

Route label: `DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`

This route is a divergent input-acquisition candidate for task-aware temporal
action detection. It is intentionally isolated from the C3 optimization route,
BH-SDC, Boundary Microscope, and Frame-token Hybrid routes.

## Purpose

The selector uses only deploy-visible preview evidence to estimate:

- temporal surprise from adjacent feature change;
- local motion/change energy;
- uncertainty from disagreement between short- and wider-window changes;
- redundancy in stable temporal spans.

Stable redundant spans receive lower priority. Surprise peaks and change-heavy
regions receive denser samples. Coverage anchors plus a max-gap repair keep the
selected temporal axis usable by the downstream detector.

## Integration Boundary

The implementation changes input sampling and the Event-Surprise inference
decode step. It does not change the Adapter, backbone, neck, detector head,
loss/assignment, token compression, or dynamic budget policy.

The selector supports two entry styles:

- standalone preview tensors: `[B,T,C]` plus `[B,T]` valid masks;
- ActionFormer keyword path: `inputs`/`masks`, returning selected
  `inputs`/`masks`/`metas`.

For ActionFormer-style selected-axis training, the route can remap training GT
segments to the selected axis. Test-time inputs reject GT, oracle, teacher,
cache, checkpoint, raw-prediction, and result metadata. Runtime/test metadata
now uses a strict allowlist: only deploy-visible dataset identifiers, temporal
scale fields, and Event-Surprise selector fields are accepted. Unexpected keys
fail closed even when their names do not match a forbidden token.

Training-time GT remap is not treated as an inference proof. The ActionFormer
test path now maps selected-axis head proposals back to the original dense
window axis through the selector's selected dense indices before returning
predictions. The mapping is piecewise-linear over selected slots and records
`event_surprise_inference_mapping.mapping_applied=True` in the route metadata.

Training GT remap filters segments and labels with the same keep mask after
checking that each batch item has aligned segment and label counts. Mismatched
counts fail before mapping, and non-tensor label sequences are filtered without
silently preserving labels for removed segments.

For 5D detector inputs, the preview feature is explicitly a prototype computed
from an already-loaded dense detector tensor. It is not a predecode hook and
does not prove raw decode, FLOPs, latency, runtime, deploy, metric, or paper
claims. The route metadata keeps `decode_saving_claim_allowed=False` and
`runtime_flops_claim_allowed=False`.

## Current Gate State

The local precheck config remains the default safe path:

- local precheck config: `event_surprise_temporal_acquisition_local_precheck.py`;
- gate-bound full-train candidate config:
  `event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`.

The full-train candidate is no longer blocked by a hard follow-up-Pro flag. It is
a train-only, external-gate-bound candidate for
`DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3`, not a C3
optimization or combined attribution route. Its config permits only
`tools/train.py` after an entrypoint gate is present. Direct evaluation,
`tools/test.py`, raw-prediction cache use, GT/teacher/oracle shortcuts,
train-validation metric claims, runtime/deploy claims, metric claims, and paper
claims remain forbidden.

Full train must fail closed unless all of the following are supplied and match:

- external full-train gate JSON path and SHA256;
- active manifest SHA256;
- resolved config SHA256;
- `EVENT_SURPRISE_RUN_TAG` / launcher `RUN_TAG`;
- route label and route id;
- fixed-budget contract:
  `budget_protocol=fixed384_over_dense768_event_surprise_train_only`,
  `dense_window_size=768`, `selected_length=384`, `target_len=384`,
  `selection_ratio=0.5`;
- user plus coordinator override statement:
  `USER_COORDINATOR_APPROVED_EVENT_SURPRISE_GATE_BOUND_FULL_TRAIN`.

Launcher enforcement is handled by `tools/bata/validate_event_surprise_gate.py`.
The config validator alone is not a training permission. Any entrypoint must
also pass an explicit action check:

- `--action precheck-only` is allowed only for the local precheck config;
- `--action full-train` fails closed without the external gate JSON/SHA and
  matching active-manifest, resolved-config, and RUN_TAG bindings; with a valid
  gate it returns `train_command_allowed=true`.

The N16R4 deployment-precheck launcher is
`scripts/run_event_surprise_temporal_acquisition_precheck_n16r4.sbatch`. It
defaults to `PRECHECK_ONLY=1` and runs validator/import/compile/focused
launcher-gate checks only. Before focused pytest it clears formal full-train
gate env vars so a stale gate cannot pollute precheck tests.

The N16R4 full-train launcher is
`scripts/run_event_surprise_temporal_acquisition_full_train_n16r4.sbatch`. It
also defaults to `PRECHECK_ONLY=1`, which computes the resolved config and active
manifest, validates the config, compiles the relevant Python files, checks bash
syntax, and proves full train remains fail-closed without an external gate. With
`PRECHECK_ONLY=0`, `ALLOW_EVENT_SURPRISE_FULL_TRAIN=1`, the explicit coordinator
override, and a valid gate JSON/SHA, it exports the entrypoint gate env to
`tools/train.py`. It never invokes `tools/test.py`.

## Validation

Use:

```powershell
python -m pytest tests/test_event_surprise_acquisition_route.py tests/test_event_surprise_config_gate.py
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py --action precheck-only
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py --action full-train
```

The last command is expected to fail without `--gate-json`, `--gate-sha256`,
`--active-manifest-sha256`, `--resolved-config-sha256`, and `--run-tag`.
