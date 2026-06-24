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

Both provided configs are fail-closed:

- local precheck config: `event_surprise_temporal_acquisition_local_precheck.py`;
- locked full-train candidate config:
  `event_surprise_temporal_acquisition_full_train_candidate_n16r4.py`.

Remote sync, Slurm full-train submission, detector training, direct evaluation,
raw-prediction cache use, runtime/deploy claims, metric claims, and paper claims
remain locked. The full-train candidate config is preserved only as a reviewed
candidate container and is marked
`FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`; an external gate JSON cannot open
full training until a follow-up Pro review replaces the pending provenance lock.

Launcher enforcement is handled by `tools/bata/validate_event_surprise_gate.py`.
The config validator alone is not a training permission. Any entrypoint must
also pass an explicit action check:

- `--action precheck-only` is allowed only for the local precheck config;
- `--action full-train` fails closed with
  `full_train remains locked pending follow-up Pro review`, even if a gate JSON
  is supplied.

The N16R4 deployment-precheck launcher is
`scripts/run_event_surprise_temporal_acquisition_precheck_n16r4.sbatch`. It
defaults to `PRECHECK_ONLY=1` and runs validator/import/compile/focused
launcher-gate checks only. Under the current follow-up-Pro lock,
`PRECHECK_ONLY=0` remains blocked by the config/validator and must not be used
to run `tools/train.py` or `tools/test.py`.

The N16R4 full-train launcher is
`scripts/run_event_surprise_temporal_acquisition_full_train_n16r4.sbatch`. It
also defaults to `PRECHECK_ONLY=1`, which computes the resolved config and active
manifest, validates the config, compiles the relevant Python files, checks bash
syntax, and proves full train remains locked. The config-level hard gate now
blocks full train before any gate JSON can authorize it; follow-up Pro review is
required before this launcher can be reconsidered.

## Validation

Use:

```powershell
python -m pytest tests/test_event_surprise_acquisition_route.py tests/test_event_surprise_config_gate.py
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py --action precheck-only
python tools/bata/validate_event_surprise_gate.py --config configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py --action full-train
```
