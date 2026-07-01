# DIVERGENT BVR-TWB Budget Fix 20260701

Timestamp: 2026-07-01 17:42:07 +08:00

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_BudgetFix_Worktree_20260701`

Branch: `codex/divergent-bvr-twb-budgetfix-20260701`

## Diagnosis

The prior BVR run trained stably and loaded pretrained weights, but Avg-mAP stayed around 21.78 to 23.92 by epoch 54. Local traces showed raw `valid_k` around 14 to 18 and detector-facing `detector_feature_valid_k` around 7 to 9, with fixed-length Adapter duplicate padding dominating the input. This is root-cause-aligned with a budget semantics failure rather than a pretrained loading failure: the selector could stop on candidate exhaustion or safety logic after satisfying only a loose raw/gap condition, while the detector received too few effective feature tokens.

The current value path remains `deploy_heuristic_voi`. `trainable_value.py` still exists but is not integrated into this route, so no learned-regret claim is made.

## Implemented Repair

- Added `feature_stride` and `min_detector_k` to `BudgetConfig`.
- Raised the effective detector-token floor into the raw selector budget. For the formal THUMOS BVR config, `bvr_twb_min_detector_keep=64` with `bvr_twb_feature_stride=2` makes the raw floor at least 128 under `max_keep=192`.
- Added controller coverage fill when candidate packets are exhausted below the configured floor. This produces explicit `selected_detector_floor_fill` rows and `under_budget_fill_count` evidence rather than silently accepting low `valid_k`.
- Added deploy and pipeline validators for:
  - raw `valid_k >= dynamic_min_k`;
  - `detector_feature_valid_k >= min_detector_feature_k`;
  - duplicate padding ratio under `max_adapter_padding_duplicate_ratio`;
  - candidate exhaustion not stopping below floor.
- Kept the fixed-length Adapter padded bridge semantics explicit: padding remains invalid, no sparse-compute claim is unlocked, and no runtime/flops claim is made.
- Updated local BVR audit and launch gate to include raw/effective floor and padding-ratio evidence.

## Changed Files

- `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`
- `opentad/acquisition/bvr_twb/budget_controller.py`
- `opentad/acquisition/bvr_twb/open_tad_bridge.py`
- `opentad/acquisition/bvr_twb/types.py`
- `opentad/acquisition/bvr_twb/validators.py`
- `opentad/datasets/transforms/end_to_end.py`
- `tests/test_bvr_twb_opentad_pipeline.py`
- `tests/test_bvr_twb_validators.py`
- `tools/bvr_twb/audit_opentad_bvr_twb_pipeline.py`
- `tools/bvr_twb/dump_bridge_roundtrip.py`
- `tools/bvr_twb/validate_bvr_twb_launch_gate.py`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_BUDGETFIX_20260701.md`

## Verification

Red tests before implementation:

- `python -m pytest tests\test_bvr_twb_validators.py::test_controller_repairs_candidate_exhaustion_to_effective_detector_floor -q`
  - Failed because `BudgetConfig` had no `feature_stride` / `min_detector_k`.
- `python -m pytest tests\test_bvr_twb_opentad_pipeline.py::test_pipeline_ledger_rejects_under_budget_and_duplicate_padding_dominance -q`
  - Failed because the validator did not reject low raw/effective token counts or duplicate padding dominance.

Fresh passing checks after implementation:

- `python -m pytest tests\test_bvr_twb_validators.py::test_controller_repairs_candidate_exhaustion_to_effective_detector_floor -q`
  - `1 passed`
- `python -m pytest tests\test_bvr_twb_opentad_pipeline.py::test_pipeline_ledger_rejects_under_budget_and_duplicate_padding_dominance -q`
  - `1 passed`
- `python -m pytest tests\test_bvr_twb_validators.py tests\test_bvr_twb_opentad_pipeline.py -q`
  - `24 passed, 7 skipped`
- PowerShell-expanded `python -m pytest tests\test_bvr_twb_*.py -q`
  - `66 passed, 17 skipped, 1 warning`
- `python -m py_compile ...`
  - Passed for changed BVR source, transform, and tools.
- `git diff --check -- ...`
  - Passed. Git emitted Windows LF to CRLF warnings only.
- `python tools\bvr_twb\audit_opentad_bvr_twb_pipeline.py --out-dir .tmp_bvr_twb_budgetfix_pipeline_audit --overwrite`
  - `ledgers=3 all_validated=True sparse_compute_claim=False blocked=False`
- `python tools\bvr_twb\validate_bvr_twb_launch_gate.py --config configs\adatad\thumos\input_bvr_twb_dynamic_adapter_irregular_headv3.py --precheck-summary .tmp_bvr_twb_budgetfix_pipeline_audit\summary.json`
  - `gate_pass=true`
  - `full_train_unlocked=false`
  - `remote_sync_unlocked_by_local_gate=false`

Local audit summary evidence:

- `min_raw_valid_k=32`, `configured_min_raw_keep=32`
- `min_detector_feature_valid_k=16`, `configured_min_detector_feature_keep=16`
- `max_adapter_padding_duplicate_ratio=0.3333333333333333`
- `max_allowed_adapter_padding_duplicate_ratio=0.5`
- `sparse_compute_claim=false`
- `no_training=true`, `no_metric_claim=true`, `no_runtime_or_flops_claim=true`

The local audit used the numpy fallback because Windows torch import failed with `c10.dll` initialization error. This is a local runtime limitation, not a BVR code pass for torch runtime. Linux torch precheck remains required before any remote sync or training.

## Locked State

No remote sync, Slurm, training, `tools/test.py`, Pro/Oracle/Rosetta submission, C3/CADF/PQR edit, or shared/root worktree write was performed.

Allowed next action is final read-only review, then Linux precheck only. Full training and remote sync remain locked.
