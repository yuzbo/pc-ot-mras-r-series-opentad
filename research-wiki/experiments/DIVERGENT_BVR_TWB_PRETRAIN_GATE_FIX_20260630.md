# DIVERGENT_BVR_TWB_PRETRAIN_GATE_FIX_20260630

Timestamp: 2026-06-30 06:46:31 +08:00

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_PretrainFix_Worktree_20260630`

Owned branch: `codex/divergent-bvr-twb-pretrainfix-20260630`

Base commit: `92ec024d1821897245a56f993ed040d97e894831`

## Scope

This stage only fixes the formal BVR-TWB pretrained/checkpoint launch blocker and hardens the local launch gate/tests/docs. No training, remote sync, Slurm, `tools/train.py`, `tools/test.py`, Pro/Rosetta/Claude/Gemini, C3, ABR, MDL, combo-route, detector-code, or acquisition-code change was performed.

## Inheritance Finding

The formal config `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py` inherits through:

`input_bvr_twb_dynamic_adapter_irregular_headv3.py`
-> `input_random_fixed_50pct_adapter_irregular_headv3_x.py`
-> `input_random_fixed_50pct_adapter_irregular_actionformer_base.py`
-> `input_random_fixed_50pct_adapter.py`
-> `e2e_thumos_videomae_s_768x1_160_adapter.py`

The ancestor VideoMAE-S adapter config declares:

`pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"`

The formal BVR config rebuilt `model.backbone.custom` with `_delete_=True` and previously did not redeclare `pretrain`, so the upstream custom pretrain field could be deleted from the resolved config. This matches the old blocker symptom: `Warning: no pretrain path is provided`.

## Changes

- `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`
  - Explicitly redeclared the required VideoMAE-S pretrain path inside the `_delete_=True` `model.backbone.custom` block.
- `tools/bvr_twb/validate_bvr_twb_launch_gate.py`
  - Added `REQUIRED_PRETRAIN_PATH`.
  - Added a fail-closed check requiring an explicit `pretrain=...` line in the formal config text.
  - Added `mmengine.config.Config.fromfile(...)` resolved-config verification that `cfg.model.backbone.custom.pretrain` is exactly the required path.
- `tests/test_bvr_twb_opentad_pipeline.py`
  - Asserted the formal config text contains the required pretrain path.
  - Added a resolved-config mmengine test for the formal config.
  - Added a regression test that removes the pretrain declaration from the `_delete_=True` custom block and expects the launch gate to reject it.

Changed surface: formal config, local launch gate, local tests, route report only.

Strict random-fixed 50% contract status: preserved; no sampling or detector behavior was changed.

GT/teacher leakage risk: unchanged; this patch only checks pretrained config retention and does not add deploy/test GT, teacher, raw prediction, cache, evaluator, or post-processing access.

Attribution boundary: BVR-TWB pretrain/config/gate fix only; no C3/C3-Pro/combo route mixing.

## Commands And Results

Command:

```powershell
python -m pytest tests/test_bvr_twb_validators.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_geometry_contracts.py -q
```

Result:

```text
26 passed, 12 skipped in 4.32s
```

Command:

```powershell
python tools/bvr_twb/validate_bvr_twb_launch_gate.py --config configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py --precheck-summary <temporary no-training/no-metric summary.json>
```

First attempt result:

```text
gate did not run because the temporary PowerShell-written JSON had a UTF-8 BOM; json.loads rejected it.
```

Second attempt result with BOM-free temporary JSON:

```json
{"adapter_bridge_mode": "adapter_fixed_length_padded_bridge", "allowed_next_action": "FINAL_READ_ONLY_REVIEW_THEN_LINUX_PRECHECK_ONLY", "full_train_unlocked": false, "gate_pass": true, "remote_sync_unlocked_by_local_gate": false, "route_label": "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3", "sparse_compute_claim": false}
```

## Still Locked

- No long training is unlocked.
- No remote sync, Slurm, `srun`, `sbatch`, `tools/train.py`, or `tools/test.py` was run.
- No final mAP, runtime/FLOPs, deployment, or paper claim exists.
- No Pro/Rosetta/Claude/Gemini review was run in this stage.
- Gate output still reports `full_train_unlocked=false` and `remote_sync_unlocked_by_local_gate=false`.

## Allowed Next Action

Only final read-only review or Linux/local precheck-only preparation may follow from this local gate result. Formal long training remains locked until the project-required later review/launch gates explicitly unlock it.
