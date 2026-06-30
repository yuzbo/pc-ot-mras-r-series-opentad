# DIVERGENT BVR-TWB Formal Readiness Gate 20260630

Timestamp: 2026-06-30 11:19:56 +08:00

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_FormalGate_Worktree_20260630`

Owned branch: `codex/divergent-bvr-twb-formalgate-20260630`

Scope: formal-readiness gate hardening only. No training, no `tools/test.py`, no evaluation, no remote sync, no Slurm, no GPU run, no Pro/Oracle/Rosetta, no C3, no combo route, no metric claim, no runtime/FLOPs claim, no deploy claim, no paper claim, and no sparse-compute claim.

## Current State

Formal full training remains locked.

Short diagnostic remains the only script-level run category prepared by this route. The short diagnostic wrapper now emits formal-readiness evidence only after it observes all required stop-condition checks, finite `reg_loss`, HeadV3 runtime debug evidence with nonzero regression samples, and no skipped optimizer/regression-head marker. That wrapper evidence still does not unlock formal full training.

The formal config still preserves:

- BVR-TWB route label and `bvr_twb_dynamic_subsample`.
- VideoMAE-S pretrain path: `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`.
- Adapter fixed-length padded bridge metadata.
- Native/detector geometry fixes through BVR detector feature positions.
- `sparse_compute_claim=False`.

## Added Gate

New validator: `tools/bvr_twb/validate_bvr_twb_formal_readiness.py`.

Inputs required:

- formal config;
- formal pipeline/launch precheck summary;
- Linux geometry summary from `python tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch`;
- short diagnostic or precheck train log.

Hard requirements:

- launch gate must pass but still report `full_train_unlocked=false`;
- pipeline summary must prove selected raw frames are handed off before decode/backbone;
- fixed padded bridge must be compatibility-only and cannot claim sparse compute;
- adapter padding must be invalid for detector/head evidence;
- geometry summary must be from Linux, must not skip torch runtime, and must pass source, numpy bridge, and torch runtime contracts;
- train log must show the required VideoMAE-S pretrain path and a real checkpoint load marker;
- train log must contain finite `Loss=...` and finite `reg_loss=...`;
- train log must contain `[Train][RuntimeDebug]` with HeadV3 fp32 regression flags and nonzero kept regression samples;
- train log must contain explicit `[bvr_twb_formal_precheck] finite_gradients=true no_skipped_optimizer_step=true no_skipped_reg_head=true`;
- log must reject historical `non-finite gradients detected ... skip optimizer step`, NaN/Inf, missing pretrain, eval/mAP/result markers, formal-train unlock markers, deploy/paper claims, and sparse-compute/FLOPs claims.

Even when all evidence is present, validator output is:

`allowed_next_action=FORMAL_REVIEW_ONLY_FULL_TRAIN_STILL_LOCKED`

and:

`formal_train_unlocked=false`, `full_train_unlocked=false`, `sparse_compute_claim=false`.

## Raw Handoff Versus Padded Bridge Audit

The BVR-TWB pipeline ledger now records:

- `raw_frame_handoff_stage=pre_decode_selected_raw_frames`;
- `selected_raw_frames_before_decode=true`;
- `decode_input_frame_inds`, whose prefix must equal `selected_frame_inds`;
- `fixed_padded_bridge_sparse_compute_claim=false`;
- `adapter_padding_role=fixed_length_decode_backbone_compatibility_invalid_observation`;
- `adapter_padding_invalid_for_detector=true`.

Validator behavior:

- `selected_frame_inds` length must equal `valid_k`.
- `decode_input_frame_inds[:valid_k]` must equal `selected_frame_inds`.
- Adapter padding must be hold-last duplicate compatibility input.
- Adapter padding cannot count as detector valid observations.
- Any fixed padded bridge sparse-compute claim is rejected.

This proves only the current handoff and geometry contract. It does not prove runtime/FLOPs reduction and must not be used as a sparse-compute claim.

## Local Evidence

Commands run in the owned worktree:

```powershell
python -m pytest tests/test_bvr_twb_formal_readiness.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_geometry_contracts.py tests/test_bvr_twb_shortdiag.py tests/test_bvr_twb_sparse_forward_audit.py -q
```

Result:

```text
38 passed, 12 skipped in 5.85s
```

```powershell
python -m py_compile tools/bvr_twb/validate_bvr_twb_formal_readiness.py tools/bvr_twb/validate_bvr_twb_launch_gate.py tools/bvr_twb/audit_opentad_bvr_twb_pipeline.py tools/bvr_twb/validate_bvr_twb_geometry_contracts.py tests/test_bvr_twb_formal_readiness.py
```

Result: pass.

```powershell
python tools/bvr_twb/validate_bvr_twb_geometry_contracts.py
```

Result summary: source and numpy bridge contracts passed; local Windows torch runtime skipped due `c10.dll` load failure; `full_training_unlocked=false`, `no_training=true`, `no_metric_claim=true`. This is not Linux torch evidence and does not unlock formal readiness.

```powershell
python tools/bvr_twb/validate_bvr_twb_launch_gate.py --config configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py --audit-out-dir .tmp_bvr_twb_launch_gate_formalreadiness
```

Result:

```json
{"adapter_bridge_mode": "adapter_fixed_length_padded_bridge", "allowed_next_action": "FINAL_READ_ONLY_REVIEW_THEN_LINUX_PRECHECK_ONLY", "full_train_unlocked": false, "gate_pass": true, "remote_sync_unlocked_by_local_gate": false, "route_label": "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3", "sparse_compute_claim": false}
```

The temporary audit directory was removed after this check.

## Still Locked

- Formal full training.
- Remote sync.
- Slurm.
- GPU execution.
- Validation/test evaluation and `tools/test.py`.
- mAP or high-IoU localization claims.
- Sparse compute, FLOPs, latency, runtime, deployment, or paper claims.
- C3/C3-Pro/GlobalRank/Interval/combo-route mixing.

## Remaining Blockers

Before any formal full train can be discussed, the route still needs:

- Linux `--require-torch` geometry precheck artifact with `torch_runtime_skipped=false`;
- real pretrain file and checkpoint-load evidence in the train/precheck log;
- finite-gradient and no skipped optimizer/regression-head evidence line from the wrapper;
- final read-only review or project-required review gate;
- explicit human/coordinator launch decision after the fail-closed formal-readiness validator passes.
