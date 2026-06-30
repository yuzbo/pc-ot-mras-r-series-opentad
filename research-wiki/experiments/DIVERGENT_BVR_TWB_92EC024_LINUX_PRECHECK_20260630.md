# DIVERGENT BVR-TWB 92ec024 Linux Precheck 20260630

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_Final_Worktree_20260630`

Owned branch: `codex/divergent-bvr-twb-final-20260630`

Remote clean clone: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024`

Remote branch/commit: `codex/divergent-bvr-twb-final-20260630` / `92ec024`

## Purpose

After the older formalgate BVR run produced a severe first-validation result, the next allowed work was bounded diagnostic evidence only. This precheck verifies the newer BVR final geometry-contract branch without launching training, evaluation, Slurm child work, or any metric-producing run.

This check does not use GPU1, does not release or modify parent hold `1118197`, and does not mix with C3/ABR/MDL routes.

## Remote Command Scope

The precheck ran on the N16R4 login side with the OpenTAD conda environment initialized from `/data/home/sczc063/run/yuzibo/conda_envs/opentad`.

Executed checks:

```bash
python tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch
python tools/bvr_twb/audit_opentad_bvr_twb_pipeline.py --out-dir "$LOGDIR/bvr_twb_pipeline_audit" --overwrite
python tools/bvr_twb/audit_sparse_forward_precheck.py --mode fake_raw --out-dir "$LOGDIR/bvr_twb_sparse_forward_fake_raw" --overwrite
python tools/bvr_twb/audit_sparse_forward_precheck.py --mode module_fake_forward --out-dir "$LOGDIR/bvr_twb_sparse_forward_module" --overwrite
```

Remote log directory:

`/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_linux_precheck_92ec024_20260630_174631_+0800_retry_env`

## Result

The Linux precheck passed.

Geometry validator:

```json
{
  "dense_T": 384,
  "detector_feature_valid_k": 14,
  "full_training_unlocked": false,
  "loader_frame_count": 192,
  "no_metric_claim": true,
  "no_training": true,
  "no_video_decode": true,
  "numpy_bridge_contract": "passed",
  "raw_valid_k": 96,
  "source_contract": "passed",
  "target_frame_num": 192,
  "torch_runtime_contract": "passed",
  "torch_runtime_skipped": false,
  "validator": "bvr_twb_geometry_contracts"
}
```

Pipeline audit:

```text
BVR-TWB OpenTAD pipeline audit: ledgers=3 all_validated=True sparse_compute_claim=False blocked=False
```

Sparse-forward fake raw audit:

```text
BVR-TWB sparse-forward precheck: mode=fake_raw ledgers=3 verdicts={'PASS_LOCAL_SHAPE_ONLY_NO_SPARSE_COMPUTE_CLAIM': 3} claim_status=sparse_forward_precheck_shape_only_no_metric_claim sparse_compute_claim=False
```

Sparse-forward module fake-forward audit:

```text
BVR-TWB sparse-forward precheck: mode=module_fake_forward ledgers=3 verdicts={'PASS_REAL_MODULE_FORWARD_NO_METRIC_CLAIM': 3} claim_status=sparse_forward_precheck_shape_only_no_metric_claim sparse_compute_claim=False
```

## Interpretation

This is useful bounded diagnostic progress for the latest BVR final branch. It proves that the Linux torch runtime geometry contract, OpenTAD pipeline audit, and sparse-forward shape/module prechecks pass on commit `92ec024`.

It does not prove BVR correctness under full detector training, does not explain the older severe result by itself, and does not unlock full training.

Still locked:

- BVR formal/full training.
- `tools/test.py`.
- official mAP or final metric claims.
- runtime/FLOPs/sparse-compute claims.
- deployment or paper claims.
- C3/BVR/ABR/MDL combo claims.
- use of GPU1.
- parent hold release/cancel/requeue/replace.

Allowed next action remains bounded BVR diagnostics or a separate reviewed launch decision after GPU0 is free and the relevant gate evidence is complete.
