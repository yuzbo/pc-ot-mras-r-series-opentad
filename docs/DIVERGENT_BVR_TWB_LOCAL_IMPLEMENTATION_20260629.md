# DIVERGENT BVR-TWB Local Implementation 2026-06-29

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Stage: `LOCAL_IMPLEMENTATION`

This implementation is local-only. It contains CPU validators, synthetic ledger smoke, selector core, witness packet generation, a heuristic/mock value predictor, dynamic budget control, sparse gather validation, and matched controls.

Locked items remain locked: remote sync, Slurm, training, validation/test evaluation, mAP, runtime/FLOPs, deploy claims, paper claims, C3 merge, T2 as a first-version dependency, and dense raw handoff sparse claims.

Implemented components:

- `StateScout`: deterministic deploy-visible curves from actionness and optional motion-like preview signal.
- `Scaffold`: thin endpoint-including linspace anchors plus hard max-gap repair.
- `BoundaryBelief` and `WitnessPackets`: scout-only brackets and role-labeled packets.
- `PacketValuePredictor`: heuristic fallback and mock constant ablation with anti-actionness-only diagnostics.
- `DynamicBudgetController`: scaffold first, min-K fill, hard gap repair, active bracket role coverage, value-per-cost additions, deterministic tie-breaks, and explicit stop reasons.
- `SparseRawGather`: local gather smoke with shape and fingerprint evidence; no sparse-compute claim without detector forward evidence.
- `Validators`: route identity, no-leakage, selected original dense indices, original-time decode metadata, sparse gather audit, dynamicity, and uniform mimicry checks.
- `MatchedControls`: same-K uniform, same-mean-K exact-uniform, random same-K, scaffold-only, and TWB no-regret controls using the same gather/time/validator path.

Synthetic smoke command:

```powershell
python tools/bvr_twb/build_synthetic_ledgers.py --out-dir .tmp_bvr_twb_ledgers --overwrite
```

Expected output files:

- `.tmp_bvr_twb_ledgers/bvr_twb_deploy_ledgers.jsonl`
- `.tmp_bvr_twb_ledgers/bvr_twb_selection_rows.jsonl`
- `.tmp_bvr_twb_ledgers/bvr_twb_candidate_packets.jsonl`
- `.tmp_bvr_twb_ledgers/bvr_twb_matched_controls.jsonl`
- `.tmp_bvr_twb_ledgers/summary.json`

Evidence status: local ledger and gather smoke only. No detector metric, FLOPs, latency, deployment readiness, or paper claim is unlocked by this implementation.
