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

## Local Pro-Fix Update

Updated after the GitHub-only Rosetta GPT-5.5 Pro `PASS_ALLOW_LOCAL_FIX_IMPLEMENTATION` decision:

- Boundary belief now records a posterior `belief_update_trace` after selected witness packets. `belief_width_safe` is based on updated witness evidence, not only initial bracket width.
- The controller records per-add `constraint_state`, `selected_decision_subreason`, marginal value components, duplicate/budget/gap checks, and incremental max-gap repair rows.
- Value scoring exposes component diagnostics: `belief_width_gain`, `role_gain`, `gap_gain`, `short_action_gain`, `redundancy_repulsion_penalty`, and `low_actionness_component`.
- Witness packet gap features receive scaffold/selected positions, and residual `gap_bridge` candidate generation uses the explicit BVR `max_gap` contract.
- StateScout deploy metadata rejection is recursive over nested dict/list structures.
- Deploy ledgers now carry explicit `claim_mode`. The local mode remains `local_gather_smoke`; sparse compute remains locked unless a future `sparse_forward_audit` proves `detector_forward_temporal_len == valid_k < dense_T` and no dense raw handoff.
- Matched controls inherit BVR max-gap by default and record random seed, jitter, per-case uniform overlap, scaffold-only policy, and `twb_no_regret` uniform fallback count/ratio.
- Synthetic summaries add posterior belief width and fallback diagnostics while keeping `claim_status` exactly `local_gather_smoke_only_no_sparse_compute_or_metric_claim`.
- Final-review blocker fix: `belief_width_safe` now requires every active bracket in `active_belief_update_trace` to be updated by selected witness packets and individually `belief_width_safe=True`. Uncovered active brackets are not excluded from the stop decision. `_infer_stop_reason` uses the same all-active-safe semantics.
- Local hardening: deploy ledger validation now rejects forged `belief_width_safe` stops, nested leakage aliases such as `ground_truth`, and synthetic builder output is schema-validated before JSONL write.

Still locked: remote sync, Slurm, training, validation/test evaluation, mAP, runtime/FLOPs, deploy claims, paper claims, C3/combo merge, T2 first-version dependency, and dense raw handoff sparse claims.
